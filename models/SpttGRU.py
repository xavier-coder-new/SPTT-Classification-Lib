import torch
import torch.nn as nn
import math
from tools.Timers import Timers
import time
from tools.GradientAccumulator_Hybrid_GRU import GradientAccumulatorGRU

timers = Timers()

def chodral_subspace_drift(Q_matrix_old: torch.Tensor, Q_matrix_new: torch.Tensor) -> torch.Tensor:
    """
    [unit_dim, krank]
    Q_matrix_old, Q_matrix_new: [d, k] with orthonormal columns.
    Returns scalar drift (chordal distance): sqrt(k - ||Q_old^T Q_new||_F^2)
    M: Q_old.T @ Q_new
    """
    k = Q_matrix_old.shape[1]
    M = Q_matrix_old.transpose(0, 1) @ Q_matrix_new
    frob_sq = torch.sum(M ** 2)
    drift_sq = torch.clamp(k - frob_sq, min=0.0)
    return torch.sqrt(drift_sq)

def qr_with_non_negative_diagonal(matrix):
    # 进行QR分解
    Q, R = torch.linalg.qr(matrix)
    
    # 获取 R 矩阵对角线元素的符号
    diagonal_sign = torch.sign(torch.diag(R))
    
    # 避免对角线元素为零的情况
    diagonal_sign[diagonal_sign == 0] = 1
    # ic(diagonal_sign)
    
    # 调整 Q 矩阵的列符号和Q = Q * diagonal_sign得到的结果一样。
    Q1 = Q * diagonal_sign.unsqueeze(0)
    
    R = diagonal_sign.unsqueeze(1) * R
    
    return Q1, R, diagonal_sign

def QR_matrix(X_matrix, target_shape):
    Q, R, diagonal_sign = qr_with_non_negative_diagonal(X_matrix)
    
    if Q.shape != target_shape:
        # 进入这个会出现nan数值错误
        raise ValueError(f"QR shape mismatch: got {Q.shape}, expected {target_shape}")

    return Q, R, diagonal_sign
    

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        
        # judge the type of input data.
        self.input_type = getattr(args, "input_type", "feature")
        if self.input_type == "text":
            self.vocab_size = args.vocab_size
            self.embed_dim = args.embed_dim
            self.pad_idx = getattr(args, "pad_idx", None)

            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.embed_dim,
                padding_idx=self.pad_idx,
            )

            rnn_input_dim = self.embed_dim
        else:
            self.embedding = None
            rnn_input_dim = args.feature_dim
        
        self.model = CustomGRU(args, input_dim=rnn_input_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        # The final logits for caching "completed samples" during stream/block training
        self.final_logits = None
        self.finished_mask = None
        
    def reset_logits(self):
        self.final_logits = None
        self.finished_mask = None
    
    def _prepare_inputs(self, inputs):
        """
        inputs:
            - feature mode: list of [B, F]
            - text mode:    list of [B]
        returns:
            processed_inputs: list of [B, F]
            
        """
        if self.input_type == "text":
            processed_inputs = []
            for x_t in inputs:
                # x_t: [B]
                if x_t.dim() != 1:
                    raise ValueError(f"Expected [B], got {x_t.shape}")
                emb_t = self.embedding(x_t.long())  # [B, E]
                processed_inputs.append(emb_t)
            return processed_inputs
        
        else:
            for x_t in inputs:
                if x_t.dim() != 2:
                    raise ValueError(f"Expected [B, F], got {x_t.shape}")
            return inputs
    
    def reset_sptt_state(self, chunk_actual_length):
        self.model.reset_sptt_runtime(chunk_actual_length)

    def forward(self, inputs, actual_length, sequence_length, ended_in_chunk):
        """
        inputs: list of [B, F], the length of inputs is sequence_length (time steps num)
        actual_length: [B], the length of chunck_actual_length for current batch. the range is [0, chunk_T]
        sequen_length: int, the length of chunk block (chunk_T)
        ended_in_chunk: [B] bool, The final logits for caching "completed samples" during stream/block training
        
        return:
            effective_logits: [B, C], C is the number of classes
                - For unfinished samples: The logits corresponding to the last valid state of the current chunk
                - For samples that have already ended: Keep the logits at the end moment and do not change it anymore, 
                which means these logits will not contribute to the loss and gradient calculation in the following 
                training of next chunks.
                
        current chunk: end_in_chunk -> [True, True, False, False, True]
        previuos chunk: finished_mask -> [False, False, False, False, True]
        ~finished_mask -> [True, True, True, True, False]
        newly_finished -> [True, True, False, False, False]
        
        The final_logits is continuously filled. Only when the finished samples (the last time steps outputs) will be
        included in the final_logits, while the unfinished parts remain at 0. The effective_logits is always full, but
        once an end is reached, the completed part of the final_logits will be transferred to the effective_logits, thus
        ensuring the freezing of the end part.
                 
        """
        processed_inputs = self._prepare_inputs(inputs)
        top_hidden = self.model(
            inputs=processed_inputs, 
            chunck_actual_length=actual_length, 
            chunk_sequence_length=sequence_length
        )
        output = self.fc(top_hidden) # [B, C]
        B, C = output.shape
        device = output.device
        
        if self.final_logits is None:
            self.final_logits = torch.zeros(B, C, device=device, dtype=output.dtype)
            # True for finished, False for unfinished
            self.finished_mask = torch.zeros(B, device=device, dtype=torch.bool)
        
        prev_finished_mask = self.finished_mask.clone()
        # For samples that have already ended before the current chunk, we need to keep the logits remain frozen.
        effective_logits = output.clone()
        
        if prev_finished_mask.any():
            # if there are already finished samples, we need to overwrite the current output of these samples with the previously saved final_logits
            # make sure the finished samples will not make new logits.
            effective_logits[prev_finished_mask] = self.final_logits[prev_finished_mask]
        
        # For the newly concluded samples in the current chunk, record and freeze the logits.
        # Find the "newly ended sample of the current chunk" (previously not ended ∧ currently ended)
        newly_finished = ended_in_chunk & (~prev_finished_mask)
        if newly_finished.any():
            self.final_logits[newly_finished] = output[newly_finished].detach()
            self.finished_mask[newly_finished] = True
            effective_logits[newly_finished] = output[newly_finished]
        
        # 当前 chunk 应该参与 loss 的样本：
        # 1) 之前没结束的样本（包括当前 newly_finished）
        # 2) 已经在更早 chunk 结束的样本不再参与
        loss_mask = ~prev_finished_mask
        
        # 当前 chunk 内哪些样本到达了最终有效位置
        final_step_mask = ended_in_chunk
        
        return effective_logits, loss_mask, final_step_mask

class CustomGRU(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.device = args.device
        self.krank = args.krank
        self.slide_window_nums  = args.slide_window_nums
        print(f"CustomGRU: input_dim={input_dim}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
        self.cells =nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            cur_input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            self.cells.append(
                CustomGRUCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim,
                    krank=self.krank,
                    slide_window_nums=self.slide_window_nums,
                    device=self.device
                )
            )
        
        self.hx_list = None
            
    def reset_state(self, batch_size: int):
        """
        When starting the training for each batch, the state needs to be reset.
        hx_list[layer]: [B, H]
        cx_list[layer]: [B, H]
        
        """
        self.hx_list = []
        
        for _ in range(self.num_layers):
            self.hx_list.append(torch.zeros(batch_size, self.hidden_dim, device=self.device))
        
        # self.hx_list = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]
        # self.cx_list = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]

    def reset_sptt_runtime(self, chunk_actual_length: torch.Tensor):
        total_valid_steps = int(chunk_actual_length.sum().item())
        for cell in self.cells:
            cell.reset_runtime_state(total_valid_steps)
            
    def reset_persistent_sptt_state(self):
        for cell in self.cells:
            cell.init_sptt_parameters()
            cell.reset_runtime_state(total_time_block_size=0)
    
    def detach_state(self):
        """When starting the training for each chunk (except the first chunk) in TBPTT, the state needs to be detached."""
        if self.hx_list is not None:
            self.hx_list = [hx.detach() for hx in self.hx_list]
    
    def _mask_state_update(self, new_state: torch.Tensor, old_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        new_state, old_state: tuple of tensors, each shape like [B, H].
        
        mask: [B] (1 means valid, 0 means padded / already ended)
        
        """
        if new_state.dim() != 2 or old_state.dim() != 2:
            raise ValueError(
                f"Expected [B, H], got new_state={new_state.shape}, old_state={old_state.shape}"
            )
            
        mask = mask.unsqueeze(1).to(device=new_state.device, dtype=new_state.dtype)  # [B, 1]
        return new_state * mask + old_state * (1.0 - mask)
            
    def forward(self, inputs, chunck_actual_length, chunk_sequence_length):
        """
        inputs: list of [batch_size, feature_dim], the length of inputs is sequence_length (time steps num)
        targets: list of [batch_size], the length of targets is sequence_length (time steps num)
        actual_length: [batch_size], the length of chunck_actual_length for current batch.
        sequence_length: the length of chunk block (chunk_T)
        
        For most cases, the hidden of per batches need to be initialized as zeros.
        
        """
        
        # TODO: Add the function of time recording
        
        if self.hx_list is None:
            raise RuntimeError("Please call reset_state(batch_size) before forward().")
        
        T = int(chunk_sequence_length)
        
        if len(inputs) != T:
            raise ValueError(f"Expected {T} inputs, got {len(inputs)}")# [B]
        
        if (chunck_actual_length < 0).any() or (chunck_actual_length > T).any():
            raise ValueError(f"actual_length must be in [0, {T}], got {chunck_actual_length}")
        
        for t in range(T):
            valid_mask = (t < chunck_actual_length)
            # for 0 layer, the input is the original input (B, F); for upper layers, the input is the hidden state of the previous layer.
            layer_input = inputs[t]
            
            for layer_idx in range(self.num_layers):
                old_h = self.hx_list[layer_idx]
                
                new_h = self.cells[layer_idx](
                    layer_input,
                    old_h, 
                    valid_mask,
                )
                
                masked_h = self._mask_state_update(new_h, old_h, valid_mask)
                
                self.hx_list[layer_idx] = masked_h
                
                # current layer output is the next layer input
                layer_input = masked_h
            
        top_h = self.hx_list[-1]
            
        return top_h
            
            
class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, krank, slide_window_nums, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.slide_window_nums = slide_window_nums
        self.krank = krank
        self.device = device
        
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, input_dim))
        self.w_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
        self.sptt_update_count = 0
        
        self.reset_parameters()
        
        self.init_sptt_parameters()
        self.accumulator = GradientAccumulatorGRU()
        self.reset_runtime_state(total_time_block_size=0)
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
            
    def init_sptt_parameters(self):
        self.X_matrix_ih = torch.randn(self.input_dim, self.krank).to(self.device)
        self.Sigma_ih = torch.randn(self.krank).to(self.device)
        # self.Sigma_ih = torch.tensor([1e-5]*self.krank).to(self.device)
        self.Sigma_matrix_ih = torch.diag(self.Sigma_ih).to(self.device)
        self.Delta_matrix_ih = torch.randn(3 * self.hidden_dim, self.krank).to(self.device)
        
        self.X_matrix_hh = torch.randn(self.hidden_dim, self.krank).to(self.device)
        self.Sigma_hh = torch.randn(self.krank).to(self.device)
        # self.Sigma_hh = torch.tensor([1e-5]*self.krank).to(self.device)
        self.Sigma_matrix_hh = torch.diag(self.Sigma_hh).to(self.device)
        self.Delta_matrix_hh = torch.randn(3 * self.hidden_dim, self.krank).to(self.device)
    
    def reset_runtime_state(self, total_time_block_size: int):
        self.accumulator.reset()
        self.flag = True
        self.compression_finished = False
        self.total_time_block_size = int(total_time_block_size)
        # self.sptt_update_count = 0
        
    def get_sptt_state(self):
        return {
            "accumulator": self.accumulator,
            "flag": self.flag,
            "compression_finished": self.compression_finished,
            "total_time_block_size": self.total_time_block_size,
            "slide_window_nums": self.slide_window_nums,
            "X_matrix_ih": self.X_matrix_ih,
            "Sigma_ih": self.Sigma_ih,
            "Sigma_matrix_ih": self.Sigma_matrix_ih,
            "Delta_matrix_ih": self.Delta_matrix_ih,
            "X_matrix_hh": self.X_matrix_hh,
            "Sigma_hh": self.Sigma_hh,
            "Sigma_matrix_hh": self.Sigma_matrix_hh,
            "Delta_matrix_hh": self.Delta_matrix_hh,
        }
    
    def set_sptt_state(
        self,
        flag,
        compression_finished,
        X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih,
        X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh,
    ):
        self.flag = flag
        self.compression_finished = compression_finished
        self.X_matrix_ih = X_matrix_ih
        self.Sigma_ih = Sigma_ih
        self.Sigma_matrix_ih = Sigma_matrix_ih
        self.Delta_matrix_ih = Delta_matrix_ih
        self.X_matrix_hh = X_matrix_hh
        self.Sigma_hh = Sigma_hh
        self.Sigma_matrix_hh = Sigma_matrix_hh
        self.Delta_matrix_hh = Delta_matrix_hh
    
    def forward(self, inputs, hx, valid_mask):
        return GRUCellFunction.apply(
            inputs, hx,
            self.w_ih, self.w_hh, 
            self.b_ih, self.b_hh,
            valid_mask,
            self,
        )


class GRUCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hidden_state, w_ih, w_hh, b_ih, b_hh, valid_mask, cell_ref):
        # Split the weight matrix and the bias vector
        w_ir, w_iz, w_in = w_ih.chunk(3, 0)
        w_hr, w_hz, w_hn = w_hh.chunk(3, 0)
        b_ir, b_iz, b_in = b_ih.chunk(3)
        b_hr, b_hz, b_hn = b_hh.chunk(3)
        
        # Calculate the reset gate
        reset_gate = torch.sigmoid(
            torch.mm(inputs, w_ir.t()) + b_ir + torch.mm(hidden_state, w_hr.t()) + b_hr
        )
        
        # Calculate the update gate
        update_gate = torch.sigmoid(
            torch.mm(inputs, w_iz.t()) + b_iz + torch.mm(hidden_state, w_hz.t()) + b_hz
        )
        
        # Calculate the candidate hidden state
        n = torch.tanh(
            torch.mm(inputs, w_in.t()) + b_in + reset_gate * (torch.mm(hidden_state, w_hn.t()) + b_hn)
        )
        
        # Calculate the new hidden state
        hy = (1 - update_gate) * n + update_gate * hidden_state

        ctx.save_for_backward(
            inputs, hidden_state, 
            w_ir, w_iz, w_in, 
            w_hr, w_hz, w_hn,
            reset_gate, update_gate, 
            n, w_ih, w_hh,
            b_ir, b_iz, b_in, b_hr, b_hz, b_hn,
            valid_mask,
        )
        
        ctx.cell_ref = cell_ref
        
        return hy
    
    @staticmethod
    def backward(ctx, grad_hy):
        (inputs, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn,
        reset_gate, update_gate, n, w_ih, w_hh, b_ir, b_iz, b_in, b_hr, b_hz, b_hn,
        valid_mask) = ctx.saved_tensors
        
        cell = ctx.cell_ref
        
        # 初始化梯度
        d_input = d_hidden_state = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None
        
        # 计算更新门的梯度
        d_update_gate = (hidden_state - n) * grad_hy
        d_update_gate = d_update_gate * update_gate * (1 - update_gate)

        # 计算候选隐藏状态的梯度
        d_n = (1 - update_gate) * grad_hy
        d_n_tanh = d_n * (1 - n ** 2)

        # 计算重置门的梯度
        hidden_w_hn = torch.mm(hidden_state, w_hn.t()) + b_hn
        d_reset_gate = d_n_tanh * hidden_w_hn
        d_reset_gate = d_reset_gate * reset_gate * (1 - reset_gate)
    
        # ---- build two deltas for GRU ----
        delta_ih = torch.cat((d_reset_gate, d_update_gate, d_n_tanh), dim=1)
        delta_hh = torch.cat((d_reset_gate, d_update_gate, d_n_tanh * reset_gate), dim=1)

        # gradients to input / hidden
        d_input = (
            torch.mm(d_reset_gate, w_ir) +
            torch.mm(d_update_gate, w_iz) +
            torch.mm(d_n_tanh, w_in)
        )

        d_hidden_state = (
            grad_hy * update_gate +
            torch.mm(d_reset_gate, w_hr) +
            torch.mm(d_update_gate, w_hz) +
            reset_gate * torch.mm(d_n_tanh, w_hn)
        )

        # bias gradients
        grad_b_ir = d_reset_gate.sum(0)
        grad_b_iz = d_update_gate.sum(0)
        grad_b_in = d_n_tanh.sum(0)
        grad_b_ih = torch.cat((grad_b_ir, grad_b_iz, grad_b_in))

        grad_b_hr = d_reset_gate.sum(0)
        grad_b_hz = d_update_gate.sum(0)
        grad_b_hn = (d_n_tanh * reset_gate).sum(0)
        grad_b_hh = torch.cat((grad_b_hr, grad_b_hz, grad_b_hn))

        # accumulate only valid rows
        if valid_mask.any():
            valid_inputs = inputs[valid_mask]
            valid_hx = hidden_state[valid_mask]
            valid_delta_ih = delta_ih[valid_mask]
            valid_delta_hh = delta_hh[valid_mask]
            cell.accumulator.accumulate(valid_inputs, valid_hx, valid_delta_ih, valid_delta_hh)

        if cell.accumulator.length >= cell.total_time_block_size and not cell.compression_finished:
            cell.compression_finished = True

        if not cell.compression_finished:
            grad_w_ih = torch.zeros_like(w_ih)
            grad_w_hh = torch.zeros_like(w_hh)

        elif cell.compression_finished and cell.flag:
            cell.flag = False

            input_cat, hx_cat, delta_ih_cat, delta_hh_cat = cell.accumulator.get_concatenated_gradients()

            if input_cat is None:
                raise ValueError("input_cat is None")

            T = input_cat.size(0)
            t = max(1, T // cell.slide_window_nums)

            X_matrix_ih = cell.X_matrix_ih
            Sigma_ih = cell.Sigma_ih
            Sigma_matrix_ih = cell.Sigma_matrix_ih
            Delta_matrix_ih = cell.Delta_matrix_ih

            X_matrix_hh = cell.X_matrix_hh
            Sigma_hh = cell.Sigma_hh
            Sigma_matrix_hh = cell.Sigma_matrix_hh
            Delta_matrix_hh = cell.Delta_matrix_hh

            inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
            inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)

            with torch.no_grad():
                # num_blocks = math.ceil(T / t)
                num_blocks = T // t
                for i in range(1, num_blocks + 1):
                    start_idx = (i - 1) * t
                    end_idx = min(i * t, T)

                    activation_input = input_cat[start_idx:end_idx]
                    activation_hx = hx_cat[start_idx:end_idx]
                    delta_ih_block = delta_ih_cat[start_idx:end_idx]
                    delta_hh_block = delta_hh_cat[start_idx:end_idx]

                    block_len = activation_input.size(0)
                    if block_len == 0:
                        raise ValueError("block_len is 0")

                    scale_factor_right_ih = delta_ih_block @ Delta_matrix_ih / block_len
                    scale_factor_left_ih = activation_input @ X_matrix_ih / block_len

                    scale_factor_right_hh = delta_hh_block @ Delta_matrix_hh / block_len
                    scale_factor_left_hh = activation_hx @ X_matrix_hh / block_len

                    history_factor = i / (i + 1)
                    update_factor = 1 / (i + 1)

                    X_ih_update = activation_input.t() @ scale_factor_right_ih @ inv_Sigma_matrix_ih
                    X_matrix_ih = history_factor * X_matrix_ih + update_factor * X_ih_update

                    Delta_ih_update = delta_ih_block.t() @ scale_factor_left_ih @ inv_Sigma_matrix_ih
                    Delta_matrix_ih = history_factor * Delta_matrix_ih + update_factor * Delta_ih_update

                    X_hh_update = activation_hx.t() @ scale_factor_right_hh @ inv_Sigma_matrix_hh
                    X_matrix_hh = history_factor * X_matrix_hh + update_factor * X_hh_update

                    Delta_hh_update = delta_hh_block.t() @ scale_factor_left_hh @ inv_Sigma_matrix_hh
                    Delta_matrix_hh = history_factor * Delta_matrix_hh + update_factor * Delta_hh_update

                    X_matrix_ih, _, Sigma_ih_signal_X = QR_matrix(X_matrix_ih, X_matrix_ih.shape)
                    Delta_matrix_ih, _, Sigma_ih_signal_Delta = QR_matrix(Delta_matrix_ih, Delta_matrix_ih.shape)
                    align_ih = Sigma_ih_signal_X * Sigma_ih_signal_Delta
                    Delta_matrix_ih = Delta_matrix_ih * align_ih.unsqueeze(0)

                    X_matrix_hh, _, Sigma_hh_signal_X = QR_matrix(X_matrix_hh, X_matrix_hh.shape)
                    Delta_matrix_hh, _, Sigma_hh_signal_Delta = QR_matrix(Delta_matrix_hh, Delta_matrix_hh.shape)
                    align_hh = Sigma_hh_signal_X * Sigma_hh_signal_Delta
                    Delta_matrix_hh = Delta_matrix_hh * align_hh.unsqueeze(0)

                    Sigma_ih_product = torch.sum(
                        ((delta_ih_block @ Delta_matrix_ih) * (activation_input @ X_matrix_ih)) / block_len,
                        dim=0,
                    )
                    Sigma_ih = history_factor * Sigma_ih + update_factor * Sigma_ih_product

                    Sigma_hh_product = torch.sum(
                        ((delta_hh_block @ Delta_matrix_hh) * (activation_hx @ X_matrix_hh)) / block_len,
                        dim=0,
                    )
                    Sigma_hh = history_factor * Sigma_hh + update_factor * Sigma_hh_product
                    
                    # XXX:
                    Sigma_ih = torch.where(Sigma_ih == 0, torch.ones_like(Sigma_ih), Sigma_ih)
                    Sigma_hh = torch.where(Sigma_hh == 0, torch.ones_like(Sigma_hh), Sigma_hh)
                    
                    # XXX：
                    Sigma_ih = torch.nan_to_num(Sigma_ih, nan=1.0)
                    Sigma_hh = torch.nan_to_num(Sigma_hh, nan=1.0)

                    Sigma_matrix_ih = torch.diag(Sigma_ih)
                    Sigma_matrix_hh = torch.diag(Sigma_hh)

                    inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
                    inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)

            grad_w_ih = (X_matrix_ih @ Sigma_matrix_ih @ Delta_matrix_ih.t()).t()
            grad_w_hh = (X_matrix_hh @ Sigma_matrix_hh @ Delta_matrix_hh.t()).t()

            cell.set_sptt_state(
                flag=cell.flag,
                compression_finished=cell.compression_finished,
                X_matrix_ih=X_matrix_ih,
                Sigma_ih=Sigma_ih,
                Sigma_matrix_ih=Sigma_matrix_ih,
                Delta_matrix_ih=Delta_matrix_ih,
                X_matrix_hh=X_matrix_hh,
                Sigma_hh=Sigma_hh,
                Sigma_matrix_hh=Sigma_matrix_hh,
                Delta_matrix_hh=Delta_matrix_hh,
            )

        else:
            grad_w_ih = torch.zeros_like(w_ih)
            grad_w_hh = torch.zeros_like(w_hh)
        
        return d_input, d_hidden_state, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None