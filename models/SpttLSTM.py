import torch
import torch.nn as nn
import math
from tools.Timers import Timers
import time
from typing import Tuple
from global_param import global_vars

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


"""Version 1"""
def QR_matrix(X_matrix, target_shape):
    Q, R, diagonal_sign = qr_with_non_negative_diagonal(X_matrix)
    # ic(Q.shape, target_shape)
    
    # 使用预分配和高效索引
    if Q.shape != target_shape:
        # 进入这个会出现nan数值错误
        assert Q.shape == target_shape, "not permit to change matrix shape"
        result = torch.zeros(target_shape, device=X_matrix.device, dtype=X_matrix.dtype)
        result[:min(Q.shape[0], target_shape[0]), :min(Q.shape[1], target_shape[1])] = \
            Q[:min(Q.shape[0], target_shape[0]), :min(Q.shape[1], target_shape[1])]
        return result, R, diagonal_sign
    # ic(Q.shape, diagonal_sign.shape)
    return Q, R, diagonal_sign
    
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

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.model = CustomLSTM(args)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        # The final logits for caching "completed samples" during stream/block training
        self.final_logits = None
        self.finished_mask = None
        
    def reset_logits(self):
        self.final_logits = None
        self.finished_mask = None

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
        
        top_hidden, next_cell = self.model(
            inputs=inputs, 
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
        
        # For samples that have already ended before the current chunk, we need to keep the logits remain frozen.
        effective_logits = output.clone()
        if self.finished_mask.any():
            # if there are already finished samples, we need to overwrite the current output of these samples with the previously saved final_logits
            # make sure the finished samples will not make new logits.
            effective_logits[self.finished_mask] = self.final_logits[self.finished_mask]
        
        # For the newly concluded samples in the current chunk, record and freeze the logits.
        # Find the "newly ended sample of the current chunk" (previously not ended ∧ currently ended)
        newly_finished = ended_in_chunk & (~self.finished_mask)
        if newly_finished.any():
            self.final_logits[newly_finished] = output[newly_finished]
            self.finished_mask[newly_finished] = True
            effective_logits[newly_finished] = self.final_logits[newly_finished]
            
        return effective_logits


class CustomLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.device = args.device
        self.krank = args.krank
        
        self.cells =nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            input_dim = self.feature_dim if layer_idx == 0 else self.hidden_dim
            self.cells.append(
                CustomLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    feature_dim=self.feature_dim,
                    krank=self.krank,
                    device=self.device
                )
            )
        
        self.hx_list = None
        self.cx_list = None
            
    def reset_state(self, batch_size: int):
        """
        When starting the training for each batch, the state needs to be reset.
        hx_list[layer]: [B, H]
        cx_list[layer]: [B, H]
        
        """
        self.hx_list = []
        self.cx_list = []
        
        for _ in range(self.num_layers):
            self.hx_list.append(torch.zeros(batch_size, self.hidden_dim, device=self.device))
            self.cx_list.append(torch.zeros(batch_size, self.hidden_dim, device=self.device))
        
        # self.hx_list = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]
        # self.cx_list = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]

    def detach_state(self):
        """When starting the training for each chunk (except the first chunk) in TBPTT, the state needs to be detached."""
        if self.hx_list is not None:
            self.hx_list = [hx.detach() for hx in self.hx_list]
        if self.cx_list is not None:
            self.cx_list = [cx.detach() for cx in self.cx_list]
    
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
        
        if self.hx_list is None or self.cx_list is None:
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
                old_c = self.cx_list[layer_idx]
                
                new_h, new_c = self.cells[layer_idx](
                    layer_input,
                    old_h, 
                    old_c,
                )
                
                masked_h = self._mask_state_update(new_h, old_h, valid_mask)
                masked_c = self._mask_state_update(new_c, old_c, valid_mask)
                
                self.hx_list[layer_idx] = masked_h
                self.cx_list[layer_idx] = masked_c
                
                # current layer output is the next layer input
                layer_input = masked_h
            
            top_h = self.hx_list[-1]
            top_c = self.cx_list[-1]
            
        return top_h, top_c
            
            
class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, krank, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.krank = krank
        self.device = device
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, input_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))
        
        self.reset_parameters()
        self.init_sptt_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
            
    def init_sptt_parameters(self):
        self.X_matrix_ih = torch.randn(self.feature_dim, self.krank).to(self.device)
        self.Sigma_ih = torch.randn(self.krank).to(self.device)
        # self.Sigma_ih = torch.tensor([1e-5]*self.krank).to(self.device)
        self.Sigma_matrix_ih = torch.diag(self.Sigma_ih).to(self.device)
        self.Delta_matrix_ih = torch.randn(4 * self.hidden_dim, self.krank).to(self.device)
        
        self.X_matrix_hh = torch.randn(self.hidden_dim, self.krank).to(self.device)
        # TODO
        self.Sigma_hh = torch.randn(self.krank).to(self.device)
        # self.Sigma_hh = torch.tensor([1e-5]*self.krank).to(self.device)
        self.Sigma_matrix_hh = torch.diag(self.Sigma_hh).to(self.device)
        self.Delta_matrix_hh = torch.randn(4 * self.hidden_dim, self.krank).to(self.device)
    
    def forward(self, inputs, hx, cx):
        return LSTMCellFunction.apply(inputs, hx, cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh)


class LSTMCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hx, cx, w_ih, w_hh, b_ih, b_hh):
        gates = (torch.mm(inputs, w_ih.t()) + b_ih + torch.mm(hx, w_hh.t()) + b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        ctx.save_for_backward(inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh)
        
        # 返回一个零时间，真正的时间将在backward中测量。
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh = ctx.saved_tensors
        
        # 初始化梯度
        grad_inputs = grad_hx = grad_cx = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None
        
        if not global_vars.compression_finished:
            # 计算梯度
            grad_outgate = grad_hy * torch.tanh(cy) * outgate * (1 - outgate)
            grad_cy = grad_hy * outgate * (1 - torch.tanh(cy) ** 2) + grad_cy
            grad_ingate = grad_cy * cellgate * ingate * (1 - ingate)
            grad_cellgate = grad_cy * ingate * (1 - cellgate ** 2)
            grad_forgetgate = grad_cy * cx * forgetgate * (1 - forgetgate)
            
            # if global_param['flag']:
            # 保存stitt所用的参数
            delta = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1)
            accumulator.accumulate(inputs, hx, delta)
            
            # 计算输入和隐藏状态的梯度
            grad_inputs = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_ih)
            grad_hx = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_hh)
            
            # 计算偏置的梯度
            grad_b_ih = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
            grad_b_hh = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
            
            # 计算上一个单元状态的梯度
            grad_cx = grad_cy * forgetgate
        
        if global_param['compression_finished'] and global_param['flag']:
            grad_w_ih = torch.zeros_like(w_ih)
            grad_w_hh = torch.zeros_like(w_hh)
            
            global_param['flag'] = False
            
            # 取stitt所需参数
            X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih, X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh = accumulator.deliver_sbtpca_parameter()
            input_cat, hx_cat, delta_cat = accumulator.get_concatenated_gradients()
            
            # len是返回张量的第一个维度的长度，对应的就是batch_size行数。
            T = len(input_cat)
            # ic(T)
            t = T // time_block_num
            
            inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
            inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)
            
            start_calculate_gradients = time.time()
            with torch.no_grad():
                for i in range(1, T // t + 1):
                    # print("运行")
                    start_idx = (i - 1) * t
                    activation_input = input_cat[start_idx:start_idx + t] # 半开区间，不包括结束索引
                    activation_hx = hx_cat[start_idx:start_idx + t]
                    delta = delta_cat[start_idx:start_idx + t]
                                        
                    scale_factor_right_ih = delta @ Delta_matrix_ih / t
                    scale_factor_left_ih = activation_input @ X_matrix_ih / t
                    
                    scale_factor_right_hh = delta @ Delta_matrix_hh / t
                    scale_factor_left_hh = activation_hx @ X_matrix_hh / t
                    
                    # 使用预计算的系数提高效率
                    i_factor = i / (i + 1)
                    update_factor = 1 / (i + 1)
        
                    # 更新X和Delta矩阵
                    X_ih_update = activation_input.t() @ scale_factor_right_ih @ inv_Sigma_matrix_ih
                    X_matrix_ih = i_factor * X_matrix_ih + update_factor * X_ih_update
                    
                    Delta_ih_update = delta.t() @ scale_factor_left_ih @ inv_Sigma_matrix_ih
                    Delta_matrix_ih = i_factor * Delta_matrix_ih + update_factor * Delta_ih_update

                    X_hh_update = activation_hx.t() @ scale_factor_right_hh @ inv_Sigma_matrix_hh
                    X_matrix_hh = i_factor * X_matrix_hh + update_factor * X_hh_update

                    Delta_hh_update = delta.t() @ scale_factor_left_hh @ inv_Sigma_matrix_hh
                    Delta_matrix_hh = i_factor * Delta_matrix_hh + update_factor * Delta_hh_update
                    # ic(Sigma_ih.shape, Sigma_hh.shape)
                    X_matrix_ih, _, Sigma_ih_signal_X = QR_matrix(X_matrix_ih, X_matrix_ih.shape)
                    Delta_matrix_ih, _, Sigma_ih_signal_Delta = QR_matrix(Delta_matrix_ih, Delta_matrix_ih.shape)
                    
                    align_ih = Sigma_ih_signal_X * Sigma_ih_signal_Delta
                    Delta_matrix_ih = Delta_matrix_ih * align_ih.unsqueeze(0)
                    
                    X_matrix_hh, _, Sigma_hh_signal_X = QR_matrix(X_matrix_hh, X_matrix_hh.shape)
                    Delta_matrix_hh, _, Sigma_hh_signal_Delta = QR_matrix(Delta_matrix_hh, Delta_matrix_hh.shape)
                    
                    align_hh = Sigma_hh_signal_X * Sigma_hh_signal_Delta
                    Delta_matrix_hh = Delta_matrix_hh * align_hh.unsqueeze(0)
                    
                    # 批量计算Sigma更新 ((i + 1) * t)
                    Sigma_ih_product = torch.sum(((delta @ Delta_matrix_ih) * (activation_input @ X_matrix_ih)) / ((i + 1) * t), dim=0)
                    Sigma_ih = (i_factor * Sigma_ih + Sigma_ih_product)
                    
                    Sigma_hh_product = torch.sum(((delta @ Delta_matrix_hh) * (activation_hx @ X_matrix_hh)) / ((i + 1) * t), dim=0)
                    Sigma_hh = (i_factor * Sigma_hh + Sigma_hh_product)
                    
                    # XXX:
                    Sigma_ih[Sigma_ih == 0] = 1
                    Sigma_hh[Sigma_hh == 0] = 1
                    
                    # 在计算完成后、更新梯度前应用对数缩放
                    Sigma_matrix_ih = torch.diag(Sigma_ih)
                    Sigma_matrix_hh = torch.diag(Sigma_hh)
                    
                    # # 更新逆矩阵
                    inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
                    inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)
            
            grad_w_ih = (X_matrix_ih @ Sigma_matrix_ih @ Delta_matrix_ih.t()).t()
            grad_w_hh = (X_matrix_hh @ Sigma_matrix_hh @ Delta_matrix_hh.t()).t()
            
            end_calculate_gradients = time.time()
            
            # ic(Sigma_hh.detach().cpu().numpy())
            
            accumulator.save_sbtpca_parameter(X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih, 
                                              X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh)
            # accumulator.save_sbtpca_parameter(X_matrix_ih_Gauss, Sigma_ih_Gauss, Sigma_ih_matrix_Gauss, Delta_matrix_ih_Gauss, X_matrix_hh_Gauss, Sigma_hh_Gauss, Sigma_hh_matrix_Gauss, Delta_matrix_hh_Gauss)
            
        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None, None