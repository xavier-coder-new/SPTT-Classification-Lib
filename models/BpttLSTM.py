import torch
import torch.nn as nn
import math
from tools.Timers import Timers
import time
from typing import Tuple
from tools.bptt_compute_profiler import BPTTComputeProfiler

timers = Timers()


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
        
        self.model = CustomLSTM(args, input_dim=rnn_input_dim)
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
        
        top_hidden, next_cell = self.model(
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
        
        loss_mask = ~prev_finished_mask
        
        final_step_mask = ended_in_chunk
        
        return effective_logits, loss_mask, final_step_mask


class CustomLSTM(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.device = args.device
        print(f"CustomLSTM: input_dim={input_dim}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
        self.cells = nn.ModuleList()
        self.bptt_profiler = BPTTComputeProfiler(
            enabled=getattr(args, "profile_bptt_compute", False),
            keep_raw=False,
            profile_epoch=1,
        )
        
        for layer_idx in range(self.num_layers):
            cur_input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            self.cells.append(
                CustomLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                    layer_idx=layer_idx,
                    profiler=self.bptt_profiler,
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
            
    def set_profile_context(
        self,
        *,
        epoch,
        batch_idx,
        chunk_idx,
        model_name,
        data_name,
        sequence_length,
        chunk_length,
        batch_size,
    ):
        if hasattr(self, "bptt_profiler") and self.bptt_profiler is not None:
            self.bptt_profiler.set_context(
                epoch=epoch,
                batch_idx=batch_idx,
                chunk_idx=chunk_idx,
                model_name=model_name,
                data_name=data_name,
                sequence_length=sequence_length,
                chunk_length=chunk_length,
                batch_size=batch_size,
            )

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
    def __init__(
        self,
        input_dim,
        hidden_dim,
        device,
        layer_idx=0,
        profiler=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, input_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))
        
        self.layer_idx = layer_idx
        self.profiler = profiler
        self.gate_multiplier = 4
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, inputs, hx, cx):
        return LSTMCellFunction.apply(
            inputs,
            hx,
            cx,
            self.w_ih,
            self.w_hh,
            self.b_ih,
            self.b_hh,
            self,
        )


class LSTMCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hx, cx, w_ih, w_hh, b_ih, b_hh, cell_ref):
        gates = (torch.mm(inputs, w_ih.t()) + b_ih + torch.mm(hx, w_hh.t()) + b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        ctx.save_for_backward(inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh)
        ctx.cell_ref = cell_ref
        
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        
        inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh = ctx.saved_tensors
        
        # calculate gradient for gates
        grad_outgate = grad_hy * torch.tanh(cy) * outgate * (1 - outgate)
        grad_cy = grad_hy * outgate * (1 - torch.tanh(cy) ** 2) + grad_cy
        grad_ingate = grad_cy * cellgate * ingate * (1 - ingate)
        grad_cellgate = grad_cy * ingate * (1 - cellgate ** 2)
        grad_forgetgate = grad_cy * cx * forgetgate * (1 - forgetgate)
        
        cell = ctx.cell_ref

        if cell.profiler is not None and cell.profiler.enabled:
            cell.profiler.sync_if_cuda(inputs.device)
            start_calculate_gradients = time.perf_counter()
        else:
            start_calculate_gradients = None

        grad_w_ih = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), inputs)
        grad_w_hh = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), hx)

        if cell.profiler is not None and cell.profiler.enabled:
            cell.profiler.sync_if_cuda(inputs.device)
            elapsed_ms = (time.perf_counter() - start_calculate_gradients) * 1000.0

            cell.profiler.add_local_record(
                layer_idx=cell.layer_idx,
                input_dim=cell.input_dim,
                hidden_dim=cell.hidden_dim,
                gate_multiplier=cell.gate_multiplier,
                batch_size=inputs.size(0),
                elapsed_ms=elapsed_ms,
            )
            
        # calculate gradient for bias
        grad_b_ih = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
        grad_b_hh = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)

        # calculate gradient for inputs, hx, cx
        grad_inputs = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_ih)
        grad_hx = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_hh)
        grad_cx = grad_cy * forgetgate

        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None