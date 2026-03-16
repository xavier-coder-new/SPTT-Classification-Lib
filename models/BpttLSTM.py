import torch
import torch.nn as nn
import math
from tools.Timers import Timers
import time
from typing import Tuple

timers = Timers()
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.model = CustomLSTM(args)
        self.fc = nn.Linear(self.hidden_dim, args.classification_num)
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
        ended_in_chunk: [B] bool, The final logits for caching "completed samples" during stream/block training
        
        return:
            effective_logits: [B, C], C is the number of classes
                - For unfinished samples: The logits corresponding to the last valid state of the current chunk
                - For samples that have already ended: Keep the logits at the end moment and do not change it anymore, 
                which means these logits will not contribute to the loss and gradient calculation in the following 
                training of next chunks.
        
        """
        
        next_hidden, next_cell = self.model(inputs, actual_length, sequence_length)
        output = self.fc(next_hidden) # [B, C]
        B, C = output.shape
        device = output.device
        
        if self.final_logits is None:
            self.final_logits = torch.zeros(B, C, device=device, dtype=output.dtype)
            self.finished_mask = torch.zeros(B, device=device, dtype=torch.bool)
        
        # For samples that have already ended before the current chunk, we need to keep the logits remain frozen.
        effective_logits = output.clone()
        if self.finished_mask.any():
            effective_logits[self.finished_mask] = self.final_logits[self.finished_mask]
        
        # For the newly concluded samples in the current chunk, record and freeze the logits.
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
        self.krank = args.krank
        self.device = args.device
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * self.hidden_dim, self.feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * self.hidden_dim, self.hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * self.hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * self.hidden_dim))
        
        self.reset_parameters()
        
        self.hx = None
        self.cx = None
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
            
    def reset_state(self, batch_size: int):
        """When starting the training for each batch, the state needs to be reset."""
        self.hx = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        self.cx = torch.zeros(batch_size, self.hidden_dim, device=self.device)
    
    def detach_state(self):
        """When starting the training for each chunk (except the first chunk) in TBPTT, the state needs to be detached."""
        if self.hx is not None:
            self.hx = self.hx.detach()
        if self.cx is not None:
            self.cx = self.cx.detach()
    
    def _mask_state_update(self, new_state: torch.Tensor, old_state: torch.Tensor, 
                           mask: torch.Tensor) -> torch.Tensor:
        """
        new_state, old_state: tuple of tensors, each shape like [B, H] or
        [num_layers, B, H].
        
        mask: [B] (1 means valid, 0 means padded / already ended)
        
        Example:
            mask = [1, 1, 0]
            old_h = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] # [B, H]
            new_h = [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]] # [B, H]
            
            if new_h.dim() == 2:
                Adjust the shape of the mask to support broadcasting
                mask = mask.unsqueeze(1) # [B] -> [B, 1]
                    # To indicate which batches are valid at the current time step, the hidden state of these time steps is updated to new_h
                    mask_reshaped = [[1], [1], [0]]
                
                masked_s = new_h * mask_reshaped + old_h * (1 - mask_reshaped)
            
            for sample 1:
                # masked_h[0] = new_h[0] * 1 + old_h[0] * 0 
                #             = [0.7, 0.8] * 1 + [0.1, 0.2] * 0
                #             = [0.7, 0.8]  # 使用新状态
            
            for sample 2:
                # masked_h[1] = new_h[1] * 1 + old_h[1] * 0
                #             = [0.9, 1.0] * 1 + [0.3, 0.4] * 0
                #             = [0.9, 1.0]  # 使用新状态
            
            for sample 3:
                # masked_h[2] = new_h[2] * 0 + old_h[2] * 1
                #             = [1.1, 1.2] * 0 + [0.5, 0.6] * 1
                #             = [0.5, 0.6]  # 使用旧状态
                # masked_h[2] = new_h[2] * 0 + old_h[2] * 1
                #             = [0.5, 0.6]
        
        """
        for new_s, old_s in zip(new_state, old_state):
            # new_s and old_s have shape [B, H] or [num_layers, B, H]
            # We want to keep new_s where mask=1 and old_s where mask=0
            
            # First, we need to reshape mask to be broadcastable to new_s/old_s
            # If new_s has shape [B, H], we want mask to be [B, 1]
            # If new_s has shape [num_layers, B, H], we want mask to be [1, B, 1]
            if new_s.dim() == 2:
                mask_reshaped = mask.unsqueeze(1).to(self.device)  # [B] -> [B, 1]
            elif new_s.dim() == 3:
                mask_reshaped = mask.unsqueeze(0).unsqueeze(2).to(self.device)  # [B] -> [1, B, 1]
            else:
                raise ValueError(f"Unexpected state tensor shape: {new_s.shape}")
            
            # assert new_s.device.type == "cuda", f"Expected new_s to be on CUDA, but got {new_s.device}"
            # assert old_s.device.type == "cuda", f"Expected old_s to be on CUDA, but got {old_s.device}"
            # assert mask_reshaped.device.type == "cuda", f"Expected mask_reshaped to be on CUDA, but got {mask_reshaped.device}"
            
            masked_s = new_s * mask_reshaped + old_s * (1 - mask_reshaped)
        
        return masked_s
    
    def forward(self, inputs, actual_length, sequence_length):
        """
        inputs: list of [batch_size, feature_dim], the length of inputs is sequence_length (time steps num)
        targets: list of [batch_size], the length of targets is sequence_length (time steps num)
        actual_length: [batch_size], the length of chunck_actual_length for current batch.
        sequence_length: the length of chunk block (chunk_T)
        
        For most cases, the hidden of per batches need to be initialized as zeros.
        
        """
        # TODO: Add the function of time recording
        
        self.T = int(sequence_length)
        
        if len(inputs) != self.T:
            raise ValueError(f"Expected {self.T} inputs, got {len(inputs)}")# [B]
        
        for t in range(self.T):
            self.old_hx, self.old_cx = self.new_hx, self.new_cx
            x_t = inputs[t]
            self.new_hx, self.new_cx = LSTMCellFunction.apply(
                x_t, self.hx, self.cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh
            )
            
            mask = (t < actual_length).float()
            
            self.mask_hx = self._mask_state_update(self.new_hx, self.old_hx, mask)
            self.mask_cx = self._mask_state_update(self.new_cx, self.old_cx, mask)
            
            # for padding samples, the hidden state should not be updated, and the output of these time steps should not contribute to the loss and gradient calculation.
            end_mask = (t == actual_length - 1)
            
        return self.mask_hx, self.mask_cx, end_mask

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
        
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        
        inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh = ctx.saved_tensors
        
        # 计算各门的梯度
        grad_outgate = grad_hy * torch.tanh(cy) * outgate * (1 - outgate)
        grad_cy = grad_hy * outgate * (1 - torch.tanh(cy) ** 2) + grad_cy
        grad_ingate = grad_cy * cellgate * ingate * (1 - ingate)
        grad_cellgate = grad_cy * ingate * (1 - cellgate ** 2)
        grad_forgetgate = grad_cy * cx * forgetgate * (1 - forgetgate)
        
        # 计算权重的梯度
        start_calculate_gradients = time.time()
        grad_w_ih = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), inputs)
        grad_w_hh = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), hx)
        end_calculate_gradients = time.time()
        time_backward = end_calculate_gradients - start_calculate_gradients
        
        timers.update_cal_gradient_time(time_backward)
        
        grad_b_ih = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
        grad_b_hh = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)

        grad_inputs = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_ih)
        grad_hx = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_hh)
        
        grad_cx = grad_cy * forgetgate

        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh