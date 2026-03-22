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


class CustomGRU(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.device = args.device
        print(f"CustomGRU: input_dim={input_dim}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
        self.cells =nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            cur_input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            self.cells.append(
                CustomGRUCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device
                )
            )
        
        self.hx_list = None
            
    def reset_state(self, batch_size: int):
        """
        When starting the training for each batch, the state needs to be reset.
        hx_list[layer]: [B, H]
        
        """
        self.hx_list = []
        
        for _ in range(self.num_layers):
            self.hx_list.append(torch.zeros(batch_size, self.hidden_dim, device=self.device))
        
        # self.hx_list = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]
        # self.cx_list = [torch.zeros(batch_size, self.hidden_dim, device=self.device) for _ in range(self.num_layers)]

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
                )
                
                masked_h = self._mask_state_update(new_h, old_h, valid_mask)
                
                self.hx_list[layer_idx] = masked_h
                
                # current layer output is the next layer input
                layer_input = masked_h
            
        top_h = self.hx_list[-1]
            
        return top_h
            
            
class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, input_dim))
        self.w_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, inputs, hx):
        return GRUCellFunction.apply(inputs, hx, self.w_ih, self.w_hh, self.b_ih, self.b_hh)


class GRUCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hidden_state, w_ih, w_hh, b_ih, b_hh):
        # Split the weight matrix and the bias vector
        w_ir, w_iz, w_in = w_ih.chunk(3, 0)
        w_hr, w_hz, w_hn = w_hh.chunk(3, 0)
        b_ir, b_iz, b_in = b_ih.chunk(3)
        b_hr, b_hz, b_hn = b_hh.chunk(3)
        
        # Calculate the reset gate
        reset_gate = torch.sigmoid(
            torch.mm(input, w_ir.t()) + b_ir + torch.mm(hidden_state, w_hr.t()) + b_hr
        )
        
        # Calculate the update gate
        update_gate = torch.sigmoid(
            torch.mm(input, w_iz.t()) + b_iz + torch.mm(hidden_state, w_hz.t()) + b_hz
        )
        
        # Calculate the candidate hidden state
        n = torch.tanh(
            torch.mm(input, w_in.t()) + b_in + reset_gate * (torch.mm(hidden_state, w_hn.t()) + b_hn)
        )
        
        # Calculate the new hidden state
        hy = (1 - update_gate) * n + update_gate * hidden_state

        ctx.save_for_backward(input, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn,
                              reset_gate, update_gate, n, b_ir, b_iz, b_in, b_hr, b_hz, b_hn)
        
        return hy

    @staticmethod
    def backward(ctx, grad_hy):
        input, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, \
        reset_gate, update_gate, n, b_ir, b_iz, b_in, b_hr, b_hz, b_hn = ctx.saved_tensors

        d_input = d_hidden_state = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None

        # Calculate the gradients of the update gate
        d_update_gate = (hidden_state - n) * grad_hy
        d_update_gate = d_update_gate * update_gate * (1 - update_gate)

        # Calculate the gradients of the candidate hidden state
        d_n = (1 - update_gate) * grad_hy
        d_n_tanh = d_n * (1 - n ** 2)

        # Calculate the gradients of the reset gate
        hidden_w_hn = torch.mm(hidden_state, w_hn.t()) + b_hn
        d_reset_gate = d_n_tanh * hidden_w_hn
        d_reset_gate = d_reset_gate * reset_gate * (1 - reset_gate)

        # Calculate the gradients of the input
        d_input = (
            torch.mm(d_reset_gate, w_ir) +
            torch.mm(d_update_gate, w_iz) +
            torch.mm(d_n_tanh, w_in)
        )

        # Calculate the gradients of the hidden state
        d_hidden_state = (
            grad_hy * update_gate +
            torch.mm(d_reset_gate, w_hr) +
            torch.mm(d_update_gate, w_hz) +
            reset_gate * torch.mm(d_n_tanh, w_hn)
        )

        # Calculate the gradients of the weights
        start_calculate_gradients = time.time()
        
        # Calculate the gradients of w_ih
        grad_w_ir = torch.mm(d_reset_gate.t(), input)
        grad_w_iz = torch.mm(d_update_gate.t(), input)
        grad_w_in = torch.mm(d_n_tanh.t(), input)
        grad_w_ih = torch.cat((grad_w_ir, grad_w_iz, grad_w_in), dim=0)

        # Calculate the gradients of w_hh
        grad_w_hr = torch.mm(d_reset_gate.t(), hidden_state)
        grad_w_hz = torch.mm(d_update_gate.t(), hidden_state)
        grad_w_hn = torch.mm((d_n_tanh * reset_gate).t(), hidden_state)
        grad_w_hh = torch.cat((grad_w_hr, grad_w_hz, grad_w_hn), dim=0)

        end_calculate_gradients = time.time()
        time_backward = end_calculate_gradients - start_calculate_gradients
        
        # Calculate the gradients of b_ih
        grad_b_ir = d_reset_gate.sum(0)
        grad_b_iz = d_update_gate.sum(0)
        grad_b_in = d_n_tanh.sum(0)
        grad_b_ih = torch.cat((grad_b_ir, grad_b_iz, grad_b_in))

        # Calculate the gradients of b_hh
        grad_b_hr = d_reset_gate.sum(0)
        grad_b_hz = d_update_gate.sum(0)
        grad_b_hn = (d_n_tanh * reset_gate).sum(0)
        grad_b_hh = torch.cat((grad_b_hr, grad_b_hz, grad_b_hn))

        return d_input, d_hidden_state, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh