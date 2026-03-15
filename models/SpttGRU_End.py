import torch
import torch.nn as nn

class Model(nn.Module):
    pass

class CustomGRU(nn.Module):
    pass

class GRUCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hidden, weight_ih, weight_hh, bias_ih=None, bias_hh=None):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass