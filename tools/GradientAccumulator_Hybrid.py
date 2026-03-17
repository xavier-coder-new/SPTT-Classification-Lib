import torch
from global_param import global_vars

class GradientAccumulator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.input = []
        self.hx = []
        self.delta = []
        self.length = 0
        # print("执行了sbtpca的reset")
    
    def accumulate(self, input, hx, delta):
        self.input.append(input)
        self.hx.append(hx)
        self.delta.append(delta)
        self.length += len(input)
        # 当满足这个判断条件时，表示所有时间步都拼接完了。
        if self.length >= global_vars.total_time_block_size:
            global_vars.compression_finished = True
        # global_vars.compression_finished = True
        
    def get_concatenated_gradients(self):
        return torch.cat(self.input, dim=0), torch.cat(self.hx, dim=0), torch.cat(self.delta, dim=0)
    
    def save_sbtpca_parameter(self, X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih,
                              X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh):
        
        self.X_matrix_ih = X_matrix_ih
        self.Sigma_ih = Sigma_ih
        self.Sigma_matrix_ih = Sigma_matrix_ih
        self.Delta_matrix_ih = Delta_matrix_ih
        
        self.X_matrix_hh = X_matrix_hh
        self.Sigma_hh = Sigma_hh
        self.Sigma_matrix_hh = Sigma_matrix_hh
        self.Delta_matrix_hh = Delta_matrix_hh
        
    def deliver_sbtpca_parameter(self):
        return self.X_matrix_ih, self.Sigma_ih, self.Sigma_matrix_ih, self.Delta_matrix_ih, \
               self.X_matrix_hh, self.Sigma_hh, self.Sigma_matrix_hh, self.Delta_matrix_hh
    
    def save_time_backward(self, time_backward):
        self.time_back_ward = time_backward
    
    def deliver_time_backward(self):
        return self.time_back_ward