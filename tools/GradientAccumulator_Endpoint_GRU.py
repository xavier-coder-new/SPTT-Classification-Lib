import torch


class GradientAccumulatorEndpointGRU:
    def __init__(self):
        self.reset()
        self.time_backward = 0.0

    def reset(self):
        self.input = None
        self.hx = None
        self.delta_ih = None
        self.delta_hh = None

    def accumulate(self, input, hx, delta_ih, delta_hh):
        self.input = input
        self.hx = hx
        self.delta_ih = delta_ih
        self.delta_hh = delta_hh


    def get_concatenated_gradients(self):
        return self.input, self.hx, self.delta_ih, self.delta_hh

    def save_sptt_parameter(
        self,
        X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih,
        X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh,
    ):
        self.X_matrix_ih = X_matrix_ih
        self.Sigma_ih = Sigma_ih
        self.Sigma_matrix_ih = Sigma_matrix_ih
        self.Delta_matrix_ih = Delta_matrix_ih

        self.X_matrix_hh = X_matrix_hh
        self.Sigma_hh = Sigma_hh
        self.Sigma_matrix_hh = Sigma_matrix_hh
        self.Delta_matrix_hh = Delta_matrix_hh

    def deliver_sptt_parameter(self):
        return (
            self.X_matrix_ih, self.Sigma_ih, self.Sigma_matrix_ih, self.Delta_matrix_ih,
            self.X_matrix_hh, self.Sigma_hh, self.Sigma_matrix_hh, self.Delta_matrix_hh,
        )
        
    def save_time_backward(self, time_backward):
        self.time_backward = time_backward

    def deliver_time_backward(self):
        return self.time_backward