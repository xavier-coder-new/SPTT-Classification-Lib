import torch


class GradientAccumulatorEndpointLSTM:
    def __init__(self):
        self.reset()

    def reset(self):
        self.input = None
        self.hx = None
        self.delta = None

    def accumulate(self, input, hx, delta):
        self.input = input
        self.hx = hx
        self.delta = delta

    def get_concatenated_gradients(self):
        return self.input, self.hx, self.delta

    def save_sbtpca_parameter(
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

    def deliver_sbtpca_parameter(self):
        return (
            self.X_matrix_ih, self.Sigma_ih, self.Sigma_matrix_ih, self.Delta_matrix_ih,
            self.X_matrix_hh, self.Sigma_hh, self.Sigma_matrix_hh, self.Delta_matrix_hh,
        )