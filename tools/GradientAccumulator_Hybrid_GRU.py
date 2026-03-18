import torch


class GradientAccumulatorGRU:
    def __init__(self):
        self.reset()

        # persistent low-rank states for ih
        self.X_matrix_ih = None
        self.Sigma_ih = None
        self.Sigma_matrix_ih = None
        self.Delta_matrix_ih = None

        # persistent low-rank states for hh
        self.X_matrix_hh = None
        self.Sigma_hh = None
        self.Sigma_matrix_hh = None
        self.Delta_matrix_hh = None

        self.time_backward = 0.0

    def reset(self):
        """
        Reset runtime accumulation buffers for the current chunk / current backward pass.
        """
        self.inputs = []
        self.hxs = []
        self.delta_ihs = []
        self.delta_hhs = []
        self.length = 0

    def accumulate(self, inputs, hx, delta_ih, delta_hh):
        """
        Accumulate valid samples for GRU SPTT compression.

        Args:
            inputs:   [N, input_dim]
            hx:       [N, hidden_dim]
            delta_ih: [N, 3H]
            delta_hh: [N, 3H]
        """
        if inputs is None or hx is None or delta_ih is None or delta_hh is None:
            return

        if inputs.numel() == 0:
            return

        if inputs.size(0) != hx.size(0) or inputs.size(0) != delta_ih.size(0) or inputs.size(0) != delta_hh.size(0):
            raise ValueError(
                f"Batch size mismatch in accumulator: "
                f"inputs={inputs.shape}, hx={hx.shape}, delta_ih={delta_ih.shape}, delta_hh={delta_hh.shape}"
            )

        self.inputs.append(inputs)
        self.hxs.append(hx)
        self.delta_ihs.append(delta_ih)
        self.delta_hhs.append(delta_hh)
        self.length += inputs.size(0)

    def get_concatenated_gradients(self):
        """
        Return concatenated runtime buffers.

        Returns:
            input_cat:    [T, input_dim]
            hx_cat:       [T, hidden_dim]
            delta_ih_cat: [T, 3H]
            delta_hh_cat: [T, 3H]

        If nothing has been accumulated, returns (None, None, None, None).
        """
        if self.length == 0:
            return None, None, None, None

        return (
            torch.cat(self.inputs, dim=0),
            torch.cat(self.hxs, dim=0),
            torch.cat(self.delta_ihs, dim=0),
            torch.cat(self.delta_hhs, dim=0),
        )

    def save_sbtpca_parameter(
        self,
        X_matrix_ih,
        Sigma_ih,
        Sigma_matrix_ih,
        Delta_matrix_ih,
        X_matrix_hh,
        Sigma_hh,
        Sigma_matrix_hh,
        Delta_matrix_hh,
    ):
        """
        Save persistent low-rank factors for the current GRU layer.
        """
        self.X_matrix_ih = X_matrix_ih
        self.Sigma_ih = Sigma_ih
        self.Sigma_matrix_ih = Sigma_matrix_ih
        self.Delta_matrix_ih = Delta_matrix_ih

        self.X_matrix_hh = X_matrix_hh
        self.Sigma_hh = Sigma_hh
        self.Sigma_matrix_hh = Sigma_matrix_hh
        self.Delta_matrix_hh = Delta_matrix_hh

    def deliver_sbtpca_parameter(self):
        """
        Return persistent low-rank factors for the current GRU layer.
        """
        return (
            self.X_matrix_ih,
            self.Sigma_ih,
            self.Sigma_matrix_ih,
            self.Delta_matrix_ih,
            self.X_matrix_hh,
            self.Sigma_hh,
            self.Sigma_matrix_hh,
            self.Delta_matrix_hh,
        )

    def save_time_backward(self, time_backward):
        self.time_backward = time_backward

    def deliver_time_backward(self):
        return self.time_backward