"""Windowed Echo State Network (WESN): a fixed (untrained) random recurrent
reservoir, rolled forward once per call over a length-T window of raw input,
then read out by ONE trained linear layer over [final reservoir state, a few
raw lagged inputs]. "Windowed" because every call starts the reservoir from
a zero state and rolls it forward T steps -- nothing is carried between
calls, unlike a stateful RNN.

    H. Jaeger, "The 'echo state' approach to analysing and training
    recurrent neural networks", GMD Report 148, German National Research
    Center for Information Technology, 2001.

Ported from drl-ntn-handover's TensorFlow ml_models/wesn.py. One
simplification versus that version: since only the last timestep's readout
is ever used, this only computes the reservoir's FINAL state and reads the
lagged inputs directly by indexing (inputs[:, -1-k, :]) instead of building
a full padded lagged-input sequence and slicing the last step at the end --
same result, no full-sequence readout wasted.
"""

import torch
from torch import nn


class EchoStateReservoir(nn.Module):
    """One fixed-weight leaky-integrator reservoir, unrolled over a window.

    :ivar kernel: [input_size, units], input -> reservoir (frozen).
    :ivar recurrent_kernel: [units, units], reservoir -> reservoir, sparse
        (connectivity) and spectral-radius-scaled so the echo state property
        holds (frozen).
    :ivar bias: [units] or None (frozen).
    """

    def __init__(self, input_size: int, units: int, connectivity: float = 0.1,
                leaky: float = 1.0, spectral_radius: float = 0.9, use_bias: bool = True):
        super().__init__()
        self.units = units
        self.leaky = leaky

        kernel = torch.empty(input_size, units).uniform_(-1.0, 1.0)

        recurrent = torch.empty(units, units).uniform_(-1.0, 1.0)
        connectivity_mask = (torch.rand(units, units) <= connectivity).float()
        recurrent = recurrent * connectivity_mask
        max_abs_eig = torch.linalg.eigvals(recurrent).abs().max()
        if max_abs_eig > 0:
            recurrent = recurrent * (spectral_radius / max_abs_eig)

        self.register_buffer("kernel", kernel)
        self.register_buffer("recurrent_kernel", recurrent)
        self.register_buffer("bias", torch.zeros(units) if use_bias else None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """:param inputs: [batch, seq_len, input_size]
        :output: [batch, units] -- reservoir state at the LAST timestep only.
        """
        batch = inputs.shape[0]
        state = inputs.new_zeros(batch, self.units)
        for t in range(inputs.shape[1]):
            pre_activation = inputs[:, t, :] @ self.kernel + state @ self.recurrent_kernel
            if self.bias is not None:
                pre_activation = pre_activation + self.bias
            candidate = torch.tanh(pre_activation)
            state = (1.0 - self.leaky) * state + self.leaky * candidate
        return state


class Wesn(nn.Module):
    """WESN Q-network: a window of raw input vectors (oldest first) ->
    reservoir's final state concatenated with the window's last `win_len`
    raw inputs -> one trained Linear readout -> one Q-value per action.
    """

    def __init__(self, input_size: int, output_size: int, units: int,
                connectivity: float = 0.1, leaky: float = 1.0,
                spectral_radius: float = 0.9, win_len: int = 0, use_bias: bool = True):
        super().__init__()
        self.win_len = win_len
        self.reservoir = EchoStateReservoir(input_size, units, connectivity, leaky,
                                            spectral_radius, use_bias)
        self.readout = nn.Linear(units + win_len * input_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """:param inputs: [batch, seq_len, input_size], seq_len >= win_len.
        :output: [batch, output_size]
        """
        if self.win_len > inputs.shape[1]:
            raise ValueError(f"win_len ({self.win_len}) cannot be greater than "
                             f"input timesteps ({inputs.shape[1]})")

        reservoir_state = self.reservoir(inputs)

        # Raw lagged inputs at the window's last step: lag 0 (current) ... lag win_len-1
        lagged = [inputs[:, inputs.shape[1] - 1 - k, :] for k in range(self.win_len)]

        features = torch.cat([reservoir_state, *lagged], dim=-1)
        return self.readout(features)
