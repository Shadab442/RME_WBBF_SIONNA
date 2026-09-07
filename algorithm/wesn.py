"""Windowed Echo State Network (WESN): a fixed (untrained) random recurrent
reservoir, rolled forward once per call over a length-T window of raw input,
then read out by a trained head over [final reservoir state, a few raw
lagged inputs ("recent", the WESN skip connection that bypasses the
reservoir)]. "Windowed" because every call starts the reservoir from a
zero state and rolls it forward T steps -- nothing is carried between
calls, unlike a stateful RNN.

H. Jaeger, "The 'echo state' approach to analysing and training
recurrent neural networks", GMD Report 148, German National Research
Center for Information Technology, 2001.

- input_kernel and the pre-mask recurrent weights are both
glorot_uniform (Xavier) initialized, matching TF's kernel_initializer/
recurrent_initializer defaults -- NOT a flat Uniform(-1,1) (an earlier
version of this port used the latter; for the recurrent matrix this
washes out after spectral-radius rescaling since that rescales by a
positive scalar regardless of the initial range, but for input_kernel
there is no such rescaling, so the distribution genuinely matters).
- the recurrent update uses recurrent_kernel TRANSPOSED
(state @ recurrent_kernel.T), matching TF's
tf.matmul(in_matrix, concat([kernel, transpose(recurrent_kernel)])) --
an earlier version of this port omitted the transpose. Same spectral
radius either way (eigenvalues of A and A^T are identical, so the
echo-state property itself is unaffected), but a different realized
matrix, hence different reservoir dynamics for a given random draw.
- the readout only computes the reservoir's FINAL-timestep state and
indexes raw lagged inputs directly (inputs[:, -1-k, :]) instead of
building a full padded lagged-input sequence and slicing the last step
at the end -- verified algebraically equivalent to TF's
tf.pad(inputs[:, :-k, :], [[0,0],[k,0],[0,0]]) construction AT the
last timestep specifically (both reduce to inputs[:, T-1-k, :] once
T >= win_len, which this port also requires), so this is a lossless
simplification, not an approximation -- no full-sequence readout
computed and thrown away for timesteps that are never used.
- There is one THING THAT CANNOT be made to match: bit-identical output
for a "matching" seed. PyTorch and TensorFlow use unrelated RNG
algorithms; there is no seed value that makes them draw the same
random numbers. This port matches formulas and distributions exactly,
not raw values.

Beyond the TF original, this also adds (both are NEW, not present in the
ported source -- see Wesn's own docstring for exact semantics):
  - an input LayerNorm, applied once before the raw input reaches EITHER
    the reservoir or the raw-lagged skip connection.
  - three selectable readout architectures via mlp_hidden_sizes/mlp_position:
    no MLP (a single trained linear layer), MLP "early" (skip-connected
    lagged inputs join before the hidden layer), MLP "late" (lagged
    inputs join only right before the final output layer).
"""

import logging

import torch
from torch import nn

from algorithm.mlp import Mlp
from helpers.utils import get_logger

logger = get_logger(__name__)


class EchoStateReservoir(nn.Module):
    """One fixed-weight leaky-integrator reservoir, unrolled over a window.

    :ivar kernel: [input_size, units], input -> reservoir (frozen,
        glorot_uniform initialized).
    :ivar recurrent_kernel: [units, units], reservoir -> reservoir, sparse
        (connectivity) and spectral-radius-scaled so the echo state property
        holds (frozen, glorot_uniform initialized before masking/scaling).
        Used TRANSPOSED in forward() -- see module docstring.
    :ivar bias: [units] or None (frozen, zero-initialized).
    """

    def __init__(self, input_size: int, units: int, connectivity: float = 0.1,
                leaky: float = 1.0, spectral_radius: float = 0.9, use_bias: bool = True):
        super().__init__()
        self.units = units
        self.leaky = leaky

        kernel = torch.empty(input_size, units)
        nn.init.xavier_uniform_(kernel)

        recurrent = torch.empty(units, units)
        nn.init.xavier_uniform_(recurrent)
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
        logger.function("EchoStateReservoir.forward start: inputs shape=%s", tuple(inputs.shape))
        batch = inputs.shape[0]
        state = inputs.new_zeros(batch, self.units)
        for t in range(inputs.shape[1]):
            pre_activation = inputs[:, t, :] @ self.kernel + state @ self.recurrent_kernel.T
            if self.bias is not None:
                pre_activation = pre_activation + self.bias
            candidate = torch.tanh(pre_activation)
            state = (1.0 - self.leaky) * state + self.leaky * candidate
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EchoStateReservoir.forward: final state mean=%.4f", state.mean().item())
        logger.function("EchoStateReservoir.forward end: state shape=%s", tuple(state.shape))
        return state


class Wesn(nn.Module):
    """WESN Q-network: LayerNorm(window of raw input vectors) -> reservoir's
    final state + the window's last `win_len` raw (normalized) inputs
    ("recent", bypassing the reservoir) -> readout -> one Q-value per
    action. Only the reservoir (kernel/recurrent_kernel/bias) is frozen;
    LayerNorm and every readout layer train.

    mlp_hidden_sizes/mlp_position select one of three readout architectures:
      mlp_hidden_sizes=() (default) -- NO MLP: a single trained Linear over
        [state, recent] directly.
      mlp_hidden_sizes non-empty, mlp_position="early" (default when an MLP
        is used) -- [state, recent] concatenated FIRST, then chained
        Linear+ReLU per entry in mlp_hidden_sizes (one entry = 1 hidden
        layer, two entries = 2, etc. -- same convention as algorithm/mlp.py's
        Mlp, reused here directly), then the final Linear -- the raw input
        and reservoir output are fused immediately.
      mlp_hidden_sizes non-empty, mlp_position="late" -- the reservoir
        state ALONE goes through the SAME chained Linear+ReLU hidden stack
        first; `recent` joins only by concatenation right before the final
        Linear (not reused from Mlp, since Mlp's own final layer is
        deliberately activation-free -- the "late" trunk needs a ReLU
        after every hidden layer including its last).
    """

    def __init__(self, input_size: int, output_size: int, units: int,
                connectivity: float = 0.1, leaky: float = 1.0,
                spectral_radius: float = 0.9, win_len: int = 0, use_bias: bool = True,
                mlp_hidden_sizes=(), mlp_position: str = "early"):
        super().__init__()
        if mlp_position not in ("early", "late"):
            raise ValueError(f"mlp_position must be 'early' or 'late', got {mlp_position!r}")
        self.win_len = win_len
        self.mlp_hidden_sizes = list(mlp_hidden_sizes)
        self.mlp_position = mlp_position
        self.norm = nn.LayerNorm(input_size)
        self.reservoir = EchoStateReservoir(input_size, units, connectivity, leaky,
                                            spectral_radius, use_bias)
        lag_features = win_len * input_size

        if not self.mlp_hidden_sizes:
            self.readout = nn.Linear(units + lag_features, output_size)
        elif mlp_position == "early":
            self.readout = Mlp(units + lag_features, output_size, self.mlp_hidden_sizes)
        else:  # "late"
            trunk_sizes = [units, *self.mlp_hidden_sizes]
            trunk_layers = []
            for in_size, out_size in zip(trunk_sizes, trunk_sizes[1:]):
                trunk_layers += [nn.Linear(in_size, out_size), nn.ReLU()]
            self.trunk = nn.Sequential(*trunk_layers)
            self.final = nn.Linear(self.mlp_hidden_sizes[-1] + lag_features, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """:param inputs: [batch, seq_len, input_size], seq_len >= win_len.
        :output: [batch, output_size]
        """
        logger.function("Wesn.forward start: inputs shape=%s", tuple(inputs.shape))
        if self.win_len > inputs.shape[1]:
            logger.warning("Wesn.forward: win_len=%d exceeds input timesteps=%d",
                          self.win_len, inputs.shape[1])
            raise ValueError(f"win_len ({self.win_len}) cannot be greater than "
                             f"input timesteps ({inputs.shape[1]})")

        inputs = self.norm(inputs)
        reservoir_state = self.reservoir(inputs)

        # Raw lagged (normalized) inputs at the window's last step: lag 0
        # (current) ... lag win_len-1 -- the WESN skip connection, bypassing
        # the reservoir entirely.
        lagged = [inputs[:, inputs.shape[1] - 1 - k, :] for k in range(self.win_len)]
        logger.debug("Wesn.forward: reservoir_state shape=%s, %d lagged inputs",
                    tuple(reservoir_state.shape), len(lagged))

        if self.mlp_hidden_sizes and self.mlp_position == "late":
            hidden = self.trunk(reservoir_state)
            features = torch.cat([hidden, *lagged], dim=-1) if lagged else hidden
            output = self.final(features)
        else:
            features = torch.cat([reservoir_state, *lagged], dim=-1)
            output = self.readout(features)
        logger.function("Wesn.forward end: output shape=%s", tuple(output.shape))
        return output
