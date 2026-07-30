#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Electrical downtilt (3GPP TR 38.901, clause 7.3.1) for a single-port antenna array."""

from typing import Union

import torch

from sionna.phy import PI, SPEED_OF_LIGHT
# Re-exported for convenience -- a single antenna import surface for this repo.
from sionna.phy.channel.tr38901 import AntennaElement, PanelArray, Antenna, AntennaArray


WINDOWS = ("rectangular", "hanning")


class ElectricalDowntilt:
    """Electrical downtilt per 3GPP TR 38.901, clause 7.3.1:

        w_m = (1/sqrt(M)) * exp(-j*2*pi*(m-1)*d_V/lambda*cos(theta_etilt)), m = 1..M

    Steers a single-column Sionna ``PanelArray``/``AntennaArray``/``Antenna``
    electronically -- the panel itself never physically rotates.

    The ``1/sqrt(M)`` amplitude term above is eq. (7.3-1)'s own literal,
    uniform ("rectangular") amplitude weighting -- the default here, matching
    the reference formula exactly. ``window="hanning"`` is an extension
    beyond eq. (7.3-1) (which only defines the phase/steering side): it
    replaces that uniform amplitude with a Hanning taper (normalized to the
    same unit total transmit power, ``sum(|w_m|^2) == 1``, so switching
    window doesn't change total radiated power) to trade some mainlobe gain
    and width for lower sidelobes/nulls.

    :param array: A Sionna TR 38.901 array with a single column
        (``num_cols_per_panel == 1``) and single polarization.
    :param carrier_frequency: Carrier frequency [Hz].
    :param downtilt_deg: Electrical downtilt [degrees]: 0 = boresight, + = down,
        - = up. Converted internally to eq. (7.3-1)'s zenith-angle convention
        (``theta_etilt = 90 + downtilt_deg``); see :attr:`theta_etilt_deg` for
        the raw value.
    :param window: Per-element amplitude taper -- ``"rectangular"`` (default,
        eq. 7.3-1's own uniform amplitude) or ``"hanning"``.
    """

    def __init__(
        self,
        array: Union[PanelArray, Antenna, AntennaArray],
        carrier_frequency: float,
        downtilt_deg: float = 0.0,
        window: str = "rectangular",
    ):
        assert array.num_cols_per_panel == 1, \
            "ElectricalDowntilt implements TR 38.901 eq. (7.3-1) for a single " \
            "vertical column (num_cols_per_panel == 1); got a 2D panel."
        assert array.polarization == 'single', \
            "ElectricalDowntilt currently only supports single polarization."
        assert window in WINDOWS, f"window must be one of {WINDOWS}; got {window!r}."

        self.array = array
        self.window = window
        wavelength = SPEED_OF_LIGHT / carrier_frequency
        # Element z-positions [multiples of wavelength] (array.ant_pos is in meters)
        self._element_pos_z = array.ant_pos[:, 2] / wavelength
        self._amplitude = self._compute_amplitude()
        self.set_tilt(downtilt_deg)

    def _compute_amplitude(self) -> torch.Tensor:
        """Per-element real amplitude taper, normalized so sum(a_m^2) == 1 --
        i.e. unit total transmit power regardless of window, so switching
        window is a pure pattern-shape change, not a power change."""
        dtype, device = self._element_pos_z.dtype, self._element_pos_z.device
        num_elements = self.num_elements
        if self.window == "rectangular":
            amplitude = torch.ones(num_elements, dtype=dtype, device=device)
        elif self.window == "hanning":
            amplitude = torch.hann_window(num_elements, periodic=False, dtype=dtype, device=device)
        return amplitude / torch.sqrt(torch.sum(amplitude ** 2))

    def set_tilt(self, downtilt_deg: float) -> None:
        """Set the electrical downtilt [degrees; 0 = boresight, + = down, - = up]"""
        self._downtilt_deg = float(downtilt_deg)

    @property
    def num_elements(self) -> int:
        """Number of elements M in the array"""
        return self.array.num_ant

    @property
    def downtilt_deg(self) -> float:
        """Current electrical downtilt [degrees; 0 = boresight, + = down, - = up]"""
        return self._downtilt_deg

    @property
    def theta_etilt_deg(self) -> float:
        """Current tilt, in TR 38.901's own zenith-angle convention (eq. 7.3-1):
        0-180 deg, 90 deg = perpendicular to the array (boresight)."""
        return 90.0 + self._downtilt_deg

    def weights(self) -> torch.Tensor:
        """Per-element complex excitation weights: eq. (7.3-1)'s steering
        phase, combined with the amplitude taper set by ``window``
        (uniform/1/sqrt(M) for the default "rectangular", matching eq.
        (7.3-1) exactly)."""
        theta_etilt = torch.tensor(
            self.theta_etilt_deg * PI / 180.0,
            dtype=self._element_pos_z.dtype, device=self._element_pos_z.device)
        phase = -2 * PI * self._element_pos_z * torch.cos(theta_etilt)
        return self._amplitude * torch.exp(1j * phase)

    def array_factor(self, theta: torch.Tensor) -> torch.Tensor:
        """|array factor|^2 as a function of zenith angle theta [radian]

        Peaks at theta == theta_etilt_deg (both in the same zenith-from-array-axis
        convention). With the default "rectangular" window, peak gain equals
        the number of elements M (linear units) -- the expected coherent-
        combining gain for a uniformly-weighted M-element array. A tapered
        window (e.g. "hanning") trades some of that peak gain for lower
        sidelobes, so the peak will be below M.
        """
        theta = theta.to(dtype=self._element_pos_z.dtype, device=self._element_pos_z.device)
        w = self.weights()
        phase = 2 * PI * self._element_pos_z[:, None] * torch.cos(theta)[None, :]
        steering = torch.exp(1j * phase)
        af = torch.sum(w[:, None] * steering, dim=0)
        return torch.abs(af) ** 2

    def gain_pattern(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Combined element-pattern * array-factor gain (linear units)

        Note: this single-column array has no azimuth steering, so its azimuth
        pattern is just the element pattern alone; ``phi`` is passed straight
        through to the element pattern.
        """
        f_theta, f_phi = self.array.ant_pol1.field(theta, phi)
        element_gain = f_theta ** 2 + f_phi ** 2
        return element_gain * self.array_factor(theta)
