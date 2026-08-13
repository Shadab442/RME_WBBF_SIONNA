"""Verification: effect of electrical downtilt on the array gain pattern.

Split out from verify_antenna_pattern.py -- that script verifies the
antenna pattern build-up itself (element -> panel, windows, element count);
this one isolates just the downtilt sweep, a different question (how does
an already-verified pattern rotate as tilt changes) from whether the
pattern's shape is correct in the first place.

Run: python scripts/verifications/verify_tilt_effect.py

Saves to results/verifications/tilt_effect/:
  tilt_effect.png -- array gain pattern (M=8, rectangular window) at
                     several downtilt settings -- the beam should visibly
                     rotate and peak at each requested downtilt.
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sionna.phy.channel.tr38901 import AntennaArray
from helpers.electrical_downtilt import ElectricalDowntilt

CARRIER_FREQUENCY = 3.5e9
THETA = torch.linspace(0.01, np.pi - 0.01, 721)
PHI0 = torch.zeros_like(THETA)
M = 8  # vertical elements

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "verifications", "tilt_effect")
os.makedirs(OUT_DIR, exist_ok=True)


def to_db(gain: torch.Tensor) -> np.ndarray:
    return 10 * np.log10(np.clip(gain.detach().cpu().numpy(), 1e-12, None))


def make_etilt(num_elements: int, vertical_spacing: float = 0.5, window: str = "rectangular") -> ElectricalDowntilt:
    array = AntennaArray(num_rows=num_elements, num_cols=1, polarization="single",
                         polarization_type="V", antenna_pattern="38.901",
                         carrier_frequency=CARRIER_FREQUENCY,
                         vertical_spacing=vertical_spacing)
    return ElectricalDowntilt(array, carrier_frequency=CARRIER_FREQUENCY, window=window)


def new_polar_fig(title):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=11)
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


etilt = make_etilt(M, window="rectangular")

fig, ax = new_polar_fig(f"Effect of electrical downtilt (M={M})")
for downtilt_deg in [-10, -5, 0, 5, 10]:
    etilt.set_tilt(downtilt_deg)
    ax.plot(THETA.numpy(), to_db(etilt.gain_pattern(THETA, PHI0)), label=f"{downtilt_deg:+.0f}°")
ax.legend(fontsize=8, loc="lower left")
save(fig, "tilt_effect.png")
