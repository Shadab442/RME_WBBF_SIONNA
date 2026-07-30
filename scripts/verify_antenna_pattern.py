"""Verification plots for helpers/electrical_downtilt.py.

Run: python scripts/verify_antenna_pattern.py

Saves seven figures to results/antenna_pattern/:
  1. element_pattern.png             -- single element pattern (omni vs TR
                                         38.901), azimuth + elevation cuts,
                                         checked point-by-point against an
                                         independent reference formula.
  2. element_pattern_3d.png          -- same element pattern, as a 3D surface
                                         colored by gain.
  3. panel_vs_element.png            -- element-only vs full-array gain at
                                         boresight (downtilt=0), fixed M, for
                                         both the "rectangular" (eq. 7.3-1's
                                         own uniform amplitude) and "hanning"
                                         windows -- shows the directivity/
                                         sidelobe tradeoff between them.
  4. panel_pattern_3d_rectangular.png -- combined array pattern (rectangular
                                         window), as a 3D surface.
  4b. panel_pattern_3d_hanning.png    -- same, but with the hanning window.
  5. tilt_effect.png                 -- effect of electrical downtilt on the
                                         array gain pattern, fixed M
                                         (rectangular window) -- the beam
                                         should visibly rotate and peak at
                                         each requested downtilt.
  6. num_elements_effect.png         -- effect of the number of vertical
                                         elements M on the pattern for one
                                         fixed, off-boresight downtilt
                                         (rectangular window) -- more elements
                                         should narrow the main lobe (and add
                                         sidelobes) while the peak stays at
                                         the same angle.
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sionna.phy.channel.tr38901 import AntennaElement, AntennaArray
from helpers.electrical_downtilt import ElectricalDowntilt

CARRIER_FREQUENCY = 3.5e9
THETA = torch.linspace(0.01, np.pi - 0.01, 721)
PHI0 = torch.zeros_like(THETA)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "antenna_pattern")
os.makedirs(OUT_DIR, exist_ok=True)


def to_db(gain: torch.Tensor) -> np.ndarray:
    return 10 * np.log10(np.clip(gain.detach().cpu().numpy(), 1e-12, None))


# TR 38.901 Table 7.3-1 constants, hardcoded independently of AntennaElement so the
# reference curve below is a genuine second implementation to check against, not a
# restatement of the same code.
THETA_3DB = PHI_3DB = 65.0
SLA_V = A_MAX = 30.0
G_E_MAX = 8.0


def reference_38901_gain_db(theta_deg, phi_deg):
    """Independent re-implementation of TR 38.901 Table 7.3-1 (plain numpy, degrees
    in/out), used only to verify AntennaElement('38.901') against the equation."""
    a_v = -np.minimum(12.0 * ((theta_deg - 90.0) / THETA_3DB) ** 2, SLA_V)
    a_h = -np.minimum(12.0 * (phi_deg / PHI_3DB) ** 2, A_MAX)
    return G_E_MAX - np.minimum(-(a_v + a_h), A_MAX)


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


# 1. Element pattern: azimuth cut + vertical cut, each checked against the
#    independently-coded reference formula above (not just plotting AntennaElement
#    and trusting it).
theta_deg = np.linspace(0.0, 180.0, 721)
phi_deg = np.linspace(-180.0, 180.0, 721)
theta_rad = torch.tensor(theta_deg * np.pi / 180.0)
phi_rad = torch.tensor(phi_deg * np.pi / 180.0)

fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(13, 7.5))


def mark_point(ax, angle_deg, r_db, text, dx=18, dy=14, color="black"):
    """Places a marker exactly ON the curve at (angle_deg, r_db), with a text
    label connected to that same point by an arrow -- so the label is anchored
    to a real, verified point on the curve, not a floating reference line."""
    xy = (angle_deg * np.pi / 180.0, r_db)
    ax.plot(*xy, "o", color=color, markersize=5, zorder=5)
    ax.annotate(text, xy=xy, xytext=(dx, dy), textcoords="offset points",
               fontsize=7.5, ha="left",
               arrowprops=dict(arrowstyle="->", color=color, lw=0.8))


def mark_span(ax, angle_a_deg, r_a, angle_b_deg, r_b):
    """Double-headed arrow directly between two verified on-curve points."""
    ax.annotate("", xy=(angle_b_deg * np.pi / 180.0, r_b),
               xytext=(angle_a_deg * np.pi / 180.0, r_a),
               arrowprops=dict(arrowstyle="<->", color="green", lw=1.1))


# -- Azimuth cut: theta = 90 deg (horizontal plane), phi swept -180..180 --
ax = axes[0]
theta_fixed_90 = torch.full_like(phi_rad, np.pi / 2)
gain_az = {}
for pattern in ["omni", "38.901"]:
    f_theta, f_phi = AntennaElement(pattern).field(theta_fixed_90, phi_rad)
    gain_az[pattern] = to_db(f_theta ** 2 + f_phi ** 2)
    ax.plot(phi_rad.numpy(), gain_az[pattern], label=f"element gain ({pattern})")
ref_az = reference_38901_gain_db(90.0, phi_deg)
ax.plot(phi_rad.numpy(), ref_az, "k--", linewidth=1, label="reference eq. (Table 7.3-1)")
ax.set_theta_zero_location("E")

# All four points below are exact solutions of the eq. (verified to match the
# plotted curve to ~1e-5 dB), so each marker sits exactly ON the curve.
mark_point(ax, 0, G_E_MAX, f"peak: {G_E_MAX:.0f} dBi", dx=10, dy=22)
mark_point(ax, PHI_3DB / 2, G_E_MAX - 3,
          f"-3 dB\n$\\Delta\\phi$={PHI_3DB/2:.1f}°")
mark_point(ax, PHI_3DB, G_E_MAX - 12,
          f"-12 dB\n$\\Delta\\phi$={PHI_3DB:.0f}°=$\\phi_{{3dB}}$\n"
          f"12$\\cdot(\\Delta\\phi/\\phi_{{3dB}})^2$=12")
mark_point(ax, 180, reference_38901_gain_db(90.0, 180.0),
          f"floor: {reference_38901_gain_db(90.0, 180.0):.0f} dBi\n(-{A_MAX:.0f} dB from peak)",
          dx=-70, dy=18)
mark_span(ax, 0, G_E_MAX, 180, reference_38901_gain_db(90.0, 180.0))
ax.set_title("Azimuth cut ($\\theta=90°$)", fontsize=10)
ax.legend(fontsize=7, loc="lower left")

# -- Vertical (elevation) cut: phi = 0 deg, theta swept 0..180 --
ax = axes[1]
phi_fixed_0 = torch.zeros_like(theta_rad)
gain_el = {}
for pattern in ["omni", "38.901"]:
    f_theta, f_phi = AntennaElement(pattern).field(theta_rad, phi_fixed_0)
    gain_el[pattern] = to_db(f_theta ** 2 + f_phi ** 2)
    ax.plot(theta_rad.numpy(), gain_el[pattern], label=f"element gain ({pattern})")
ref_el = reference_38901_gain_db(theta_deg, 0.0)
ax.plot(theta_rad.numpy(), ref_el, "k--", linewidth=1, label="reference eq. (Table 7.3-1)")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

end_fire_db = reference_38901_gain_db(180.0, 0.0)
mark_point(ax, 90, G_E_MAX, f"peak: {G_E_MAX:.0f} dBi", dx=10, dy=22)
mark_point(ax, 90 + THETA_3DB / 2, G_E_MAX - 3,
          f"-3 dB\n$\\Delta\\theta$={THETA_3DB/2:.1f}°")
mark_point(ax, 90 + THETA_3DB, G_E_MAX - 12,
          f"-12 dB\n$\\Delta\\theta$={THETA_3DB:.0f}°=$\\theta_{{3dB}}$\n"
          f"12$\\cdot(\\Delta\\theta/\\theta_{{3dB}})^2$=12")
mark_point(ax, 180, end_fire_db,
          f"endfire: {end_fire_db:.1f} dBi\n(-{G_E_MAX - end_fire_db:.1f} dB from peak,\n"
          f"floor NOT reached, needs -{A_MAX:.0f} dB)",
          dx=-100, dy=15)
mark_span(ax, 90, G_E_MAX, 180, end_fire_db)
ax.set_title("Vertical/elevation cut ($\\phi=0°$)", fontsize=10)
ax.legend(fontsize=7, loc="lower left")

fig.suptitle("TR 38.901 Element Pattern Verification", fontsize=13)

# Numeric verification: AntennaElement('38.901') vs. the independently-coded
# reference equation, plus explicit 3dB-beamwidth and floor checks. Rendered onto
# the figure itself (not just printed) so the verification travels with the plot.
def find_3db_angle(angle_deg, gain_db, peak_db, positive_side=True):
    target = peak_db - 3.0
    mask = angle_deg >= 0 if positive_side else angle_deg <= 0
    idx = np.argmin(np.abs(gain_db[mask] - target))
    return angle_deg[mask][idx]


max_diff_az = np.max(np.abs(gain_az["38.901"] - ref_az))
max_diff_el = np.max(np.abs(gain_el["38.901"] - ref_el))
az_3db = find_3db_angle(phi_deg, ref_az, G_E_MAX)
el_3db = find_3db_angle(theta_deg - 90.0, ref_el, G_E_MAX)
az_floor_val = reference_38901_gain_db(90.0, 180.0)
el_edge_val = reference_38901_gain_db(0.0, 0.0)

verification_text = (
    f"Verification vs. reference eq. (Table 7.3-1) -- max diff: "
    f"azimuth {max_diff_az:.1e} dB, vertical {max_diff_el:.1e} dB (both $\\approx$0)\n"
    f"3dB angle found: azimuth $\\pm${az_3db:.1f}° (exp. $\\pm${PHI_3DB/2:.1f}°), "
    f"vertical $\\pm${el_3db:.1f}° (exp. $\\pm${THETA_3DB/2:.1f}°)\n"
    f"Floor check: azimuth reaches {az_floor_val:.1f} dBi at $\\phi$=180° (floor IS reached); "
    f"vertical reaches {el_edge_val:.1f} dBi at $\\theta$=0° (floor NOT reached)"
)
fig.text(0.5, -0.05, verification_text, ha="center", va="top", fontsize=8.5, wrap=True)

save(fig, "element_pattern.png")

# 1c. Element pattern: full 3D radiation pattern (TR 38.901), shape = linear gain,
#     color = gain in dB (heatmap), matching standard antenna-pattern 3D plots.
theta3d_deg = np.linspace(0.0, 180.0, 91)
phi3d_deg = np.linspace(-180.0, 180.0, 181)
phi3d_grid_deg, theta3d_grid_deg = np.meshgrid(phi3d_deg, theta3d_deg, indexing="xy")
theta3d_grid = torch.tensor(theta3d_grid_deg * np.pi / 180.0)
phi3d_grid = torch.tensor(phi3d_grid_deg * np.pi / 180.0)

f_theta, f_phi = AntennaElement("38.901").field(theta3d_grid, phi3d_grid)
gain_linear_3d = (f_theta ** 2 + f_phi ** 2).detach().cpu().numpy()
gain_db_3d = to_db(f_theta ** 2 + f_phi ** 2)

theta_np, phi_np = theta3d_grid.cpu().numpy(), phi3d_grid.cpu().numpy()
x = gain_linear_3d * np.sin(theta_np) * np.cos(phi_np)
y = gain_linear_3d * np.sin(theta_np) * np.sin(phi_np)
z = gain_linear_3d * np.cos(theta_np)

norm = mcolors.Normalize(vmin=gain_db_3d.min(), vmax=gain_db_3d.max())
surf_colors = cm.viridis(norm(gain_db_3d))

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.plot_surface(x, y, z, facecolors=surf_colors, rstride=1, cstride=1,
               linewidth=0, antialiased=True, shade=False)
ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Element Gain Pattern (TR 38.901), peak at boresight "
            "($\\theta$=90°, $\\phi$=0°)", fontsize=11)
ax.view_init(elev=25, azim=-60)

mappable = cm.ScalarMappable(cmap="viridis", norm=norm)
mappable.set_array(gain_db_3d)
fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Power (dB)")

save(fig, "element_pattern_3d.png")

# 2. Element-only vs full-panel gain at boresight (downtilt = 0 deg), rectangular
#    vs. hanning window -- shows the sidelobe-suppression/peak-gain tradeoff
#    directly: hanning gives up some peak gain for far fewer/shallower nulls.
M = 8
etilts = {window: make_etilt(M, window=window) for window in ("rectangular", "hanning")}
for etilt in etilts.values():
    etilt.set_tilt(0.0)
f_theta, f_phi = etilts["rectangular"].array.ant_pol1.field(THETA, PHI0)
fig, ax = new_polar_fig(f"Panel vs. element gain (downtilt = 0°, M={M})")
ax.plot(THETA.numpy(), to_db(f_theta ** 2 + f_phi ** 2), label="element only")
for window, etilt in etilts.items():
    ax.plot(THETA.numpy(), to_db(etilt.gain_pattern(THETA, PHI0)), label=f"panel ({window}, M={M})")
ax.legend(fontsize=8)
save(fig, "panel_vs_element.png")
for window, etilt in etilts.items():
    peak_gain = etilt.array_factor(torch.tensor([np.pi / 2])).item()
    print(f"[2] peak array gain at boresight ({window}): {peak_gain:.2f} "
         f"({10 * np.log10(peak_gain):.2f} dB)")

# 2b. Composite panel pattern in 3D (M=8, downtilt=0), one figure per window,
#     same style as the element 3D plot, but the panel has much more dynamic
#     range (sidelobes/nulls from the array factor) than the element's ~23 dB --
#     a linear-gain radius would make the sidelobes invisible (collapsed to a
#     sliver next to the main lobe). So here the SHAPE is dB-with-floor (dB
#     above a fixed dB-below-peak floor, clipped at 0) instead of linear gain,
#     which is what actually makes the array's sidelobe structure visible;
#     color is still gain in dB. Rectangular and hanning use the same floor
#     (relative to each one's own peak) so the two figures are visually
#     comparable despite hanning's lower peak gain.
FLOOR_BELOW_PEAK_DB = 40.0
for window, etilt in etilts.items():
    etilt.set_tilt(0.0)
    grid_shape = theta3d_grid.shape
    gain_panel_flat = etilt.gain_pattern(theta3d_grid.reshape(-1), phi3d_grid.reshape(-1))
    gain_panel_db_3d = to_db(gain_panel_flat).reshape(grid_shape)

    peak_db_panel = gain_panel_db_3d.max()
    radius_panel = np.clip(gain_panel_db_3d - (peak_db_panel - FLOOR_BELOW_PEAK_DB), 0, None)

    x_p = radius_panel * np.sin(theta_np) * np.cos(phi_np)
    y_p = radius_panel * np.sin(theta_np) * np.sin(phi_np)
    z_p = radius_panel * np.cos(theta_np)

    # Color normalization matches the same floor as the shape -- otherwise the
    # deep nulls would dominate the color scale and everything visible would
    # look the same.
    norm_p = mcolors.Normalize(vmin=peak_db_panel - FLOOR_BELOW_PEAK_DB, vmax=peak_db_panel)
    surf_colors_p = cm.viridis(norm_p(np.clip(gain_panel_db_3d, peak_db_panel - FLOOR_BELOW_PEAK_DB, None)))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(x_p, y_p, z_p, facecolors=surf_colors_p, rstride=1, cstride=1,
                   linewidth=0, antialiased=True, shade=False)
    ax.set_box_aspect((np.ptp(x_p), np.ptp(y_p), np.ptp(z_p)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D Composite Panel Gain Pattern ({window}, M={M}, downtilt=0°)", fontsize=11)
    ax.view_init(elev=25, azim=-60)

    mappable_p = cm.ScalarMappable(cmap="viridis", norm=norm_p)
    mappable_p.set_array(gain_panel_db_3d)
    fig.colorbar(mappable_p, ax=ax, shrink=0.6, pad=0.1, label="Power (dB)")

    save(fig, f"panel_pattern_3d_{window}.png")

etilt = etilts["rectangular"]  # used unchanged by sections 3-4 below

# 3. Effect of electrical downtilt, fixed M
fig, ax = new_polar_fig(f"Effect of electrical downtilt (M={M})")
for downtilt_deg in [-10, -5, 0, 5, 10]:
    etilt.set_tilt(downtilt_deg)
    ax.plot(THETA.numpy(), to_db(etilt.gain_pattern(THETA, PHI0)), label=f"{downtilt_deg:+.0f}°")
ax.legend(fontsize=8, loc="lower left")
save(fig, "tilt_effect.png")

# 4. Effect of number of elements M, fixed off-boresight downtilt
FIXED_DOWNTILT = -5.0
fig, ax = new_polar_fig(f"Effect of M (downtilt = {FIXED_DOWNTILT:+.0f}°)")
for num_elements in [2, 4, 8, 16]:
    e = make_etilt(num_elements)
    e.set_tilt(FIXED_DOWNTILT)
    ax.plot(THETA.numpy(), to_db(e.gain_pattern(THETA, PHI0)), label=f"M={num_elements}")
ax.legend(fontsize=8)
save(fig, "num_elements_effect.png")
