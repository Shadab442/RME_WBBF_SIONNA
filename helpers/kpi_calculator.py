"""Per-UE downlink KPIs (SINR, RSRP) across candidate sectors.

Large-scale only: pathloss + shadow fading, no fast/frequency-selective
fading. At the tilt-control timescale (hours-to-a-day), a control loop only
ever acts on a population/time-aggregated KPI, over which fast fading has
already averaged out; modeling it per-decision only adds noise unrelated to
what a coverage/SINR-percentile metric is meant to measure -- see the design
discussion this was built from (converged on independently by every RET
paper checked: Buenestado, Dandanov, Partov, Vannella, Shafin et al.).

Pathloss and shadow fading come directly from Sionna's own 3GPP TR 38.901
scenario/LSP machinery (``channel_model._lsp_sampler``/``._lsp``), reusing
whatever the channel model itself already cached for the current topology --
no channel impulse response is ever generated. Antenna gain toward each UE
is read directly off each sector's ElectricalDowntilt radiation pattern at
the link's LOS angle, rotated from the global frame into the antenna's own
local frame via 3GPP TR 38.901 eq. (7.1-4)/(7.1-7)/(7.1-8) -- mirrors
Sionna's own (private) ChannelCoefficientsGenerator._gcs_to_lcs, reimplemented
here since it isn't part of Sionna's public API.

No fixed rx-per-tx association anywhere: every sector is evaluated as a
candidate server for every UE, every time compute_ue_sinr_db()/
compute_ue_sinr_rsrp() is called.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from sionna.phy.channel.tr38901 import SystemLevelChannel
from sionna.phy.channel.utils import deg_2_rad

from .electrical_downtilt import ElectricalDowntilt


def _forward_rotation_matrix(orientations: torch.Tensor) -> torch.Tensor:
    """3GPP TR 38.901 eq. (7.1-4): composite yaw/pitch/roll rotation matrix.

    :param orientations: [..., 3] (yaw, pitch, roll) [radian].
    :output: [..., 3, 3].
    """
    a, b, c = orientations[..., 0], orientations[..., 1], orientations[..., 2]
    row_1 = torch.stack([
        torch.cos(a) * torch.cos(b),
        torch.cos(a) * torch.sin(b) * torch.sin(c) - torch.sin(a) * torch.cos(c),
        torch.cos(a) * torch.sin(b) * torch.cos(c) + torch.sin(a) * torch.sin(c),
    ], dim=-1)
    row_2 = torch.stack([
        torch.sin(a) * torch.cos(b),
        torch.sin(a) * torch.sin(b) * torch.sin(c) + torch.cos(a) * torch.cos(c),
        torch.sin(a) * torch.sin(b) * torch.cos(c) - torch.cos(a) * torch.sin(c),
    ], dim=-1)
    row_3 = torch.stack([
        -torch.sin(b),
        torch.cos(b) * torch.sin(c),
        torch.cos(b) * torch.cos(c),
    ], dim=-1)
    return torch.stack([row_1, row_2, row_3], dim=-2)


def gcs_to_lcs(orientations: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor):
    """3GPP TR 38.901 eq. (7.1-7)/(7.1-8): rotate a global-frame (theta, phi)
    direction into an array's local frame given its (yaw, pitch, roll)
    ``orientations`` -- e.g. a sector's boresight + mechanical downtilt.

    :param orientations: [..., 3] (yaw, pitch, roll) [radian], broadcastable
        against theta/phi.
    :param theta: Zenith [radian].
    :param phi: Azimuth [radian], same shape as theta.
    :output: (theta_prime, phi_prime), each theta's shape.
    """
    rho_hat = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1).unsqueeze(-1)  # [..., 3, 1]
    rot_inv = _forward_rotation_matrix(orientations).mT
    rot_rho = torch.matmul(rot_inv, rho_hat)  # [..., 3, 1]
    z = rot_rho[..., 2, 0].clamp(-1.0, 1.0)
    theta_prime = torch.acos(z)
    phi_prime = torch.atan2(rot_rho[..., 1, 0], rot_rho[..., 0, 0])
    return theta_prime, phi_prime


@dataclass
class LargeScaleState:
    """Pathloss+shadow-fading and each sector's LOS angle (in its own local
    frame), for the topology's CURRENT positions -- everything needed to
    compute received power at any tilt, without redrawing per tilt (tilt
    only changes antenna gain, never pathloss/shadowing).

    :ivar total_pathloss_db: [batch, num_sectors, num_ue].
    :ivar theta_lcs: [batch, num_sectors, num_ue] zenith [radian], in each
        sector's own local frame.
    :ivar phi_lcs: [batch, num_sectors, num_ue] azimuth [radian], likewise.
    """
    total_pathloss_db: torch.Tensor
    theta_lcs: torch.Tensor
    phi_lcs: torch.Tensor


class KpiCalculator:
    """
    :param channel_model: A Sionna tr38901 channel model (UMi/UMa/RMa).
    :param sector_etilts: One :class:`~helpers.electrical_downtilt.ElectricalDowntilt`
        per sector, in the same order as the channel model's tx (BS) dimension.
    :param tx_power_w: Total sector transmit power [W].
    :param noise_power_w: Total-bandwidth noise power [W].
    """

    def __init__(
        self,
        channel_model: SystemLevelChannel,
        sector_etilts: List[ElectricalDowntilt],
        tx_power_w: float,
        noise_power_w: float,
    ):
        self.channel_model = channel_model
        self.sector_etilts = sector_etilts
        self.tx_power_w = tx_power_w
        self.noise_power_w = noise_power_w

    def generate_large_scale_state(self) -> LargeScaleState:
        """Pathloss+shadow fading for the topology's CURRENT positions, plus
        each sector's LOS angle in its own local frame -- no fast/frequency-
        selective fading (see module docstring).

        Shadow fading is whatever ``channel_model.set_topology()`` already
        cached for this topology (``channel_model._lsp.sf``) -- the same
        draw the old fast-fading path would have reused across a tilt sweep,
        just without a channel impulse response layered on top of it.

        Pass the returned state into ``compute_power_matrix_w``/
        ``compute_ue_sinr_db``/``compute_ue_sinr_rsrp``/``compute_power_table``
        to evaluate several tilts against this exact same large-scale
        realization -- the common-random-numbers pattern needed for a clean
        tilt study, where only the tilt should differ between comparisons.
        """
        scenario = self.channel_model._scenario
        pathloss_db = self.channel_model._lsp_sampler.sample_pathloss()  # [batch, num_bs, num_ue]
        shadow_fading_db = 10.0 * torch.log10(self.channel_model._lsp.sf)
        total_pathloss_db = pathloss_db + shadow_fading_db

        theta_lcs, phi_lcs = gcs_to_lcs(
            scenario.bs_orientations.unsqueeze(2),  # [batch, num_bs, 1, 3]
            deg_2_rad(scenario.los_zod), deg_2_rad(scenario.los_aod),
        )
        return LargeScaleState(total_pathloss_db, theta_lcs, phi_lcs)

    def compute_power_matrix_w(self, state: LargeScaleState = None) -> torch.Tensor:
        """Received power [num_sectors, batch, num_ue] in Watts, at each
        sector's *current* electrical downtilt. ``batch`` is a real,
        vectorized dimension -- the topology's own batch of independent
        realizations, not a repeated draw.

        :param state: A large-scale state from ``generate_large_scale_state()``,
            to evaluate the current tilts against a specific realization. If
            `None` (default), a fresh one is drawn internally.
        """
        if state is None:
            state = self.generate_large_scale_state()
        num_sectors = len(self.sector_etilts)
        batch, num_ue = state.total_pathloss_db.shape[0], state.total_pathloss_db.shape[2]

        power = torch.zeros(num_sectors, batch, num_ue,
                            dtype=state.total_pathloss_db.dtype, device=state.total_pathloss_db.device)
        for s, etilt in enumerate(self.sector_etilts):
            # gain_pattern only supports flat angle arrays (see
            # verify_tilt_effect.py's 1-D sweeps) -- flatten, evaluate, reshape.
            theta_flat = state.theta_lcs[:, s, :].reshape(-1)
            phi_flat = state.phi_lcs[:, s, :].reshape(-1)
            gain = etilt.gain_pattern(theta_flat, phi_flat).reshape(batch, num_ue)
            pathloss_lin = 10.0 ** (state.total_pathloss_db[:, s, :] / 10.0)
            power[s] = self.tx_power_w * gain / pathloss_lin
        return power

    def compute_ue_sinr_db(self, state: LargeScaleState = None) -> np.ndarray:
        """Per-UE SINR [dB], shape [batch, num_ue]: for every UE, every
        sector is evaluated as a candidate server (signal = that sector's
        power, interference = the sum of every other sector's power); the
        reported value is whichever candidate sector gives the highest SINR.

        :param state: see ``compute_power_matrix_w``.
        """
        sinr_db, _ = self.compute_ue_sinr_rsrp(state)
        return sinr_db

    def compute_ue_sinr_rsrp(self, state: LargeScaleState = None, return_serving_idx: bool = False):
        """Per-UE SINR [dB] and RSRP [dBm], both shape [batch, num_ue] --
        RSRP is the *serving* sector's own received power (same argmax-SINR
        sector as compute_ue_sinr_db, not a separate argmax-power sector),
        so it answers "how much power is the sector a UE would actually
        attach to delivering," not just its ratio to interference. SINR can
        stay comparable across very different tilts even where RSRP shows a
        real coverage hole, since a shared tilt suppresses interference by
        a similar factor to signal -- RSRP doesn't get that cancellation.

        :param state: see ``compute_power_matrix_w``.
        :param return_serving_idx: if True, also return each UE's serving
            (argmax-SINR) sector index -- e.g. for a tilt controller that
            needs to know which sector's attached population a UE belongs
            to, not just its SINR.
        :output: (sinr_db, rsrp_dbm), or (sinr_db, rsrp_dbm, serving_idx) if
            return_serving_idx is True. sinr_db/rsrp_dbm are numpy arrays
            [batch, num_ue]; serving_idx is a numpy int array [batch, num_ue].
        """
        power_w = self.compute_power_matrix_w(state)  # [num_sectors, batch, num_ue]
        total_w = power_w.sum(dim=0, keepdim=True)
        interference_w = total_w - power_w
        sinr = power_w / (interference_w + self.noise_power_w)
        best_sinr, best_idx = sinr.max(dim=0)  # [batch, num_ue]
        best_power_w = torch.gather(power_w, 0, best_idx.unsqueeze(0)).squeeze(0)  # [batch, num_ue]
        sinr_db = (10 * torch.log10(best_sinr)).detach().cpu().numpy()
        rsrp_dbm = (10 * torch.log10(best_power_w) + 30).detach().cpu().numpy()  # W -> dBm
        if return_serving_idx:
            return sinr_db, rsrp_dbm, best_idx.detach().cpu().numpy()
        return sinr_db, rsrp_dbm

    def compute_power_table(self, tilt_values, state: LargeScaleState) -> np.ndarray:
        """[num_tilts, num_sectors, batch*num_ue] received power [W]. Row
        t, sector s is sector s's power at tilt_values[t] -- computed by
        applying tilt_values[t] to EVERY sector (an ordinary sweep step,
        via compute_power_matrix_w) and keeping only sector s's own row,
        which is valid regardless of what any other sector's tilt is (a
        sector's own power depends only on its own antenna weights and the
        channel to each UE). Any per-sector tilt assignment can then be
        evaluated by array lookup alone (see coverage_from_assignment) --
        no repeated channel computation.

        :param tilt_values: candidate downtilts [deg].
        :param state: a fixed large-scale state from
            ``generate_large_scale_state`` -- reused across every tilt in
            ``tilt_values`` (common random numbers), never redrawn here.
        """
        power_rows = []
        for downtilt_deg in tilt_values:
            for etilt in self.sector_etilts:
                etilt.set_tilt(downtilt_deg)
            power_w = self.compute_power_matrix_w(state)  # [num_sectors, batch, num_ue]
            power_rows.append(power_w.detach().cpu().numpy().reshape(power_w.shape[0], -1))
        return np.stack(power_rows, axis=0)

    def coverage_from_assignment(self, power_table: np.ndarray, tilt_idx_per_sector, threshold_db: float,
                                 return_details: bool = False, weights: np.ndarray = None):
        """Pooled coverage (fraction of samples with best-serving-sector
        SINR > threshold_db) for one per-sector tilt assignment, read
        directly off a power_table from compute_power_table -- no channel
        recomputation. Same signal/interference/SINR formula as
        compute_ue_sinr_rsrp, just applied to a pre-gathered numpy power
        table (letting an arbitrary per-sector assignment be scored) rather
        than a fresh torch computation from a large-scale state (which only
        supports every sector sharing the SAME current tilt).

        :param tilt_idx_per_sector: [num_sectors] int array/list, each
            entry an index into the tilt values ``power_table`` was built
            from.
        :param return_details: if True, also return each sample's SINR
            [dB] and serving-sector index (the argmax-SINR sector under
            THIS assignment).
        :param weights: [num_samples] optional per-sample weight (e.g. a
            radio map's grid points reweighted by estimated UE density --
            see helpers.radio_map.grid_density_weights). None (default)
            weights every sample equally, i.e. plain coverage.
        """
        num_sectors = power_table.shape[1]
        tilt_idx_per_sector = np.asarray(tilt_idx_per_sector, dtype=int)
        power_w = power_table[tilt_idx_per_sector, np.arange(num_sectors)]  # [num_sectors, num_samples]
        total_w = power_w.sum(axis=0, keepdims=True)
        interference_w = total_w - power_w
        sinr = power_w / (interference_w + self.noise_power_w)
        best_idx = sinr.argmax(axis=0)
        best_sinr = sinr.max(axis=0)
        sinr_db = 10.0 * np.log10(best_sinr)
        is_covered = sinr_db > threshold_db
        coverage = float(np.mean(is_covered)) if weights is None else float(np.average(is_covered, weights=weights))
        if return_details:
            return coverage, sinr_db, best_idx
        return coverage
