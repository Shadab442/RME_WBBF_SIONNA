"""Per-UE downlink KPIs (SINR, RSRP) across candidate sectors.

Uses Sionna's own GenerateOFDMChannel for a genuine per-subcarrier
(frequency-selective) channel -- the channel model must be built with the
per-sector antenna array, so the per-element channel this produces is physically
correct. Each sector's own current ElectricalDowntilt.weights() combines its
elements into that sector's port, per subcarrier.

No fixed rx-per-tx association anywhere: every sector is evaluated as a
candidate server for every UE, every time compute_ue_sinr_db()/
compute_ue_sinr_rsrp() is called.
"""

from typing import List

import numpy as np
import torch

from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import SystemLevelChannel
from sionna.phy.ofdm import ResourceGrid

from .electrical_downtilt import ElectricalDowntilt


class KpiCalculator:
    """
    :param channel_model: A Sionna tr38901 channel model (UMi/UMa/RMa)
    :param resource_grid: A :class:`~sionna.phy.ofdm.ResourceGrid`
    :param sector_etilts: One :class:`~functions.electrical_downtilt.ElectricalDowntilt`
        per sector, in the same order as the channel model's tx (BS) dimension.
    :param tx_power_w: Transmit power per subcarrier [W] -- i.e. the
        sector's total conducted power already divided by however many
        subcarriers share that power budget (the full channel bandwidth).
    :param noise_power_w: Noise power per subcarrier [W].
    """

    def __init__(
        self,
        channel_model: SystemLevelChannel,
        resource_grid: ResourceGrid,
        sector_etilts: List[ElectricalDowntilt],
        tx_power_w: float,
        noise_power_w: float,
    ):
        self.sector_etilts = sector_etilts
        self.tx_power_w = tx_power_w
        self.noise_power_w = noise_power_w
        self._ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid)

    def generate_h_freq(self, batch_size: int = 1) -> torch.Tensor:
        """Draws one channel realization (topology's current shadow fading
        and pathloss, plus a *fresh* fast-fading/multipath draw).

        Pass the returned tensor into ``compute_power_matrix_w``/
        ``compute_ue_sinr_db``/``compute_ue_sinr_rsrp`` to evaluate several
        tilts against this exact same realization -- the common-random-numbers
        pattern needed for a clean tilt study, where only the tilt should
        differ between comparisons, not the channel.

        # h_freq: [batch, num_ue, num_ue_ant, num_sectors, M, num_ofdm_symbols, num_subcarriers]
        """
        h_freq = self._ofdm_channel(batch_size)
        assert h_freq.shape[2] == 1, (
            f"KpiCalculator assumes a single UE antenna (num_ue_ant == 1); "
            f"got {h_freq.shape[2]}. A multi-antenna UE would need receive "
            "combining, which isn't implemented here."
        )
        return h_freq

    def compute_power_matrix_w(self, h_freq: torch.Tensor = None, batch_size: int = 1) -> torch.Tensor:
        """Received power [num_sectors, batch, num_ue] in Watts, averaged
        over the resource grid, at each sector's *current* electrical
        downtilt. ``batch`` is a real, vectorized dimension: e.g. several
        independent (topology, channel) realizations drawn at once via
        ``generate_h_freq(batch_size=...)``, not just a repeated draw.

        :param h_freq: A channel realization from ``generate_h_freq()``, to
            evaluate the current tilts against a specific, fixed realization
            (or batch of realizations). If `None` (default), a fresh one is
            drawn internally.
        """
        if h_freq is None:
            h_freq = self.generate_h_freq(batch_size)
        num_sectors = len(self.sector_etilts)
        batch, num_ue = h_freq.shape[0], h_freq.shape[1]

        power = torch.zeros(num_sectors, batch, num_ue, dtype=h_freq.real.dtype, device=h_freq.device)
        for s, etilt in enumerate(self.sector_etilts):
            w = etilt.weights().to(h_freq.dtype)  # [M], complex
            h_s = h_freq[:, :, 0, s, :, :, :]  # [batch, num_ue, M, num_ofdm_symbols, num_subcarriers]
            combined = torch.einsum("e,buest->bust", w, h_s)  # [batch, num_ue, num_ofdm_symbols, num_subcarriers]
            power[s] = self.tx_power_w * (combined.abs() ** 2).mean(dim=(-2, -1))
        return power

    def compute_ue_sinr_db(self, h_freq: torch.Tensor = None, batch_size: int = 1) -> np.ndarray:
        """Per-UE SINR [dB], shape [batch, num_ue]: for every UE, every
        sector is evaluated as a candidate server (signal = that sector's
        power, interference = the sum of every other sector's power); the
        reported value is whichever candidate sector gives the highest SINR.

        :param h_freq: see ``compute_power_matrix_w``.
        """
        sinr_db, _ = self.compute_ue_sinr_rsrp(h_freq, batch_size)
        return sinr_db

    def compute_ue_sinr_rsrp(self, h_freq: torch.Tensor = None, batch_size: int = 1):
        """Per-UE SINR [dB] and RSRP [dBm], both shape [batch, num_ue] --
        RSRP is the *serving* sector's own received power (same argmax-SINR
        sector as compute_ue_sinr_db, not a separate argmax-power sector),
        so it answers "how much power is the sector a UE would actually
        attach to delivering," not just its ratio to interference. SINR can
        stay comparable across very different tilts even where RSRP shows a
        real coverage hole, since a shared tilt suppresses interference by
        a similar factor to signal -- RSRP doesn't get that cancellation.

        :param h_freq: see ``compute_power_matrix_w``.
        :output: (sinr_db, rsrp_dbm), each a numpy array [batch, num_ue].
        """
        power_w = self.compute_power_matrix_w(h_freq, batch_size)  # [num_sectors, batch, num_ue]
        total_w = power_w.sum(dim=0, keepdim=True)
        interference_w = total_w - power_w
        sinr = power_w / (interference_w + self.noise_power_w)
        best_sinr, best_idx = sinr.max(dim=0)  # [batch, num_ue]
        best_power_w = torch.gather(power_w, 0, best_idx.unsqueeze(0)).squeeze(0)  # [batch, num_ue]
        sinr_db = (10 * torch.log10(best_sinr)).detach().cpu().numpy()
        rsrp_dbm = (10 * torch.log10(best_power_w) + 30).detach().cpu().numpy()  # W -> dBm
        return sinr_db, rsrp_dbm
