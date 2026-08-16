"""Large-scale (pathloss + shadow fading, no fast/frequency-selective fading)
channel state, read directly off a Sionna channel model's own cached LSP/
topology state.

Pathloss and shadow fading come from Sionna's own 3GPP TR 38.901 LSP
machinery (``channel_model._lsp_sampler``/``._lsp``) 
Each sector's LOS angle (needed for an antenna gain-pattern lookup) 
is rotated from the global frame into the antenna's own local frame 
via Sionna's own ``ChannelCoefficientsGenerator._gcs_to_lcs`` 
(3GPP TR 38.901 eq. (7.1-7)/(7.1-8)), through ``channel_model._cir_sampler`` 
"""

from dataclasses import dataclass

import torch

from sionna.phy.channel.tr38901 import SystemLevelChannel
from sionna.phy.channel.utils import deg_2_rad


@dataclass
class LargeScaleState:
    """Pathloss+shadow-fading and each sector's LOS angle (in its own local
    frame), for the topology's CURRENT positions 

    :ivar total_pathloss_db: [batch, num_sectors, num_ue].
    :ivar theta_lcs: [batch, num_sectors, num_ue] zenith [radian], in each
        sector's own local frame.
    :ivar phi_lcs: [batch, num_sectors, num_ue] azimuth [radian], likewise.
    """
    total_pathloss_db: torch.Tensor
    theta_lcs: torch.Tensor
    phi_lcs: torch.Tensor


class LargeScaleChannel:
    """Wraps a Sionna channel model, exposing only its large-scale (pathloss
    + shadow fading + LOS angle) state

    :param channel_model: A Sionna tr38901 channel model (UMi/UMa/RMa).
    """

    def __init__(self, channel_model: SystemLevelChannel):
        self.channel_model = channel_model

    def generate_state(self) -> LargeScaleState:
        """Pathloss+shadow fading for the topology's CURRENT positions, plus
        each sector's LOS angle in its own local frame -- no fast/frequency-
        selective fading.
        """
        scenario = self.channel_model._scenario
        pathloss_db = self.channel_model._lsp_sampler.sample_pathloss()  # [batch, num_bs, num_ue]
        shadow_fading_db = 10.0 * torch.log10(self.channel_model._lsp.sf)
        total_pathloss_db = pathloss_db + shadow_fading_db

        theta_lcs, phi_lcs = self.channel_model._cir_sampler._gcs_to_lcs(
            scenario.bs_orientations.unsqueeze(2),  # [batch, num_bs, 1, 3]
            deg_2_rad(scenario.los_zod), deg_2_rad(scenario.los_aod),
        )
        return LargeScaleState(total_pathloss_db, theta_lcs, phi_lcs)
