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

import logging
from dataclasses import dataclass

import torch

from sionna.phy.channel.tr38901 import SystemLevelChannel
from sionna.phy.channel.utils import deg_2_rad
from helpers.utils import get_logger

logger = get_logger(__name__)


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

        Reads the channel model's CACHED ``_lsp.sf`` -- it is only redrawn
        by the channel model's own ``set_topology()``, not by this call. Two
        calls in a row with no intervening ``set_topology()`` return
        identical shadow fading; this is intentional (repeated queries of
        the same topology snapshot should agree), not a missed resample.
        """
        logger.function("LargeScaleChannel.generate_state start")
        scenario = self.channel_model._scenario
        pathloss_db = self.channel_model._lsp_sampler.sample_pathloss()  # [batch, num_bs, num_ue]
        # Sionna's own SystemLevelChannel._step_12 applies sf as a LINEAR
        # POWER GAIN: amplitude *= 10**(-pathloss_db/20) * sqrt(sf). So a
        # sf > 1 (favorable shadowing) must REDUCE total loss -- subtract
        # its dB equivalent, don't add it.
        shadow_fading_db = 10.0 * torch.log10(self.channel_model._lsp.sf)
        total_pathloss_db = pathloss_db - shadow_fading_db
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("generate_state: total_pathloss_db mean=%.2f min=%.2f max=%.2f",
                        total_pathloss_db.mean().item(), total_pathloss_db.min().item(),
                        total_pathloss_db.max().item())

        theta_lcs, phi_lcs = self.channel_model._cir_sampler._gcs_to_lcs(
            scenario.bs_orientations.unsqueeze(2),  # [batch, num_bs, 1, 3]
            deg_2_rad(scenario.los_zod), deg_2_rad(scenario.los_aod),
        )
        logger.debug("generate_state: theta_lcs shape=%s phi_lcs shape=%s",
                    tuple(theta_lcs.shape), tuple(phi_lcs.shape))
        logger.function("LargeScaleChannel.generate_state end")
        return LargeScaleState(total_pathloss_db, theta_lcs, phi_lcs)
