"""StickyLosState: optional, removable approximation of 3GPP TR 38.901 Sec
7.6.3's spatial-consistency procedure for LOS/NLOS state.

Sionna's own stochastic LOS model (channel.force_los: null) redraws every
UE-sector link's LOS/NLOS state independently from scratch on EVERY
set_topology() call -- i.e. every tilt-control interval, regardless of
whether a UE actually moved -- since Sionna's tr38901 implementation does
not itself implement Sec 7.6.3's spatially-correlated-field procedure (no
"spatial consistency"/"drop" concept anywhere in that module, confirmed by
reading its source -- Sionna 2.0.0 ships the real Table 7.6.3.1-2-style
corrDist numbers in its own RMa_O2I.json model data but never actually
reads them anywhere; there is no enable_spatial_consistency option in this
version). That produces the exact "hard transition" artifact Sec 7.6.3.2
names as the reason the real spec wants LOS to be spatially correlated: a
UE's LOS state can flip every interval purely from randomness, never from
any real position/environment change.

This module does NOT implement the full spec procedure (a correlated 2D
random field evaluated at each UE's continuous position, per-scenario
correlation distance). It's a simpler, CLUSTER-level approximation
(2026-10-12, replacing an earlier per-UE distance-threshold design):
every member of a hotspot/RPGM cluster shares one LOS draw, held fixed for
as long as that cluster is in its HOLD period (its own mobility.ref_xy
reference point unchanged interval to interval -- a member's own small
per-step random walk within the cluster does NOT trigger a redraw), and
redrawn fresh for the WHOLE cluster the instant it jumps to a new
reference point. This matches the mobility model's own actual hold/jump
structure directly, rather than approximating it with a distance
threshold a member's own within-cluster wiggle could still accidentally
cross during a hold period.

Entirely opt-in and self-contained: SimulationEngine only ever touches this
through one guarded call (StickyLosState.apply, right after
channel_model.set_topology()), itself gated by self.sticky_los is not None
(built only when channel.sticky_los_enabled is true). Deleting this file,
its one call site, and the config key fully removes the feature with no
other code depending on it.
"""

import numpy as np
import torch


class StickyLosState:
    """Keeps per-(sector, UE) LOS/NLOS state fixed for as long as that UE's
    own cluster is holding (mobility.ref_xy unchanged for its group), and
    redraws the WHOLE cluster fresh the instant its reference point jumps
    -- see this module's own docstring for the 3GPP motivation and what
    this deliberately does NOT implement.
    """

    def __init__(self):
        self._los_state = None    # [batch, num_bs, num_ut] bool, our own persisted copy
        self._last_ref_xy = None  # [num_clusters, 2] float -- ref_xy as of each cluster's own last redraw

    def apply(self, channel_model, member_group_idx: np.ndarray, ref_xy: np.ndarray) -> None:
        """Call immediately after channel_model.set_topology(..., los=None,
        ...) has run this interval -- channel_model._scenario._los at that
        point already holds a FRESH, independent draw for every link, and
        channel_model._lsp (shadow fading etc., read directly by
        helpers/large_scale_channel.py) has already been sampled FROM that
        draw's own LOS-dependent mean/std/correlation. This selectively
        reverts the LOS draw for every UE whose own cluster is still
        holding (ref_xy bit-for-bit unchanged since last call -- mobility's
        own step() never touches a holding cluster's ref_xy at all, so
        plain equality is exact, no epsilon needed), then replays EVERY
        downstream step that depends on LOS state -- not just pathloss,
        but also the LSP (shadow fading/K-factor/delay-and-angle-spread)
        sampler's own topology-dependent correlation matrices and its
        cached draw -- so a held-LOS UE's shadow fading stays statistically
        consistent with LOS, not leftover NLOS stats from the draw
        channel_model.set_topology() already made before this call. Mirrors
        channel_model.set_topology()'s own `need_for_update` branch exactly
        (scenario pathloss recompute + LSP sampler topology_updated_callback
        + resample), just re-run AFTER our override instead of after
        Sionna's own fresh draw.

        :param channel_model: SimulationEngine.channel_model (reaches into
            its private _scenario/_lsp_sampler/_lsp -- same pattern already
            used by helpers/large_scale_channel.py to read this same
            object's other private LSP/CIR state directly).
        :param member_group_idx: [num_ut] int -- which cluster each UE
            belongs to (mobility.member_group_idx).
        :param ref_xy: [num_clusters, 2] float -- THIS interval's own
            cluster reference points (mobility.ref_xy), BEFORE
            mobility.step() advances them for the next interval.
        """
        scenario = channel_model._scenario
        fresh_los = scenario._los  # [batch, num_bs, num_ut] bool, just (re)drawn by set_topology
        if self._last_ref_xy is None:
            # First call ever -- nothing to hold yet, accept this draw (and
            # its already-sampled downstream LSPs) as-is, just start tracking.
            self._los_state = fresh_los.clone()
            self._last_ref_xy = ref_xy.copy()
            return

        cluster_moved = ~np.all(ref_xy == self._last_ref_xy, axis=-1)  # [num_clusters] -- jumped this interval
        hold = ~cluster_moved[member_group_idx]  # [num_ut] -- True: keep the OLD state
        hold_t = torch.as_tensor(hold, device=fresh_los.device).view(1, 1, -1)
        merged = torch.where(hold_t, self._los_state, fresh_los)

        if torch.equal(merged, fresh_los):
            # Every cluster jumped (or nothing to hold) -- Sionna's own draw
            # and its already-sampled LSPs are already fully consistent, no
            # replay needed.
            self._los_state = merged.clone()
            self._last_ref_xy = ref_xy.copy()
            return

        scenario._update_attr("_los", merged)
        scenario._compute_lsp_log_mean_std()
        scenario._compute_pathloss_basic()
        # Same two calls channel_model.set_topology() itself makes right
        # after scenario.set_topology() (see that method's own
        # `need_for_update` branch) -- replayed here so shadow fading/
        # K-factor/etc. get redrawn from the CORRECTED (post-override) LOS
        # state's own mean/std/cross-correlation, not the stale pre-override
        # ones they were already sampled from once this call started.
        channel_model._lsp_sampler.topology_updated_callback()
        if not channel_model._always_generate_lsp:
            channel_model._lsp = channel_model._lsp_sampler()

        self._los_state = merged.clone()
        self._last_ref_xy = ref_xy.copy()
