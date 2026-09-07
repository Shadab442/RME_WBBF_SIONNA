"""Per-sector staggered decision scheduling for the async DRL tilt-control
loop -- decouples each sector's own decision cadence from the shared
interval boundary. See memory project_drl_state_taxonomy_v2 for the async
redesign rationale.
"""

import numpy as np

from helpers.utils import get_logger, greedy_graph_coloring

logger = get_logger(__name__)


class SectorTiltControlScheduler:
    """Each sector gets a fixed phase offset into its own
    measurement_slots_per_interval-length window, closing/deciding
    independently on its own schedule.

    The actual requirement for async scheduling is narrower than "every
    sector gets a unique offset": two ADJACENT sectors must never decide in
    the same window (otherwise a local reward/state change can't be
    attributed to either one specifically -- both changed at once). Two
    sectors that don't interfere can safely share a phase. Graph-coloring
    sector_adjacency finds the minimum number of phases that satisfies
    this exactly, rather than a manually-tuned group size that can
    accidentally group same-site (mutually adjacent) sectors together.

    :ivar sector_offset: [num_bs] int, each sector's phase offset [slots].
    """

    def __init__(self, num_bs: int, measurement_slots_per_interval: int,
                async_schedule: str, sector_adjacency: np.ndarray = None):
        """
        :param async_schedule: "sync" | "async" -- "async" colors
            sector_adjacency so adjacent sectors never share a phase.
        :param sector_adjacency: [num_bs, num_bs] bool, symmetric --
            required (and only used) for "async".
        """
        assert async_schedule in ("sync", "async"), "async_schedule must be 'sync' or 'async'"
        self.measurement_slots_per_interval = measurement_slots_per_interval

        if async_schedule == "sync":
            self.sector_offset = np.zeros(num_bs, dtype=int)
        else:
            assert sector_adjacency is not None, "async_schedule='async' requires sector_adjacency"
            self.sector_offset = greedy_graph_coloring(sector_adjacency)
            num_phases = len(set(self.sector_offset.tolist()))
            # A period shorter than the color count reuses offsets among
            # colors mod measurement_slots_per_interval, which can silently
            # collide two adjacent (differently-colored) sectors back onto
            # the same closing slot -- defeating the whole point of coloring.
            assert measurement_slots_per_interval >= num_phases, (
                f"measurement_slots_per_interval ({measurement_slots_per_interval}) is shorter than the "
                f"number of adjacency colors ({num_phases}) -- some adjacent sectors would collide onto "
                "the same closing slot; increase the interval or reduce sector density.")

        logger.info("SectorTiltControlScheduler constructed: num_bs=%d async_schedule=%s "
                   "num_phases=%d offset_range=[%d, %d]",
                   num_bs, async_schedule, len(set(self.sector_offset.tolist())),
                   int(self.sector_offset.min()), int(self.sector_offset.max()))

    def closes(self, global_slot) -> np.ndarray:
        """[num_bs] bool -- which sectors' own staggered window (length
        measurement_slots_per_interval) closes exactly at this global
        measurement-slot index (cumulative across the whole run, not reset
        per interval).
        """
        logger.function("SectorTiltControlScheduler.closes start: global_slot=%d", global_slot)
        slots = self.measurement_slots_per_interval
        closes = (global_slot >= self.sector_offset) & \
                ((global_slot - self.sector_offset) % slots == slots - 1)
        logger.debug("SectorTiltControlScheduler.closes: %d/%d sectors closing", int(closes.sum()), closes.size)
        logger.function("SectorTiltControlScheduler.closes end")
        return closes
