"""Per-sector staggered decision scheduling for the async DRL tilt-control
loop -- decouples each sector's own decision cadence from the shared
interval boundary. See memory project_drl_state_taxonomy_v2 for the async
redesign rationale.
"""

import numpy as np


class SectorTiltControlScheduler:
    """Each sector gets a fixed phase offset into its own
    measurement_slots_per_interval-length window, closing/deciding
    independently on its own schedule.

    :ivar sector_offset: [num_bs] int, each sector's phase offset [slots].
    """

    def __init__(self, num_bs: int, measurement_slots_per_interval: int,
                async_schedule: str, async_group_size: int, seed: int):
        """
        :param async_schedule: "round_robin" | "random" | "sync".
        :param async_group_size: sectors given a new decision per interval.
        :param seed: seeds the "random" schedule's fixed permutation.
        """
        assert async_schedule in ("round_robin", "random", "sync"), \
            "async_schedule must be 'round_robin', 'random', or 'sync'"
        self.measurement_slots_per_interval = measurement_slots_per_interval

        # Per-sector phase offset
        if async_schedule == "sync":
            self.sector_offset = np.zeros(num_bs, dtype=int)
        else:
            if async_schedule == "round_robin":
                order = np.arange(num_bs)
            else:  # random -- a fixed permutation, drawn once
                order = np.random.default_rng(seed).permutation(num_bs)
            self.sector_offset = np.empty(num_bs, dtype=int)
            self.sector_offset[order] = np.arange(num_bs) // async_group_size

    def closes(self, global_slot) -> np.ndarray:
        """[num_bs] bool -- which sectors' own staggered window (length
        measurement_slots_per_interval) closes exactly at this global
        measurement-slot index (cumulative across the whole run, not reset
        per interval).
        """
        slots = self.measurement_slots_per_interval
        return (global_slot >= self.sector_offset) & \
              ((global_slot - self.sector_offset) % slots == slots - 1)
