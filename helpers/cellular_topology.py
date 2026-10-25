# Hex-grid BS site/sector layout (a thing wrapper around sionna.sys.topology.HexGrid),
import math

import numpy as np
import torch

from sionna.phy import PI
from sionna.sys.topology import HexGrid
from helpers.utils import get_logger

logger = get_logger(__name__)

NUM_SECTORS_PER_SITE = 3  # same 120-degree, 3-sector-per-site convention used throughout


class CellularTopology:
    """BS site/sector placement

    :ivar grid: the underlying :class:`~sionna.sys.topology.HexGrid`.
    :ivar bs_loc: [batch_size, num_bs, 3]. BS/sector positions [m].
    :ivar bs_orientations: [batch_size, num_bs, 3]. BS orientations
        [radian]; index 0 is yaw (sector boresight).
    :ivar site_loc: [num_cells, 2]. BS site (x, y) positions [m].
    :ivar num_cells: number of BS sites.
    :ivar num_bs: total sectors (``num_cells * num_sectors_per_site``).
    :ivar isd: inter-site distance [m].
    :ivar sector_adjacency: [num_bs, num_bs] bool, symmetric -- see
        _compute_sector_adjacency.
    :ivar neighbor_ids: [num_bs, max_neighbors] int, -1-padded fixed
        per-sector neighbor ordering, for per-neighbor (not pooled)
        breakdowns.
    :ivar max_neighbors: widest neighbor count across all sectors.
    """

    def __init__(
        self,
        scenario_params,
        num_rings: int,
        num_sectors_per_site: int = NUM_SECTORS_PER_SITE,
        batch_size: int = 1,
        precision=None,
        device=None,
        num_sites: int = None,
        azimuth_offset_deg: float = 0.0,
    ):
        """
        :param scenario_params: "isd", "bs_height", "min_bs_ut_dist"
        :param num_rings: HexGrid rings (1 ring = 7 sites) -- Sionna's
            HexGrid requires num_rings >= 1 (7 sites minimum), so a smaller
            deployment isn't reachable through num_rings alone.
        :param num_sectors_per_site: sectors co-located at each site.
        :param batch_size: batch dimension for bs_loc/bs_orientations.
        :param azimuth_offset_deg: added to EVERY sector's own standard
            boresight yaw (60/180/300 deg) uniformly, e.g. for testing how
            shifting the horizontal beam pointing direction away from each
            sector's own wedge center reshapes coverage. 0.0 (default) is
            the standard tri-sector convention -- every sector still ends
            up 120 deg apart from the others (the offset is added to all
            three identically, so sector spacing/symmetry is preserved,
            just rotated). Purely a diagnostic/experimentation knob for
            now -- sector_adjacency, hotspot_region_membership, and every
            other geometry-derived quantity below are computed from the
            OFFSET yaw, so they stay self-consistent, but nothing yet
            depends on this being 0.
        :param num_sites: if given, builds the normal num_rings-sized grid
            then keeps only the first num_sites sites (e.g. num_sites=1 for
            a standalone single-site/3-sector diagnostic) -- sector_adjacency
            is recomputed on the truncated set using the SAME general-purpose
            geometry code, so a truncated topology correctly reports zero
            cross-site neighbors (there's nothing left to be adjacent to)
            rather than needing separate logic.
        """
        # Assertion check for 3 sector site
        assert num_sectors_per_site == 3, (
            "sector_yaws below hardcodes the standard 3-sector, 120-degree "
            "boresight convention (60/180/300 deg); a different "
            "num_sectors_per_site would need a different yaw formula."
        )

        # Read input parameters
        self.num_sectors_per_site = num_sectors_per_site
        self.batch_size = batch_size
        self.min_bs_ut_dist = scenario_params["min_bs_ut_dist"]
        isd = scenario_params["isd"]
        bs_height = scenario_params["bs_height"]

        # Create hexagonal cellular grid
        self.grid = HexGrid(
            isd=isd.item(),
            cell_height=bs_height.item(),
            num_rings=num_rings,
            precision=precision,
            device=device,
        )
        num_cells = self.grid.num_cells

        # num_sectors_per_site co-located sectors per site
        bs_loc = self.grid.cell_loc.repeat_interleave(num_sectors_per_site, dim=0)
        bs_loc = bs_loc.unsqueeze(0).expand(batch_size, -1, -1).clone()
        dtype = bs_loc.dtype

        # Standard 3-sector boresight yaws (60, 180, 300 deg), uniformly
        # rotated by azimuth_offset_deg if given (see this class's own
        # docstring -- sector spacing/symmetry is preserved, just rotated).
        sector_yaws = torch.tensor([PI / 3.0, PI, 5.0 * PI / 3.0], dtype=dtype, device=bs_loc.device)
        sector_yaws = sector_yaws + math.radians(azimuth_offset_deg)
        bs_yaw = sector_yaws.repeat(num_cells)  # [num_bs]
        bs_yaw = bs_yaw.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [batch, num_bs, 1]

        # Mechanical pitch: horizontal (0 deg, boresight parallel to ground)
        # -- NOT aimed at a computed "sector center" distance (2026-10-08,
        # replacing the old atan2-toward-sector_center formula, which gave
        # e.g. UMa a 9.36 deg baked-in mechanical downtilt before any
        # electrical tilt was even applied; real BS antennas aren't
        # mechanically aimed at a calculated sector-center point like that).
        # Every degree of downtilt now comes from ElectricalDowntilt alone.
        num_bs = num_cells * num_sectors_per_site
        bs_pitch = torch.zeros(batch_size, num_bs, 1, dtype=dtype, device=bs_loc.device)
        bs_roll = torch.zeros(batch_size, num_bs, 1, dtype=dtype, device=bs_loc.device)

        self.bs_loc = bs_loc
        self.bs_orientations = torch.cat([bs_yaw, bs_pitch, bs_roll], dim=-1)  # [batch, num_bs, 3]
        self.site_loc = bs_loc[0, ::num_sectors_per_site, :2]  # [num_cells, 2]
        self.num_cells = num_cells
        self.num_bs = num_bs
        self.isd = float(isd.item())

        # Truncate to a smaller standalone deployment, if requested.
        if num_sites is not None:
            assert 1 <= num_sites <= num_cells, f"num_sites must be in [1, {num_cells}], got {num_sites}"
            num_bs = num_sites * num_sectors_per_site
            self.bs_loc = self.bs_loc[:, :num_bs, :]
            self.bs_orientations = self.bs_orientations[:, :num_bs, :]
            self.site_loc = self.site_loc[:num_sites]
            self.num_cells = num_sites
            self.num_bs = num_bs
            # mirror_cell_loc is a read-only HexGrid property (computed from
            # the full num_rings grid) -- can't truncate it in place. With
            # only num_sites real sites left, wraparound mirroring has
            # nothing to do (there's no other tile copy to prefer), so
            # override with a no-op: every UE's virtual BS location is just
            # the real one, broadcast to its shape.
            def mirror_bs_loc_truncated(ut_loc):
                # Matches mirror_bs_loc's own [batch, num_bs, num_ut, 3] output shape.
                batch_size, num_ut, _ = ut_loc.shape
                return self.bs_loc[:, :, None, :].expand(batch_size, -1, num_ut, -1)
            self.mirror_bs_loc = mirror_bs_loc_truncated
            logger.info("CellularTopology: truncated to num_sites=%d (num_bs=%d); "
                       "mirror_bs_loc overridden to a no-op (nothing to wrap around to)", num_sites, num_bs)

        # Sector adjacency -- needed by KpiManager for overshoot/per-neighbor
        # KPIs, computed here since it's pure site/sector geometry.
        bs_xy = self.bs_loc[0, :, :2].detach().cpu().numpy()
        boresight_rad = self.bs_orientations[0, :, 0].detach().cpu().numpy()
        self.sector_adjacency = self._compute_sector_adjacency(bs_xy, boresight_rad, self.isd)
        neighbor_lists = [np.where(row)[0] for row in self.sector_adjacency]
        self.max_neighbors = max((len(nl) for nl in neighbor_lists), default=0)
        self.neighbor_ids = np.full((num_bs, self.max_neighbors), -1, dtype=int)
        for i, nl in enumerate(neighbor_lists):
            self.neighbor_ids[i, :len(nl)] = nl

        logger.info("CellularTopology constructed: num_cells=%d num_bs=%d isd=%.1f max_neighbors=%d",
                   self.num_cells, self.num_bs, self.isd, self.max_neighbors)

    @staticmethod
    def _compute_sector_adjacency(bs_xy: np.ndarray, boresight_rad: np.ndarray, isd: float) -> np.ndarray:
        """[num_sectors, num_sectors] bool, symmetric, diagonal False --
        sector j is a neighbor of sector i if sector i's own rhombus
        territory shares an OUTER edge with sector j's (a cross-site
        neighbor). A regular hexagon splits into 3 congruent rhombi when
        cut from its center to alternating vertices, one per sector --
        each rhombus has 4 edges: 2 RADIAL edges shared with its 2
        same-site siblings, and 2 OUTER (hex-perimeter) edges, each shared
        with exactly ONE specific sector at a neighboring site -- the
        outward direction through each outer edge's midpoint is boresight
        +-30 deg, at distance ISD. Which of that neighboring site's 3
        sectors actually owns the shared edge is found by the SAME test
        run in reverse (does that candidate sector's own outer-edge
        direction point back at this site).

        Same-site siblings are deliberately EXCLUDED (2026-09-20, were
        previously always adjacent) -- measured: a sector's own tilt
        change produces ~0.03dB mean |SINR delta| on its same-site
        siblings vs ~0.32dB on its genuine cross-site front neighbors (the
        ones this method finds), a ~10x gap. Same-site co-location means
        near-zero path-loss differentiation and no antenna-reach effect
        between siblings, so they were never really "coupled" in the
        sense this adjacency is meant to capture (overshoot, per-neighbor
        state, reward) -- they were only ever adjacent by a geometric
        technicality (sharing a radial division line, not a real
        propagation boundary).

        This definition supersedes an earlier, coarser site-distance-only
        version (any sector at a nearest-neighbor SITE counted as a
        neighbor, regardless of orientation) -- that overcounted badly
        (e.g. 20 neighbors for a sector whose own beam only ever faces 2
        of its site's up-to-6 nearest sites) and is used consistently
        everywhere adjacency matters (overshoot, per-neighbor state),
        not just for sizing a state vector.
        """
        logger.function("_compute_sector_adjacency start: num_sectors=%d", bs_xy.shape[0])

        def wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        num_sectors = bs_xy.shape[0]
        site_dist = np.linalg.norm(bs_xy[:, None, :] - bs_xy[None, :, :], axis=-1)
        same_site = site_dist < 1e-6

        adjacency = np.zeros((num_sectors, num_sectors), dtype=bool)
        for i in range(num_sectors):
            for edge_dir in (boresight_rad[i] - np.pi / 6, boresight_rad[i] + np.pi / 6):
                target = bs_xy[i] + isd * np.array([np.cos(edge_dir), np.sin(edge_dir)])
                candidates = np.where((np.linalg.norm(bs_xy - target[None, :], axis=1) < isd * 0.05)
                                      & ~same_site[i])[0]
                if candidates.size == 0:
                    continue
                reverse_dir = wrap(edge_dir + np.pi)
                for j in candidates:
                    for j_edge_dir in (boresight_rad[j] - np.pi / 6, boresight_rad[j] + np.pi / 6):
                        if abs(wrap(j_edge_dir - reverse_dir)) < np.radians(5):
                            adjacency[i, j] = True
        np.fill_diagonal(adjacency, False)

        neighbor_counts = adjacency.sum(axis=1)
        logger.debug("_compute_sector_adjacency: neighbor count min=%d max=%d mean=%.1f",
                    int(neighbor_counts.min()), int(neighbor_counts.max()), float(neighbor_counts.mean()))
        if (neighbor_counts == 0).any():
            logger.warning("_compute_sector_adjacency: %d sector(s) with ZERO neighbors",
                          int((neighbor_counts == 0).sum()))
        logger.function("_compute_sector_adjacency end")
        return adjacency

    @property
    def default_drop_radius(self) -> float:
        """Smallest disk radius that fully covers the hex-grid footprint
        (farthest site center + one hex radius)."""
        max_site_dist = torch.linalg.norm(self.site_loc, dim=-1).max()
        return float(max_site_dist.item() + self.grid.cell_radius.item())

    def is_within_coverage(self, xy: torch.Tensor) -> torch.Tensor:
        """True for points that lie within their nearest site's actual
        hexagon -- not just within the `default_drop_radius` disk,

        :param xy: [N, 2] candidate (x, y) positions [m].
        :output: [N] bool tensor.
        """
        logger.function("is_within_coverage start: xy shape=%s", tuple(xy.shape))
        site_loc = self.site_loc.to(dtype=xy.dtype, device=xy.device)
        cell_radius = self.grid.cell_radius.to(dtype=xy.dtype, device=xy.device)

        dist_to_sites = torch.linalg.norm(xy[:, None, :] - site_loc[None, :, :], dim=-1)
        nearest_dist, nearest_idx = dist_to_sites.min(dim=-1)

        # Hex boundary radius varies with angle:
        # largest at vertices, smallest at edge midpoints.
        delta = xy - site_loc[nearest_idx]
        angle = torch.atan2(delta[:, 1], delta[:, 0])
        offset = torch.remainder(angle, math.pi / 3.0) - math.pi / 6.0
        hex_boundary = cell_radius * math.cos(math.pi / 6.0) / torch.cos(offset)
        within = nearest_dist <= hex_boundary
        logger.debug("is_within_coverage: %d/%d points inside", int(within.sum()), within.numel())
        logger.function("is_within_coverage end")
        return within

    def mirror_bs_loc(self, ut_loc: torch.Tensor) -> torch.Tensor:
        """Wraparound topology: 
        use the nearest mirror of each BS site for every UE to avoid edge effects.

        :param ut_loc: [batch_size, num_ut, 3].
        :output bs_virtual_loc: [batch_size, num_bs, num_ut, 3]. Feeds
            directly into `channel_model.set_topology`'s
            ``bs_virtual_loc`` argument.
        """
        logger.function("mirror_bs_loc start: ut_loc shape=%s", tuple(ut_loc.shape))
        batch_size, num_ut, _ = ut_loc.shape
        # [num_cells, 7, 3]: each site's own position + its 6 mirror images.
        mirror_cell_loc = self.grid.mirror_cell_loc.to(dtype=ut_loc.dtype, device=ut_loc.device)

        # [batch, num_ut, num_cells, 7]
        dist = torch.norm(
            ut_loc[:, :, None, None, :] - mirror_cell_loc[None, None, :, :, :], dim=-1
        )
        closest_idx = dist.argmin(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        mirror_expand = mirror_cell_loc[None, None].expand(batch_size, num_ut, -1, -1, -1)
        # [batch, num_ut, num_cells, 3]
        virtual_loc = torch.gather(mirror_expand, dim=3, index=closest_idx).squeeze(3)

        # Co-located sectors at a site share the same virtual site position.
        virtual_loc = virtual_loc.repeat_interleave(self.num_sectors_per_site, dim=2)
        result = virtual_loc.permute(0, 2, 1, 3)  # [batch, num_bs, num_ut, 3]
        logger.debug("mirror_bs_loc: result shape=%s", tuple(result.shape))
        logger.function("mirror_bs_loc end")
        return result

    def build_sector_rhombus_grid(self, cell_size: float) -> tuple:
        """Per-sector rhombus spatial sub-grid for the spatial-coverage DRL
        state -- NOT self.grid (the coarser per-site hex layout used for
        rendering). A regular hexagon splits into 3 congruent rhombi when
        cut from center to alternating vertices, one per sector; grid
        cells are laid out in the rhombus's own oblique basis (mirror
        images of each other about boresight, so plain row-major
        (alpha, beta) indexing is automatically boresight-mirror-
        symmetric). See memory project_drl_state_taxonomy_v2 for the full
        design rationale. Caches the basis vectors internally for
        assign_to_sector_rhombus_grid.

        :param cell_size: target grid cell size [m] -- divisions per side
            is ceil(hex circumradius / cell_size).
        :output: (grid_points [num_bs, n*n, 2], n) -- grid_points are
            world-coordinate cell centers, row-major over (alpha, beta);
            n is divisions per side (n*n points per sector).
        """
        logger.function("build_sector_rhombus_grid start: cell_size=%.2f", cell_size)
        bs_xy = self.bs_loc[0, :, :2].detach().cpu().numpy()
        boresight_rad = self.bs_orientations[0, :, 0].detach().cpu().numpy()
        side_length = float(self.grid.cell_radius.item())

        # Grid resolution
        n = int(np.ceil(side_length / cell_size))

        # Two rhombus oblique basis vectors in +-60deg directions
        e1 = side_length * np.stack(
            [np.cos(boresight_rad - np.pi / 3), np.sin(boresight_rad - np.pi / 3)], axis=-1)
        e2 = side_length * np.stack(
            [np.cos(boresight_rad + np.pi / 3), np.sin(boresight_rad + np.pi / 3)], axis=-1)

        # Grid center coordinates
        frac = (np.arange(n) + 0.5) / n
        alpha, beta = np.meshgrid(frac, frac, indexing="ij")
        alpha, beta = alpha.reshape(-1), beta.reshape(-1)  # [n*n], row-major (a, b)
        grid_points = (bs_xy[:, None, :] + alpha[None, :, None] * e1[:, None, :]
                      + beta[None, :, None] * e2[:, None, :])  # [num_bs, n*n, 2]

        self._sector_grid_bs_xy = bs_xy
        self._sector_grid_e1 = e1
        self._sector_grid_e2 = e2
        self._sector_grid_n = n
        logger.info("build_sector_rhombus_grid: n=%d (%d cells/sector), grid_points shape=%s",
                   n, n * n, grid_points.shape)
        logger.function("build_sector_rhombus_grid end")
        return grid_points, n

    def assign_to_sector_rhombus_grid(self, ut_xy: np.ndarray) -> tuple:
        """For each UE position, find which sector's rhombus it
        geographically falls in (if any) and which grid cell within that
        sector -- keyed by GEOGRAPHY, not serving-sector assignment.
        Requires build_sector_rhombus_grid to have been called first.

        :param ut_xy: [num_ut, 2].
        :output: (sector_idx, cell_idx) each [num_ut] int, -1 where a UE
            falls outside every sector's rhombus (e.g. near the deployment
            boundary, where no site's rhombus covers that point).
        """
        logger.function("assign_to_sector_rhombus_grid start: ut_xy shape=%s", ut_xy.shape)
        bs_xy, e1, e2, n = self._sector_grid_bs_xy, self._sector_grid_e1, self._sector_grid_e2, self._sector_grid_n

        # Initialization
        num_ut = ut_xy.shape[0]
        num_sectors = bs_xy.shape[0]
        sector_idx = np.full(num_ut, -1, dtype=int)
        cell_idx = np.full(num_ut, -1, dtype=int)
        unresolved = np.ones(num_ut, dtype=bool)

        # Check in each sector
        for s in range(num_sectors):
            # Check for the unresolved UEs
            if not unresolved.any():
                break
            idx_unresolved = np.where(unresolved)[0]

            # Check if the UE is inside this sector
            alpha, beta = self.sector_alpha_beta(s, ut_xy[idx_unresolved])

            inside = (alpha >= 0) & (alpha < 1) & (beta >= 0) & (beta < 1)
            matched = idx_unresolved[inside]

            # Convert fractional coordinates into grid indices
            a_idx = np.clip((alpha[inside] * n).astype(int), 0, n - 1)
            b_idx = np.clip((beta[inside] * n).astype(int), 0, n - 1)

            # Update matched sector and grid cell index
            sector_idx[matched] = s
            cell_idx[matched] = a_idx * n + b_idx

            # Update unresolved UEs
            unresolved[matched] = False

        logger.debug("assign_to_sector_rhombus_grid: resolved=%d/%d", int((sector_idx >= 0).sum()), num_ut)
        logger.function("assign_to_sector_rhombus_grid end")
        return sector_idx, cell_idx

    def sector_basis(self, sector_idx: int) -> tuple:
        """(bs_xy, e1, e2) for one sector -- thin accessor over the cache
        build_sector_rhombus_grid populates (must be called first)."""
        return (self._sector_grid_bs_xy[sector_idx],
               self._sector_grid_e1[sector_idx], self._sector_grid_e2[sector_idx])

    def sector_alpha_beta(self, sector_idx: int, xy: np.ndarray) -> tuple:
        """Raw (alpha, beta) oblique coords of xy [N,2] in sector_idx's own
        rhombus basis -- NOT clipped to [0,1); the same Cramer's-rule
        projection assign_to_sector_rhombus_grid uses inline, factored out
        so both share one implementation.
        """
        bs_xy, e1, e2 = self.sector_basis(sector_idx)
        d = xy - bs_xy
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        alpha = (d[:, 0] * e2[1] - d[:, 1] * e2[0]) / cross
        beta = (e1[0] * d[:, 1] - e1[1] * d[:, 0]) / cross
        return alpha, beta

    def set_hotspot_region_lut(self, near_mask: np.ndarray, mid_mask: np.ndarray) -> None:
        """Installs the calibrated near/mid (i, j) lookup table that
        hotspot_region_membership reads from -- see that method's own
        docstring. Must be called once, before any hotspot_region_membership
        call (SimulationEngine._calibrate_hotspot_region_lut does this in
        __init__, before mobility/UE placement, which is the first real
        consumer). :param near_mask, mid_mask: [n, n] bool, n =
        self._sector_grid_n (set by build_sector_rhombus_grid, must be
        called first) -- mutually exclusive, both False = far.
        """
        n = self._sector_grid_n
        assert near_mask.shape == (n, n) and mid_mask.shape == (n, n), \
            f"hotspot region LUT shape must be ({n},{n}) to match the current grid resolution"
        assert not (near_mask & mid_mask).any(), "near_mask/mid_mask must be mutually exclusive"
        self._hotspot_region_lut_near = near_mask
        self._hotspot_region_lut_mid = mid_mask

    def hotspot_region_membership(self, sector_idx: int, xy: np.ndarray) -> dict:
        """Near/Mid/Far hotspot region membership for points xy [N,2] --
        from sector_idx's OWN rhombus geometry only, no neighbor-sector
        extension.

        near/mid/far come from a calibrated (i, j) lookup table (see
        set_hotspot_region_lut), built from each cell's RADIAL DISTANCE to
        its own site (2026-09-29, see SimulationEngine's own
        _HOTSPOT_NEAR_DIST_M/_HOTSPOT_FAR_DIST_M -- 900/2300 m). Two
        earlier rules were tried and rejected:
          - a closed-form "closeness to the boresight diagonal" rule
            (2026-09-20): even after excluding the min_bs_ut_dist zone,
            its near/mid bands formed a long diagonal WEDGE stretching all
            the way to the sector's outer edge, nothing like the real
            high-SINR footprint (confirmed against real simulated SINR and
            visually against spatial_heatmaps_by_tilt_binned.png).
          - a real-SINR-VALUE rule (2026-09-22): classified each cell by
            its simulated SINR at a frozen 0-deg/all-sectors/forced-LOS/
            no-shadow reference condition, 5/0 dB cuts. Worked, but
            entangled RADIAL distance with AZIMUTHAL interference geometry
            (a cell's SINR depends on proximity to a shared sector
            boundary too, not just distance from its own site) -- its
            near/mid/far bands didn't line up with the two-population
            tilt-coverage reversal found empirically by sweeping tilt
            against PURE distance bins (scripts/radial_swap_plot.png,
            2026-09-28/29): cells under ~900m genuinely prefer downtilt,
            cells from ~900-2300m genuinely prefer uptilt, roles reversing
            cleanly. Tilt is an elevation-only control with NO azimuthal
            effect, so radial distance -- not SINR value, not diagonal
            proximity -- is the axis that actually matches what tilt can
            influence. A pure distance rule also needs no frozen reference
            condition and no radio simulation pass at all to calibrate,
            unlike the SINR-value rule it replaced.

        Points within min_bs_ut_dist of the site are excluded from
        `inside` (and therefore from near/mid/far) entirely, and never
        enter the calibration ranking either: no real UE can ever be
        placed there (helpers/ue_drop.py enforces this everywhere UEs are
        dropped), so any point that close is either outside the reachable
        network or an antenna near-field null -- confirmed to be one of
        the WORST-SINR cells in its sector, not the best, at both 50m and
        20m. This is the same exclusion helpers/ue_drop.py:290-296
        already had to bolt on by hand at every "near" UE-placement call
        site, and helpers/spatial_grid_estimator.py's own
        estimable_cell_mask applies for the same underlying reason (no
        measurement can ever exist there) -- centralized here instead of
        re-derived per caller.

        :output: dict(alpha, beta, eta, inside, near, mid, far) -- each
            [N]. Every `inside` point gets exactly one of near/mid/far --
            no excluded "hole" near the site beyond the min_bs_ut_dist
            exclusion above.
        """
        bs_xy, _, _ = self.sector_basis(sector_idx)
        alpha, beta = self.sector_alpha_beta(sector_idx, xy)
        dist_to_site = np.linalg.norm(xy - bs_xy[None, :], axis=-1)
        inside = ((alpha >= 0) & (alpha < 1) & (beta >= 0) & (beta < 1)
                 & (dist_to_site >= float(self.min_bs_ut_dist)))
        eta = np.maximum(alpha, beta)
        n = self._sector_grid_n
        assert hasattr(self, "_hotspot_region_lut_near"), \
            "hotspot_region_membership needs set_hotspot_region_lut() called first (see SimulationEngine.__init__)"
        i_idx = np.clip(np.floor(alpha * n).astype(int), 0, n - 1)
        j_idx = np.clip(np.floor(beta * n).astype(int), 0, n - 1)
        near = inside & self._hotspot_region_lut_near[i_idx, j_idx]
        mid = inside & ~near & self._hotspot_region_lut_mid[i_idx, j_idx]
        far = inside & ~near & ~mid
        # near_mid: purely derived (near | mid), added for consumers that
        # only need a coarser 2-way (near_mid vs far) split -- e.g.
        # HotspotDemandMobility's own demand-state clustering -- without
        # changing the near/mid/far split itself for every other consumer
        # (RegionalInfoProvider, OccupancyEstimator, debug_pipeline_audit.py).
        near_mid = near | mid
        return dict(alpha=alpha, beta=beta, eta=eta, inside=inside, near=near, mid=mid, far=far,
                    near_mid=near_mid)

