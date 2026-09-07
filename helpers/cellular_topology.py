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
    ):
        """
        :param scenario_params: "isd", "bs_height", "min_bs_ut_dist"
        :param num_rings: HexGrid rings (1 ring = 7 sites) -- Sionna's
            HexGrid requires num_rings >= 1 (7 sites minimum), so a smaller
            deployment isn't reachable through num_rings alone.
        :param num_sectors_per_site: sectors co-located at each site.
        :param batch_size: batch dimension for bs_loc/bs_orientations.
        :param num_sites: if given, builds the normal num_rings-sized grid
            then keeps only the first num_sites sites (e.g. num_sites=1 for
            a standalone single-site/3-sector diagnostic) -- sector_adjacency
            and corner_neighbor_ids are recomputed on the truncated set using
            the SAME general-purpose geometry code, so a truncated topology
            correctly reports zero cross-site neighbors (there's nothing left
            to be adjacent to) rather than needing separate logic.
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

        # Standard 3-sector boresight yaws (60, 180, 300 deg)
        sector_yaws = torch.tensor([PI / 3.0, PI, 5.0 * PI / 3.0], dtype=dtype, device=bs_loc.device)
        bs_yaw = sector_yaws.repeat(num_cells)  # [num_bs]
        bs_yaw = bs_yaw.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [batch, num_bs, 1]

        # Mechanical downtilt toward a nominal sector-center distance.
        # Vertical drop is to typical UE height, not all the way to ground.
        ut_height = scenario_params.get("min_ut_height", torch.zeros_like(bs_height))
        sector_center = (self.min_bs_ut_dist + 0.5 * isd) * 0.5
        bs_downtilt = torch.atan2(bs_height - ut_height, sector_center)
        num_bs = num_cells * num_sectors_per_site
        bs_pitch = torch.full((batch_size, num_bs, 1), bs_downtilt.item(), dtype=dtype, device=bs_loc.device)
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

        # Corner-only neighbors (share a hex vertex, not an edge) -- used by
        # compute_observation_windows and for expanded neighbor-tilt visibility.
        self.corner_neighbor_ids = self._compute_corner_neighbor_ids()
        self.max_corner_neighbors = self.corner_neighbor_ids.shape[1]
        self.expanded_neighbor_mask = self.sector_adjacency.copy()
        for i in range(num_bs):
            for j in self.corner_neighbor_ids[i][self.corner_neighbor_ids[i] >= 0]:
                self.expanded_neighbor_mask[i, j] = True

        logger.info("CellularTopology constructed: num_cells=%d num_bs=%d isd=%.1f max_neighbors=%d "
                   "max_corner_neighbors=%d", self.num_cells, self.num_bs, self.isd, self.max_neighbors,
                   self.max_corner_neighbors)

    @staticmethod
    def _compute_sector_adjacency(bs_xy: np.ndarray, boresight_rad: np.ndarray, isd: float) -> np.ndarray:
        """[num_sectors, num_sectors] bool, symmetric, diagonal False --
        sector j is a neighbor of sector i if they're co-located (same
        site), or if sector i's own rhombus territory shares an outer edge
        with sector j's. A regular hexagon splits into 3 congruent rhombi
        when cut from its center to alternating vertices, one per sector
        -- each rhombus has 4 edges: 2 RADIAL edges shared with its 2
        same-site siblings (handled by the same-site case), and 2 OUTER
        (hex-perimeter) edges, each shared with exactly ONE specific
        sector at a neighboring site -- the outward direction through
        each outer edge's midpoint is boresight +-30 deg, at distance ISD.
        Which of that neighboring site's 3 sectors actually owns the
        shared edge is found by the SAME test run in reverse (does that
        candidate sector's own outer-edge direction point back at this
        site).

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

        adjacency = same_site.copy()
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
            d = ut_xy[idx_unresolved] - bs_xy[s]
            cross = e1[s, 0] * e2[s, 1] - e1[s, 1] * e2[s, 0]
            alpha = (d[:, 0] * e2[s, 1] - d[:, 1] * e2[s, 0]) / cross
            beta = (e1[s, 0] * d[:, 1] - e1[s, 1] * d[:, 0]) / cross

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

    def _compute_corner_neighbor_ids(self) -> np.ndarray:
        """[num_bs, max_corner_neighbors] int, -1-padded: sectors sharing a
        rhombus CORNER (a hex vertex) with sector i but NOT already an
        edge-adjacent neighbor (those are in neighbor_ids). Only 3 of a
        rhombus's 4 corners are hex vertices -- E1 (boresight-60deg), Far
        (boresight), E2 (boresight+60deg); the 4th corner is the site
        center, shared only with the 2 same-site siblings already covered
        via edges. Each hex vertex is shared by up to 3 hexagons in a full
        tiling; num_rings=1 leaves some vertices on the network boundary
        with fewer, so counts are NOT uniform across sectors.
        """
        bs_xy = self.bs_loc[0, :, :2].detach().cpu().numpy()
        boresight_rad = self.bs_orientations[0, :, 0].detach().cpu().numpy()
        side_length = float(self.grid.cell_radius.item())
        num_bs = self.num_bs

        e1 = side_length * np.stack([np.cos(boresight_rad - np.pi / 3), np.sin(boresight_rad - np.pi / 3)], axis=-1)
        e2 = side_length * np.stack([np.cos(boresight_rad + np.pi / 3), np.sin(boresight_rad + np.pi / 3)], axis=-1)
        corner_pts = {"e1": bs_xy + e1, "far": bs_xy + e1 + e2, "e2": bs_xy + e2}

        tol = self.isd * 0.02
        corner_only = [set() for _ in range(num_bs)]
        for pts in corner_pts.values():
            for other_pts in corner_pts.values():
                for i in range(num_bs):
                    d = np.linalg.norm(other_pts - pts[i], axis=-1)
                    for j in np.where(d < tol)[0]:
                        if j != i:
                            corner_only[i].add(int(j))
        for i in range(num_bs):
            corner_only[i] -= set(self.neighbor_ids[i][self.neighbor_ids[i] >= 0].tolist())

        max_corner_neighbors = max((len(s) for s in corner_only), default=0)
        corner_neighbor_ids = np.full((num_bs, max_corner_neighbors), -1, dtype=int)
        for i, s in enumerate(corner_only):
            ids = sorted(s)
            corner_neighbor_ids[i, :len(ids)] = ids

        counts = [len(s) for s in corner_only]
        logger.info("_compute_corner_neighbor_ids: counts min=%d max=%d mean=%.1f",
                   min(counts, default=0), max(counts, default=0), float(np.mean(counts)) if counts else 0.0)
        return corner_neighbor_ids

    def compute_observation_windows(self, cell_size_m: float) -> list:
        """Nine fixed-purpose spatial observation windows per sector, over
        the sector's own rhombus grid (built via build_sector_rhombus_grid)
        plus context borrowed from neighboring sectors' own grids -- NOT an
        information-preserving partition of the 36 own cells. Windows are
        deliberately allowed to overlap (an own boundary cell can be a
        member of both its interior partition context and a seam window)
        and to vary slightly in member count at the network boundary
        (fewer real sectors meet there) -- see project memory
        project_drl_state_taxonomy_v2 and the halo/pooling design
        discussion for rationale. Requires build_sector_rhombus_grid to
        have been called first with the SAME cell_size_m.

        Nine windows per sector, in a fixed order:
          "center"      -- own middle 2x2, pure interior context.
          "edge_beta0", "edge_alpha0", "edge_alphaN", "edge_betaN"
                        -- seam context: 2 own boundary cells (nearest the
                           seam's midpoint) + 2 matching cells from the
                           neighbor across that edge (fewer if no neighbor
                           exists there, e.g. a network-boundary sector).
          "corner_C", "corner_E1", "corner_Far", "corner_E2"
                        -- vertex context: this sector's own nearest cell
                           to that vertex + the nearest cell of every OTHER
                           sector (edge- or corner-only-adjacent) that also
                           touches that same vertex.

        :param cell_size_m: must match the value already passed to
            build_sector_rhombus_grid (reuses its cached grid_points/n).
        :output: [num_bs] list of dict[window_name -> list[(sector_idx, cell_idx)]].
        """
        logger.function("compute_observation_windows start: cell_size_m=%.2f", cell_size_m)
        grid_points, n = self.build_sector_rhombus_grid(cell_size_m)  # [num_bs, n*n, 2]
        num_bs = self.num_bs
        mid = n // 2
        cell_size = float(self.grid.cell_radius.item()) / n
        edge_match_threshold = cell_size * 1.5
        # A cell's centroid sits well off its own corner point (~1.5-1.7 cell
        # widths for this rhombus geometry), unlike an edge midpoint where the
        # boundary cell centroid is much closer -- needs a looser threshold.
        corner_match_threshold = cell_size * 2.5

        def flat(a, b):
            return a * n + b

        edge_names = ["beta0", "alpha0", "alphaN", "betaN"]
        edge_cells = {
            "beta0": [(mid - 1, 0), (mid, 0)],
            "alpha0": [(0, mid - 1), (0, mid)],
            "alphaN": [(n - 1, mid - 1), (n - 1, mid)],
            "betaN": [(mid - 1, n - 1), (mid, n - 1)],
        }
        corner_cell = {"C": (0, 0), "E1": (n - 1, 0), "Far": (n - 1, n - 1), "E2": (0, n - 1)}
        # Which 2 edges meet at each corner (for finding that corner's side-neighbors)
        corner_edges = {"C": ("beta0", "alpha0"), "E1": ("beta0", "alphaN"),
                        "Far": ("alphaN", "betaN"), "E2": ("alpha0", "betaN")}

        corner_neighbor_ids = self.corner_neighbor_ids

        def nearest_cell(sector_j, world_pt):
            d = np.linalg.norm(grid_points[sector_j] - world_pt[None, :], axis=-1)
            return int(d.argmin())

        def match_edges_to_neighbors(i):
            """Greedy 1-to-1 assignment of sector i's up-to-4 edges to its
            listed neighbor_ids, by minimum boundary-cell distance --
            verified against neighbor_ids counts across all 21 sectors."""
            cands = self.neighbor_ids[i][self.neighbor_ids[i] >= 0]
            edge_pts = {e: np.stack([grid_points[i, flat(a, b)] for a, b in cells])
                       for e, cells in edge_cells.items()}
            pairs = []
            for ename in edge_names:
                for ji, j in enumerate(cands):
                    d = np.linalg.norm(grid_points[j][:, None, :] - edge_pts[ename][None, :, :], axis=-1).min()
                    pairs.append((d, ename, ji))
            pairs.sort(key=lambda t: t[0])
            assignment, used_j = {e: None for e in edge_names}, set()
            for d, ename, ji in pairs:
                if d > edge_match_threshold:
                    break
                if assignment[ename] is not None or ji in used_j:
                    continue
                assignment[ename] = int(cands[ji])
                used_j.add(ji)
            return assignment

        windows = [dict() for _ in range(num_bs)]
        for i in range(num_bs):
            edge_neighbor = match_edges_to_neighbors(i)

            # Center window: own middle 2x2
            windows[i]["center"] = [(i, flat(a, b)) for a in (mid - 1, mid) for b in (mid - 1, mid)]

            # Edge windows: my 2 boundary cells + neighbor's 2 nearest matching cells (if any)
            for ename in edge_names:
                members = [(i, flat(a, b)) for a, b in edge_cells[ename]]
                j = edge_neighbor[ename]
                if j is not None:
                    for a, b in edge_cells[ename]:
                        members.append((j, nearest_cell(j, grid_points[i, flat(a, b)])))
                windows[i][f"edge_{ename}"] = members

            # Corner windows: my 1 nearest cell + every other sector touching that vertex's 1 nearest cell
            for cname, (a, b) in corner_cell.items():
                vertex_pt = grid_points[i, flat(a, b)]
                members = [(i, flat(a, b))]
                for ename in corner_edges[cname]:
                    j = edge_neighbor[ename]
                    if j is not None:
                        members.append((j, nearest_cell(j, vertex_pt)))
                for j in corner_neighbor_ids[i][corner_neighbor_ids[i] >= 0]:
                    # Only sectors whose OWN nearest cell is actually close to this vertex
                    # (a corner-only neighbor may sit at a DIFFERENT one of sector i's 3 hex vertices)
                    j_cell = nearest_cell(j, vertex_pt)
                    if np.linalg.norm(grid_points[j, j_cell] - vertex_pt) < corner_match_threshold:
                        members.append((int(j), j_cell))
                windows[i][f"corner_{cname}"] = members

        sizes = [len(w) for win in windows for w in win.values()]
        logger.info("compute_observation_windows: %d windows/sector, member count min=%d max=%d mean=%.1f",
                   len(windows[0]), min(sizes), max(sizes), float(np.mean(sizes)))
        logger.function("compute_observation_windows end")
        return windows
