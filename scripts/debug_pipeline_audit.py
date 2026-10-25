"""Pipeline debug audit: does each stage of the environment (raw UE ->
grid -> RME -> region -> state/reward) produce physically meaningful,
sufficiently variable, decision-relevant information before it reaches the
RL agent? Built in 8 independent phases (see phaseN_* functions below),
run and verified ONE AT A TIME -- no changes to any existing production
code; this script only drives SimulationEngine's own public
attributes/methods (topology, engine.grid_points, engine.num_own_regions,
engine.regional_info_provider, engine.step_environment/drl_process_turn, ...).

Debug scenario (shared by every phase): num_tilt_control_intervals=50 --
intervals 0-19 held at 0 deg (no tilt),
intervals 20-49 driven by RandomPolicy, so later phases can compare a
static baseline against actual tilt perturbation.

Visual convention (shared by every figure in this script):
  green solid  -- site (hex) boundary
  green dashed -- sector division within a site
  red          -- region_aligned region boundary + region label -- the ONE
                  region definition in this codebase (near/mid/far hotspot
                  regions, CellularTopology.hotspot_region_membership),
                  classified over the sector's own n x n rhombus fine grid
                  so boundaries land on real fine-grid-cell edges. The same
                  near/mid/far rule applies identically in every sector's
                  own oblique (alpha, beta) grid -- comparable ACROSS
                  sectors. Regions from different sectors never overlap:
                  the 3 sectors' own rhombi exactly partition their site's
                  hexagon.
  dark blue dots -- rhombus fine-grid cell centers
  sector ID labels are the GLOBAL index (0..num_bs-1), never the
  per-site local 1/2/3 sionna's own HexGrid.show() would draw.

Run: python scripts/debug_pipeline_audit.py --phase 1 --use-cache/--no-use-cache
(--use-cache/--no-use-cache is required -- see load_or_run_scenario's own
docstring for why: a stale cached scenario reused across a config.yaml
change produces results that look valid but are quietly wrong.)

Saves each phase's figures to results/debug/phase<N>_<name>/.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import f_oneway, mannwhitneyu
import sionna
import torch

from drl.interface import compute_region_state, compute_reward, num_features_for
from helpers.utils import add_cluster_ellipses, load_config
from helpers.occupancy_estimator import OccupancyEstimator
from helpers.simulation_engine import SimulationEngine
from helpers.spatial_grid_estimator import SpatialGridEstimator

# =============================================================
# CONFIGURATIONS
# =============================================================

# Directories
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "debug")
SCENARIO_CACHE_PATH = os.path.join(OUT_ROOT, "_scenario_cache.npz")

# Visualization configurations
SITE_COLOR = "black"
SECTOR_COLOR = "darkred"
REGION_COLOR = "darkblue"
GRID_COLOR = "darkgreen"
HOTSPOT_REGION_COLORS = {"near": "#d7191c", "mid": "#f6d55c", "far": "#2c7bb6"}  # red/yellow/blue
NON_ANCHOR_COLOR = "#808080"  # gray -- a non-anchor (non-location-reporting) UE, see location_reporting_mask

# Colorbar convention -- EVERY continuous/quantitative color scale in this
# script (heatmaps, spatial scatter colorbars, filled regions) uses this
# Red-Yellow-Blue scheme, RED = HIGH value (blue = low); keep any NEW
# colorbar consistent with it.
COLORBAR_CMAP = "RdYlBu_r"   # low -> high: blue -> yellow -> red
COLORBAR_CMAP_R = "RdYlBu"   # reversed relative to COLORBAR_CMAP -- red = low, blue = high
DENSITY_CMAP = COLORBAR_CMAP

# Script specific configurations
NUM_NO_TILT, NUM_RANDOM = 20, 30
DEBUG_SEED = 1
# Reference tilt for phase 2's single-sector perturbation map ("all yellow"
# baseline, zero delta everywhere) -- nearest value in engine.downtilt_sweep_deg
# is used if this isn't exactly in the sweep (see compute_single_sector_tilt_perturbation).
# 9 deg (2026-10-08) -- the nominal/reference tilt for all sectors now that
# CellularTopology's own mechanical pitch is horizontal (0 deg) and the
# committed downtilt sweep is 5-13 deg; 9 deg sits at the middle of that
# sweep. Previously 0 deg, back when the sweep was symmetric around 0 and
# mechanical pitch carried a scenario-dependent baseline tilt on its own.
SINGLE_SECTOR_NOMINAL_TILT_DEG = 9.0

# Cluster/UE synthetic setup
NUM_BS = 21                  # num_rings=1 -- 7 sites x 3 sectors/site (matches every "21 sectors" assumption in this script)
NUM_SECTORS_PER_SITE = 3     # CellularTopology's own fixed invariant (asserted == 3 there)
NUM_SITES = NUM_BS // NUM_SECTORS_PER_SITE
SITE_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
assert len(SITE_MARKERS) == NUM_SITES, "need one marker shape per site"


# =============================================================
# ACCESSORIES
# =============================================================

# Build simulation engine from config
def build_debug_engine(num_ut=None, cluster_mobility_mode=None, hotspot=None,
                       location_reporting_fraction=None):
    """The shared 50-interval (20 NO_TILT and 30 RANDOM) debug scenario.

    :param num_ut: override config.yaml -- must satisfy num_bs *
        mobility.num_clusters_per_sector * mobility.num_ues_per_cluster
        (shared by every cluster_mobility_mode now, not just
        "hotspot_demand") -- so pass cluster_mobility_mode explicitly too
        whenever overriding num_ut.
    :param cluster_mobility_mode: override config.yaml's own default
        (currently "hotspot_demand-markov") -- "<family>-<submode>", e.g.
        "rpgm-periodic" for phase2's legacy-waypoint-cycling-based snapshot
        figure.
    :param hotspot: dict merged into cfg["mobility"]["hotspot"] -- e.g.
        phase2's own demand_states/deviation_radius_m/hotspot_hold_steps
        overrides (num_clusters_per_sector/num_ues_per_cluster now live at
        mobility's own top level, shared by every mode -- not here anymore).
    :param location_reporting_fraction: override config.yaml's own value --
        e.g. a phase wanting a specific anchor/non-anchor mix regardless of
        the real scenario's config.yaml setting. None (default) uses
        config.yaml's own value as-is.
    """
    cfg = load_config()
    cfg["simulation"]["num_tilt_control_intervals"] = NUM_NO_TILT + NUM_RANDOM
    if cluster_mobility_mode is not None:
        cfg["mobility"]["common"]["cluster_mobility_mode"] = cluster_mobility_mode
    if num_ut is not None:
        cfg["topology"]["num_ut"] = num_ut
    if hotspot is not None:
        cfg["mobility"]["hotspot"].update(hotspot)
    if location_reporting_fraction is not None:
        cfg["algorithms"]["drl"]["location_reporting_fraction"] = location_reporting_fraction
    return SimulationEngine(cfg)

# Choose between load / run + save scenario
# This function is shared by multiple debugging phases; called only once.
def load_or_run_scenario(engine, use_cache: bool) -> dict:
    """:param use_cache: caller-required, no implicit default (see --use-cache/
        --no-use-cache in main()) -- a stale cache silently reused across a
        config.yaml change (e.g. coverage_threshold_db) produces results that
        look valid but are quietly wrong, which is exactly the confusion this
        forces the caller to rule out every run.
    """
    if use_cache and os.path.exists(SCENARIO_CACHE_PATH):
        print(f"[scenario] loading cached scenario from {SCENARIO_CACHE_PATH} (--use-cache) -- "
             "stale if config.yaml (or anything else run_debug_scenario reads) changed since it was "
             "generated; pass --no-use-cache to force a fresh run.")
        data = np.load(SCENARIO_CACHE_PATH)
        return {k: data[k] for k in data.files}

    # Otherwise run and save it
    print(f"[scenario] running a fresh {NUM_NO_TILT + NUM_RANDOM}-interval scenario "
         f"({'no cache found' if use_cache else '--no-use-cache'})...")
    scenario = run_debug_scenario(engine)
    os.makedirs(OUT_ROOT, exist_ok=True)
    np.savez(SCENARIO_CACHE_PATH, **scenario)
    return scenario

# Actual function for running scenario
def run_debug_scenario(engine, rng=None):
    """Drives mobility + channel across every interval, capturing each
    stage's output for every sector -- read-only: never calls
    drl_controller.update(), so nothing here is learned or fed back.

    Tilt schedule: intervals 0..NUM_NO_TILT-1 held at 0 deg (a static
    baseline); intervals NUM_NO_TILT.. use an independent per-sector
    UNIFORM-RANDOM tilt each interval (like RandomPolicy, but generated
    here directly since we deliberately never touch drl_controller) -- so
    later phases can compare a static baseline against actual tilt
    perturbation. async_schedule is NOT used here: every interval is
    processed as a SINGLE sync-like turn (closes=all sectors at once),
    regardless of config.yaml's own async_schedule -- this audit cares
    about information CONTENT per interval, not the sequential-turn
    training mechanics.

    :output: dict of [num_intervals, ...]-shaped arrays: ut_loc, ref_xy
        (each cluster's own reference/center point, e.g. for
        add_cluster_ellipses), live_tilt_deg, own_occupancy, own_sinr_p25/
        p50/p75, all_occupancy, all_sinr_p50 (every sector's own whole-grid
        coarse pair, see RegionalInfoProvider.compute's own docstring),
        reward, state_ready, observation (the full
        fixed-width state vector), plus observed_mask [num_intervals,
        num_bs, num_grid_cells] (True where a real UE was observed this
        interval, BEFORE Kriging/CGR fills the rest), coverage_per_cell
        (engine.spatial_grid_estimator's own full Kriged grid, called
        directly here via .compute() -- production's own drl_process_turn
        no longer consumes a coverage map at all, only ever calling
        .reset() on this same instance, see RegionalInfoProvider's own
        module docstring; this and cgr_coverage_per_cell exist purely for
        phase 5's own standalone Kriging-vs-CGR validation), and
        cgr_coverage_per_cell (a SECOND,
        CGR-configured SpatialGridEstimator, fed the exact same real
        accumulate() inputs every interval -- so it's directly comparable
        to coverage_per_cell, one unified real scenario instead of two
        separately-realized runs; see phase5's own rme_validation figure).
    """
    # Reading configurations
    if rng is None:
        rng = np.random.default_rng(DEBUG_SEED)
    # Capped at NUM_NO_TILT+NUM_RANDOM regardless of engine.
    # num_tilt_control_intervals (config.yaml's own REAL production
    # training length, e.g. 1200) -- this is a quick diagnostic snapshot,
    # not a training run; blindly inheriting production's length made every
    # needs_scenario=True phase silently pay for a full 1200-interval real
    # channel simulation (fixed 2026-09-20, caught when it made phase 5/6
    # hang far longer than expected).
    num_intervals = min(engine.num_tilt_control_intervals, NUM_NO_TILT + NUM_RANDOM)
    num_bs = engine.num_bs
    tilt_min, tilt_max = engine.tilt_deg_min, engine.tilt_deg_max

    # A SECOND, CGR-configured SpatialGridEstimator, run alongside
    # engine.spatial_grid_estimator (production's own, Kriging by
    # config.yaml's default) and fed the exact same per-interval
    # accumulate() inputs -- the fairness pattern from
    # scripts/compare_cgr_vs_kriging.py's own module docstring, folded
    # into this ONE real scenario so Kriging and CGR stay directly
    # comparable to each other and to ground truth, without a separate run.
    spatial_grid_cell_size_m = load_config()["topology"]["spatial_grid_cell_size_m"]
    cgr_estimator = SpatialGridEstimator(
        num_bs, engine.num_grid_cells, engine.grid_points,
        length_scale=spatial_grid_cell_size_m, coverage_threshold_db=engine.coverage_threshold_db,
        radio_map_method="rsrp_per_sector", noise_power_w=engine.kpi_manager.noise_power_w,
        topology=engine.topology,
        radio_map_estimator="cgr", cgr_num_gaussian=engine.cgr_cfg.get("num_gaussian", 12),
        cgr_ridge_lambda=engine.cgr_cfg.get("ridge_lambda", 1e-4),
        cgr_baseline_method=engine.cgr_cfg.get("baseline_method", "global_mean"),
        cgr_baseline_length_scale_m=engine.cgr_cfg.get("baseline_length_scale_m", spatial_grid_cell_size_m),
        cgr_gaussian_config={k: v for k, v in engine.cgr_cfg.items()
                             if k not in ("num_gaussian", "ridge_lambda", "baseline_method",
                                         "baseline_length_scale_m")})

    # Initializations
    out = {
        "ut_loc": np.zeros((num_intervals, engine.num_ut, 3)),
        "ref_xy": np.zeros((num_intervals, engine.mobility.ref_xy.shape[0], 2)),
        "live_tilt_deg": np.zeros((num_intervals, num_bs)),
        "own_occupancy": np.zeros((num_intervals, num_bs, engine.num_own_regions)),
        "own_sinr_p25": np.zeros((num_intervals, num_bs, engine.num_own_regions)),
        "own_sinr_p50": np.zeros((num_intervals, num_bs, engine.num_own_regions)),
        "own_sinr_p75": np.zeros((num_intervals, num_bs, engine.num_own_regions)),
        "all_occupancy": np.zeros((num_intervals, num_bs)),
        "all_sinr_p50": np.zeros((num_intervals, num_bs)),
        "reward": np.zeros((num_intervals, num_bs)),
        "state_ready": np.zeros((num_intervals, num_bs), dtype=bool),
        "observation": np.zeros((num_intervals, num_bs,
                                 num_features_for("region_aligned", num_own_regions=engine.num_own_regions))),
        "observed_mask": np.zeros((num_intervals, num_bs, engine.num_grid_cells), dtype=bool),
        "coverage_per_cell": np.zeros((num_intervals, num_bs, engine.num_grid_cells)),
        "cgr_coverage_per_cell": np.zeros((num_intervals, num_bs, engine.num_grid_cells)),
    }
    closes = np.ones(num_bs, dtype=bool)

    # For each interval collect run the scenario and collect KPIs
    for interval in range(num_intervals):
        # choose NO TILT or RANDOM depending on interval
        live_tilt_deg = (np.zeros(num_bs) if interval < NUM_NO_TILT
                         else rng.uniform(tilt_min, tilt_max, size=num_bs))

        # Check if last interval
        is_last = interval == num_intervals - 1

        # Get UT locations
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        ref_xy = engine.mobility.ref_xy.detach().cpu().numpy().copy()

        # Get states
        state = engine.step_environment(is_last)

        # Get sector and cell IDs
        sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])

        # Get coverages
        drl_power_w = engine.kpi_manager.resolve_power_at_tilt(state, live_tilt_deg)[0]
        kpis = engine.kpi_manager.compute_ue_kpis(drl_power_w[None, :, :])
        sinr_db = kpis["sinr_db"][0]
        covered_ue = sinr_db > engine.coverage_threshold_db

        # Accumulate coverages from visited locations
        engine.regional_info_provider.accumulate(sector_idx, cell_idx, covered_ue, sinr_db)

        # Updates coverages for unvisited locations with radio map estimation
        power_db_all_sectors = 10.0 * np.log10(drl_power_w)
        coverage_per_cell = None
        if engine.spatial_grid_estimator is not None:
            engine.spatial_grid_estimator.accumulate(sector_idx, cell_idx, covered_ue,
                                                     power_db_all_sectors=power_db_all_sectors)
            out["observed_mask"][interval] = engine.spatial_grid_estimator.visit_count > 0
            _, coverage_per_cell, _ = engine.spatial_grid_estimator.compute(closes, current_tilts_deg=live_tilt_deg)
            out["coverage_per_cell"][interval] = coverage_per_cell

        # Second, CGR-configured estimator -- same real inputs, unconditional
        # (independent of whether engine.spatial_grid_estimator itself was
        # even built -- see location_reporting_fraction's own docs).
        cgr_estimator.accumulate(sector_idx, cell_idx, covered_ue, power_db_all_sectors=power_db_all_sectors)
        if engine.spatial_grid_estimator is None:
            out["observed_mask"][interval] = cgr_estimator.visit_count > 0
        _, cgr_coverage_per_cell, _ = cgr_estimator.compute(closes, current_tilts_deg=live_tilt_deg)
        out["cgr_coverage_per_cell"][interval] = cgr_coverage_per_cell

        # Resolve occupancy, coverage, reward, and readiness for this turn --
        # always real covered_count/visit_count now (own_coverage/
        # all_coverage removed 2026-09-21, replaced by all_sinr_p50 for the
        # neighbor role -- see RegionalInfoProvider's own module
        # docstring); coverage_per_cell above is still computed and saved
        # to out["coverage_per_cell"] purely for phase 5's own separate
        # whole-grid Kriging/CGR validation.
        (own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75,
         all_occupancy, all_sinr_p50, reward_per_sector, state_ready) = (
            engine.regional_info_provider.compute(closes))

        # Min-max normalization for tilts to go into states
        own_tilt_norm = (live_tilt_deg - tilt_min) / (tilt_max - tilt_min)

        # Compute states based on tilt, coverage and occupancy
        observations = compute_region_state(
            own_tilt_norm=own_tilt_norm, own_occupancy=own_occupancy,
            own_sinr_p25=own_sinr_p25, own_sinr_p50=own_sinr_p50, own_sinr_p75=own_sinr_p75,
            all_occupancy=all_occupancy, all_sinr_p50=all_sinr_p50,
            neighbor_ids=engine.topology.neighbor_ids,
        )

        # Save Scenario KPIs, state and reward for each interval
        out["ut_loc"][interval] = ut_loc
        out["ref_xy"][interval] = ref_xy
        out["live_tilt_deg"][interval] = live_tilt_deg
        out["own_occupancy"][interval] = own_occupancy
        out["own_sinr_p25"][interval] = own_sinr_p25
        out["own_sinr_p50"][interval] = own_sinr_p50
        out["own_sinr_p75"][interval] = own_sinr_p75
        out["all_occupancy"][interval] = all_occupancy
        out["all_sinr_p50"][interval] = all_sinr_p50
        out["reward"][interval] = compute_reward(engine.reward_type, reward_per_sector) # compute reward
        out["state_ready"][interval] = state_ready
        out["observation"][interval] = observations

    return out


def _randomize_location_reporting_mask(engine: SimulationEngine, non_anchor_fraction: float = None,
                                        seed: int = 0) -> None:
    """i.i.d. per-UE Bernoulli reporting assignment (NOT per-cluster):
    each UE independently becomes non-anchor with probability
    non_anchor_fraction (defaults to 1 - engine.location_reporting_fraction).

    Unlike the per-cluster-uniform rule SimulationEngine.__init__ uses in
    PRODUCTION (deliberately, to remove run-to-run variance -- untouched
    here), random per-UE assignment lets a cluster end up with fewer, or
    occasionally ZERO, anchors purely by chance -- the actual condition
    needed to see anchor-only measurement genuinely degrade. The
    per-cluster rule guarantees >=1 anchor in every 5-member cluster
    unless non_anchor_fraction exceeds ~90% (round(5*0.9)=4, not 5) --
    confirmed 2026-09-20 as the reason anchor-only tracked ground truth
    almost exactly regardless of the configured reporting fraction: every
    non-anchor UE always had 1-4 same-cluster anchors within
    deviation_radius_m of it, so nothing was ever really "missing" at
    region granularity. This is a debug-only override -- production's own
    per-cluster rule is untouched.
    """
    frac = non_anchor_fraction if non_anchor_fraction is not None else (1.0 - engine.location_reporting_fraction)
    rng = np.random.default_rng(seed)
    engine.location_reporting_mask = rng.random(engine.num_ut) >= frac


def _force_fixed_anchors_per_sector_region(engine: SimulationEngine, anchors_per_region,
                                           seed: int = 0) -> None:
    """2026-10-13, explicit request: the "[1,1,1]"/"[0,1,1]" anchor stress
    condition -- exactly `anchors_per_region` anchor CLUSTER(s) (not
    individual UEs) picked PER SECTOR PER REGION (near/mid/far), ALL of
    that cluster's own UEs becoming anchors -- matching PRODUCTION's own
    per-cluster-uniform anchor-reporting convention (see
    _randomize_location_reporting_mask's own docstring for the contrast
    with the per-UE random rule everywhere else in this debug script).
    Overrides whatever engine.location_reporting_mask the last
    _force_synchronized_demand_state/_randomize_location_reporting_mask
    call set.

    :param anchors_per_region: int (same count for every region) or
        array-like [num_regions] (region order near=0/mid=1/far=2, per
        OccupancyEstimator.region_names) -- e.g. [0,1,1] gives the near
        region's own cluster(s) in every sector ZERO reporting clusters
        this trial, mid/far get 1 each.

    Why per-SECTOR, not one anchor cluster per region network-wide: a
    single network-wide anchor per region would leave any one target
    sector's own near/mid/far groups anchor-less almost always (that
    sector is only ~1/num_bs of the network), making a sector-local
    comparison (e.g. compute_real_ue_tilt_sensitivity_anchor_compare's own
    sector-0 sweep) undefined. Per-sector-per-region instead gives EVERY
    sector exactly `anchors_per_region` anchor cluster(s) in each of its
    own near/mid/far groups whenever that group is non-empty.

    A cluster's own region is classified from its ref_xy (center point),
    not a majority vote over its own (possibly boundary-straddling, see
    hotspot_region_points's own narrow-band fallback) members -- matches
    how every other region-membership check in this file treats a single
    point's own real geometric region.

    Must be called AFTER mobility/ref_xy is already set for this trial
    (i.e. after _force_synchronized_demand_state).
    """
    mobility = engine.mobility
    cell_to_region = engine.regional_info_provider.cell_to_region
    ref_xy_np = mobility.ref_xy.detach().cpu().numpy()
    cluster_sector = mobility.cluster_sector  # [num_clusters] int, plain numpy already
    _, ref_cell_idx = engine.topology.assign_to_sector_rhombus_grid(ref_xy_np)
    cluster_region = cell_to_region[np.clip(ref_cell_idx, 0, len(cell_to_region) - 1)]  # [num_clusters]

    num_regions = len(OccupancyEstimator.region_names)
    counts = (np.full(num_regions, anchors_per_region) if np.isscalar(anchors_per_region)
             else np.asarray(anchors_per_region))

    member_group_idx = mobility.member_group_idx.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    anchor_clusters = []
    for s in range(engine.num_bs):
        for r in range(num_regions):
            if counts[r] <= 0:
                continue
            clusters_here = np.flatnonzero((cluster_sector == s) & (cluster_region == r))
            if clusters_here.size == 0:
                continue
            chosen = rng.choice(clusters_here, size=min(int(counts[r]), clusters_here.size), replace=False)
            anchor_clusters.extend(chosen.tolist())

    engine.location_reporting_mask = np.isin(member_group_idx, np.asarray(anchor_clusters, dtype=int))


def _force_synchronized_demand_state(engine: SimulationEngine, state_idx: int,
                                     non_anchor_fraction: float = None, mask_seed: int = 0) -> None:
    """Overrides an already-built HotspotDemandMobility so every sector
    starts in the SAME demand state (index into mobility.demand_states),
    for a controlled single-distribution measurement run -- phase 4's
    measurement-aggregation comparison needs one clean "the network is
    near-heavy" condition, but demand_states normally runs as per-sector
    INDEPENDENT Markov/periodic chains (see HotspotDemandMobility), so no
    single global interval is naturally "the whole network in state K".

    Must be called immediately after build_debug_engine(), before any
    step_environment() call -- replays the exact sequence
    HotspotDemandMobility.__init__ itself uses (state_idx -> cluster_type
    -> ref_xy -> clustered -> ut_loc/member_group_idx), so there's no
    intervening per-step state (hold timers, walk offsets) to go stale.
    Also re-draws engine.location_reporting_mask via
    _randomize_location_reporting_mask (SimulationEngine.__init__ built it
    from the ORIGINAL random member_group_idx, now replaced, AND used the
    per-cluster rule -- see that function's own docstring for why this
    phase deliberately overrides it).

    :param non_anchor_fraction: passed straight through to
        _randomize_location_reporting_mask -- defaults to
        1 - engine.location_reporting_fraction (config.yaml's own value)
        when omitted.
    :param mask_seed: passed to _randomize_location_reporting_mask as
        `seed` -- vary this across repeated calls on the SAME engine (a
        multi-trial average) so each trial draws a genuinely different
        anchor subset, not the same one every time. Mobility's own cluster
        positions already vary automatically across repeated calls (the
        UeDropper's torch.Generator advances statefully each call), no
        seed needed for that part.
    """
    mobility = engine.mobility
    num_bs = engine.topology.num_bs
    mobility._state_idx[:] = state_idx
    counts_per_sector = mobility.demand_states[mobility._state_idx]
    region_types = OccupancyEstimator.region_names  # ("near","mid","far") -- demand_states column order
    mobility.cluster_type = np.concatenate(
        [np.repeat(np.array(region_types, dtype=object), counts_per_sector[s])
         for s in range(num_bs)])
    mobility.ref_xy = mobility._sample_all_ref_points(mobility.cluster_sector, mobility.cluster_type)
    # Same per-cluster list engine.__init__ itself derived num_ut from (see
    # its own comment there) -- a plain scalar floor-divide here (as this
    # used to do) silently drops engine.num_ut's own remainder clusters
    # entirely whenever num_ut doesn't divide evenly by num_clusters
    # (2026-10-11 fix: 300 UEs / 105 clusters produced only 210, a real
    # shape-mismatch crash against every other num_ut=300-sized engine buffer).
    ut_loc, member_group_idx = mobility._sampler.clustered(
        mobility.ref_xy.tolist(), engine.num_ues_per_cluster, mobility.deviation_radius, engine.ut_height)
    mobility.member_group_idx = member_group_idx
    mobility._z = ut_loc[:, 2:3].clone()
    mobility.ut_loc = ut_loc

    _randomize_location_reporting_mask(engine, non_anchor_fraction=non_anchor_fraction, seed=mask_seed)


def _force_specific_demand_states(engine: SimulationEngine, sector_state_idx: dict,
                                  default_state_idx: int, non_anchor_fraction: float = None,
                                  mask_seed: int = 0) -> None:
    """Same mechanics as _force_synchronized_demand_state (see its own
    docstring for the full replay-sequence rationale), but each sector can
    be pinned to its OWN demand state instead of every sector sharing one --
    e.g. the neighbor-effect joint sweep below, which needs sector 0/N1/N2
    independently set to near-heavy or far-heavy while every OTHER sector
    stays at one fixed background state.

    :param sector_state_idx: {sector_idx: demand_states row index} -- only
        the sectors named here deviate from default_state_idx.
    :param default_state_idx: demand_states row index for every sector NOT
        named in sector_state_idx.
    """
    mobility = engine.mobility
    num_bs = engine.topology.num_bs
    mobility._state_idx[:] = default_state_idx
    for s, idx in sector_state_idx.items():
        mobility._state_idx[s] = idx
    counts_per_sector = mobility.demand_states[mobility._state_idx]
    region_types = OccupancyEstimator.region_names  # ("near","mid","far") -- demand_states column order
    mobility.cluster_type = np.concatenate(
        [np.repeat(np.array(region_types, dtype=object), counts_per_sector[s])
         for s in range(num_bs)])
    mobility.ref_xy = mobility._sample_all_ref_points(mobility.cluster_sector, mobility.cluster_type)
    ut_loc, member_group_idx = mobility._sampler.clustered(
        mobility.ref_xy.tolist(), engine.num_ues_per_cluster, mobility.deviation_radius, engine.ut_height)
    mobility.member_group_idx = member_group_idx
    mobility._z = ut_loc[:, 2:3].clone()
    mobility.ut_loc = ut_loc

    _randomize_location_reporting_mask(engine, non_anchor_fraction=non_anchor_fraction, seed=mask_seed)


def run_measurement_aggregation_scenario(engine: SimulationEngine, num_intervals: int = 1) -> dict:
    """Runs num_intervals fixed-tilt (SINGLE_SECTOR_NOMINAL_TILT_DEG, a
    realistic operating-range baseline -- 2026-10-13, was 0 deg, which sits
    outside config.yaml's own configured downtilt_sweep_deg range [5,13]
    entirely and, under the current antenna config, collapsed SINR to
    around -3 to -4.5dB network-wide: below coverage_threshold_db for
    EVERY UE regardless of anchor status, making ground_truth and
    anchor_only's own coverage figures identically 0.0 and uninformative --
    see _tilt_sensitivity_single_sector_figure's own 2026-10-08 fix for the
    same root issue in a different function) intervals on an engine already
    forced into one demand_states distribution (_force_synchronized_demand_state), and
    pools every real UE's own ground-truth (SINR, RSRP, coverage, TRUE
    region) each interval, plus which UEs are anchors this run
    (engine.location_reporting_mask).

    num_intervals=1 (2026-09-20, was 15): production's own own_occupancy
    is a SINGLE interval's raw instantaneous value now (EMA smoothing
    removed the same day) -- pooling many intervals here gave anchor-only
    a far larger sample than production's real estimate ever gets, making
    it look artificially robust to missing UEs. One interval per
    distribution is what actually matches how the real state is computed.

    No Kriging/CGR/region-inference of any kind -- this phase compares
    only ground truth (every real UE, known) against anchor-only (the
    SAME quantities restricted to anchor UEs, raw and uninterpolated, no
    filling-in for non-anchor UEs). RME/radio-map-assisted reconstruction
    is deliberately out of scope here (deferred, see phase 5 for
    Kriging/CGR's own validation).

    :output: dict --
        ue_region_true, ue_sinr_db, ue_rsrp_db, ue_covered: [total_ue_obs]
            arrays, one row per (interval, UE) real observation, UEs in a
            CONSISTENT order every interval (so a per-UE mask can be
            np.tile'd across intervals -- see the percentage-sweep figure).
        ue_is_anchor: [total_ue_obs] bool.
        num_ut, num_intervals: for reconstructing/re-tiling a per-UE mask.
        member_group_idx: [num_ut] int, each UE's own cluster id -- fixed
            for the whole run (cluster MEMBERSHIP doesn't change with
            demand-state transitions, only which region a cluster is
            assigned to does), captured once for a cluster-level (not
            per-UE) anchor mask -- see
            _randomize_cluster_level_reporting_mask.
    """
    num_bs = engine.num_bs
    cell_to_region = engine.regional_info_provider.cell_to_region
    member_group_idx = engine.mobility.member_group_idx.detach().cpu().numpy().copy()

    ue_region_true, ue_sinr_db, ue_rsrp_db, ue_covered, ue_is_anchor = [], [], [], [], []

    for interval in range(num_intervals):
        live_tilt_deg = np.full(num_bs, SINGLE_SECTOR_NOMINAL_TILT_DEG)
        state = engine.step_environment(interval == num_intervals - 1)
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        _, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])

        power_w = engine.kpi_manager.resolve_power_at_tilt(state, live_tilt_deg)[0]
        kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
        sinr_db, rsrp_db = kpis["sinr_db"][0], kpis["rsrp_dbm"][0]
        covered_ue = sinr_db > engine.coverage_threshold_db

        region_true = cell_to_region[np.clip(cell_idx, 0, len(cell_to_region) - 1)]

        ue_region_true.append(region_true)
        ue_sinr_db.append(sinr_db)
        ue_rsrp_db.append(rsrp_db)
        ue_covered.append(covered_ue)
        ue_is_anchor.append(engine.location_reporting_mask.copy())

    return dict(
        ue_region_true=np.concatenate(ue_region_true),
        ue_sinr_db=np.concatenate(ue_sinr_db), ue_rsrp_db=np.concatenate(ue_rsrp_db),
        ue_covered=np.concatenate(ue_covered), ue_is_anchor=np.concatenate(ue_is_anchor),
        num_ut=engine.num_ut, num_intervals=num_intervals, member_group_idx=member_group_idx,
    )


def run_measurement_aggregation_scenario_rme(engine: SimulationEngine, num_intervals: int = 1) -> dict:
    """Same real per-(interval,UE) collection as
    run_measurement_aggregation_scenario, PLUS -- for phase 5's own RME-
    based region-inference comparison -- a Kriging AND a CGR
    SpatialGridEstimator, each fed ONLY anchor UEs' real (location, RSRP)
    every interval (never non-anchor UEs' -- exactly what a real
    deployment would actually have), producing region-membership
    PROBABILITIES for every UE via infer_region_probabilities: one-hot at
    the real region for an anchor UE (no inference attempted -- its region
    IS known), the Kriging/CGR fit's own inferred distribution for a non-
    anchor UE. A non-anchor UE's own real SINR/RSRP/coverage is NEVER
    estimated -- only which region to file that already-known value under
    (see RegionalInfoProvider's own module docstring).

    :output: run_measurement_aggregation_scenario's own dict, PLUS
        ue_region_probs_kriging, ue_region_probs_cgr: [total_ue_obs,
        num_regions], each row summing to 1.
    """
    num_bs = engine.num_bs
    cell_to_region = engine.regional_info_provider.cell_to_region
    num_regions = engine.regional_info_provider.num_own_regions
    member_group_idx = engine.mobility.member_group_idx.detach().cpu().numpy().copy()
    spatial_grid_cell_size_m = load_config()["topology"]["spatial_grid_cell_size_m"]

    common = dict(num_bs=num_bs, num_grid_cells=engine.num_grid_cells, grid_points=engine.grid_points,
                 length_scale=spatial_grid_cell_size_m, coverage_threshold_db=engine.coverage_threshold_db,
                 radio_map_method="rsrp_per_sector", noise_power_w=engine.kpi_manager.noise_power_w,
                 topology=engine.topology)
    kriging_estimator = SpatialGridEstimator(**common, radio_map_estimator="kriging")
    cgr_estimator = SpatialGridEstimator(
        **common, radio_map_estimator="cgr", cgr_num_gaussian=engine.cgr_cfg.get("num_gaussian", 12),
        cgr_ridge_lambda=engine.cgr_cfg.get("ridge_lambda", 1e-4),
        cgr_baseline_method=engine.cgr_cfg.get("baseline_method", "global_mean"),
        cgr_baseline_length_scale_m=engine.cgr_cfg.get("baseline_length_scale_m", spatial_grid_cell_size_m),
        cgr_gaussian_config={k: v for k, v in engine.cgr_cfg.items()
                             if k not in ("num_gaussian", "ridge_lambda", "baseline_method",
                                         "baseline_length_scale_m")})

    (ue_region_true, ue_sinr_db, ue_rsrp_db, ue_covered, ue_is_anchor,
     ue_region_probs_kriging, ue_region_probs_cgr) = [], [], [], [], [], [], []
    # 2026-10-13: was np.zeros(num_bs) (0 deg) -- same degenerate-coverage
    # bug as run_measurement_aggregation_scenario's own fix, see there.
    live_tilt_deg = np.full(num_bs, SINGLE_SECTOR_NOMINAL_TILT_DEG)

    for interval in range(num_intervals):
        state = engine.step_environment(interval == num_intervals - 1)
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])

        power_w = engine.kpi_manager.resolve_power_at_tilt(state, live_tilt_deg)[0]
        kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
        sinr_db, rsrp_db = kpis["sinr_db"][0], kpis["rsrp_dbm"][0]
        covered_ue = sinr_db > engine.coverage_threshold_db
        region_true = cell_to_region[np.clip(cell_idx, 0, len(cell_to_region) - 1)]
        is_anchor = engine.location_reporting_mask.copy()

        power_db_all_sectors = 10.0 * np.log10(power_w)
        reporting_sector_idx = np.where(is_anchor, sector_idx, -1)
        kriging_estimator.accumulate(reporting_sector_idx, cell_idx, covered_ue,
                                     power_db_all_sectors=power_db_all_sectors)
        cgr_estimator.accumulate(reporting_sector_idx, cell_idx, covered_ue,
                                 power_db_all_sectors=power_db_all_sectors)

        num_ue = len(sector_idx)
        probs_kriging = np.zeros((num_ue, num_regions))
        probs_cgr = np.zeros((num_ue, num_regions))
        probs_kriging[np.arange(num_ue), region_true] = 1.0
        probs_cgr[np.arange(num_ue), region_true] = 1.0
        non_anchor = ~is_anchor
        if non_anchor.any():
            safe_sector_idx = np.clip(sector_idx, 0, num_bs - 1)
            measured_rsrp_db = power_db_all_sectors[safe_sector_idx, np.arange(num_ue)]
            probs_kriging[non_anchor] = kriging_estimator.infer_region_probabilities(
                safe_sector_idx[non_anchor], measured_rsrp_db[non_anchor],
                cell_to_region, num_regions, engine.region_inference_sigma_db)
            probs_cgr[non_anchor] = cgr_estimator.infer_region_probabilities(
                safe_sector_idx[non_anchor], measured_rsrp_db[non_anchor],
                cell_to_region, num_regions, engine.region_inference_sigma_db,
                radio_map_estimator="cgr", current_tilts_deg=live_tilt_deg)

        kriging_estimator.reset()
        cgr_estimator.reset()

        ue_region_true.append(region_true)
        ue_sinr_db.append(sinr_db)
        ue_rsrp_db.append(rsrp_db)
        ue_covered.append(covered_ue)
        ue_is_anchor.append(is_anchor)
        ue_region_probs_kriging.append(probs_kriging)
        ue_region_probs_cgr.append(probs_cgr)

    return dict(
        ue_region_true=np.concatenate(ue_region_true),
        ue_sinr_db=np.concatenate(ue_sinr_db), ue_rsrp_db=np.concatenate(ue_rsrp_db),
        ue_covered=np.concatenate(ue_covered), ue_is_anchor=np.concatenate(ue_is_anchor),
        ue_region_probs_kriging=np.concatenate(ue_region_probs_kriging),
        ue_region_probs_cgr=np.concatenate(ue_region_probs_cgr),
        num_ut=engine.num_ut, num_intervals=num_intervals, member_group_idx=member_group_idx,
    )

# =============================================================
# SECTOR VISUALIZATION
# =============================================================

def draw_sites_and_sectors(ax, topo, sector_ids=None, site_color=SITE_COLOR,
                           sector_color=SECTOR_COLOR, sector_linestyle="--",
                           site_zorder=5, sector_zorder=5, id_fontsize=7) -> None:
    """Hex (site) boundaries + sector-division lines, drawn directly from
    HexGrid's own cell corners (NOT via its show(show_sectors=True), which
    also stamps a confusing per-site-local "1/2/3" label we don't want).
    Labels every sector in `sector_ids` (default: all) with its GLOBAL
    index, placed the same way sionna's own show() places its own labels
    (inside the wedge, near its outer edge) -- just matched to the correct
    global sector by nearest boresight angle, not sionna's own local
    numbering.

    :param site_color, sector_color, sector_linestyle: site boundary is
        always solid; sector division lines use sector_linestyle. Default
        colors/style match the original green-solid/green-dashed convention.
    :param site_zorder, sector_zorder: paint order -- pass site_zorder >
        sector_zorder (and both above draw_regions's own zorder) so
        coincident/overlapping boundaries show the higher-priority one on
        top. ID labels are always placed above both, so they stay legible
        regardless of underlying line overlaps.
    """
    if sector_ids is None:
        sector_ids = range(topo.num_bs)
    boresight = topo.bs_orientations[0, :, 0].cpu().numpy()
    n_per_site = topo.num_sectors_per_site
    sector_ids = set(sector_ids)
    label_zorder = max(site_zorder, sector_zorder) + 2

    for cell_idx, cell in topo.grid._grid.items():
        corners = cell.corners().cpu().numpy()
        center = cell.coord_euclid.cpu().numpy()

        # Site boundary: always solid.
        ax.plot([corners[-1][0]] + [c[0] for c in corners],
               [corners[-1][1]] + [c[1] for c in corners],
               color=site_color, linewidth=1.2, zorder=site_zorder)

        # Sector division lines: one per wedge corner (0, 2, 4).
        site_sectors = [s for s in range(cell_idx * n_per_site, (cell_idx + 1) * n_per_site)]
        for ii in (0, 2, 4):
            ax.plot([center[0], corners[ii][0]], [center[1], corners[ii][1]],
                   linestyle=sector_linestyle, color=sector_color, linewidth=1.0, zorder=sector_zorder)
            # Match this wedge to whichever of the site's sectors points
            # closest to it (angle from center to the wedge's own far corner).
            wedge_angle = np.arctan2(corners[ii + 1][1] - center[1], corners[ii + 1][0] - center[0])
            angle_diff = np.abs(np.angle(np.exp(1j * (boresight[site_sectors] - wedge_angle))))
            sector = site_sectors[int(np.argmin(angle_diff))]
            if sector in sector_ids:
                label_xy = (center + corners[ii + 1]) / 2
                # This point sits right where a near/mid/far region's own
                # label can land too (see draw_regions) -- an opaque bbox
                # cleanly occludes whichever one collides, regardless of
                # which region it happens to be, rather than guessing a
                # single region name to suppress.
                ax.annotate(str(sector), label_xy, fontsize=id_fontsize, ha="center", va="center",
                           color="#0b0b0b", fontweight="bold", zorder=label_zorder,
                           bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def classify_fine_grid(topo, s: int, n: int) -> np.ndarray:
    """Classifies a sector's n x n rhombus fine grid into near/mid/far via
    CellularTopology.hotspot_region_membership, evaluated at each cell's
    CENTER -- the same rule OccupancyEstimator's own cell_to_region uses.

    :output: [n, n] str array, row=i (alpha index), col=j (beta index).
    """
    b0, v1, v2 = topo._sector_grid_bs_xy[s], topo._sector_grid_e1[s], topo._sector_grid_e2[s]
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    alpha_c, beta_c = (i_idx + 0.5) / n, (j_idx + 0.5) / n
    xy = b0[None, None, :] + alpha_c[..., None] * v1[None, None, :] + beta_c[..., None] * v2[None, None, :]
    m = topo.hotspot_region_membership(s, xy.reshape(-1, 2))
    return np.where(m["near"], "near", np.where(m["mid"], "mid", "far")).reshape(n, n)


def draw_region_boundaries(ax, engine, sector_ids, color=REGION_COLOR, linestyle="-",
                           linewidth=0.5, zorder=3) -> None:
    """Outer-boundary lines ONLY (no fill, no label) for each sector's own
    near/mid/far hotspot regions -- the ONE region definition in this
    codebase (CellularTopology.hotspot_region_membership), classified over
    the sector's own n x n rhombus fine grid (n = topo._sector_grid_n) so
    boundaries land exactly on real fine-grid-cell edges. Only the OUTER
    boundary of each region is drawn (a cell edge is drawn iff its
    neighboring cell belongs to a different region, or it's the grid's own
    outer edge) -- not every individual fine-cell edge. Factored out of
    draw_regions so draw_regions_filled (phase6/9) can overlay the SAME
    boundary convention on top of a value-colored fill, making a real
    inter-region edge visually distinct from an ordinary intra-region
    fine-cell edge.

    :param zorder: pass BELOW draw_sites_and_sectors's own site_zorder/
        sector_zorder so region boundaries never paint over site/sector
        lines where they coincide (site > sector > region priority).
    """
    topo = engine.topology
    bs_xy, e1, e2, n = (topo._sector_grid_bs_xy, topo._sector_grid_e1,
                        topo._sector_grid_e2, topo._sector_grid_n)
    for s in sector_ids:
        region_grid = classify_fine_grid(topo, s, n)
        b0, v1, v2 = bs_xy[s], e1[s], e2[s]
        for i in range(n):
            for j in range(n):
                here = region_grid[i, j]
                edges = []
                if i == n - 1 or region_grid[i + 1, j] != here:
                    edges.append(((i + 1) / n, j / n, (i + 1) / n, (j + 1) / n))
                if j == n - 1 or region_grid[i, j + 1] != here:
                    edges.append((i / n, (j + 1) / n, (i + 1) / n, (j + 1) / n))
                if i == 0:
                    edges.append((0.0, j / n, 0.0, (j + 1) / n))
                if j == 0:
                    edges.append((i / n, 0.0, (i + 1) / n, 0.0))
                for a0, b_0, a1, b1 in edges:
                    p0, p1 = b0 + a0 * v1 + b_0 * v2, b0 + a1 * v1 + b1 * v2
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linestyle=linestyle,
                           linewidth=linewidth, zorder=zorder)


def draw_regions(ax, engine, sector_ids, label: bool = True, color=REGION_COLOR,
                 linestyle="-", zorder=3, id_fontsize=4.5, hide_region_ids=frozenset()) -> None:
    """draw_region_boundaries plus a region-name label per sector, plus (for
    visual clarity -- boundary lines alone are easy to miss) a light
    near/mid/far color shading using the SAME HOTSPOT_REGION_COLORS
    convention phase 3's draw_hotspot_regions uses, aligned exactly to the
    real fine grid (resolution=n) so the shading and the boundary lines
    coincide pixel-for-pixel.

    :param hide_region_ids: region names ("near"/"mid"/"far") to skip
        LABELING for (boundary + shading still drawn) -- e.g. a region's
        label can land right where the caller also places its own
        sector-ID label; draw_sites_and_sectors's own opaque bbox handles
        that collision, so this is rarely needed now.
    """
    topo = engine.topology
    bs_xy, e1, e2, n = (topo._sector_grid_bs_xy, topo._sector_grid_e1,
                        topo._sector_grid_e2, topo._sector_grid_n)
    draw_hotspot_regions(ax, topo, sector_ids, resolution=n, fill_alpha=0.35, zorder=max(zorder - 1, 0))
    draw_region_boundaries(ax, engine, sector_ids, color=color, linestyle=linestyle, linewidth=0.5, zorder=zorder)
    if label:
        label_zorder = zorder + 2
        for s in sector_ids:
            region_grid = classify_fine_grid(topo, s, n)
            b0, v1, v2 = bs_xy[s], e1[s], e2[s]
            for region_type in ("near", "mid", "far"):
                if region_type in hide_region_ids:
                    continue
                mask = region_grid == region_type
                if not mask.any():
                    continue
                # "mid"/"far" are RING-shaped (an annulus around "near"), so
                # the naive mean of (i,j) indices lands near the ring's own
                # hole, not inside the ring itself -- snap to whichever
                # actual in-region cell sits closest to that mean instead,
                # guaranteeing the label lands on a real cell of that region
                # (2026-10-07 fix -- most visible at coarse grid resolutions,
                # where the ring is thin/jagged enough for this to be obvious).
                ii, jj = np.nonzero(mask)
                mean_i, mean_j = ii.mean(), jj.mean()
                nearest = np.argmin((ii - mean_i) ** 2 + (jj - mean_j) ** 2)
                pos = b0 + ((ii[nearest] + 0.5) / n) * v1 + ((jj[nearest] + 0.5) / n) * v2
                ax.annotate(region_type, pos, fontsize=id_fontsize, ha="center",
                           va="center", color=color, zorder=label_zorder)

def draw_hotspot_regions(ax, topo, sector_ids,
                         colors=HOTSPOT_REGION_COLORS, resolution=20, fill_alpha=0.45, zorder=0) -> None:
    """Fills each sector's own near/mid/far hotspot region with a distinct
    flat color -- classified EXACTLY via
    CellularTopology.hotspot_region_membership on a resolution x
    resolution grid of (alpha, beta) parallelogram cells per sector
    (independent of the rhombus fine grid used elsewhere in this script).
    Every cell gets exactly one of near/mid/far (no excluded hole, unlike
    the earlier eta-threshold definition this replaces).
    """
    bs_xy, e1, e2 = topo._sector_grid_bs_xy, topo._sector_grid_e1, topo._sector_grid_e2
    edges = np.linspace(0.0, 1.0, resolution + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    aa, bb = np.meshgrid(centers, centers, indexing="ij")
    ai, bi = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    aa, bb, ai, bi = aa.ravel(), bb.ravel(), ai.ravel(), bi.ravel()
    for s in sector_ids:
        b0, v1, v2 = bs_xy[s], e1[s], e2[s]
        xy = b0[None, :] + aa[:, None] * v1[None, :] + bb[:, None] * v2[None, :]
        m = topo.hotspot_region_membership(s, xy)
        for region_type, color in colors.items():
            for idx in np.flatnonzero(m[region_type]):
                i, j = ai[idx], bi[idx]
                corners = np.array([
                    b0 + edges[i] * v1 + edges[j] * v2, b0 + edges[i + 1] * v1 + edges[j] * v2,
                    b0 + edges[i + 1] * v1 + edges[j + 1] * v2, b0 + edges[i] * v1 + edges[j + 1] * v2,
                ])
                ax.add_patch(mpatches.Polygon(corners, closed=True, facecolor=color,
                                              edgecolor="none", alpha=fill_alpha, zorder=zorder))


def draw_regions_filled(ax, engine, sector_ids, values: np.ndarray, cmap=DENSITY_CMAP, vmin=0.0, vmax=1.0) -> None:
    """Same near/mid/far region geometry as draw_regions, but filled by
    color instead of outlined -- `values` is [num_bs, num_own_regions]
    (region order: near=0, mid=1, far=2, matching OccupancyEstimator's own
    region_names) (e.g. occupancy or coverage). Every fine-grid cell in a
    region is filled with that region's single scalar value's color -- a
    thin gray edge on each cell keeps the underlying fine-grid visible, and
    a crisp black overlay (draw_region_boundaries, same boundary convention
    as draw_regions) marks the real near/mid/far borders, which the
    value-driven fill color alone can't reliably show (two adjacent regions
    can land on a similar value/color).
    """
    topo = engine.topology
    bs_xy, e1, e2, n = (topo._sector_grid_bs_xy, topo._sector_grid_e1, topo._sector_grid_e2, topo._sector_grid_n)
    colormap = plt.get_cmap(cmap)
    region_index = {"near": 0, "mid": 1, "far": 2}
    for s in sector_ids:
        region_grid = classify_fine_grid(topo, s, n)
        b0, v1, v2 = bs_xy[s], e1[s], e2[s]
        for i in range(n):
            for j in range(n):
                region_idx = region_index[region_grid[i, j]]
                corners = np.array([
                    b0 + i / n * v1 + j / n * v2,
                    b0 + (i + 1) / n * v1 + j / n * v2,
                    b0 + (i + 1) / n * v1 + (j + 1) / n * v2,
                    b0 + i / n * v1 + (j + 1) / n * v2,
                ])
                norm_value = np.clip((values[s, region_idx] - vmin) / (vmax - vmin), 0.0, 1.0)
                ax.add_patch(mpatches.Polygon(corners, closed=True, facecolor=colormap(norm_value),
                                              edgecolor="gray", linewidth=0.3, zorder=2))
    draw_region_boundaries(ax, engine, sector_ids, color="black", linestyle="-", linewidth=1.2, zorder=3)


def draw_grid_points(ax, engine, sector_ids=None, color=GRID_COLOR, zorder=1, size=2) -> None:
    """Dots at every rhombus fine-grid cell center."""
    pts = engine.grid_points if sector_ids is None else engine.grid_points[list(sector_ids)]
    xy = pts.reshape(-1, 2)
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, alpha=0.5, linewidths=0, zorder=zorder)

# =============================================================
# PHASE 1
# =============================================================
def phase1_static_geometry(engine: SimulationEngine, out_dir: str) -> None:
    """
    One overview figure (all sites/sectors/grid points)
    One combined panel figure (all sectors' own zoomed neighborhoods)
    One standalone zoomed figure per sector (same content as its panel)

    No region overlay here (2026-10-10): region membership isn't defined
    off anything real yet at this phase -- no measurement exists to pick
    between the radial-distance (CellularTopology.hotspot_region_membership)
    and SINR-measurement-based definitions, so drawing either one here would
    be premature. The radial-distance overlay now lives in
    phase2_radio_environment_validation instead, where it's drawn against
    real measured RSRP/SINR.

    Shared visual convention: black solid=site, blue densely-dashed=sector,
    red dots=grid point, explained via a legend (not a text caption). Draw
    priority when boundaries overlap is site > sector (see zorder args
    below); ID labels always sit above every line.
    """
    os.makedirs(out_dir, exist_ok=True)
    topo = engine.topology
    num_bs = topo.num_bs

    sector_linestyle = (0, (5, 1))  # densely dashed
    legend_handles = [
        Line2D([0], [0], color=SITE_COLOR, linewidth=1.5, linestyle="-", label="site"),
        Line2D([0], [0], color=SECTOR_COLOR, linewidth=1.2, linestyle=sector_linestyle, label="sector"),
        Line2D([0], [0], color=GRID_COLOR, linewidth=0, marker="o", markersize=5, label="grid point"),
    ]

    def draw_layer(ax, sector_ids=None) -> None:
        draw_sites_and_sectors(ax, topo, sector_ids=sector_ids, site_color=SITE_COLOR, sector_color=SECTOR_COLOR,
                               sector_linestyle=sector_linestyle, site_zorder=5, sector_zorder=4, id_fontsize=10)
        draw_grid_points(ax, engine, sector_ids=sector_ids, color=GRID_COLOR, zorder=1, size=6)

    # Overview: every sector's own sites/sectors/grid points at once.
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_layer(ax)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.set_title(f"Network overview: {num_bs} sectors", fontsize=10)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, "overview_sites_sectors_grid.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Per-sector zoomed neighborhood -- own X_s (sector + side-sharing
    # neighbors) only. Own sector marked with a star in a color distinct
    # from every other element.
    bs_xy = topo.bs_loc[0, :, :2].cpu().numpy()
    cell_radius = float(topo.grid.cell_radius.item())

    def draw_sector_panel(ax, s: int) -> None:
        sector_set = [s] + engine.regional_info_provider.neighbor_sets[s]
        draw_layer(ax, sector_ids=sector_set)
        ax.plot(*bs_xy[s], marker="*", markersize=16, color="darkorange", zorder=6)
        member_xy = bs_xy[sector_set]
        pad = 1.3 * cell_radius
        ax.set_xlim(member_xy[:, 0].min() - pad, member_xy[:, 0].max() + pad)
        ax.set_ylim(member_xy[:, 1].min() - pad, member_xy[:, 1].max() + pad)
        ax.set_title(f"sector {s} + {len(sector_set) - 1} neighbors", fontsize=9)
        ax.set_aspect("equal")

    # Combined panel: all sectors' own zoomed neighborhoods in one figure.
    ncols = 7
    nrows = int(np.ceil(num_bs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    for s in range(num_bs):
        ax = axes[s // ncols, s % ncols]
        draw_sector_panel(ax, s)
        ax.tick_params(labelsize=6)
    for s in range(num_bs, nrows * ncols):
        axes[s // ncols, s % ncols].axis("off")
    # ncols evenly divides num_bs here, so every grid cell is a subplot --
    # no free space for the legend inside the grid; reserve a top margin
    # for suptitle + legend instead (same fix as save_mobility_animation's
    # own title-clipping issue).
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Per-sector neighborhood (star = own sector)", fontsize=12, y=0.99)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, fontsize=9)
    fig.savefig(os.path.join(out_dir, "per_sector_neighborhoods.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Same content, one standalone figure per sector.
    for s in range(num_bs):
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_sector_panel(ax, s)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        fig.savefig(os.path.join(out_dir, f"sector_{s}_neighborhood.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Phase 1 done -- saved to {out_dir}/")

# =============================================================
# PHASE 2
# =============================================================

def phase3_raw_ue_behavior(engine: SimulationEngine, scenario: dict, out_dir: str,
                           run_neighbor_joint_sweep: bool = False,
                           run_neighbor_joint_sweep_grouped: bool = False) -> None:
    """
    Three raw-UE spatial checks are performed before grid/channel processing:

    1) cluster_snapshots.png: Use the full config.yaml hotspot settings to
    visualize D1/D2/D3 at NUM_SNAPSHOTS successive hold periods (more than
    len(demand_states) under state_transition_mode="periodic" so the
    cycle's own wraparound/repeat is visible, not just each state once),
    including cluster boundaries and near/mid/far regions. Non-anchor UEs
    (location_reporting_mask False -- see SimulationEngine) are drawn in
    NON_ANCHOR_COLOR (gray) instead of their cluster color, so we can
    visually confirm where they actually are even though the real
    controller never gets to see this -- only their own serving-sector
    measurement, never their position.

    2) sector_ue_counts_over_time.png: Track UE population for all 21
    sectors over time to verify distinct, time-varying spatial demand and
    ensure occupancy features are informative.

    3) region_occupancy_validation.png: does the REAL simulated UE
    population's near/mid/far split, pooled over several full periodic
    cycles and every sector, actually match each demand_states row's own
    intended ratio? A real empirical check, not a tautology -- a cluster's
    reference point is sampled inside its target region, but members then
    random-walk up to deviation_radius away from it, and near/mid's own
    bands can be narrow relative to deviation_radius (see
    hotspot_region_points' own "falling back to unbuffered" warning), so
    the realized population can plausibly drift from the intended ratio.

    :param run_neighbor_joint_sweep: opt-in (default False), expensive
        (6 cases x 125 own/N1/N2 tilt combos = 750 full-network power
        resolves) sub-experiment -- see _neighbor_joint_sweep_figures' own
        docstring for the 6-case table. Saves
        neighbor_joint_sweep_<case>.png, one per case, into
        out_dir/neighbor_joint_sweep/.
    :param run_neighbor_joint_sweep_grouped: opt-in (default False), same
        cost/cases as run_neighbor_joint_sweep, but sweeps EVERY sector's
        tilt+demand state via its own phase group (not just 3 isolated
        sectors) -- see compute_neighbor_joint_tilt_sweep_grouped's own
        docstring. Saves neighbor_joint_sweep_grouped_<case>.png into
        out_dir/neighbor_joint_sweep_grouped/.
    """

    # Intiialization
    os.makedirs(out_dir, exist_ok=True)
    topo = engine.topology
    ut_loc = scenario["ut_loc"]
    num_intervals = ut_loc.shape[0]

    # Cluster snapshots -- uses the SAME cluster_mobility_mode `engine` (the
    # shared, config.yaml-selected one) was built with, not a hardcoded
    # submode (2026-10-11 fix -- was always "hotspot_demand-markov"
    # regardless of config.yaml), so this figure faithfully reflects the
    # actual configured mobility mode and anchor/non-anchor mix.
    snap_engine = build_debug_engine(cluster_mobility_mode=engine.mobility.cluster_mobility_mode)
    snap_mobility = snap_engine.mobility
    is_anchor = snap_engine.location_reporting_mask

    NUM_SNAPSHOTS = 5  # > len(demand_states) under periodic mode so the cycle's wraparound/repeat is visible too
    if hasattr(snap_mobility, "num_waypoints"):
        snapshot_intervals = [i * snap_mobility.waypoint_hold_steps for i in range(snap_mobility.num_waypoints)]
    elif hasattr(snap_mobility, "demand_states"):
        snapshot_intervals = [i * snap_mobility.hotspot_hold_steps for i in range(NUM_SNAPSHOTS)]
    else:
        snapshot_intervals = [0, NUM_NO_TILT - 1, NUM_NO_TILT, num_intervals - 1]

    # Pure mobility replay
    max_interval = max(snapshot_intervals)
    snap_ut_loc, snap_ref_xy = {}, {}
    for interval in range(max_interval + 1):
        if interval in snapshot_intervals:
            snap_ut_loc[interval] = snap_mobility.ut_loc.detach().cpu().numpy().copy()
            snap_ref_xy[interval] = snap_mobility.ref_xy.detach().cpu().numpy().copy()
        if interval < max_interval:
            snap_mobility.step()

    num_clusters_per_site = snap_mobility.num_clusters_per_sector * NUM_SECTORS_PER_SITE
    cluster_color_pool = plt.get_cmap("tab20", num_clusters_per_site)(range(num_clusters_per_site))
    home_sector, _ = topo.assign_to_sector_rhombus_grid(snap_ref_xy[0])
    home_site = home_sector // NUM_SECTORS_PER_SITE
    num_groups = snap_ref_xy[0].shape[0]
    color_idx = np.zeros(num_groups, dtype=int)
    for site in range(NUM_SITES):
        members = np.where(home_site == site)[0]
        color_idx[members] = np.arange(len(members)) % num_clusters_per_site
    cluster_colors = [cluster_color_pool[c] for c in color_idx]
    member_group_idx = snap_mobility.member_group_idx.cpu().numpy()

    fig, axes = plt.subplots(1, len(snapshot_intervals), figsize=(5.5 * len(snapshot_intervals), 5.5))
    axes = np.atleast_1d(axes)
    for ax, interval in zip(axes, snapshot_intervals):
        # resolution=own grid_n -- same fine-grid quantization phase 1's
        # draw_regions uses, so the region boundary renders identically
        # across phases instead of at this function's smoother 20x20
        # sampling default.
        draw_hotspot_regions(ax, topo, range(topo.num_bs), resolution=topo._sector_grid_n)
        draw_sites_and_sectors(ax, topo)
        xy = snap_ut_loc[interval][:, :2]
        for site in range(NUM_SITES):
            group_ids = np.where(home_site == site)[0]
            member_mask = np.isin(member_group_idx, group_ids)
            if not member_mask.any():
                continue
            point_colors = [cluster_colors[g] if a else NON_ANCHOR_COLOR
                           for g, a in zip(member_group_idx[member_mask], is_anchor[member_mask])]
            ax.scatter(xy[member_mask, 0], xy[member_mask, 1], s=26, c=point_colors,
                      marker=SITE_MARKERS[site], edgecolors="black", linewidths=0.4, zorder=4)
        add_cluster_ellipses(ax, snap_ref_xy[interval], snap_mobility.deviation_radius,
                             edgecolors=cluster_colors, zorder=3)
        ax.set_title(f"interval {interval}", fontsize=10)
        ax.set_aspect("equal")
    marker_handles = [Line2D([0], [0], color="black", marker=SITE_MARKERS[s], linestyle="None",
                             markersize=7, markerfacecolor="none", label=f"site {s}") for s in range(NUM_SITES)]
    region_handles = [mpatches.Patch(facecolor=color, alpha=0.45, label=region_type)
                      for region_type, color in HOTSPOT_REGION_COLORS.items()]
    non_anchor_handle = [Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=7,
                                markerfacecolor=NON_ANCHOR_COLOR, label="non-anchor UE")]
    fig.legend(handles=marker_handles + region_handles + non_anchor_handle, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=NUM_SITES + len(region_handles) + 1, fontsize=8)
    fig.suptitle(f"UE cluster snapshots -- marker=site, color=cluster rank within its site "
                f"(circle = matching cluster boundary; gray = non-anchor, see legend), "
                f"cluster_mobility_mode={snap_mobility.cluster_mobility_mode}, "
                f"anchor_fraction={snap_engine.location_reporting_mask.mean():.2f}", fontsize=12, y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cluster_snapshots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Raw UE count per sector per interval -- all 21 sectors, geography only.
    sector_counts = np.zeros((num_intervals, topo.num_bs))
    for interval in range(num_intervals):
        sector_idx, _ = topo.assign_to_sector_rhombus_grid(ut_loc[interval, :, :2])
        sector_counts[interval] = np.bincount(sector_idx[sector_idx >= 0], minlength=topo.num_bs)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(sector_counts.T, aspect="auto", cmap=DENSITY_CMAP, interpolation="nearest")
    ax.axvline(NUM_NO_TILT - 0.5, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("interval")
    ax.set_ylabel("sector")
    ax.set_title("Raw UE count per sector per interval (dashed = no-tilt/random-tilt boundary)")
    fig.colorbar(im, ax=ax, label="# UEs")
    fig.savefig(os.path.join(out_dir, "sector_ue_counts_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- region occupancy validation: does the REAL simulated UE
    # population's near/mid/far split actually match each demand_states
    # row's own intended ratio? Cluster reference points are sampled
    # inside their target region, but members then random-walk up to
    # deviation_radius away from it (and near/mid's own bands can be
    # narrow relative to deviation_radius -- see hotspot_region_points'
    # own "falling back to unbuffered" warning), so the realized member
    # population can plausibly drift from the intended ratio -- this is a
    # real empirical check, not a tautology. Pooled across every sector
    # and several full periodic cycles (not per-sector -- the intended
    # ratio is the same for every sector regardless of which one is
    # currently in a given state, so pooling just gives more samples).
    if hasattr(engine.mobility, "demand_states"):
        # Same config.yaml-selected submode as `engine` itself (2026-10-11
        # fix -- was always hardcoded to "hotspot_demand-markov"), not just
        # whether it's a hotspot_demand family at all (the hasattr check
        # above already gates that).
        val_engine = build_debug_engine(cluster_mobility_mode=engine.mobility.cluster_mobility_mode)
        val_mobility = val_engine.mobility
        num_states = len(val_mobility.demand_states)
        num_cycles = 3
        num_replay_intervals = val_mobility.hotspot_hold_steps * num_states * num_cycles

        cell_to_region = val_engine.regional_info_provider.cell_to_region
        region_names = OccupancyEstimator.region_names
        num_own_regions = len(region_names)
        counts = np.zeros((num_states, num_own_regions))
        for interval in range(num_replay_intervals):
            state_idx_per_sector = val_mobility._state_idx.copy()
            xy = val_mobility.ut_loc[:, :2].detach().cpu().numpy()
            sector_idx, cell_idx = topo.assign_to_sector_rhombus_grid(xy)
            valid = sector_idx >= 0
            region_idx = cell_to_region[cell_idx[valid]]
            state_idx_per_ue = state_idx_per_sector[sector_idx[valid]]
            np.add.at(counts, (state_idx_per_ue, region_idx), 1.0)
            if interval < num_replay_intervals - 1:
                val_mobility.step()

        actual_frac = counts / counts.sum(axis=1, keepdims=True)
        intended_frac = val_mobility.demand_states / val_mobility.demand_states.sum(axis=1, keepdims=True)

        print(f"  [region occupancy validation] pooled over {num_cycles} periodic cycles "
             f"({num_replay_intervals} intervals, {int(counts.sum()):d} UE-interval observations)")
        fig, axes = plt.subplots(1, num_states, figsize=(5.5 * num_states, 5))
        axes = np.atleast_1d(axes)
        x = np.arange(num_own_regions)
        width = 0.35
        for s, ax in enumerate(axes):
            ax.bar(x - width / 2, intended_frac[s], width, label="intended", color="tab:gray")
            ax.bar(x + width / 2, actual_frac[s], width, label="actual", color="tab:purple")
            ax.set_xticks(x); ax.set_xticklabels(region_names)
            ax.set_ylim(0, 1)
            ax.set_ylabel("fraction of UEs")
            state_label = ",".join(str(v) for v in val_mobility.demand_states[s])
            ax.set_title(f"state {s}: [{state_label}]", fontsize=10)
            ax.legend(fontsize=8)
            print(f"    state {s} [{state_label}]: intended={np.round(intended_frac[s], 3).tolist()} "
                 f"actual={np.round(actual_frac[s], 3).tolist()}")
        fig.suptitle("Region occupancy: intended (demand_states) vs. actual (real simulated UEs), "
                    "pooled over all sectors", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "region_occupancy_validation.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- real-UE tilt sensitivity (2026-10-11) -- phase 2's own
        # tilt_sensitivity_single_sector.png characterizes a UNIFORM
        # virtual grid; this instead measures the engine's own REAL UEs
        # (same cluster_mobility_mode as `engine`, dedicated engine so the
        # shared one isn't mutated -- _force_synchronized_demand_state
        # overwrites mobility state in place), one demand state at a time,
        # local sweep (only target_sector's own tilt varies). Pooled over
        # several trials per state (see compute_real_ue_tilt_sensitivity's
        # own docstring for why) rather than one snapshot.
        target_sector = 0
        num_trials = 5
        tilt_engine = build_debug_engine(cluster_mobility_mode=engine.mobility.cluster_mobility_mode)
        num_states = len(val_mobility.demand_states)
        state_labels = []
        real_results = []
        for s in range(num_states):
            row = val_mobility.demand_states[s]
            heavy_region = region_names[int(np.argmax(row))]
            state_labels.append(f"[{','.join(str(v) for v in row)}]\n({heavy_region}-heavy)")
            real_results.append(compute_real_ue_tilt_sensitivity(
                tilt_engine, s, target_sector=target_sector, num_trials=num_trials))
            n_total = {r: int(real_results[-1][f"{r}_n"].sum()) for r in region_names}
            print(f"  [real-UE tilt sensitivity] state {s} ({heavy_region}-heavy): "
                 f"pooled UE-trial observations per region={n_total}")

        fig, axes = plt.subplots(num_states, 3, figsize=(16, 4.5 * num_states), squeeze=False)
        for s in range(num_states):
            res = real_results[s]
            for col, (metric, ylabel) in enumerate((
                    ("coverage", "coverage fraction"), ("rsrp_mean", "mean RSRP (dBm)"),
                    ("sinr_mean", "mean SINR (dB)"))):
                ax = axes[s, col]
                for r in region_names:
                    ax.plot(res["tilt_values"], res[f"{r}_{metric}"], "-o", color=HOTSPOT_REGION_COLORS[r], label=r)
                ax.plot(res["tilt_values"], res[f"whole_{metric}"], "--s", color="black", label="whole area")
                ax.set_xlabel("tilt (deg)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"state {s} {state_labels[s].splitlines()[0]}: {metric}", fontsize=9)
                if col == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"Real-UE tilt sensitivity (sector {target_sector}'s own local sweep), "
                    f"by demand state -- pooled over {num_trials} trials", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "real_ue_tilt_sensitivity.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- real-UE vs. virtual-grid comparison -- same local sweep,
        # same target_sector, computed via compute_single_sector_tilt_
        # perturbation (phase 2's own machinery) + the SAME per-region
        # pooling tilt_sensitivity_single_sector_figure uses internally.
        topo = tilt_engine.topology
        tilt_values = tilt_engine.downtilt_sweep_deg
        own_mask_geo = topo.hotspot_region_membership(target_sector, tilt_engine.grid_points[target_sector])
        pert = compute_single_sector_tilt_perturbation(
            target_sector, tilt_values, other_sectors_tilt_deg=SINGLE_SECTOR_NOMINAL_TILT_DEG,
            reference_tilt_deg=SINGLE_SECTOR_NOMINAL_TILT_DEG, engine=tilt_engine)
        own_slice = slice(target_sector * tilt_engine.num_grid_cells, (target_sector + 1) * tilt_engine.num_grid_cells)
        virtual_coverage, virtual_rsrp, virtual_sinr = {}, {}, {}
        for r in region_names:
            rmask = own_mask_geo[r]
            sinr_r = pert["sinr_db"][:, own_slice][:, rmask]
            virtual_coverage[r] = (sinr_r > tilt_engine.coverage_threshold_db).mean(axis=1)
            virtual_rsrp[r] = pert["rsrp_dbm"][:, own_slice][:, rmask].mean(axis=1)
            virtual_sinr[r] = sinr_r.mean(axis=1)
        # "whole": every one of target_sector's own virtual grid cells,
        # not restricted to any one region -- same convention as "whole"
        # above/real_ue_tilt_sensitivity.png's own addition.
        sinr_whole = pert["sinr_db"][:, own_slice]
        virtual_coverage["whole"] = (sinr_whole > tilt_engine.coverage_threshold_db).mean(axis=1)
        virtual_rsrp["whole"] = pert["rsrp_dbm"][:, own_slice].mean(axis=1)
        virtual_sinr["whole"] = sinr_whole.mean(axis=1)

        fig, axes = plt.subplots(num_states, 3, figsize=(16, 4.5 * num_states), squeeze=False)
        for s in range(num_states):
            res = real_results[s]
            for col, (metric, virtual_vals, ylabel) in enumerate((
                    ("coverage", virtual_coverage, "coverage fraction"),
                    ("rsrp_mean", virtual_rsrp, "mean RSRP (dBm)"),
                    ("sinr_mean", virtual_sinr, "mean SINR (dB)"))):
                ax = axes[s, col]
                for r in region_names:
                    ax.plot(res["tilt_values"], res[f"{r}_{metric}"], "-o", color=HOTSPOT_REGION_COLORS[r],
                           label=f"{r} (real UE)")
                    ax.plot(tilt_values, virtual_vals[r], "--s", color=HOTSPOT_REGION_COLORS[r],
                           alpha=0.55, label=f"{r} (virtual grid)")
                ax.plot(res["tilt_values"], res[f"whole_{metric}"], "-o", color="black", label="whole area (real UE)")
                ax.plot(tilt_values, virtual_vals["whole"], "--s", color="black", alpha=0.55,
                       label="whole area (virtual grid)")
                ax.set_xlabel("tilt (deg)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"state {s} {state_labels[s].splitlines()[0]}: {metric}", fontsize=9)
                if col == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"Real UE (solid, demand-state-specific) vs. virtual grid (dashed, uniform) "
                    f"tilt sensitivity, sector {target_sector}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "real_ue_vs_virtual_grid_tilt_sensitivity.png"),
                   dpi=150, bbox_inches="tight")
        plt.close(fig)

    if run_neighbor_joint_sweep:
        _neighbor_joint_sweep_figures(engine, os.path.join(out_dir, "neighbor_joint_sweep"))

    if run_neighbor_joint_sweep_grouped:
        _neighbor_joint_sweep_grouped_figures(engine, os.path.join(out_dir, "neighbor_joint_sweep_grouped"))

    print(f"Phase 3 done -- saved to {out_dir}/")

# =============================================================
# PHASE 3
# =============================================================

def _virtual_grid_state(ground_truth_engine: SimulationEngine, xy: np.ndarray = None, all_indoor: bool = False):
    """Places one virtual UE at every point in `xy` (default: every
    (sector, grid-cell) on ground_truth_engine's own grid_points -- cell
    CENTERS) and returns the resulting LargeScaleState. Shared by
    compute_ground_truth_grid, compute_hotspot_ground_truth, and
    compute_ground_truth_grid_fraction -- each then resolves POWER
    differently (one per-sector tilt vector vs. a tilt-value sweep).

    :param xy: [M, 2] world points to override the default cell-center set
        -- e.g. compute_ground_truth_grid_fraction's own within-cell
        sub-points, used to build a continuous per-cell coverage fraction
        instead of a single center-point binary decision.
    :param all_indoor: False (default) -- every virtual UE is OUTDOOR
        (in_state=False), REGARDLESS of the scenario's own derived
        anchor_ue_ratio (this function never reads SimulationEngine's
        derivation at all) -- the normal diagnostic convention, showing the best-case
        reachable radio environment. True forces every virtual UE INDOOR
        instead (in_state=True) -- per Sionna's own set_topology (los is
        ANDed with ~in_state), this also forces NLOS regardless of
        force_los, plus whatever O2I penetration-loss model the channel
        model applies for non-RMa scenarios (2026-10-08, added to test an
        all-indoor UMa condition).

    reset_topology() first: the channel model's own LSP spatial-
    correlation buffers are plain instance state, not released just by a
    fresh set_topology() call with a same-shaped or differently-shaped
    virtual-UE batch -- calling this function repeatedly (e.g. once per
    sub-point chunk) without reset_topology() first either raises ("Cannot
    change shape of '_ut_loc'...") on a batch-size change or silently
    accumulates GPU memory across calls even at a CONSTANT batch size.
    """
    topo = ground_truth_engine.topology
    if xy is None:
        xy = ground_truth_engine.grid_points.reshape(-1, 2)
    num_virtual = xy.shape[0]
    z = np.full((num_virtual, 1), ground_truth_engine.ut_height)
    virtual_loc = torch.tensor(np.concatenate([xy, z], axis=1),
                               dtype=topo.bs_loc.dtype, device=topo.bs_loc.device).unsqueeze(0)
    bs_virtual_loc = topo.mirror_bs_loc(virtual_loc)
    zeros = torch.zeros(1, num_virtual, 3, dtype=topo.bs_loc.dtype, device=topo.bs_loc.device)
    in_state = torch.full((1, num_virtual), all_indoor, dtype=torch.bool, device=topo.bs_loc.device)
    ground_truth_engine.channel_model.reset_topology()
    ground_truth_engine.channel_model.set_topology(virtual_loc, topo.bs_loc, zeros, topo.bs_orientations,
                                                   zeros, in_state, ground_truth_engine.force_los, bs_virtual_loc)
    return ground_truth_engine.large_scale_channel.generate_state()


def compute_hotspot_ground_truth(tilt_values: np.ndarray, batch_size: int = 800, all_indoor: bool = False) -> dict:
    """Ground-truth radio + hotspot-region geometry (near/mid/far, see
    CellularTopology.hotspot_region_membership) at EVERY sector's own grid
    cell, swept across every tilt in `tilt_values` -- one virtual-UE
    population, one dedicated engine (see compute_ground_truth_grid's own
    caveat: an independent shadow-fading realization, not a sample-exact
    replay of any real interval), one channel state reused for every
    candidate tilt (large-scale fading is tilt-independent -- see
    KpiManager.resolve_power_at_tilt/compute_tilt_sector_ue_rx_power).

    Builds its OWN dedicated engine internally (unlike
    compute_ground_truth_grid, which is called twice per phase-4 run and so
    takes a pre-built engine to avoid rebuilding it) -- phase 3 calls this
    exactly once.

    :output: dict, all arrays over the flat [num_bs*num_grid_cells]
        virtual-UE population unless noted --
        sector_tag: int, each cell's own (geometric) sector.
        eta, near, mid, far: from hotspot_region_membership -- every
            `inside` cell gets exactly one of these (no excluded hole).
        tilt_values: [T] echoed back.
        rsrp_dbm, sinr_db: [T, N] serving-cell KPIs per tilt.
        power_db_full: [T, num_bs, N] every TX sector's own received power
            at every virtual UE, per tilt -- needed for neighbor-edge
            coupling (check 5), which needs the non-serving sectors' power
            too, not just the winning one.
    """
    gt_engine = build_debug_engine()
    topo = gt_engine.topology
    num_bs, num_grid_cells = gt_engine.num_bs, gt_engine.num_grid_cells
    num_virtual = num_bs * num_grid_cells

    # Geometric hotspot-region tags -- tilt-independent, computed once.
    sector_tag = np.repeat(np.arange(num_bs), num_grid_cells)
    eta = np.zeros(num_virtual)
    near = np.zeros(num_virtual, dtype=bool)
    mid = np.zeros(num_virtual, dtype=bool)
    far = np.zeros(num_virtual, dtype=bool)
    for s in range(num_bs):
        m = topo.hotspot_region_membership(s, gt_engine.grid_points[s])
        sl = slice(s * num_grid_cells, (s + 1) * num_grid_cells)
        eta[sl], near[sl], mid[sl], far[sl] = m["eta"], m["near"], m["mid"], m["far"]

    # Radio, swept across the real (Oracle/DRL) action space -- CHUNKED
    # (batch_size virtual UEs per call, reset_topology() before each -- see
    # _virtual_grid_state's own docstring) since the channel model's LSP
    # spatial-correlation matrix is O(N^2) in the virtual-UE count: at a
    # fine grid (e.g. 20m cells -> ~4700 virtual UEs network-wide) a single
    # unchunked call can exceed available GPU memory even alongside modest
    # other GPU usage, whereas per-chunk correlation matrices are tiny by
    # comparison. Cross-chunk shadow-fading correlation is lost, but
    # virtual UEs in different chunks are typically many decorrelation-
    # distances apart anyway (see spatial_grid_cell_size_m's own UMa
    # decorrelation-distance comment), so this doesn't meaningfully change
    # the result -- same chunking pattern as
    # compute_ground_truth_grid_fraction above.
    xy_all = gt_engine.grid_points.reshape(-1, 2)
    tilt_values = np.asarray(tilt_values)
    num_tilts = len(tilt_values)
    rsrp_dbm = np.zeros((num_tilts, num_virtual))
    sinr_db = np.zeros((num_tilts, num_virtual))
    power_db_full = np.zeros((num_tilts, num_bs, num_virtual))
    with torch.no_grad():
        for start in range(0, num_virtual, batch_size):
            end = min(start + batch_size, num_virtual)
            chunk_state = _virtual_grid_state(gt_engine, xy=xy_all[start:end], all_indoor=all_indoor)
            power_table_chunk = gt_engine.kpi_manager.compute_tilt_sector_ue_rx_power(
                chunk_state, tilt_values)  # [T,1,num_bs,chunk_N]
            for t in range(num_tilts):
                kpis = gt_engine.kpi_manager.compute_ue_kpis(power_table_chunk[t])
                rsrp_dbm[t, start:end] = kpis["rsrp_dbm"][0]
                sinr_db[t, start:end] = kpis["sinr_db"][0]
                power_db_full[t, :, start:end] = 10.0 * np.log10(power_table_chunk[t, 0])
            del chunk_state, power_table_chunk
            torch.cuda.empty_cache()

    return dict(sector_tag=sector_tag, eta=eta, near=near, mid=mid, far=far,
               tilt_values=tilt_values, rsrp_dbm=rsrp_dbm, sinr_db=sinr_db, power_db_full=power_db_full)


def compute_single_sector_tilt_perturbation(target_sector: int, tilt_values: np.ndarray,
                                            other_sectors_tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                            reference_tilt_deg: float = None, batch_size: int = 800,
                                            engine: SimulationEngine = None, all_indoor: bool = False) -> dict:
    """Holds EVERY sector except `target_sector` fixed at `other_sectors_tilt_deg`
    and sweeps ONLY target_sector's own tilt across `tilt_values` -- isolates
    one sector's own tilt-change effect (including how it leaks into
    neighboring sectors' coverage) from compute_hotspot_ground_truth's
    UNIFORM/global sweep above, where every sector changes together and a
    single sector's own contribution can't be told apart from the rest of
    the network's.

    One dedicated engine + one channel state (large-scale fading is
    tilt-independent, see compute_ground_truth_grid's own reasoning), swept
    via `resolve_power_at_tilt`'s per-sector tilt VECTOR (not
    compute_hotspot_ground_truth's uniform-tilt-per-row sweep).

    :param other_sectors_tilt_deg: fixed tilt for every sector EXCEPT
        target_sector, held constant across the whole sweep.
    :param reference_tilt_deg: which of target_sector's OWN swept tilt
        values is the zero-delta baseline (defaults to other_sectors_tilt_deg
        if omitted) -- independent of other_sectors_tilt_deg, so the "all
        yellow" reference column doesn't have to coincide with what the
        rest of the network is held at.
    :param engine: use THIS engine instead of building a fresh
        build_debug_engine() -- e.g. a caller with its own antenna-config-
        overridden engine, so the sweep actually reflects that config's own
        antenna.m/antenna_window instead of silently falling back to
        config.yaml's default (a real bug this param fixes, 2026-09-22 --
        every OTHER caller omits this and keeps the original
        build_debug_engine() behavior).
    :output: dict with tilt_values [T] echoed back; rsrp_dbm, sinr_db
        [T, N] (absolute, N = num_bs*num_grid_cells virtual UEs);
        rsrp_delta_dbm, sinr_delta_db [T, N] (relative to the
        reference_tilt_deg baseline row -- exactly zero everywhere at
        whichever tilt equals reference_tilt_deg).
    """
    gt_engine = engine if engine is not None else build_debug_engine()
    num_bs = gt_engine.num_bs
    tilt_values = np.asarray(tilt_values)
    num_tilts = tilt_values.size
    num_virtual = num_bs * gt_engine.num_grid_cells
    rsrp_dbm = np.zeros((num_tilts, num_virtual))
    sinr_db = np.zeros((num_tilts, num_virtual))
    # CHUNKED -- see compute_hotspot_ground_truth's own comment for why
    # (the channel model's LSP spatial-correlation matrix is O(N^2) in the
    # virtual-UE count).
    xy_all = gt_engine.grid_points.reshape(-1, 2)
    with torch.no_grad():
        for start in range(0, num_virtual, batch_size):
            end = min(start + batch_size, num_virtual)
            chunk_state = _virtual_grid_state(gt_engine, xy=xy_all[start:end], all_indoor=all_indoor)
            for t, tilt in enumerate(tilt_values):
                tilt_vec = np.full(num_bs, other_sectors_tilt_deg)
                tilt_vec[target_sector] = tilt
                power = gt_engine.kpi_manager.resolve_power_at_tilt(chunk_state, tilt_vec)[0]
                kpis = gt_engine.kpi_manager.compute_ue_kpis(power[None, :, :])
                rsrp_dbm[t, start:end], sinr_db[t, start:end] = kpis["rsrp_dbm"][0], kpis["sinr_db"][0]
            del chunk_state
            torch.cuda.empty_cache()

    reference_tilt_deg = other_sectors_tilt_deg if reference_tilt_deg is None else reference_tilt_deg
    reference_idx = int(np.argmin(np.abs(tilt_values - reference_tilt_deg)))
    return dict(tilt_values=tilt_values, rsrp_dbm=rsrp_dbm, sinr_db=sinr_db,
               rsrp_delta_dbm=rsrp_dbm - rsrp_dbm[reference_idx],
               sinr_delta_db=sinr_db - sinr_db[reference_idx],
               other_sectors_tilt_deg=other_sectors_tilt_deg, reference_tilt_deg=reference_tilt_deg)


def compute_real_ue_tilt_sensitivity(engine: SimulationEngine, state_idx: int, target_sector: int = 0,
                                     other_sectors_tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                     num_trials: int = 5) -> dict:
    """Real-UE analog of compute_single_sector_tilt_perturbation above --
    same LOCAL sweep (only target_sector's own tilt varies, every other
    sector fixed at other_sectors_tilt_deg), but measures the engine's own
    REAL (hotspot_demand-periodic, or whatever config.yaml selects) UEs
    instead of a uniform virtual grid, under ONE forced demand_states
    distribution (state_idx, via _force_synchronized_demand_state).

    Pooled over num_trials independent re-draws of that forced population
    (mask_seed=trial also advances the UeDropper's own torch.Generator
    statefully -- see _force_synchronized_demand_state's own docstring --
    so each trial is a genuinely different concrete cluster-placement
    realization of the SAME demand TYPE, not a repeat) -- characterizes the
    demand state's own TYPICAL real-UE experience, not one lucky/unlucky
    draw. non_anchor_fraction=0.0 (every UE counts) since this measures raw
    per-UE RSRP/SINR/coverage directly, never routes through the anchor-
    gated production state/reward pipeline.

    Region membership is the REAL geometric region (same
    CellularTopology.hotspot_region_membership-derived cell_to_region
    region_occupancy_validation.png uses), not the cluster's own INTENDED
    region -- members random-walk away from their cluster's own target
    region, so this is "where these UEs actually ended up", matching that
    figure's own "actual, not intended" convention.

    One fixed channel/position snapshot per trial, tilt swept via
    resolve_power_at_tilt (same pattern compute_single_sector_tilt_
    perturbation/_virtual_grid_state use) -- isolates the tilt effect from
    position noise WITHIN a trial; position noise across trials is exactly
    the thing num_trials pools over.

    :output: dict with tilt_values [T]; per region name ("near"/"mid"/
        "far"/"whole" -- "whole" is every target_sector UE regardless of
        region, 2026-10-12 addition): "{region}_coverage"/"{region}_sinr_mean"/
        "{region}_rsrp_mean" [T] (pooled over every target_sector UE across
        every trial, at each tilt; NaN where zero UEs landed in that region
        at that tilt across every trial) and "{region}_n" [T] (pooled
        UE-trial observation count).
    """
    tilt_values = engine.downtilt_sweep_deg
    num_bs = engine.num_bs
    cell_to_region = engine.regional_info_provider.cell_to_region
    region_names = OccupancyEstimator.region_names
    result_groups = region_names + ("whole",)
    sinr_by_region = {r: [[] for _ in tilt_values] for r in result_groups}
    rsrp_by_region = {r: [[] for _ in tilt_values] for r in result_groups}

    for trial in range(num_trials):
        _force_synchronized_demand_state(engine, state_idx, non_anchor_fraction=0.0, mask_seed=trial)
        state = engine.step_environment(True)  # True: no mobility.step() needed, about to be overwritten next trial
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
        own_mask = sector_idx == target_sector
        region_idx_own = cell_to_region[cell_idx[own_mask]]

        for t, tilt in enumerate(tilt_values):
            tilt_vec = np.full(num_bs, other_sectors_tilt_deg)
            tilt_vec[target_sector] = tilt
            power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
            kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
            sinr_db_own = kpis["sinr_db"][0][own_mask]
            rsrp_dbm_own = kpis["rsrp_dbm"][0][own_mask]
            sinr_by_region["whole"][t].extend(sinr_db_own.tolist())
            rsrp_by_region["whole"][t].extend(rsrp_dbm_own.tolist())
            for ri, r in enumerate(region_names):
                rmask = region_idx_own == ri
                sinr_by_region[r][t].extend(sinr_db_own[rmask].tolist())
                rsrp_by_region[r][t].extend(rsrp_dbm_own[rmask].tolist())

    result = dict(tilt_values=tilt_values)
    for r in result_groups:
        result[f"{r}_coverage"] = np.array([
            float((np.asarray(vals) > engine.coverage_threshold_db).mean()) if len(vals) else np.nan
            for vals in sinr_by_region[r]])
        result[f"{r}_sinr_mean"] = np.array([
            float(np.mean(vals)) if len(vals) else np.nan for vals in sinr_by_region[r]])
        result[f"{r}_rsrp_mean"] = np.array([
            float(np.mean(vals)) if len(vals) else np.nan for vals in rsrp_by_region[r]])
        result[f"{r}_n"] = np.array([len(vals) for vals in sinr_by_region[r]])
    return result


def compute_real_ue_tilt_sensitivity_anchor_compare(engine: SimulationEngine, state_idx: int,
                                                     target_sector: int = 0,
                                                     other_sectors_tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                                     num_trials: int = 5,
                                                     non_anchor_fraction: float = 0.8,
                                                     anchor_override_fn=None) -> dict:
    """Phase 4's own tilt-decision-accuracy question: does the anchor-only
    coverage-vs-tilt curve pick a different "best tilt" than the
    ground-truth curve would? Same LOCAL single-sector sweep methodology
    as compute_real_ue_tilt_sensitivity (target_sector's own tilt varies,
    every other sector fixed at other_sectors_tilt_deg, resolve_power_at_tilt
    per tilt value, num_trials independent re-draws pooled), but every
    trial uses _force_synchronized_demand_state's own non_anchor_fraction
    (phase 4's convention, default 0.8 -- NOT 0.0) so engine.location_
    reporting_mask is a genuine ~80% non-anchor split, and BOTH
    ground_truth (every real target_sector UE) and anchor_only (the SAME
    draw's own anchor subset only) are pooled from that ONE population per
    trial -- paired, not two independent re-draws, matching how
    _situations_for pairs them elsewhere in this phase.

    :output: dict with tilt_values [T]; ground_truth, anchor_only -- each a
        dict with "{region}_coverage"/"{region}_sinr_mean"/
        "{region}_rsrp_mean" [T] and "{region}_n" [T] for region in
        region_names + ("whole",), same naming convention as
        compute_real_ue_tilt_sensitivity's own output.
    """
    tilt_values = engine.downtilt_sweep_deg
    num_bs = engine.num_bs
    cell_to_region = engine.regional_info_provider.cell_to_region
    region_names = OccupancyEstimator.region_names
    result_groups = region_names + ("whole",)
    situations = ("ground_truth", "anchor_only")
    sinr_by = {sit: {r: [[] for _ in tilt_values] for r in result_groups} for sit in situations}
    rsrp_by = {sit: {r: [[] for _ in tilt_values] for r in result_groups} for sit in situations}

    for trial in range(num_trials):
        _force_synchronized_demand_state(engine, state_idx, non_anchor_fraction=non_anchor_fraction,
                                         mask_seed=trial)
        if anchor_override_fn is not None:
            anchor_override_fn(engine, state_idx, trial)
        state = engine.step_environment(True)  # True: no mobility.step() needed, about to be overwritten next trial
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
        own_mask = sector_idx == target_sector
        is_anchor_own = engine.location_reporting_mask[own_mask]
        region_idx_own = cell_to_region[cell_idx[own_mask]]

        for t, tilt in enumerate(tilt_values):
            tilt_vec = np.full(num_bs, other_sectors_tilt_deg)
            tilt_vec[target_sector] = tilt
            power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
            kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
            sinr_db_own = kpis["sinr_db"][0][own_mask]
            rsrp_dbm_own = kpis["rsrp_dbm"][0][own_mask]
            for sit, mask in (("ground_truth", slice(None)), ("anchor_only", is_anchor_own)):
                sinr_by[sit]["whole"][t].extend(sinr_db_own[mask].tolist())
                rsrp_by[sit]["whole"][t].extend(rsrp_dbm_own[mask].tolist())
                for ri, r in enumerate(region_names):
                    rmask = region_idx_own[mask] == ri
                    sinr_by[sit][r][t].extend(sinr_db_own[mask][rmask].tolist())
                    rsrp_by[sit][r][t].extend(rsrp_dbm_own[mask][rmask].tolist())

    result = dict(tilt_values=tilt_values)
    for sit in situations:
        result[sit] = {}
        for r in result_groups:
            result[sit][f"{r}_coverage"] = np.array([
                float((np.asarray(vals) > engine.coverage_threshold_db).mean()) if len(vals) else np.nan
                for vals in sinr_by[sit][r]])
            result[sit][f"{r}_sinr_mean"] = np.array([
                float(np.mean(vals)) if len(vals) else np.nan for vals in sinr_by[sit][r]])
            result[sit][f"{r}_rsrp_mean"] = np.array([
                float(np.mean(vals)) if len(vals) else np.nan for vals in rsrp_by[sit][r]])
            result[sit][f"{r}_n"] = np.array([len(vals) for vals in sinr_by[sit][r]])
    return result


def _tilt_decision_accuracy_figure(out_dir: str, engine: SimulationEngine, demand_states: np.ndarray,
                                   state_labels: list, target_sector: int = 0,
                                   num_trials: int = 5, non_anchor_fraction: float = 0.8,
                                   anchor_override_fn=None) -> list:
    """The tilt-decision-accuracy deliverable: for every demand state,
    overlay ground_truth vs anchor_only coverage-vs-tilt curves (one panel
    per region + whole), mark each curve's OWN argmax tilt, and -- the
    actual decision-relevant number -- evaluate the GROUND-TRUTH curve at
    BOTH tilt choices (own optimum vs. anchor-only's optimum), since the
    real-world consequence of picking the wrong tilt is always governed by
    ground truth, never by what anchor-only merely believed. Marks both
    points + an arrow on the ground-truth curve itself (not just a console
    number) showing that coverage-loss gap, per [[feedback_plot_verification_style]]
    convention. Saves tilt_decision_accuracy.png and a plain-text summary
    table (tilt_decision_accuracy_summary.txt) of (state, region,
    gt_optimal_tilt_deg, anchor_optimal_tilt_deg, coverage_loss).

    :output: list of (state_label, region, gt_tilt, anchor_tilt, loss) rows
        -- also printed and written to the summary file, returned too so
        callers can fold it into their own console summary.
    """
    region_names = OccupancyEstimator.region_names
    columns = region_names + ("whole",)
    num_states = demand_states.shape[0]
    num_cols = len(columns)
    fig, axes = plt.subplots(num_states, num_cols, figsize=(5 * num_cols, 4 * num_states), squeeze=False)
    summary_rows = []

    for row, state_idx in enumerate(range(num_states)):
        res = compute_real_ue_tilt_sensitivity_anchor_compare(
            engine, state_idx, target_sector=target_sector, num_trials=num_trials,
            non_anchor_fraction=non_anchor_fraction, anchor_override_fn=anchor_override_fn)
        tilt_values = res["tilt_values"]
        for col, r in enumerate(columns):
            ax = axes[row, col]
            gt_cov = res["ground_truth"][f"{r}_coverage"]
            anc_cov = res["anchor_only"][f"{r}_coverage"]
            ax.plot(tilt_values, gt_cov, "-o", color=_SITUATION_COLORS["ground_truth"], label="ground truth")
            ax.plot(tilt_values, anc_cov, "-s", color=_SITUATION_COLORS["anchor_only"], label="anchor-only")
            ax.set_ylim(0, 1)
            ax.set_xlim(tilt_values.min(), tilt_values.max())
            if row == 0:
                ax.set_title(r, fontsize=11)
            if col == 0:
                ax.set_ylabel(state_labels[row], fontsize=9)
            if row == num_states - 1:
                ax.set_xlabel("tilt (deg)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

            if np.isfinite(gt_cov).any() and np.isfinite(anc_cov).any():
                gt_opt_idx = int(np.nanargmax(gt_cov))
                anc_opt_idx = int(np.nanargmax(anc_cov))
                gt_opt_tilt, anc_opt_tilt = tilt_values[gt_opt_idx], tilt_values[anc_opt_idx]
                gt_cov_at_gt_opt = gt_cov[gt_opt_idx]
                gt_cov_at_anc_opt = gt_cov[anc_opt_idx]  # real consequence of trusting anchor-only's pick
                loss = gt_cov_at_gt_opt - gt_cov_at_anc_opt
                ax.plot(gt_opt_tilt, gt_cov_at_gt_opt, "*", color=_SITUATION_COLORS["ground_truth"],
                       markersize=16, markeredgecolor="black", zorder=5)
                if anc_opt_idx != gt_opt_idx:
                    ax.plot(anc_opt_tilt, gt_cov_at_anc_opt, "*", color=_SITUATION_COLORS["anchor_only"],
                           markersize=16, markeredgecolor="black", zorder=5)
                    ax.annotate("", xy=(anc_opt_tilt, gt_cov_at_anc_opt), xytext=(gt_opt_tilt, gt_cov_at_gt_opt),
                               arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
                    ax.text(0.5, 0.05, f"loss={loss:.3f}", transform=ax.transAxes, ha="center", fontsize=8,
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax.text(0.5, 0.05, "same tilt", transform=ax.transAxes, ha="center", fontsize=8,
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                summary_rows.append((state_labels[row], r, float(gt_opt_tilt), float(anc_opt_tilt), float(loss)))

    fig.suptitle(f"Tilt-decision accuracy: ground-truth vs. anchor-only coverage-vs-tilt "
                f"(sector {target_sector}, {num_trials} trials, non_anchor_fraction={non_anchor_fraction})",
                fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tilt_decision_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(out_dir, "tilt_decision_accuracy_summary.txt")
    with open(summary_path, "w") as f:
        header = f"{'state':<28}{'region':<8}{'gt_opt_deg':<12}{'anchor_opt_deg':<16}{'coverage_loss':<14}"
        print(header, file=f)
        print("  [tilt decision accuracy]")
        print(f"  {header}")
        for state_label, r, gt_tilt, anc_tilt, loss in summary_rows:
            line = f"{state_label:<28}{r:<8}{gt_tilt:<12.0f}{anc_tilt:<16.0f}{loss:<14.4f}"
            print(line, file=f)
            print(f"  {line}")
    print(f"  [tilt decision accuracy] done -- saved to {out_dir}/tilt_decision_accuracy.png, "
         f"{summary_path}")
    return summary_rows


# L="near-heavy"/H="edge(far)-heavy" demand-state ROW INDICES into the
# committed config.yaml's own demand_states table (index 0=mid-heavy,
# 1=near-heavy, 2=far-heavy -- mid-heavy deliberately unused here, per
# explicit request). NEIGHBOR_JOINT_CASES below name sector 0's own state,
# N1's (topology.neighbor_ids[0][0]), and N2's (...[1]) independently --
# see phase3_raw_ue_behavior's own neighbor-joint-sweep docstring for the
# full 6-case table and each one's expected-behavior rationale.
_NEAR_HEAVY_STATE_IDX, _FAR_HEAVY_STATE_IDX = 1, 2
NEIGHBOR_JOINT_CASES = {
    # name -> (sector_0, n1, n2), each "L" or "H"
    "case1_LLL": ("L", "L", "L"),
    "case2_HLL": ("H", "L", "L"),
    "case3_LHL": ("L", "H", "L"),
    "case4_HHL": ("H", "H", "L"),
    "case5_HHH": ("H", "H", "H"),
    "case6_LHH": ("L", "H", "H"),
}


def compute_neighbor_joint_tilt_sweep(engine: SimulationEngine, target_sector: int, n1: int, n2: int,
                                      own_state: str, n1_state: str, n2_state: str,
                                      default_state_idx: int = 0,
                                      other_sectors_tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG) -> dict:
    """ONE population snapshot (target_sector/n1/n2 pinned to their own
    "L"=near-heavy/"H"=far-heavy demand state via _force_specific_demand_states,
    every OTHER sector at default_state_idx -- mid-heavy), then an
    EXHAUSTIVE 5x5x5=125-combo sweep of (target_sector, n1, n2)'s own tilt
    -- every other sector fixed at other_sectors_tilt_deg throughout. No
    trial-averaging layer (explicit request -- the 125-combo grid itself
    IS the whole experiment, evaluated once against one real population,
    not pooled over repeated re-draws).

    GLOBAL coverage/covered-count (every real UE in the whole network, not
    just target_sector/n1/n2's own) at each of the 125 tilt combos --
    network-wide effect of this own-sector/2-real-neighbor joint tilt
    decision, not a local one.

    :output: dict with tilt_values [5]; global_coverage, global_covered_count
        [5, 5, 5] indexed [own_tilt_idx, n1_tilt_idx, n2_tilt_idx].
    """
    state_map = {}
    for sector, state in ((target_sector, own_state), (n1, n1_state), (n2, n2_state)):
        state_map[sector] = _FAR_HEAVY_STATE_IDX if state == "H" else _NEAR_HEAVY_STATE_IDX
    _force_specific_demand_states(engine, state_map, default_state_idx, non_anchor_fraction=0.0)

    state = engine.step_environment(True)  # True: single fixed snapshot, no mobility.step() needed
    tilt_values = engine.downtilt_sweep_deg
    num_bs = engine.num_bs
    num_t = len(tilt_values)
    global_coverage = np.zeros((num_t, num_t, num_t))
    global_covered_count = np.zeros((num_t, num_t, num_t))

    base_tilt_vec = np.full(num_bs, other_sectors_tilt_deg)
    for oi, own_tilt in enumerate(tilt_values):
        for ni, n1_tilt in enumerate(tilt_values):
            for nj, n2_tilt in enumerate(tilt_values):
                tilt_vec = base_tilt_vec.copy()
                tilt_vec[target_sector] = own_tilt
                tilt_vec[n1] = n1_tilt
                tilt_vec[n2] = n2_tilt
                power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
                kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
                covered = kpis["sinr_db"][0] > engine.coverage_threshold_db
                global_covered_count[oi, ni, nj] = float(covered.sum())
                global_coverage[oi, ni, nj] = float(covered.mean())

    return dict(tilt_values=tilt_values, global_coverage=global_coverage,
               global_covered_count=global_covered_count)


def compute_neighbor_joint_tilt_sweep_grouped(engine: SimulationEngine, target_sector: int, n1: int, n2: int,
                                              own_state: str, n1_state: str, n2_state: str,
                                              own_group: np.ndarray, n1_group: np.ndarray,
                                              n2_group: np.ndarray) -> dict:
    """GROUPED variant of compute_neighbor_joint_tilt_sweep (2026-10-12,
    explicit request) -- addresses that function's own global-coverage
    DILUTION (target_sector/n1/n2 are only ~15% of the network; the other
    85% sits at one fixed background state/tilt, barely responding to this
    sector's own tilt choice since tilt effects are local).

    Here, EVERY sector in the network is swept, not just 3: target_sector's
    own "phase group" (every sector sharing its NATURAL, pre-override
    mobility._state_idx -- i.e. every sector that stays permanently
    synchronized with it under periodic cycling, per the graph-coloring fix
    in HotspotDemandMobility.__init__) is forced to own_state and given
    own_tilt; n1's phase group to n1_state/n1_tilt; n2's to n2_state/n2_tilt.
    Since target_sector/n1/n2 always get DIFFERENT starting phases (that's
    what the graph coloring guarantees), these 3 groups are disjoint, and
    since there are exactly num_states=3 possible phases total, they
    PARTITION the entire network -- no "other" category left over, no
    sector sits out the sweep.

    :param own_group, n1_group, n2_group: [sector indices] -- CALLER-OWNED,
        computed ONCE from mobility._state_idx BEFORE the first case's own
        _force_specific_demand_states call (2026-10-12 fix: this function
        used to re-derive the groups from mobility._state_idx itself every
        call, but that attribute IS what _force_specific_demand_states
        overwrites -- after case 1 forces every sector in all 3 groups to
        state indices that can coincide, e.g. case1_LLL sets ALL THREE
        groups to the SAME near-heavy index, every later case's own
        re-derivation collapsed into one 21-sector group instead of 3
        disjoint ones).

    Same ONE-population-snapshot, no-trial-averaging, 5x5x5=125-combo
    methodology as the ungrouped version otherwise.

    :output: dict with tilt_values [5]; global_coverage, global_covered_count
        [5, 5, 5] indexed [own_tilt_idx, n1_tilt_idx, n2_tilt_idx]; group_sizes
        {"own": int, "n1": int, "n2": int} (sector counts per group, for the
        caller's own sanity-check/labeling).
    """
    assert len(own_group) + len(n1_group) + len(n2_group) == engine.num_bs, (
        "own/n1/n2 phase groups should partition the whole network -- got "
        f"{len(own_group)}+{len(n1_group)}+{len(n2_group)} != {engine.num_bs}")

    own_idx = _FAR_HEAVY_STATE_IDX if own_state == "H" else _NEAR_HEAVY_STATE_IDX
    n1_idx = _FAR_HEAVY_STATE_IDX if n1_state == "H" else _NEAR_HEAVY_STATE_IDX
    n2_idx = _FAR_HEAVY_STATE_IDX if n2_state == "H" else _NEAR_HEAVY_STATE_IDX
    state_map = {}
    for group, idx in ((own_group, own_idx), (n1_group, n1_idx), (n2_group, n2_idx)):
        for s in group:
            state_map[int(s)] = idx
    _force_specific_demand_states(engine, state_map, default_state_idx=0, non_anchor_fraction=0.0)

    state = engine.step_environment(True)  # True: single fixed snapshot, no mobility.step() needed
    tilt_values = engine.downtilt_sweep_deg
    num_bs = engine.num_bs
    num_t = len(tilt_values)
    global_coverage = np.zeros((num_t, num_t, num_t))
    global_covered_count = np.zeros((num_t, num_t, num_t))

    for oi, own_tilt in enumerate(tilt_values):
        for ni, n1_tilt in enumerate(tilt_values):
            for nj, n2_tilt in enumerate(tilt_values):
                tilt_vec = np.zeros(num_bs)
                tilt_vec[own_group] = own_tilt
                tilt_vec[n1_group] = n1_tilt
                tilt_vec[n2_group] = n2_tilt
                power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
                kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
                covered = kpis["sinr_db"][0] > engine.coverage_threshold_db
                global_covered_count[oi, ni, nj] = float(covered.sum())
                global_coverage[oi, ni, nj] = float(covered.mean())

    return dict(tilt_values=tilt_values, global_coverage=global_coverage,
               global_covered_count=global_covered_count,
               group_sizes=dict(own=len(own_group), n1=len(n1_group), n2=len(n2_group)))


def _neighbor_joint_sweep_figures(engine: SimulationEngine, out_dir: str) -> None:
    """Runs compute_neighbor_joint_tilt_sweep for all 6 NEIGHBOR_JOINT_CASES
    (sector 0 + its 2 real neighbors), saving one figure per case: a 1x5
    row of heatmaps (one per N2 tilt value), each heatmap x=sector 0's own
    tilt, y=N1's own tilt, color=global coverage fraction.

    Case table (L=near-heavy, H=far(edge)-heavy, mid-heavy unused):
      1 LLL: higher downtilt should be safe/best
      2 HLL: sector 0 should prefer lower downtilt
      3 LHL: sector 0 should prefer higher downtilt to protect N1
      4 HHL: genuine competition between sector 0 and N1
      5 HHH: strong contention, best config determined empirically
      6 LHH: sector 0 should strongly favor higher downtilt
    """
    os.makedirs(out_dir, exist_ok=True)
    target_sector = 0
    nbr = engine.topology.neighbor_ids[target_sector]
    nbr = nbr[nbr >= 0]
    assert len(nbr) >= 2, f"sector {target_sector} needs >=2 real neighbors for this sweep, has {len(nbr)}"
    n1, n2 = int(nbr[0]), int(nbr[1])
    print(f"  [neighbor joint sweep] target_sector={target_sector} n1={n1} n2={n2}")

    for case_name, (own_s, n1_s, n2_s) in NEIGHBOR_JOINT_CASES.items():
        res = compute_neighbor_joint_tilt_sweep(engine, target_sector, n1, n2, own_s, n1_s, n2_s)
        tilt_values = res["tilt_values"]
        num_t = len(tilt_values)
        vmin, vmax = res["global_coverage"].min(), res["global_coverage"].max()

        fig, axes = plt.subplots(1, num_t, figsize=(4.2 * num_t, 4.5))
        for nj in range(num_t):
            ax = axes[nj]
            grid = res["global_coverage"][:, :, nj]  # [own, n1]
            im = ax.imshow(grid.T, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(range(num_t)); ax.set_xticklabels([f"{t:.0f}" for t in tilt_values])
            ax.set_yticks(range(num_t)); ax.set_yticklabels([f"{t:.0f}" for t in tilt_values])
            ax.set_xlabel(f"sector {target_sector} (own) tilt (deg)")
            if nj == 0:
                ax.set_ylabel(f"N1 (sector {n1}) tilt (deg)")
            ax.set_title(f"N2 (sector {n2})={tilt_values[nj]:.0f} deg", fontsize=10)
            for oi in range(num_t):
                for ni in range(num_t):
                    ax.text(oi, ni, f"{grid[oi, ni]:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if grid[oi, ni] < (vmin + vmax) / 2 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
        own_s, n1_s, n2_s = NEIGHBOR_JOINT_CASES[case_name]
        fig.suptitle(f"{case_name}: sector 0={own_s}, N1(sector {n1})={n1_s}, N2(sector {n2})={n2_s} -- "
                    f"global coverage fraction", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"neighbor_joint_sweep_{case_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    {case_name}: global coverage range=[{res['global_coverage'].min():.3f}, "
             f"{res['global_coverage'].max():.3f}]")
    print(f"  [neighbor joint sweep] done -- saved to {out_dir}/")


def _neighbor_joint_sweep_grouped_figures(engine: SimulationEngine, out_dir: str) -> None:
    """GROUPED variant of _neighbor_joint_sweep_figures -- see
    compute_neighbor_joint_tilt_sweep_grouped's own docstring for why
    (dilution fix: every sector in the network is swept via its own phase
    group, not just 3 isolated sectors). Same 6-case/1x5-heatmap-per-case
    layout and output convention otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_sector = 0
    nbr = engine.topology.neighbor_ids[target_sector]
    nbr = nbr[nbr >= 0]
    assert len(nbr) >= 2, f"sector {target_sector} needs >=2 real neighbors for this sweep, has {len(nbr)}"
    n1, n2 = int(nbr[0]), int(nbr[1])
    # Captured ONCE, before any case's own _force_specific_demand_states
    # call -- see compute_neighbor_joint_tilt_sweep_grouped's own docstring
    # for why this can't be re-derived from mobility._state_idx per case.
    base_state_idx = engine.mobility._state_idx.copy()
    own_group = np.flatnonzero(base_state_idx == base_state_idx[target_sector])
    n1_group = np.flatnonzero(base_state_idx == base_state_idx[n1])
    n2_group = np.flatnonzero(base_state_idx == base_state_idx[n2])
    print(f"  [neighbor joint sweep GROUPED] target_sector={target_sector} n1={n1} n2={n2} "
         f"group_sizes=own:{len(own_group)}/n1:{len(n1_group)}/n2:{len(n2_group)}")

    for case_name, (own_s, n1_s, n2_s) in NEIGHBOR_JOINT_CASES.items():
        res = compute_neighbor_joint_tilt_sweep_grouped(
            engine, target_sector, n1, n2, own_s, n1_s, n2_s, own_group, n1_group, n2_group)
        tilt_values = res["tilt_values"]
        num_t = len(tilt_values)
        vmin, vmax = res["global_coverage"].min(), res["global_coverage"].max()
        gs = res["group_sizes"]

        fig, axes = plt.subplots(1, num_t, figsize=(4.2 * num_t, 4.5))
        for nj in range(num_t):
            ax = axes[nj]
            grid = res["global_coverage"][:, :, nj]  # [own, n1]
            im = ax.imshow(grid.T, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(range(num_t)); ax.set_xticklabels([f"{t:.0f}" for t in tilt_values])
            ax.set_yticks(range(num_t)); ax.set_yticklabels([f"{t:.0f}" for t in tilt_values])
            ax.set_xlabel(f"own group ({gs['own']} sectors) tilt (deg)")
            if nj == 0:
                ax.set_ylabel(f"N1 group ({gs['n1']} sectors) tilt (deg)")
            ax.set_title(f"N2 group ({gs['n2']} sectors)={tilt_values[nj]:.0f} deg", fontsize=10)
            for oi in range(num_t):
                for ni in range(num_t):
                    ax.text(oi, ni, f"{grid[oi, ni]:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if grid[oi, ni] < (vmin + vmax) / 2 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"GROUPED {case_name}: own group={own_s}, N1 group={n1_s}, N2 group={n2_s} "
                    f"(sizes {gs['own']}/{gs['n1']}/{gs['n2']}, sector {target_sector}/N1={n1}/N2={n2}) -- "
                    f"global coverage fraction", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"neighbor_joint_sweep_grouped_{case_name}.png"),
                   dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    {case_name}: global coverage range=[{res['global_coverage'].min():.3f}, "
             f"{res['global_coverage'].max():.3f}]")
    print(f"  [neighbor joint sweep GROUPED] done -- saved to {out_dir}/")


def phase2_radio_environment_validation(engine: SimulationEngine, scenario: dict, out_dir: str,
                                        all_indoor: bool = False) -> None:
    """7 checks that the near/mid/far hotspot regions (see
    CellularTopology.hotspot_region_membership) actually carry physically
    meaningful, distinct radio characteristics -- the motivation for making
    the RL state hotspot-region-aware at all. Pure radio-environment
    geometry/RF characterization -- runs entirely on virtual grid points via
    its own dedicated engine(s), needs no real UE deployment at all (the
    `scenario` argument is accepted only for main()'s shared phase-runner
    signature and is never actually read below) -- so it can, and does, run
    BEFORE phase3_raw_ue_behavior's real mobility scenario exists.

    Checks 1-5 use SINGLE_SECTOR_NOMINAL_TILT_DEG as a FIXED baseline tilt --
    deliberately isolating geometry-driven regional effects from tilt as a
    confound. Checks 6-7
    sweep the FULL downtilt_sweep_deg (the real DRL/Oracle action space,
    not an invented finer grid).

    1.   measurement_sanity.png -- RSRP/SINR distributions, sanity flags.
    2+4. radial_profile_and_separability.png -- RSRP/SINR by region, ANOVA.
    3.   intra_region_coherence.png -- same-region vs. arbitrary-pair spread.
    5.   neighbor_edge_coupling.png -- own vs. cross-site-neighbor power gap.
    6+7. tilt_sensitivity_single_sector.png -- per-region coverage/RSRP/SINR
         vs. tilt (LOCAL sweep: only sector 0's own tilt changes, every
         other sector fixed at SINGLE_SECTOR_NOMINAL_TILT_DEG), restricted to sector 0's own cells --
         what a single-sector-controlled agent's own tilt choice actually
         buys. (An earlier version used the UNIFORM sweep -- every sector's
         tilt changing together -- instead; dropped as misleading, since
         that reads as "this region's own coverage vs. its own tilt" but is
         actually dominated by network-wide interference coordination, not
         this sector's own decision -- see single_sector_* figures below
         for the <1 dB-vs-several-dB comparison this was based on.)
         tilt_sensitivity_single_sector_neighbors_aggressive_downtilt.png /
         ..._aggressive_uptilt.png -- SAME layout, but with every OTHER
         sector fixed at the sweep's max/min tilt instead of the nominal -- tests
         whether a neighbor-coordinated extreme changes what sector 0's own
         tilt can achieve for its own regions. Confirmed: neighbors at max
         downtilt lifts sector 0's own far coverage from 0% to 50-65%, but
         ONLY while sector 0 itself stays at a moderate tilt -- matching
         its own tilt to its neighbors' collapses that gain back down,
         since neighbor-interference relief and sector 0's own signal loss
         pull in opposite directions for its own far users.
         spatial_heatmaps_by_tilt.png (+ ..._binned.png) -- same uniform
         sweep, but as RSRP/SINR spatial maps (one column per tilt) instead
         of per-region averages; the _binned variant discretizes the color
         scale into fixed-width dB bins instead of a continuous gradient.
         single_sector_rsrp.png / single_sector_sinr.png -- the
         complementary LOCAL sweep: only sector 0's own tilt changes (every
         other sector held at nominal), zoomed on sector 0 + its neighbors,
         absolute value (top row) directly above its own change-vs-baseline
         (bottom row) per tilt column -- isolates one sector's own effect,
         including its leak into neighboring sectors, from the uniform
         sweep above. single_sector_coverage.png -- same local sweep,
         absolute coverage (top) + a 4-way still-uncovered/gained/lost/
         still-covered transition map (bottom) vs. the baseline tilt.
         single_sector_rsrp_binned.png / single_sector_sinr_binned.png --
         same local sweep, absolute (top, same convention as
         spatial_heatmaps_by_tilt_binned.png above) + change-vs-baseline
         (bottom, symmetric bins around 0) -- both binned color scales.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Grouped output (2026-10-10): measurement_sanity.png alone stays
    # directly in out_dir; everything else nests under one of these three,
    # by which question it answers -- region definition/separability
    # (regions/), own-sector local-sweep spatial detail (tilt_sensitivity/),
    # or own-vs-network-wide tilt-sensitivity curves (regional_tilt_sensitivity/).
    regions_dir = os.path.join(out_dir, "regions")
    tilt_sensitivity_dir = os.path.join(out_dir, "tilt_sensitivity")
    regional_tilt_sensitivity_dir = os.path.join(out_dir, "regional_tilt_sensitivity")
    for d in (regions_dir, tilt_sensitivity_dir, regional_tilt_sensitivity_dir):
        os.makedirs(d, exist_ok=True)
    topo = engine.topology
    gt = compute_hotspot_ground_truth(engine.downtilt_sweep_deg, all_indoor=all_indoor)
    valid = gt["near"] | gt["mid"] | gt["far"]
    region_of = np.full(valid.size, "", dtype=object)
    region_of[gt["near"]] = "near"
    region_of[gt["mid"]] = "mid"
    region_of[gt["far"]] = "far"
    region_order = ["near", "mid", "far"]
    print(f"Phase 2: {int((~valid).sum())}/{valid.size} virtual cells excluded "
         "(outside every sector's own rhombus, or within min_bs_ut_dist of the site)")

    # --- figure 0: region definition overview (2026-10-10, moved from
    # phase 1) --- the radial-distance region boundaries
    # (CellularTopology.hotspot_region_membership), the SAME definition
    # mobility/phase3 already relies on -- drawn here instead of phase 1
    # since phase 1 is pure static geometry with no measurement to justify
    # picking a region definition at all. A SINR-measurement-based
    # alternative definition is a separate, not-yet-designed TODO (would
    # classify region from real measured RSRP/SINR, e.g. via
    # SpatialGridEstimator.infer_region_probabilities, rather than this
    # fixed geometric rule) -- not drawn here yet.
    region_linestyle = (0, (5, 10))  # loosely dashed
    region_legend_handles = [
        Line2D([0], [0], color=SITE_COLOR, linewidth=1.5, linestyle="-", label="site"),
        Line2D([0], [0], color=SECTOR_COLOR, linewidth=1.2, linestyle=(0, (5, 1)), label="sector"),
        Line2D([0], [0], color=REGION_COLOR, linewidth=1.0, linestyle=region_linestyle, label="region boundary"),
        *(mpatches.Patch(facecolor=c, edgecolor="none", alpha=0.35, label=r)
         for r, c in HOTSPOT_REGION_COLORS.items()),
    ]
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_sites_and_sectors(ax, topo, site_color=SITE_COLOR, sector_color=SECTOR_COLOR,
                           sector_linestyle=(0, (5, 1)), site_zorder=5, sector_zorder=4, id_fontsize=10)
    draw_regions(ax, engine, range(topo.num_bs), label=True, color=REGION_COLOR,
                linestyle=region_linestyle, zorder=3, id_fontsize=7)
    ax.legend(handles=region_legend_handles, loc="upper right", fontsize=9)
    ax.set_title(f"Region definition (radial-distance): {topo.num_bs} sectors", fontsize=10)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(regions_dir, "region_definition_overview.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- figure 1: measurement sanity [check 1] ---
    # Checks 1-5's own FIXED baseline tilt is SINGLE_SECTOR_NOMINAL_TILT_DEG
    # (isolates geometry-driven regional effects from tilt as a confound --
    # see this function's own docstring) -- looked up BY VALUE, not by
    # position, since downtilt_sweep_deg's own start/end (see config.yaml)
    # don't pin any particular index to this value.
    tilt0_idx = int(np.argmin(np.abs(gt["tilt_values"] - SINGLE_SECTOR_NOMINAL_TILT_DEG)))
    rsrp0, sinr0 = gt["rsrp_dbm"][tilt0_idx], gt["sinr_db"][tilt0_idx]
    noise_floor_dbm = 10.0 * np.log10(engine.kpi_manager.noise_power_w) + 30.0
    bad = (~np.isfinite(rsrp0)) | (~np.isfinite(sinr0)) | (rsrp0 > 0.0) | (rsrp0 < noise_floor_dbm)
    print(f"  [check 1] RSRP range=[{rsrp0[valid].min():.1f}, {rsrp0[valid].max():.1f}] dBm, "
         f"SINR range=[{sinr0[valid].min():.1f}, {sinr0[valid].max():.1f}] dB, "
         f"noise floor={noise_floor_dbm:.1f} dBm, {int((bad & valid).sum())} flagged cell(s)")
    xy_all = engine.grid_points.reshape(-1, 2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].hist(rsrp0[valid], bins=40, color="tab:blue")
    axes[0, 0].set_title("RSRP (dBm), all valid grid cells")
    axes[0, 1].hist(sinr0[valid], bins=40, color="tab:orange")
    axes[0, 1].set_title("SINR (dB), all valid grid cells")
    for ax, values, label, cmap in ((axes[1, 0], rsrp0, "RSRP (dBm)", COLORBAR_CMAP),
                                    (axes[1, 1], sinr0, "SINR (dB)", COLORBAR_CMAP)):
        draw_sites_and_sectors(ax, topo)
        sc = ax.scatter(xy_all[valid, 0], xy_all[valid, 1], s=14, c=values[valid], cmap=cmap, linewidths=0, zorder=2)
        ax.set_title(f"{label} spatial heatmap (tilt={SINGLE_SECTOR_NOMINAL_TILT_DEG:.0f})", fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.03, label=label)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "measurement_sanity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 2: radial profile + separability [checks 2, 4] ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for ax, values, label in ((axes[0], rsrp0, "RSRP (dBm)"), (axes[1], sinr0, "SINR (dB)")):
        data = [values[valid & (region_of == r)] for r in region_order]
        bp = ax.boxplot(data, tick_labels=region_order, patch_artist=True)
        for patch, r in zip(bp["boxes"], region_order):
            patch.set_facecolor(HOTSPOT_REGION_COLORS[r])
        f_stat, p_val = f_oneway(*data)
        ax.set_title(f"{label} by region (F={f_stat:.1f}, p={p_val:.2e})", fontsize=10)
    eta_bins = np.linspace(0.0, 1.0, 21)
    bin_idx = np.digitize(gt["eta"][valid], eta_bins) - 1
    bin_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
    for ax, values, label in ((axes[2], rsrp0, "RSRP (dBm)"), (axes[3], sinr0, "SINR (dB)")):
        v = values[valid]
        means = [v[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(len(bin_centers))]
        stds = [v[bin_idx == i].std() if (bin_idx == i).any() else np.nan for i in range(len(bin_centers))]
        ax.errorbar(bin_centers, means, yerr=stds, fmt="-o", markersize=3, color="tab:purple")
        ax.set_xlabel("eta (radial position, 0=site, 1=outer boundary)")
        ax.set_title(f"{label} vs. eta (mean +/- std)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(regions_dir, "radial_profile_and_separability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 3: intra-region coherence [check 3] ---
    # NOTE: near/mid/far each still spans a real range of distance-to-site
    # and diagonal-offset (see hotspot_region_membership), so some spread
    # WITHIN one (sector, region) bucket is expected even where the region
    # classification itself is working correctly -- this checks for a
    # real, if possibly modest, coherence effect (via a one-sided
    # Mann-Whitney U test), not a "tight cluster vs wide spread" visual on
    # its own.
    rng = np.random.default_rng(0)
    valid_idx = np.flatnonzero(valid)
    num_pairs = 5000
    buckets = defaultdict(list)
    for idx in valid_idx:
        buckets[(gt["sector_tag"][idx], region_of[idx])].append(idx)
    bucket_list = [np.array(b) for b in buckets.values() if len(b) >= 2]
    pairs_a = np.stack([rng.choice(bucket_list[i], size=2, replace=False)
                        for i in rng.integers(len(bucket_list), size=num_pairs)])
    pairs_b = np.stack([rng.choice(valid_idx, size=num_pairs), rng.choice(valid_idx, size=num_pairs)], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, values, label in ((axes[0], rsrp0, "|delta RSRP| (dB)"), (axes[1], sinr0, "|delta SINR| (dB)")):
        delta_a = np.abs(values[pairs_a[:, 0]] - values[pairs_a[:, 1]])
        delta_b = np.abs(values[pairs_b[:, 0]] - values[pairs_b[:, 1]])
        _, p_val = mannwhitneyu(delta_a, delta_b, alternative="less")
        ax.boxplot([delta_a, delta_b], tick_labels=["same sector+region", "arbitrary pair"])
        ax.set_title(f"{label}: mean same-region={delta_a.mean():.2f} vs. arbitrary={delta_b.mean():.2f} "
                    f"(Mann-Whitney p={p_val:.2e})", fontsize=9)
        print(f"  [check 3] {label}: same-region mean={delta_a.mean():.2f}/median={np.median(delta_a):.2f}, "
             f"arbitrary mean={delta_b.mean():.2f}/median={np.median(delta_b):.2f}, "
             f"one-sided Mann-Whitney p={p_val:.2e} (H1: same-region deltas are smaller)")
    fig.tight_layout()
    fig.savefig(os.path.join(regions_dir, "intra_region_coherence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 4: neighbor-edge coupling [check 5] ---
    # Restricted to CROSS-SITE neighbors only -- same-site siblings share a
    # RADIAL sector-division line, not the sector's outer rhombus
    # boundary, so including them would confound this check with an
    # unrelated proximity effect.
    gap_by_region = {r: [] for r in region_order}
    excluded_no_neighbor = 0
    power_db0 = gt["power_db_full"][tilt0_idx]  # [num_bs, N] -- nominal-tilt baseline, see tilt0_idx above
    for s in range(topo.num_bs):
        raw = topo.neighbor_ids[s]
        cross_site = raw[(raw >= 0) & (raw // topo.num_sectors_per_site != s // topo.num_sectors_per_site)]
        own_mask = valid & (gt["sector_tag"] == s)
        if cross_site.size == 0:
            excluded_no_neighbor += int(own_mask.sum())
            continue
        own_db = power_db0[s, own_mask]
        neighbor_best_db = power_db0[cross_site][:, own_mask].max(axis=0)
        gap = own_db - neighbor_best_db
        own_region = region_of[own_mask]
        for r in region_order:
            gap_by_region[r].extend(gap[own_region == r].tolist())
    print(f"  [check 5] {excluded_no_neighbor} virtual cell(s) excluded (own sector has no cross-site neighbor)")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    data = [gap_by_region[r] for r in region_order]
    bp = ax.boxplot(data, tick_labels=region_order, patch_artist=True)
    for patch, r in zip(bp["boxes"], region_order):
        patch.set_facecolor(HOTSPOT_REGION_COLORS[r])
    ax.set_ylabel("own sector power - best cross-site-neighbor power (dB)")
    ax.set_title("Neighbor-edge coupling: smaller gap = stronger neighbor interaction", fontsize=10)
    fig.savefig(os.path.join(regions_dir, "neighbor_edge_coupling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 5: per-region coverage/RSRP/SINR vs. tilt, LOCAL
    # single-sector sweep only (only sector 0's own tilt changes, every
    # other sector held fixed at other_sectors_tilt_deg) -- what a
    # single-sector-controlled agent's own tilt choice actually buys, as
    # opposed to what the whole network moving together buys. An earlier
    # version of this figure used the UNIFORM sweep instead (every sector's
    # tilt changing together) -- dropped: it reads as "this region's own
    # coverage vs. its own tilt" but is actually dominated by network-wide
    # interference coordination (confirmed empirically: a single sector's
    # own tilt alone moves RSRP/SINR by <1 dB vs. several dB for the
    # uniform sweep -- see single_sector_* figures below), so it was
    # answering a different, easily-conflated question and not actually
    # informative for what a single sector's own decision buys.
    tilt_values = gt["tilt_values"]
    # Restricted to target_sector's own cells only (own_mask below) -- a
    # neighboring sector's cells are a different geographic sector's own
    # near/mid/far, not this sweep's target.
    target_sector = 0
    own_mask = gt["sector_tag"] == target_sector
    m = topo.hotspot_region_membership(target_sector, engine.grid_points[target_sector])

    def _tilt_sensitivity_single_sector_figure(other_sectors_tilt_deg, fname):
        """Sweeps ONLY target_sector's own tilt, holding every other sector
        fixed at other_sectors_tilt_deg -- neighbors_tilt=SINGLE_SECTOR_NOMINAL_TILT_DEG
        is the nominal baseline; neighbors_tilt=tilt_values.max()/.min() (aggressive
        downtilt/uptilt) tests whether a NEIGHBOR-coordinated extreme
        changes what this sector's own tilt choice can achieve for its own
        regions (confirmed empirically: neighbors at max downtilt lifts
        this sector's own far coverage from 0% to 50-65%, but ONLY while
        this sector itself stays at a moderate tilt -- matching its own
        aggressive downtilt collapses that gain back down, since the
        neighbor-interference relief and this sector's own signal loss
        pull in opposite directions for its own far users). Only absolute
        rsrp_dbm/sinr_db are plotted here (never the delta fields), so
        reference_tilt_deg is fixed at SINGLE_SECTOR_NOMINAL_TILT_DEG
        regardless of other_sectors_tilt_deg -- keeps the returned pert's
        delta-vs-baseline semantics identical to the original single call
        for callers downstream that DO use those fields (the
        other_sectors_tilt_deg=SINGLE_SECTOR_NOMINAL_TILT_DEG case, reused
        for single_sector_* below).
        """
        pert_local = compute_single_sector_tilt_perturbation(target_sector, tilt_values, other_sectors_tilt_deg,
                                                              reference_tilt_deg=SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                                              all_indoor=all_indoor)
        # "whole" (2026-10-12): every one of target_sector's own cells,
        # not restricted to any one region -- plotted alongside near/mid/
        # far, not replacing them, so the regional breakdown and the
        # whole-sector aggregate are directly comparable on the same axes.
        plot_groups = region_order + ["whole"]
        coverage_by_region = {r: [] for r in plot_groups}
        rsrp_mean_by_region = {r: [] for r in plot_groups}
        sinr_mean_by_region = {r: [] for r in plot_groups}
        for t in range(len(tilt_values)):
            rsrp_t, sinr_t = pert_local["rsrp_dbm"][t][own_mask], pert_local["sinr_db"][t][own_mask]
            for r in plot_groups:
                region_mask = m[r] if r != "whole" else np.ones_like(sinr_t, dtype=bool)
                coverage_by_region[r].append(float((sinr_t[region_mask] > engine.coverage_threshold_db).mean()))
                rsrp_mean_by_region[r].append(float(rsrp_t[region_mask].mean()))
                sinr_mean_by_region[r].append(float(sinr_t[region_mask].mean()))
        whole_style = dict(color="black", linestyle="--", marker="s")
        fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
        for r in region_order:
            axes[0].plot(tilt_values, coverage_by_region[r], "-o", color=HOTSPOT_REGION_COLORS[r], label=r)
            axes[1].plot(tilt_values, rsrp_mean_by_region[r], "-o", color=HOTSPOT_REGION_COLORS[r], label=r)
            axes[2].plot(tilt_values, sinr_mean_by_region[r], "-o", color=HOTSPOT_REGION_COLORS[r], label=r)
        axes[0].plot(tilt_values, coverage_by_region["whole"], label="whole area", **whole_style)
        axes[1].plot(tilt_values, rsrp_mean_by_region["whole"], label="whole area", **whole_style)
        axes[2].plot(tilt_values, sinr_mean_by_region["whole"], label="whole area", **whole_style)
        axes[0].set_xlabel("tilt (deg)"); axes[0].set_ylabel("coverage fraction"); axes[0].legend(fontsize=8)
        axes[0].set_title(f"Coverage vs. sector {target_sector}'s OWN tilt, by region\n"
                          f"(others fixed at {other_sectors_tilt_deg:.1f} deg)", fontsize=10)
        axes[1].set_xlabel("tilt (deg)"); axes[1].set_ylabel("mean RSRP (dBm)"); axes[1].legend(fontsize=8)
        axes[1].set_title(f"Mean RSRP vs. sector {target_sector}'s OWN tilt, by region", fontsize=10)
        axes[2].set_xlabel("tilt (deg)"); axes[2].set_ylabel("mean SINR (dB)"); axes[2].legend(fontsize=8)
        axes[2].set_title(f"Mean SINR vs. sector {target_sector}'s OWN tilt, by region", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(regional_tilt_sensitivity_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return pert_local

    pert = _tilt_sensitivity_single_sector_figure(SINGLE_SECTOR_NOMINAL_TILT_DEG, "tilt_sensitivity_single_sector.png")
    reference_tilt_deg = SINGLE_SECTOR_NOMINAL_TILT_DEG
    reference_idx = int(np.argmin(np.abs(tilt_values - reference_tilt_deg)))
    # nominal baseline for EVERY sector, not just target_sector's own
    # reference point -- 2026-10-08, was 0.0 deg, which (now that mechanical
    # pitch is horizontal) meant every OTHER sector pointed at the horizon,
    # not a realistic baseline. kept for the single_sector_* figures below,
    # which reuse this `pert`.
    other_sectors_tilt_deg = SINGLE_SECTOR_NOMINAL_TILT_DEG
    _tilt_sensitivity_single_sector_figure(
        float(tilt_values.max()), "tilt_sensitivity_single_sector_neighbors_aggressive_downtilt.png")
    _tilt_sensitivity_single_sector_figure(
        float(tilt_values.min()), "tilt_sensitivity_single_sector_neighbors_aggressive_uptilt.png")

    # --- figure 5c: fine-grained RADIAL distance crossover, averaged over
    # several seeds -- figure 5b's near/mid/far buckets above pool a WIDE
    # span of distances into just 3 numbers each (e.g. mid = everything
    # 900-2300m averaged together), which can visually wash out a sharp,
    # localized crossover. This instead bins by raw distance-to-site
    # (NUM_RADIAL_BINS fine bins, not 3 coarse regions) and compares the
    # tilt sweep's own two ENDPOINTS directly, averaged over
    # NUM_RADIAL_SEEDS independent random seeds (a single seed's own
    # shadow-fading draw can make a real crossover look noisier/weaker
    # than it robustly is -- see check 1/measurement_sanity.png for the
    # same single-vs-averaged tension elsewhere in this phase). Adapted
    # 2026-10-05 from a one-off investigation script into a permanent,
    # reusable diagnostic -- no config overrides here (unlike that
    # script), always reflects config.yaml's own current settings, same
    # as every other figure in this phase.
    NUM_RADIAL_SEEDS = 3
    NUM_RADIAL_BINS = 15
    tilt_lo, tilt_hi = float(tilt_values.min()), float(tilt_values.max())
    site_xy = topo.bs_loc[0, target_sector, :2].cpu().numpy()
    xy_own = engine.grid_points[target_sector]
    dist_own = np.linalg.norm(xy_own - site_xy[None, :], axis=1)
    radial_bins = np.linspace(0, float(dist_own.max()), NUM_RADIAL_BINS + 1)
    radial_bin_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_bin_idx = np.digitize(dist_own, radial_bins) - 1

    all_cov_lo, all_cov_hi = [], []
    for seed in range(1, NUM_RADIAL_SEEDS + 1):
        sionna.phy.config.seed = seed
        seed_engine = build_debug_engine()
        seed_pert = compute_single_sector_tilt_perturbation(
            target_sector, np.array([tilt_lo, tilt_hi]), other_sectors_tilt_deg=SINGLE_SECTOR_NOMINAL_TILT_DEG,
            engine=seed_engine, all_indoor=all_indoor)
        seed_num_grid_cells = seed_engine.num_grid_cells
        seed_sl = slice(target_sector * seed_num_grid_cells, (target_sector + 1) * seed_num_grid_cells)
        cov_lo = seed_pert["sinr_db"][0][seed_sl] > seed_engine.coverage_threshold_db
        cov_hi = seed_pert["sinr_db"][1][seed_sl] > seed_engine.coverage_threshold_db
        cov_lo_by_bin = np.full(NUM_RADIAL_BINS, np.nan)
        cov_hi_by_bin = np.full(NUM_RADIAL_BINS, np.nan)
        for b in range(NUM_RADIAL_BINS):
            mask = radial_bin_idx == b
            if mask.sum() > 0:
                cov_lo_by_bin[b] = cov_lo[mask].mean()
                cov_hi_by_bin[b] = cov_hi[mask].mean()
        all_cov_lo.append(cov_lo_by_bin)
        all_cov_hi.append(cov_hi_by_bin)
    sionna.phy.config.seed = DEBUG_SEED  # restore -- every other figure in this phase assumes this

    all_cov_lo, all_cov_hi = np.array(all_cov_lo), np.array(all_cov_hi)
    mean_lo, mean_hi = np.nanmean(all_cov_lo, axis=0), np.nanmean(all_cov_hi, axis=0)
    std_lo, std_hi = np.nanstd(all_cov_lo, axis=0), np.nanstd(all_cov_hi, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(radial_bin_centers, mean_lo, yerr=std_lo, marker="o", label=f"tilt={tilt_lo:.1f} deg",
               color="tab:blue", capsize=3)
    ax.errorbar(radial_bin_centers, mean_hi, yerr=std_hi, marker="s", label=f"tilt={tilt_hi:.1f} deg",
               color="tab:red", capsize=3)
    ax.set_xlabel("distance from site (m)")
    ax.set_ylabel(f"coverage fraction (mean +/- std across {NUM_RADIAL_SEEDS} seeds)")
    ax.set_title(f"Sector {target_sector}: coverage vs distance at the tilt sweep's own endpoints\n"
                f"(averaged over {NUM_RADIAL_SEEDS} seeds -- finer-grained than the near/mid/far buckets above)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(regional_tilt_sensitivity_dir, "radial_crossover_single_sector.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [radial crossover] sector={target_sector} tilt={tilt_lo:.1f} vs {tilt_hi:.1f} deg, "
         f"{NUM_RADIAL_SEEDS} seeds, {NUM_RADIAL_BINS} distance bins:")
    for i, bc in enumerate(radial_bin_centers):
        print(f"    {bc:7.0f}m  tilt{tilt_lo:.1f}={mean_lo[i]:.3f}  tilt{tilt_hi:.1f}={mean_hi[i]:.3f}  "
             f"diff={mean_hi[i]-mean_lo[i]:+.3f}")

    # --- figure 6: spatial RSRP/SINR heatmaps at every grid location, one
    # column per tilt [checks 1, 6 -- spatial companion to the aggregate
    # per-region tilt curves above: same style as the RME validation
    # coverage maps (draw_sites_and_sectors + a colored grid-point scatter),
    # one row for RSRP and one for SINR, sharing a fixed color scale across
    # tilts (computed from valid cells across the WHOLE sweep) so panels
    # are directly comparable.
    rsrp_vmin, rsrp_vmax = gt["rsrp_dbm"][:, valid].min(), gt["rsrp_dbm"][:, valid].max()
    sinr_vmin, sinr_vmax = gt["sinr_db"][:, valid].min(), gt["sinr_db"][:, valid].max()
    fig, axes = plt.subplots(2, len(tilt_values), figsize=(5.5 * len(tilt_values), 11), squeeze=False)
    for t, tilt in enumerate(tilt_values):
        for row, (values, vmin, vmax, cmap, label) in enumerate((
                (gt["rsrp_dbm"][t], rsrp_vmin, rsrp_vmax, COLORBAR_CMAP, "RSRP (dBm)"),
                (gt["sinr_db"][t], sinr_vmin, sinr_vmax, COLORBAR_CMAP, "SINR (dB)"))):
            ax = axes[row, t]
            draw_sites_and_sectors(ax, topo)
            sc = ax.scatter(xy_all[valid, 0], xy_all[valid, 1], s=14, c=values[valid],
                            cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, zorder=2)
            ax.set_title(f"{label}, tilt={tilt:.1f} deg", fontsize=10)
            ax.set_aspect("equal")
            fig.colorbar(sc, ax=ax, fraction=0.03, label=label)
    fig.tight_layout()
    fig.savefig(os.path.join(tilt_sensitivity_dir, "spatial_heatmaps_by_tilt.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 6b: SAME data as figure 6, but with a BINNED (discretized)
    # color scale instead of continuous -- a separate, additional figure,
    # not a replacement. Values outside the outermost bin edges are clipped
    # into the nearest bin (BoundaryNorm(..., clip=True)) rather than left
    # off-scale.
    NUM_BINS = 3

    def _binned_norm_cmap(values):
        """Data-driven: exactly NUM_BINS=3 equal-width bins spanning the
        data's own [min, max] -- used for RSRP, where a fixed
        connectivity-style cutoff (like SINR's 0/5 dB) doesn't apply.
        Always exactly 3 bins regardless of the data's own range (2026-10-08
        -- previously a FIXED 10 dB bin width with NUM_BINS=4, which could
        read as many more than 3 visually distinct bins once combined with
        the diverging delta panels below; standardized to a clean 3-bin
        read everywhere in this phase)."""
        edges = np.linspace(values.min(), values.max(), NUM_BINS + 1)
        norm = BoundaryNorm(edges, NUM_BINS, clip=True)
        cmap = ListedColormap([plt.get_cmap(COLORBAR_CMAP)(i / (NUM_BINS - 1)) for i in range(NUM_BINS)])
        return norm, cmap, edges

    def _binned_norm_cmap_fixed(values, interior_edges):
        """Fixed interior cut points (e.g. SINR's connectivity-meaningful
        0 dB / 5 dB), with the data's own min/max as the outer bin edges."""
        edges = np.concatenate([[values.min()], interior_edges, [values.max()]])
        num_bins = len(edges) - 1
        norm = BoundaryNorm(edges, num_bins, clip=True)
        cmap = ListedColormap([plt.get_cmap(COLORBAR_CMAP)(i / (num_bins - 1)) for i in range(num_bins)])
        return norm, cmap, edges

    rsrp_norm, rsrp_cmap, rsrp_edges = _binned_norm_cmap(gt["rsrp_dbm"][:, valid])
    # SINR: fixed at 0/5 dB (below 0 / 0-5 / above 5) -- matches the
    # connectivity-threshold discussion (0 dB ~ coverage floor, 5 dB ~
    # a step toward usable service), not an arbitrary data-driven cut.
    sinr_norm, sinr_cmap, sinr_edges = _binned_norm_cmap_fixed(gt["sinr_db"][:, valid], [0.0, 5.0])
    fig, axes = plt.subplots(2, len(tilt_values), figsize=(5.5 * len(tilt_values), 11), squeeze=False)
    for t, tilt in enumerate(tilt_values):
        for row, (values, norm, cmap, edges, label) in enumerate((
                (gt["rsrp_dbm"][t], rsrp_norm, rsrp_cmap, rsrp_edges, "RSRP (dBm)"),
                (gt["sinr_db"][t], sinr_norm, sinr_cmap, sinr_edges, "SINR (dB)"))):
            ax = axes[row, t]
            draw_sites_and_sectors(ax, topo)
            sc = ax.scatter(xy_all[valid, 0], xy_all[valid, 1], s=14, c=values[valid],
                            cmap=cmap, norm=norm, linewidths=0, zorder=2)
            ax.set_title(f"{label}, tilt={tilt:.1f} deg", fontsize=10)
            ax.set_aspect("equal")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, label=label, ticks=edges)
            cbar.ax.set_yticklabels([f"{e:.0f}" for e in edges])
    fig.tight_layout()
    fig.savefig(os.path.join(tilt_sensitivity_dir, "spatial_heatmaps_by_tilt_binned.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 7: single-sector tilt perturbation -- LOCAL sweep, holds
    # every OTHER sector fixed at other_sectors_tilt_deg and varies only
    # target_sector's own tilt. Zoomed on target_sector + its own
    # neighbor_sets only (same zoom convention as phase1's per-sector
    # panel) -- a single sector's tilt is a genuinely LOCAL perturbation
    # (confirmed empirically: only ~5% of network-wide cells exceed 1 dB of
    # change at all), so a whole-network view is mostly uninformative
    # near-zero space; dropped in favor of keeping each figure to one
    # sector's own neighborhood, at full readable detail.
    # Three separate figures (RSRP, SINR, coverage) so no single one gets
    # overloaded, each with absolute value (top row) directly above its own
    # change-vs-reference_tilt_deg baseline (bottom row) for the SAME tilt
    # column -- lets a reader connect "what the value actually is" with
    # "how much THIS tilt moved it" without cross-referencing a separate
    # figure. Plus two binned-color variants (RSRP, SINR; absolute only,
    # same fixed/data-driven bin convention as figure 6b above) -- a coarse,
    # connectivity-style read that's easier to eyeball than a continuous
    # gradient at this zoom level. Reuses target_sector/other_sectors_tilt_deg/
    # reference_tilt_deg/pert/reference_idx computed for figure 5b above.
    sector_set = [target_sector] + engine.regional_info_provider.neighbor_sets[target_sector]
    bs_xy_all = topo.bs_loc[0, :, :2].cpu().numpy()
    cell_radius = float(topo.grid.cell_radius.item())
    pad = 1.3 * cell_radius
    member_xy = bs_xy_all[sector_set]
    local_mask = np.isin(gt["sector_tag"], sector_set)
    # Marker size/edge scaled to point density -- a fixed s=40 + black edge
    # fully saturates into a near-solid black mass once the grid gets dense
    # enough. Baseline is 3364 cells/sector (2026-10-08, corrected from a
    # stale 36 -- that was UMa's own cell count back when
    # spatial_grid_cell_size_m was still RMa's unscaled 50m default,
    # producing a degenerate 6x6 grid; UMa's own committed value is 5m,
    # giving the SAME ~3364 cells/sector as RMa, so both scenarios now share
    # one real calibration point instead of the old UMa-only one). Keeps the
    # SAME total plotted "ink" roughly constant instead of a per-marker
    # constant that only works at one scale.
    local_n = max(int(local_mask.sum()), 1)
    scatter_s = float(np.clip(40.0 * np.sqrt(3364.0 / local_n), 0.3, 40.0))
    scatter_edge_kwargs = (dict(linewidths=0.3, edgecolors="black") if local_n <= 200
                          else dict(linewidths=0.0, edgecolors="none"))

    def _zoomed_ax(ax):
        # site/sector boundary lines at a HIGHER zorder than the scatter
        # (zorder=2 below) -- otherwise a dense point cloud (thousands of
        # cells/sector) draws over and completely buries the thin dashed
        # boundary lines, since equal zorder falls back to call order and
        # this was being called before the scatter every time (2026-10-08
        # fix -- sector boundaries were invisible under the real grid
        # density for exactly this reason).
        draw_sites_and_sectors(ax, topo, sector_ids=sector_set, site_zorder=5, sector_zorder=5)
        ax.set_xlim(member_xy[:, 0].min() - pad, member_xy[:, 0].max() + pad)
        ax.set_ylim(member_xy[:, 1].min() - pad, member_xy[:, 1].max() + pad)
        ax.set_aspect("equal")

    def _title(tilt):
        return f"sector {target_sector} tilt={tilt:.1f} deg (others={other_sectors_tilt_deg:.1f} deg)"

    def _abs_and_delta_figure(absolute, delta, label, fname):
        """2 rows (absolute, change-vs-baseline) x len(tilt_values) columns."""
        abs_vmin, abs_vmax = np.percentile(absolute[:, local_mask], [2, 98])
        delta_vmax = max(np.percentile(np.abs(delta[:, local_mask]), 98), 1e-6)
        fig, axes = plt.subplots(2, len(tilt_values), figsize=(5.0 * len(tilt_values), 10.5), squeeze=False)
        for t, tilt in enumerate(tilt_values):
            ax = axes[0, t]
            _zoomed_ax(ax)
            sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=scatter_s, c=absolute[t], cmap=COLORBAR_CMAP,
                            vmin=abs_vmin, vmax=abs_vmax, zorder=2, **scatter_edge_kwargs)
            ax.set_title(f"{_title(tilt)}\n{label}, absolute", fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=0.03, label=label)

            ax = axes[1, t]
            _zoomed_ax(ax)
            sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=scatter_s, c=delta[t], cmap=COLORBAR_CMAP,
                            vmin=-delta_vmax, vmax=delta_vmax, zorder=2, **scatter_edge_kwargs)
            ax.set_title(f"{_title(tilt)}\n{label} change vs {reference_tilt_deg:.1f} deg", fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=0.03, label=f"{label} change")
        fig.tight_layout()
        fig.savefig(os.path.join(tilt_sensitivity_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    _abs_and_delta_figure(pert["rsrp_dbm"], pert["rsrp_delta_dbm"], "RSRP (dBm)", "single_sector_rsrp.png")
    _abs_and_delta_figure(pert["sinr_db"], pert["sinr_delta_db"], "SINR (dB)", "single_sector_sinr.png")

    # --- coverage: absolute (covered/uncovered) + a 4-way transition map
    # vs. reference_tilt_deg (still-uncovered / gained / lost / still-covered)
    # -- a plain +-1/0 delta would collapse "always fine" and "always bad"
    # into the same 0, losing exactly the distinction that matters here.
    covered = (pert["sinr_db"] > engine.coverage_threshold_db).astype(int)  # [T, N]
    transition = np.where(covered[reference_idx][None, :] == 0,
                          np.where(covered == 1, 1, 0),      # was uncovered: gained(1) or still-uncovered(0)
                          np.where(covered == 1, 3, 2))      # was covered: still-covered(3) or lost(2)
    cov_cmap = ListedColormap(["#a6bddb", "#1a9850", "#d73027", "#67000d"])
    trans_labels = ["still uncovered", "gained", "lost", "still covered"]
    fig, axes = plt.subplots(2, len(tilt_values), figsize=(5.0 * len(tilt_values), 10.5), squeeze=False)
    for t, tilt in enumerate(tilt_values):
        ax = axes[0, t]
        _zoomed_ax(ax)
        sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=scatter_s, c=covered[t], cmap=ListedColormap(["#2c7bb6", "#d7191c"]),
                        vmin=0, vmax=1, zorder=2, **scatter_edge_kwargs)
        ax.set_title(f"{_title(tilt)}\ncoverage (SINR > {engine.coverage_threshold_db:.0f} dB), absolute", fontsize=9)

        ax = axes[1, t]
        _zoomed_ax(ax)
        sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=scatter_s, c=transition[t], cmap=cov_cmap,
                        vmin=-0.5, vmax=3.5, zorder=2, **scatter_edge_kwargs)
        ax.set_title(f"{_title(tilt)}\ncoverage change vs {reference_tilt_deg:.1f} deg", fontsize=9)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(trans_labels)
    fig.tight_layout()
    fig.savefig(os.path.join(tilt_sensitivity_dir, "single_sector_coverage.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 7b: own tilt x neighbor tilt coverage tradeoff (2026-10-10,
    # explicit request) -- single_sector_coverage.png above only ever holds
    # every OTHER sector fixed at ONE value (other_sectors_tilt_deg) while
    # target_sector's own tilt sweeps; this instead sweeps BOTH
    # independently across the SAME real downtilt_sweep_deg grid (5 values
    # each, 25 combos), to see whether the own/neighbor-tilt interaction
    # (qualitatively confirmed via the aggressive_downtilt/aggressive_uptilt
    # companion figures above) resolves into a cleaner tradeoff as a full 2D
    # grid than as two single-axis slices. Both outputs below share the SAME
    # baseline: own=neighbor=SINGLE_SECTOR_NOMINAL_TILT_DEG (9 deg), the
    # same nominal point every other figure in this phase references.
    own_tilt_values = tilt_values
    neighbor_tilt_values = tilt_values
    num_own, num_nbr = len(own_tilt_values), len(neighbor_tilt_values)
    sinr_grid = np.zeros((num_nbr, num_own, xy_all.shape[0]))
    for ni, nbr_tilt in enumerate(neighbor_tilt_values):
        pert_nbr = compute_single_sector_tilt_perturbation(
            target_sector, own_tilt_values, other_sectors_tilt_deg=nbr_tilt,
            reference_tilt_deg=SINGLE_SECTOR_NOMINAL_TILT_DEG, engine=engine, all_indoor=all_indoor)
        sinr_grid[ni] = pert_nbr["sinr_db"]
    own_baseline_idx = int(np.argmin(np.abs(own_tilt_values - SINGLE_SECTOR_NOMINAL_TILT_DEG)))
    nbr_baseline_idx = int(np.argmin(np.abs(neighbor_tilt_values - SINGLE_SECTOR_NOMINAL_TILT_DEG)))
    covered_grid = sinr_grid > engine.coverage_threshold_db  # [num_nbr, num_own, N]
    baseline_covered = covered_grid[nbr_baseline_idx, own_baseline_idx]  # [N]

    # --- 7b-i: aggregate heatmap -- one coverage-fraction GAIN/LOSS number
    # per (own tilt, neighbor tilt) combo, relative to the (9, 9) baseline,
    # for target_sector's own near/mid/far regions + the whole sector.
    region_panels = region_order + ["whole"]
    fig, heat_axes = plt.subplots(1, len(region_panels), figsize=(5.2 * len(region_panels), 5.5))
    for ax, r in zip(heat_axes, region_panels):
        region_sel = m[r] if r != "whole" else slice(None)
        baseline_frac = float(baseline_covered[own_mask][region_sel].mean())
        grid = np.zeros((num_nbr, num_own))
        for ni in range(num_nbr):
            for oi in range(num_own):
                grid[ni, oi] = float(covered_grid[ni, oi][own_mask][region_sel].mean()) - baseline_frac
        vmax = max(np.abs(grid).max(), 1e-6)
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_xticks(range(num_own)); ax.set_xticklabels([f"{t:.0f}" for t in own_tilt_values])
        ax.set_yticks(range(num_nbr)); ax.set_yticklabels([f"{t:.0f}" for t in neighbor_tilt_values])
        ax.set_xlabel(f"sector {target_sector}'s own tilt (deg)")
        ax.set_ylabel("every other sector's tilt (deg)")
        ax.set_title(f"{r}: coverage gain/loss vs ({SINGLE_SECTOR_NOMINAL_TILT_DEG:.0f},"
                    f"{SINGLE_SECTOR_NOMINAL_TILT_DEG:.0f}) baseline\n(baseline coverage={baseline_frac:.2f})",
                    fontsize=9)
        for ni in range(num_nbr):
            for oi in range(num_own):
                ax.text(oi, ni, f"{grid[ni, oi]:+.2f}", ha="center", va="center", fontsize=7,
                       color="white" if abs(grid[ni, oi]) > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="coverage fraction change")
    fig.suptitle(f"Sector {target_sector}: own tilt x neighbor tilt coverage tradeoff", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(tilt_sensitivity_dir, "single_sector_coverage_tiltgrid_heatmap.png"),
               dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 7b-ii: full spatial detail -- same 4-way still-uncovered/gained/
    # lost/still-covered transition convention as single_sector_coverage.png
    # above, but a 5(neighbor tilt)x5(own tilt) grid of panels instead of
    # one row, baseline fixed at (9, 9) for every panel (not a per-row
    # reference), so panels are directly comparable to EACH OTHER, not just
    # to their own row's own baseline.
    transition_grid = np.where(baseline_covered[None, None, :] == 0,
                              np.where(covered_grid, 1, 0),
                              np.where(covered_grid, 3, 2))  # [num_nbr, num_own, N]
    # Panels here are 3.3in vs single_sector_coverage.png's own 5.0in --
    # scatter_s (calibrated for THAT panel size) scaled down by the
    # resulting area ratio so points don't smear into solid noise bands.
    grid_scatter_s = scatter_s * (3.3 / 5.0) ** 2
    fig, grid_axes = plt.subplots(num_nbr, num_own, figsize=(3.3 * num_own, 3.3 * num_nbr), squeeze=False)
    for ni, nbr_tilt in enumerate(neighbor_tilt_values):
        for oi, own_tilt in enumerate(own_tilt_values):
            ax = grid_axes[num_nbr - 1 - ni, oi]  # neighbor tilt increases upward, matches heat_axes' origin="lower"
            _zoomed_ax(ax)
            sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=grid_scatter_s, c=transition_grid[ni, oi],
                            cmap=cov_cmap, vmin=-0.5, vmax=3.5, zorder=2, linewidths=0.0, edgecolors="none")
            ax.set_title(f"own={own_tilt:.0f}, nbr={nbr_tilt:.0f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=grid_axes, fraction=0.015, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(trans_labels)
    fig.suptitle(f"Sector {target_sector}: coverage change vs ({SINGLE_SECTOR_NOMINAL_TILT_DEG:.0f},"
                f"{SINGLE_SECTOR_NOMINAL_TILT_DEG:.0f}) baseline -- own tilt (columns) x neighbor tilt (rows)",
                fontsize=11, y=0.91)
    fig.savefig(os.path.join(tilt_sensitivity_dir, "single_sector_coverage_tiltgrid_spatial.png"),
               dpi=130, bbox_inches="tight")
    plt.close(fig)

    def _binned_figure(absolute, abs_norm, abs_cmap, abs_edges, delta, delta_norm, delta_cmap, delta_edges,
                       label, fname):
        """2 rows (absolute, change-vs-baseline) x len(tilt_values) columns, both binned color scales."""
        fig, axes = plt.subplots(2, len(tilt_values), figsize=(5.0 * len(tilt_values), 10.5), squeeze=False)
        for t, tilt in enumerate(tilt_values):
            ax = axes[0, t]
            _zoomed_ax(ax)
            sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=scatter_s, c=absolute[t], cmap=abs_cmap, norm=abs_norm,
                            zorder=2, **scatter_edge_kwargs)
            ax.set_title(f"{_title(tilt)}\n{label}, absolute", fontsize=9)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, label=label, ticks=abs_edges)
            cbar.ax.set_yticklabels([f"{e:.0f}" for e in abs_edges])

            ax = axes[1, t]
            _zoomed_ax(ax)
            sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=scatter_s, c=delta[t], cmap=delta_cmap, norm=delta_norm,
                            zorder=2, **scatter_edge_kwargs)
            ax.set_title(f"{_title(tilt)}\n{label} change vs {reference_tilt_deg:.1f} deg", fontsize=9)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, label=f"{label} change", ticks=delta_edges)
            cbar.ax.set_yticklabels([f"{e:.1f}" for e in delta_edges])
        fig.tight_layout()
        fig.savefig(os.path.join(tilt_sensitivity_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    local_rsrp_norm, local_rsrp_cmap, local_rsrp_edges = _binned_norm_cmap(pert["rsrp_dbm"][:, local_mask])
    local_sinr_norm, local_sinr_cmap, local_sinr_edges = _binned_norm_cmap_fixed(
        pert["sinr_db"][:, local_mask], [0.0, 5.0])
    rsrp_delta_norm, rsrp_delta_cmap, rsrp_delta_edges = _binned_norm_cmap_fixed(
        pert["rsrp_delta_dbm"][:, local_mask], [0.0, 5.0])
    sinr_delta_norm, sinr_delta_cmap, sinr_delta_edges = _binned_norm_cmap_fixed(
        pert["sinr_delta_db"][:, local_mask], [0.0, 5.0])
    _binned_figure(pert["rsrp_dbm"], local_rsrp_norm, local_rsrp_cmap, local_rsrp_edges,
                  pert["rsrp_delta_dbm"], rsrp_delta_norm, rsrp_delta_cmap, rsrp_delta_edges,
                  "RSRP (dBm)", "single_sector_rsrp_binned.png")
    _binned_figure(pert["sinr_db"], local_sinr_norm, local_sinr_cmap, local_sinr_edges,
                  pert["sinr_delta_db"], sinr_delta_norm, sinr_delta_cmap, sinr_delta_edges,
                  "SINR (dB)", "single_sector_sinr_binned.png")

    # Region-pooled means (near/mid/far, target_sector's own cells only) at
    # each tilt -- the same single-sector-alone-vs-uniform-sweep contrast
    # discussed above, printed as a quick console summary. Reuses own_mask/m
    # computed for figure 5b above.
    print(f"  [single-sector, sector={target_sector}] "
         f"{'tilt':>5} {'near_rsrp':>10} {'mid_rsrp':>10} {'far_rsrp':>10} "
         f"{'near_sinr':>10} {'mid_sinr':>10} {'far_sinr':>10} "
         f"{'near_cov':>9} {'mid_cov':>9} {'far_cov':>9}")
    for t, tilt in enumerate(tilt_values):
        rsrp_own, sinr_own = pert["rsrp_dbm"][t][own_mask], pert["sinr_db"][t][own_mask]
        cov_own = (sinr_own > engine.coverage_threshold_db).astype(float)
        row = [rsrp_own[m[r]].mean() for r in region_order] + [sinr_own[m[r]].mean() for r in region_order] \
            + [cov_own[m[r]].mean() for r in region_order]
        print(f"  {'':>19}{tilt:5.1f} " + " ".join(f"{v:10.3f}" for v in row[:6]) + " "
             + " ".join(f"{v:9.3f}" for v in row[6:]))

    print(f"Phase 2 done -- saved to {out_dir}/")


def compute_ground_truth_grid(engine: SimulationEngine, tilt_deg: np.ndarray,
                              batch_size: int = 800) -> np.ndarray:
    """Ground-truth per-cell coverage at EVERY sector's own grid cell, by
    treating grid cell centers as virtual UEs and resolving power at
    tilt_deg directly -- independent of where real UEs happen to be,
    unlike the Kriged reconstruction (which only sees real UEs' samples).

    CAVEAT: the channel model redraws shadow fading on every
    set_topology() call (real UEs included -- see project memory on this),
    so this is a statistically representative INDEPENDENT realization at
    the same positions/tilt, not an exact twin of this interval's real
    measurement. Fair for checking whether Kriging recovers the right
    large-scale coverage STRUCTURE; not a sample-exact replay.

    Chunked (batch_size virtual UEs per call, reset_topology() before
    each -- see _virtual_grid_state's own docstring) since the channel
    model's own LSP spatial-correlation matrix is O(N^2) in the virtual-UE
    count -- num_bs*num_grid_cells can be several thousand at fine grid
    resolutions (e.g. 4725 at 20m), same OOM risk as
    compute_hotspot_ground_truth/compute_ground_truth_grid_fraction.

    :output: [num_bs, num_grid_cells] coverage (0./1.).
    """
    xy_all = engine.grid_points.reshape(-1, 2)
    coverage_chunks = []
    with torch.no_grad():
        for start in range(0, xy_all.shape[0], batch_size):
            chunk = xy_all[start:start + batch_size]
            state = _virtual_grid_state(engine, xy=chunk)
            power = engine.kpi_manager.resolve_power_at_tilt(state, tilt_deg)[0]
            kpis = engine.kpi_manager.compute_ue_kpis(power[None, :, :])
            coverage_chunks.append((kpis["sinr_db"][0] > engine.coverage_threshold_db).astype(float))
            del state, power, kpis
            torch.cuda.empty_cache()
    coverage = np.concatenate(coverage_chunks)
    return coverage.reshape(engine.num_bs, engine.num_grid_cells)


def _cell_sub_points(topo, sub_samples_per_side: int) -> np.ndarray:
    """[num_bs, num_grid_cells, K, 2] world xy for every sub-point of
    every fine-grid cell of every sector (K = sub_samples_per_side**2),
    same rhombus (alpha, beta) basis as classify_fine_grid/_sector_grid_*
    -- evenly-spaced sub-cell CENTERS within each cell's own [0,1/n) x
    [0,1/n) footprint (same "cell-center-of-subcell" convention
    draw_hotspot_regions uses for its own sampled fill)."""
    n = topo._sector_grid_n
    bs_xy = np.asarray(topo._sector_grid_bs_xy)
    e1 = np.asarray(topo._sector_grid_e1)
    e2 = np.asarray(topo._sector_grid_e2)
    num_grid_cells = n * n

    sub_edges = np.linspace(0.0, 1.0 / n, sub_samples_per_side + 1)
    sub_centers = (sub_edges[:-1] + sub_edges[1:]) / 2
    sub_a, sub_b = np.meshgrid(sub_centers, sub_centers, indexing="ij")
    sub_a, sub_b = sub_a.ravel(), sub_b.ravel()  # [K]

    i_idx = np.arange(num_grid_cells) // n
    j_idx = np.arange(num_grid_cells) % n
    alpha = i_idx[:, None] / n + sub_a[None, :]  # [num_grid_cells, K]
    beta = j_idx[:, None] / n + sub_b[None, :]

    return (bs_xy[:, None, None, :] + alpha[None, :, :, None] * e1[:, None, None, :]
           + beta[None, :, :, None] * e2[:, None, None, :])  # [num_bs, num_grid_cells, K, 2]


def compute_ground_truth_grid_fraction(engine: SimulationEngine, tilt_deg: np.ndarray,
                                       sub_samples_per_side: int = 3, batch_size: int = 800) -> np.ndarray:
    """Continuous per-cell ground-truth COVERAGE FRACTION -- unlike
    compute_ground_truth_grid's own single cell-center point (whose binary
    decision can flip depending on exactly which spot inside a genuinely-
    mixed cell it lands on -- see the within-cell antenna-pattern RSRP
    variation this session traced: a narrow vertical array beam pattern
    plus cells close to a site sweeping several degrees of elevation angle
    across one 50m cell), this samples a sub_samples_per_side x
    sub_samples_per_side sub-grid spanning EACH cell's own footprint,
    thresholds every sub-point, and averages -- a continuous [0,1]
    fraction representing what share of the cell's own area is actually
    covered, not one arbitrary point's answer.

    Chunked (batch_size virtual UEs per call, reset_topology() before
    each -- see _virtual_grid_state's own docstring for why).

    :output: [num_bs, num_grid_cells] continuous coverage fraction in [0,1].
    """
    sub_xy = _cell_sub_points(engine.topology, sub_samples_per_side)
    num_bs, num_grid_cells, K, _ = sub_xy.shape
    flat = sub_xy.reshape(-1, 2)
    covered_chunks = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            chunk = flat[start:start + batch_size]
            state = _virtual_grid_state(engine, xy=chunk)
            power = engine.kpi_manager.resolve_power_at_tilt(state, tilt_deg)[0]
            kpis = engine.kpi_manager.compute_ue_kpis(power[None, :, :])
            covered_chunks.append((kpis["sinr_db"][0] > engine.coverage_threshold_db).astype(float))
            del state, power, kpis
            torch.cuda.empty_cache()
    covered = np.concatenate(covered_chunks)
    return covered.reshape(num_bs, num_grid_cells, K).mean(axis=2)


def _nmse(predicted: np.ndarray, true: np.ndarray) -> float:
    """MSE / variance(true) -- 0=perfect, 1=no better than predicting the
    mean."""
    mse = float(np.mean((predicted - true) ** 2))
    var = float(np.var(true))
    return mse / var if var > 0 else float("nan")


def compute_grid_validation_for_distribution(engine: SimulationEngine, state_idx: int,
                                             sub_samples_per_side: int = 3) -> dict:
    """Grid-CELL-level Kriging/CGR validation for ONE forced demand_states
    distribution (see _force_synchronized_demand_state) at tilt=0 (matching
    phase4's own ground-truth reference condition) -- fresh Kriging/CGR
    estimators, fed EVERY real UE's own (location, RSRP) regardless of
    anchor status: this checks pure spatial extrapolation quality (sparse
    real UE positions -> full grid), nothing to do with the anchor/non-
    anchor overlay phase 4/5's own per-UE situations use.

    :param engine: a build_debug_engine() the CALLER owns and reuses across
        every state_idx it calls this with -- _force_synchronized_demand_state
        fully overwrites mobility state from scratch every call (no
        intervening per-step state, since callers always pass is_last=True
        to step_environment, skipping mobility.step()), so there is no
        need to rebuild a fresh engine (~115s each) per distribution
        (2026-09-20: was building its own fresh engine every call, ~3x
        more engine builds than necessary across phase 5 alone).

    Continuous-coverage-fraction validation against a continuous, AREA-
    AVERAGED ground truth (compute_ground_truth_grid_fraction) -- unlike a
    binary match-rate check, this isn't brittle to which exact spot inside
    a genuinely-mixed cell the comparison lands on (see this session's own
    traced mechanism: a narrow vertical-array antenna beam pattern,
    combined with cells close to a site sweeping several degrees of
    elevation angle across one 50m cell, caps ANY binary per-cell
    comparison at ~88.7% regardless of estimator quality). Both ground
    truth AND each estimator's own prediction are the average of
    sub_samples_per_side^2 sub-points spanning each cell's own footprint,
    thresholded individually then averaged -- scored via NMSE by the
    caller (see _nmse's own docstring).

    :output: dict -- observed_mask [num_bs, num_grid_cells] bool (real UE
        observed this interval); ground_truth, kriging, cgr [num_bs,
        num_grid_cells] continuous [0,1] coverage fraction.
    """
    _force_synchronized_demand_state(engine, state_idx)
    num_bs = engine.num_bs
    # 2026-10-13: was np.zeros(num_bs) (0 deg) -- same degenerate-coverage
    # bug as run_measurement_aggregation_scenario's own fix, see there.
    live_tilt_deg = np.full(num_bs, SINGLE_SECTOR_NOMINAL_TILT_DEG)

    spatial_grid_cell_size_m = load_config()["topology"]["spatial_grid_cell_size_m"]
    common = dict(num_bs=num_bs, num_grid_cells=engine.num_grid_cells, grid_points=engine.grid_points,
                 length_scale=spatial_grid_cell_size_m, coverage_threshold_db=engine.coverage_threshold_db,
                 radio_map_method="rsrp_per_sector", noise_power_w=engine.kpi_manager.noise_power_w,
                 topology=engine.topology)
    kriging_estimator = SpatialGridEstimator(**common, radio_map_estimator="kriging")
    cgr_estimator = SpatialGridEstimator(
        **common, radio_map_estimator="cgr", cgr_num_gaussian=engine.cgr_cfg.get("num_gaussian", 12),
        cgr_ridge_lambda=engine.cgr_cfg.get("ridge_lambda", 1e-4),
        cgr_baseline_method=engine.cgr_cfg.get("baseline_method", "global_mean"),
        cgr_baseline_length_scale_m=engine.cgr_cfg.get("baseline_length_scale_m", spatial_grid_cell_size_m),
        cgr_gaussian_config={k: v for k, v in engine.cgr_cfg.items()
                             if k not in ("num_gaussian", "ridge_lambda", "baseline_method",
                                         "baseline_length_scale_m")})

    state = engine.step_environment(True)
    ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
    sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
    power_w = engine.kpi_manager.resolve_power_at_tilt(state, live_tilt_deg)[0]
    kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
    covered_ue = kpis["sinr_db"][0] > engine.coverage_threshold_db
    power_db_all_sectors = 10.0 * np.log10(power_w)

    kriging_estimator.accumulate(sector_idx, cell_idx, covered_ue, power_db_all_sectors=power_db_all_sectors)
    cgr_estimator.accumulate(sector_idx, cell_idx, covered_ue, power_db_all_sectors=power_db_all_sectors)
    observed_mask = kriging_estimator.visit_count > 0

    ground_truth = compute_ground_truth_grid_fraction(engine, live_tilt_deg, sub_samples_per_side)

    sub_xy = _cell_sub_points(engine.topology, sub_samples_per_side)  # [num_bs, num_grid_cells, K, 2]
    pred_fractions = {}
    for name, est in (("kriging", kriging_estimator), ("cgr", cgr_estimator)):
        pred_fraction = np.zeros((num_bs, engine.num_grid_cells))
        for i in range(num_bs):
            known = est.visit_count[i] > 0
            unknown = ~known & est.estimable_cell_mask[i]
            # observed cells: real per-UE fraction, same definition ground truth uses.
            pred_fraction[i, known] = est.covered_count[i, known] / est.visit_count[i, known]
            if not unknown.any():
                continue
            sub_targets = sub_xy[i, unknown]  # [n_unknown, K, 2]
            n_unknown, K, _ = sub_targets.shape
            sub_targets_flat = sub_targets.reshape(-1, 2)

            predicted_rsrp_db = np.zeros((num_bs, sub_targets_flat.shape[0]))
            for b in range(num_bs):
                mean_rsrp_db_b = np.where(known, est.rsrp_db_sum[b, i] / np.where(known, est.visit_count[i], 1), 0.0)
                if name == "cgr":
                    context = est._cgr_context(b, i, live_tilt_deg[b])
                    pred = est.cgr.predict(est.grid_points[i, known], mean_rsrp_db_b[known],
                                           sub_targets_flat, context=context)
                else:
                    pred = est.kriging.predict(est.grid_points[i, known], mean_rsrp_db_b[known], sub_targets_flat)
                predicted_rsrp_db[b] = pred
            predicted_rsrp_w = np.power(10.0, predicted_rsrp_db / 10.0)
            serving_w = predicted_rsrp_w.max(axis=0)
            interference_w = predicted_rsrp_w.sum(axis=0) - serving_w
            predicted_sinr_db = 10.0 * np.log10(serving_w / (interference_w + engine.kpi_manager.noise_power_w))
            predicted_covered = (predicted_sinr_db > engine.coverage_threshold_db).astype(float)
            pred_fraction[i, unknown] = predicted_covered.reshape(n_unknown, K).mean(axis=1)
        pred_fractions[name] = pred_fraction

    return dict(observed_mask=observed_mask, ground_truth=ground_truth, **pred_fractions)


# =============================================================
# PHASE 4
# =============================================================

def phase5_grid_measurements_and_rme(engine: SimulationEngine, scenario: dict, out_dir: str) -> None:
    """Grid-CELL-level measurement sparsity, then RME (Kriging/CGR)
    validation against ground truth, then the phase-4-style per-region
    comparison extended with RME-based non-anchor region inference -- the
    full motivating story for radio map estimation in one phase:

    (1) a whole-network map, one PER DISTRIBUTION (the same [3,1,1]/
    [2,1,2]/[1,1,3] near/balanced/far-heavy conditions phase 4 uses, via
    _force_synchronized_demand_state -- changed 2026-09-20 from an
    arbitrary interval index into the big random-tilt scenario, for
    consistency with every other per-region comparison in this codebase),
    of which fine-grid cells got a real UE observation vs which didn't;

    (2) the fraction of a sector's own grid cells observed per interval,
    all 21 sectors over the WHOLE random-tilt run (unlike (1)/(3)/(4),
    this is inherently a full-timeline trend, not a single-condition
    snapshot, so it still reads from `scenario` un-changed) -- together,
    (1)/(2) are why Kriging is needed at all;

    (3) continuous coverage-FRACTION validation against ground truth, for
    the SAME 3 distributions as (1) (changed 2026-09-20, same reason) --
    compute_grid_validation_for_distribution scores a continuous, area-
    averaged per-cell coverage FRACTION via NMSE (see its own and _nmse's
    docstrings): ONE spatial-map figure (rme_validation_fraction_state_
    *.png, ground truth/Kriged/CGR fraction) and a SEPARATE spatial error-
    map figure (rme_validation_fraction_error_state_*.png, |predicted -
    true| fraction). Deliberately NEVER a rounded binary match/mismatch
    decision (deleted 2026-09-20) -- that was capped at ~88.7% regardless
    of estimator quality purely from real, physically-grounded within-cell
    antenna-pattern RSRP variation (a narrow vertical-array beam pattern,
    plus cells close to a site sweeping several degrees of elevation angle
    across one 50m cell), never actually discriminating Kriging/CGR
    quality. See compute_ground_truth_grid's own caveat -- this checks
    whether each estimator recovers the right large-scale coverage
    STRUCTURE, not sample-exact agreement.

    (4) phase 4's own ground_truth/anchor_only per-region comparison
    (occupancy/coverage/sinr/rsrp_distribution.png), extended with
    kriging_hard/kriging_soft/cgr_hard/cgr_soft situations -- can RME
    recover a non-anchor UE's own region well enough to beat simply
    excluding it (anchor_only)? A non-anchor UE's own real (SINR, RSRP,
    coverage) is ALREADY known (never estimated -- see
    RegionalInfoProvider's own module docstring); RME's only job is
    inferring which region to file that already-known value under, via
    SpatialGridEstimator.infer_region_probabilities (Kriging fit from
    anchor UEs only, or its CGR-analog). "hard" = argmax(region_probs), one
    full unit in one region, matching production's own 2026-09-20 design;
    "soft" = the raw probability vector as fractional weights across
    regions, for comparison against production's choice.

    (5) Kriging vs. CGR summary bar chart (rme_method_comparison.png,
    continuous coverage-fraction NMSE only, from (3)).
    """
    os.makedirs(out_dir, exist_ok=True)
    topo, grid_points = engine.topology, engine.grid_points
    xy = grid_points.reshape(-1, 2)

    # Distribution setup, shared by (1)/(3)/(4) below.
    region_names = OccupancyEstimator.region_names
    num_regions = len(region_names)
    num_trials, non_anchor_fraction = 1, 0.8

    # ONE dedicated engine (~115s to build), reused for EVERY distribution
    # below via _force_synchronized_demand_state -- which fully overwrites
    # mobility state from scratch every call, so a fresh engine per
    # distribution was never actually needed (2026-09-20: was ~7 separate
    # builds across this phase alone). Kept separate from the `engine`
    # PARAMETER (the shared, already-run scenario engine every phase in
    # this invocation receives) rather than mutating that one, since other
    # phases in the same run (e.g. --phase 5 6 7) expect it untouched.
    dedicated_engine = build_debug_engine()
    demand_states = dedicated_engine.mobility.demand_states
    state_labels = []
    for state_idx in range(demand_states.shape[0]):
        row = demand_states[state_idx]
        heavy_region = region_names[int(np.argmax(row))]
        state_labels.append(f"[{','.join(str(v) for v in row)}] ({heavy_region}-heavy)")

    # compute_grid_validation_for_distribution's own result is computed
    # ONCE per distribution here and reused by both (1) and (3) below (was
    # two separate calls per distribution until 2026-09-20, doubling this
    # section's own cost for nothing).
    grid_val_per_state = [compute_grid_validation_for_distribution(dedicated_engine, s)
                          for s in range(demand_states.shape[0])]

    # --- (1) whole-network observed/unobserved snapshot, one per distribution.
    fig, axes = plt.subplots(1, demand_states.shape[0], figsize=(10 * demand_states.shape[0], 9), squeeze=False)
    for state_idx, state_label in enumerate(state_labels):
        ax = axes[0, state_idx]
        draw_sites_and_sectors(ax, topo)
        flat_mask = grid_val_per_state[state_idx]["observed_mask"].reshape(-1)
        ax.scatter(xy[~flat_mask, 0], xy[~flat_mask, 1], s=4, c="lightgray", linewidths=0, zorder=1,
                  label="unobserved")
        ax.scatter(xy[flat_mask, 0], xy[flat_mask, 1], s=10, c="#1baf7a", linewidths=0, zorder=2, label="observed")
        ax.set_title(f"{state_label}\n({int(flat_mask.sum())}/{flat_mask.size} cells = {flat_mask.mean():.1%})",
                    fontsize=10)
        ax.set_aspect("equal")
        if state_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.savefig(os.path.join(out_dir, "observed_cells_snapshot.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- (2) observed-fraction heatmap over the WHOLE random-tilt run (a
    # full-timeline trend, unlike (1)/(3)/(4) -- still reads from `scenario`).
    observed_mask = scenario["observed_mask"]  # [num_intervals, num_bs, num_grid_cells]
    fraction_observed = observed_mask.mean(axis=2)  # [num_intervals, num_bs]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(fraction_observed.T, aspect="auto", cmap=DENSITY_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax.axvline(NUM_NO_TILT - 0.5, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("interval")
    ax.set_ylabel("sector")
    ax.set_title("Fraction of a sector's own grid cells observed per interval")
    fig.colorbar(im, ax=ax, label="fraction observed")
    fig.savefig(os.path.join(out_dir, "observed_fraction_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ({fraction_observed.mean():.1%} mean cells observed/interval)")

    # --- (3) RME (Kriging/CGR) validation against ground truth -- continuous
    # coverage FRACTION only (NMSE), never a rounded binary match/mismatch
    # decision (deleted 2026-09-20 -- capped at ~88.7% regardless of
    # estimator quality purely from real, physically-grounded within-cell
    # antenna-pattern RSRP variation, not estimator error, so it never
    # actually discriminated Kriging/CGR quality -- see NMSE's own
    # docstring for the real, physically-meaningful check instead).
    nmse_records = []  # pooled for the summary bar chart at the end (5)
    for state_idx, state_label in enumerate(state_labels):
        grid_val = grid_val_per_state[state_idx]
        true_fraction = grid_val["ground_truth"].reshape(-1)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        panels = [(true_fraction, "Ground truth coverage fraction (area-averaged, sub-sampled)"),
                 (grid_val["kriging"].reshape(-1), "Kriged coverage fraction"),
                 (grid_val["cgr"].reshape(-1), "CGR coverage fraction")]
        for ax, (values, title) in zip(axes, panels):
            draw_sites_and_sectors(ax, engine.topology)
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=8, c=values, cmap=COLORBAR_CMAP, vmin=0, vmax=1,
                            linewidths=0, zorder=2)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            fig.colorbar(sc, ax=ax, fraction=0.03, label="coverage fraction (continuous, 0-1)")
        fig.suptitle(f"Continuous coverage-fraction validation, {state_label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rme_validation_fraction_state_{state_idx}.png"), dpi=150,
                   bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, method_label in zip(axes, ("kriging", "cgr")):
            pred_fraction = grid_val[method_label].reshape(-1)
            this_nmse = _nmse(pred_fraction, true_fraction)
            draw_sites_and_sectors(ax, engine.topology)
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=8, c=np.abs(pred_fraction - true_fraction), cmap=COLORBAR_CMAP,
                            vmin=0, vmax=1, linewidths=0, zorder=2)
            ax.set_title(f"{method_label.capitalize()} vs. ground truth (NMSE={this_nmse:.3f})", fontsize=10)
            ax.set_aspect("equal")
            fig.colorbar(sc, ax=ax, fraction=0.03, label="|predicted - true| coverage fraction")
            print(f"  {state_label} {method_label} continuous coverage-fraction NMSE: {this_nmse:.4f}")
            nmse_records.append((method_label, this_nmse))
        fig.suptitle(f"Continuous coverage-fraction error, {state_label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rme_validation_fraction_error_state_{state_idx}.png"), dpi=150,
                   bbox_inches="tight")
        plt.close(fig)

    # --- (4) phase 4's own per-region comparison, extended with RME-based
    # non-anchor region inference -- can Kriging/CGR beat simply excluding
    # a non-anchor UE (anchor_only)? Same 3-distribution/num_trials=1/
    # non_anchor_fraction=0.8 convention as phase4_measurement_aggregation
    # itself (see its own docstring for why num_trials=1 is deliberate).
    all_results, situations_per_state = [], []
    for state_idx, state_label in enumerate(state_labels):
        print(f"[phase5] distribution {state_idx}: {state_label} ({num_trials} trials)")

        situations, pooled_result = run_multi_trial_aggregation_rme(
            dedicated_engine, state_idx, non_anchor_fraction, num_trials, num_regions)
        all_results.append(pooled_result)
        situations_per_state.append(situations)
        for sit in _RME_SITUATION_ORDER:
            print(f"  {sit}: occupancy={situations[sit]['occupancy'].round(3).tolist()} "
                 f"coverage={np.round(situations[sit]['coverage'], 3).tolist()} "
                 f"sinr={np.round(situations[sit]['sinr'], 2).tolist()} "
                 f"rsrp={np.round(situations[sit]['rsrp'], 1).tolist()}")

    _metric_histogram_grid_figure(
        out_dir, "sinr_distribution.png", "SINR by region: ground truth vs. anchor-only vs. RME-inferred",
        "sinr", "SINR (dB)", region_names, state_labels, all_results, situations_per_state, num_regions,
        bin_range=(-10, 25), situation_order=_RME_SITUATION_ORDER)
    _metric_histogram_grid_figure(
        out_dir, "rsrp_distribution.png", "RSRP by region: ground truth vs. anchor-only vs. RME-inferred",
        "rsrp", "RSRP (dBm)", region_names, state_labels, all_results, situations_per_state, num_regions,
        bin_range=(-50, -20), situation_order=_RME_SITUATION_ORDER)
    _occupancy_figure(out_dir, region_names, state_labels, situations_per_state, num_regions,
                      situation_order=_RME_SITUATION_ORDER,
                      title_suffix="ground truth vs. anchor-only vs. RME-inferred")
    _coverage_figure(out_dir, region_names, state_labels, situations_per_state, num_regions,
                     situation_order=_RME_SITUATION_ORDER,
                     title_suffix="ground truth vs. anchor-only vs. RME-inferred")

    # --- (5) Kriging vs. CGR summary: which method is actually better,
    # one bar chart instead of reading printed NMSE lines, averaged across
    # representative_intervals.
    methods = ["Kriged", "CGR"]
    nmse_method_key = {"Kriged": "kriging", "CGR": "cgr"}
    nmse_means = [np.mean([n for meth, n in nmse_records if meth == nmse_method_key[method]]) for method in methods]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(methods, nmse_means, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("mean NMSE (lower is better)")
    ax.set_title("Kriging vs. CGR -- continuous coverage-fraction NMSE", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rme_method_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Phase 5 done -- saved to {out_dir}/")


_SITUATION_ORDER = ["ground_truth", "anchor_only"]
_SITUATION_COLORS = {"ground_truth": "#333333", "anchor_only": "#d7191c",
                     "kriging_hard": "#1a9850", "kriging_soft": "#91cf60",
                     "cgr_hard": "#4575b4", "cgr_soft": "#91bfdb"}
_SITUATION_LABELS = {"ground_truth": "ground truth (all real UEs, known)",
                     "anchor_only": "anchor-only (raw, uninterpolated)",
                     "kriging_hard": "Kriging-inferred region (hard)",
                     "kriging_soft": "Kriging-inferred region (soft)",
                     "cgr_hard": "CGR-inferred region (hard)",
                     "cgr_soft": "CGR-inferred region (soft)"}
# phase 5's own extended order (ground_truth/anchor_only + RME-inferred
# non-anchor region assignment, both hard-argmax and soft/fractional --
# see run_measurement_aggregation_scenario_rme/_rme_situations_for).
_RME_SITUATION_ORDER = ["ground_truth", "anchor_only", "kriging_hard", "kriging_soft", "cgr_hard", "cgr_soft"]


def _region_pooled_stats(region_idx: np.ndarray, sinr_db: np.ndarray, rsrp_db: np.ndarray,
                         covered: np.ndarray, num_regions: int) -> dict:
    """Per-region (occupancy, coverage, sinr, rsrp) means for one situation
    -- region_idx is each observation's own region (near=0/mid=1/far=2).
    NaN for a region with zero members (nothing to average). Each array is
    length num_regions+1 -- index num_regions ("whole") is the SAME metric
    pooled over every observation passed in, regardless of region (not a
    4th real region -- just the whole-area aggregate of whichever
    population this call was already given, e.g. anchor_only's own "whole"
    pools only anchor UEs, matching that situation's own already-filtered
    near/mid/far entries). occupancy's own "whole" is trivially 1.0 (the
    whole population IS 100% of itself) -- kept anyway so the figure grid
    stays rectangular and so it visually cross-checks that near+mid+far
    occupancy sums to 1.0.
    """
    return dict(
        occupancy=np.array([(region_idx == r).mean() for r in range(num_regions)] + [1.0]),
        coverage=np.array([covered[region_idx == r].mean() if (region_idx == r).any() else np.nan
                           for r in range(num_regions)]
                          + [covered.mean() if len(covered) else np.nan]),
        sinr=np.array([sinr_db[region_idx == r].mean() if (region_idx == r).any() else np.nan
                      for r in range(num_regions)]
                     + [sinr_db.mean() if len(sinr_db) else np.nan]),
        rsrp=np.array([rsrp_db[region_idx == r].mean() if (region_idx == r).any() else np.nan
                      for r in range(num_regions)]
                     + [rsrp_db.mean() if len(rsrp_db) else np.nan]),
    )


def _situations_for(result: dict, num_regions: int) -> dict:
    """ground_truth (every real UE) vs anchor_only (anchor UEs' own real
    values, raw, no filling-in for non-anchor UEs) -- see
    run_measurement_aggregation_scenario's own docstring."""
    region_true = result["ue_region_true"]
    sinr_db, rsrp_db = result["ue_sinr_db"], result["ue_rsrp_db"]
    covered = result["ue_covered"].astype(float)
    is_anchor = result["ue_is_anchor"]
    return {
        "ground_truth": _region_pooled_stats(region_true, sinr_db, rsrp_db, covered, num_regions),
        "anchor_only": _region_pooled_stats(region_true[is_anchor], sinr_db[is_anchor], rsrp_db[is_anchor],
                                            covered[is_anchor], num_regions),
    }


def _region_weighted_stats(weights: np.ndarray, sinr_db: np.ndarray, rsrp_db: np.ndarray,
                           covered: np.ndarray, num_regions: int) -> dict:
    """Per-region (occupancy, coverage, sinr, rsrp) means for one situation,
    generalizing _region_pooled_stats to FRACTIONAL per-UE region weights
    (weights[u, r] = how much of UE u's own real, already-known value
    counts toward region r, each row summing to 1) instead of one hard
    region per UE -- a hard assignment is just the special case where
    weights is one-hot (see _rme_situations_for's own argmax)."""
    total_weight = weights.sum(axis=0)  # [num_regions]
    total = total_weight.sum()
    occupancy = total_weight / total if total > 0 else np.full(num_regions, np.nan)

    def wmean(values):
        with np.errstate(invalid="ignore"):
            return np.where(total_weight > 0,
                            (weights * values[:, None]).sum(axis=0) / np.where(total_weight > 0, total_weight, 1),
                            np.nan)
    return dict(occupancy=occupancy, coverage=wmean(covered), sinr=wmean(sinr_db), rsrp=wmean(rsrp_db))


def _rme_situations_for(result: dict, num_regions: int) -> dict:
    """_situations_for's own ground_truth/anchor_only, PLUS
    kriging_hard/kriging_soft/cgr_hard/cgr_soft -- every real UE's own
    already-known (SINR, RSRP, coverage) value, bucketed by region: real
    region for an anchor UE, RME-INFERRED region for a non-anchor UE (see
    run_measurement_aggregation_scenario_rme's own docstring for what
    ue_region_probs_kriging/ue_region_probs_cgr mean). "hard" =
    argmax(region_probs), one full unit in one region (production's own
    2026-09-20 design); "soft" = the raw probability vector as fractional
    weights across regions."""
    situations = _situations_for(result, num_regions)
    sinr_db, rsrp_db = result["ue_sinr_db"], result["ue_rsrp_db"]
    covered = result["ue_covered"].astype(float)
    for method, probs_key in (("kriging", "ue_region_probs_kriging"), ("cgr", "ue_region_probs_cgr")):
        probs = result[probs_key]
        situations[f"{method}_soft"] = _region_weighted_stats(probs, sinr_db, rsrp_db, covered, num_regions)
        situations[f"{method}_hard"] = _region_pooled_stats(probs.argmax(axis=1), sinr_db, rsrp_db, covered,
                                                             num_regions)
    return situations


_SITUATION_SHORT_LABELS = {"ground_truth": "ground\ntruth", "anchor_only": "anchor\nonly",
                          "kriging_hard": "Kriging\nhard", "kriging_soft": "Kriging\nsoft",
                          "cgr_hard": "CGR\nhard", "cgr_soft": "CGR\nsoft"}


def _metric_histogram_grid_figure(out_dir: str, fname: str, title: str, metric: str, xlabel: str,
                                  region_names, state_labels, all_results, situations_per_state,
                                  num_regions: int, bin_range: tuple, situation_order=_SITUATION_ORDER) -> None:
    """One figure, grid = distributions (rows) x regions (cols): overlaid
    histograms of ground_truth vs. anchor_only (whole population in that
    region vs. anchor UEs in that region), each situation's MEAN marked as
    a vertical dashed line (in the histogram itself -- no separate bar
    panel for "where's the center", per 2026-09-20 redesign). Histogram
    BARS are always just ground_truth/anchor_only (a UE's own real value
    doesn't change across situations, only which region it's filed under
    does, so a third/fourth histogram would just be a re-sliced view of
    the SAME values already shown -- situation_order beyond those two only
    adds its own mean line, e.g. phase 5's kriging/cgr situations)."""
    values_key = f"ue_{metric}_db" if metric in ("sinr", "rsrp") else f"ue_{metric}"
    num_states = len(state_labels)
    columns = region_names + ("whole",)
    num_cols = len(columns)
    fig, axes = plt.subplots(num_states, num_cols, figsize=(5.5 * num_cols, 4 * num_states), squeeze=False)
    bins = np.linspace(bin_range[0], bin_range[1], 30)
    for row, (result, situations, state_label) in enumerate(zip(all_results, situations_per_state, state_labels)):
        region_true = result["ue_region_true"]
        is_anchor = result["ue_is_anchor"]
        values = result[values_key]
        anchor_region, anchor_values = region_true[is_anchor], values[is_anchor]
        for col in range(num_cols):
            ax = axes[row, col]
            is_whole = col == num_regions
            gt_vals_col = values if is_whole else values[region_true == col]
            anchor_vals_col = anchor_values if is_whole else anchor_values[anchor_region == col]
            ax.hist(gt_vals_col, bins=bins, density=True, alpha=0.4,
                   label="ground truth", color=_SITUATION_COLORS["ground_truth"])
            if len(anchor_vals_col):
                ax.hist(anchor_vals_col, bins=bins, density=True, alpha=0.4,
                       label="anchor-only", color=_SITUATION_COLORS["anchor_only"])
            for sit in situation_order:
                mean_val = situations[sit][metric][col]
                if np.isfinite(mean_val):
                    ax.axvline(mean_val, color=_SITUATION_COLORS[sit], linestyle="--", linewidth=1.5)
            gt_mean, anchor_mean = situations["ground_truth"][metric][col], situations["anchor_only"][metric][col]
            if np.isfinite(gt_mean) and np.isfinite(anchor_mean):
                ax.text(0.97, 0.95, f"Δ={anchor_mean - gt_mean:+.2f}", transform=ax.transAxes,
                       ha="right", va="top", fontsize=8)
            if row == 0:
                ax.set_title(columns[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(state_label, fontsize=9)
            if row == num_states - 1:
                ax.set_xlabel(xlabel, fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _occupancy_figure(out_dir: str, region_names, state_labels, situations_per_state, num_regions: int,
                      situation_order=_SITUATION_ORDER, title_suffix: str = "ground truth vs. anchor-only") -> None:
    """Occupancy has no per-UE continuous value to histogram (a UE's own
    region membership is a single fact, not a spread) -- shown instead as
    grouped bars (fraction of UEs per region, one bar per situation), one
    subplot per distribution. Saved as occupancy_by_region.png (renamed
    2026-10-13 from occupancy_distribution.png -- it was never actually a
    distribution plot either, same reasoning as coverage_by_region.png)."""
    num_states = len(state_labels)
    num_sit = len(situation_order)
    columns = region_names + ("whole",)
    num_cols = len(columns)
    fig, axes = plt.subplots(1, num_states, figsize=(6 * num_states, 5), squeeze=False)
    x = np.arange(num_cols)
    width = 0.8 / num_sit
    for col, (situations, state_label) in enumerate(zip(situations_per_state, state_labels)):
        ax = axes[0, col]
        for i, sit in enumerate(situation_order):
            ax.bar(x + (i - (num_sit - 1) / 2) * width, situations[sit]["occupancy"], width,
                  label=_SITUATION_LABELS[sit], color=_SITUATION_COLORS[sit])
        if "ground_truth" in situation_order and "anchor_only" in situation_order:
            for c in range(num_cols):
                gt, anc = situations["ground_truth"]["occupancy"][c], situations["anchor_only"]["occupancy"][c]
                if np.isfinite(gt) and np.isfinite(anc):
                    ax.text(x[c], max(gt, anc) + 0.02, f"Δ={anc - gt:+.2f}",
                           ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(columns)
        ax.set_ylabel("fraction of UEs")
        ax.set_title(state_label, fontsize=10)
        ax.set_ylim(0, 1.15)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Occupancy: {title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "occupancy_by_region.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coverage_figure(out_dir: str, region_names, state_labels, situations_per_state, num_regions: int,
                     situation_order=_SITUATION_ORDER, title_suffix: str = "ground truth vs. anchor-only") -> None:
    """Coverage is binary per UE -- just an average of 0/1 values, shown as
    grouped bars (P(covered) per situation), NOT a histogram/distribution
    (there's no shape to show for a binary quantity) -- same state x region
    grid layout as the continuous-metric histograms, for visual consistency
    only. Saved as coverage_by_region.png (renamed 2026-10-13 from
    coverage_distribution.png -- it was never actually a distribution
    plot)."""
    num_states = len(state_labels)
    columns = region_names + ("whole",)
    num_cols = len(columns)
    fig, axes = plt.subplots(num_states, num_cols, figsize=(4 * num_cols, 3.5 * num_states), squeeze=False)
    for row, (situations, state_label) in enumerate(zip(situations_per_state, state_labels)):
        for col in range(num_cols):
            ax = axes[row, col]
            vals = [situations[sit]["coverage"][col] for sit in situation_order]
            ax.bar(range(len(situation_order)), vals, color=[_SITUATION_COLORS[s] for s in situation_order])
            if "ground_truth" in situation_order and "anchor_only" in situation_order:
                gt, anc = situations["ground_truth"]["coverage"][col], situations["anchor_only"]["coverage"][col]
                if np.isfinite(gt) and np.isfinite(anc):
                    ax.text(0.97, 0.95, f"Δ={anc - gt:+.3f}", transform=ax.transAxes,
                           ha="right", va="top", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(situation_order)))
            ax.set_xticklabels([_SITUATION_SHORT_LABELS[s] for s in situation_order], fontsize=8)
            if row == 0:
                ax.set_title(columns[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(state_label + "\ncoverage fraction", fontsize=8)
    fig.suptitle(f"Coverage (mean of per-UE 0/1 values): {title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coverage_by_region.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ensure_min_one_anchor_per_region(region_true: np.ndarray, is_anchor: np.ndarray, num_regions: int,
                                      seed: int = 0) -> np.ndarray:
    """Guarantees every region has >=1 anchor observation. At the stress-
    test fractions this phase now uses (50-90% non-anchor), a region could
    otherwise end up with ZERO anchor UEs purely by chance, making its
    coverage/SINR/RSRP mean undefined (NaN) instead of just noisy/small-
    sample. Promotes one randomly-picked (interval, UE) observation
    already known to be in that region to anchor -- doesn't need to touch
    every interval of that UE's own run, just enough to give the region
    something to average (2026-09-20).
    """
    is_anchor = is_anchor.copy()
    rng = np.random.default_rng(seed)
    for r in range(num_regions):
        if not (is_anchor & (region_true == r)).any():
            candidates = np.where(region_true == r)[0]
            if candidates.size > 0:
                is_anchor[rng.choice(candidates)] = True
    return is_anchor


def run_multi_trial_aggregation(engine: SimulationEngine, state_idx: int, non_anchor_fraction: float,
                                num_trials: int, num_regions: int, anchor_override_fn=None) -> tuple:
    """num_trials INDEPENDENT single-interval draws of the SAME
    demand_states distribution -- fresh mobility positions AND a fresh
    random anchor selection EVERY trial (_force_synchronized_demand_state
    re-randomizes both each call: mobility's own torch.Generator advances
    statefully, and mask_seed=trial varies the anchor draw) -- then
    averages each trial's OWN already-computed (ground_truth, anchor_only)
    region-level statistic.

    This is deliberately NOT the same as pooling num_trials intervals'
    raw UE data into one big sample before computing the statistic once
    (that was run_measurement_aggregation_scenario's own num_intervals=15
    behavior until 2026-09-20, and it artificially inflated the effective
    sample size, masking the real anchor-only effect). Averaging
    trial-level statistics keeps each trial's own small (single-interval)
    sample size intact -- a real, persistent bias across trials shows up
    clearly in the average; pure per-trial noise shrinks toward zero
    instead, which is the correct way to separate the two.

    :output: (situations_avg -- same shape as _situations_for's own
        output, each entry the MEAN across num_trials trial-level values;
        pooled_result -- raw per-UE arrays concatenated across all trials,
        for histogram SHAPE only, e.g. _metric_histogram_grid_figure --
        never used here to compute a summary statistic).

    :param anchor_override_fn: optional (engine, state_idx, trial) -> None,
        called right after _force_synchronized_demand_state each trial,
        e.g. a _force_fixed_anchors_per_sector_region closure -- overrides
        engine.location_reporting_mask with a non-fractional anchor rule
        (2026-10-13). When set, _ensure_min_one_anchor_per_region's own
        floor-of-1-anchor-per-region safety net is SKIPPED -- a region
        with deliberately ZERO anchor clusters (e.g. anchors_per_region=
        [0,1,1]) is the whole point of that override, not an accidental
        edge case to silently patch over.
    """
    trial_situations = []
    pooled = {k: [] for k in ("ue_region_true", "ue_sinr_db", "ue_rsrp_db", "ue_covered", "ue_is_anchor")}
    result = None
    for trial in range(num_trials):
        _force_synchronized_demand_state(engine, state_idx, non_anchor_fraction=non_anchor_fraction,
                                         mask_seed=trial)
        if anchor_override_fn is not None:
            anchor_override_fn(engine, state_idx, trial)
        result = run_measurement_aggregation_scenario(engine, num_intervals=1)
        if anchor_override_fn is None:
            result["ue_is_anchor"] = _ensure_min_one_anchor_per_region(
                result["ue_region_true"], result["ue_is_anchor"], num_regions)
        trial_situations.append(_situations_for(result, num_regions))
        for k in pooled:
            pooled[k].append(result[k])

    situations_avg = {}
    for sit in _SITUATION_ORDER:
        situations_avg[sit] = {}
        for metric in ("occupancy", "coverage", "sinr", "rsrp"):
            stacked = np.stack([ts[sit][metric] for ts in trial_situations])  # [num_trials, num_regions]
            situations_avg[sit][metric] = np.nanmean(stacked, axis=0)

    pooled_result = {k: np.concatenate(v) for k, v in pooled.items()}
    pooled_result["num_ut"] = result["num_ut"]
    pooled_result["num_intervals"] = num_trials
    return situations_avg, pooled_result


def run_multi_trial_aggregation_rme(engine: SimulationEngine, state_idx: int, non_anchor_fraction: float,
                                    num_trials: int, num_regions: int) -> tuple:
    """run_multi_trial_aggregation's own trial-averaging design (see its
    docstring), but using run_measurement_aggregation_scenario_rme/
    _rme_situations_for -- adds kriging_hard/kriging_soft/cgr_hard/cgr_soft
    to every trial's own situations before averaging.
    """
    trial_situations = []
    pooled = {k: [] for k in ("ue_region_true", "ue_sinr_db", "ue_rsrp_db", "ue_covered", "ue_is_anchor",
                              "ue_region_probs_kriging", "ue_region_probs_cgr")}
    result = None
    for trial in range(num_trials):
        _force_synchronized_demand_state(engine, state_idx, non_anchor_fraction=non_anchor_fraction,
                                         mask_seed=trial)
        result = run_measurement_aggregation_scenario_rme(engine, num_intervals=1)
        result["ue_is_anchor"] = _ensure_min_one_anchor_per_region(
            result["ue_region_true"], result["ue_is_anchor"], num_regions)
        trial_situations.append(_rme_situations_for(result, num_regions))
        for k in pooled:
            pooled[k].append(result[k])

    situations_avg = {}
    for sit in _RME_SITUATION_ORDER:
        situations_avg[sit] = {}
        for metric in ("occupancy", "coverage", "sinr", "rsrp"):
            stacked = np.stack([ts[sit][metric] for ts in trial_situations])  # [num_trials, num_regions]
            situations_avg[sit][metric] = np.nanmean(stacked, axis=0)

    pooled_result = {k: np.concatenate(v) for k, v in pooled.items()}
    pooled_result["num_ut"] = result["num_ut"]
    pooled_result["num_intervals"] = num_trials
    return situations_avg, pooled_result


def _non_anchor_percentage_sweep_figure(out_dir: str, region_names, num_regions: int, engine: SimulationEngine,
                                        state_idx: int, state_label: str, fname: str, num_trials: int,
                                        fractions=(0.5, 0.6, 0.7, 0.8, 0.9)) -> None:
    """How anchor-only coverage AND SINR vary with the non-anchor fraction
    (50-90%), each point averaged over num_trials independent single-
    interval draws (see run_multi_trial_aggregation) -- NOT one trial per
    point (that was noisy enough to obscure any real trend, 2026-09-20).
    Ground truth is its own num_trials-average too (non_anchor_fraction
    doesn't affect it, but real UE positions still vary trial to trial),
    computed once, reused as the flat reference line for every fraction.
    """
    # SINR's fixed y-pad (not auto-scaled per subplot, unlike coverage's
    # natural 0-1 bound): auto-scaling would zoom into a sub-0.2dB window
    # and make ordinary sampling noise look like a real effect -- +-2dB is
    # a physically meaningful scale to judge the wobble against.
    metric_specs = [("coverage", "coverage fraction", "fixed01"), ("sinr", "mean SINR (dB)", "pad2")]
    columns = region_names + ("whole",)
    num_cols = len(columns)

    gt_situations, _ = run_multi_trial_aggregation(engine, state_idx, fractions[0], num_trials, num_regions)
    gt_vals = {m: gt_situations["ground_truth"][m] for m, _, _ in metric_specs}

    anchor_vals = {m: np.zeros((len(fractions), num_cols)) for m, _, _ in metric_specs}
    for i, frac in enumerate(fractions):
        situations, _ = run_multi_trial_aggregation(engine, state_idx, frac, num_trials, num_regions)
        for m, _, _ in metric_specs:
            anchor_vals[m][i] = situations["anchor_only"][m]

    fig, axes = plt.subplots(len(metric_specs), num_cols, figsize=(5.5 * num_cols, 4.5 * len(metric_specs)),
                             squeeze=False)
    x_pct = [f * 100 for f in fractions]
    for row, (metric, ylabel, ylim_mode) in enumerate(metric_specs):
        for r in range(num_cols):
            ax = axes[row, r]
            ax.axhline(gt_vals[metric][r], color=_SITUATION_COLORS["ground_truth"], linestyle="--",
                      linewidth=1.5, label="ground truth")
            ax.plot(x_pct, anchor_vals[metric][:, r], "-o", color=_SITUATION_COLORS["anchor_only"],
                   label="anchor-only")
            ax.set_xlabel("non-anchor UEs (%)")
            ax.set_ylabel(ylabel)
            if ylim_mode == "fixed01":
                ax.set_ylim(0, 1)
            elif ylim_mode == "pad2" and np.isfinite(gt_vals[metric][r]):
                ax.set_ylim(gt_vals[metric][r] - 2.0, gt_vals[metric][r] + 2.0)
            # else: region has zero UEs under the committed pure demand split
            # (e.g. state 1=[10,0,0] has no mid/far UEs at all) -- gt_vals is
            # NaN, let matplotlib auto-scale instead of crashing on NaN limits.
            if row == 0:
                ax.set_title(columns[r], fontsize=11)
            if row == 0 and r == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Anchor-only coverage/SINR vs. non-anchor percentage "
                f"({num_trials}-trial average per point) -- {state_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def phase4_measurement_aggregation(engine: SimulationEngine, scenario: dict, out_dir: str,
                                   num_trials: int = 1,
                                   non_anchor_fraction: float = 0.8,
                                   anchor_override_fn=None) -> None:
    """Two situations, per demand_states distribution (each sector forced
    into the SAME state -- see _force_synchronized_demand_state):
      1) ground_truth -- every real UE's own true region/SINR/RSRP/
         coverage, known here only because this is the simulator (a real
         deployment could never know this for non-anchor UEs).
      2) anchor_only -- the SAME quantities, but computed ONLY from anchor
         (location-reporting) UEs, at face value, no interpolation:
         occupancy undercounts (non-anchor UEs invisible), coverage is
         silent on non-anchor UEs.

    RME/radio-map-assisted reconstruction is OUT OF SCOPE here (deferred
    2026-09-20 pending a region_inference_sigma_db recalibration -- see
    phase 5 for Kriging/CGR's own validation instead). Anchor assignment
    is i.i.d. per-UE random (_randomize_location_reporting_mask), not
    production's per-cluster-uniform rule -- see that function's own
    docstring for why (the per-cluster rule guarantees an anchor in every
    cluster at any realistic reporting fraction, making anchor-only look
    unrealistically good). Whole-CLUSTER random exclusion was also tried
    and dropped (2026-09-20) -- everything here is per-UE random now.

    Every number here goes through run_multi_trial_aggregation, which
    supports averaging num_trials INDEPENDENT single-interval draws (fresh
    mobility positions + fresh anchor selection each trial) rather than
    pooling their raw UE data into one big sample first (that was this
    phase's own original design, num_intervals=15 -- it artificially
    inflated the effective sample size and masked the real anchor-only
    effect entirely). num_trials=1 (the current default, 2026-09-20) is a
    DELIBERATE choice, not a fallback: production's own real estimate is
    also always exactly one noisy single-interval draw, never an average
    of many, so num_trials=1 is the more honest depiction of what the
    system actually experiences moment-to-moment. Set num_trials>1 instead
    to separate a persistent bias (survives averaging) from pure
    draw-to-draw noise (shrinks toward zero) -- a different, complementary
    question from "how bad does any one real moment actually look".

    :param non_anchor_fraction: applied to the main 3-distribution
        comparison (not just the percentage-sweep figure) -- locked to 0.8
        (2026-09-20), within the sweep's own 50-90% range. Not config.yaml's
        own location_reporting_fraction.

    Evaluated at two levels (2026-10-13):
      (1) Spatial-state accuracy -- occupancy_by_region.png and
          coverage_by_region.png (grouped bars of ground_truth vs.
          anchor_only, each region's own |anchor-ground_truth| gap labeled
          in-plot -- occupancy is inherently categorical, a UE's region IS
          the category, not a value to histogram; coverage is just a mean
          of per-UE 0/1 values, also bars, never a histogram); sinr_
          distribution.png/rsrp_distribution.png (a distribution x region
          grid, histogram SHAPE pooled from all num_trials trials -- for
          visualization only -- vs. ground_truth, mean marked in-plot using
          the CORRECT trial-averaged value, not the pooled histogram's own
          mean, gap between the two means labeled in-plot too); and
          non_anchor_percentage_sweep.png (coverage AND SINR vs. non-anchor
          % 50-90, one representative distribution, each point its own
          num_trials-trial average). Every one of these four figures also
          carries a "whole" column alongside near/mid/far (region_names +
          ("whole",)) -- pooled over whichever population that situation
          already saw (so anchor_only's own "whole" only ever pools anchor
          UEs, never silently reusing ground-truth-only values for the
          region split it couldn't actually see).
      (2) Tilt-decision accuracy -- tilt_decision_accuracy.png (see
          _tilt_decision_accuracy_figure): ground_truth vs. anchor_only
          coverage-vs-tilt curves overlaid per region + whole, each curve's
          own argmax tilt marked, and the REAL coverage loss from trusting
          anchor-only's pick (ground-truth curve evaluated at anchor-only's
          optimal tilt vs. at ground-truth's own optimal tilt) marked with
          an arrow on the ground-truth curve itself, plus written out to
          tilt_decision_accuracy_summary.txt.

    `scenario` is unused -- builds its OWN dedicated engine, reused across
    every distribution AND the percentage sweep below (needs_scenario=False
    in PHASES, same convention as phase 2; a single reused engine rather
    than one per distribution + a separate sweep engine, 2026-09-20 -- see
    compute_grid_validation_for_distribution's own docstring for why this
    is safe).
    """
    os.makedirs(out_dir, exist_ok=True)
    region_names = OccupancyEstimator.region_names
    num_regions = len(region_names)

    dedicated_engine = build_debug_engine()
    demand_states = dedicated_engine.mobility.demand_states

    state_labels, all_results, situations_per_state = [], [], []
    for state_idx in range(demand_states.shape[0]):
        row = demand_states[state_idx]
        heavy_region = region_names[int(np.argmax(row))]
        state_label = f"[{','.join(str(v) for v in row)}] ({heavy_region}-heavy)"
        state_labels.append(state_label)
        print(f"[phase4] distribution {state_idx}: {state_label} ({num_trials} trials)")

        situations, pooled_result = run_multi_trial_aggregation(
            dedicated_engine, state_idx, non_anchor_fraction, num_trials, num_regions,
            anchor_override_fn=anchor_override_fn)
        all_results.append(pooled_result)
        situations_per_state.append(situations)
        for sit in _SITUATION_ORDER:
            print(f"  {sit}: occupancy={situations[sit]['occupancy'].round(3).tolist()} "
                 f"coverage={np.round(situations[sit]['coverage'], 3).tolist()} "
                 f"sinr={np.round(situations[sit]['sinr'], 2).tolist()} "
                 f"rsrp={np.round(situations[sit]['rsrp'], 1).tolist()}")

    all_sinr = np.concatenate([r["ue_sinr_db"] for r in all_results])
    all_rsrp = np.concatenate([r["ue_rsrp_db"] for r in all_results])
    _metric_histogram_grid_figure(out_dir, "sinr_distribution.png", "SINR: ground truth vs. anchor-only",
                                  "sinr", "SINR (dB)", region_names, state_labels, all_results,
                                  situations_per_state, num_regions,
                                  (min(all_sinr.min(), -10.0), max(all_sinr.max(), 15.0)))
    _metric_histogram_grid_figure(out_dir, "rsrp_distribution.png", "RSRP: ground truth vs. anchor-only",
                                  "rsrp", "RSRP (dBm)", region_names, state_labels, all_results,
                                  situations_per_state, num_regions,
                                  (all_rsrp.min() - 2.0, all_rsrp.max() + 2.0))
    _occupancy_figure(out_dir, region_names, state_labels, situations_per_state, num_regions)
    _coverage_figure(out_dir, region_names, state_labels, situations_per_state, num_regions)

    # Percentage sweep sweeps non_anchor_fraction itself -- meaningless
    # under a fixed-cluster anchor_override_fn (a different, non-fractional
    # anchor rule), so skip it in that mode rather than produce a figure
    # that silently ignores the override.
    if anchor_override_fn is None:
        # the "[2,1,2] balanced" distribution (index 1) as the
        # representative middle-of-the-road case, not near/far-heavy extremes.
        sweep_state_idx = 1 if demand_states.shape[0] > 1 else 0
        _non_anchor_percentage_sweep_figure(out_dir, region_names, num_regions, dedicated_engine, sweep_state_idx,
                                            state_labels[sweep_state_idx], "non_anchor_percentage_sweep.png",
                                            num_trials)

    # (2) Tilt-decision accuracy -- reuses dedicated_engine (safe: every
    # trial re-forces demand state from scratch via _force_synchronized_
    # demand_state, same pattern run_multi_trial_aggregation already uses
    # repeatedly above, no stale per-step state carried across calls).
    _tilt_decision_accuracy_figure(out_dir, dedicated_engine, demand_states, state_labels,
                                   non_anchor_fraction=non_anchor_fraction, anchor_override_fn=anchor_override_fn)

    print(f"Phase 4 done -- saved to {out_dir}/")


def phase6_region_pooling(engine: SimulationEngine, scenario: dict, out_dir: str) -> None:
    """Regional pooling CURVE: how region-pooled occupancy/coverage shifts
    across the SAME 3 demand_states distributions phase 4/5 use (near/
    balanced/far-heavy, via _force_synchronized_demand_state), comparing
    ground_truth (all real UEs, known) against anchor_only (raw,
    uninterpolated) -- one line per (situation, region), x-axis =
    distribution. Reuses phase4_measurement_aggregation's own
    run_multi_trial_aggregation/num_trials=1/non_anchor_fraction=0.8
    convention. ground_truth/anchor_only ONLY (2026-09-20 rewrite,
    replacing the old spatial-fill-map + across-region-variance content,
    which mostly duplicated what phase 4/5 already show) -- RME situations
    deliberately deferred until the region-inference approach itself is
    fixed (see phase 5's own kriging_hard/soft, cgr_hard/soft situations
    and their negative finding this session).

    `scenario` is unused -- builds its OWN dedicated engine, reused across
    every distribution (needs_scenario=False in PHASES, same convention as
    phase 4; a single reused engine rather than one per distribution,
    2026-09-20 -- see compute_grid_validation_for_distribution's own
    docstring for why this is safe).
    """
    os.makedirs(out_dir, exist_ok=True)
    region_names = OccupancyEstimator.region_names
    num_regions = len(region_names)
    num_trials, non_anchor_fraction = 1, 0.8

    dedicated_engine = build_debug_engine()
    demand_states = dedicated_engine.mobility.demand_states
    state_labels = []
    for state_idx in range(demand_states.shape[0]):
        row = demand_states[state_idx]
        heavy_region = region_names[int(np.argmax(row))]
        state_labels.append(f"[{','.join(str(v) for v in row)}]\n({heavy_region}-heavy)")

    situations_per_state = []
    for state_idx, state_label in enumerate(state_labels):
        print(f"[phase6] distribution {state_idx}: {state_label!r} ({num_trials} trials)")
        situations, _ = run_multi_trial_aggregation(
            dedicated_engine, state_idx, non_anchor_fraction, num_trials, num_regions)
        situations_per_state.append(situations)
        for sit in _SITUATION_ORDER:
            print(f"  {sit}: occupancy={situations[sit]['occupancy'].round(3).tolist()} "
                 f"coverage={np.round(situations[sit]['coverage'], 3).tolist()}")

    x = np.arange(len(state_labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in ((axes[0], "occupancy", "Regional pooling: occupancy"),
                              (axes[1], "coverage", "Regional pooling: coverage")):
        for region in region_names:
            r = region_names.index(region)
            for sit, linestyle in (("ground_truth", "-"), ("anchor_only", "--")):
                y = [situations_per_state[s][sit][metric][r] for s in range(len(state_labels))]
                ax.plot(x, y, linestyle, marker="o", color=HOTSPOT_REGION_COLORS[region],
                       label=f"{region} ({sit.replace('_', ' ')})")
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=11)
        if metric == "coverage":
            ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Regional pooling curve: ground truth vs. anchor-only, across UE distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_pooling_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Across-region variance, one bar per distribution: how much near/mid/
    # far differ from EACH OTHER within one distribution (ground_truth vs
    # anchor_only) -- the natural analog, in this distribution-based
    # design, of the old region_variance_over_time.png's own across-region
    # variance (that one plotted it over an interval/time axis; this
    # design has no time axis, just 3 forced distributions).
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in ((axes[0], "occupancy", "Across-region variance: occupancy"),
                              (axes[1], "coverage", "Across-region variance: coverage")):
        for i, sit in enumerate(_SITUATION_ORDER):
            variances = [np.var(situations_per_state[s][sit][metric]) for s in range(len(state_labels))]
            ax.bar(x + (i - 0.5) * width, variances, width, label=_SITUATION_LABELS[sit],
                  color=_SITUATION_COLORS[sit])
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels, fontsize=8)
        ax.set_ylabel("variance across near/mid/far")
        ax.set_title(title, fontsize=11)
    axes[0].legend(fontsize=8)
    fig.suptitle("Across-region variance per distribution: ground truth vs. anchor-only", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_variance_across_regions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 6 done -- saved to {out_dir}/")


def compute_interference_decomposition(engine: SimulationEngine, state_idx: int, target_sector: int = 0,
                                       tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                       num_trials: int = 5, seed_offset: int = 0) -> dict:
    """How much of target_sector's own UEs' INTERFERENCE comes from EACH
    of the other num_bs-1 sectors, compared to target_sector's own signal
    power -- direct quantification of how much a state representation
    loses by only looking at a handful of "local" neighbors instead of the
    whole network (2026-10-13, explicit request).

    power_w (from resolve_power_at_tilt) is [sector, ue] received power in
    WATTS from EVERY sector at EVERY UE, not just the serving one (see
    KpiManager._compute_serving_sinr, whose own interference_w = total_w -
    power_w this directly generalizes to a PER-SECTOR breakdown instead of
    one pooled "interference" scalar). For target_sector's own UEs (home
    sector via assign_to_sector_rhombus_grid, same convention as every
    other per-sector grouping in this file): own_signal = power_w[target_sector],
    interference_from_k = power_w[k] for every other sector k.

    ONE fixed uniform tilt (every sector at tilt_deg, default
    SINGLE_SECTOR_NOMINAL_TILT_DEG) -- a representative operating-point
    snapshot, not a sweep (this is about WHICH sectors matter, not how
    their effect changes with tilt). num_trials independent re-draws
    (fresh positions each trial, non_anchor_fraction=0.0 -- every real UE
    counts, this never routes through the anchor-gated pipeline) averaged.

    :output: dict -- own_signal_power_w [scalar]; interference_power_w
        [num_bs-1] mean power received from each OTHER sector, in sector-id
        order; sector_ids [num_bs-1] (matching order);
        total_interference_power_w [scalar].
    """
    num_bs = engine.num_bs
    other_sectors = [s for s in range(num_bs) if s != target_sector]
    trial_own, trial_interf = [], []
    for trial in range(num_trials):
        _force_synchronized_demand_state(engine, state_idx, non_anchor_fraction=0.0, mask_seed=seed_offset + trial)
        state = engine.step_environment(True)
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        sector_idx, _ = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
        own_mask = sector_idx == target_sector
        tilt_vec = np.full(num_bs, tilt_deg)
        power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]  # [num_bs, num_ue]
        trial_own.append(float(power_w[target_sector][own_mask].mean()))
        trial_interf.append(power_w[other_sectors][:, own_mask].mean(axis=1))

    own_signal_power_w = float(np.mean(trial_own))
    interference_power_w = np.mean(trial_interf, axis=0)
    return dict(own_signal_power_w=own_signal_power_w, interference_power_w=interference_power_w,
               sector_ids=np.array(other_sectors), total_interference_power_w=float(interference_power_w.sum()))


def _interference_decomposition_figure(out_dir: str, engine: SimulationEngine, demand_states: np.ndarray,
                                       state_labels: list, target_sector: int = 0,
                                       tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG) -> None:
    """interference_decomposition.png: per demand state (rows), LEFT panel
    -- every other sector's own mean interference power at target_sector's
    UEs (dBm, sorted descending), with target_sector's own mean signal
    power and the total pooled interference power marked as reference
    lines, so "how does interference compare to signal" is read directly
    off the same axis. RIGHT panel -- cumulative fraction of TOTAL
    interference captured by the top-K sectors (K=1..num_bs-1) -- the
    direct "how much information would a K-neighbor-only state lose"
    answer (e.g. K=2 -- this session's earlier front-neighbor state design
    -- read straight off the curve at x=2).
    """
    num_states = demand_states.shape[0]
    fig, axes = plt.subplots(num_states, 2, figsize=(15, 4.5 * num_states), squeeze=False)
    for row, state_idx in enumerate(range(num_states)):
        res = compute_interference_decomposition(engine, state_idx, target_sector=target_sector, tilt_deg=tilt_deg)
        order = np.argsort(-res["interference_power_w"])
        sorted_ids = res["sector_ids"][order]
        sorted_db = 10.0 * np.log10(res["interference_power_w"][order] * 1000.0)  # dBm
        own_db = 10.0 * np.log10(res["own_signal_power_w"] * 1000.0)
        total_interf_db = 10.0 * np.log10(res["total_interference_power_w"] * 1000.0)

        ax = axes[row, 0]
        ax.bar(range(len(sorted_ids)), sorted_db, color="#d7191c")
        ax.axhline(own_db, color="#333333", linestyle="--", linewidth=2,
                  label=f"own signal power ({own_db:.1f} dBm)")
        ax.axhline(total_interf_db, color="#1a9850", linestyle="--", linewidth=2,
                  label=f"TOTAL interference, all 20 sectors ({total_interf_db:.1f} dBm)")
        ax.set_xticks(range(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, fontsize=7, rotation=90)
        ax.set_xlabel("other sector id (sorted by own interference contribution)")
        ax.set_ylabel("interference power at sector 0's own UEs (dBm)")
        ax.set_title(f"{state_labels[row]}: per-sector interference vs. own signal", fontsize=10)
        ax.legend(fontsize=7)

        cum_frac = np.cumsum(res["interference_power_w"][order]) / res["total_interference_power_w"]
        ax2 = axes[row, 1]
        ax2.plot(range(1, len(sorted_ids) + 1), cum_frac, "-o", color="#4575b4")
        ax2.axhline(1.0, color="#888888", linestyle=":", linewidth=1)
        for k_mark in (2, 5, 10):
            if k_mark <= len(sorted_ids):
                ax2.axvline(k_mark, color="#888888", linestyle=":", linewidth=1)
                ax2.annotate(f"K={k_mark}\n{cum_frac[k_mark - 1]:.1%}", (k_mark, cum_frac[k_mark - 1]),
                           textcoords="offset points", xytext=(5, -15), fontsize=7)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("K (top-K interfering sectors included)")
        ax2.set_ylabel("cumulative fraction of TOTAL interference captured")
        ax2.set_title(f"{state_labels[row]}: information captured by a K-neighbor-only state", fontsize=10)

    fig.suptitle(f"Interference decomposition: sector {target_sector}'s own UEs, every other sector's own "
                f"contribution vs. own signal power (uniform tilt={tilt_deg:.0f} deg)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interference_decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [interference decomposition] done -- saved to {out_dir}/interference_decomposition.png")


_K_POLICY_COLORS = {"global": "#333333", "k10": "#2a78d6", "k8": "#4daf4a", "k5": "#ff7f00",
                    "k2": "#984ea3", "own": "#d7191c"}


def compute_neighbor_visibility_tilt_decision(engine: SimulationEngine, state_idx: int, tilt_values: np.ndarray,
                                               target_sector: int = 0,
                                               other_sectors_tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                               k_values: tuple = (10, 8, 5, 2), num_trials: int = 5,
                                               seed_offset: int = 0) -> dict:
    """2026-10-14, explicit request -- directly connects
    compute_interference_decomposition's own "who matters" ranking to an
    actual DECISION-QUALITY question: if target_sector could only see
    coverage from itself plus its own top-K interferers (by interference
    power, same ranking/mechanism as compute_interference_decomposition),
    how good is the tilt it would end up picking, measured by the REAL
    global-network consequence of that pick -- not just how different the
    chosen tilt looks.

    Every OTHER sector fixed at other_sectors_tilt_deg throughout (nominal
    by default) -- only target_sector's own tilt varies. Six policies, each
    a different coverage METRIC a decision-maker with that visibility would
    maximize over the sweep:
      "global" -- every real UE in the whole 21-sector network (the TRUE
        optimum this experiment judges every other policy against).
      "k10"/"k8"/"k5"/"k2" -- target_sector's own UEs POOLED with its own
        top-K interfering sectors' UEs (K by interference power, re-ranked
        fresh each trial at the other_sectors_tilt_deg baseline -- same
        method as compute_interference_decomposition, not a fixed list).
      "own" -- target_sector's own UEs only (the worst case: zero neighbor
        visibility).
    Ground-truth coverage throughout (non_anchor_fraction=0.0) -- this
    experiment isolates NEIGHBOR VISIBILITY SCOPE specifically, deliberately
    not conflated with phase 4's separate anchor-sparsity question.

    Each policy's own metric curve is trial-averaged first (same precedent
    as every other tilt-sweep diagnostic this session), THEN its own argmax
    tilt is read off that averaged curve -- not a per-trial vote.

    :output: dict with tilt_values; "global"/"k10"/"k8"/"k5"/"k2"/"own"
        [T] trial-averaged coverage curves (global = whole-network, the
        rest = that policy's own pooled visibility, NOT whole-network);
        "neighbor_sectors" -- dict {k: [sector ids]} -- the LAST trial's own
        top-K ranking, for labeling only.
    """
    num_bs = engine.num_bs
    num_t = len(tilt_values)
    policies = ["global"] + [f"k{k}" for k in k_values] + ["own"]
    trial_metric = {p: np.zeros((num_trials, num_t)) for p in policies}
    neighbor_sectors = {}

    for trial in range(num_trials):
        _force_synchronized_demand_state(engine, state_idx, non_anchor_fraction=0.0, mask_seed=seed_offset + trial)
        state = engine.step_environment(True)
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        sector_idx, _ = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
        own_mask = sector_idx == target_sector

        # Rank by interference power at the other_sectors_tilt_deg baseline
        # -- one ranking per trial, reused across sector 0's whole sweep
        # (a decision-maker doesn't re-rank "who matters" at every
        # hypothetical tilt it's considering).
        baseline_tilt_vec = np.full(num_bs, other_sectors_tilt_deg)
        baseline_power_w = engine.kpi_manager.resolve_power_at_tilt(state, baseline_tilt_vec)[0]
        other_sectors = [s for s in range(num_bs) if s != target_sector]
        interf = baseline_power_w[other_sectors][:, own_mask].mean(axis=1)
        ranked = np.array(other_sectors)[np.argsort(-interf)]

        masks = {"global": np.ones(engine.num_ut, dtype=bool), "own": own_mask}
        for k in k_values:
            neighbor_sectors[k] = ranked[:k].tolist()
            masks[f"k{k}"] = own_mask | np.isin(sector_idx, ranked[:k])

        for t, tilt in enumerate(tilt_values):
            tilt_vec = np.full(num_bs, other_sectors_tilt_deg)
            tilt_vec[target_sector] = tilt
            power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
            kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
            covered = (kpis["sinr_db"][0] > engine.coverage_threshold_db).astype(float)
            for p in policies:
                trial_metric[p][trial, t] = covered[masks[p]].mean()

    result = dict(tilt_values=tilt_values, neighbor_sectors=neighbor_sectors)
    for p in policies:
        result[p] = trial_metric[p].mean(axis=0)
    return result


def _neighbor_visibility_tilt_decision_figure(out_dir: str, engine: SimulationEngine, demand_states: np.ndarray,
                                              state_labels: list, target_sector: int = 0,
                                              other_sectors_tilt_deg: float = SINGLE_SECTOR_NOMINAL_TILT_DEG,
                                              k_values: tuple = (10, 8, 5, 2)) -> list:
    """neighbor_visibility_tilt_decision.png: per demand state (rows), LEFT
    panel -- all 6 policies' own coverage-vs-tilt curves overlaid, each
    curve's own argmax tilt marked with a star. RIGHT panel -- the REAL
    global-network coverage that results from trusting each policy's own
    pick (global curve evaluated AT that policy's chosen tilt, per
    [[feedback_plot_verification_style]] convention: the decision-relevant
    number anchored directly on a bar, not left to a console table), sorted
    global/k10/k8/k5/k2/own so the coverage lost as visibility shrinks
    reads directly off the bar heights. Also writes
    neighbor_visibility_tilt_decision_summary.txt (state, policy, chosen
    tilt, resultant global coverage, loss vs. the global-optimal pick).
    """
    policies = ["global"] + [f"k{k}" for k in k_values] + ["own"]
    policy_labels = {"global": "global (true optimum)", **{f"k{k}": f"K={k} neighbors" for k in k_values},
                     "own": "own sector only (worst)"}
    num_states = demand_states.shape[0]
    fig, axes = plt.subplots(num_states, 2, figsize=(15, 4.5 * num_states), squeeze=False)
    summary_rows = []

    for row, state_idx in enumerate(range(num_states)):
        res = compute_neighbor_visibility_tilt_decision(engine, state_idx, engine.downtilt_sweep_deg,
                                                         target_sector=target_sector,
                                                         other_sectors_tilt_deg=other_sectors_tilt_deg,
                                                         k_values=k_values)
        tilt_values = res["tilt_values"]
        global_curve = res["global"]
        gt_opt_idx = int(np.argmax(global_curve))

        ax = axes[row, 0]
        for p in policies:
            ax.plot(tilt_values, res[p], "-o", color=_K_POLICY_COLORS[p], label=policy_labels[p])
            opt_idx = int(np.argmax(res[p]))
            ax.plot(tilt_values[opt_idx], res[p][opt_idx], "*", color=_K_POLICY_COLORS[p],
                   markersize=16, markeredgecolor="black", zorder=5)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"sector {target_sector} tilt (deg) -- others fixed at {other_sectors_tilt_deg:.0f} deg")
        ax.set_ylabel("coverage fraction (policy's own visibility)")
        ax.set_title(f"{state_labels[row]}: coverage-vs-tilt by visibility policy", fontsize=10)
        if row == 0:
            ax.legend(fontsize=7)

        ax2 = axes[row, 1]
        chosen_tilts, resultant_global, losses = [], [], []
        for p in policies:
            opt_idx = int(np.argmax(res[p]))
            chosen_tilts.append(tilt_values[opt_idx])
            resultant_global.append(global_curve[opt_idx])
            losses.append(global_curve[gt_opt_idx] - global_curve[opt_idx])
            summary_rows.append((state_labels[row], p, float(tilt_values[opt_idx]), float(global_curve[opt_idx]),
                                float(global_curve[gt_opt_idx] - global_curve[opt_idx])))
        bars = ax2.bar(range(len(policies)), resultant_global, color=[_K_POLICY_COLORS[p] for p in policies])
        for i, (bar, tilt, loss) in enumerate(zip(bars, chosen_tilts, losses)):
            label = f"{tilt:.0f}°" + (f"\n-{loss:.3f}" if loss > 1e-6 else "\n(optimal)")
            ax2.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom",
                        fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(len(policies)))
        ax2.set_xticklabels([policy_labels[p] for p in policies], fontsize=8, rotation=20, ha="right")
        ax2.set_ylabel("REAL global-network coverage at the policy's own chosen tilt")
        ax2.set_title(f"{state_labels[row]}: coverage lost by limited visibility", fontsize=10)

    fig.suptitle(f"Neighbor-visibility tilt decision: sector {target_sector}, {num_states} demand states "
                f"(others fixed at {other_sectors_tilt_deg:.0f} deg)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "neighbor_visibility_tilt_decision.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(out_dir, "neighbor_visibility_tilt_decision_summary.txt")
    with open(summary_path, "w") as f:
        header = f"{'state':<28}{'policy':<10}{'chosen_tilt':<13}{'resultant_global_cov':<22}{'loss':<10}"
        print(header, file=f)
        print("  [neighbor visibility tilt decision]")
        print(f"  {header}")
        for state_label, p, tilt, cov, loss in summary_rows:
            line = f"{state_label:<28}{p:<10}{tilt:<13.0f}{cov:<22.4f}{loss:<10.4f}"
            print(line, file=f)
            print(f"  {line}")
    print(f"  [neighbor visibility tilt decision] done -- saved to {out_dir}/neighbor_visibility_tilt_decision.png, "
         f"{summary_path}")
    return summary_rows


def phase7_state_representation(engine: SimulationEngine, scenario: dict, out_dir: str) -> None:
    """State REPRESENTATION comparison (2026-10-15 rewrite -- the
    resolution-comparison part (cell/polar/region/whole anchor-limited
    reconstruction) was removed entirely, explicit request: that design
    tested MEASUREMENT fidelity under anchor sparsity, not actual DECISION
    quality -- a mean-coverage objective is a weighted average that's
    mathematically invariant to binning once you have an accurate per-bin
    estimate, so it was never going to show "resolution changes the
    decision" the way intended. See this session's own back-and-forth for
    the full reasoning; postponed pending a decision-quality-sensitive
    policy design (not a plain mean-maximizer) to test resolution with
    instead.

    What remains -- the GLOBAL VS LOCAL information-scope comparison, which
    is a different, still-valid question (how much information a decision-
    maker has ACCESS to, not how finely it's binned):

    (interference_decomposition.png, see compute_interference_decomposition/
    _interference_decomposition_figure's own docstrings, 2026-10-13 explicit
    request): how much of sector 0's own interference comes from EACH of
    the other 20 sectors individually, vs. sector 0's own signal power, at
    one representative uniform tilt.

    (neighbor_visibility_tilt_decision.png, see
    compute_neighbor_visibility_tilt_decision/
    _neighbor_visibility_tilt_decision_figure's own docstrings, 2026-10-14
    explicit request) connects that "who matters" interference ranking to
    an actual DECISION outcome: sector 0's tilt picked from 6 visibility
    policies (global/K=10/8/5/2/own-sector-only), each policy's own chosen
    tilt then evaluated by its REAL effect on true global-network coverage.

    `scenario` is unused -- builds its OWN dedicated engine (needs_scenario=
    False in PHASES, same convention as phase 4/6).
    """
    os.makedirs(out_dir, exist_ok=True)
    dedicated_engine = build_debug_engine()
    demand_states = dedicated_engine.mobility.demand_states
    region_names = OccupancyEstimator.region_names

    target_sector = 0
    nbr = dedicated_engine.topology.neighbor_ids[target_sector]
    nbr = nbr[nbr >= 0]
    assert len(nbr) >= 2, f"sector {target_sector} needs >=2 real neighbors, has {len(nbr)}"
    n1, n2 = int(nbr[0]), int(nbr[1])
    print(f"[phase7] target_sector={target_sector} n1={n1} n2={n2}")

    state_labels = []
    for state_idx in range(demand_states.shape[0]):
        row = demand_states[state_idx]
        heavy_region = region_names[int(np.argmax(row))]
        state_labels.append(f"[{','.join(str(v) for v in row)}] ({heavy_region}-heavy)")

    _interference_decomposition_figure(out_dir, dedicated_engine, demand_states, state_labels,
                                       target_sector=target_sector)

    _neighbor_visibility_tilt_decision_figure(out_dir, dedicated_engine, demand_states, state_labels,
                                              target_sector=target_sector)

    print(f"Phase 7 done -- saved to {out_dir}/")


def phase8_reward_behavior(engine: SimulationEngine, scenario: dict, out_dir: str) -> None:
    """Global reward over time, all 21 sectors: (1) a [sector x interval]
    heatmap (same convention as earlier phases' heatmaps) -- every row is
    now IDENTICAL within a given interval column (reward changed from
    per-sector "sector_neighbors" to a single network-wide scalar
    2026-09-20, see RegionalInfoProvider's own module docstring), so this
    mainly confirms the broadcast is working, not sector-to-sector
    variation; (2) reward's own variance no-tilt vs random-tilt phase --
    since tilt is literally the only thing that changes between the two
    phases, reward should respond (higher variance) once random tilt
    starts, otherwise reward isn't sensitive to the action at all; (3)
    state_ready coverage, confirming reward is usable often enough to train on.
    """
    os.makedirs(out_dir, exist_ok=True)
    reward, state_ready = scenario["reward"], scenario["state_ready"]
    num_intervals, num_bs = reward.shape

    # A LINE plot, not a heatmap -- reward is identical across all 21
    # sectors every interval (global, by design), so a [sector x interval]
    # heatmap's own sector axis is redundant (every column one solid
    # color) and a fixed vmin=0 buries the real interval-to-interval
    # variation in a narrow high range, making it look uniformly saturated
    # (fixed 2026-09-20, per direct user feedback -- "the plot is all red").
    reward_per_interval = reward[:, 0]  # same value every sector -- one representative column
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(reward_per_interval, color="#2a78d6", linewidth=1.2, marker="o", markersize=3)
    ax.axvline(NUM_NO_TILT - 0.5, color="black", linestyle="--", linewidth=1, label="no-tilt / random-tilt boundary")
    ax.set_xlabel("interval")
    ax.set_ylabel("reward (global coverage ratio)")
    ax.set_title("Global reward per interval (same value every sector)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "reward_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    no_tilt_var = float(np.nanvar(reward[:NUM_NO_TILT]))
    random_tilt_var = float(np.nanvar(reward[NUM_NO_TILT:]))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["no-tilt (0 deg)", "random-tilt"], [no_tilt_var, random_tilt_var], color=["#888888", "#2a78d6"])
    ax.set_ylabel("variance of reward across (interval, sector)")
    ax.set_title("Reward responds to tilt? (should be higher once tilt varies)", fontsize=10)
    fig.savefig(os.path.join(out_dir, "reward_variance_by_phase.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ready_fraction = state_ready.mean()
    print(f"Phase 8 done (no_tilt reward var={no_tilt_var:.4f}, random_tilt reward var={random_tilt_var:.4f}, "
         f"state_ready={ready_fraction:.1%} of (interval,sector)) -- saved to {out_dir}/")


def _r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    residual = actual - predicted
    total = actual - actual.mean()
    denom = float(np.sum(total ** 2))
    return 1.0 - float(np.sum(residual ** 2)) / denom if denom > 0 else float("nan")


# =============================================================
# DEBUG PHASES REGISTRY DICT
# =============================================================

# Config keys phase1_static_geometry's own output actually depends on --
# pure site/sector/grid geometry, nothing UE/mobility/channel-related (see
# phase1_static_geometry's own docstring). main() uses this to skip
# regenerating phase 1 when none of these changed since its last run
# (2026-10-10, explicit request).
PHASE1_TOPOLOGY_KEYS = ("scenario", "num_rings", "spatial_grid_cell_size_m")
PHASE1_ANTENNA_KEYS = ("azimuth_offset_deg",)
PHASE1_CONFIG_SENTINEL = "_phase1_config.json"


def _phase1_relevant_config(cfg: dict) -> dict:
    return {**{k: cfg["topology"][k] for k in PHASE1_TOPOLOGY_KEYS},
           **{k: cfg["antenna"][k] for k in PHASE1_ANTENNA_KEYS}}


# Phase index: (Name, Function)
# name, fn, needs_scenario -- needs_scenario=False means fn's own
# `scenario` argument (present on every fn below for one shared call
# signature) is never actually read, so main() can skip building/loading
# the real mobility scenario entirely when only such a phase is requested.
PHASES = {
    1: ("static_geometry", phase1_static_geometry, False),
    2: ("radio_environment_validation", phase2_radio_environment_validation, False),
    3: ("raw_ue_behavior", phase3_raw_ue_behavior, True),
    4: ("measurement_aggregation", phase4_measurement_aggregation, False),
    5: ("grid_measurements_and_rme", phase5_grid_measurements_and_rme, True),
    6: ("region_pooling", phase6_region_pooling, False),
    7: ("state_representation", phase7_state_representation, False),
    8: ("reward_behavior", phase8_reward_behavior, True),
}

# =============================================================
# MAIN FUNCTION
# =============================================================

def main():
    # Parse which phase(s) to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, nargs="+", choices=sorted(PHASES.keys()),
                       help="One or more phase numbers, run in order, sharing one engine and one "
                            "debug scenario (built/loaded at most once, not once per phase).")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, required=True,
                       help="--use-cache reuses results/debug/_scenario_cache.npz if present; "
                            "--no-use-cache always re-runs the 50-interval scenario fresh (overwriting "
                            "the cache). Required, no default -- a stale cache silently reused across a "
                            "config.yaml change (e.g. coverage_threshold_db) produces results that look "
                            "valid but are quietly wrong, so the choice must be made explicitly every run.")
    args = parser.parse_args()

    # ENVIRONMENT seed (topology/mobility/UE-drop/pathloss/shadow-fading/
    # LOS-NLOS) -- same convention as main.py: pin ONCE, here, before any
    # engine is built, so this whole run (the shared engine here, plus
    # every phase's own additional ground-truth engines, e.g. phase 3/4's
    # build_debug_engine() calls) draws from one deterministic stream and
    # is fully reproducible run-to-run.
    sionna.phy.config.seed = DEBUG_SEED

    # Build the simulation engine
    engine = build_debug_engine()

    # Run the debug phase functions
    scenario = None
    for phase in args.phase:
        name, fn, needs_scenario = PHASES[phase]
        out_dir = os.path.join(OUT_ROOT, f"phase{phase}_{name}")
        if phase in (1, 2, 3, 4):
            # Phases 1-4 accumulate many hand-run config variants (scenario,
            # grid size, shadow fading, LOS condition, antenna, demand
            # splits, outdoor ratio, ...) alongside the ordinary
            # config.yaml-default run -- nest the default run under its own
            # subfolder so it doesn't get lost among the variants (which
            # live as their own named subfolders, e.g. .../uma/,
            # .../grid20m/, .../grouped_rect_910/, .../outdoor_020/, placed
            # there by hand after a temp config edit or a dedicated
            # scenario script, same pattern as this session's own variant
            # runs). 2026-10-13: extended from (1, 2) to (1, 2, 3, 4) once
            # phase 3/4 started accumulating the same kind of variants.
            out_dir = os.path.join(out_dir, "default")
        if phase == 1:
            sentinel_path = os.path.join(out_dir, PHASE1_CONFIG_SENTINEL)
            current_config = _phase1_relevant_config(load_config())
            previous_config = None
            if os.path.exists(sentinel_path):
                with open(sentinel_path) as f:
                    previous_config = json.load(f)
            if previous_config == current_config:
                print(f"Phase 1 already done (static geometry config unchanged) -- saved in {out_dir}/")
            else:
                fn(engine, out_dir)
                with open(sentinel_path, "w") as f:
                    json.dump(current_config, f, indent=2)
        elif not needs_scenario:
            # Pure radio-environment/geometry phase -- never reads its own
            # `scenario` argument, so skip building/loading the real
            # (potentially expensive) mobility scenario entirely.
            fn(engine, None, out_dir)
        else:
             # Run or load the scenario for phase(s) if scenario not already loaded
            if scenario is None:
                scenario = load_or_run_scenario(engine, args.use_cache)
            fn(engine, scenario, out_dir)


if __name__ == "__main__":
    main()
