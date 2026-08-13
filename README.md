# E-Tilt RL/Rule-Based Controller — Design Notes

Written from a long discussion chat, meant to give a fresh implementation chat full context
without replaying the whole conversation. Point a new chat at this file and ask it to
implement accordingly.

## Environment Setup

Use the `sionna-wbbf` virtualenv at `/home/shadab/venvs/sionna-wbbf` for everything in
this repo — do not use `/usr/virtualenvs/sionna2`. `sionna2`'s torch build
(`2.10.0+cu128`) has a real cuBLAS bug on this machine's RTX 5090: any batched GEMM
(`set_topology()`/`generate_h_freq()` with `batch_size > 1`, i.e. `MAX_REALIZATION_CUDA
> 1`) throws `CUBLAS_STATUS_INVALID_VALUE`. `sionna-wbbf`'s torch (`2.11.0+cu128`) does
not have this bug, confirmed by direct testing.

To create it from scratch (or reproduce it elsewhere):

```bash
python3.12 -m venv /home/shadab/venvs/sionna-wbbf
/home/shadab/venvs/sionna-wbbf/bin/pip install -r requirements.txt
```

To activate it (once per shell session):

```bash
source /home/shadab/venvs/sionna-wbbf/bin/activate
```

Then run any script in this repo normally:

```bash
python scripts/<script_name>.py
```

Deactivate with `deactivate`. To run a one-off script without activating, call the
venv's Python directly instead: `/home/shadab/venvs/sionna-wbbf/bin/python3
scripts/<script_name>.py`.

**`requirements.txt` must stay in sync**: whenever a new package is installed into
`sionna-wbbf`, add it (with its pinned version) to `requirements.txt` in the same change.

### Long-running scripts: use `run_in_tmux.sh`

Some scripts here (e.g. `test_dynamic_scenario_tilts_effect.py` at `NUM_SLOTS=500`) run for
minutes. Since this repo is normally used over an SSH/VS Code Remote connection that can
drop and kill whatever was running in its terminal, launch anything long-running inside
a detached tmux session instead, so it keeps running on the remote machine independent
of your client connection:

```bash
./run_in_tmux.sh scripts/tests/test_dynamic_scenario_tilts_effect.py
```

This starts a tmux session named after the script (refuses to start a second one if a
session with that name is already running), tees its output to
`results/tests/_run_logs/<script>_<timestamp>.log`, and prints how to reattach
(`tmux attach -t <session_name>`; detach again with `Ctrl+B` then `D`, which leaves it
running). It always runs scripts with the `sionna-wbbf` venv's Python, regardless of
whether that venv is active in your current shell.

## Goal

Build a controller that takes UE SINR measurements as feedback and adjusts a shared
**electrical downtilt** angle per sector, closed-loop — a minimal Remote Electrical Tilt
(RET) optimization loop, aligned with this repo's existing simulation machinery.

System model in one paragraph: a hexagonal macro layout (N rings, 3 sectors/site). Each
sector transmits through a **single logical antenna port** (a broadcast/SSB-like common
beam every UE in range measures identically — not a per-UE precoded beam), formed by a
uniform linear array of M elements whose per-element phase weights realize **electrical**
downtilt (3GPP TR 38.901, clause 7.3.1, eq. 7.3-1) — the antenna panel itself never
physically rotates. UEs are dropped uniformly and move under a simple mobility model.
Each UE's serving sector is whichever sector gives it the highest measured SINR (SS-SINR
analog; no RRC attach/handover signaling is modeled). Each sector periodically aggregates
its associated UEs' SINR and adjusts its own downtilt angle from that feedback. Explicitly
out of scope: MAC scheduler, link adaptation (MCS/CQI), MIMO precoder/codebook, multi-user
MIMO/spatial multiplexing, real data traffic, explicit HO signaling.

## RESOLVED: 38.901-only, built on Sionna's own antenna classes (not a local reimplementation)

the target is the plain TR 38.901 clause 7.3.1 electrical-tilt formula

```
w_m = (1/sqrt(M)) * exp(-j * 2*pi * (m-1) * d_V * cos(theta_etilt)),  m = 1..M
```

No sub-array structure, no correlation factor, no azimuth scan (d_V in multiples of
wavelength, so no separate carrier-frequency term in the phase itself).

**Key finding: Sionna's installed package already provides the element pattern and
array geometry standard-compliant, so none of it needed reimplementing.**
`sionna.phy.channel.tr38901.antenna` ships `AntennaElement` (omni + TR 38.901 Table
7.3-1 — verified byte-identical to what the deleted `antenna_38922.py` had copied from
it) and `AntennaPanel`/`PanelArray`/`Antenna`/`AntennaArray` for element geometry.
`PanelArray`'s panel-group counts (`num_rows`/`num_cols`) default to 1, and
`Antenna`/`AntennaArray` are thin convenience subclasses that just delegate to
`PanelArray` with renamed arguments — verified they produce byte-identical `ant_pos` for
the same configuration. So a single vertical column (our single-port array) is just
`AntennaArray(num_rows=M, num_cols=1, polarization='single', polarization_type='V',
antenna_pattern='38.901', carrier_frequency=fc)` (or the equivalent `PanelArray` call) —
**no custom panel/element class needed at all.**

`helpers/electrical_downtilt.py` (implemented, see below) contains only the one
genuinely new piece: the `ElectricalDowntilt` class implementing eq. (7.3-1) on top of
any such single-column Sionna array (asserts `num_cols_per_panel == 1` and
`polarization == 'single'`).

**Angle convention, verified numerically (was an open question, now closed)**:
`theta_etilt` and the observation zenith angle `theta` use the *same* convention
directly, no offset/complementary-angle conversion needed — `array_factor(theta)` peaks
exactly at `theta == theta_etilt` for the same value in degrees, confirmed for
theta_etilt in {60, 90, 120} deg with array gain = M (linear) at each peak, plus a
`show()`-produced plot visually confirming the beam rotates correctly across a tilt
sweep.

## Key architectural understanding (established through extensive back-and-forth)

- **Unit note (current API)**: `ElectricalDowntilt.set_tilt(downtilt_deg)` and the
  constructor's `downtilt_deg` argument are both **degrees of downtilt** (0 =
  boresight, + = down, - = up) — **not** the raw TR 38.901 zenith angle. The class
  converts internally (`theta_etilt = 90 + downtilt_deg`) and exposes the converted
  zenith value read-only via the `theta_etilt_deg` property if it's ever needed
  directly. (Earlier versions of this class took the raw zenith angle as
  `theta_etilt_deg` directly — changed because the natural caller-facing unit for a
  RET/SON control loop is downtilt, not zenith; everything internal still works in
  radians, fully encapsulated, callers never pass radians directly.)
- In this repo's model, each RF port/subarray is a **fixed, mechanically pre-combined**
  group of elements. Sionna's channel model (`h_freq_fading`) only ever sees the **port**
  level — raw elements never appear as separate channel dimensions.
- Applying **one shared tilt** to all UEs in a sector is mathematically a **diagonal
  matrix** multiply (`x = diag(w)·u ≡ w ⊙ u`, elementwise) — not the dense/padded matrix
  the repo's existing `naive_analog_beamforming`-style code (in `dmimo.py`,
  `build_local_analog_precoders`) builds for *per-UE* beam pointing. **Do not reuse that
  per-UE code for shared-tilt control** — it computes a different steering vector per UE
  and pads/repeats columns to fake a fixed matrix width, which is a hack specific to
  per-UE beam-pointing (and has a silent-truncation bug if
  `num_ut_per_sector > num_tx_ant`; `dmimo.py` has since been deleted along with the rest
  of the MU-MIMO/dMIMO cluster, noted here only so a fresh implementer doesn't go looking
  for it or reinvent the same pattern). Instead: one
  `ElectricalDowntilt.set_tilt(theta_etilt_deg=<controller state>)` call per control epoch,
  applied identically to every UE in the sector.
- **Two-timescale loop**, matching real RET/hybrid-MIMO practice: SINR sampling happens
  every **slot** (fast, tracks fading/mobility, already supported — see topology/mobility
  note below); tilt only updates once per **epoch** (many slots, e.g. 50–100), using an
  aggregate of that epoch's SINR samples (mean, or 5th-percentile for a cell-edge-
  protecting objective — leaning toward 5th-percentile as default, matching standard RET
  literature).

## Topology, mobility, and attachment

- Sionna's own `sionna.sys.topology.gen_hexgrid_topology` handles: hex-ring site layout
  (N rings, 3 sectors/site), uniform UE dropping per sector, and **UE mobility**
  (`min_ut_velocity`/`max_ut_velocity` draw a per-UE velocity, applied via
  `ut_loc += ut_velocities * slot_duration` each slot, then re-call
  `channel_model.set_topology(...)`). **Mobility is not a gap — it's already built into
  Sionna itself,** confirmed directly (not just inherited from the now-deleted
  repo-local wrapper, which had the same behavior — see "Files removed" below for why
  it was removed anyway).
- Sionna's `gen_hexgrid_topology` requires `num_rings >= 1` (asserts "must be positive")
  — confirmed by direct call, `num_rings=0` fails. A true single-site view needs slicing
  down from a 1-ring (7-site) result; Step 1's verification instead just shows the full
  1-ring layout, which is fine for that check.
- `gen_hexgrid_topology` also supports `downtilt_to_sector_center` (a **mechanical**
  baseline downtilt toward each sector's center) — open design decision below.
- **Real gap**: `gen_hexgrid_topology` assigns each UE to a sector *by construction*
  (geometric drop within that sector's angular wedge), not by measured signal quality.
  There is no argmax-SINR/RSRP re-attachment logic anywhere in this repo.
  `sionna-sls/functions/initial_attachment.py` (a sibling repo) already has this:
  `RsrpBasedAttachment` (and `NearestSectorAttachment`, `compute_attachment_diagnostics`)
  — port this over rather than writing new attachment logic from scratch.
- Per-UE zenith/azimuth angle relative to a candidate sector's boresight (needed for
  antenna-gain lookups) has no reusable helper left in this repo (it lived in the
  now-deleted `SystemLevelSimulator._compute_ue_theta_phi`) — recompute this directly,
  and for **every** candidate sector (not just one), when doing argmax attachment.

## Files to reuse as-is (no modification needed)

- Sionna's own `sionna.sys.topology` package — `gen_hexgrid_topology`, `HexGrid` (hex
  layout, UE drop, mobility — all already present, confirmed identical in spirit to
  what the repo-local version used to wrap).
- `helpers/scenario_view.py` — `plot_scenario`/`save_scenario`, combines a `HexGrid` +
  UE positions and adds the one thing `HexGrid.show()` doesn't do (plot the UEs),
  without reimplementing any hexagon/sector drawing. Deliberately plain functions, kept
  separate from both the topology and UE-drop layers, since it's the one place that
  legitimately depends on both at once.
- Sionna's own `sionna.phy.channel.tr38901` package — `AntennaElement`, `PanelArray`
  (or its `Antenna`/`AntennaArray` convenience subclasses). Standard-compliant element
  pattern and array geometry, reused directly with **no local antenna file at all** for
  this part (see the RESOLVED section above).
- Sionna's `UMi`/`UMa`/`RMa` channel models, for Step 3's channel/pathloss computation
  (construct directly with our own `bs_array`/`ut_array`; no repo-local setup helper
  needed for this — see "Files removed" below).

## Files removed from the repo

**Different use case (don't look for these — not needed at all):**
`functions/dmimo.py`, `functions/slnr_precoder.py`, `functions/antenna_38922.py`,
`tests/test_MU_MIMO_cellular.py`, `tests/test_DMIMO_intercell.py`,
`tests/test_dmimo_helpers.py`, `tests/test_SU_MIMO_cellular.py`,
`tests/plot_composite_pattern.py`, `tests/plot_capacity_cdf.py`,
`tests/plot_dmimo_logdet_cdf.py`, `scripts/plot_dmimo_comparison.py`,
`scripts/plot_intercell_dmimo_precoder_comparison.py`, `scripts/run_dmimo_comparison.sh`
— all built around per-UE precoding / MU-MIMO / dMIMO / the 38.922 composite model, none
of which this design needs. (`test_SU_MIMO_cellular.py`'s deletion also took down the
only consumer of `functions/simulation.py`'s `SystemLevelSimulator` — SU-MIMO doesn't
fit this project's single-port/rank-1 design either, so it wasn't worth keeping even as
a reference.)

**Redundant duplicates, not a different-use-case removal (available in the sibling
`sionna-sls` repo — byte-identical — if ever needed as reference):**
`functions/topology.py`, `functions/simulation.py`, `functions/utils.py`. These were
repo-local forks of/wrappers around Sionna's own topology and channel-model tooling.
Once `SystemLevelSimulator` (in `simulation.py`) had no remaining consumer in this repo
and Step 1's verification moved to using Sionna's own topology tools directly (see
above), nothing here imported any of the three anymore, so they were deleted rather than
kept unused. `functions/__init__.py` had a bare `from functions import utils` that would
have broken on any import once `utils.py` was gone — emptied that file when deleting.

## Files to add (new)

- `helpers/initial_attachment.py` — port `RsrpBasedAttachment` from `sionna-sls`.
  **Not yet done** (Step 4 below).
- `helpers/electrical_downtilt.py` — **done.** Contains only `ElectricalDowntilt`
  (eq. 7.3-1), built on Sionna's own `AntennaElement`/`PanelArray`/`AntennaArray` (see
  RESOLVED section above). Numerically verified: peak gain lands exactly at the
  requested `theta_etilt` (checked at 60/90/120 deg) with array gain = M (linear) at
  each peak; correctly asserts/rejects a real 2D panel; `show()` produces a polar plot
  confirming the beam visibly rotates across a tilt sweep. (File was briefly named
  `antenna_panel.py` during an intermediate version that still had its own
  reimplemented `AntennaElement`/`AntennaPanel`; renamed once those were replaced by
  Sionna's own classes, since by then the file only contained the tilt mechanism.)
- `helpers/tilt_controller.py` — **not yet done.** The control law:
  `update_tilt(prev_tilt_deg, sinr_db) -> new_tilt_deg`. Algorithm choice (rule-based
  proportional controller vs. bandit vs. RL) not yet decided; start rule-based.
- `run_etilt_control_loop.py` — new driver script (note: the `tests/` directory this was
  originally sketched under no longer exists; place this alongside the other verification
  scripts under `scripts/`, or a new location, when it's built). **One persistent
  topology** (not an outer "num_drops" random-redrop loop — tilt control needs a stable
  scene across time), nested slot/epoch loop (slot = SINR/attachment update, epoch = tilt
  update), logging `(epoch, sector_id, tilt_deg, sinr_db)`. No scheduler/OLLA/power
  control in this loop — there's no `SystemLevelSimulator` in this repo to accidentally
  pull that in from anymore anyway (see "Files removed" above).

## Step-by-step incremental plan (each step independently verifiable)

0. **Baseline** — confirm existing tests still import/run untouched (all changes above
   are additive new files; nothing shared is edited). **Not formally re-run** since the
   MU-MIMO/dMIMO test files were deleted rather than left untouched — there's nothing
   left that could regress from that cluster. (`topology.py`/`utils.py`/`simulation.py`
   were also later deleted, once unused and confirmed redundant — see "Files removed"
   below — so there's nothing there to regress either.)
1. **Topology + mobility standalone** — **done**, via `scripts/verifications/verify_topology.py`.
   **Revised to use Sionna's own topology tools directly, not the repo-local ones.**
   Checked `functions/topology.py`'s `gen_hexgrid_topology` and `functions/simulation.py`'s
   `CenterCellGrid` against Sionna's own `sionna.sys.topology` package and found most of
   what they add is redundant with what Sionna already provides (`HexGrid`/
   `gen_hexgrid_topology` there already do hex layout, UE drop, and sector-geometry
   plotting). The only genuinely new pieces in the repo-local versions are: (a) UE-drop
   constraints (`min_ue_azimuth_separation_deg`, `ue_elevation_mode`/`fixed_ue_height`) —
   not needed for this check, and (b) plotting the actual UE positions, which Sionna's
   own `HexGrid.show()` doesn't do. So this step now uses Sionna's `gen_hexgrid_topology`
   directly, plus one new small file, `helpers/scenario_view.py` (`plot_scenario`/
   `save_scenario`, plain functions) that combines an existing `HexGrid` + UE positions
   and calls the grid's own `show()` before scattering the UEs on top — no hexagon/sector
   drawing is reimplemented. `functions/topology.py`/`functions/simulation.py`/`functions/utils.py`
   were kept unused for a short while after this, then deleted once confirmed
   byte-identical to the copies in the sibling `sionna-sls` repo and confirmed nothing
   in this repo still imported them (see "Files removed" above) — available there for
   reference if ever needed, rather than carrying dead duplicate files here.
   *Verified*: Sionna's own `HexGrid` requires `num_rings >= 1` (confirmed via a direct
   call — `num_rings=0` raises `AssertionError`), so a 1-ring (7-site, 21-sector) layout
   is generated and shown in full (no center-cell crop, unlike the old `CenterCellGrid`
   version) — hex boundaries, per-site sector numbering, and 105 UEs (5/sector) all
   render correctly, saved to `results/verifications/topology/hex_topology.png`. One
   `+= ut_velocities * slot_duration` step produces real UE displacement (0.0015 m over
   0.5 ms, consistent with the 1-3 m/s velocity range used).
2. **`helpers/electrical_downtilt.py`** — **done**, see "Files to add" above for the
   verification results (peak-tracking, array gain, 2D-panel rejection, visual sweep).
3. **Per-UE SINR vs. every sector** — **done**, via `scripts/tests/test_static_global_tilts_effect.py`
   (new file). Genuinely frequency-selective, not scalar pathloss: the channel model is
   built with the *real* per-sector array (same one each `ElectricalDowntilt` wraps, not
   a placeholder — an earlier draft used a placeholder omni array for the channel model,
   which was wrong once per-element combining is in play), and Sionna's own
   `GenerateOFDMChannel` produces the full per-element, per-subcarrier channel (pathloss +
   shadow fading + multipath fast fading, all as the 3GPP model generates them). Each
   sector's own current `ElectricalDowntilt.weights()` combines its elements into that
   sector's port, per subcarrier, before power/SINR are computed — no fixed rx-per-tx
   association anywhere (every sector evaluated as a candidate server for every UE, fresh,
   each call). New reusable pieces (not ad-hoc script-level functions): originally one
   file (`UniformDropTopology`, copied from `sionna-sls` with one adaptation to use
   Sionna's own `HexGrid` directly -- a live cross-repo import was tried first and
   rejected, since both repos have a top-level `functions` package and
   `sionna-sls/functions/__init__.py`'s own absolute self-import breaks as soon as
   another `functions` package is loaded in the same process), later split (once
   mobility/clustered drops made the original name misleading -- it was doing site
   topology, UE dropping, *and* coverage-boundary math under a name that only
   described one of those) into `helpers/cellular_topology.py` (`CellularTopology` --
   site/sector geometry plus `is_within_coverage`/`mirror_bs_loc`/`default_drop_radius`
   only) and `helpers/ue_drop.py` (`sample_uniform_ut_loc`/`sample_clustered_ut_loc`,
   plain functions taking a topology and returning initial UE positions). Plus
   `helpers/kpi_calculator.py` (`KpiCalculator` — takes a configured channel model, a
   `ResourceGrid`, and the list of per-sector `ElectricalDowntilt`s; exposes
   `compute_power_matrix_w()`/`compute_ue_sinr_db()`).
   UEs are dropped uniformly at random over the whole hex-grid area (not a fixed count per
   sector — confirmed Sionna's own `HexGrid.call()` hardcodes per-sector-wedge dropping
   with no parameter to disable it, so `sample_uniform_ut_loc()` does the drop itself,
   independently, using only the site positions from Sionna's `HexGrid`).
   `NUM_RINGS` is a plain script parameter (not hardcoded) for comparing ring counts.
   *Verified*: with `NUM_RINGS=1` (7 sites, 21 sectors, 150 UEs), median SINR rises
   monotonically from 4.2 dB (downtilt −10°) to 6.8 dB (downtilt +10°) — a real,
   physically sensible effect (more downtilt → less inter-site interference). A
   center-cell-only (single-site) version was tried first and gave an almost perfectly
   flat CDF across the same tilt sweep — correct, not a bug: with all sectors co-located
   at one site, a shared symmetric tilt changes signal and interference by nearly the same
   factor for most UEs, so it cancels in the ratio. That result is exactly why moving to
   the full ring mattered — inter-*site* interference (a different location, so a
   genuinely different elevation angle) is where tilt actually shows up.
4. **`initial_attachment.py`** (ported `RsrpBasedAttachment`) — wire it to Step 3's SINR
   values. *Verify*: every UE gets exactly one serving sector; assignment matches manual
   argmax spot-checks for a handful of UEs; reassignment changes sensibly when a sector's
   tilt is manually changed.
5. **Mobility + repeated re-attachment across slots** — run several slots, updating UE
   position each slot and rerunning Steps 3–4. *Verify*: plot serving-sector-id per UE
   over time — changes smoothly near real sector boundaries, no crashes/shape errors
   across slots.
6. **`tilt_controller.py`** — rule-based/proportional controller first. *Verify*: unit-test
   in isolation with synthetic SINR inputs (low SINR in → tilt moves the expected
   direction, output stays within configured min/max bounds), independent of the rest of
   the simulation.
7. **Full closed loop** (`run_etilt_control_loop.py`) — persistent topology, nested
   slot(fast)/epoch(slow) loop: each slot recompute channel/SINR/attachment (Steps 3–5),
   each epoch aggregate each sector's associated-UE SINR (mean or 5th-percentile) and call
   `update_tilt`, apply the new tilt to that sector for the next epoch. *Verify*: multi-
   epoch run is stable (no NaNs/divergence); tilt-vs-epoch and SINR-vs-epoch plots per
   sector show convergence, not wild oscillation.
8. *(Optional/stretch)* Compare aggregation objectives (mean vs. 5th-percentile) and
   controller gains — tuning, not correctness; not required for the loop to "work."

## Open design decisions (not yet settled)

- Single tilt shared across the whole grid vs. one independent controller per sector
  (current lean: independent per sector, matching real RET).
- SINR aggregation objective: mean vs. 5th-percentile vs. a weighted combination.
- Controller algorithm: simple rule-based/proportional first, RL later once the loop is
  validated (mirrors how real RET research has progressed — see references below).
- Whether to disable `gen_hexgrid_topology`'s automatic mechanical
  `downtilt_to_sector_center` to isolate pure electrical tilt, or treat it as a fixed
  baseline the electrical controller tunes around (currently leaning toward the latter —
  more realistic, matches how real deployments layer electrical tilt on top of a fixed
  mechanical tilt).
- Whether attachment re-evaluation happens every slot (tracks fast fading/mobility
  closely, more churn) or only once per epoch alongside the tilt update (cheaper, matches
  the two-timescale spirit more closely) — leaning toward once per epoch, revisit after
  Step 5's stability check.
