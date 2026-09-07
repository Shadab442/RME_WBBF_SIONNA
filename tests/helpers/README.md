# Helper contract verification

Black-box tests written from the supplied reference. Helper implementations were
not opened or inspected. Runtime imports, public signatures, attribute names,
outputs, and exception messages were used to construct valid fixtures. No helper
code is changed. The reference plus the user-approved conventions below define
the expected behavior.

From `sionna-rme-bbf/`:

```bash
CUDA_VISIBLE_DEVICES='' /home/shadab/venvs/sionna-wbbf/bin/python tests/helpers/run_verification.py
```

The runner uses standard-library `unittest` (no pytest installation needed),
returns a nonzero status on failure, and prints exception messages without source
tracebacks. Each `test_*.py` can also run directly, or use unittest discovery.
Direct execution/discovery uses normal tracebacks, which can display source lines.
CPU execution keeps the tiny integration tests independent of CUDA availability.

| File | Verification |
| --- | --- |
| `test_topology.py` | Layout, yaw, adjacency, neighbor IDs, hex coverage, drop radius, mirrors, grid coordinates/assignment, window membership |
| `test_electrical_downtilt.py` | Invalid arrays, rectangular peak power, signed steering, gain, idempotence |
| `test_kpi_manager.py` | Repeatable power, counterfactuals, sweep equivalence, attachment, thresholds, counts, table selection, interval pooling/reset |
| `test_spatial_grid_estimator.py` | Configuration rejection, invalid rows, selective resets, empty rewards, coverage range, EMA, window sum/mean |
| `test_tilt_controller.py` | Exhaustive global optimum, local monotonicity, dynamic warm starts, adaptive bounds, RL scheduling and evaluation |
| `test_ue_drop_mobility.py` | Uniform-area sampling, angular restriction, valid drops, warning path, cluster membership/heights, clone isolation, static offsets, periodic cycle, Markov validation |
| `test_simulation_engine.py` | Small simulation, output histories, disabled causal baseline, last-slot immobility, channel shapes/ranges and fresh shadow fading |

Fixtures use a 500 m inter-site distance; the hex circumradius is independently
computed as `500/sqrt(3)`. Sector yaw is read from `bs_orientations`. KPI positions
include a batch dimension. Pooling concatenates realization tables on the batch
axis and realization-zero samples on the UE axis. The reference did not specify
these API layout details. Mobility fixtures avoid the base-station exclusion zone.
An omnidirectional element isolates array-factor steering from element-pattern
attenuation. The engine fixture reads the public `config.yaml` and overrides sizes
and iteration counts; it does not edit the file.

This is broad contract coverage, not exhaustive verification. In particular, the
full seven-candidate wraparound oracle, per-neighbor overshoot, cluster allocation
balance, engine slot scheduling, and statistical Markov transition frequencies
are not independently verified. `utils.py` has no supplied contract. Fresh channel
states are tested through `SimulationEngine.step_environment` rather than a mocked
channel implementation.

## Accepted conventions

The user approved these corrections to the original reference:

- Steering peaks at zenith angle `90° + tilt`.
- Uncomputed grid coverage uses `-1`; computed coverage must remain in `[0,1]`.
  The test checks these rows separately so the sentinel cannot mask invalid
  computed coverage.

## Verification result

Rechecked 2026-09-04: **32 test methods passed**. No setup/import errors remain.
The RL initialization test now accepts initial internal buffer allocation and
checks that later unscheduled updates preserve the initialized state.

Passing these contracts does not cover every implementation path. See the
[repository-wide review](../../REPOSITORY_REVIEW.md) and the additional
[review regressions](../review/README.md) for remaining correctness issues.
