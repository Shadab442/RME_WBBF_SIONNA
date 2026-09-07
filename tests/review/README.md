# Repository review regression checks

These implementation-informed checks complement the original black-box helper tests. The full findings and scope are in [REPOSITORY_REVIEW.md](../../REPOSITORY_REVIEW.md).

Run from `sionna-rme-bbf`:

```bash
/home/shadab/venvs/sionna-wbbf/bin/python tests/review/test_repository_correctness.py
/home/shadab/venvs/sionna-wbbf/bin/python -m unittest discover -s tests/helpers -v
```

Review baseline, 2026-09-04: the 15 new tests have 7 passes, 7 failures and 1 error. The existing 32 helper tests pass. The new failures are deliberate assertions of expected correct behavior, not tests that are supposed to accept the bugs; their exit status is nonzero until the implementation is fixed.

The failures cover checkpoint loading, WESN evaluation history, recurrent sequence boundaries, static cluster radius at a topology boundary, clone RNG isolation, and pooled CDF shape. Passing checks cover Double DQN targets, target synchronization, MLP learning/masking, reservoir recurrence, kriging, asynchronous occupancy, and small engine runs for MLP/WESN.

Tests use CPU and temporary output paths. They do not run the historical plotting/experiment scripts at module scope or overwrite their saved results. A pure plotting function is extracted using Python's AST to test it without triggering top-level file operations. Not every source-review finding has a regression here; see the report for source-only findings and execution limitations.
