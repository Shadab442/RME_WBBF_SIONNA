"""Black-box tests from the supplied SectorTiltControlScheduler contract.

No scheduler source is read. Runtime construction and public outputs only.
Run directly for a compact report without implementation source tracebacks:
    python tests/drl/test_scheduler.py

The supplied contract does not specify zero- versus one-based slot indexing.
Tests accept either common origin, consistently across sectors, while requiring
exact periods and offsets after the first period of startup warmup. 'Minimum phases' is tested literally, independently
of valid coloring. Graph fixtures have symmetric adjacency and no self-loops.

Verification coverage:
- Constructor: caller adjacency is unchanged; offsets have one integer per sector
  and lie within the configured period.
- closes(): boolean output per sector, one close per period after warmup, phase
  alignment, adjacent-sector separation, and stable results for arbitrary queries.
- Sync/async: common sync phases, valid async phases across representative graphs,
  exact-fit periods, impossible clique rejection, and a one-slot edgeless case.
- Minimum phases: known graph optima and an independent exhaustive coloring oracle
  for all 1,024 labeled five-vertex graphs. This is not a proof for larger graphs.
The startup allowance does not verify an exact first-close time. The scheduler
implementation is imported and called, but its source is not inspected.
"""
import logging
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from drl.scheduler import SectorTiltControlScheduler


def graph(n, edges=()):
    adjacency = np.zeros((n,n), dtype=bool)
    for a,b in edges:
        adjacency[a,b] = adjacency[b,a] = True
    return adjacency


def minimum_colors(adjacency):
    """Independent exhaustive oracle for small graphs; no greedy heuristic."""
    n = len(adjacency)
    labels = [-1]*n

    def assign(vertex, count):
        if vertex == n:
            return True
        for color in range(count):
            if all(not adjacency[vertex,j] or labels[j] != color for j in range(vertex)):
                labels[vertex] = color
                if assign(vertex+1, count):
                    return True
        labels[vertex] = -1
        return False

    for count in range(1,n+1):
        if assign(0,count):
            return count
    raise AssertionError('No coloring found')


class SchedulerContract(unittest.TestCase):
    def make(self, adjacency, period=12, mode='async'):
        # Verify construction preserves the input graph and exposes a valid integer phase for each sector.
        original = adjacency.copy()
        scheduler = SectorTiltControlScheduler(len(adjacency), period, mode, adjacency)
        np.testing.assert_array_equal(adjacency, original,
                                      err_msg='Constructor mutated caller adjacency')
        offsets = np.asarray(scheduler.sector_offset)
        self.assertEqual(offsets.shape, (len(adjacency),))
        self.assertTrue(np.issubdtype(offsets.dtype, np.integer))
        self.assertTrue(np.all((offsets >= 0) & (offsets < period)))
        return scheduler

    def check_timing(self, scheduler, period, adjacency=None):
        # Verify public closes() outputs over multiple periods and repeated/arbitrary slot queries.
        offsets = np.array(scheduler.sector_offset, copy=True)
        n = len(offsets)
        rows = []
        for slot in range(4*period):
            result = np.asarray(scheduler.closes(slot))
            self.assertEqual(result.shape, (n,))
            self.assertEqual(result.dtype, np.dtype(bool))
            rows.append(result.copy())
        rows = np.stack(rows)
        # Skip initial warmup; the steady-state schedule must repeat every period.
        np.testing.assert_array_equal(rows[period:3*period], rows[2*period:])
        # Every sliding full-period window must contain exactly one close per sector.
        for start in range(period,3*period+1):
            np.testing.assert_array_equal(rows[start:start+period].sum(axis=0), np.ones(n))
        # A phase offset translates the common closure origin modulo the period.
        residues = np.argmax(rows[period:2*period], axis=0)
        origins = (residues-offsets) % period
        self.assertEqual(len(np.unique(origins)), 1)
        self.assertIn(int(origins[0]), (0,period-1))
        # Check actual simultaneous closures, not merely different assigned offsets.
        if adjacency is not None:
            for row in rows:
                self.assertFalse(np.any(adjacency & row[:,None] & row[None,:]),
                                 'Adjacent sectors close at the same slot')
        # Query order must not affect a schedule defined by a global index.
        for slot in (3*period-1,0,period+1,0,2*period):
            np.testing.assert_array_equal(scheduler.closes(slot), rows[slot])
        # Cumulative slot indices must preserve phase even far into a simulation.
        large = 1_000_000*period + 1
        np.testing.assert_array_equal(scheduler.closes(large), rows[period+1])
        np.testing.assert_array_equal(scheduler.sector_offset, offsets)

    def test_sync_all_sectors_close_together(self):
        # Verify sync assigns one common phase, even when all sectors are mutually adjacent.
        a = graph(5, [(i,j) for i in range(5) for j in range(i)])
        for period in (1,2,7):
            with self.subTest(period=period):
                s = self.make(a, period, 'sync')
                self.assertEqual(len(np.unique(s.sector_offset)),1)
                self.check_timing(s,period)

    def test_async_named_graphs_valid_offsets_and_periodic_closes(self):
        # Verify separation and timing on isolated, path, star, cycle, clique, and disconnected graphs.
        fixtures = [graph(1), graph(5), graph(5,[(i,i+1) for i in range(4)]),
                    graph(5,[(0,i) for i in range(1,5)]),
                    graph(4,[(0,1),(1,2),(2,3),(3,0)]),
                    graph(5,[(0,1),(1,2),(2,3),(3,4),(4,0)]),
                    graph(4,[(i,j) for i in range(4) for j in range(i)]),
                    graph(6,[(0,1),(1,2),(0,2),(3,4)])]
        for index,a in enumerate(fixtures):
            with self.subTest(graph=index):
                s = self.make(a)
                self.assertFalse(np.any(a & (s.sector_offset[:,None] == s.sector_offset[None,:])))
                self.check_timing(s,12,a)

    def test_minimum_phases_on_known_graphs(self):
        # Verify edgeless/star/odd-cycle/clique graphs use respectively 1/2/3/4 phases.
        fixtures = [(graph(5),1), (graph(6,[(0,i) for i in range(1,6)]),2),
                    (graph(5,[(i,(i+1)%5) for i in range(5)]),3),
                    (graph(4,[(i,j) for i in range(4) for j in range(i)]),4)]
        for a,expected in fixtures:
            with self.subTest(expected=expected):
                self.assertEqual(len(np.unique(self.make(a).sector_offset)),expected)

    def test_minimum_phases_against_exhaustive_small_graph_oracle(self):
        # Compare phase counts with exhaustive optima, rather than assuming any valid coloring is minimal.
        # All 1,024 labeled undirected graphs with five vertices.
        pairs = [(i,j) for i in range(5) for j in range(i)]
        for bits in range(1 << len(pairs)):
            edges = [edge for k,edge in enumerate(pairs) if bits & (1 << k)]
            a = graph(5,edges)
            actual = len(np.unique(self.make(a).sector_offset))
            expected = minimum_colors(a)
            self.assertEqual(actual,expected,
                             f'Minimum-phase contract violated: edges={edges}; used {actual}, optimum {expected}')

    def test_exactly_enough_slots_for_clique(self):
        # Verify four mutually adjacent sectors fit exactly in four distinct slots.
        a = graph(4,[(i,j) for i in range(4) for j in range(i)])
        self.check_timing(self.make(a,4),4,a)

    def test_impossible_adjacent_phase_assignment_is_rejected(self):
        # Verify four mutually adjacent sectors cannot be silently scheduled in only three slots.
        a = graph(4,[(i,j) for i in range(4) for j in range(i)])
        with self.assertRaises((ValueError,AssertionError)):
            SectorTiltControlScheduler(4,3,'async',a)

    def test_one_slot_async_for_noninterfering_sectors(self):
        # Verify all nonadjacent sectors may share the only available slot.
        a = graph(4)
        self.check_timing(self.make(a,1),1,a)


class CompactResult(unittest.TextTestResult):
    def _exc_info_to_string(self, err, test):
        # Avoid normal traceback output, which can expose implementation lines.
        return f'{err[0].__name__}: {err[1]}\n'


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2, resultclass=CompactResult))
