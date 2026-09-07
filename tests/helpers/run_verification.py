"""Run with exception messages only, without rendering implementation source."""
import sys
import unittest
from pathlib import Path

class ContractResult(unittest.TextTestResult):
    def _exc_info_to_string(self, err, test):
        return f'{err[0].__name__}: {err[1]}\n'

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2, resultclass=ContractResult).run(suite)
    sys.exit(not result.wasSuccessful())
