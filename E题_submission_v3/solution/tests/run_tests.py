"""Run revised protocol regression tests; historical data audits live in run_tests_legacy.py."""
import unittest
from tests.test_revision import RevisionTests
from tests.test_optimization import OptimizationTests
from tests.test_distillation import DistillationTests
if __name__ == "__main__":
    unittest.main()
