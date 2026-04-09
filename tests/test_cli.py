"""
Tests for robo_eval/cli.py — CLI command logic.

Covers:
- _required_sim_workers() — compute sim workers for various configs
- _version_callback behavior
- Module-level constants/config
"""

from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# _required_sim_workers
# ---------------------------------------------------------------------------

class TestRequiredSimWorkers:
    def _call(
        self,
        parallel: bool = True,
        tasks_parallel: int = 10,
        num_tasks: int = 10,
        suites_parallel: Optional[int] = None,
        num_suites: int = 1,
        debug_window: bool = False,
    ) -> int:
        from robo_eval.cli import _required_sim_workers
        return _required_sim_workers(
            parallel=parallel,
            tasks_parallel=tasks_parallel,
            num_tasks=num_tasks,
            suites_parallel=suites_parallel,
            num_suites=num_suites,
            debug_window=debug_window,
        )

    def test_sequential_mode(self):
        # Sequential always needs 1 worker
        assert self._call(parallel=False) == 1

    def test_debug_window(self):
        # Debug window always needs 1 worker
        assert self._call(debug_window=True, parallel=True, tasks_parallel=10) == 1

    def test_parallel_default(self):
        # 10 tasks, 10 tasks_parallel -> 10 workers
        assert self._call(parallel=True, tasks_parallel=10, num_tasks=10) == 10

    def test_parallel_fewer_tasks_than_parallel(self):
        # Only 3 tasks, tasks_parallel=10 -> min(10, 3) = 3
        assert self._call(parallel=True, tasks_parallel=10, num_tasks=3) == 3

    def test_parallel_limited_tasks_parallel(self):
        # 10 tasks, but only tasks_parallel=5 -> 5
        assert self._call(parallel=True, tasks_parallel=5, num_tasks=10) == 5

    def test_suites_parallel_single_suite(self):
        # Even if suites_parallel=3, with only 1 suite it's still 1x
        assert self._call(
            parallel=True, tasks_parallel=10, num_tasks=10,
            suites_parallel=3, num_suites=1,
        ) == 10

    def test_suites_parallel_multiple_suites(self):
        # 2 suites in parallel, 10 workers each -> 20
        assert self._call(
            parallel=True, tasks_parallel=10, num_tasks=10,
            suites_parallel=2, num_suites=4,
        ) == 20

    def test_suites_parallel_capped_by_num_suites(self):
        # suites_parallel=5 but only 2 suites -> 2 concurrent suites
        assert self._call(
            parallel=True, tasks_parallel=10, num_tasks=10,
            suites_parallel=5, num_suites=2,
        ) == 20

    def test_tasks_parallel_one(self):
        # tasks_parallel=1 -> 1 worker per suite
        assert self._call(parallel=True, tasks_parallel=1, num_tasks=10) == 1

    def test_combined_small(self):
        # 5 tasks, 3 tasks_parallel, 2 suites parallel -> min(3,5) * min(2,2) = 3 * 2 = 6
        assert self._call(
            parallel=True, tasks_parallel=3, num_tasks=5,
            suites_parallel=2, num_suites=2,
        ) == 6


# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------

class TestVersionInfo:
    def test_version_exists(self):
        from robo_eval import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        # Should look like a semver
        parts = __version__.split(".")
        assert len(parts) >= 2
