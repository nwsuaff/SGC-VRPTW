"""Smoke tests for the PyVRP solver."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import read_instance
from src.solvers.pyvrp_solver import solve_pyvrp
from src.solvers.feasibility_checker import check_feasibility


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(
    Path(__file__).parent.parent.joinpath("src/solvers/pyvrp_solver.py").exists() and
    not bool(__import__("importlib").util.find_spec("pyvrp")),
    reason="PyVRP not installed",
)
class TestPyVRPSmoke:
    """Smoke tests for PyVRP integration."""

    def test_pyvrp_solves_toy_instance(self):
        """PyVRP should solve the toy instance without errors."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        solution, _ = solve_pyvrp(instance, time_limit=2, seed=42)

        assert solution is not None
        assert solution.runtime_sec >= 0
        assert solution.vehicles_used >= 1
        assert solution.total_distance >= 0

    def test_pyvrp_output_normalized(self):
        """PyVRP output should be normalized to RouteSolution format."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        solution, _ = solve_pyvrp(instance, time_limit=2, seed=42)

        assert hasattr(solution, "routes")
        assert hasattr(solution, "vehicles_used")
        assert hasattr(solution, "total_distance")
        assert hasattr(solution, "feasible")
        assert hasattr(solution, "runtime_sec")

    def test_pyvrp_feasibility_check(self):
        """The feasibility checker should validate PyVRP solutions."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        solution, _ = solve_pyvrp(instance, time_limit=2, seed=42)
        report = check_feasibility(instance, solution)

        assert hasattr(report, "feasible")
        assert hasattr(report, "late_violations")
        assert hasattr(report, "capacity_violations")

    def test_pyvrp_deterministic_with_same_seed(self):
        """PyVRP should produce deterministic results with the same seed."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        s1, _ = solve_pyvrp(instance, time_limit=1, seed=42)
        s2, _ = solve_pyvrp(instance, time_limit=1, seed=42)

        assert s1.total_distance == s2.total_distance
        assert s1.vehicles_used == s2.vehicles_used
