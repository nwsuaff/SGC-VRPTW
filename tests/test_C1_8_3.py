"""Tests for the C1_8_3 Homberger VRPTW instance.

C1_8_3 is a Homberger 800-customer benchmark instance (clustered, type 1 time windows).
Stored at: data/VRPTW/GH800/C1_8_3.vrp
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.homberger_loader import (
    load_homberger_instance,
    _derive_family_homberger,
    _derive_size_homberger,
)
from src.domain.schema import VRPTWInstance, ValidationReport
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.utils.io import read_instance


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GH800_DIR = DATA_DIR / "VRPTW" / "GH800"
C1_8_3_PATH = GH800_DIR / "C1_8_3.vrp"

pytestmark = pytest.mark.skipif(
    not C1_8_3_PATH.exists(),
    reason="Homberger benchmark instance is not available at the configured test path.",
)


class TestC183Instance:
    """Tests for the C1_8_3 instance file."""

    def test_instance_file_exists(self):
        """The C1_8_3 instance file should exist."""
        assert C1_8_3_PATH.exists(), f"C1_8_3 instance not found at {C1_8_3_PATH}"

    def test_load_via_read_instance(self):
        """read_instance should auto-detect Homberger format."""
        instance = read_instance(str(C1_8_3_PATH))
        assert instance is not None
        assert isinstance(instance, VRPTWInstance)

    def test_load_via_homberger_loader(self):
        """homberger_loader should produce the same result."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance is not None
        assert isinstance(instance, VRPTWInstance)

    def test_instance_name(self):
        """Instance name should be C1_8_3."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.name == "C1_8_3"

    def test_instance_family(self):
        """Instance family should be C1."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.family == "C1"

    def test_instance_source(self):
        """Instance source should be homberger."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.source == "homberger"

    def test_instance_size(self):
        """Instance should have exactly 800 customers."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.size == 800

    def test_depot_id(self):
        """Depot should have id 0."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.depot_id == 0

    def test_customer_ids_count(self):
        """customer_ids list should have exactly 800 entries."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert len(instance.customer_ids) == 800
        assert instance.customer_ids == list(range(1, 801))

    def test_depot_in_coordinates(self):
        """Depot (id=0) should be present in x_coords and y_coords."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert 0 in instance.x_coords
        assert 0 in instance.y_coords

    def test_all_customers_in_coordinates(self):
        """All 800 customers should be present in x_coords and y_coords."""
        instance = load_homberger_instance(C1_8_3_PATH)
        for cid in instance.customer_ids:
            assert cid in instance.x_coords, f"Customer {cid} missing from x_coords"
            assert cid in instance.y_coords, f"Customer {cid} missing from y_coords"

    def test_vehicle_capacity(self):
        """Vehicle capacity should be 200."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.vehicle_capacity == 200

    def test_depot_demand_is_zero(self):
        """Depot demand should be 0."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.demand[0] == 0

    def test_all_demands_non_negative(self):
        """All customer demands should be non-negative."""
        instance = load_homberger_instance(C1_8_3_PATH)
        for cid in instance.customer_ids:
            assert instance.demand[cid] >= 0, f"Customer {cid} has negative demand"

    def test_demand_within_capacity(self):
        """No single customer demand should exceed vehicle capacity."""
        instance = load_homberger_instance(C1_8_3_PATH)
        for cid in instance.customer_ids:
            assert instance.demand[cid] <= instance.vehicle_capacity, (
                f"Customer {cid} demand {instance.demand[cid]} exceeds capacity"
            )

    def test_time_windows_valid(self):
        """All time windows should have ready_time <= due_time."""
        instance = load_homberger_instance(C1_8_3_PATH)
        for cid in instance.customer_ids:
            ready = instance.ready_time.get(cid, 0)
            due = instance.due_time.get(cid, 999999)
            assert ready <= due, f"Customer {cid}: ready_time({ready}) > due_time({due})"

    def test_depot_time_window(self):
        """Depot should have a valid time window (ready=0, due large)."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.ready_time[0] == 0
        assert instance.due_time[0] == 999999

    def test_distance_matrix_precomputed(self):
        """Distance matrix should be precomputed and symmetric."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.distance_matrix is not None
        n = instance.num_nodes()
        assert len(instance.distance_matrix) == n
        for i in range(n):
            assert len(instance.distance_matrix[i]) == n
            for j in range(n):
                assert instance.distance_matrix[i][j] >= 0
                assert abs(instance.distance_matrix[i][j] - instance.distance_matrix[j][i]) < 1e-9

    def test_distance_matrix_matches_euclidean(self):
        """Precomputed distances should match Euclidean calculation."""
        instance = load_homberger_instance(C1_8_3_PATH)
        n = instance.num_nodes()
        for i in range(min(n, 20)):
            for j in range(n):
                expected = instance.dist(i, j)
                actual = instance.distance_matrix[i][j]
                assert abs(expected - actual) < 1e-6, (
                    f"Distance mismatch at ({i},{j}): expected {expected}, got {actual}"
                )

    def test_distance_is_symmetric(self):
        """Distance should be symmetric: dist(i,j) == dist(j,i)."""
        instance = load_homberger_instance(C1_8_3_PATH)
        n = instance.num_nodes()
        for i in range(n):
            for j in range(i + 1, n):
                dij = instance.dist(i, j)
                dji = instance.dist(j, i)
                assert abs(dij - dji) < 1e-9, f"Distance asymmetry: ({i},{j})={dij} vs ({j},{i})={dji}"

    def test_distance_diagonal_zero(self):
        """Distance from a node to itself should be zero."""
        instance = load_homberger_instance(C1_8_3_PATH)
        n = instance.num_nodes()
        for i in range(n):
            assert instance.dist(i, i) == 0.0

    def test_objective_lexicographic(self):
        """Default objective should be lexicographic (minimize vehicles, then distance)."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.objective == "lexicographic_vehicles_distance"

    def test_vehicle_count_from_file(self):
        """Vehicle count from file header should be 200."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.vehicle_count == 200

    def test_num_nodes(self):
        """num_nodes() should return size + 1 (depot)."""
        instance = load_homberger_instance(C1_8_3_PATH)
        assert instance.num_nodes() == 801

    def test_get_all_node_ids(self):
        """get_all_node_ids should return [0] + customer_ids."""
        instance = load_homberger_instance(C1_8_3_PATH)
        all_ids = instance.get_all_node_ids()
        assert all_ids[0] == 0
        assert len(all_ids) == 801
        assert set(all_ids) == set(instance.customer_ids) | {0}

    def test_service_times_present(self):
        """All customers should have service_time values."""
        instance = load_homberger_instance(C1_8_3_PATH)
        for cid in instance.customer_ids:
            assert cid in instance.service_time, f"Customer {cid} missing service_time"
            assert instance.service_time[cid] >= 0


class TestC183FilenameParsing:
    """Tests for filename-derived properties of C1_8_3."""

    def test_derive_family(self):
        """_derive_family_homberger should return C1 for C1_8_3."""
        assert _derive_family_homberger("C1_8_3") == "C1"

    def test_derive_size(self):
        """_derive_size_homberger should return 800 for C1_8_3."""
        assert _derive_size_homberger("C1_8_3") == 800


class TestC183Solver:
    """Tests for solving the C1_8_3 instance with OR-Tools."""

    def test_ortools_finds_feasible_solution(self):
        """OR-Tools should find a feasible solution for C1_8_3 within time limit."""
        instance = load_homberger_instance(C1_8_3_PATH)
        solution, _meta = solve_ortools(instance, time_limit=30)

        assert solution is not None
        assert solution.routes is not None
        assert len(solution.routes) > 0
        assert solution.feasible is True or solution.vehicles_used > 0

    def test_solution_routes_valid(self):
        """Solution routes should cover all 800 customers exactly once."""
        instance = load_homberger_instance(C1_8_3_PATH)
        solution, _meta = solve_ortools(instance, time_limit=30)

        visited = []
        for route in solution.routes:
            visited.extend(route)

        assert len(visited) == instance.size, (
            f"Solution visits {len(visited)} customers, expected {instance.size}"
        )
        assert set(visited) == set(range(1, 801)), "Solution should visit each customer exactly once"

    def test_solution_respects_capacity(self):
        """Solution should not exceed vehicle capacity on any route."""
        instance = load_homberger_instance(C1_8_3_PATH)
        solution, _meta = solve_ortools(instance, time_limit=30)
        report = check_feasibility(instance, solution)

        assert report.capacity_violations == 0, (
            f"Solution has {report.capacity_violations} capacity violations"
        )

    def test_solution_validated_feasible(self):
        """Solution should pass full feasibility validation."""
        instance = load_homberger_instance(C1_8_3_PATH)
        solution, _meta = solve_ortools(instance, time_limit=30)
        report = check_feasibility(instance, solution)

        assert report.feasible, (
            f"Solution infeasible: "
            f"{report.capacity_violations} cap violations, "
            f"{report.late_violations} time violations"
        )
