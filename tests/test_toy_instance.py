"""Tests for toy instance loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.schema import VRPTWInstance, RouteSolution
from src.utils.io import read_instance, read_solution


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TOY_DIR = DATA_DIR / "toy"


class TestToyInstance:
    """Tests for the toy VRPTW instance."""

    def test_toy_instance_exists(self):
        """The toy instance file should exist."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        assert toy_path.exists(), f"Toy instance not found at {toy_path}"

    def test_toy_instance_loads(self):
        """The toy instance should load without errors."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)
        assert instance.name == "vrptw_tiny"

    def test_toy_schema_validates(self):
        """The toy instance should pass Pydantic validation."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        assert instance.size == 4
        assert instance.depot_id == 0
        assert len(instance.customer_ids) == instance.size
        assert set(instance.customer_ids) == {1, 2, 3, 4}
        assert instance.vehicle_capacity == 20
        assert instance.source == "toy"

    def test_toy_instance_has_valid_coords(self):
        """The toy instance should have valid coordinates."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        assert 0 in instance.x_coords
        assert 0 in instance.y_coords
        for cid in instance.customer_ids:
            assert cid in instance.x_coords
            assert cid in instance.y_coords

    def test_toy_instance_has_time_windows(self):
        """The toy instance should have valid time windows."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        for cid in instance.customer_ids:
            ready = instance.ready_time.get(cid, 0)
            due = instance.due_time.get(cid, 999999)
            assert ready <= due, f"Customer {cid}: ready_time > due_time"

    def test_toy_instance_has_demands(self):
        """The toy instance should have valid demands."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        total_demand = sum(instance.demand.get(cid, 0) for cid in instance.customer_ids)
        assert total_demand <= instance.vehicle_capacity * 2, "Total demand should be reasonable"

    def test_expected_solution_loads(self):
        """The expected solution should load and have valid structure."""
        sol_path = TOY_DIR / "expected_tiny_solution.json"
        assert sol_path.exists()

        solution = read_solution(sol_path)
        assert solution.vehicles_used == 1
        assert solution.feasible is True

    def test_depot_in_all_coords(self):
        """Depot should be present in x_coords and y_coords."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        assert instance.depot_id in instance.x_coords
        assert instance.depot_id in instance.y_coords

    def test_distance_computation(self):
        """The instance should compute distances correctly."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        d = instance.dist(0, 1)
        assert d >= 0
        assert d < 1000

    def test_all_node_ids(self):
        """get_all_node_ids should return depot + all customers."""
        toy_path = TOY_DIR / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        all_ids = instance.get_all_node_ids()
        assert all_ids[0] == 0
        assert len(all_ids) == instance.size + 1
        assert set(all_ids) == set(instance.customer_ids) | {0}
