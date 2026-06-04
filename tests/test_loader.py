"""Tests for data loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset_registry import DatasetRegistry
from src.data.split_manager import SplitManager, Split
from src.data.solomon_loader import load_solomon_instance, _derive_family
from src.data.homberger_loader import _derive_family_homberger, _derive_size_homberger
from src.utils.io import read_instance


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class TestSolomonLoader:
    """Tests for the Solomon loader."""

    def test_derive_family_c1(self):
        assert _derive_family("C101") == "C1"
        assert _derive_family("C208") == "C2"

    def test_derive_family_r1(self):
        assert _derive_family("R101") == "R1"
        assert _derive_family("R211") == "R2"

    def test_derive_family_rc(self):
        assert _derive_family("RC101") == "RC1"
        assert _derive_family("RC208") == "RC2"

    def test_derive_family_unknown(self):
        assert _derive_family("") is None
        assert _derive_family("UNKNOWN") == "UNKNOWN1"

    def test_load_toy_as_solomon(self):
        """Loading the toy instance via solomon loader should work."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)
        assert instance.name == "vrptw_tiny"


class TestHombergerLoader:
    """Tests for the Homberger loader."""

    def test_derive_family_homberger(self):
        assert _derive_family_homberger("C1_2_1") == "C1"
        assert _derive_family_homberger("RC2_8_4") == "RC2"
        assert _derive_family_homberger("R1_4_2") == "R1"

    def test_derive_size_homberger(self):
        assert _derive_size_homberger("C1_2_1") == 200
        assert _derive_size_homberger("RC2_8_4") == 800
        assert _derive_size_homberger("R1_4_2") == 400


class TestDatasetRegistry:
    """Tests for the dataset registry."""

    def test_registry_starts_empty(self):
        """Registry should start empty or load from manifest."""
        registry = DatasetRegistry()
        assert len(registry) == 0

    def test_registry_load_toy_instance(self):
        """Registry should be able to load the toy instance."""
        registry = DatasetRegistry()
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"

        loaded = registry.load_from_json(toy_path)
        assert loaded is True
        assert len(registry) == 1

    def test_registry_get_instance(self):
        """Registry should retrieve loaded instances by name."""
        registry = DatasetRegistry()
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"

        registry.load_from_json(toy_path)
        inst = registry.get_instance("vrptw_tiny")
        assert inst is not None
        assert inst["size"] == 4

    def test_registry_filter_by_source(self):
        """Registry should filter by source."""
        registry = DatasetRegistry()
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"

        registry.load_from_json(toy_path)
        toy_instances = registry.filter_by_source("toy")
        assert len(toy_instances) >= 1

    def test_registry_filter_by_family(self):
        """Registry should filter by family (toy has no family)."""
        registry = DatasetRegistry()
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"

        registry.load_from_json(toy_path)
        instances = registry.filter_by_family("C1")
        assert len(instances) == 0


class TestSplitManager:
    """Tests for the split manager."""

    def test_split_manager_loads(self):
        """Split manager should load from configs/splits/."""
        splits_dir = Path(__file__).parent.parent / "configs" / "splits"
        manager = SplitManager(splits_dir)

        split_names = manager.list_splits()
        assert "debug" in split_names
        assert "family_shift" in split_names
        assert "size_shift" in split_names

    def test_get_debug_split(self):
        """Should retrieve the debug split."""
        splits_dir = Path(__file__).parent.parent / "configs" / "splits"
        manager = SplitManager(splits_dir)

        split = manager.get_split("debug")
        assert split is not None
        assert split.name == "debug"
        assert "C101" in split.instances

    def test_get_family_shift_split(self):
        """Should retrieve the family shift split."""
        splits_dir = Path(__file__).parent.parent / "configs" / "splits"
        manager = SplitManager(splits_dir)

        split = manager.get_split("family_shift")
        assert split is not None
        assert split.seen_families == ["C1", "R1", "RC1"]
        assert split.unseen_families == ["C2", "R2", "RC2"]

    def test_get_unknown_split(self):
        """Should return None for unknown splits."""
        splits_dir = Path(__file__).parent.parent / "configs" / "splits"
        manager = SplitManager(splits_dir)

        split = manager.get_split("nonexistent")
        assert split is None
