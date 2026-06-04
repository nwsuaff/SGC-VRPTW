"""Dataset split manager: handles train/test splits defined in YAML configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.data.dataset_registry import DatasetRegistry


@dataclass
class Split:
    """Represents a dataset split definition."""

    name: str
    instances: list[str]
    seen_families: list[str] | None = None
    unseen_families: list[str] | None = None
    train_sizes: list[int] | None = None
    test_sizes: list[int] | None = None


class SplitManager:
    """Manages dataset splits defined in YAML files."""

    def __init__(self, splits_dir: str | Path):
        """Initialize the split manager.

        Args:
            splits_dir: Directory containing YAML split definition files.
        """
        self._splits_dir = Path(splits_dir)
        self._splits: dict[str, Split] = {}
        self._load_builtin_splits()

    def _load_builtin_splits(self) -> None:
        """Load all YAML files from the splits directory."""
        if not self._splits_dir.exists():
            return
        for yaml_file in sorted(self._splits_dir.glob("*.yaml")):
            name = yaml_file.stem
            self._splits[name] = self._load_split_file(yaml_file)

    def _load_split_file(self, path: Path) -> Split:
        """Load a single YAML split file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return Split(
            name=path.stem,
            instances=data.get("instances", []),
            seen_families=data.get("seen_families"),
            unseen_families=data.get("unseen_families"),
            train_sizes=data.get("train_sizes"),
            test_sizes=data.get("test_sizes"),
        )

    def get_split(self, name: str) -> Split | None:
        """Get a split by name."""
        return self._splits.get(name)

    def list_splits(self) -> list[str]:
        """List all available split names."""
        return list(self._splits.keys())

    def get_train_instances(
        self, split_name: str, registry: DatasetRegistry
    ) -> list[str]:
        """Get training instance names for a given split.

        If the split uses family/size-based selection, queries the registry.
        If it uses an explicit instance list, returns that list.
        """
        split = self._splits.get(split_name)
        if split is None:
            return []

        if split.instances:
            return split.instances

        train_instances = []
        if split.seen_families:
            for family in split.seen_families:
                for inst in registry.filter_by_family(family):
                    train_instances.append(inst.name)
        if split.train_sizes:
            for size in split.train_sizes:
                for inst in registry.filter_by_size(size):
                    if inst.name not in train_instances:
                        train_instances.append(inst.name)

        return train_instances

    def get_test_instances(
        self, split_name: str, registry: DatasetRegistry
    ) -> list[str]:
        """Get test instance names for a given split."""
        split = self._splits.get(split_name)
        if split is None:
            return []

        test_instances = []
        if split.unseen_families:
            for family in split.unseen_families:
                for inst in registry.filter_by_family(family):
                    test_instances.append(inst.name)
        if split.test_sizes:
            for size in split.test_sizes:
                for inst in registry.filter_by_size(size):
                    if inst.name not in test_instances:
                        test_instances.append(inst.name)

        return test_instances
