"""File I/O utilities for the VRPTW project."""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.schema import VRPTWInstance, RouteSolution


def read_instance(path: str | Path) -> VRPTWInstance:
    """Load a VRPTWInstance from a file.

    Supports:
    - JSON files (.json) - direct VRPTWInstance format
    - Solomon format files (.vrp) - standard Solomon benchmark format
    - Homberger format files (.vrp) - extended Solomon format
    - ORTEC format files (.txt, .vrp) - EURO NeurIPS 2022 competition format
      with explicit duration matrices

    Args:
        path: Path to the instance file.

    Returns:
        VRPTWInstance parsed from the file.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        return _read_json_instance(path)
    elif suffix in (".vrp", ".txt"):
        return _read_vrp_instance(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _read_json_instance(path: Path) -> VRPTWInstance:
    """Load a VRPTWInstance from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return VRPTWInstance.model_validate(data)


def _read_vrp_instance(path: Path) -> VRPTWInstance:
    """Load a VRPTWInstance from a .vrp file (Solomon, Homberger, or ORTEC format)."""
    from src.data.solomon_loader import load_solomon_instance
    from src.data.homberger_loader import load_homberger_instance
    from src.data.ortec_loader import load_ortec_instance

    name = path.stem

    if "ORTEC" in str(path).upper() or "ortec" in path.parent.name.lower():
        return load_ortec_instance(path)
    elif "GH800" in str(path) or "homberger" in path.parent.name.lower():
        return load_homberger_instance(path)
    else:
        return load_solomon_instance(path)


def write_instance(instance: VRPTWInstance, path: str | Path) -> None:
    """Write a VRPTWInstance to a JSON file.

    Args:
        instance: The instance to write.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance.model_dump(), f, indent=2)


def read_solution(path: str | Path) -> RouteSolution:
    """Load a RouteSolution from a JSON file.

    Args:
        path: Path to a JSON file.

    Returns:
        RouteSolution parsed from the file.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RouteSolution.model_validate(data)


def write_solution(solution: RouteSolution, path: str | Path) -> None:
    """Write a RouteSolution to a JSON file.

    Args:
        solution: The solution to write.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution.model_dump(), f, indent=2)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        The Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_project_root() -> Path:
    """Return the project root directory.

    Assumes this file is at src/utils/io.py and navigates up.
    """
    return Path(__file__).resolve().parent.parent.parent
