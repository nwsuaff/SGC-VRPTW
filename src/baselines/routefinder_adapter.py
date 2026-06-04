"""RouteFinder adapter.

This is a thin adapter for the RouteFinder neural construction heuristic.
In v1, it is a placeholder that checks for the external dependency.
Actual RouteFinder integration is stage 2.
"""

from __future__ import annotations

import os
from pathlib import Path


_CHECKPOINT_ENV = "ROUTEFINDER_CHECKPOINT"
_CHECKPOINT_DEFAULT = Path.home() / ".routefinder" / "checkpoints" / "vrptw.pt"


def is_routefinder_available() -> bool:
    """Check whether a RouteFinder installation and checkpoint are available.

    Returns:
        True if the external RouteFinder dependency is present.
    """
    checkpoint = os.environ.get(_CHECKPOINT_ENV, str(_CHECKPOINT_DEFAULT))
    return Path(checkpoint).exists()


def get_routefinder_info() -> dict[str, str]:
    """Return RouteFinder configuration information.

    Returns:
        Dict with 'checkpoint_path', 'dataset_path', and 'available' keys.
    """
    checkpoint = os.environ.get(_CHECKPOINT_ENV, str(_CHECKPOINT_DEFAULT))
    return {
        "checkpoint_path": checkpoint,
        "available": str(is_routefinder_available()),
        "installation_note": (
            "RouteFinder is a neural construction heuristic. "
            "Install from: https://github.com/your-org/RouteFinder "
            "and set ROUTEFINDER_CHECKPOINT environment variable."
        ),
    }


def solve_with_routefinder(
    instance_path: str | Path,
    checkpoint_path: str | None = None,
    **kwargs,
) -> dict:
    """Run RouteFinder on a VRPTW instance.

    Args:
        instance_path: Path to the VRPTW instance JSON file.
        checkpoint_path: Optional path to the RouteFinder checkpoint.
            Defaults to ROUTEFINDER_CHECKPOINT env var or ~/.routefinder/.

    Raises:
        NotImplementedError: Always raised in v1.
        RuntimeError: If the RouteFinder checkpoint is not found.
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(_CHECKPOINT_ENV, str(_CHECKPOINT_DEFAULT))
    else:
        checkpoint_path = str(checkpoint_path)

    if not Path(checkpoint_path).exists():
        raise RuntimeError(
            f"RouteFinder checkpoint not found at '{checkpoint_path}'. "
            f"Set the { _CHECKPOINT_ENV} environment variable or install RouteFinder. "
            "See: https://github.com/your-org/RouteFinder"
        )

    raise NotImplementedError(
        "RouteFinder integration is planned for stage 2. "
        "The full adapter (checkpoint loading, inference, route conversion) "
        "will be implemented in a future version."
    )
