"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across common libraries.

    Sets seeds for:
    - Python random
    - NumPy

    Args:
        seed: The seed value.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_global_seed() -> int | None:
    """Get the current global seed.

    Returns:
        The current seed if set, None otherwise.
    """
    return getattr(get_global_seed, "_seed", None)
