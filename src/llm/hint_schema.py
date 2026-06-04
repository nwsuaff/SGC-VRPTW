"""Hint Schema - DEPRECATED.

This module is kept for backward compatibility with ablation experiments.
For new development, use the DRoC-style code generation approach.

DRoC Pipeline:
    - LLM generates solver code (see augmented_generator.py)
    - Code is executed and self-debugged (see self_debugger.py)
    
Hint-based Pipeline: DEPRECATED - Use DRoC instead
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

# Suppress deprecation warnings for now
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class LLMHints:
    """Structured hints from LLM - DEPRECATED, use DRoC instead."""
    
    priority_customers: list[int] = field(default_factory=list)
    locked_subroutes: list[list[int]] = field(default_factory=list)
    avoid_pairs: list[list[int]] = field(default_factory=list)
    suggested_moves: list = field(default_factory=list)
    solver_control: Optional["SolverControl"] = None
    rationale: Optional[str] = None


@dataclass
class SolverControl:
    """Solver runtime parameters - DEPRECATED."""
    extra_seconds: Optional[float] = None
    intensify: bool = False
    diversify: bool = False


@dataclass
class RelocateMove:
    """Relocate move suggestion - DEPRECATED."""
    from_route: int = 0
    to_route: int = 0
    customer: int = 0
    after_customer: int = 0


@dataclass
class SwapMove:
    """Swap move suggestion - DEPRECATED."""
    route1: int = 0
    customer1: int = 0
    route2: int = 0
    customer2: int = 0


def empty_hints() -> LLMHints:
    """Return an empty hints object."""
    return LLMHints()


__deprecated__ = True
