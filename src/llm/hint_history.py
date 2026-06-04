"""Hint History - Tracks accepted/rejected hints and their outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class HintRecord:
    """Record of a hint interaction."""
    timestamp: str
    hint: dict[str, Any]
    accepted: bool
    outcome: str | None = None
    improvement: float | None = None
    notes: str | None = None


@dataclass
class HintHistory:
    """History of hints for a single instance."""
    instance_name: str
    records: list[HintRecord] = field(default_factory=list)
    
    def add(self, hint: dict[str, Any], accepted: bool, outcome: str | None = None) -> None:
        """Add a hint record."""
        record = HintRecord(
            timestamp=datetime.now().isoformat(),
            hint=hint,
            accepted=accepted,
            outcome=outcome,
        )
        self.records.append(record)
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate."""
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.accepted) / len(self.records)
    
    @property
    def total_hints(self) -> int:
        return len(self.records)
    
    def get_accepted(self) -> list[HintRecord]:
        return [r for r in self.records if r.accepted]
    
    def get_rejected(self) -> list[HintRecord]:
        return [r for r in self.records if not r.accepted]


class HintHistoryManager:
    """Manages hint histories across multiple instances."""
    
    def __init__(self):
        self._histories: dict[str, HintHistory] = {}
    
    def get_or_create(self, instance_name: str) -> HintHistory:
        """Get or create history for an instance."""
        if instance_name not in self._histories:
            self._histories[instance_name] = HintHistory(instance_name)
        return self._histories[instance_name]
    
    def add_hint(
        self,
        instance_name: str,
        hint: dict[str, Any],
        accepted: bool,
        outcome: str | None = None,
    ) -> None:
        """Add a hint record to an instance's history."""
        history = self.get_or_create(instance_name)
        history.add(hint, accepted, outcome)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics."""
        total = sum(h.total_hints for h in self._histories.values())
        accepted = sum(len(h.get_accepted()) for h in self._histories.values())
        
        return {
            "total_instances": len(self._histories),
            "total_hints": total,
            "accepted_hints": accepted,
            "acceptance_rate": accepted / total if total > 0 else 0.0,
        }
