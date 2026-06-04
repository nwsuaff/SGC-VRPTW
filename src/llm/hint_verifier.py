"""Hint Verifier - Validates LLM hints for safety and correctness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VerificationResult:
    """Result of hint verification."""
    valid: bool
    error: str | None = None
    warnings: list[str] | None = None


class HintVerifier:
    """Verifies LLM hints before they are used.
    
    Ensures hints are:
    - Syntactically valid
    - Semantically meaningful for VRPTW
    - Safe (no harmful operations)
    """
    
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    def verify(self, hint: dict[str, Any]) -> VerificationResult:
        """Verify a hint dictionary.
        
        Args:
            hint: Hint to verify.
        
        Returns:
            VerificationResult with status and any errors.
        """
        warnings = []
        
        # Check required fields
        if "type" not in hint:
            return VerificationResult(False, "Missing 'type' field")
        
        # Verify type-specific constraints
        hint_type = hint.get("type", "")
        
        if hint_type == "route_modification":
            if "route_id" not in hint:
                return VerificationResult(False, "Missing 'route_id' in route_modification")
            if "action" not in hint:
                return VerificationResult(False, "Missing 'action' in route_modification")
        
        elif hint_type == "customer_reassignment":
            if "customer_id" not in hint:
                return VerificationResult(False, "Missing 'customer_id'")
        
        elif hint_type == "parameter_change":
            if "param" not in hint or "value" not in hint:
                return VerificationResult(False, "Missing 'param' or 'value'")
        
        # Safety checks
        if hint_type == "code_change":
            code = hint.get("code", "")
            dangerous_patterns = ["eval(", "exec(", "import os", "import sys", "subprocess"]
            for pattern in dangerous_patterns:
                if pattern in code:
                    warnings.append(f"Dangerous pattern detected: {pattern}")
                    if self.strict:
                        return VerificationResult(False, f"Dangerous pattern: {pattern}")
        
        return VerificationResult(True, warnings=warnings if warnings else None)


def verify_hints(hints: list[dict[str, Any]], strict: bool = False) -> list[VerificationResult]:
    """Verify multiple hints.
    
    Args:
        hints: List of hint dictionaries.
        strict: Whether to enforce strict validation.
    
    Returns:
        List of VerificationResults.
    """
    verifier = HintVerifier(strict=strict)
    return [verifier.verify(hint) for hint in hints]
