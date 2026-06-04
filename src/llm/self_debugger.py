"""Self-Debugger for DRoC - Automatic code error correction.

This module implements the self-debugging capability for the DRoC
pipeline. When generated code has errors, this debugger analyzes the
error and generates a fix using the LLM and constraint context.

Based on DRoC's self_debug() function.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance
    from src.llm.code_summarizer import ConstraintContext

logger = logging.getLogger(__name__)


@dataclass
class DebugResult:
    """Result of a self-debug attempt."""
    
    success: bool
    fixed_code: str
    fixed_imports: str
    error_type: str | None = None
    error_message: str | None = None
    explanation: str = ""
    attempt_count: int = 0
    runtime_ms: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        return bool(self.fixed_code and self.fixed_imports)


@dataclass
class ErrorAnalysis:
    """Analysis of an error in generated code."""
    
    error_type: str
    error_message: str
    likely_cause: str
    suggested_fix: str
    constraint_affected: str | None = None


class SelfDebugger:
    """Self-debugger for LLM-generated solver code.
    
    This debugger:
    1. Analyzes errors in generated code
    2. Determines the best fix strategy
    3. Calls LLM to fix the code with constraint context
    4. Validates the fix
    
    Example:
        debugger = SelfDebugger(llm_client=client)
        result = debugger.debug(
            broken_code=code,
            error="IndexError: list index out of range",
            instance=instance,
            constraint_context=ctx,
        )
    """
    
    # Common error patterns and their analysis
    ERROR_PATTERNS = {
        "IndexError": {
            "likely_cause": "Incorrect index handling, off-by-one errors, or matrix dimension mismatch",
            "suggested_fix": "Check array bounds and index calculations",
            "priority_constraint": "Time Windows",
        },
        "TypeError": {
            "likely_cause": "Type mismatch in function calls or operations",
            "suggested_fix": "Ensure correct types for all operations",
            "priority_constraint": None,
        },
        "NameError": {
            "likely_cause": "Undefined variable or incorrect variable name",
            "suggested_fix": "Check variable names and ensure all are defined",
            "priority_constraint": None,
        },
        "SyntaxError": {
            "likely_cause": "Missing brackets, colons, or incorrect Python syntax",
            "suggested_fix": "Fix Python syntax errors",
            "priority_constraint": None,
        },
        "AttributeError": {
            "likely_cause": "Incorrect use of OR-Tools API, missing attributes",
            "suggested_fix": "Check OR-Tools API documentation for correct usage",
            "priority_constraint": None,
        },
        "ValueError": {
            "likely_cause": "Invalid value passed to function or constraint",
            "suggested_fix": "Check valid value ranges and constraints",
            "priority_constraint": "Time Windows",
        },
        "ImportError": {
            "likely_cause": "Missing or incorrect import statement",
            "suggested_fix": "Fix import statements for ortools modules",
            "priority_constraint": None,
        },
        "KeyError": {
            "likely_cause": "Missing key in dictionary access",
            "suggested_fix": "Check dictionary keys before access",
            "priority_constraint": "Multiple Depots",
        },
    }
    
    SYSTEM_PROMPT = """You are an expert in Python programming for operations research by calling {solver}.
Your responsibility is to debug the code snippet with errors and fix them.

When fixing code, you must:
1. Analyze the error and understand its cause
2. Use the provided context to understand correct implementation
3. Fix the code to resolve the error
4. Return the complete fixed function
5. Ensure all imports are correct"""

    USER_PROMPT_TEMPLATE = """Fix the following Python code that has an error.

## The Problem
Problem name: {problem_name}
Solver: {solver}

## Broken Code
```python
{broken_code}
```

## Error Information
Error Type: {error_type}
Error Message: {error_message}

## Constraint Context (for reference)
{constraint_context}

## Instructions
1. Analyze the error and determine its cause
2. Fix the code to resolve the error
3. Ensure the fix is consistent with the constraint implementation
4. Return the complete fixed code with all imports

## Output Format
Structure your answer with:
1. Brief explanation of the error and fix strategy
2. Required imports
3. Complete functioning code block

IMPORTANT: Output ONLY the imports and code. No additional explanation outside the structure."""

    def __init__(
        self,
        llm_client: Any,
        solver: str = "OR-tools",
        max_attempts: int = 4,
    ):
        """Initialize the self-debugger.
        
        Args:
            llm_client: LLM client for generating fixes.
            solver: Solver backend ("OR-tools" or "Gurobi").
            max_attempts: Maximum number of debug attempts.
        """
        self.llm_client = llm_client
        self.solver = solver
        self.max_attempts = max_attempts
    
    def debug(
        self,
        broken_code: str,
        error_message: str,
        instance: "VRPTWInstance | None" = None,
        constraint_context: dict[str, "ConstraintContext"] | None = None,
        imports: str = "",
        error_type: str | None = None,
    ) -> DebugResult:
        """Debug and fix broken code.
        
        Args:
            broken_code: The code with errors.
            error_message: The error message from execution.
            instance: The VRPTW instance (optional).
            constraint_context: RAG context for constraints (optional).
            imports: Import statements.
            error_type: Type of error (auto-detected if None).
        
        Returns:
            DebugResult with fixed code.
        """
        start_time = time.perf_counter()
        
        # Analyze the error
        if error_type is None:
            error_type = self._detect_error_type(error_message)
        
        error_analysis = self._analyze_error(error_message, error_type)
        
        logger.info(f"Error analysis: {error_analysis.likely_cause}")
        
        # Build constraint context string
        context_str = self._build_context_string(constraint_context)
        
        # Build the problem name
        problem_name = instance.name if instance else "VRPTW"
        
        # Build the prompt
        full_code = imports + "\n" + broken_code if imports else broken_code
        
        prompt = self.USER_PROMPT_TEMPLATE.format(
            problem_name=problem_name,
            solver=self.solver,
            broken_code=full_code,
            error_type=error_type,
            error_message=error_message,
            constraint_context=context_str or "No specific context available.",
        )
        
        # Call LLM to fix
        for attempt in range(self.max_attempts):
            try:
                response = self._call_llm(prompt)
                fixed = self._parse_response(response)
                
                runtime = (time.perf_counter() - start_time) * 1000
                
                return DebugResult(
                    success=True,
                    fixed_code=fixed["code"],
                    fixed_imports=fixed["imports"],
                    error_type=error_type,
                    error_message=error_message,
                    explanation=fixed.get("explanation", ""),
                    attempt_count=attempt + 1,
                    runtime_ms=runtime,
                )
                
            except Exception as e:
                logger.warning(f"Debug attempt {attempt + 1} failed: {e}")
                # Update prompt with more specific guidance
                prompt = self._update_prompt_with_feedback(
                    prompt, str(e), attempt
                )
        
        # All attempts failed
        runtime = (time.perf_counter() - start_time) * 1000
        
        return DebugResult(
            success=False,
            fixed_code="",
            fixed_imports="",
            error_type=error_type,
            error_message=error_message,
            explanation=f"Failed to fix after {self.max_attempts} attempts",
            attempt_count=self.max_attempts,
            runtime_ms=runtime,
        )
    
    def debug_simple(
        self,
        broken_code: str,
        error_message: str,
        imports: str = "",
    ) -> DebugResult:
        """Simple debugging without context (fallback mode).
        
        Args:
            broken_code: The code with errors.
            error_message: The error message.
            imports: Import statements.
        
        Returns:
            DebugResult with fixed code.
        """
        return self.debug(
            broken_code=broken_code,
            error_message=error_message,
            instance=None,
            constraint_context=None,
            imports=imports,
        )
    
    def _detect_error_type(self, error_message: str) -> str:
        """Detect the error type from the error message."""
        message_lower = error_message.lower()
        
        # Check each pattern
        for pattern, info in self.ERROR_PATTERNS.items():
            if pattern.lower() in message_lower:
                return pattern
        
        # Default to generic error
        if "error" in message_lower:
            return "RuntimeError"
        
        return "UnknownError"
    
    def _analyze_error(self, error_message: str, error_type: str) -> ErrorAnalysis:
        """Analyze an error and provide fix guidance."""
        pattern_info = self.ERROR_PATTERNS.get(
            error_type,
            {
                "likely_cause": "Unknown error",
                "suggested_fix": "Review the error message and code carefully",
                "priority_constraint": None,
            }
        )
        
        return ErrorAnalysis(
            error_type=error_type,
            error_message=error_message,
            likely_cause=pattern_info["likely_cause"],
            suggested_fix=pattern_info["suggested_fix"],
            constraint_affected=pattern_info.get("priority_constraint"),
        )
    
    def _build_context_string(
        self,
        constraint_context: dict[str, "ConstraintContext"] | None,
    ) -> str:
        """Build context string from constraint contexts."""
        if not constraint_context:
            return ""
        
        parts = []
        for keyword, ctx in constraint_context.items():
            if ctx.is_empty:
                continue
            parts.append(f"## {keyword}\n{ctx.combined_context}\n")
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to generate a fix."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(solver=self.solver)},
            {"role": "user", "content": prompt},
        ]
        
        if hasattr(self.llm_client, "invoke"):
            response = self.llm_client.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        elif hasattr(self.llm_client, "generate_code"):
            return self.llm_client.generate_code(prompt)
        else:
            raise NotImplementedError("LLM client doesn't support code generation")
    
    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse LLM response to extract fixed code."""
        import re
        
        code = response
        imports = ""
        explanation = ""
        
        # Extract explanation
        explanation_pattern = r"(?:explanation|reason|analysis|description):(.+?)(?=(?:```|import|def |class |important))"
        explanation_matches = re.findall(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        if explanation_matches:
            explanation = explanation_matches[0].strip()[:200]
        
        # Extract imports
        import_pattern = r"(?:from ortools[^\n]+\n|import ortools[^\n]+\n|import [^\n]+\n|from [^\n]+\n)+"
        import_matches = re.findall(import_pattern, response)
        if import_matches:
            imports = "".join(import_matches).strip()
            code = re.sub(import_pattern, "", code)
        
        # Extract code blocks
        code_pattern = r"```(?:\w+)?\n?(.*?)```"
        code_matches = re.findall(code_pattern, code, re.DOTALL)
        if code_matches:
            code = max(code_matches, key=len).strip()
        
        # Clean up
        code = code.strip()
        
        return {
            "code": code,
            "imports": imports,
            "explanation": explanation,
        }
    
    def _update_prompt_with_feedback(
        self,
        original_prompt: str,
        error: str,
        attempt: int,
    ) -> str:
        """Update the prompt with feedback from failed attempt."""
        feedback = f"\n\n## Previous Attempt {attempt + 1} Failed\nThe LLM returned an invalid response: {error}\n\nPlease ensure you output valid Python code with imports."
        return original_prompt + feedback


class DebugLoop:
    """Manages the self-debug loop with multiple iterations.
    
    This class orchestrates the self-debug process, tracking state
    across iterations and determining when to stop.
    """
    
    def __init__(
        self,
        debugger: SelfDebugger,
        max_iterations: int = 4,
    ):
        """Initialize the debug loop.
        
        Args:
            debugger: SelfDebugger instance.
            max_iterations: Maximum iterations before giving up.
        """
        self.debugger = debugger
        self.max_iterations = max_iterations
    
    def run(
        self,
        initial_code: str,
        initial_imports: str,
        error_message: str,
        instance: "VRPTWInstance | None" = None,
        constraint_context: dict[str, "ConstraintContext"] | None = None,
        execute_fn: Any = None,
    ) -> tuple[str, str, bool]:
        """Run the self-debug loop.
        
        Args:
            initial_code: The initial broken code.
            initial_imports: Import statements.
            error_message: The error message.
            instance: The VRPTW instance.
            constraint_context: RAG context for constraints.
            execute_fn: Optional function to execute and validate code.
        
        Returns:
            Tuple of (fixed_code, fixed_imports, success).
        """
        current_code = initial_code
        current_imports = initial_imports
        current_error = error_message
        
        for iteration in range(self.max_iterations):
            logger.info(f"Debug iteration {iteration + 1}/{self.max_iterations}")
            
            # Debug the code
            result = self.debugger.debug(
                broken_code=current_code,
                error_message=current_error,
                instance=instance,
                constraint_context=constraint_context,
                imports=current_imports,
            )
            
            if result.success and result.is_valid:
                # Validate if execution function provided
                if execute_fn:
                    try:
                        exec_result = execute_fn(
                            result.fixed_imports + "\n" + result.fixed_code
                        )
                        if exec_result.success:
                            logger.info("Debug succeeded and code validated")
                            return result.fixed_code, result.fixed_imports, True
                        else:
                            current_error = exec_result.error_message or "Execution failed"
                            current_code = result.fixed_code
                            current_imports = result.fixed_imports
                            logger.warning("Code fixed but execution failed, continuing...")
                    except Exception as e:
                        current_error = str(e)
                        current_code = result.fixed_code
                        current_imports = result.fixed_imports
                        logger.warning(f"Validation failed: {e}, continuing...")
                else:
                    # No validation, trust the fix
                    logger.info("Debug succeeded (no validation)")
                    return result.fixed_code, result.fixed_imports, True
            else:
                logger.warning(f"Debug attempt {iteration + 1} unsuccessful")
                if not result.success:
                    current_error = result.explanation
                else:
                    current_error = "Invalid fixed code"
        
        logger.error(f"Debug loop failed after {self.max_iterations} iterations")
        return initial_code, initial_imports, False


# Convenience functions
def self_debug(
    code: str,
    error: str,
    llm_client: Any,
    solver: str = "OR-tools",
) -> DebugResult:
    """Simple self-debug function.
    
    Args:
        code: The broken code.
        error: The error message.
        llm_client: LLM client.
        solver: Solver backend.
    
    Returns:
        DebugResult with fixed code.
    """
    debugger = SelfDebugger(llm_client=llm_client, solver=solver)
    return debugger.debug_simple(broken_code=code, error_message=error)


def self_debug_with_context(
    code: str,
    error: str,
    instance: "VRPTWInstance",
    constraint_context: dict[str, "ConstraintContext"],
    llm_client: Any,
    solver: str = "OR-tools",
) -> DebugResult:
    """Self-debug with constraint context.
    
    Args:
        code: The broken code.
        error: The error message.
        instance: The VRPTW instance.
        constraint_context: RAG context for constraints.
        llm_client: LLM client.
        solver: Solver backend.
    
    Returns:
        DebugResult with fixed code.
    """
    debugger = SelfDebugger(llm_client=llm_client, solver=solver)
    return debugger.debug(
        broken_code=code,
        error_message=error,
        instance=instance,
        constraint_context=constraint_context,
    )
