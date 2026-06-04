"""Augmented Code Generator for DRoC - RAG-enhanced code generation.

This module implements the retrieval-augmented code generation step
in the DRoC pipeline. It combines problem templates, constraint
context from RAG, and LLM capabilities to generate solver code.

Based on DRoC's retrieval_augmented_generate() function.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.llm.vrptw_templates import (
    VRPTW_BASE_TEMPLATE,
    VRPTW_LEXICOGRAPHIC_TEMPLATE,
    instance_to_solve_params,
)

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance
    from src.llm.code_summarizer import ConstraintContext

logger = logging.getLogger(__name__)


@dataclass
class GenerationPrompt:
    """A prompt for code generation."""
    
    system_prompt: str
    user_prompt: str
    template: str
    context: str
    
    def to_messages(self) -> list[dict[str, str]]:
        """Convert to message format for LLM."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass
class GeneratedCode:
    """Result of code generation."""
    
    code: str
    imports: str
    description: str
    generation_time_ms: float = 0.0
    iteration: int = 0
    
    def is_valid(self) -> bool:
        """Check if generated code is valid."""
        return bool(self.code and self.imports)


@dataclass
class AugmentedGenerationState:
    """State during augmented code generation."""
    
    instance: "VRPTWInstance"
    keywords: list[str]
    constraint_context: dict[str, "ConstraintContext"]
    generated_code: GeneratedCode | None = None
    error: str = "no"
    messages: list[str] = field(default_factory=list)
    iterations: int = 0


class AugmentedCodeGenerator:
    """Generates solver code with RAG-enhanced context.
    
    This generator combines:
    1. Problem description from instance
    2. Code templates
    3. Retrieved constraint context from RAG
    4. LLM capabilities
    
    The result is higher quality code that addresses each constraint
    based on relevant examples from the RAG system.
    
    Example:
        generator = AugmentedCodeGenerator(llm_client=client)
        result = generator.generate(instance, constraint_context)
    """
    
    SYSTEM_PROMPT = """You are an expert in Python programming for operations research 
and combinatorial optimization. You are good at calling {solver} in Python and solving problems.

When generating code, you must:
1. Follow the template format exactly
2. Use the provided context to understand how to implement each constraint
3. Return syntactically correct Python code
4. Ensure all imports are included
5. Return the solution by the 'solve' function"""

    USER_PROMPT_TEMPLATE = """Generate Python code to solve a {problem_name} using {solver}.

## Problem Description
{instance_description}

## Required Constraints
The following constraints must be implemented based on the problem:
{constraint_list}

## Code Template
Use this template structure:
```python
{template}
```

## Context: Example Code for Constraints
The following examples show how to implement each constraint:

{constraint_context}

## Requirements
1. Read the template and understand the parameters in 'solve' function
2. Implement all constraints mentioned in the problem description
3. Follow the template format strictly
4. Return the objective value by the 'solve' function
5. Ensure all code is executable with required imports

## Output Format
Structure your answer with:
1. Brief description of the solution approach
2. Required imports (starting with 'from ortools...' or 'import ortools')
3. The complete Python code block

IMPORTANT: Output ONLY the imports and code. No additional explanation outside the structure."""

    def __init__(
        self,
        llm_client: Any,
        solver: str = "OR-tools",
        template: str | None = None,
        max_retries: int = 3,
    ):
        """Initialize the augmented code generator.
        
        Args:
            llm_client: LLM client for code generation.
            solver: Solver backend ("OR-tools" or "Gurobi").
            template: Custom code template (uses default if None).
            max_retries: Maximum retry attempts on failure.
        """
        self.llm_client = llm_client
        self.solver = solver
        self.template = template or VRPTW_BASE_TEMPLATE
        self.max_retries = max_retries
    
    def generate(
        self,
        instance: "VRPTWInstance",
        constraint_context: dict[str, "ConstraintContext"] | None = None,
        keywords: list[str] | None = None,
    ) -> GeneratedCode:
        """Generate solver code with RAG context.
        
        Args:
            instance: The VRPTW instance to solve.
            constraint_context: Retrieved context for each constraint.
            keywords: List of constraint keywords.
        
        Returns:
            GeneratedCode with code and imports.
        """
        start_time = time.perf_counter()
        
        # Build the generation prompt
        prompt = self._build_prompt(instance, constraint_context, keywords)
        
        # Generate with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                code = self._parse_response(response)
                
                generation_time = (time.perf_counter() - start_time) * 1000
                
                return GeneratedCode(
                    code=code["code"],
                    imports=code["imports"],
                    description=code.get("description", ""),
                    generation_time_ms=generation_time,
                    iteration=attempt,
                )
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                # Try to fix the prompt and retry
                prompt = self._build_prompt(
                    instance, constraint_context, keywords,
                    error_context=last_error
                )
        
        # All retries failed
        return GeneratedCode(
            code="",
            imports="",
            description=f"Generation failed after {self.max_retries} attempts: {last_error}",
            generation_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    def _build_prompt(
        self,
        instance: "VRPTWInstance",
        constraint_context: dict[str, "ConstraintContext"] | None,
        keywords: list[str] | None,
        error_context: str | None = None,
    ) -> GenerationPrompt:
        """Build the generation prompt."""
        # Build instance description
        instance_desc = self._build_instance_description(instance)
        
        # Build constraint list
        constraint_list = self._build_constraint_list(keywords or [])
        
        # Build context string
        context_str = self._build_context_string(constraint_context)
        
        # Build user prompt
        problem_name = keywords[0] if keywords else "VRPTW" if keywords is None else ", ".join(keywords)
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            problem_name=problem_name,
            solver=self.solver,
            instance_description=instance_desc,
            constraint_list=constraint_list,
            template=self.template,
            constraint_context=context_str or "No specific context available. Use standard VRPTW implementation.",
        )
        
        # Add error context if provided
        if error_context:
            user_prompt += f"\n\n## Previous Error\nThe previous attempt had this error:\n{error_context}\n\nPlease fix the code to avoid this error."
        
        return GenerationPrompt(
            system_prompt=self.SYSTEM_PROMPT.format(solver=self.solver),
            user_prompt=user_prompt,
            template=self.template,
            context=context_str,
        )
    
    def _build_instance_description(self, instance: "VRPTWInstance") -> str:
        """Build detailed instance description for prompt."""
        n = instance.size
        
        # Distance matrix preview
        if instance.distance_matrix:
            matrix_preview = "Distance matrix (first 5x5):\n"
            for i in range(min(5, len(instance.distance_matrix))):
                row = [f"{d:.1f}" for d in instance.distance_matrix[i][:5]]
                matrix_preview += f"  {row}\n"
        else:
            matrix_preview = "Distance matrix: compute from coordinates"
        
        lines = [
            f"- **Instance name**: {instance.name}",
            f"- **Number of customers**: {n}",
            f"- **Depot ID**: {instance.depot_id}",
            f"- **Vehicle capacity**: {instance.vehicle_capacity}",
            f"- **Vehicle count**: {instance.vehicle_count or 'unlimited'}",
            "",
            f"- **Customer coordinates** (sample):",
        ]
        
        # Add customer coordinates
        for cid in instance.customer_ids[:5]:
            x = instance.x_coords.get(cid, 0)
            y = instance.y_coords.get(cid, 0)
            d = instance.demand.get(cid, 0)
            lines.append(f"  Customer {cid}: ({x:.1f}, {y:.1f}), demand={d}")
        
        lines.extend([
            "",
            f"- **Time windows** (sample):",
        ])
        
        for cid in instance.customer_ids[:5]:
            ready = instance.ready_time.get(cid, 0)
            due = instance.due_time.get(cid, 999999)
            lines.append(f"  Customer {cid}: [{ready}, {due}]")
        
        lines.extend([
            "",
            "- **Service times** (sample):",
        ])
        
        for cid in instance.customer_ids[:5]:
            st = instance.service_time.get(cid, 0)
            lines.append(f"  Customer {cid}: {st}")
        
        lines.extend([
            "",
            f"- **Objective**: {instance.objective}",
            "",
            matrix_preview,
        ])
        
        return "\n".join(lines)
    
    def _build_constraint_list(self, keywords: list[str]) -> str:
        """Build constraint list from keywords."""
        if not keywords:
            return "- Capacitated (vehicle capacity constraints)\n- Time Windows (customer time window constraints)"
        
        return "\n".join(f"- {kw}" for kw in keywords)
    
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
            
            parts.append(f"\n{'=' * 50}")
            parts.append(f"## {keyword}")
            parts.append(f"{'=' * 50}")
            parts.append(ctx.combined_context)
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: GenerationPrompt) -> str:
        """Call the LLM to generate code."""
        messages = prompt.to_messages()
        
        if hasattr(self.llm_client, "invoke"):
            # LangChain style
            response = self.llm_client.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        elif hasattr(self.llm_client, "generate_code"):
            # Custom generate_code method
            full_prompt = f"{prompt.system_prompt}\n\n{prompt.user_prompt}"
            return self.llm_client.generate_code(full_prompt)
        else:
            # Generic fallback
            raise NotImplementedError(
                f"LLM client {type(self.llm_client)} doesn't support invoke() or generate_code()"
            )
    
    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse LLM response to extract code and imports."""
        import re
        
        code = response
        imports = ""
        description = ""
        
        # Extract imports
        import_pattern = r"(?:from ortools[^\n]+\n|import ortools[^\n]+\n|import [^\n]+\n|from [^\n]+\n)+"
        import_matches = re.findall(import_pattern, response)
        if import_matches:
            imports = "".join(import_matches).strip()
            # Remove imports from code
            code = re.sub(import_pattern, "", code)
        
        # Extract description
        desc_pattern = r"(?:description|approach|solution):(.+?)(?=(?:```|import|def |class ))"
        desc_matches = re.findall(desc_pattern, response, re.IGNORECASE | re.DOTALL)
        if desc_matches:
            description = desc_matches[0].strip()[:200]  # Limit length
        
        # Extract code from markdown blocks
        code_pattern = r"```(?:\w+)?\n?(.*?)```"
        code_matches = re.findall(code_pattern, code, re.DOTALL)
        if code_matches:
            # Take the longest code block
            code = max(code_matches, key=len).strip()
        
        # Clean up
        code = code.strip()
        
        return {
            "code": code,
            "imports": imports,
            "description": description,
        }


class RefinementCodeGenerator(AugmentedCodeGenerator):
    """Generator for refining existing code with RAG context.
    
    This generator takes existing code with errors and refines it
    using the constraint context from RAG.
    """
    
    REFINE_PROMPT_TEMPLATE = """Refine the following code which has an error.

## Problem Description
{instance_description}

## Broken Code
```python
{broken_code}
```

## Error Message
{error_message}

## Context: Example Code for Constraints
{constraint_context}

## Instructions
1. Analyze the error and understand what caused it
2. Use the context examples to fix the constraint implementation
3. Return the complete corrected code
4. Do not change the function signature

## Output Format
1. Brief explanation of the error and fix
2. Required imports
3. Complete corrected Python code

IMPORTANT: Output ONLY the imports and code."""

    def refine(
        self,
        instance: "VRPTWInstance",
        broken_code: str,
        error_message: str,
        constraint_context: dict[str, "ConstraintContext"] | None = None,
        keywords: list[str] | None = None,
    ) -> GeneratedCode:
        """Refine broken code using RAG context.
        
        Args:
            instance: The VRPTW instance.
            broken_code: The code with errors.
            error_message: The error message.
            constraint_context: Retrieved context for each constraint.
            keywords: List of constraint keywords.
        
        Returns:
            GeneratedCode with refined code.
        """
        start_time = time.perf_counter()
        
        # Build context string
        context_str = self._build_context_string(constraint_context)
        
        # Build instance description
        instance_desc = self._build_instance_description(instance)
        
        # Build user prompt
        user_prompt = self.REFINE_PROMPT_TEMPLATE.format(
            instance_description=instance_desc,
            broken_code=broken_code,
            error_message=error_message,
            constraint_context=context_str or "No specific context available.",
        )
        
        # Call LLM
        system_prompt = self.SYSTEM_PROMPT.format(solver=self.solver)
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            if hasattr(self.llm_client, "invoke"):
                response = self.llm_client.invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)
            else:
                response_text = self.llm_client.generate_code(f"{system_prompt}\n\n{user_prompt}")
            
            code = self._parse_response(response_text)
            
            generation_time = (time.perf_counter() - start_time) * 1000
            
            return GeneratedCode(
                code=code["code"],
                imports=code["imports"],
                description=f"Refined based on error: {error_message[:100]}",
                generation_time_ms=generation_time,
                iteration=0,
            )
            
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return GeneratedCode(
                code="",
                imports="",
                description=f"Refinement failed: {e}",
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
            )


# Convenience functions
def generate_with_context(
    instance: "VRPTWInstance",
    constraint_context: dict[str, "ConstraintContext"],
    llm_client: Any,
    solver: str = "OR-tools",
    template: str | None = None,
) -> GeneratedCode:
    """Convenience function to generate code with RAG context.
    
    Args:
        instance: The VRPTW instance.
        constraint_context: Retrieved context for each constraint.
        llm_client: LLM client.
        solver: Solver backend.
        template: Optional custom template.
    
    Returns:
        GeneratedCode with code and imports.
    """
    generator = AugmentedCodeGenerator(
        llm_client=llm_client,
        solver=solver,
        template=template,
    )
    keywords = list(constraint_context.keys())
    return generator.generate(instance, constraint_context, keywords)


def refine_with_context(
    instance: "VRPTWInstance",
    broken_code: str,
    error_message: str,
    constraint_context: dict[str, "ConstraintContext"],
    llm_client: Any,
    solver: str = "OR-tools",
) -> GeneratedCode:
    """Convenience function to refine code with RAG context.
    
    Args:
        instance: The VRPTW instance.
        broken_code: The code with errors.
        error_message: The error message.
        constraint_context: Retrieved context for each constraint.
        llm_client: LLM client.
        solver: Solver backend.
    
    Returns:
        GeneratedCode with refined code.
    """
    generator = RefinementCodeGenerator(
        llm_client=llm_client,
        solver=solver,
    )
    keywords = list(constraint_context.keys())
    return generator.refine(
        instance, broken_code, error_message, constraint_context, keywords
    )
