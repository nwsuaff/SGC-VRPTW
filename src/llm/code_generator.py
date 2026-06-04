"""Code generator for LLM-based VRPTW solving (DRoC-style).

This module generates Python solver code using an LLM with retrieval-augmented
generation (RAG) from example codes and templates.

Workflow:
1. Build prompt with problem description and templates
2. Query LLM to generate code
3. Execute code and check result
4. Self-debug if errors occur (up to max_iterations)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.domain.schema import VRPTWInstance, RouteSolution
from src.llm.vrptw_templates import (
    VRPTW_BASE_TEMPLATE,
    VRPTW_WARMSTART_TEMPLATE,
    VRPTW_LEXICOGRAPHIC_TEMPLATE,
    instance_to_solve_params,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeGenResult:
    """Result of code generation attempt."""
    
    success: bool
    code: str | None = None
    imports: str = ""
    execution_result: "ExecutionResult | None" = None
    error_message: str | None = None
    iterations: int = 0
    total_runtime_sec: float = 0.0
    llm_latency_ms: float = 0.0


@dataclass 
class GenerationState:
    """State during code generation process."""
    
    code: str
    imports: str = ""
    messages: list[str] = field(default_factory=list)
    error: str = "no"
    iterations: int = 0


class CodeGenerator:
    """DRoC-style code generator for VRPTW.
    
    This generator:
    1. Builds prompts with templates and instance data
    2. Queries LLM to generate solver code
    3. Executes code and validates results
    4. Self-debugs on errors (up to max_iterations)
    """
    
    def __init__(
        self,
        llm_client: Any,
        max_iterations: int = 4,
        enable_self_debug: bool = True,
        enable_rag: bool = True,
        objective: str = "lexicographic",
        exec_timeout: float = 60.0,
        llm_timeout: float = 120.0,
        outer_timeout: float | None = None,
    ):
        """Initialize the code generator.

        Args:
            llm_client: LLM client (e.g., OpenAIClient, AnthropicClient).
            max_iterations: Maximum self-debug iterations.
            enable_self_debug: Whether to enable self-debugging on errors.
            enable_rag: Whether to use RAG with example codes.
            objective: Optimization objective ('lexicographic', 'distance', 'warmstart').
            exec_timeout: Maximum seconds to let generated solver code run (default 60s).
            llm_timeout: Maximum seconds to wait for each LLM response (default 120s).
            outer_timeout: Hard ceiling for the entire generate() call. If None, no ceiling is applied.
                This should be set to match the outer executor timeout so the loop exits
                before the executor forcibly kills the thread.
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.enable_self_debug = enable_self_debug
        self.enable_rag = enable_rag
        self.objective = objective
        self.exec_timeout = exec_timeout
        self.llm_timeout = llm_timeout
        self.outer_timeout = outer_timeout
    
    def generate(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None = None,
        optimal: float | None = None,
    ) -> CodeGenResult:
        """Generate and execute VRPTW solver code.
        
        Args:
            instance: The VRPTW instance to solve.
            incumbent: Current incumbent solution (for warm-start mode).
            optimal: Known optimal value (for validation).
        
        Returns:
            CodeGenResult with generated code and execution result.
        """
        start_time = time.perf_counter()

        # Select template based on objective
        template = self._select_template()

        # Build prompt
        prompt = self._build_generation_prompt(instance, incumbent, template)

        # Query LLM with timeout guard
        llm_start = time.perf_counter()
        try:
            response = self._llm_call_with_timeout(prompt)
            llm_latency = (time.perf_counter() - llm_start) * 1000
        except TimeoutError:
            logger.error(
                f"LLM query timed out after {self.llm_timeout:.0f}s "
                f"(elapsed since start: {time.perf_counter() - start_time:.1f}s)"
            )
            return CodeGenResult(
                success=False,
                error_message=f"LLM query timed out after {self.llm_timeout:.0f}s",
                iterations=0,
                llm_latency_ms=(time.perf_counter() - llm_start) * 1000,
                total_runtime_sec=time.perf_counter() - start_time,
            )
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return CodeGenResult(
                success=False,
                error_message=f"LLM query failed: {e}",
                iterations=0,
                llm_latency_ms=0,
                total_runtime_sec=time.perf_counter() - start_time,
            )

        # Parse response
        state = self._parse_llm_response(response)

        # Execute code
        solve_params = instance_to_solve_params(instance)

        # Add warm-start routes if available
        if incumbent and self.objective == "warmstart":
            solve_params["initial_routes"] = incumbent.routes

        # First execution with timeout
        exec_result = self._execute_with_timeout(
            state, solve_params, optimal, self.exec_timeout
        )

        if not exec_result.success:
            logger.warning(
                f"[CodeGen] Execution failed: {exec_result.error_type} - "
                f"{str(exec_result.error_message)[:200]}"
            )

        iterations = 0
        consecutive_exec_timeouts = 0
        MAX_CONSECUTIVE_EXEC_TIMEOUTS = 2  # give up after 2 straight execution timeouts

        # Self-debug loop — track elapsed time and respect remaining budget
        while iterations < self.max_iterations:
            elapsed = time.perf_counter() - start_time

            if exec_result.success:
                break

            # Give up if total time budget is exhausted (per-iteration budget)
            if elapsed >= self.llm_timeout * self.max_iterations:
                logger.warning(
                    f"[CodeGen] Per-iteration time budget exhausted "
                    f"(elapsed={elapsed:.1f}s, budget={self.llm_timeout * self.max_iterations:.1f}s), "
                    f"stopping self-debug at iteration {iterations}"
                )
                break

            # Give up if outer hard timeout is exceeded (this is the real ceiling).
            # Without this check, the loop can run for longer than the outer
            # executor timeout (660s) with no output, appearing to hang.
            if self.outer_timeout is not None and elapsed >= self.outer_timeout:
                logger.warning(
                    f"[CodeGen] Outer hard timeout reached (elapsed={elapsed:.1f}s >= outer_timeout={self.outer_timeout:.1f}s), "
                    f"stopping self-debug at iteration {iterations}"
                )
                break

            if not self.enable_self_debug:
                break

            # Detect consecutive execution timeouts — the LLM keeps generating code
            # that OR-Tools can't solve within exec_timeout. No point wasting more
            # LLM calls on the same problem pattern.
            if exec_result.error_type == "TimeoutError" or getattr(exec_result, "_is_timeout", False):
                consecutive_exec_timeouts += 1
                if consecutive_exec_timeouts >= MAX_CONSECUTIVE_EXEC_TIMEOUTS:
                    logger.warning(
                        f"[CodeGen] {consecutive_exec_timeouts} consecutive execution timeouts, "
                        f"stopping self-debug — LLM code keeps timing out, "
                        f"giving up to save time"
                    )
                    break
            else:
                consecutive_exec_timeouts = 0

            iterations += 1
            logger.info(
                f"[CodeGen] Self-debug iteration {iterations} "
                f"(elapsed={elapsed:.1f}s, remaining budget~{self.llm_timeout * (self.max_iterations - iterations):.1f}s)"
            )

            # Build debug prompt
            debug_prompt = self._build_debug_prompt(
                state, exec_result, instance, incumbent, template
            )

            # Query LLM again with timeout guard
            llm_start = time.perf_counter()
            try:
                response = self._llm_call_with_timeout(debug_prompt)
                llm_latency += (time.perf_counter() - llm_start) * 1000
            except TimeoutError:
                logger.warning(
                    f"LLM debug query timed out at iteration {iterations}, aborting self-debug"
                )
                break
            except Exception as e:
                logger.error(f"LLM debug query failed at iteration {iterations}: {e}")
                break

            # Parse and execute
            state = self._parse_llm_response(response)
            exec_result = self._execute_with_timeout(
                state, solve_params, optimal, self.exec_timeout
            )
        
        total_runtime = time.perf_counter() - start_time
        
        if exec_result.success:
            return CodeGenResult(
                success=True,
                code=state.code,
                imports=state.imports,
                execution_result=exec_result,
                iterations=iterations,
                llm_latency_ms=llm_latency,
                total_runtime_sec=total_runtime,
            )
        else:
            return CodeGenResult(
                success=False,
                code=state.code,
                imports=state.imports,
                error_message=exec_result.error_message,
                iterations=iterations,
                llm_latency_ms=llm_latency,
                total_runtime_sec=total_runtime,
            )
    
    def _select_template(self) -> str:
        """Select appropriate template based on objective."""
        if self.objective == "warmstart":
            return VRPTW_WARMSTART_TEMPLATE
        elif self.objective == "distance":
            return VRPTW_BASE_TEMPLATE
        elif self.objective == "lexicographic":
            return VRPTW_LEXICOGRAPHIC_TEMPLATE
        return VRPTW_BASE_TEMPLATE
    
    def _build_generation_prompt(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None,
        template: str,
    ) -> str:
        """Build prompt for code generation."""
        from src.llm.code_executor import build_execution_context

        # Build instance description
        instance_desc = self._build_instance_description(instance)

        # Build warm-start hint from incumbent solution (e.g. baseline solver output)
        incumbent_hint = ""
        if incumbent and incumbent.routes:
            route_previews = []
            for i, route in enumerate(incumbent.routes[:5]):
                route_previews.append(f"  Route {i+1} ({len(route)} customers): {route[:8]}{'...' if len(route) > 8 else ''}")
            incumbent_hint = f"""
### Known Reference Solution (Warm-Start)
A baseline solver found a feasible solution with these properties:
- Vehicles used: {incumbent.vehicles_used}
- Total distance: {incumbent.total_distance:.1f}
- Sample routes (first 5 vehicles):
{chr(10).join(route_previews)}
{("  ... and " + str(max(0, incumbent.vehicles_used - 5)) + " more routes") if incumbent.vehicles_used > 5 else ""}

IMPORTANT: Use this as a TARGET. Your solution should:
1. Have FEWER or EQUAL vehicles compared to the reference ({incumbent.vehicles_used})
2. Have LESS or EQUAL total distance compared to the reference ({incumbent.total_distance:.1f})
3. Ensure ALL time window and capacity constraints are satisfied
"""

        # Add execution-time guidance for large instances
        time_guidance = ""
        n = instance.size
        if n >= 600:
            time_guidance = f"""
### Execution Time Guidance
This instance has {n} customers. To avoid timeouts:
- Use greedy construction heuristics (savings algorithm, nearest neighbor) to build an initial solution quickly
- Apply local search (2-opt, Or-opt) rather than exhaustive enumeration
- Target producing a feasible solution within the first 60 seconds
- DO NOT use brute-force or exhaustive search strategies
"""

        prompt = (
            "## TASK: Generate VRPTW Solver Code\n"
            "\n"
            "You are an expert in Python programming for vehicle routing problems.\n"
            "Generate syntactically correct Python code to solve the following VRPTW instance.\n"
            "\n"
            "### Problem Description\n"
            f"{instance_desc}\n"
            f"{incumbent_hint}\n"
            f"{time_guidance}"
            "### Requirements\n"
            "1. Read the template below. Understand the parameters in the 'solve' function.\n"
            "2. Follow the format of the template strictly to complete the code.\n"
            "3. Ensure all parameters in the template are used.\n"
            "4. Return the solution by the 'solve' function with keys: 'routes', 'total_distance', 'vehicles_used'.\n"
            "5. Ensure any code you provide can be executed with all required imports.\n"
            "6. The objective is LEXICOGRAPHIC: minimize vehicles first, then minimize distance.\n"
            "\n"
            "### Template\n"
            "```\n"
            f"{template}\n"
            "```\n"
            "\n"
            "### Anti-Patterns: NEVER do these\n"
            "- Do NOT use hardcoded numbers from other instances (e.g., '74 vehicles', '30254 distance').\n"
            "  Your code must work for ANY instance based on its actual parameters.\n"
            "- Do NOT return early with empty routes based on magic thresholds like 'if num_vehicle > X: return {}'.\n"
            "- Do NOT add service time to the FROM node in time callbacks. The correct pattern is:\n"
            "    service_times[to_node]  NOT  service_times[from_node]\n"
            "- Do NOT use exhaustive search, brute force, or enumeration strategies for large instances.\n"
            "- Ensure all YOUR_CODE_HERE placeholders in the template are filled in completely.\n"
            "\n"
            "### Output Format\n"
            "Structure your answer with:\n"
            "1. Brief description of the solution approach\n"
            "2. Required imports (starting with 'from ortools...' or 'import ortools')\n"
            "3. The complete Python code block\n"
            "\n"
            "IMPORTANT: Output ONLY the imports and code. No markdown fences, no additional explanation.\n"
        )
        return prompt
    
    def _build_instance_description(self, instance: VRPTWInstance) -> str:
        """Build detailed instance description for the prompt."""
        n = instance.size
        
        # Compute distance matrix preview (first 5x5)
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
        
        # Add some customer coordinates
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
            "- **Service times**:",
        ])
        
        for cid in instance.customer_ids[:5]:
            st = instance.service_time.get(cid, 0)
            lines.append(f"  Customer {cid}: {st}")
        
        lines.extend([
            "",
            f"- **Objective**: {instance.objective}",
        ])
        
        return "\n".join(lines)
    
    def _build_debug_prompt(
        self,
        state: GenerationState,
        exec_result: "ExecutionResult",
        instance: VRPTWInstance,
        incumbent: RouteSolution | None,
        template: str,
    ) -> str:
        """Build prompt for self-debugging."""
        from src.llm.code_executor import extract_error_for_llm

        error_context = extract_error_for_llm(exec_result)
        instance_desc = self._build_instance_description(instance)

        broken_code = state.imports + "\n" + state.code if state.imports else state.code

        # Build incumbent hint for the debug prompt
        incumbent_hint = ""
        if incumbent and incumbent.routes:
            incumbent_hint = f"""
### Known Reference Solution (do NOT exceed these targets)
- Vehicles: {incumbent.vehicles_used}  (your solution should use <= {incumbent.vehicles_used})
- Distance: {incumbent.total_distance:.1f}  (your solution should have <= {incumbent.total_distance:.1f})
"""

        # Build execution result context
        result_context = ""
        if exec_result.vehicles_used is not None:
            result_context += f"- Your solution uses: {exec_result.vehicles_used} vehicles, {exec_result.total_distance:.1f} distance\n"
        if exec_result.runtime_sec > 0:
            result_context += f"- Execution time: {exec_result.runtime_sec:.1f}s\n"

        # Determine error category and specific guidance
        error_type = exec_result.error_type or ""
        specific_guidance = ""
        if "TimeoutError" in error_type or "timeout" in error_type.lower():
            specific_guidance = """
### Timeout Guidance
The solver timed out. This means the search strategy is too slow for an 800-customer instance.
Fix strategy:
- Replace the current approach with a greedy construction heuristic (e.g., savings algorithm, nearest-neighbor insertion)
- Add local search (2-opt, Or-opt) AFTER the initial solution is built
- Set a short time limit for the initial solution construction phase
- Do NOT run exhaustive search or enumeration
"""
        elif "ValidationError" in error_type or "Syntax" in error_type:
            specific_guidance = """
### Syntax/Validation Fix
The generated code has syntax errors or doesn't follow the template format.
Fix: Carefully check indentation, matching brackets, and ensure all YOUR_CODE_HERE sections are replaced.
"""
        elif "RuntimeError" in error_type or "Exception" in error_type:
            specific_guidance = """
### Runtime Error Fix
The code crashes during execution.
Fix: Check array indices, make sure all required OR-Tools objects are initialized, verify callback functions return correct types.
"""

        prompt = (
            "## TASK: Debug VRPTW Solver Code\n"
            "\n"
            "The following code has a problem:\n"
            "\n"
            "### Current Solution (if available)\n"
            f"{result_context.strip()}\n"
            f"{incumbent_hint}"
            "### Broken Code\n"
            "```python\n"
            f"{broken_code}\n"
            "```\n"
            "\n"
            "### Error Information\n"
            "```\n"
            f"{error_context}\n"
            "```\n"
            f"{specific_guidance}"
            "### Problem Description (for reference)\n"
            f"{instance_desc}\n"
            "\n"
            "### Instructions\n"
            "1. First analyze the error: reason about what caused it.\n"
            "2. Then fix the code and return the complete corrected function.\n"
            "3. Ensure the fixed code follows the template structure.\n"
            "4. Do not change the function signature or parameter names.\n"
            "5. Keep the same overall algorithm strategy — only fix the specific bug or timeout issue.\n"
            "6. For timeout errors: switch to a faster heuristic. For syntax errors: fix the code. For runtime errors: fix the crash.\n"
            "\n"
            "### Critical Anti-Patterns to Avoid\n"
            "- NEVER use hardcoded numbers from other instances (e.g., '74 vehicles', '30254 distance').\n"
            "- NEVER return early with empty routes based on magic thresholds like 'if num_vehicle > X: return {}'.\n"
            "- NEVER add service time to the FROM node — always use service_times[to_node], not service_times[from_node].\n"
            "\n"
            "### Template (for reference)\n"
            "```\n"
            f"{template}\n"
            "```\n"
            "\n"
            "### Output Format\n"
            "1. Brief explanation of the error and fix strategy\n"
            "2. Required imports\n"
            "3. Complete corrected Python code\n"
            "\n"
            "IMPORTANT: Output ONLY the imports and code. No markdown fences.\n"
        )
        return prompt
    
    def _parse_llm_response(self, response: str | dict) -> GenerationState:
        """Parse LLM response into code and imports."""
        code = ""
        imports = ""
        
        if isinstance(response, dict):
            # Structured response from some LLM clients
            code = response.get("code", response.get("content", ""))
            imports = response.get("imports", "")
        else:
            # Raw text response - need to parse
            text = response
            lines = text.split("\n")
            
            # Find imports section
            import_lines = []
            code_lines = []
            in_imports = False
            in_code = False
            
            for line in lines:
                stripped = line.strip().lower()
                
                # Detect imports section
                if "from ortools" in stripped or "import ortools" in stripped:
                    in_imports = True
                    in_code = False
                
                # Detect code section (after imports or after description)
                if in_imports and ("def " in line or "```" in stripped):
                    in_imports = False
                    in_code = True
                
                if stripped.startswith("```"):
                    if in_imports:
                        in_imports = False
                    in_code = not in_code
                    continue
                
                if in_imports:
                    import_lines.append(line)
                elif in_code or (not import_lines and not in_imports and line.strip()):
                    if "def " in line:
                        in_code = True
                    if in_code:
                        code_lines.append(line)
            
            imports = "\n".join(import_lines).strip()
            code = "\n".join(code_lines).strip()
        
        # Clean up markdown fences if present
        code = code.replace("```python", "").replace("```", "").strip()
        imports = imports.replace("```python", "").replace("```", "").strip()
        
        return GenerationState(code=code, imports=imports)
    
    def _execute_with_validation(
        self,
        state: GenerationState,
        solve_params: dict[str, Any],
        optimal: float | None,
    ) -> "ExecutionResult":
        """Execute code and validate result."""
        from src.llm.code_executor import execute_solve_code, validate_code, ExecutionResult
        
        # Validate syntax first (check full code including imports)
        full_code = state.imports + "\n" + state.code if state.imports else state.code
        validation = validate_code(full_code)
        if not validation.valid:
            # Prepend standard ortools imports if missing and re-validate
            if validation.import_error and "ortools" in validation.import_error:
                standard_imports = (
                    "from ortools.constraint_solver import pywrapcp, routing_enums_pb2\n"
                )
                full_code = standard_imports + state.code
                state.imports = standard_imports
                validation = validate_code(full_code)
            if not validation.valid:
                return ExecutionResult(
                    success=False,
                    error_message=validation.syntax_error or validation.import_error,
                    error_type="ValidationError",
                )
        
        # Execute
        full_code = state.imports + "\n" + state.code if state.imports else state.code
        exec_result = execute_solve_code(full_code, solve_params)
        
        # Check accuracy if optimal known
        if exec_result.success and optimal is not None:
            # Add accuracy check message if result is far from optimal
            dist = exec_result.total_distance or 0
            if optimal > 0 and abs(dist - optimal) / optimal > 0.1:
                state.error = f"The objective {dist:.2f} is far from the optimum {optimal:.2f}"
        
        return exec_result

    def _llm_call_with_timeout(self, prompt: str) -> str:
        """Call LLM with a hard timeout via ThreadPoolExecutor.

        Some API clients only bound the socket wait, while server-side queueing
        is not counted. We add an outer
        ThreadPoolExecutor layer so the entire call (queue + generate) is bounded.
        """
        client = self.llm_client

        def _call() -> str:
            return client.generate_code(prompt)

        # Wrap in a thread so we can cancel on overall timeout
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            try:
                return future.result(timeout=self.llm_timeout)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"LLM call exceeded llm_timeout={self.llm_timeout:.0f}s "
                    f"(server queueing is included in this limit)"
                )

    def _execute_with_timeout(
        self,
        state: GenerationState,
        solve_params: dict[str, Any],
        optimal: float | None,
        timeout: float,
    ) -> "ExecutionResult":
        """Execute code with a hard timeout.

        Calls through to execute_solve_code which now has built-in
        ThreadPoolExecutor-based timeout protection.
        """
        from src.llm.code_executor import execute_solve_code
        return execute_solve_code(
            state.imports + "\n" + state.code if state.imports else state.code,
            solve_params,
            timeout=timeout,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Code Generator - Full DRoC Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

from src.llm.problem_decomposer import VRPTWDecomposer, decompose_vrptw
from src.llm.constraint_retriever import ConstraintRetriever, retrieve_constraint_codes
from src.llm.code_summarizer import CodeSummarizer, filter_relevant_codes
from src.llm.augmented_generator import (
    AugmentedCodeGenerator,
    RefinementCodeGenerator,
    generate_with_context,
    refine_with_context,
)
from src.llm.self_debugger import SelfDebugger, DebugLoop
from src.llm.vrptw_templates import (
    VRPTW_CONSTRAINT_SNIPPETS,
    get_constraint_snippet,
    build_constraint_context_from_snippets,
    get_template_for_objective,
)


class EnhancedCodeGenerator(CodeGenerator):
    """Enhanced code generator with full DRoC pipeline.
    
    This generator integrates all DRoC components:
    - Problem decomposition
    - RAG retrieval
    - Augmented code generation
    - Self-debugging
    
    Inherits from CodeGenerator and extends it with RAG capabilities.
    """
    
    def __init__(
        self,
        llm_client: Any,
        max_iterations: int = 4,
        enable_self_debug: bool = True,
        enable_rag: bool = True,
        objective: str = "lexicographic",
        solver: str = "OR-tools",
        rag_top_k: int = 3,
        exec_timeout: float = 60.0,
        llm_timeout: float = 120.0,
    ):
        """Initialize the enhanced code generator.

        Args:
            llm_client: LLM client.
            max_iterations: Maximum self-debug iterations.
            enable_self_debug: Whether to enable self-debugging.
            enable_rag: Whether to use RAG.
            objective: Optimization objective.
            solver: Solver backend.
            rag_top_k: Number of RAG results per keyword.
            exec_timeout: Maximum seconds for solver code execution.
            llm_timeout: Maximum seconds for each LLM call.
        """
        super().__init__(
            llm_client=llm_client,
            max_iterations=max_iterations,
            enable_self_debug=enable_self_debug,
            enable_rag=enable_rag,
            objective=objective,
            exec_timeout=exec_timeout,
            llm_timeout=llm_timeout,
        )
        self.solver = solver
        self.rag_top_k = rag_top_k
        
        # Initialize DRoC components
        self._init_droc_components()
    
    def _init_droc_components(self) -> None:
        """Initialize DRoC-specific components."""
        # Problem decomposer
        self.decomposer = VRPTWDecomposer()
        
        # RAG retriever
        self.retriever = ConstraintRetriever(
            solver=self.solver,
            top_k=self.rag_top_k,
        )
        
        # Code summarizer
        self.summarizer = CodeSummarizer(
            solver=self.solver,
            llm_client=self.llm_client,
        )
        
        # Augmented generator
        self.aug_generator = AugmentedCodeGenerator(
            llm_client=self.llm_client,
            solver=self.solver,
        )
        
        # Refinement generator
        self.refiner = RefinementCodeGenerator(
            llm_client=self.llm_client,
            solver=self.solver,
        )
        
        # Self-debugger
        self.debugger = SelfDebugger(
            llm_client=self.llm_client,
            solver=self.solver,
            max_attempts=self.max_iterations,
        )
        
        # Debug loop
        self.debug_loop = DebugLoop(
            debugger=self.debugger,
            max_iterations=self.max_iterations,
        )

    # ── Helper methods for timeout-aware LLM calls ──────────────────────────────

    def _build_augmented_prompt(
        self,
        instance: VRPTWInstance,
        constraint_context: str,
        template: str,
        incumbent: RouteSolution | None = None,
    ) -> str:
        """Build the augmented generation prompt (used when falling back from old API)."""
        from src.llm.code_executor import build_execution_context
        instance_desc = self._build_instance_description(instance)

        incumbent_hint = ""
        if incumbent and incumbent.routes:
            route_previews = []
            for i, route in enumerate(incumbent.routes[:5]):
                route_previews.append(f"  Route {i+1} ({len(route)} customers): {route[:8]}{'...' if len(route) > 8 else ''}")
            incumbent_hint = f"""
### Known Reference Solution (Warm-Start)
A baseline solver found a feasible solution with these properties:
- Vehicles used: {incumbent.vehicles_used}
- Total distance: {incumbent.total_distance:.1f}
- Sample routes (first 5 vehicles):
{chr(10).join(route_previews)}

IMPORTANT: Your solution should have <= {incumbent.vehicles_used} vehicles and <= {incumbent.total_distance:.1f} distance.
"""

        time_guidance = ""
        n = instance.size
        if n >= 600:
            time_guidance = f"""
### Execution Time Guidance
This instance has {n} customers. To avoid timeouts:
- Use greedy construction heuristics (savings algorithm, nearest neighbor) to build an initial solution quickly
- Apply local search (2-opt, Or-opt) rather than exhaustive enumeration
- Target producing a feasible solution within the first 60 seconds
- DO NOT use brute-force or exhaustive search strategies
"""

        return (
            "## TASK: Generate VRPTW Solver Code\n"
            "\n"
            "You are an expert in Python programming for vehicle routing problems.\n"
            "Generate syntactically correct Python code to solve the following VRPTW instance.\n"
            "\n"
            "### Problem Description\n"
            f"{instance_desc}\n"
            f"{incumbent_hint}"
            f"{time_guidance}"
            "### RAG Constraint Context\n"
            f"{constraint_context}\n"
            "\n"
            "### Requirements\n"
            "1. Follow the template strictly to complete the code.\n"
            "2. Return the solution via the 'solve' function with keys: 'routes', 'total_distance', 'vehicles_used'.\n"
            "3. Ensure all required imports are included.\n"
            "4. The objective is LEXICOGRAPHIC: minimize vehicles first, then minimize distance.\n"
            "\n"
            "### Anti-Patterns: NEVER do these\n"
            "- Do NOT use hardcoded numbers from other instances (e.g., '74 vehicles', '30254 distance').\n"
            "  Your code must work for ANY instance based on its actual parameters.\n"
            "- Do NOT return early with empty routes based on magic thresholds like 'if num_vehicle > X: return {}'.\n"
            "- Do NOT add service time to the FROM node in time callbacks. Always use service_times[to_node].\n"
            "- Do NOT use exhaustive search, brute force, or enumeration strategies for large instances.\n"
            "- Ensure all YOUR_CODE_HERE placeholders in the template are filled in completely.\n"
            "\n"
            "### Template\n"
            "```\n"
            f"{template}\n"
            "```\n"
            "\n"
            "IMPORTANT: Output ONLY the imports and code. No markdown fences.\n"
        )

    def _build_refinement_prompt(
        self,
        instance: VRPTWInstance,
        broken_code: str,
        exec_result: "ExecutionResult",
        constraint_context: str,
        incumbent: RouteSolution | None = None,
    ) -> str:
        """Build the RAG-enhanced refinement prompt."""
        from src.llm.code_executor import extract_error_for_llm
        instance_desc = self._build_instance_description(instance)
        error_context = extract_error_for_llm(exec_result)

        incumbent_hint = ""
        if incumbent and incumbent.routes:
            incumbent_hint = f"""
### Known Reference Solution (do NOT exceed these targets)
- Vehicles: {incumbent.vehicles_used}  (your solution should use <= {incumbent.vehicles_used})
- Distance: {incumbent.total_distance:.1f}  (your solution should have <= {incumbent.total_distance:.1f})
"""

        result_context = ""
        if exec_result.vehicles_used is not None:
            result_context += f"- Your solution uses: {exec_result.vehicles_used} vehicles, {exec_result.total_distance:.1f} distance\n"
        if exec_result.runtime_sec > 0:
            result_context += f"- Execution time: {exec_result.runtime_sec:.1f}s\n"

        error_type = exec_result.error_type or ""
        specific_guidance = ""
        if "TimeoutError" in error_type or "timeout" in error_type.lower():
            specific_guidance = """
### Timeout Guidance
The solver timed out. Use a greedy construction heuristic (savings, nearest-neighbor) + local search instead.
"""
        elif "ValidationError" in error_type or "Syntax" in error_type:
            specific_guidance = """
### Syntax/Validation Fix
Check indentation, matching brackets, and ensure all YOUR_CODE_HERE sections are replaced.
"""
        elif "RuntimeError" in error_type or "Exception" in error_type:
            specific_guidance = """
### Runtime Error Fix
Check array indices, OR-Tools initialization, and callback return types.
"""

        return (
            "## TASK: Debug VRPTW Solver Code\n"
            "\n"
            "The following code has a problem:\n"
            "\n"
            "### Current Solution (if available)\n"
            f"{result_context.strip()}\n"
            f"{incumbent_hint}"
            "### Broken Code\n"
            "```python\n"
            f"{broken_code}\n"
            "```\n"
            "\n"
            "### Error Information\n"
            "```\n"
            f"{error_context}\n"
            "```\n"
            + specific_guidance +
            "### RAG Constraint Context (for reference)\n"
            f"{constraint_context}\n"
            "\n"
            "### Problem Description (for reference)\n"
            f"{instance_desc}\n"
            "\n"
            "### Instructions\n"
            "1. Analyze the error and fix the code.\n"
            "2. Return the complete corrected code.\n"
            "3. Follow the template structure.\n"
            "4. Keep the same algorithm strategy — only fix the specific bug or timeout issue.\n"
            "\n"
            "### Critical Anti-Patterns to Avoid\n"
            "- NEVER use hardcoded numbers from other instances (e.g., '74 vehicles', '30254 distance').\n"
            "- NEVER return early with empty routes based on magic thresholds like 'if num_vehicle > X: return {}'.\n"
            "- NEVER add service time to the FROM node — always use service_times[to_node], not service_times[from_node].\n"
            "\n"
            "IMPORTANT: Output ONLY the imports and code. No markdown fences.\n"
        )

    def generate(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None = None,
        optimal: float | None = None,
    ) -> CodeGenResult:
        """Generate code with full DRoC pipeline.
        
        This method overrides the parent to use the enhanced DRoC pipeline.
        """
        if not self.enable_rag:
            # Fall back to simple generation
            return super().generate(instance, incumbent, optimal)
        
        start_time = time.perf_counter()

        # Step 1: Problem decomposition
        decomposed = self.decomposer.decompose(instance)
        keywords = decomposed.keywords

        # Step 2: RAG retrieval
        retrieved = self.retriever.retrieve(keywords)
        constraint_context = filter_relevant_codes(
            retrieved,
            solver=self.solver,
            llm_client=self.llm_client,
        )

        # Step 3: Augmented code generation with LLM timeout
        template = get_template_for_objective(self.objective)
        llm_start = time.perf_counter()

        try:
            generated = self._llm_call_with_timeout(
                self._build_augmented_prompt(
                    instance, constraint_context, template, incumbent
                )
            )
        except TimeoutError:
            logger.error(
                f"[EnhancedCodeGen] LLM generation timed out "
                f"({self.llm_timeout:.0f}s), falling back to simple generation"
            )
            return super().generate(instance, incumbent, optimal)
        except Exception as e:
            logger.error(f"RAG generation failed: {e}, falling back to simple generation")
            return super().generate(instance, incumbent, optimal)

        llm_latency = (time.perf_counter() - llm_start) * 1000

        parsed = self._parse_llm_response(generated)
        state = GenerationState(
            code=parsed.code or generated,
            imports=parsed.imports,
            iterations=0,
        )

        # Get solve params
        solve_params = instance_to_solve_params(instance)
        if incumbent and self.objective == "warmstart":
            solve_params["initial_routes"] = incumbent.routes

        # Execute with timeout
        exec_result = self._execute_with_timeout(
            state, solve_params, optimal, self.exec_timeout
        )

        if not exec_result.success:
            logger.warning(
                f"[EnhancedCodeGen] Execution failed: {exec_result.error_type} - "
                f"{str(exec_result.error_message)[:200]}"
            )

        iterations = 0

        # Self-debug loop with RAG context — respect elapsed time budget
        while iterations < self.max_iterations:
            elapsed = time.perf_counter() - start_time
            if exec_result.success:
                break
            if elapsed >= self.llm_timeout * self.max_iterations:
                logger.warning(
                    f"[EnhancedCodeGen] Time budget exhausted "
                    f"(elapsed={elapsed:.1f}s), stopping self-debug"
                )
                break
            if not self.enable_self_debug:
                break

            iterations += 1
            logger.info(
                f"[EnhancedCodeGen] Self-debug iteration {iterations} "
                f"(elapsed={elapsed:.1f}s)"
            )

            # Use RAG-enhanced refinement with timeout
            try:
                refined_prompt = self._build_refinement_prompt(
                    instance, state.code, exec_result, constraint_context, incumbent
                )
                refined_response = self._llm_call_with_timeout(refined_prompt)
            except TimeoutError:
                logger.warning(
                    f"[EnhancedCodeGen] LLM refinement timed out at iteration {iterations}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"[EnhancedCodeGen] RAG refinement failed at iteration {iterations}: {e}"
                )
                break

            refined_parsed = self._parse_llm_response(refined_response)
            state = GenerationState(
                code=refined_parsed.code,
                imports=refined_parsed.imports,
                iterations=iterations,
            )
            exec_result = self._execute_with_timeout(
                state, solve_params, optimal, self.exec_timeout
            )
        
        total_runtime = time.perf_counter() - start_time
        
        if exec_result.success:
            return CodeGenResult(
                success=True,
                code=state.code,
                imports=state.imports,
                execution_result=exec_result,
                iterations=iterations,
                llm_latency_ms=llm_latency,
                total_runtime_sec=total_runtime,
            )
        else:
            return CodeGenResult(
                success=False,
                code=state.code,
                imports=state.imports,
                error_message=exec_result.error_message,
                iterations=iterations,
                llm_latency_ms=llm_latency,
                total_runtime_sec=total_runtime,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Simple LLM client wrapper that extends existing clients
# ─────────────────────────────────────────────────────────────────────────────

def extend_llm_client_with_code_gen(client: Any) -> None:
    """Extend an existing LLM client with code generation capability.
    
    Adds a 'generate_code' method to the client if it doesn't exist.
    """
    if hasattr(client, "generate_code"):
        return
    
    def generate_code(prompt: str) -> str:
        """Generate code from prompt using the LLM."""
        # Try structured output first
        if hasattr(client, "with_structured_output"):
            try:
                response = client.invoke(prompt)
                if hasattr(response, "content"):
                    return response.content
                return str(response)
            except Exception:
                pass
        
        # Fall back to regular query
        if hasattr(client, "query"):
            response = client.query(prompt)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        
        raise NotImplementedError("LLM client doesn't support code generation")
    
    client.generate_code = generate_code
