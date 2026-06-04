"""Enhanced DRoC Pipeline - Complete RAG-based VRPTW solver.

This module implements the complete DRoC pipeline integrating:
1. Problem decomposition
2. Constraint retrieval (RAG)
3. Code summarization and filtering
4. Augmented code generation
5. Self-debugging loop

The pipeline provides an end-to-end solution for VRPTW instances
using LLM-guided code generation with retrieval augmentation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.domain.schema import VRPTWInstance, RouteSolution, ExperimentResult
from src.domain.metrics import lexicographic_compare
from src.solvers.feasibility_checker import check_feasibility
from src.llm.code_executor import execute_solve_code, ExecutionResult, validate_code
from src.llm.vrptw_templates import instance_to_solve_params, VRPTW_LEXICOGRAPHIC_TEMPLATE

from src.llm.problem_decomposer import (
    VRPTWDecomposer,
    DecomposedProblem,
    decompose_vrptw,
)
from src.llm.constraint_retriever import (
    ConstraintRetriever,
    retrieve_constraint_codes,
    build_constraint_context,
)
from src.llm.code_summarizer import (
    CodeSummarizer,
    ConstraintContextBuilder,
    filter_relevant_codes,
)
from src.llm.augmented_generator import (
    AugmentedCodeGenerator,
    RefinementCodeGenerator,
    GeneratedCode,
    generate_with_context,
    refine_with_context,
)
from src.llm.self_debugger import (
    SelfDebugger,
    DebugLoop,
    DebugResult,
    self_debug_with_context,
)

if TYPE_CHECKING:
    from src.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """State during pipeline execution."""
    
    instance: VRPTWInstance
    decomposed: DecomposedProblem | None = None
    constraint_context: dict[str, Any] | None = None
    generated_code: GeneratedCode | None = None
    execution_result: ExecutionResult | None = None
    error: str = "no"
    messages: list[str] = field(default_factory=list)
    iterations: int = 0
    total_runtime_sec: float = 0.0
    
    @property
    def has_solution(self) -> bool:
        return (
            self.execution_result is not None 
            and self.execution_result.success
        )


@dataclass
class PipelineConfig:
    """Configuration for the enhanced DRoC pipeline."""
    
    solver: str = "OR-tools"
    max_iterations: int = 4
    enable_rag: bool = True
    enable_self_debug: bool = True
    enable_constraint_decomposition: bool = True
    rag_top_k: int = 3
    time_limit: float = 60.0
    seed: int = 42
    output_dir: str | Path | None = None
    optimal: float | None = None


class EnhancedDrocPipeline:
    """Enhanced DRoC pipeline with full RAG support.
    
    This pipeline integrates all DRoC components:
    - Problem decomposition
    - RAG retrieval
    - Code generation
    - Self-debugging
    
    Example:
        config = PipelineConfig(solver="OR-tools", max_iterations=4)
        pipeline = EnhancedDrocPipeline(llm_client=client, config=config)
        
        result = pipeline.run(instance)
        print(f"Solution: {result.routes}")
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: PipelineConfig | None = None,
    ):
        """Initialize the pipeline.
        
        Args:
            llm_client: LLM client for code generation.
            config: Pipeline configuration.
        """
        self.llm_client = llm_client
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Decomposer
        self.decomposer = VRPTWDecomposer()
        
        # Retriever
        self.retriever = ConstraintRetriever(
            solver=self.config.solver,
            top_k=self.config.rag_top_k,
        )
        
        # Summarizer
        self.summarizer = CodeSummarizer(
            solver=self.config.solver,
            llm_client=self.llm_client,
        )
        
        # Context builder
        self.context_builder = ConstraintContextBuilder(self.summarizer)
        
        # Generators
        self.generator = AugmentedCodeGenerator(
            llm_client=self.llm_client,
            solver=self.config.solver,
        )
        
        self.refiner = RefinementCodeGenerator(
            llm_client=self.llm_client,
            solver=self.config.solver,
        )
        
        # Debugger
        self.debugger = SelfDebugger(
            llm_client=self.llm_client,
            solver=self.config.solver,
            max_attempts=self.config.max_iterations,
        )
        
        # Debug loop
        self.debug_loop = DebugLoop(
            debugger=self.debugger,
            max_iterations=self.config.max_iterations,
        )
    
    def run(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None = None,
    ) -> tuple[RouteSolution, PipelineState, ExperimentResult]:
        """Run the complete DRoC pipeline.
        
        Args:
            instance: The VRPTW instance to solve.
            incumbent: Optional incumbent solution for warm-start.
        
        Returns:
            Tuple of (solution, state, experiment_result).
        """
        logger.info(f"[EnhancedDRoC] Starting on {instance.name}")
        start_time = time.perf_counter()
        
        # Initialize state
        state = PipelineState(instance=instance)
        
        # Phase 1: Problem decomposition
        if self.config.enable_constraint_decomposition:
            state.decomposed = self._phase1_decompose(instance)
            logger.info(f"Decomposed keywords: {state.decomposed.keywords}")
        
        # Phase 2: RAG retrieval
        if self.config.enable_rag:
            state.constraint_context = self._phase2_retrieve(state.decomposed)
            logger.info(f"Retrieved context for {len(state.constraint_context)} constraints")
        
        # Phase 3: Code generation
        state.generated_code = self._phase3_generate(state)
        
        if not state.generated_code or not state.generated_code.is_valid():
            logger.error("Code generation failed")
            return self._return_failure(state, start_time)
        
        # Phase 4: Execution and self-debug loop
        state = self._phase4_execute_and_debug(state)
        
        # Phase 5: Build solution
        runtime = time.perf_counter() - start_time
        state.total_runtime_sec = runtime
        
        if state.has_solution:
            solution = self._build_solution(state.execution_result, runtime)
            validation = check_feasibility(instance, solution)
            
            result = self._build_experiment_result(
                instance, solution, validation, runtime, state
            )
            
            # Save outputs
            if self.config.output_dir:
                self._save_outputs(instance, state, solution, result)
            
            logger.info(
                f"[EnhancedDRoC] Success: vehicles={solution.vehicles_used}, "
                f"distance={solution.total_distance:.2f}"
            )
            
            return solution, state, result
        else:
            return self._return_failure(state, start_time)
    
    def _phase1_decompose(self, instance: VRPTWInstance) -> DecomposedProblem:
        """Phase 1: Decompose problem into constraints."""
        logger.debug("[Phase 1] Decomposing problem")
        return self.decomposer.decompose(instance)
    
    def _phase2_retrieve(
        self,
        decomposed: DecomposedProblem | None,
    ) -> dict[str, Any]:
        """Phase 2: Retrieve constraint codes via RAG."""
        logger.debug("[Phase 2] Retrieving constraint codes")
        
        if decomposed is None:
            # Default keywords
            keywords = ["Capacitated", "Time Windows"]
        else:
            keywords = decomposed.keywords
        
        # Retrieve codes for each keyword
        retrieved = self.retriever.retrieve(keywords)
        
        # Filter to relevant codes
        filtered = filter_relevant_codes(
            retrieved,
            solver=self.config.solver,
            llm_client=self.llm_client,
        )
        
        return filtered
    
    def _phase3_generate(self, state: PipelineState) -> GeneratedCode | None:
        """Phase 3: Generate code with RAG context."""
        logger.debug("[Phase 3] Generating code")
        
        keywords = state.decomposed.keywords if state.decomposed else None
        
        try:
            code = generate_with_context(
                instance=state.instance,
                constraint_context=state.constraint_context,
                llm_client=self.llm_client,
                solver=self.config.solver,
            )
            return code
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            state.messages.append(str(e))
            return None
    
    def _phase4_execute_and_debug(self, state: PipelineState) -> PipelineState:
        """Phase 4: Execute code and run self-debug loop if needed."""
        logger.debug("[Phase 4] Executing and debugging")
        
        # Get solve params
        solve_params = instance_to_solve_params(state.instance)
        
        # Full code with imports
        code = state.generated_code
        full_code = code.imports + "\n" + code.code if code else ""
        
        # Validate syntax first
        validation = validate_code(code.code if code else "")
        if not validation.valid:
            state.error = f"Syntax error: {validation.syntax_error}"
            state.messages.append(state.error)
            logger.warning(f"Code validation failed: {state.error}")
            # Continue to self-debug to fix syntax errors
        
        # Execute
        exec_result = execute_solve_code(full_code, solve_params)
        state.execution_result = exec_result
        
        if not exec_result.success:
            state.error = exec_result.error_message or "Execution failed"
            state.messages.append(state.error)
            logger.warning(f"Execution failed: {state.error}")
            
            # Run self-debug loop if enabled
            if self.config.enable_self_debug and code:
                state = self._run_self_debug(state)
        else:
            logger.info("Code executed successfully")
        
        return state
    
    def _run_self_debug(self, state: PipelineState) -> PipelineState:
        """Run the self-debug loop."""
        logger.info(f"[Self-Debug] Starting debug loop (max {self.config.max_iterations})")
        
        code = state.generated_code
        if not code:
            return state
        
        # Define execution function for validation
        def execute_and_validate(code_str: str) -> ExecutionResult:
            solve_params = instance_to_solve_params(state.instance)
            return execute_solve_code(code_str, solve_params)
        
        # Run debug loop
        fixed_code, fixed_imports, success = self.debug_loop.run(
            initial_code=code.code,
            initial_imports=code.imports,
            error_message=state.error,
            instance=state.instance,
            constraint_context=state.constraint_context,
            execute_fn=execute_and_validate,
        )
        
        if success:
            # Update state with fixed code
            state.generated_code = GeneratedCode(
                code=fixed_code,
                imports=fixed_imports,
                description="Fixed via self-debug",
            )
            
            # Re-execute to confirm
            full_code = fixed_imports + "\n" + fixed_code
            exec_result = execute_solve_code(full_code, instance_to_solve_params(state.instance))
            state.execution_result = exec_result
            
            if exec_result.success:
                state.error = "no"
                logger.info("[Self-Debug] Debug succeeded")
            else:
                state.error = exec_result.error_message or "Still failed"
                logger.warning(f"[Self-Debug] Still failed: {state.error}")
        else:
            logger.warning("[Self-Debug] Debug loop failed to fix code")
        
        return state
    
    def _build_solution(
        self,
        exec_result: ExecutionResult,
        runtime: float,
    ) -> RouteSolution:
        """Build RouteSolution from execution result."""
        return RouteSolution(
            routes=exec_result.routes or [],
            vehicles_used=exec_result.vehicles_used or len(exec_result.routes or []),
            total_distance=exec_result.total_distance or 0.0,
            total_duration=0.0,
            feasible=True,
            runtime_sec=runtime,
        )
    
    def _build_experiment_result(
        self,
        instance: VRPTWInstance,
        solution: RouteSolution,
        validation: Any,
        runtime: float,
        state: PipelineState,
    ) -> ExperimentResult:
        """Build ExperimentResult."""
        # Calculate gap if optimal known
        gap_to_optimal = None
        if self.config.optimal and self.config.optimal > 0:
            gap_to_optimal = abs(solution.total_distance - self.config.optimal) / self.config.optimal
        
        return ExperimentResult(
            experiment_id=f"{instance.name}_enhanced_droc_{int(time.time())}",
            instance_name=instance.name,
            solver="enhanced_droc",
            vehicles_used=solution.vehicles_used,
            total_distance=solution.total_distance,
            total_duration=solution.total_duration,
            feasible=validation.feasible if validation else True,
            late_violations=getattr(validation, "late_violations", 0) if validation else 0,
            capacity_violations=getattr(validation, "capacity_violations", 0) if validation else 0,
            runtime_sec=runtime,
            gap_to_bks=gap_to_optimal,
            metadata={
                "keywords": state.decomposed.keywords if state.decomposed else [],
                "iterations": state.iterations,
                "code_iterations": state.generated_code.iteration if state.generated_code else 0,
                "error": state.error,
            },
        )
    
    def _return_failure(
        self,
        state: PipelineState,
        start_time: float,
    ) -> tuple[RouteSolution, PipelineState, ExperimentResult]:
        """Return failure result."""
        runtime = time.perf_counter() - start_time
        state.total_runtime_sec = runtime
        
        solution = RouteSolution(
            routes=[],
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            runtime_sec=runtime,
        )
        
        result = ExperimentResult(
            experiment_id=f"{state.instance.name}_enhanced_droc_fail_{int(time.time())}",
            instance_name=state.instance.name,
            solver="enhanced_droc",
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            late_violations=0,
            capacity_violations=0,
            runtime_sec=runtime,
            metadata={
                "error": state.error,
                "iterations": state.iterations,
            },
        )
        
        return solution, state, result
    
    def _save_outputs(
        self,
        instance: VRPTWInstance,
        state: PipelineState,
        solution: RouteSolution,
        result: ExperimentResult,
    ) -> None:
        """Save pipeline outputs to disk."""
        import json
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exp_dir = output_dir / result.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save instance
        with open(exp_dir / "instance.json", "w", encoding="utf-8") as f:
            json.dump(instance.model_dump(), f, indent=2)
        
        # Save generated code
        if state.generated_code:
            code = state.generated_code
            with open(exp_dir / "generated_code.py", "w", encoding="utf-8") as f:
                if code.imports:
                    f.write(code.imports + "\n\n")
                f.write(code.code)
        
        # Save solution
        with open(exp_dir / "solution.json", "w", encoding="utf-8") as f:
            json.dump(solution.model_dump(), f, indent=2)
        
        # Save result
        with open(exp_dir / "experiment_result.json", "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        
        # Save decomposed problem
        if state.decomposed:
            with open(exp_dir / "decomposed.json", "w", encoding="utf-8") as f:
                json.dump({
                    "problem_name": state.decomposed.problem_name,
                    "keywords": state.decomposed.keywords,
                    "metadata": state.decomposed.metadata,
                }, f, indent=2)
        
        logger.info(f"[EnhancedDRoC] Outputs saved to {exp_dir}")


# Convenience function
def run_enhanced_droc(
    instance: VRPTWInstance,
    llm_client: Any,
    config: PipelineConfig | None = None,
) -> tuple[RouteSolution, ExperimentResult]:
    """Run the enhanced DRoC pipeline.
    
    Args:
        instance: The VRPTW instance.
        llm_client: LLM client.
        config: Pipeline configuration.
    
    Returns:
        Tuple of (solution, experiment_result).
    """
    pipeline = EnhancedDrocPipeline(llm_client=llm_client, config=config)
    solution, state, result = pipeline.run(instance)
    return solution, result


# Alias for backward compatibility
def run_droc_v2(
    instance: VRPTWInstance,
    llm_client: Any,
    **kwargs,
) -> tuple[RouteSolution, Any, ExperimentResult]:
    """Alias for run_enhanced_droc with extended return."""
    config = PipelineConfig(**kwargs) if kwargs else None
    pipeline = EnhancedDrocPipeline(llm_client=llm_client, config=config)
    return pipeline.run(instance)
