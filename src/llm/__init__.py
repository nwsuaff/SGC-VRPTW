"""LLM integration module.

Components:
- diagnostic_encoder: Extracts structured diagnostics from VRPTW instances and solutions
- hint_verifier: Validates LLM hints for safety and correctness
- hint_history: Tracks accepted/rejected hints and their outcomes
- prompt_builder: Constructs prompts with diagnostics, history, and constraints
- code_generator: DRoC-style code generation for VRPTW
- code_executor: Executes and validates LLM-generated solver code
- vrptw_templates: Code templates for VRPTW solvers
"""

from src.llm.diagnostic_encoder import (
    compute_diagnostics,
    compute_infeasibility_reasons,
    CustomerDiagnostics,
    EdgeDiagnostics,
    FullDiagnostics,
    RouteDiagnostics,
    SolverDiagnostics,
)

from src.llm.hint_verifier import (
    HintVerifier,
    VerificationResult,
    verify_hints,
)

from src.llm.hint_history import (
    HintHistory,
    HintHistoryManager,
    HintRecord,
)

from src.llm.code_generator import (
    CodeGenerator,
    CodeGenResult,
    GenerationState,
)

from src.llm.code_executor import (
    ExecutionResult,
    CodeValidation,
    validate_code,
    execute_solve_code,
)

from src.llm.vrptw_templates import (
    VRPTW_BASE_TEMPLATE,
    VRPTW_WARMSTART_TEMPLATE,
    VRPTW_LEXICOGRAPHIC_TEMPLATE,
    instance_to_solve_params,
)
