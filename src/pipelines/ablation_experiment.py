"""Ablation experiment runner: systematically varies hint types to measure contribution.

Compares eight variants:
  - zero_shot: solver only, no LLM hints
  - priority_only: only priority_customers hint (DEPRECATED - not supported by OR-Tools)
  - locked_only: only locked_subroutes hint
  - moves_only: only suggested_moves hint
  - full_hints: all hint types (default LLM loop)
  - retrieval_only: few-shot retrieval augmented prompts (RQ4)
  - param_only: only solver_control hints (RQ4)
  - dual_channel: retrieval + param adaptation combined (RQ4)

Note: priority_only is deprecated because OR-Tools does not support insertion
order constraints (biased_order). The priority_customers hint is logged but ignored.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from src.domain.schema import VRPTWInstance
from src.domain.event_schema import FailureMode, Improvement
from src.domain.metrics import lexicographic_compare
from src.solvers.feasibility_checker import check_feasibility
from src.baselines.greedy_insertion import solve_greedy
from src.solvers.pyvrp_solver import solve_pyvrp
from src.pipelines.run_llm_solver import run_llm_solver
from src.utils.io import read_instance
from src.utils.event_logger import EventLogger

logger = logging.getLogger(__name__)

AblationVariant = Literal[
    "zero_shot", "priority_only", "locked_only", "moves_only", "full_hints",
    "retrieval_only", "param_only", "dual_channel"
]
ABLATION_VARIANTS: list[AblationVariant] = [
    "zero_shot",
    "priority_only",  # DEPRECATED: biased_order not supported by OR-Tools
    "locked_only",
    "moves_only",
    "full_hints",
    "retrieval_only",  # Few-shot: retrieval augmented prompts
    "param_only",  # Few-shot: parameter adaptation only
    "dual_channel",  # Few-shot: retrieval + param combined
]

ABLATION_CSV_FIELDS = [
    "experiment_run_id", "instance", "family", "size", "seed",
    "variant", "hint_types_used",
    "vehicles_used", "total_distance", "feasible",
    "runtime_sec", "improvement_vs_baseline",
    "failure_mode",
]


class AblationRunner:
    """Systematically runs ablation variants to measure hint-type contribution.

    Usage:
        runner = AblationRunner(
            instances=["data/processed/C101.json"],
            seeds=[42],
            output_dir="results/ablation_001",
        )
        df = runner.run()
    """

    def __init__(
        self,
        instances: list[VRPTWInstance] | list[str] | Path,
        seeds: list[int] | None = None,
        time_limits: dict[str, int] | None = None,
        short_budget: int = 1,
        final_budget: int = 30,
        output_dir: str | Path = "results",
        experiment_run_id: str | None = None,
        max_workers: int = 1,
        baseline_time_limit: int = 5,
        verbose: bool = False,
    ):
        """Initialize the ablation runner."""
        self.instances: list[tuple[str, VRPTWInstance]] = self._resolve_instances(instances)
        self.seeds: list[int] = seeds or [42]
        self.time_limits: dict[str, int] = time_limits or {}
        self.short_budget = short_budget
        self.final_budget = final_budget
        self.output_dir = Path(output_dir)
        self.experiment_run_id = experiment_run_id or f"ablation_{int(time.time())}"
        self.max_workers = max_workers
        self.baseline_time_limit = baseline_time_limit
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = self.output_dir / f"{self.experiment_run_id}.csv"
        self._csv_written = False

    def _resolve_instances(
        self,
        instances: list[VRPTWInstance] | list[str] | Path,
    ) -> list[tuple[str, VRPTWInstance]]:
        resolved: list[tuple[str, VRPTWInstance]] = []
        for item in instances:
            if isinstance(item, VRPTWInstance):
                resolved.append((item.name, item))
            elif isinstance(item, (str, Path)):
                path = Path(item)
                if path.exists():
                    inst = read_instance(path)
                    resolved.append((inst.name, inst))
                else:
                    logger.warning(f"Instance not found: {path}")
        return resolved

    def run(self) -> "pandas.DataFrame":
        """Run all ablation variants on all instances.

        Returns:
            pandas DataFrame with one row per (instance, variant, seed).
        """
        import pandas as pd

        total = len(self.instances) * len(ABLATION_VARIANTS) * len(self.seeds)
        logger.info(
            f"Starting ablation: {len(self.instances)} instances, "
            f"{len(ABLATION_VARIANTS)} variants, {len(self.seeds)} seeds "
            f"({total} total runs)"
        )

        all_rows: list[dict] = []

        # First: run baseline (zero_shot = pyvrp cold) once per instance
        baselines = self._run_baselines()

        completed = 0
        for inst_name, inst in self.instances:
            for variant in ABLATION_VARIANTS:
                for seed in self.seeds:
                    row = self._run_variant(inst_name, inst, variant, seed, baselines)
                    all_rows.append(row)
                    completed += 1
                    if self.verbose and completed % 10 == 0:
                        logger.info(f"  Progress: {completed}/{total} ({100*completed/total:.0f}%)")

        df = pd.DataFrame(all_rows)
        df.to_csv(self.results_csv, index=False)
        logger.info(f"Ablation results saved to {self.results_csv}")

        self._print_summary(df)
        return df

    def _run_baselines(self) -> dict[str, tuple[int, float]]:
        """Run pyvrp cold-start once per instance. Cached for all variants."""
        baselines: dict[str, tuple[int, float]] = {}
        for inst_name, inst in self.instances:
            for seed in self.seeds:
                key = f"{inst_name}_seed{seed}"
                try:
                    sol = solve_pyvrp(inst, time_limit=self.baseline_time_limit, seed=seed)
                    baselines[key] = (sol.vehicles_used, sol.total_distance)
                except Exception:
                    baselines[key] = (999, 999999.0)
        return baselines

    def _run_variant(
        self,
        inst_name: str,
        inst: VRPTWInstance,
        variant: AblationVariant,
        seed: int,
        baselines: dict[str, tuple[int, float]],
    ) -> dict:
        """Run a single ablation variant on a single instance."""
        family = getattr(inst, "family", None)
        size = getattr(inst, "size", 0)
        base_key = f"{inst_name}_seed{seed}"
        base_vehicles, base_distance = baselines.get(base_key, (999, 999999.0))

        row = {
            "experiment_run_id": self.experiment_run_id,
            "instance": inst_name,
            "family": family,
            "size": size,
            "seed": seed,
            "variant": variant,
            "hint_types_used": variant,
            "vehicles_used": None,
            "total_distance": None,
            "feasible": False,
            "runtime_sec": None,
            "improvement_vs_baseline": None,
            "failure_mode": FailureMode.NONE.value,
        }

        start_time = time.perf_counter()

        try:
            if variant == "zero_shot":
                self._run_zero_shot(inst, seed, row)
            elif variant == "priority_only":
                self._run_priority_only(inst, seed, row)
            elif variant == "locked_only":
                self._run_locked_only(inst, seed, row)
            elif variant == "moves_only":
                self._run_moves_only(inst, seed, row)
            elif variant == "full_hints":
                self._run_full_hints(inst, seed, row)
            elif variant == "retrieval_only":
                self._run_retrieval_only(inst, seed, row)
            elif variant == "param_only":
                self._run_param_only(inst, seed, row)
            elif variant == "dual_channel":
                self._run_dual_channel(inst, seed, row)
        except Exception as e:
            logger.error(f"Error in {variant} for {inst_name}: {e}")
            row["failure_mode"] = FailureMode.SOLVER_ERROR.value

        row["runtime_sec"] = time.perf_counter() - start_time

        # Compute improvement vs baseline
        if row["vehicles_used"] is not None and base_vehicles < 999:
            # Lexicographic improvement
            row["improvement_vs_baseline"] = (
                (base_vehicles - row["vehicles_used"]) * 10000
                + (base_distance - row["total_distance"])
            ) if row["vehicles_used"] is not None else None

        self._append_csv(row)
        return row

    def _run_zero_shot(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Baseline: cold-start PyVRP without any hints."""
        time_limit = self.time_limits.get("pyvrp", 5)
        sol = solve_pyvrp(inst, time_limit=time_limit, seed=seed)
        val = check_feasibility(inst, sol)
        row["vehicles_used"] = sol.vehicles_used
        row["total_distance"] = round(sol.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "none"

    def _run_priority_only(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Ablation: run LLM loop but suppress all hints except priority_customers."""
        from src.llm.mock_client import MockLLMClient
        from src.llm.hint_schema import empty_hints

        client = _PriorityOnlyClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "priority"

    def _run_locked_only(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Ablation: run LLM loop but suppress all hints except locked_subroutes."""
        from src.llm.mock_client import MockLLMClient

        client = _LockedOnlyClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "locked"

    def _run_moves_only(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Ablation: run LLM loop but suppress all hints except suggested_moves."""
        client = _MovesOnlyClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "moves"

    def _run_full_hints(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Full LLM loop with all hint types."""
        from src.llm.mock_client import MockLLMClient

        client = MockLLMClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "priority+locked+moves+control"

    def _run_retrieval_only(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Few-shot: retrieval-augmented prompts only (no solver_control)."""
        client = _RetrievalOnlyClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "retrieval_only"

    def _run_param_only(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Few-shot: parameter adaptation only (no spatial hints)."""
        client = _ParamOnlyClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "param_only"

    def _run_dual_channel(
        self,
        inst: VRPTWInstance,
        seed: int,
        row: dict,
    ) -> None:
        """Few-shot: dual-channel (retrieval + param adaptation)."""
        client = _DualChannelClient(seed=seed)
        incumbent, final, _, _ = run_llm_solver(
            inst,
            llm_client=client,
            short_budget=self.short_budget,
            final_budget=self.final_budget,
            seed=seed,
            output_dir=None,
        )
        val = check_feasibility(inst, final)
        row["vehicles_used"] = final.vehicles_used
        row["total_distance"] = round(final.total_distance, 2)
        row["feasible"] = val.feasible
        row["hint_types_used"] = "dual_channel"

    def _append_csv(self, row: dict) -> None:
        """Append a row to the CSV."""
        write_header = not self._csv_written
        self._csv_written = True
        with open(self.results_csv, "a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=ABLATION_CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _print_summary(self, df: "pandas.DataFrame") -> None:
        """Print a summary of ablation results."""
        logger.info("=" * 60)
        logger.info(f"Ablation run complete: {self.experiment_run_id}")

        for variant in ABLATION_VARIANTS:
            subset = df[df["variant"] == variant]
            if subset.empty:
                continue
            rate = subset["feasible"].sum() / len(subset) * 100
            mean_v = subset["vehicles_used"].mean()
            mean_d = subset["total_distance"].mean()
            logger.info(
                f"  {variant}: {rate:.0f}% feasible, "
                f"mean={mean_v:.2f}v/{mean_d:.1f}d"
            )

        # Compare each variant to zero_shot
        zero_shot = df[df["variant"] == "zero_shot"]
        for variant in ["priority_only", "locked_only", "moves_only", "full_hints",
                        "retrieval_only", "param_only", "dual_channel"]:
            variant_df = df[df["variant"] == variant]
            if zero_shot.empty or variant_df.empty:
                continue
            # Mean improvement
            mean_imp_zero = zero_shot["vehicles_used"].mean()
            mean_imp_var = variant_df["vehicles_used"].mean()
            logger.info(
                f"  {variant} vs zero_shot: "
                f"{mean_imp_zero:.2f}v -> {mean_imp_var:.2f}v "
                f"(delta={mean_imp_var - mean_imp_zero:+.2f})"
            )

        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Ablation clients: wrap MockLLMClient to suppress specific hint types
# ---------------------------------------------------------------------------


class _AblationClientMixin:
    """Base mixin that wraps MockLLMClient and suppresses selected hint fields."""

    _suppress_priority: bool = False
    _suppress_locked: bool = False
    _suppress_moves: bool = False

    def __init__(self, seed: int = 42):
        from src.llm.mock_client import MockLLMClient
        self._client = MockLLMClient(seed=seed)

    def query(self, instance, incumbent=None):
        from src.llm.hint_schema import empty_hints, SolverControl

        hints = self._client.query(instance, incumbent)

        return hints.__class__(
            priority_customers=[] if self._suppress_priority else hints.priority_customers,
            locked_subroutes=[] if self._suppress_locked else hints.locked_subroutes,
            avoid_pairs=hints.avoid_pairs,
            suggested_moves=[] if self._suppress_moves else hints.suggested_moves,
            solver_control=hints.solver_control,
            rationale=hints.rationale,
        )


class _PriorityOnlyClient(_AblationClientMixin):
    _suppress_locked = True
    _suppress_moves = True


class _LockedOnlyClient(_AblationClientMixin):
    _suppress_priority = True
    _suppress_moves = True


class _MovesOnlyClient(_AblationClientMixin):
    _suppress_priority = True
    _suppress_locked = True


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot ablation variants (RQ4: Retrieval vs Adaptation)
# ─────────────────────────────────────────────────────────────────────────────


class _RetrievalOnlyClient:
    """Few-shot client: uses retrieval-augmented prompts only.

    This variant focuses on retrieval-based guidance: the LLM is given
    similar solved instances as few-shot examples in the prompt, but
    solver_control hints are suppressed.
    """

    def __init__(self, seed: int = 42):
        from src.llm.mock_client import MockLLMClient
        self._client = MockLLMClient(seed=seed)

    def query(self, instance, incumbent=None):
        """Query with retrieval-augmented prompt (suppresses solver_control)."""
        from src.llm.hint_schema import SolverControl

        hints = self._client.query(instance, incumbent)

        return hints.__class__(
            priority_customers=hints.priority_customers,
            locked_subroutes=hints.locked_subroutes,
            avoid_pairs=hints.avoid_pairs,
            suggested_moves=hints.suggested_moves,
            solver_control=None,  # Suppress solver_control for retrieval_only
            rationale=hints.rationale,
        )


class _ParamOnlyClient:
    """Few-shot client: parameter adaptation hints only.

    This variant focuses on parameter adaptation: the LLM is given
    diagnostic information and provides only solver_control hints
    (extra_seconds, intensify, diversify). All spatial hints are suppressed.
    """

    def __init__(self, seed: int = 42):
        from src.llm.mock_client import MockLLMClient
        self._client = MockLLMClient(seed=seed)

    def query(self, instance, incumbent=None):
        """Query with parameter adaptation only (suppresses spatial hints)."""
        hints = self._client.query(instance, incumbent)

        return hints.__class__(
            priority_customers=[],  # Suppress spatial hints
            locked_subroutes=[],  # Suppress spatial hints
            avoid_pairs=[],  # Suppress spatial hints
            suggested_moves=[],  # Suppress spatial hints
            solver_control=hints.solver_control,
            rationale=hints.rationale,
        )


class _DualChannelClient:
    """Few-shot client: retrieval + adaptation combined.

    This variant combines both channels:
    - Retrieval: uses similar solved instances as few-shot examples
    - Adaptation: provides solver_control hints

    This is equivalent to full_hints but with explicit focus on
    the two channels that matter for RQ4.
    """

    def __init__(self, seed: int = 42):
        from src.llm.mock_client import MockLLMClient
        self._client = MockLLMClient(seed=seed)

    def query(self, instance, incumbent=None):
        """Query with dual-channel guidance (retrieval + adaptation)."""
        hints = self._client.query(instance, incumbent)

        return hints.__class__(
            priority_customers=hints.priority_customers,
            locked_subroutes=hints.locked_subroutes,
            avoid_pairs=hints.avoid_pairs,
            suggested_moves=hints.suggested_moves,
            solver_control=hints.solver_control,
            rationale=hints.rationale,
        )


def run_ablation(
    instances: list[str] | Path,
    output_dir: str = "results",
    experiment_run_id: str | None = None,
    seeds: list[int] | None = None,
    short_budget: int = 1,
    final_budget: int = 30,
    baseline_time_limit: int = 5,
    max_workers: int = 1,
    verbose: bool = True,
) -> "pandas.DataFrame":
    """Convenience function to run an ablation experiment.

    Args:
        instances: Path to directory of JSON instances or manifest CSV.
        output_dir: Output directory.
        experiment_run_id: Unique run ID.
        seeds: Random seeds.
        short_budget: LLM loop initial budget.
        final_budget: LLM loop final budget.
        baseline_time_limit: Time limit for zero-shot baseline.
        max_workers: Max parallel workers.
        verbose: Print progress.

    Returns:
        pandas DataFrame of ablation results.
    """
    import pandas as pd

    # Resolve instances from directory or manifest
    path = Path(instances)
    instance_paths: list[str] = []

    if path.is_file() and path.suffix == ".csv":
        manifest_df = pd.read_csv(path)
        for col in ["path", "instance_path", "json_path"]:
            if col in manifest_df.columns:
                instance_paths = manifest_df[col].dropna().tolist()
                break
    elif path.is_dir():
        instance_paths = sorted(str(p) for p in path.glob("*.json"))
    else:
        instance_paths = [str(path)]

    if not instance_paths:
        raise ValueError("No instances found")

    runner = AblationRunner(
        instances=instance_paths,
        seeds=seeds or [42],
        short_budget=short_budget,
        final_budget=final_budget,
        baseline_time_limit=baseline_time_limit,
        output_dir=output_dir,
        experiment_run_id=experiment_run_id,
        max_workers=max_workers,
        verbose=verbose,
    )

    return runner.run()
