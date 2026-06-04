#!/usr/bin/env python3
"""Run missing conclusion-level experiments with traceable CSV outputs.

This script is intentionally conservative. It only treats variants as executable
when the current DRoC pipeline has a real switch for that component. Unsupported
component claims are written as explicit rows rather than silently fabricating an
ablation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.schema import RouteSolution, VRPTWInstance
from src.solvers.feasibility_checker import check_feasibility
from src.solvers.ortools_solver import solve_ortools
from src.pipelines.droc_pipeline import run_droc_solver
from src.llm.client_factory import create_llm_client
from src.utils.io import read_instance


ABLATION_FIELDS = [
    "dataset",
    "unit_id",
    "scale",
    "seed",
    "variant",
    "variant_label",
    "time_limit_sec",
    "llm_model",
    "solver_version",
    "feasible",
    "served_rate",
    "vehicles",
    "distance",
    "runtime_sec",
    "debug_iterations",
    "failure_category",
    "baseline_feasible",
    "baseline_vehicles",
    "baseline_distance",
    "baseline_runtime_sec",
    "variant_supported",
    "error",
    "timestamp",
]

ANYTIME_FIELDS = [
    "dataset",
    "unit_id",
    "scale",
    "seed",
    "method",
    "time_checkpoint_sec",
    "incumbent_feasible",
    "incumbent_vehicles",
    "incumbent_distance",
    "served_rate",
    "best_known_vehicles",
    "best_known_distance",
    "runtime_sec",
    "trace_type",
    "timestamp",
]

REPLICATE_FIELDS = [
    "dataset",
    "unit_id",
    "scale",
    "seed",
    "method",
    "aggregation_rule",
    "feasible",
    "vehicles",
    "distance",
    "served_rate",
    "runtime_sec",
    "timestamp",
]


VARIANT_LABELS = {
    "solver_only": "OR-Tools baseline",
    "full_droc": "Full DRoC",
    "no_self_debug": "DRoC without self-debug",
    "no_incumbent": "DRoC without incumbent reference",
    "no_retrieval": "DRoC without retrieval",
    "no_scene_card": "DRoC without scene card",
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def solver_version() -> str:
    try:
        import ortools

        return f"ortools-{getattr(ortools, '__version__', 'unknown')}"
    except Exception:
        return "ortools-unknown"


def parse_scale(path: Path) -> int:
    match = re.search(r"ORTEC-SYNTH-(\d+)-", path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"n(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return 0


def select_instances(
    input_dir: Path,
    sample_size: int | None,
    scales: set[int] | None,
) -> list[Path]:
    files = sorted(input_dir.glob("ORTEC-SYNTH-*.txt"))
    if scales:
        files = [p for p in files if parse_scale(p) in scales]
    if sample_size is None or sample_size <= 0 or sample_size >= len(files):
        return files

    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        grouped[parse_scale(path)].append(path)

    selected: list[Path] = []
    scale_order = sorted(grouped)
    cursor = 0
    while len(selected) < sample_size and scale_order:
        scale = scale_order[cursor % len(scale_order)]
        group = grouped[scale]
        if group:
            selected.append(group.pop(0))
        scale_order = [s for s in scale_order if grouped[s]]
        cursor += 1
    return selected


def append_rows(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def classify_solution(solution: RouteSolution, feasible: bool, error: str = "") -> str:
    if error:
        return "runtime_error"
    if feasible:
        return "none"
    if not solution.routes:
        return "no_solution"
    return "constraint_violation"


def finite_distance(solution: RouteSolution) -> float | str:
    value = solution.total_distance
    if value is None or not math.isfinite(float(value)):
        return ""
    return round(float(value), 6)


def solve_baseline(instance: VRPTWInstance, time_limit: float) -> tuple[RouteSolution, bool]:
    solution, _ = solve_ortools(instance, time_limit=time_limit, warm_start=None)
    validation = check_feasibility(instance, solution)
    solution.feasible = validation.feasible
    solution.late_violations = validation.late_violations
    solution.capacity_violations = validation.capacity_violations
    return solution, validation.feasible


def make_baseline_row(
    instance: VRPTWInstance,
    seed: int,
    baseline: RouteSolution,
    baseline_feasible: bool,
    model_label: str,
    variant: str = "solver_only",
) -> dict:
    return {
        "dataset": "ortec_static",
        "unit_id": instance.name,
        "scale": instance.size,
        "seed": seed,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "time_limit_sec": "",
        "llm_model": model_label,
        "solver_version": solver_version(),
        "feasible": baseline_feasible,
        "served_rate": "",
        "vehicles": baseline.vehicles_used,
        "distance": finite_distance(baseline),
        "runtime_sec": round(baseline.runtime_sec, 6),
        "debug_iterations": 0,
        "failure_category": classify_solution(baseline, baseline_feasible),
        "baseline_feasible": baseline_feasible,
        "baseline_vehicles": baseline.vehicles_used,
        "baseline_distance": finite_distance(baseline),
        "baseline_runtime_sec": round(baseline.runtime_sec, 6),
        "variant_supported": True,
        "error": "",
        "timestamp": now(),
    }


def unsupported_row(
    instance: VRPTWInstance,
    seed: int,
    variant: str,
    baseline: RouteSolution,
    baseline_feasible: bool,
    model_label: str,
    time_limit: float,
    reason: str,
) -> dict:
    row = make_baseline_row(instance, seed, baseline, baseline_feasible, model_label, variant="solver_only")
    row.update(
        {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "time_limit_sec": time_limit,
            "feasible": "",
            "vehicles": "",
            "distance": "",
            "runtime_sec": "",
            "debug_iterations": "",
            "failure_category": "unsupported_current_pipeline",
            "variant_supported": False,
            "error": reason,
            "timestamp": now(),
        }
    )
    return row


def run_droc_variant(
    instance: VRPTWInstance,
    seed: int,
    variant: str,
    baseline: RouteSolution,
    baseline_feasible: bool,
    llm_name: str,
    model: str | None,
    time_limit: float,
    max_iterations: int,
    artifacts_dir: Path,
) -> dict:
    model_label = model or llm_name

    if variant == "solver_only":
        row = make_baseline_row(instance, seed, baseline, baseline_feasible, model_label)
        row["time_limit_sec"] = time_limit
        return row

    if variant == "no_retrieval":
        return unsupported_row(
            instance,
            seed,
            variant,
            baseline,
            baseline_feasible,
            model_label,
            time_limit,
            "CodeGenerator.enable_rag is currently accepted but not used to change prompt retrieval; this is not a valid isolated ablation.",
        )

    if variant == "no_scene_card":
        return unsupported_row(
            instance,
            seed,
            variant,
            baseline,
            baseline_feasible,
            model_label,
            time_limit,
            "Current DRoC pipeline has no independent scene-card switch; prompt construction must be refactored before this ablation is meaningful.",
        )

    enable_self_debug = variant != "no_self_debug"
    incumbent = None if variant == "no_incumbent" else baseline

    llm_client = create_llm_client(llm_name, model=model)
    error = ""
    try:
        solution, gen_result, exp_result = run_droc_solver(
            instance=instance,
            llm_client=llm_client,
            max_iterations=max_iterations,
            time_limit=time_limit,
            seed=seed,
            output_dir=artifacts_dir / variant / instance.name / f"seed_{seed}",
            enable_self_debug=enable_self_debug,
            enable_rag=True,
            incumbent=incumbent,
            droc_timeout=time_limit,
        )
        feasible = bool(exp_result.feasible)
        debug_iterations = gen_result.iterations
    except Exception as exc:  # keep the run table complete even when one run fails
        solution = RouteSolution(
            routes=[],
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            runtime_sec=0.0,
        )
        feasible = False
        debug_iterations = 0
        error = repr(exc)

    return {
        "dataset": "ortec_static",
        "unit_id": instance.name,
        "scale": instance.size,
        "seed": seed,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "time_limit_sec": time_limit,
        "llm_model": model_label,
        "solver_version": solver_version(),
        "feasible": feasible,
        "served_rate": "",
        "vehicles": solution.vehicles_used,
        "distance": finite_distance(solution),
        "runtime_sec": round(solution.runtime_sec, 6),
        "debug_iterations": debug_iterations,
        "failure_category": classify_solution(solution, feasible, error),
        "baseline_feasible": baseline_feasible,
        "baseline_vehicles": baseline.vehicles_used,
        "baseline_distance": finite_distance(baseline),
        "baseline_runtime_sec": round(baseline.runtime_sec, 6),
        "variant_supported": True,
        "error": error,
        "timestamp": now(),
    }


def cmd_ablation(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(VARIANT_LABELS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    paths = select_instances(input_dir, args.sample_size, parse_scales(args.scales))
    if not paths:
        raise FileNotFoundError(f"No ORTEC static .txt files found in {input_dir}")

    csv_path = output_dir / "matched_ablation_runs.csv"
    artifacts_dir = output_dir / "artifacts"
    model_label = args.model or args.llm

    for seed in parse_int_list(args.seeds):
        for path in paths:
            print(f"[ablation] seed={seed} instance={path.name}", flush=True)
            instance = read_instance(path)
            baseline, baseline_feasible = solve_baseline(instance, args.baseline_time_limit)
            rows = []
            for variant in variants:
                rows.append(
                    run_droc_variant(
                        instance=instance,
                        seed=seed,
                        variant=variant,
                        baseline=baseline,
                        baseline_feasible=baseline_feasible,
                        llm_name=args.llm,
                        model=args.model,
                        time_limit=args.time_limit,
                        max_iterations=args.max_iterations,
                        artifacts_dir=artifacts_dir,
                    )
                )
            append_rows(csv_path, ABLATION_FIELDS, rows)

    write_run_readme(
        output_dir,
        "Matched component ablation runs",
        [
            f"input_dir: `{input_dir}`",
            f"instances: `{len(paths)}`",
            f"seeds: `{args.seeds}`",
            f"variants: `{','.join(variants)}`",
            f"llm: `{model_label}`",
            f"output: `{csv_path}`",
        ],
    )
    print(f"Wrote {csv_path}")


def cmd_anytime(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    paths = select_instances(input_dir, args.sample_size, parse_scales(args.scales))
    checkpoints = parse_float_list(args.checkpoints)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = parse_int_list(args.seeds)
    if not paths:
        raise FileNotFoundError(f"No ORTEC static .txt files found in {input_dir}")

    anytime_csv = output_dir / "anytime_quality_traces.csv"
    replicate_csv = output_dir / "protocol_replicates_by_seed.csv"
    artifacts_dir = output_dir / "artifacts"

    for seed in seeds:
        for path in paths:
            print(f"[anytime] seed={seed} instance={path.name}", flush=True)
            instance = read_instance(path)
            best_vehicles = ""
            best_distance = ""
            final_rows = []

            for method in methods:
                for checkpoint in checkpoints:
                    started = time.perf_counter()
                    if method == "ortools":
                        solution, feasible = solve_baseline(instance, checkpoint)
                    elif method == "full_droc":
                        baseline, baseline_feasible = solve_baseline(instance, args.baseline_time_limit)
                        row = run_droc_variant(
                            instance=instance,
                            seed=seed,
                            variant="full_droc",
                            baseline=baseline,
                            baseline_feasible=baseline_feasible,
                            llm_name=args.llm,
                            model=args.model,
                            time_limit=checkpoint,
                            max_iterations=args.max_iterations,
                            artifacts_dir=artifacts_dir,
                        )
                        solution = RouteSolution(
                            routes=[],
                            vehicles_used=int(row["vehicles"] or 0),
                            total_distance=float(row["distance"] or 0.0),
                            total_duration=0.0,
                            feasible=bool(row["feasible"]),
                            runtime_sec=float(row["runtime_sec"] or 0.0),
                        )
                        feasible = bool(row["feasible"])
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    runtime = time.perf_counter() - started
                    if feasible and (best_vehicles == "" or solution.vehicles_used < int(best_vehicles)):
                        best_vehicles = solution.vehicles_used
                        best_distance = finite_distance(solution)
                    elif (
                        feasible
                        and best_vehicles != ""
                        and solution.vehicles_used == int(best_vehicles)
                        and best_distance != ""
                        and finite_distance(solution) != ""
                        and float(finite_distance(solution)) < float(best_distance)
                    ):
                        best_distance = finite_distance(solution)

                    append_rows(
                        anytime_csv,
                        ANYTIME_FIELDS,
                        [
                            {
                                "dataset": "ortec_static",
                                "unit_id": instance.name,
                                "scale": instance.size,
                                "seed": seed,
                                "method": method,
                                "time_checkpoint_sec": checkpoint,
                                "incumbent_feasible": feasible,
                                "incumbent_vehicles": solution.vehicles_used,
                                "incumbent_distance": finite_distance(solution),
                                "served_rate": "",
                                "best_known_vehicles": best_vehicles,
                                "best_known_distance": best_distance,
                                "runtime_sec": round(runtime, 6),
                                "trace_type": "independent_budget_rerun",
                                "timestamp": now(),
                            }
                        ],
                    )

                final_rows.append(
                    {
                        "dataset": "ortec_static",
                        "unit_id": instance.name,
                        "scale": instance.size,
                        "seed": seed,
                        "method": method,
                        "aggregation_rule": "largest_checkpoint",
                        "feasible": feasible,
                        "vehicles": solution.vehicles_used,
                        "distance": finite_distance(solution),
                        "served_rate": "",
                        "runtime_sec": round(solution.runtime_sec, 6),
                        "timestamp": now(),
                    }
                )

            append_rows(replicate_csv, REPLICATE_FIELDS, final_rows)

    write_run_readme(
        output_dir,
        "Anytime and seed-replicate runs",
        [
            f"input_dir: `{input_dir}`",
            f"instances: `{len(paths)}`",
            f"seeds: `{args.seeds}`",
            f"methods: `{args.methods}`",
            f"checkpoints: `{args.checkpoints}`",
            f"trace_type: `independent_budget_rerun`",
            f"outputs: `{anytime_csv}`, `{replicate_csv}`",
        ],
    )
    print(f"Wrote {anytime_csv}")
    print(f"Wrote {replicate_csv}")


def write_run_readme(output_dir: Path, title: str, lines: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", "", f"Generated: `{now()}`", ""]
    body.extend(f"- {line}" for line in lines)
    body.append("")
    body.append(
        "Rows are append-only. Remove or archive the output directory before a clean full rerun."
    )
    (output_dir / "README.md").write_text("\n".join(body), encoding="utf-8")


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_scales(text: str | None) -> set[int] | None:
    if not text:
        return None
    return set(parse_int_list(text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-dir", default="data/VRPTW/ORTEC/static")
    common.add_argument("--output", required=True)
    common.add_argument("--sample-size", type=int, default=None)
    common.add_argument("--scales", default=None, help="Comma-separated scales, e.g. 300,500")
    common.add_argument("--seeds", default="42")
    common.add_argument("--llm", default="mock")
    common.add_argument("--model", default=None)
    common.add_argument("--baseline-time-limit", type=float, default=10.0)
    common.add_argument("--time-limit", type=float, default=300.0)
    common.add_argument("--max-iterations", type=int, default=4)

    p_ablation = sub.add_parser("ablation", parents=[common])
    p_ablation.add_argument(
        "--variants",
        default="solver_only,full_droc,no_self_debug,no_incumbent,no_retrieval,no_scene_card",
    )
    p_ablation.set_defaults(func=cmd_ablation)

    p_anytime = sub.add_parser("anytime", parents=[common])
    p_anytime.add_argument("--methods", default="ortools,full_droc")
    p_anytime.add_argument("--checkpoints", default="90,180,300")
    p_anytime.set_defaults(func=cmd_anytime)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
