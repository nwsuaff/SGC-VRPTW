"""CLI for VRPTW LLM Solver."""

import logging
import os
from pathlib import Path

import yaml

import typer

# Load .env file if it exists
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

import typer
from typing import Optional

from src.pipelines.run_llm_solver import run_llm_solver
from src.pipelines.run_baseline import run_baseline
from src.pipelines.batch_experiment import run_batch
from src.pipelines.ablation_experiment import run_ablation
from src.pipelines.evaluate_experiment import evaluate_results
from src.pipelines.droc_pipeline import run_droc_solver, run_ablation_comparison
from src.pipelines.comparison_pipeline import (
    run_comparison,
    run_batch_comparison,
    run_batch_comparison_queued,
)
from src.data.preprocess import prepare_dataset
from src.data.dataset_registry import DatasetRegistry
from src.domain.schema import VRPTWInstance
from src.domain.dynamic_schema import DynamicVRPTWInstance
from src.utils.io import read_instance as load_instance
from src.llm.client_factory import create_llm_client, is_llm_available
from src.pipelines.dynamic_dispatch_pipeline import (
    DynamicDispatchPipeline,
    run_dynamic_comparison,
    load_scenario,
)

app = typer.Typer(help="SGC-VRPTW experiment CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def smoke_test():
    """Run smoke tests on all components."""
    import subprocess
    result = subprocess.run(["pytest", "tests/", "-v", "--tb=short"], cwd=".")
    raise typer.Exit(code=result.returncode)


@app.command()
def llm_loop(
    instance_path: str,
    llm: str = "mock",
    short_budget: int = 1,
    final_budget: int = 2,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Run LLM-guided solver loop on a single instance."""
    logger.info(f"Loading instance: {instance_path}")
    instance = load_instance(instance_path)

    logger.info(f"LLM client: {llm}")

    if llm != "mock" and not is_llm_available(llm):
        logger.warning(f"LLM '{llm}' not available, falling back to mock")
        llm = "mock"

    llm_client = create_llm_client(llm)

    output = Path(output_dir) if output_dir else None

    result, final, hints, exp_result = run_llm_solver(
        instance,
        llm_client=llm_client,
        short_budget=short_budget,
        final_budget=final_budget,
        seed=seed,
        output_dir=output,
    )

    logger.info(f"Results saved to {exp_result.experiment_id}")


@app.command()
def solve(
    solver: str,
    instance_path: str,
    time_limit: int = 5,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Run a baseline solver on an instance."""
    instance = load_instance(instance_path)
    solution, result = run_baseline(
        solver=solver,
        instance=instance,
        time_limit=time_limit,
        seed=seed,
        output_dir=Path(output_dir) if output_dir else None,
    )
    logger.info(f"Solution: vehicles={result.vehicles_used}, "
                f"distance={result.total_distance:.2f}, "
                f"feasible={result.feasible}")


@app.command()
def prepare_data(
    dataset: str,
    raw_dir: str = "data/raw/solomon",
    processed_dir: str = "data/processed",
    manifest_path: str = "data/manifests/manifest.csv",
    dry_run: bool = False,
):
    """Prepare dataset from raw files."""
    registry = DatasetRegistry()
    prepare_dataset(
        dataset_name=dataset,
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        manifest_path=Path(manifest_path),
        dry_run=dry_run,
    )
    logger.info(f"Prepared {dataset} dataset")


@app.command()
def run_experiments(
    data_dir: str,
    methods: str = "pyvrp,llm_loop",
    seeds: str = "42",
    output: str = "results/batch",
    llm: str = "mock",
):
    """Run batch experiments."""
    method_list = methods.split(",")
    seed_list = [int(s) for s in seeds.split(",")]

    run_batch(
        data_dir=Path(data_dir),
        methods=method_list,
        seeds=seed_list,
        output_dir=Path(output),
        llm_client_name=llm,
    )


@app.command()
def ablation(
    data_dir: str,
    seeds: str = "42",
    output: str = "results/ablation",
):
    """Run ablation experiments."""
    seed_list = [int(s) for s in seeds.split(",")]

    run_ablation(
        data_dir=Path(data_dir),
        seeds=seed_list,
        output_dir=Path(output),
    )


@app.command()
def ping(
    llm: str = "openai",
):
    """Test API connectivity with a ping."""
    if not is_llm_available(llm):
        logger.error(f"LLM '{llm}' not available.")
        raise typer.Exit(code=1)

    llm_client = create_llm_client(llm)

    if not hasattr(llm_client, "ping"):
        logger.error(f"LLM client '{llm}' does not support ping.")
        raise typer.Exit(code=1)

    result = llm_client.ping()

    print("\n--- API Ping Result ---")
    print(f"  Status:     {result['status']}")
    print(f"  Latency:    {result['latency_ms']:.0f} ms")
    print(f"  Model:      {result['model']}")
    print(f"  Base URL:   {result['base_url']}")
    if result["error"]:
        print(f"  Error:      {result['error']}")
    print("-----------------------\n")

    if result["status"] != "ok":
        raise typer.Exit(code=1)


def _run_sgc_instance(
    instance_path: str,
    llm: str,
    max_iterations: int,
    time_limit: int,
    seed: int,
    output_dir: Optional[str],
):
    """Shared implementation for SGC command aliases."""
    logger.info(f"Loading instance: {instance_path}")
    instance = load_instance(instance_path)

    logger.info(f"LLM client: {llm}")

    if llm != "mock" and not is_llm_available(llm):
        logger.warning(f"LLM '{llm}' not available, falling back to mock")
        llm = "mock"

    llm_client = create_llm_client(llm)

    output = Path(output_dir) if output_dir else None

    solution, gen_result, exp_result = run_droc_solver(
        instance,
        llm_client=llm_client,
        max_iterations=max_iterations,
        time_limit=time_limit,
        seed=seed,
        output_dir=output,
    )

    logger.info(f"Results: vehicles={exp_result.vehicles_used}, "
                f"distance={exp_result.total_distance:.2f}, "
                f"feasible={exp_result.feasible}, "
                f"iterations={gen_result.iterations}")


@app.command()
def sgc(
    instance_path: str,
    llm: str = "mock",
    max_iterations: int = 4,
    time_limit: int = 60,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Run the SGC-VRPTW generated-solver workflow on one instance."""
    _run_sgc_instance(instance_path, llm, max_iterations, time_limit, seed, output_dir)


@app.command()
def droc(
    instance_path: str,
    llm: str = "mock",
    max_iterations: int = 4,
    time_limit: int = 60,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Backward-compatible alias for the SGC-VRPTW workflow."""
    _run_sgc_instance(instance_path, llm, max_iterations, time_limit, seed, output_dir)


@app.command()
def droc_ablation(
    data_dir: str,
    llm: str = "mock",
    seeds: str = "42",
    max_iterations: int = 4,
    time_limit: int = 60,
    output: str = "results/droc_batch",
):
    """Run SGC-VRPTW batch experiments on multiple instances.
    
    This compares SGC-VRPTW with different configurations (e.g., with/without RAG,
    different max_iterations).
    """
    import pandas as pd

    seed_list = [int(s) for s in seeds.split(",")]

    # Load instances
    instance_paths = sorted(Path(data_dir).glob("*.json"))
    logger.info(f"Found {len(instance_paths)} instances in {data_dir}")
    
    if not instance_paths:
        logger.warning(f"No instances found in {data_dir}")
        return

    instances = [load_instance(str(p)) for p in instance_paths[:10]]  # Limit to 10 for quick test

    llm_client = create_llm_client(llm) if llm != "mock" else None

    # Run SGC-VRPTW on each instance
    results = []
    for instance in instances:
        for seed in seed_list:
            logger.info(f"Running SGC-VRPTW on {instance.name} (seed={seed})")
            try:
                solution, gen_result, exp_result = run_droc_solver(
                    instance,
                    llm_client=llm_client,
                    max_iterations=max_iterations,
                    time_limit=time_limit,
                    seed=seed,
                    output_dir=None,
                )
                results.append({
                    "instance": instance.name,
                    "seed": seed,
                    "vehicles": exp_result.vehicles_used,
                    "distance": exp_result.total_distance,
                    "feasible": exp_result.feasible,
                    "iterations": gen_result.iterations,
                    "runtime": exp_result.runtime_sec,
                })
            except Exception as e:
                logger.error(f"Failed on {instance.name}: {e}")
                results.append({
                    "instance": instance.name,
                    "seed": seed,
                    "vehicles": None,
                    "distance": None,
                    "feasible": False,
                    "iterations": None,
                    "runtime": None,
                    "error": str(e),
                })

    # Save results
    df = pd.DataFrame(results)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "droc_results.csv", index=False)
    logger.info(f"Results saved to {output}/droc_results.csv")


@app.command()
def compare(
    solver: str = "ortools",
    data_dir: str = "data/VRPTW/",
    instances: str = "gh800",
    llm: str = "mock",
    seeds: str = "42",
    max_iterations: int = 6,
    time_limit: int = 300,
    output: str = "results/comparison",
    max_instances: int = None,
    per_instance_timeout: int = 900,
    resume: bool = True,
    incremental: bool = True,
):
    """Compare SGC-VRPTW generated code against a baseline solver.

    Runs instances one by one, saves results after each instance completes.
    If interrupted and re-run with --resume (default), already-completed instances
    are skipped. If an instance runs longer than --timeout, it is recorded as
    timed-out and the next instance starts.

    Examples:
        # Full run with default 15min timeout per instance
        python -m src.cli compare --solver ortools --instances gh800 --llm openai

        # Quick test on 5 instances
        python -m src.cli compare --solver ortools --instances gh800 --llm mock --max-instances 5

        # Re-run everything from scratch (don't resume)
        python -m src.cli compare --solver ortools --instances gh800 --llm openai --no-resume

        # 5min timeout per instance
        python -m src.cli compare --solver ortools --instances gh800 --llm openai --per-instance-timeout 300
    """
    import pandas as pd

    seed_list = [int(s) for s in seeds.split(",")]

    logger.info(f"Setting up comparison: SGC-VRPTW vs {solver}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Instance filter: {instances}")
    logger.info(f"LLM: {llm}")
    logger.info(f"Per-instance timeout: {per_instance_timeout}s")
    logger.info(f"Incremental save + resume: {incremental and resume}")

    if not is_llm_available(llm):
        logger.error(f"LLM '{llm}' is not available. Please check your API key.")
        return

    llm_client = create_llm_client(llm)

    if hasattr(llm_client, "ping"):
        ping_result = llm_client.ping()
        if ping_result["status"] != "ok":
            logger.error(
                f"LLM connectivity check failed: {ping_result['error']} "
                f"(latency={ping_result['latency_ms']:.0f}ms). "
                f"Aborting comparison."
            )
            return
        logger.info(
            f"[Compare] LLM ping OK — model={ping_result['model']}, "
            f"latency={ping_result['latency_ms']:.0f}ms"
        )

    if incremental:
        # Incremental: one by one, saves after each, resume on restart
        output_data = run_batch_comparison(
            data_dir=data_dir,
            llm_client=llm_client,
            solver=solver,
            instances=instances,
            max_iterations=max_iterations,
            time_limit=time_limit,
            seeds=seed_list,
            output_dir=output,
            max_instances=max_instances,
            per_instance_timeout=float(per_instance_timeout),
            resume=resume,
        )
    else:
        # Parallel queued version (original behavior)
        output_data = run_batch_comparison_queued(
            data_dir=data_dir,
            llm_client=llm_client,
            solver=solver,
            instances=instances,
            max_iterations=max_iterations,
            time_limit=time_limit,
            seeds=seed_list,
            output_dir=output,
            max_instances=max_instances,
            max_workers=4,
        )

    df = output_data["df"]
    summary = output_data["summary"]

    print("\n" + "=" * 60)
    print(f"COMPARISON SUMMARY: SGC-VRPTW vs {solver}")
    print("=" * 60)
    print(f"Total runs:  {len(df)}")
    success_cnt = int(df["droc_feasible"].sum()) if "droc_feasible" in df.columns else 0
    fail_cnt = len(df) - success_cnt
    print(f"SGC OK:      {success_cnt} ({success_cnt / len(df) * 100:.0f}%)")
    print(f"SGC FAIL:    {fail_cnt} ({fail_cnt / len(df) * 100:.0f}%)")
    print()
    if "droc_feasible_rate" in summary:
        print(f"SGC feasible rate:     {summary.get('droc_feasible_rate', 0):.1%}")
        print(f"Baseline feasible rate: {summary.get('baseline_feasible_rate', 0):.1%}")
    if summary.get("droc_avg_vehicles"):
        print()
        print(f"SGC avg vehicles:         {summary['droc_avg_vehicles']:.1f}")
        print(f"Baseline avg vehicles:    {summary['baseline_avg_vehicles']:.1f}")
        print(f"Avg vehicle improvement:  {summary.get('avg_vehicle_improvement', 0):.2f}%")
    print()
    print(f"Results: {output}/comparison_results.csv")
    print(f"Summary: {output}/summary.csv")
    print("=" * 60)


@app.command()
def evaluate(output_dir: str):
    """Evaluate experiment results."""
    evaluate_results(Path(output_dir))


@app.command()
def generate_ortec(
    sizes: str = "300,500,800",
    instances_per_size: int = 3,
    start_seed: int = 42,
    output_dir: str = "data/VRPTW/ORTEC/static",
):
    """Generate synthetic ORTEC-style static VRPTW instances.

    Creates .txt (VRPLIB) and .json files for each instance.

    Examples:
        python -m src.cli generate-ortec
        python -m src.cli generate-ortec --sizes 200,400,600 --instances-per-size 5
    """
    from src.data.dynamic_ortec_generator import generate_ortec_benchmark_suite

    size_list = [int(s.strip()) for s in sizes.split(",")]
    logger.info(f"Generating ORTEC instances: sizes={size_list}, "
                f"instances_per_size={instances_per_size}")

    instances = generate_ortec_benchmark_suite(
        sizes=size_list,
        instances_per_size=instances_per_size,
        output_dir=Path(output_dir),
        start_seed=start_seed,
    )
    logger.info(f"Generated {len(instances)} instances in {output_dir}")


@app.command()
def generate_dynamic(
    instance_path: str,
    num_epochs: int = 5,
    orders_per_epoch: int | None = None,
    output_path: str | None = None,
):
    """Generate a dynamic dispatch scenario from a static VRPTW instance.

    The static instance provides the customer base. Orders are sampled from
    this base and released across epochs following the ORTEC competition protocol:
    - Each epoch lasts EPOCH_DURATION = 3600s
    - New orders are released at the start of each epoch
    - Orders with near-closing time windows become must-go

    Examples:
        python -m src.cli generate-dynamic data/VRPTW/ORTEC/static/ORTEC-SYNTH-300-s42.txt
    """
    import json
    import random
    import math

    instance = load_instance(instance_path)
    logger.info(f"Loaded {instance.name}: {instance.size} customers")

    if instance.duration_matrix is None and instance.distance_matrix is not None:
        duration_matrix = instance.distance_matrix
    elif instance.duration_matrix is not None:
        duration_matrix = instance.duration_matrix
    else:
        logger.warning("No duration/distance matrix; computing Euclidean")
        n = instance.size + 1
        duration_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dx = instance.x_coords[i] - instance.x_coords[j]
                dy = instance.y_coords[i] - instance.y_coords[j]
                duration_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

    n_customers = instance.size
    if orders_per_epoch is None:
        orders_per_epoch = max(1, n_customers // (num_epochs * 2))

    rng = random.Random(hash(instance.name) & 0xFFFFFFFF)

    EPOCH_DURATION = 3600
    MARGIN_DISPATCH = 3600
    depot_ready = instance.ready_time.get(0, 0)
    depot_due = instance.due_time.get(0, 999999)

    start_epoch = max(0, int(
        (min(instance.ready_time.get(i, 0) for i in instance.customer_ids) - MARGIN_DISPATCH)
        / EPOCH_DURATION
    ))
    end_epoch = max(0, int(
        (max(instance.due_time.get(i, 0) for i in instance.customer_ids) - MARGIN_DISPATCH)
        / EPOCH_DURATION
    ))
    num_epochs = max(num_epochs, end_epoch - start_epoch + 1)

    depot = {
        "request_id": 0,
        "customer_idx": 0,
        "x": float(instance.x_coords.get(0, 0.0)),
        "y": float(instance.y_coords.get(0, 0.0)),
        "demand": 0,
        "ready_time": depot_ready,
        "due_time": depot_due,
        "service_time": instance.service_time.get(0, 0),
        "release_epoch": 0,
        "is_dispatched": False,
        "must_dispatch": False,
    }

    all_orders = []
    for cid in instance.customer_ids:
        all_orders.append({
            "customer_idx": cid,
            "demand": instance.demand.get(cid, 0),
            "service_time": instance.service_time.get(cid, 0),
            "ready_time": instance.ready_time.get(cid, 0),
            "due_time": instance.due_time.get(cid, 0),
            "x": float(instance.x_coords.get(cid, 0.0)),
            "y": float(instance.y_coords.get(cid, 0.0)),
        })

    rng.shuffle(all_orders)

    orders_by_epoch: list[list] = [[] for _ in range(num_epochs)]
    for i, order in enumerate(all_orders):
        epoch = i % num_epochs
        orders_by_epoch[epoch].append(order)

    orders_list = [depot]
    request_id = 1

    for epoch in range(num_epochs):
        for order in orders_by_epoch[epoch]:
            planning_time = epoch * EPOCH_DURATION + MARGIN_DISPATCH
            earliest_arrival = max(planning_time + duration_matrix[0][order["customer_idx"] + 1],
                                   order["ready_time"])
            is_must_go = False
            if epoch < num_epochs - 1:
                next_planning = (epoch + 1) * EPOCH_DURATION + MARGIN_DISPATCH
                earliest_next = max(
                    next_planning + duration_matrix[0][order["customer_idx"] + 1],
                    order["ready_time"]
                )
                is_must_go = (
                    earliest_next > order["due_time"] or
                    (earliest_next + order["service_time"] + duration_matrix[order["customer_idx"] + 1][0]
                     > depot_due)
                )

            orders_list.append({
                "request_id": request_id,
                "customer_idx": order["customer_idx"],
                "x": order["x"],
                "y": order["y"],
                "demand": order["demand"],
                "ready_time": order["ready_time"],
                "due_time": order["due_time"],
                "service_time": order["service_time"],
                "release_epoch": epoch,
                "is_dispatched": False,
                "must_dispatch": is_must_go,
            })
            request_id += 1

    scenario = {
        "name": f"{instance.name}_dyn{num_epochs}e",
        "base_instance_name": instance.name,
        "source": "ortec_synthetic",
        "num_epochs": num_epochs,
        "epoch_duration": EPOCH_DURATION,
        "margin_dispatch": MARGIN_DISPATCH,
        "num_orders_total": len(all_orders),
        "num_orders_per_epoch": orders_per_epoch,
        "orders": orders_list,
        "duration_matrix": duration_matrix,
        "vehicle_capacity": instance.vehicle_capacity,
        "vehicle_count": instance.vehicle_count,
        "metadata": {
            "generated_from": instance.name,
            "seed": rng.randint(0, 2**31 - 1),
        },
    }

    if output_path is None:
        output_path = f"data/VRPTW/ORTEC/dynamic/{instance.name}_dyn{num_epochs}e.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, default=str)

    logger.info(f"Generated dynamic scenario: {output_path}")
    logger.info(f"  {num_epochs} epochs, {len(all_orders)} total orders")
    logger.info(f"  Orders per epoch: {orders_per_epoch}")


@app.command()
def dynamic_dispatch(
    scenario_path: str,
    instance_path: str,
    method: str = "ortools",
    llm: str = "mock",
    num_epochs: int = 5,
    time_limit_per_epoch: int = 60,
    output: str = "results/dynamic",
):
    """Run dynamic dispatch on a scenario with OR-Tools or SGC-VRPTW.

    Evaluates solver performance on an ORTEC-style order-arrival scenario
    with per-epoch served-rate tracking.

    Examples:
        # OR-Tools baseline on a synthetic dynamic scenario
        python -m src.cli dynamic-dispatch \\
            --scenario-path data/VRPTW/ORTEC/dynamic/ORTEC-SYNTH-300-s42_dyn5e.json \\
            --instance-path data/VRPTW/ORTEC/static/ORTEC-SYNTH-300-s42.txt \\
            --method ortools

        # SGC-VRPTW on the same scenario
        python -m src.cli dynamic-dispatch \\
            --scenario-path data/VRPTW/ORTEC/dynamic/ORTEC-SYNTH-300-s42_dyn5e.json \\
            --instance-path data/VRPTW/ORTEC/static/ORTEC-SYNTH-300-s42.txt \\
            --method droc --llm openai
    """
    logger.info(f"Loading scenario: {scenario_path}")
    scenario = load_scenario(scenario_path)

    logger.info(f"Loading base instance: {instance_path}")
    base_instance = load_instance(instance_path)

    logger.info(f"Method: {method}, epochs: {num_epochs}, "
                f"time_limit: {time_limit_per_epoch}s")

    if method == "droc":
        if not is_llm_available(llm):
            logger.warning(f"LLM '{llm}' not available, using mock")
            llm = "mock"
        llm_client = create_llm_client(llm)
    else:
        llm_client = None

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DynamicDispatchPipeline(
        solver=method,
        llm_client=llm_client,
        time_limit_per_epoch=float(time_limit_per_epoch),
        droc_timeout=float(time_limit_per_epoch * 3),
        output_dir=output_dir,
    )

    result = pipeline.run(scenario=scenario, base_instance=base_instance)

    print("\n" + "=" * 60)
    print(f"DYNAMIC DISPATCH RESULTS: {scenario.name}")
    print("=" * 60)
    print(f"Method:           {method}")
    print(f"Epochs:           {result.num_epochs}")
    print(f"Orders served:    {result.overall_served_rate:.1%}")
    print(f"Overall feasible: {result.overall_feasible}")
    print(f"Total runtime:   {result.total_runtime_sec:.1f}s")
    print()
    for epoch_res in result.epoch_results:
        print(f"  Epoch {epoch_res.epoch}: "
              f"served_rate={epoch_res.served_rate:.1%}, "
              f"feasible={epoch_res.feasible}, "
              f"runtime={epoch_res.runtime_sec:.1f}s, "
              f"late={epoch_res.late_violations}")
    print()
    print(f"Result saved to: {output_dir / 'result.json'}")
    print("=" * 60)


@app.command()
def dynamic_compare(
    scenario_path: str,
    instance_path: str,
    methods: str = "ortools,greedy",
    llm: str = "mock",
    time_limit_per_epoch: int = 60,
    output: str = "results/dynamic_compare",
):
    """Compare multiple dynamic dispatch solvers on a scenario.

    Runs the same scenario with different solvers (ortools, droc, greedy)
    and produces comparison results.

    Examples:
        python -m src.cli dynamic-compare \\
            --scenario-path data/VRPTW/ORTEC/dynamic/ORTEC-SYNTH-300-s42_dyn5e.json \\
            --instance-path data/VRPTW/ORTEC/static/ORTEC-SYNTH-300-s42.txt \\
            --methods ortools,greedy
    """
    import pandas as pd

    scenario = load_scenario(scenario_path)
    base_instance = load_instance(instance_path)
    method_list = [m.strip() for m in methods.split(",")]

    logger.info(f"Comparing methods: {method_list}")

    if "droc" in method_list:
        if not is_llm_available(llm):
            logger.warning(f"LLM '{llm}' not available, removing droc")
            method_list.remove("droc")
        else:
            llm_client = create_llm_client(llm)
    else:
        llm_client = None

    results = run_dynamic_comparison(
        scenario=scenario,
        base_instance=base_instance,
        methods=method_list,
        llm_client=llm_client,
        time_limit_per_epoch=float(time_limit_per_epoch),
        output_dir=Path(output),
    )

    print("\n" + "=" * 60)
    print(f"DYNAMIC COMPARISON: {scenario.name}")
    print("=" * 60)
    for method, result in results.items():
        print(f"\n  [{method}]")
        print(f"    Served rate:    {result.overall_served_rate:.1%}")
        print(f"    Feasible:       {result.overall_feasible}")
        print(f"    Total distance: {result.total_distance:.0f}")
        print(f"    Runtime:        {result.total_runtime_sec:.1f}s")
        for epoch_res in result.epoch_results:
            print(f"      E{epoch_res.epoch}: "
                  f"rate={epoch_res.served_rate:.1%} "
                  f"feas={epoch_res.feasible}")

    rows = []
    for method, result in results.items():
        for epoch_res in result.epoch_results:
            rows.append({
                "method": method,
                "epoch": epoch_res.epoch,
                "served_rate": epoch_res.served_rate,
                "feasible": epoch_res.feasible,
                "total_distance": epoch_res.total_distance,
                "runtime_sec": epoch_res.runtime_sec,
                "late_violations": epoch_res.late_violations,
                "capacity_violations": epoch_res.capacity_violations,
            })

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dynamic_comparison.csv", index=False)
    print(f"\nResults: {output}/dynamic_comparison.csv")
    print("=" * 60)


if __name__ == "__main__":
    app()
