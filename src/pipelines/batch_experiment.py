"""Batch Experiment Runner - Runs multiple solver methods across multiple instances."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from src.domain.schema import VRPTWInstance
from src.utils.io import read_instance

logger = logging.getLogger(__name__)


def run_batch(
    data_dir: Path,
    methods: list[str],
    seeds: list[int],
    output_dir: Path,
    llm_client_name: str = "mock",
) -> None:
    """Run batch experiments across multiple instances and methods.
    
    Args:
        data_dir: Directory containing VRPTW instance files.
        methods: List of method names ("pyvrp", "llm_loop", "ortools").
        seeds: List of random seeds.
        output_dir: Directory to save results.
        llm_client_name: LLM client name for llm_loop method.
    """
    import pandas as pd
    
    instance_paths = sorted(data_dir.glob("*.json"))
    logger.info(f"Found {len(instance_paths)} instances in {data_dir}")
    
    if not instance_paths:
        logger.warning(f"No .json instances found in {data_dir}")
        return
    
    rows = []
    for path in instance_paths:
        inst = read_instance(str(path))
        for method in methods:
            for seed in seeds:
                try:
                    row = _run_single(
                        instance=inst,
                        method=method,
                        seed=seed,
                        llm_client_name=llm_client_name,
                    )
                    rows.append(row)
                except Exception as e:
                    logger.error(f"Failed {method} on {inst.name}: {e}")
                    rows.append({
                        "instance": inst.name,
                        "method": method,
                        "seed": seed,
                        "vehicles": None,
                        "distance": None,
                        "feasible": False,
                        "runtime": None,
                        "error": str(e),
                    })
    
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "batch_results.csv", index=False)
    logger.info(f"Batch results saved to {output_dir / 'batch_results.csv'}")


def _run_single(
    instance: VRPTWInstance,
    method: str,
    seed: int,
    llm_client_name: str,
) -> dict:
    """Run a single experiment."""
    from src.pipelines.run_baseline import run_baseline
    
    start = time.perf_counter()
    
    if method == "llm_loop":
        from src.pipelines.run_llm_solver import run_llm_solver
        from src.llm.client_factory import create_llm_client
        
        client = create_llm_client(llm_client_name)
        _, final, _, result = run_llm_solver(
            instance,
            llm_client=client,
            seed=seed,
        )
        elapsed = time.perf_counter() - start
        return {
            "instance": instance.name,
            "method": method,
            "seed": seed,
            "vehicles": result.vehicles_used,
            "distance": result.total_distance,
            "feasible": result.feasible,
            "runtime": elapsed,
        }
    else:
        solver = "ortools" if method == "ortools" else method
        _, result = run_baseline(
            instance=instance,
            solver=solver,
            time_limit=30,
            seed=seed,
        )
        elapsed = time.perf_counter() - start
        return {
            "instance": instance.name,
            "method": method,
            "seed": seed,
            "vehicles": result.vehicles_used,
            "distance": result.total_distance,
            "feasible": result.feasible,
            "runtime": elapsed,
        }
