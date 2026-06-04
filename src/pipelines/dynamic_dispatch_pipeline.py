"""Dynamic dispatch pipeline for ORTEC-style order-arrival scenarios.

This pipeline orchestrates epoch-by-epoch DRoC solving for dynamic VRPTW.
At each epoch:
1. New orders arrive (released from the static base)
2. DRoC receives the set of open orders + previous solution as warm-start
3. DRoC generates code to solve the current epoch's VRPTW sub-instance
4. The solution is validated and the next epoch begins

The pipeline produces per-epoch results tracking:
- Served order rate (cumulative)
- Driving duration
- Feasibility
- Runtime
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.domain.dynamic_schema import (
    DynamicSolverResult,
    DynamicVRPTWInstance,
    EpochResult,
)
from src.domain.schema import RouteSolution, VRPTWInstance
from src.llm.code_generator import CodeGenerator
from src.solvers.dynamic_ortools_solver import (
    DynamicORtoolsConfig,
    solve_dynamic_scenario,
)
from src.utils.io import read_instance, write_solution

logger = logging.getLogger(__name__)


class DynamicDispatchPipeline:
    """Epoch-by-epoch dynamic dispatch pipeline using DRoC.

    Supports three solver strategies:
    - "droc": DRoC with self-debugging LLM code generation
    - "ortools": OR-Tools with warm-start from previous epoch
    - "greedy": Greedy insertion baseline
    """

    def __init__(
        self,
        solver: str = "droc",
        llm_client: Any | None = None,
        time_limit_per_epoch: float = 60.0,
        droc_max_iterations: int = 4,
        droc_timeout: float = 180.0,
        enable_self_debug: bool = True,
        enable_rag: bool = True,
        output_dir: str | Path | None = None,
    ):
        self.solver = solver
        self.llm_client = llm_client
        self.time_limit_per_epoch = time_limit_per_epoch
        self.droc_max_iterations = droc_max_iterations
        self.droc_timeout = droc_timeout
        self.enable_self_debug = enable_self_debug
        self.enable_rag = enable_rag
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ortools_config = DynamicORtoolsConfig(
            time_limit_per_epoch=time_limit_per_epoch,
            time_limit_first_epoch=time_limit_per_epoch * 2,
            warm_start_enabled=True,
        )

    def run(
        self,
        scenario: DynamicVRPTWInstance,
        base_instance: VRPTWInstance,
    ) -> DynamicSolverResult:
        """Run the full dynamic dispatch scenario.

        Args:
            scenario: The dynamic scenario with orders by epoch.
            base_instance: The static VRPTWInstance (used for coordinates etc.)

        Returns:
            DynamicSolverResult with per-epoch metrics.
        """
        if self.solver == "ortools":
            return self._run_ortools(scenario, base_instance)
        elif self.solver == "droc":
            return self._run_droc(scenario, base_instance)
        elif self.solver == "greedy":
            return self._run_greedy(scenario, base_instance)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def _run_ortools(
        self,
        scenario: DynamicVRPTWInstance,
        base_instance: VRPTWInstance,
    ) -> DynamicSolverResult:
        """Use OR-Tools with warm-start for all epochs."""
        return solve_dynamic_scenario(
            scenario=scenario,
            base=base_instance,
            config=self.ortools_config,
        )

    def _run_droc(
        self,
        scenario: DynamicVRPTWInstance,
        base_instance: VRPTWInstance,
    ) -> DynamicSolverResult:
        """Use DRoC (LLM code generation + execution) per epoch."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        epoch_states: list[dict] = []
        warm_start: list[list[int]] | None = None
        total_runtime = 0.0
        total_late = 0
        total_capacity = 0
        total_distance = 0.0
        total_driving = 0.0
        dispatched_ids: set[int] = set()

        logger.info(
            f"[DynamicDRoC] Starting {scenario.name} "
            f"({scenario.num_orders_total} orders, {scenario.num_epochs} epochs)"
        )

        for epoch in range(scenario.num_epochs):
            open_orders = scenario.get_open_orders(epoch)
            new_orders = scenario.get_orders_by_epoch(epoch)
            total_orders = scenario.num_orders_total

            logger.info(
                f"[DynamicDRoC] Epoch {epoch}: "
                f"{len(new_orders)} new, {len(open_orders)} total open, "
                f"must_go={sum(1 for o in open_orders if o.must_dispatch)}"
            )

            if not open_orders:
                result = EpochResult(
                    epoch=epoch,
                    routes=[],
                    orders_dispatched=0,
                    orders_pending=0,
                    served_rate=1.0 if epoch == 0 else (
                        len(dispatched_ids) / total_orders if total_orders > 0 else 1.0
                    ),
                    total_distance=0.0,
                    driving_duration=0.0,
                    driving_duration_cumulative=total_driving,
                    feasible=True,
                    runtime_sec=0.0,
                )
                epoch_states.append(result.model_dump())
                continue

            epoch_result = self._solve_epoch_with_droc(
                open_orders=open_orders,
                base_instance=base_instance,
                scenario=scenario,
                epoch=epoch,
                warm_start=warm_start,
            )

            if epoch_result.routes:
                warm_start = epoch_result.routes
            total_runtime += epoch_result.runtime_sec
            total_late += epoch_result.late_violations
            total_capacity += epoch_result.capacity_violations
            total_distance += epoch_result.total_distance
            total_driving += epoch_result.driving_duration

            for route in epoch_result.routes:
                for req_id in route:
                    if req_id != 0:
                        dispatched_ids.add(req_id)

            served = len(dispatched_ids)
            served_rate = served / total_orders if total_orders > 0 else 1.0

            final_result = EpochResult(
                epoch=epoch,
                routes=epoch_result.routes,
                orders_dispatched=epoch_result.orders_dispatched,
                orders_pending=epoch_result.orders_pending,
                served_rate=served_rate,
                total_distance=epoch_result.total_distance,
                driving_duration=epoch_result.driving_duration,
                driving_duration_cumulative=total_driving,
                feasible=epoch_result.feasible,
                runtime_sec=epoch_result.runtime_sec,
                late_violations=epoch_result.late_violations,
                capacity_violations=epoch_result.capacity_violations,
            )
            epoch_states.append(final_result.model_dump())

            if self.output_dir:
                epoch_dir = self.output_dir / f"epoch_{epoch}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                (epoch_dir / "scenario.json").write_text(
                    json.dumps(scenario.model_dump(), indent=2)
                )

        served = len(dispatched_ids)
        served_rate = served / total_orders if total_orders > 0 else 1.0

        result = DynamicSolverResult(
            scenario_name=scenario.name,
            base_instance_name=base_instance.name,
            num_epochs=scenario.num_epochs,
            epoch_results=[EpochResult(**e) for e in epoch_states],
            total_distance=total_distance,
            total_driving_duration=total_driving,
            overall_served_rate=served_rate,
            overall_feasible=all(e.get("feasible", True) for e in epoch_states),
            total_late_violations=total_late,
            total_capacity_violations=total_capacity,
            total_runtime_sec=total_runtime,
        )

        logger.info(
            f"[DynamicDRoC] Complete: {served}/{total_orders} orders served "
            f"({served_rate:.1%}), feasible={result.overall_feasible}, "
            f"runtime={total_runtime:.1f}s"
        )

        if self.output_dir:
            result_path = self.output_dir / "result.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

        return result

    def _solve_epoch_with_droc(
        self,
        open_orders,
        base_instance: VRPTWInstance,
        scenario: DynamicVRPTWInstance,
        epoch: int,
        warm_start: list[list[int]] | None,
    ) -> EpochResult:
        """Solve one epoch using DRoC (LLM code generation + execution)."""
        from src.llm.code_generator import CodeGenResult

        epoch_inst = self._build_epoch_vrptw(open_orders, base_instance, scenario, epoch)
        if epoch_inst.size == 0:
            return EpochResult(
                epoch=epoch, routes=[], orders_dispatched=0,
                orders_pending=0, served_rate=0.0,
                total_distance=0.0, driving_duration=0.0,
                driving_duration_cumulative=0.0, feasible=True,
            )

        generator = CodeGenerator(
            llm_client=self.llm_client,
            max_iterations=self.droc_max_iterations,
            enable_self_debug=self.enable_self_debug,
            enable_rag=self.enable_rag,
            exec_timeout=min(self.droc_timeout * 0.25, 250.0),
            llm_timeout=min(self.droc_timeout * 0.5, 180.0),
            outer_timeout=self.droc_timeout * 0.95,
        )

        # Build incumbent from previous epoch's routes if available
        from src.domain.schema import RouteSolution
        incumbent: RouteSolution | None = None
        if warm_start:
            total_v = sum(len(r) for r in warm_start)
            total_d = 0.0  # unknown for epoch-level warm-start
            incumbent = RouteSolution(
                routes=warm_start,
                vehicles_used=len(warm_start),
                total_distance=total_d,
                total_duration=0.0,
                feasible=True,
                late_violations=0,
                capacity_violations=0,
                runtime_sec=0.0,
            )

        start_time = time.time()
        gen_result: CodeGenResult = generator.generate(
            instance=epoch_inst,
            incumbent=incumbent,
        )
        elapsed = time.time() - start_time

        from src.pipelines.droc_pipeline import _execution_to_solution

        if gen_result.execution_result is not None:
            solution = _execution_to_solution(
                gen_result.execution_result, epoch_inst, elapsed
            )
        else:
            solution = None
            logger.warning(f"[DynamicDRoC] Epoch {epoch}: No solution from DRoC")
            return EpochResult(
                epoch=epoch, routes=[],
                orders_dispatched=0, orders_pending=len(open_orders),
                served_rate=0.0, total_distance=0.0,
                driving_duration=0.0, driving_duration_cumulative=0.0,
                feasible=False, runtime_sec=elapsed,
                late_violations=0, capacity_violations=0,
            )

        return EpochResult(
            epoch=epoch,
            routes=solution.routes,
            orders_dispatched=solution.route_count(),
            orders_pending=epoch_inst.size - solution.route_count(),
            served_rate=0.0,
            total_distance=solution.total_distance,
            driving_duration=solution.total_duration,
            driving_duration_cumulative=0.0,
            feasible=solution.feasible,
            runtime_sec=elapsed,
            late_violations=solution.late_violations,
            capacity_violations=solution.capacity_violations,
        )

    def _build_epoch_vrptw(
        self,
        open_orders,
        base_instance: VRPTWInstance,
        scenario: DynamicVRPTWInstance,
        epoch: int,
    ) -> VRPTWInstance:
        """Build a VRPTWInstance from open orders for the current epoch."""
        from src.domain.dynamic_schema import DynamicOrder

        if not open_orders:
            return VRPTWInstance(
                name=f"{base_instance.name}_epoch{epoch}",
                source="ortec_dynamic",
                family="ORTEC",
                size=0,
                depot_id=0,
                customer_ids=[],
                x_coords={0: 0.0},
                y_coords={0: 0.0},
                demand={0: 0},
                service_time={0: 0},
                ready_time={0: 0},
                due_time={0: 999999},
                vehicle_capacity=scenario.vehicle_capacity,
                vehicle_count=scenario.vehicle_count,
            )

        x_coords: dict[int, float] = {}
        y_coords: dict[int, float] = {}
        demand: dict[int, int] = {}
        service_time: dict[int, int] = {}
        ready_time: dict[int, int] = {}
        due_time: dict[int, int] = {}
        customer_ids: list[int] = []

        x_coords[0] = 0.0
        y_coords[0] = 0.0
        demand[0] = 0
        service_time[0] = 0
        ready_time[0] = 0
        due_time[0] = 999999

        for i, order in enumerate(open_orders):
            nid = i + 1
            customer_ids.append(nid)
            x_coords[nid] = order.x
            y_coords[nid] = order.y
            demand[nid] = order.demand
            service_time[nid] = order.service_time
            ready_time[nid] = order.ready_time
            due_time[nid] = order.due_time

        duration_matrix: list[list[float]] | None = None
        if scenario.duration_matrix is not None:
            base_dm = scenario.duration_matrix
            n = len(open_orders) + 1
            duration_matrix = [[0.0] * n for _ in range(n)]
            for new_i in range(n):
                for new_j in range(n):
                    if new_i == 0 or new_j == 0:
                        continue
                    oi = open_orders[new_i - 1]
                    oj = open_orders[new_j - 1]
                    if oi.customer_idx < len(base_dm) and oj.customer_idx < len(base_dm):
                        duration_matrix[new_i][new_j] = base_dm[oi.customer_idx][oj.customer_idx]

        return VRPTWInstance(
            name=f"{base_instance.name}_epoch{epoch}",
            source="ortec_dynamic",
            family="ORTEC",
            size=len(open_orders),
            depot_id=0,
            customer_ids=customer_ids,
            x_coords=x_coords,
            y_coords=y_coords,
            demand=demand,
            service_time=service_time,
            ready_time=ready_time,
            due_time=due_time,
            vehicle_capacity=scenario.vehicle_capacity,
            vehicle_count=scenario.vehicle_count,
            duration_matrix=duration_matrix,
            objective="distance",
            metadata={"epoch": epoch},
        )

    def _run_greedy(
        self,
        scenario: DynamicVRPTWInstance,
        base_instance: VRPTWInstance,
    ) -> DynamicSolverResult:
        """Greedy baseline: dispatch all orders in each epoch."""
        return solve_dynamic_scenario(
            scenario=scenario,
            base=base_instance,
            config=DynamicORtoolsConfig(
                time_limit_per_epoch=self.time_limit_per_epoch,
                time_limit_first_epoch=self.time_limit_per_epoch,
                warm_start_enabled=False,
            ),
        )


def run_dynamic_comparison(
    scenario: DynamicVRPTWInstance,
    base_instance: VRPTWInstance,
    methods: list[str] | None = None,
    llm_client=None,
    time_limit_per_epoch: float = 60.0,
    droc_max_iterations: int = 4,
    droc_timeout: float = 180.0,
    output_dir: str | Path | None = None,
) -> dict[str, DynamicSolverResult]:
    """Run dynamic scenario with multiple methods and compare results.

    Args:
        scenario: The dynamic ORTEC scenario.
        base_instance: Static VRPTWInstance.
        methods: List of solver methods ["droc", "ortools", "greedy"].
        llm_client: LLM client for droc.
        time_limit_per_epoch: Seconds per epoch.
        droc_max_iterations: Max self-debug iterations.
        droc_timeout: Timeout per DRoC generation.
        output_dir: Where to save results.

    Returns:
        Dict mapping method name -> DynamicSolverResult.
    """
    if methods is None:
        methods = ["ortools", "droc"]

    results = {}
    for method in methods:
        logger.info(f"[DynamicComparison] Running method: {method}")
        pipeline = DynamicDispatchPipeline(
            solver=method,
            llm_client=llm_client,
            time_limit_per_epoch=time_limit_per_epoch,
            droc_max_iterations=droc_max_iterations,
            droc_timeout=droc_timeout,
            output_dir=Path(output_dir) / method if output_dir else None,
        )
        result = pipeline.run(scenario=scenario, base_instance=base_instance)
        results[method] = result

    return results


def load_scenario(path: str | Path) -> DynamicVRPTWInstance:
    """Load a dynamic scenario from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return DynamicVRPTWInstance.model_validate(data)
