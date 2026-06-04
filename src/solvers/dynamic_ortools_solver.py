"""Dynamic OR-Tools solver for ORTEC-style order-arrival scenarios.

Epoch-by-epoch dynamic dispatch using OR-Tools as the VRPTW solver.
At each epoch, we solve for ALL open orders (released by current epoch, not yet dispatched).
Key metrics: served_rate (cumulative % served), driving_duration, feasibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.domain.dynamic_schema import (
    DynamicOrder,
    DynamicSolverResult,
    DynamicVRPTWInstance,
    EpochResult,
)
from src.domain.schema import RouteSolution, VRPTWInstance
from src.solvers.ortools_solver import solve_ortools

logger = logging.getLogger(__name__)


@dataclass
class DynamicORtoolsConfig:
    time_limit_per_epoch: float = 60.0
    time_limit_first_epoch: float = 120.0
    warm_start_enabled: bool = True


def _build_epoch_vrptw(
    open_orders: list[DynamicOrder],
    scenario: DynamicVRPTWInstance,
    epoch: int,
    base_name: str,
) -> tuple[VRPTWInstance, dict[int, DynamicOrder]]:
    """Build VRPTWInstance from open orders.

    open_orders: ALL orders with release_epoch <= epoch that are not dispatched.
    Returns (instance, remap) where remap[node_id] -> DynamicOrder.
    """
    depot = next((o for o in scenario.orders if o.request_id == 0))

    idx_to_order: dict[int, DynamicOrder] = {0: depot}
    x_c, y_c = {0: depot.x}, {0: depot.y}
    demand_c = {0: depot.demand}
    service_c = {0: depot.service_time}
    ready_c = {0: depot.ready_time}
    due_c = {0: depot.due_time}
    customer_ids: list[int] = []

    for i, order in enumerate(open_orders):
        nid = i + 1
        idx_to_order[nid] = order
        x_c[nid] = order.x
        y_c[nid] = order.y
        demand_c[nid] = order.demand
        service_c[nid] = order.service_time
        ready_c[nid] = order.ready_time
        due_c[nid] = order.due_time
        customer_ids.append(nid)

    dur_mat: list[list[float]] | None = None
    if scenario.duration_matrix is not None:
        bdm = scenario.duration_matrix
        n = len(open_orders) + 1
        dur_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ci = idx_to_order[i].customer_idx
                cj = idx_to_order[j].customer_idx
                if 0 <= ci < len(bdm) and 0 <= cj < len(bdm):
                    dur_mat[i][j] = bdm[ci][cj]

    inst = VRPTWInstance(
        name=f"{base_name}_ep{epoch}",
        source="ortec_dynamic",
        family="ORTEC",
        size=len(open_orders),
        depot_id=0,
        customer_ids=customer_ids,
        x_coords=x_c,
        y_coords=y_c,
        demand=demand_c,
        service_time=service_c,
        ready_time=ready_c,
        due_time=due_c,
        vehicle_capacity=scenario.vehicle_capacity,
        vehicle_count=scenario.vehicle_count,
        duration_matrix=dur_mat,
        objective="distance",
        metadata={"epoch": epoch},
    )
    return inst, idx_to_order


def _remap_to_request_ids(
    routes: list[list[int]], remap: dict[int, DynamicOrder]
) -> list[list[int]]:
    """Convert internal node IDs back to original request IDs (excluding depot)."""
    result = []
    for route in routes:
        ids = [remap[n].request_id for n in route if n != 0 and n in remap and remap[n].request_id != 0]
        if ids:
            result.append(ids)
    return result


def solve_dynamic_scenario(
    scenario: DynamicVRPTWInstance,
    base: VRPTWInstance,
    config: Optional[DynamicORtoolsConfig] = None,
) -> DynamicSolverResult:
    """Run full dynamic scenario with OR-Tools per epoch.

    At each epoch:
    1. Collect all open orders (release_epoch <= epoch, not yet dispatched)
    2. Build a VRPTW sub-instance with those orders
    3. Solve with OR-Tools (warm-start from previous epoch)
    4. Update dispatched set and advance to next epoch
    """
    if config is None:
        config = DynamicORtoolsConfig()

    dispatched_ids: set[int] = set()
    epoch_results: list[EpochResult] = []
    total_dist = 0.0
    total_driving = 0.0
    total_late = 0
    total_cap = 0
    total_runtime = 0.0
    warm_start: list[list[int]] | None = None

    logger.info(
        f"[DynamicORtools] {scenario.name}: "
        f"{scenario.num_orders_total} orders, {scenario.num_epochs} epochs"
    )

    for epoch in range(scenario.num_epochs):
        # All open orders = released by this epoch, not yet dispatched
        open_orders = [
            o for o in scenario.orders
            if o.request_id != 0
            and o.release_epoch <= epoch
            and o.request_id not in dispatched_ids
        ]

        tlim = (
            config.time_limit_first_epoch
            if epoch == 0
            else config.time_limit_per_epoch
        )

        logger.info(
            f"[DynamicORtools] Epoch {epoch}: "
            f"{len(open_orders)} open orders, "
            f"must_go={sum(1 for o in open_orders if o.must_dispatch)}"
        )

        if not open_orders:
            served = len(dispatched_ids)
            rate = served / scenario.num_orders_total
            er = EpochResult(
                epoch=epoch, routes=[], orders_dispatched=0,
                orders_pending=0, served_rate=rate,
                total_distance=0.0, driving_duration=0.0,
                driving_duration_cumulative=total_driving,
                feasible=True, runtime_sec=0.0,
            )
            epoch_results.append(er)
            continue

        epoch_inst, remap = _build_epoch_vrptw(
            open_orders, scenario, epoch, base.name
        )

        start = time.time()
        try:
            # NOTE: Warm-start compatibility requires matching node IDs between epochs.
            # Since open_orders change every epoch (different customer sets), we do NOT
            # pass warm_start. Each epoch is solved independently from scratch.
            solution, meta = solve_ortools(
                instance=epoch_inst,
                time_limit=tlim,
                warm_start=None,
            )
        except Exception as e:
            logger.error(f"[DynamicORtools] Epoch {epoch} failed: {e}")
            solution = RouteSolution(
                routes=[], vehicles_used=0,
                total_distance=0.0, total_duration=0.0, feasible=False,
            )
        elapsed = time.time() - start
        total_runtime += elapsed

        # Track newly dispatched orders (track how many NEW ones were dispatched this epoch)
        dispatched_this_epoch = set()
        for route in solution.routes:
            for nid in route:
                if nid in remap:
                    req_id = remap[nid].request_id
                    if req_id not in dispatched_ids:
                        dispatched_this_epoch.add(req_id)
                    dispatched_ids.add(req_id)

        # Remap routes to original request IDs
        original_routes = _remap_to_request_ids(solution.routes, remap)

        total_dist += solution.total_distance
        total_driving += solution.total_distance
        total_late += solution.late_violations
        total_cap += solution.capacity_violations

        # Cumulative served rate
        total_served = len(dispatched_ids)
        served_rate = total_served / scenario.num_orders_total

        # Orders pending = open orders not yet dispatched (cumulative undispatched)
        pending_undispatched = len(open_orders) - len(dispatched_this_epoch)
        # Note: some open_orders may remain undispatched this epoch due to constraints;
        # they will appear in open_orders of subsequent epochs.

        er = EpochResult(
            epoch=epoch,
            routes=original_routes,
            orders_dispatched=len(dispatched_this_epoch),
            orders_pending=pending_undispatched,
            served_rate=served_rate,
            total_distance=solution.total_distance,
            driving_duration=solution.total_duration,
            driving_duration_cumulative=total_driving,
            feasible=solution.feasible,
            runtime_sec=elapsed,
            late_violations=solution.late_violations,
            capacity_violations=solution.capacity_violations,
        )
        epoch_results.append(er)

        logger.info(
            f"[DynamicORtools] Epoch {epoch} done: "
            f"{total_served}/{scenario.num_orders_total} served ({served_rate:.1%}), "
            f"dispatched={len(dispatched_this_epoch)}, feasible={solution.feasible}, runtime={elapsed:.1f}s"
        )

    final_served = len(dispatched_ids)
    final_rate = final_served / scenario.num_orders_total

    result = DynamicSolverResult(
        scenario_name=scenario.name,
        base_instance_name=base.name,
        num_epochs=scenario.num_epochs,
        epoch_results=epoch_results,
        total_distance=total_dist,
        total_driving_duration=total_driving,
        overall_served_rate=final_rate,
        overall_feasible=all(e.feasible for e in epoch_results),
        total_late_violations=total_late,
        total_capacity_violations=total_cap,
        total_runtime_sec=total_runtime,
    )

    logger.info(
        f"[DynamicORtools] Complete: {final_served}/{scenario.num_orders_total} "
        f"served ({final_rate:.1%}), feasible={result.overall_feasible}, "
        f"runtime={total_runtime:.1f}s"
    )

    return result
