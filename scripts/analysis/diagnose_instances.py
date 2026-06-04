"""Quick diagnostic: check which instances are solvable within time limits."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.solomon_loader import load_solomon_instance


def check_tw_widths(inst):
    widths = {}
    for cid in inst.customer_ids:
        r = inst.ready_time.get(cid, 0)
        d = inst.due_time.get(cid, 999999)
        widths[cid] = d - r
    return widths


def quick_pyvrp(inst, limit=5.0, seed=42):
    from src.solvers.pyvrp_solver import solve_pyvrp
    from src.solvers.feasibility_checker import check_feasibility
    t0 = time.perf_counter()
    try:
        sol, _ = solve_pyvrp(inst, time_limit=limit, seed=seed)
        elapsed = time.perf_counter() - t0
        rep = check_feasibility(inst, sol)
        return sol, rep, elapsed, None
    except Exception as e:
        return None, None, time.perf_counter() - t0, str(e)


def quick_ortools(inst, limit=10.0):
    from src.solvers.ortools_solver import solve_ortools
    from src.solvers.feasibility_checker import check_feasibility
    t0 = time.perf_counter()
    try:
        sol, meta = solve_ortools(inst, time_limit=limit)
        elapsed = time.perf_counter() - t0
        rep = check_feasibility(inst, sol)
        return sol, rep, elapsed, meta.get("ortools_status")
    except Exception as e:
        return None, None, time.perf_counter() - t0, str(e)


def main():
    root = Path(__file__).parent / "data" / "VRPTW" / "Solomon"

    print(f"\n{'─'*80}")
    print(f"  Solomon-100 Instance Diagnostic (PyVRP 5s, OR-Tools 10s)")
    print(f"{'─'*80}")
    print(f"  {'Instance':<8} {'Type':<4} {'TW-width (avg/min/max)':<22} "
          f"{'PyVRP(5s)':<20} {'OR-Tools(10s)':<20}")
    print(f"{'─'*80}")

    for fname in sorted(root.glob("*.vrp")):
        name = fname.stem
        inst = load_solomon_instance(str(fname))
        inst.name = name
        tw_widths = check_tw_widths(inst)
        if not tw_widths:
            continue
        avg_tw = sum(tw_widths.values()) / len(tw_widths)
        min_tw = min(tw_widths.values())
        max_tw = max(tw_widths.values())

        family = name[:2]
        inst_type = name[2] if len(name) > 2 else "?"

        sol_p, rep_p, t_p, err_p = quick_pyvrp(inst, limit=5.0)
        sol_o, rep_o, t_o, err_o = quick_ortools(inst, limit=10.0)

        p_str = f"{rep_p.feasible} v={sol_p.vehicles_used} d={sol_p.total_distance:.0f} ({t_p:.1f}s)" if rep_p else f"ERROR: {err_p}"
        o_str = f"{rep_o.feasible} v={sol_o.vehicles_used} d={sol_o.total_distance:.0f} ({t_o:.1f}s)" if rep_o else f"FAIL / {err_o}"

        flag = "  ← recommended" if (rep_p and rep_p.feasible and min_tw >= 50) or (rep_o and rep_o.feasible and min_tw >= 50) else ""
        print(f"  {name:<8} {family}/{inst_type:<3} avg={avg_tw:.0f} min={min_tw} max={max_tw:<5} "
              f"{p_str:<20} {o_str:<20}{flag}")

    print(f"{'─'*80}")


if __name__ == "__main__":
    main()
