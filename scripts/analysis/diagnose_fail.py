import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import read_instance
from src.solvers.feasibility_checker import check_feasibility, _check_single_route

# Check both C1_8_8 and C1_8_9
for name in ["C1_8_8", "C1_8_9"]:
    vrp_files = list(Path("data/VRPTW").glob(f"**/{name}.vrp"))
    instance = read_instance(str(vrp_files[0]))

    print(f"\n{'='*60}")
    print(f"{name}:")
    print(f"  Customers: {len(instance.customer_ids)}")
    print(f"  Vehicle capacity: {instance.vehicle_capacity}")
    print(f"  Depot: {instance.depot_id}")

    # Check sample time windows and demands
    print(f"  Sample customer data (first 3 customers):")
    for cid in instance.customer_ids[:3]:
        ready = instance.ready_time.get(cid, 0)
        due = instance.due_time.get(cid, 999999)
        demand = instance.demand.get(cid, 0)
        service = instance.service_time.get(cid, 0)
        x = instance.x_coords.get(cid, 0)
        y = instance.y_coords.get(cid, 0)
        print(f"    Customer {cid}: ({x:.1f},{y:.1f}), demand={demand}, window=[{ready},{due}], service={service}")

    # Get baseline solution from CSV
    import csv
    with open("results/test_warmstart/comparison_results.csv") as f:
        for row in csv.DictReader(f):
            if row["instance"] == name:
                print(f"\n  CSV data:")
                print(f"    Baseline: {row['baseline_vehicles']}v/{row['baseline_distance']}d, feas={row['baseline_feasible']}")
                print(f"    DRoC:     {row['droc_vehicles']}v/{row['droc_distance']}d, feas={row['droc_feasible']}")
                break

    # Check feasibility of a trivially empty solution
    from src.domain.schema import RouteSolution
    empty_soln = RouteSolution(routes=[], vehicles_used=0, total_distance=0.0, total_duration=0.0, feasible=False)
    empty_report = check_feasibility(instance, empty_soln)
    print(f"\n  Empty solution validation: feasible={empty_report.feasible}")

    # Check a single-depot single-customer route
    single_route = RouteSolution(
        routes=[[1]],  # route with just customer 1
        vehicles_used=1,
        total_distance=instance.dist(instance.depot_id, 1) + instance.dist(1, instance.depot_id),
        total_duration=0.0,
        feasible=False
    )
    single_report = check_feasibility(instance, single_route)
    print(f"  Single-customer route validation: feasible={single_report.feasible}")
    if single_report.route_reports:
        rr = single_report.route_reports[0]
        print(f"    load={rr.load}, time_violations={rr.time_violations}")
        print(f"    arrival_times={rr.arrival_times}")
        print(f"    waiting_times={rr.waiting_times}")

    # Check distance matrix consistency
    dm = instance.distance_matrix
    if dm:
        print(f"\n  Distance matrix check (first 3x3):")
        for i in range(min(3, len(dm))):
            row = [f"{dm[i][j]:.1f}" for j in range(min(3, len(dm[i])))]
            print(f"    {row}")
        # Check if distance == travel_duration
        d01 = dm[0][1]
        td01 = instance.travel_duration(0, 1)
        print(f"  dist(0,1)={d01:.2f}, travel_duration(0,1)={td01:.2f}")
