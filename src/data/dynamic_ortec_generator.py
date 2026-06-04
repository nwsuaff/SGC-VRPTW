"""Synthetic ORTEC static instance generator.

Generates ORTEC-style static VRPTW instances following the same format
as the EURO Meets NeurIPS 2022 competition. Each instance has:
- Pre-computed explicit duration matrix (realistic asymmetric road times)
- Realistic time windows
- Variable customer counts (300, 500, 800)

The generator creates instances by:
1. Placing customers in a 2D region with depot at the center
2. Generating demands from a log-normal distribution
3. Generating time windows based on customer distance from depot
4. Computing an asymmetric duration matrix with road-network-like properties
  (non-Euclidean, triangle-inequality-satisfying, scaled by a speed factor)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from src.data.ortec_loader import _derive_family_ortec
from src.domain.schema import VRPTWInstance


# Default parameters for synthetic instances
DEFAULT_SIZES = [300, 500, 800]
DEFAULT_CAPACITY = 200
DEFAULT_SPEED = 10.0  # units per second
DEFAULT_REGION_SIZE = 5000.0  # 5km x 5km region


def _generate_coords(
    num_customers: int,
    seed: int,
    region_size: float = DEFAULT_REGION_SIZE,
    cluster_centers: int = 3,
) -> list[tuple[float, float]]:
    """Generate customer coordinates with optional clustering."""
    rng = random.Random(seed)
    half = region_size / 2.0

    coords = []

    if cluster_centers > 1:
        cluster_xs = [rng.uniform(-half * 0.8, half * 0.8) for _ in range(cluster_centers)]
        cluster_ys = [rng.uniform(-half * 0.8, half * 0.8) for _ in range(cluster_centers)]
        cluster_sd = region_size / (cluster_centers * 2)

        for i in range(num_customers):
            cx = rng.choice(range(cluster_centers))
            x = rng.gauss(cluster_xs[cx], cluster_sd)
            y = rng.gauss(cluster_ys[cx], cluster_sd)
            x = max(-half, min(half, x))
            y = max(-half, min(half, y))
            coords.append((x, y))
    else:
        for i in range(num_customers):
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
            coords.append((x, y))

    depot_x = 0.0
    depot_y = 0.0
    return [(depot_x, depot_y)] + coords


def _generate_demands(
    num_customers: int,
    seed: int,
    mean: float = 15.0,
    std: float = 8.0,
) -> list[int]:
    """Generate customer demands from a log-normal-ish distribution."""
    rng = random.Random(seed + 1000)
    demands = [0]
    for _ in range(num_customers):
        d = max(1, int(rng.gauss(mean, std)))
        d = max(1, min(d, 50))
        demands.append(d)
    return demands


def _generate_time_windows(
    num_customers: int,
    depot_x: float,
    depot_y: float,
    coords: list[tuple[float, float]],
    seed: int,
    depot_ready: int = 0,
    depot_due: int = 28800,
    service_time_base: int = 600,
) -> list[tuple[int, int]]:
    """Generate time windows based on distance from depot.

    Follows the ORTEC/Homberger pattern:
    - Ready time: proportional to distance from depot + buffer
    - Due time: ready time + time_window_width
    - Time window width decreases with distance (nearby customers need tighter scheduling)
    """
    rng = random.Random(seed + 2000)
    tws = [(depot_ready, depot_due)]

    for i in range(1, num_customers + 1):
        x, y = coords[i]
        dist = math.sqrt((x - depot_x) ** 2 + (y - depot_y) ** 2)
        travel_time = dist / DEFAULT_SPEED

        ready = depot_ready + int(travel_time)
        window_width = rng.randint(3600, 14400)
        due = ready + window_width
        due = min(due, depot_due)
        if due <= ready:
            due = ready + 3600

        tws.append((ready, due))

    return tws


def _generate_service_times(
    num_customers: int,
    seed: int,
    base: int = 600,
    std: int = 300,
) -> list[int]:
    """Generate service times (unloading/loading time at each customer)."""
    rng = random.Random(seed + 3000)
    times = [0]
    for _ in range(num_customers):
        t = int(rng.gauss(base, std))
        times.append(max(60, t))
    return times


def _compute_duration_matrix(
    coords: list[tuple[float, float]],
    seed: int,
    speed: float = DEFAULT_SPEED,
    asymmetry_factor: float = 0.15,
) -> list[list[float]]:
    """Compute an asymmetric duration matrix.

    Uses a gravity-model-like approach:
    - Base duration = Euclidean distance / speed
    - Add asymmetric perturbation to simulate road network effects
    - Ensure triangle inequality is approximately satisfied via a
      "speed reduction" on long direct routes vs. going via intermediate nodes
    """
    n = len(coords)
    rng = random.Random(seed + 4000)

    raw_durations: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                raw_durations[i][j] = 0.0
            else:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                base = math.sqrt(dx * dx + dy * dy) / speed
                perturbation = 1.0 + rng.uniform(-asymmetry_factor, asymmetry_factor)
                raw_durations[i][j] = base * perturbation

    for k in range(n):
        for i in range(n):
            for j in range(n):
                alt = raw_durations[i][k] + raw_durations[k][j] + 600.0
                if alt < raw_durations[i][j]:
                    raw_durations[i][j] = alt

    return raw_durations


def generate_synthetic_ortec_instance(
    num_customers: int,
    seed: int,
    capacity: int = DEFAULT_CAPACITY,
    name_prefix: str = "ORTEC-SYNTH",
    output_dir: Path | str | None = None,
) -> VRPTWInstance:
    """Generate a synthetic ORTEC-style VRPTW instance.

    Args:
        num_customers: Number of customers (excluding depot).
        seed: Random seed for reproducibility.
        capacity: Vehicle capacity.
        name_prefix: Prefix for the instance name.
        output_dir: If provided, save the instance as .vrp file and JSON.

    Returns:
        VRPTWInstance in our unified format.
    """
    rng = random.Random(seed)
    name = f"{name_prefix}-{num_customers}-s{seed}"

    coords = _generate_coords(num_customers, seed)
    depot_x, depot_y = coords[0]
    num_nodes = num_customers + 1

    demands = _generate_demands(num_customers, seed)
    demands = [0] + demands[:num_customers]

    service_times = _generate_service_times(num_customers, seed)
    service_times = [0] + service_times[:num_customers]

    time_windows = _generate_time_windows(
        num_customers, depot_x, depot_y, coords, seed
    )

    duration_matrix = _compute_duration_matrix(coords, seed)

    x_coords: dict[int, float] = {i: coords[i][0] for i in range(num_nodes)}
    y_coords: dict[int, float] = {i: coords[i][1] for i in range(num_nodes)}
    demand: dict[int, int] = {i: demands[i] for i in range(num_nodes)}
    service_time: dict[int, int] = {i: service_times[i] for i in range(num_nodes)}
    ready_time: dict[int, int] = {i: time_windows[i][0] for i in range(num_nodes)}
    due_time: dict[int, int] = {i: time_windows[i][1] for i in range(num_nodes)}
    customer_ids = list(range(1, num_nodes))

    instance = VRPTWInstance(
        name=name,
        source="ortec_synthetic",
        family=_derive_family_ortec(name),
        size=num_customers,
        depot_id=0,
        customer_ids=customer_ids,
        x_coords=x_coords,
        y_coords=y_coords,
        demand=demand,
        service_time=service_time,
        ready_time=ready_time,
        due_time=due_time,
        vehicle_capacity=capacity,
        vehicle_count=None,
        distance_matrix=None,
        duration_matrix=duration_matrix,
        objective="distance",
        metadata={
            "edge_weight_type": "EXPLICIT",
            "asymmetry_factor": 0.15,
            "speed": DEFAULT_SPEED,
            "region_size": DEFAULT_REGION_SIZE,
            "seed": seed,
        },
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vrp_path = output_dir / f"{name}.txt"
        with open(vrp_path, "w", encoding="utf-8") as f:
            f.write(f"NAME : {name}\n")
            f.write(f"COMMENT : Synthetic ORTEC-style instance\n")
            f.write(f"TYPE : VRPTW\n")
            f.write(f"DIMENSION : {num_nodes}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write(f"VEHICLES : 999\n")
            f.write(f"CAPACITY : {capacity}\n\n")

            f.write("EDGE_WEIGHT_SECTION\n")
            for row in duration_matrix:
                f.write("\t".join(str(int(d)) for d in row) + "\n")
            f.write("\n")

            f.write("NODE_COORD_SECTION\n")
            for i in range(num_nodes):
                f.write(f"{i+1}\t{int(x_coords[i])}\t{int(y_coords[i])}\n")
            f.write("\n")

            f.write("DEMAND_SECTION\n")
            for i in range(num_nodes):
                f.write(f"{i+1}\t{demand[i]}\n")
            f.write("\n")

            f.write("DEPOT_SECTION\n")
            f.write("1\n")
            f.write("-1\n\n")

            f.write("SERVICE_TIME_SECTION\n")
            for i in range(num_nodes):
                f.write(f"{i+1}\t{service_time[i]}\n")
            f.write("\n")

            f.write("TIME_WINDOW_SECTION\n")
            for i in range(num_nodes):
                f.write(f"{i+1}\t{ready_time[i]}\t{due_time[i]}\n")
            f.write("\n")

            f.write("EOF\n")

        from src.utils.io import write_instance

        json_path = output_dir / f"{name}.json"
        write_instance(instance, json_path)

        print(f"Generated: {vrp_path} ({num_nodes} nodes, {num_customers} customers)")

    return instance


def generate_ortec_benchmark_suite(
    sizes: list[int] = None,
    instances_per_size: int = 3,
    seeds_per_instance: int = 1,
    output_dir: Path | str = "data/VRPTW/ORTEC/static",
    start_seed: int = 42,
) -> list[VRPTWInstance]:
    """Generate the full synthetic ORTEC benchmark suite.

    Args:
        sizes: List of customer counts. Defaults to [300, 500, 800].
        instances_per_size: Number of instances per size.
        seeds_per_instance: Number of seeds per instance (for multiple variants).
        output_dir: Where to save generated files.
        start_seed: Starting random seed.

    Returns:
        List of generated VRPTWInstance objects.
    """
    if sizes is None:
        sizes = DEFAULT_SIZES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = []
    seed = start_seed

    for size in sizes:
        for inst_idx in range(instances_per_size):
            inst = generate_synthetic_ortec_instance(
                num_customers=size,
                seed=seed,
                output_dir=output_dir,
            )
            instances.append(inst)
            seed += 1

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("name,source,family,size,file\n")
        for inst in instances:
            f.write(f"{inst.name},ortec_synthetic,{inst.family},{inst.size},{inst.name}.txt\n")
    print(f"Manifest written to {manifest_path}")

    return instances
