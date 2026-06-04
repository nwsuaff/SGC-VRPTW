"""Homberger benchmark loader.

Loads Homberger 200-1000 VRPTW instances using the same unified format.
Homberger files use a variant of Solomon format with extended header fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from src.domain.schema import VRPTWInstance


def _derive_family_homberger(name: str) -> str | None:
    """Derive family label from Homberger filename.

    Filename format: {family}{type}_{size}_{number}.txt
    e.g. C1_2_1.txt -> family=C1, size=200
         RC2_8_4.txt -> family=RC2, size=800
    """
    match = re.match(r"^([A-Z]+)([12])_(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    family_prefix = match.group(1).upper()
    tw_type = match.group(2)
    return f"{family_prefix}{tw_type}"


def _derive_size_homberger(name: str) -> int:
    """Derive customer count from Homberger filename.

    Homberger filenames encode size as: {Family}{Type}_{SizeCode}_{Number}.txt
    The SizeCode is 1-10, representing 100-1000 customers.
    E.g. C1_2_1 -> size=2 -> 200 customers
         RC2_8_4 -> size=8 -> 800 customers
    """
    match = re.match(r"^([A-Z]+)([12])_(\d+)", name, re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(3)) * 100


def load_homberger_instance(path: str | Path) -> VRPTWInstance:
    """Load a Homberger VRPTW instance from a text file.

    Homberger format is similar to Solomon but with extended header
    (no VEHICLE section, capacity/vehicle count in comment).
    Node IDs are 1-based in the file (1=depot, 2..n+1=customers).
    We normalize to internal convention: 0=depot, 1..n=customers.

    Args:
        path: Path to the Homberger .vrp benchmark file.

    Returns:
        VRPTWInstance in our unified format.
    """
    import math

    path = Path(path)
    name = path.stem

    content = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()]

    capacity = 200
    num_vehicles: int | None = None
    depot_x = 0.0
    depot_y = 0.0
    raw_x: dict[int, float] = {}
    raw_y: dict[int, float] = {}
    raw_demand: dict[int, int] = {}
    raw_service: dict[int, int] = {}
    raw_ready: dict[int, int] = {}
    raw_due: dict[int, int] = {}
    phase = "header"
    raw_customer_ids: list[int] = []

    for line in lines:
        upper = line.upper()
        if "CAPACITY" in upper:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    capacity = int(p)
                    break
        elif "VEHICLES" in upper:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    num_vehicles = int(p)
                    break
        elif "NODE_COORD_SECTION" in upper:
            phase = "coords"
        elif "DEMAND_SECTION" in upper:
            phase = "demand"
        elif "TIME_WINDOW_SECTION" in upper:
            phase = "time_window"
        elif "DEPOT_SECTION" in upper:
            phase = "depot"
        elif phase == "coords" and "-1" not in upper:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    raw_x[nid] = x
                    raw_y[nid] = y
                    if nid == 1:
                        depot_x, depot_y = x, y
                except ValueError:
                    pass
        elif phase == "demand" and "-1" not in upper:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 2:
                try:
                    nid = int(parts[0])
                    raw_demand[nid] = int(parts[1])
                    if len(parts) >= 3:
                        raw_service[nid] = int(parts[2])
                    else:
                        raw_service.setdefault(nid, 0)
                except ValueError:
                    pass
        elif phase == "time_window" and "-1" not in upper:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    raw_ready[nid] = int(parts[1])
                    raw_due[nid] = int(parts[2])
                except ValueError:
                    pass

    if not raw_customer_ids:
        raw_customer_ids = sorted(set(raw_x.keys()) - {1})

    num_customers = len(raw_customer_ids)
    num_nodes = num_customers + 1

    # Normalize: depot 1 -> 0, customer k -> k-1
    x_coords: dict[int, float] = {0: depot_x}
    y_coords: dict[int, float] = {0: depot_y}
    demand: dict[int, int] = {0: 0}
    service_time: dict[int, int] = {0: 0}
    ready_time: dict[int, int] = {0: 0}
    due_time: dict[int, int] = {0: 999999}
    customer_ids: list[int] = []

    for raw_id in sorted(raw_customer_ids):
        internal_id = raw_id - 1
        customer_ids.append(internal_id)
        x_coords[internal_id] = raw_x[raw_id]
        y_coords[internal_id] = raw_y[raw_id]
        demand[internal_id] = raw_demand.get(raw_id, 0)
        service_time[internal_id] = raw_service.get(raw_id, 0)
        ready_time[internal_id] = raw_ready.get(raw_id, 0)
        due_time[internal_id] = raw_due.get(raw_id, 999999)

    # Precompute distance matrix to avoid coordinate lookup issues
    distance_matrix: list[list[float]] = [
        [0.0 for _ in range(num_nodes)] for _ in range(num_nodes)
    ]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

    return VRPTWInstance(
        name=name,
        source="homberger",
        family=_derive_family_homberger(name),
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
        vehicle_count=num_vehicles,
        distance_matrix=distance_matrix,
        objective="lexicographic_vehicles_distance",
    )
