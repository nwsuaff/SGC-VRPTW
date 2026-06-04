"""Solomon benchmark loader.

Loads Solomon 100 VRPTW instances. Uses vrplib as the primary parser,
with a fallback manual parser if needed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from src.domain.schema import VRPTWInstance


# Family patterns extracted from Solomon instance naming conventions.
# C = clustered, R = random, RC = mixed.
# 1 = tight time windows, 2 = loose time windows.
_NAME_TO_FAMILY = {
    "C1": "C1", "C2": "C2",
    "R1": "R1", "R2": "R2",
    "RC1": "RC1", "RC2": "RC2",
}


def _derive_family(name: str) -> str | None:
    """Extract family label from instance name like 'C101', 'RC208'."""
    if not name:
        return None
    prefix_match = re.match(r"^([A-Z]+)", name.upper())
    if not prefix_match:
        return None
    prefix = prefix_match.group(1)
    if len(name) > len(prefix):
        suffix = name[len(prefix):]
        if suffix.startswith("1"):
            return f"{prefix}1"
        if suffix.startswith("2"):
            return f"{prefix}2"
    return f"{prefix}1"  # default to type 1


def load_solomon_instance(path: str | Path) -> VRPTWInstance:
    """Load a Solomon 100 VRPTW instance from a text file.

    Args:
        path: Path to the .vrp Solomon benchmark file.

    Returns:
        VRPTWInstance in our unified format.
    """
    path = Path(path)
    name = path.stem
    
    # Always use manual parser since vrplib doesn't handle VEHICLES plural
    return _manual_parse_solomon(path, name)


def _convert_vrplib_solomon(raw: dict, name: str) -> VRPTWInstance:
    """Convert a vrplib-parsed Solomon dict to our VRPTWInstance."""
    # Solomon uses node 1 as depot, we use node 0
    depot_x = float(raw["depot"][0])
    depot_y = float(raw["depot"][1])
    
    # Remap coordinates: node 1 (depot) -> 0, node N -> N-1
    x_coords = {0: depot_x}
    y_coords = {0: depot_y}
    demand = {0: 0}
    service_time = {0: 0}
    ready_time = {0: raw.get("time_window", [(0, 999999)])[0][0]}
    due_time = {0: raw.get("time_window", [(0, 999999)])[0][1]}
    customer_ids = []

    if "node_coord" in raw:
        for node in raw["node_coord"]:
            nid = int(node[0])
            if nid == 1:
                continue  # depot, already handled
            new_id = nid - 1  # remap: 2 -> 1, 3 -> 2, etc.
            x_coords[new_id] = float(node[1])
            y_coords[new_id] = float(node[2])
            customer_ids.append(new_id)

    if "demand" in raw:
        for nid, d in enumerate(raw["demand"]):
            if nid == 0:
                continue
            new_id = nid - 1
            demand[new_id] = int(d)

    if "service_time" in raw:
        for nid, st in enumerate(raw["service_time"]):
            if nid == 0:
                continue
            new_id = nid - 1
            service_time[new_id] = int(st)

    if "time_window" in raw:
        for nid, tw in enumerate(raw["time_window"]):
            if nid == 0:
                continue
            new_id = nid - 1
            ready_time[new_id] = int(tw[0])
            due_time[new_id] = int(tw[1])

    capacity = int(raw.get("capacity", 200))

    return VRPTWInstance(
        name=name,
        source="solomon",
        family=_derive_family(name),
        size=len(customer_ids),
        depot_id=0,
        customer_ids=sorted(customer_ids),
        x_coords=x_coords,
        y_coords=y_coords,
        demand=demand,
        service_time=service_time,
        ready_time=ready_time,
        due_time=due_time,
        vehicle_capacity=capacity,
        vehicle_count=raw.get("num_vehicles"),
        objective="lexicographic_vehicles_distance",
    )


def _manual_parse_solomon(path: Path, name: str) -> VRPTWInstance:
    """Manually parse a Solomon .vrp file.

    Solomon format (C101.vrp example):
    - NAME : C101
    - TYPE : CVRPTW  
    - DIMENSION : 101
    - VEHICLES : 25
    - CAPACITY : 200
    - SERVICE_TIME : 90
    - NODE_COORD_SECTION
      1 40 50  (node 1 = depot!)
      2 45 68
      ...
    - DEMAND_SECTION
    - TIME_WINDOW_SECTION
    """
    content = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()]

    phase = "header"
    capacity = 200
    num_vehicles: int | None = None
    service_time_global = 0
    depot_x = 0.0
    depot_y = 0.0
    x_coords: dict[int, float] = {}
    y_coords: dict[int, float] = {}
    demand: dict[int, int] = {}
    ready_time: dict[int, int] = {}
    due_time: dict[int, int] = {}
    customer_ids: list[int] = []

    for line in lines:
        upper = line.upper()
        
        # Parse header fields
        if "CAPACITY" in upper:
            parts = re.split(r":\s*", line)
            for p in parts:
                if p.isdigit():
                    capacity = int(p)
                    break
        elif "VEHICLES" in upper:
            parts = re.split(r":\s*", line)
            for p in parts:
                if p.isdigit():
                    num_vehicles = int(p)
                    break
        elif "SERVICE_TIME" in upper:
            parts = re.split(r":\s*", line)
            for p in parts:
                if p.isdigit():
                    service_time_global = int(p)
                    break
        
        # Parse sections
        elif "NODE_COORD_SECTION" in upper:
            phase = "coords"
        elif "DEMAND_SECTION" in upper:
            phase = "demand"
        elif "TIME_WINDOW_SECTION" in upper:
            phase = "time_window"
        elif "DEPOT_SECTION" in upper:
            phase = "depot"
        
        # Parse coordinates
        elif phase == "coords":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3 and parts[0].isdigit():
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                # Remap: depot (node 1) -> 0, customers shift down by 1
                if nid == 1:
                    x_coords[0] = x
                    y_coords[0] = y
                    depot_x, depot_y = x, y
                else:
                    new_id = nid - 1
                    x_coords[new_id] = x
                    y_coords[new_id] = y
                    customer_ids.append(new_id)
        
        # Parse demand
        elif phase == "demand":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 2 and parts[0].isdigit():
                nid = int(parts[0])
                d = int(parts[1])
                if nid == 1:
                    demand[0] = 0
                else:
                    demand[nid - 1] = d
        
        # Parse time windows
        elif phase == "time_window":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3 and parts[0].isdigit():
                nid = int(parts[0])
                ready = int(parts[1])
                due = int(parts[2])
                if nid == 1:
                    ready_time[0] = ready
                    due_time[0] = due
                else:
                    ready_time[nid - 1] = ready
                    due_time[nid - 1] = due
        
        elif phase == "depot" and line.strip() == "1":
            phase = "depot_wait"

    # Fill defaults
    demand.setdefault(0, 0)
    ready_time.setdefault(0, 0)
    due_time.setdefault(0, 999999)

    return VRPTWInstance(
        name=name,
        source="solomon",
        family=_derive_family(name),
        size=len(customer_ids),
        depot_id=0,
        customer_ids=sorted(customer_ids),
        x_coords=x_coords,
        y_coords=y_coords,
        demand=demand,
        service_time={k: service_time_global for k in demand.keys()},
        ready_time=ready_time,
        due_time=due_time,
        vehicle_capacity=capacity,
        vehicle_count=num_vehicles,
        objective="lexicographic_vehicles_distance",
    )
