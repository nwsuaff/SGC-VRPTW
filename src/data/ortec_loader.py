"""ORTEC VRPTW benchmark loader.

Loads ORTEC VRPTW instances from the EURO Meets NeurIPS 2022 competition
(https://euro-neurips-vrp-2022.challenges.ortec.com/).
Uses the same unified format as Solomon/Homberger loaders.

ORTEC format key differences from Solomon/Homberger:
- EDGE_WEIGHT_TYPE = EXPLICIT  (pre-computed real-world road driving times)
- EDGE_WEIGHT_FORMAT = FULL_MATRIX
- SERVICE_TIME_SECTION is mandatory
- DIMENSION includes depot
- The duration matrix is the actual driving time, not Euclidean distance
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from src.domain.schema import VRPTWInstance


def load_ortec_instance(path: str | Path) -> VRPTWInstance:
    """Load an ORTEC VRPTW instance from a .txt VRPLIB-format file.

    Args:
        path: Path to the ORTEC .txt file.

    Returns:
        VRPTWInstance in our unified format.
    """
    path = Path(path)
    return _parse_ortec_file(path)


def _parse_ortec_file(path: Path) -> VRPTWInstance:
    """Parse an ORTEC VRPTW .txt file.

    ORTEC file format (based on ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt):
    - NAME : ORTEC-VRPTW-ASYM-{hash}-d{n_depots}-n{nodes}-k{vehicles}
    - TYPE : VRPTW
    - DIMENSION : N  (includes depot)
    - EDGE_WEIGHT_TYPE : EXPLICIT
    - EDGE_WEIGHT_FORMAT : FULL_MATRIX
    - CAPACITY : Q
    - EDGE_WEIGHT_SECTION
      N lines, each with N integer durations (seconds)
    - NODE_COORD_SECTION
      N lines: "node x y"
    - DEMAND_SECTION
      N lines: "node demand"
    - DEPOT_SECTION
      depot_id (typically 1)
      -1
    - SERVICE_TIME_SECTION
      N lines: "node service_time"
    - TIME_WINDOW_SECTION
      N lines: "node ready due"
    - EOF
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in content.splitlines() if l.strip()]

    name = path.stem
    phase = "header"
    capacity = 200
    num_vehicles: int | None = None
    edge_weight_type = "EUC_2D"
    num_nodes = 0

    raw_x: dict[int, float] = {}
    raw_y: dict[int, float] = {}
    raw_demand: dict[int, int] = {}
    raw_service: dict[int, int] = {}
    raw_ready: dict[int, int] = {}
    raw_due: dict[int, int] = {}
    duration_matrix: list[list[float]] | None = None

    matrix_row = 0

    for line in lines:
        upper = line.upper()

        if "NAME" in upper:
            continue
        elif "COMMENT" in upper:
            continue
        elif "TYPE" in upper:
            continue
        elif "DIMENSION" in upper:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    num_nodes = int(p)
                    break
        elif "EDGE_WEIGHT_TYPE" in upper:
            for p in line.split(" : "):
                if p.strip() not in ("EDGE_WEIGHT_TYPE", ""):
                    edge_weight_type = p.strip()
        elif "VEHICLES" in upper:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    num_vehicles = int(p)
                    break
        elif "CAPACITY" in upper:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    capacity = int(p)
                    break

        elif "EDGE_WEIGHT_SECTION" in upper:
            phase = "edge_weights"
            duration_matrix = [[0.0] * num_nodes for _ in range(num_nodes)]
            matrix_row = 0
        elif "NODE_COORD_SECTION" in upper:
            phase = "coords"
        elif "DEMAND_SECTION" in upper:
            phase = "demand"
        elif "DEPOT_SECTION" in upper:
            phase = "depot"
        elif "SERVICE_TIME_SECTION" in upper:
            phase = "service"
        elif "TIME_WINDOW_SECTION" in upper:
            phase = "time_window"
        elif "EOF" in upper:
            break

        elif phase == "edge_weights":
            vals = [float(x) for x in re.split(r"\s+", line.strip()) if x]
            if matrix_row < num_nodes and len(vals) == num_nodes:
                duration_matrix[matrix_row] = vals
                matrix_row += 1
            elif matrix_row < num_nodes:
                existing = duration_matrix[matrix_row]
                offset = 0
                for v in vals:
                    existing[offset] = v
                    offset += 1
                    if offset >= num_nodes:
                        matrix_row += 1
                        if matrix_row < num_nodes:
                            existing = duration_matrix[matrix_row]
                            offset = 0

        elif phase == "coords":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    raw_x[nid] = x
                    raw_y[nid] = y
                except ValueError:
                    pass

        elif phase == "demand":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 2:
                try:
                    nid = int(parts[0])
                    raw_demand[nid] = int(parts[1])
                except ValueError:
                    pass

        elif phase == "depot":
            if "-1" in line:
                phase = "depot_wait"
            continue

        elif phase == "service":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 2:
                try:
                    nid = int(parts[0])
                    raw_service[nid] = int(parts[1])
                except ValueError:
                    pass

        elif phase == "time_window":
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    raw_ready[nid] = int(parts[1])
                    raw_due[nid] = int(parts[2])
                except ValueError:
                    pass

    if num_nodes == 0:
        raise ValueError(f"Could not parse DIMENSION from ORTEC file: {path}")

    depot_id = 1
    for nid in list(raw_ready.keys()):
        if nid != depot_id:
            continue
        break

    depot_x = raw_x.get(depot_id, 0.0)
    depot_y = raw_y.get(depot_id, 0.0)

    x_coords: dict[int, float] = {0: depot_x}
    y_coords: dict[int, float] = {0: depot_y}
    demand: dict[int, int] = {0: 0}
    service_time: dict[int, int] = {0: 0}
    ready_time: dict[int, int] = {0: 0}
    due_time: dict[int, int] = {0: 999999}
    customer_ids: list[int] = []

    raw_customer_ids = sorted(n for n in raw_x.keys() if n != depot_id)

    for raw_id in raw_customer_ids:
        internal_id = raw_id - 1
        customer_ids.append(internal_id)
        x_coords[internal_id] = raw_x[raw_id]
        y_coords[internal_id] = raw_y[raw_id]
        demand[internal_id] = raw_demand.get(raw_id, 0)
        service_time[internal_id] = raw_service.get(raw_id, 0)
        ready_time[internal_id] = raw_ready.get(raw_id, 0)
        due_time[internal_id] = raw_due.get(raw_id, 999999)

    final_duration_matrix: list[list[float]] | None = None
    if duration_matrix is not None and all(len(row) == num_nodes for row in duration_matrix):
        final_duration_matrix = [[0.0] * (len(customer_ids) + 1) for _ in range(len(customer_ids) + 1)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                ri = i - 1 if i != depot_id else depot_id - 1
                rj = j - 1 if j != depot_id else depot_id - 1
                if 0 <= ri < len(customer_ids) + 1 and 0 <= rj < len(customer_ids) + 1:
                    final_duration_matrix[ri][rj] = duration_matrix[i][j]

    distance_matrix: list[list[float]] | None = None
    if final_duration_matrix is None:
        n = len(customer_ids) + 1
        distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = x_coords.get(i, 0) - x_coords.get(j, 0)
                    dy = y_coords.get(i, 0) - y_coords.get(j, 0)
                    distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

    family = _derive_family_ortec(name)

    return VRPTWInstance(
        name=name,
        source="ortec",
        family=family,
        size=len(customer_ids),
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
        duration_matrix=final_duration_matrix,
        objective="distance",
        metadata={
            "edge_weight_type": edge_weight_type,
            "original_dimension": num_nodes,
        },
    )


def _derive_family_ortec(name: str) -> str | None:
    """Derive family label from ORTEC filename.

    Filename pattern: ORTEC-VRPTW-ASYM-{hash}-{d_ndepots}-n{nodes}-k{vehicles}
    e.g. ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35
    The 'n' component gives the total node count (including depot).
    ORTEC instances are treated as a single family since they don't follow
    Solomon's C/R/RC family convention.
    """
    m = re.search(r"-n(\d+)-", name, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n <= 100:
            return "ORTEC_S"
        elif n <= 300:
            return "ORTEC_M"
        elif n <= 500:
            return "ORTEC_L"
        else:
            return "ORTEC_XL"
    return "ORTEC"


def _derive_size_ortec(name: str) -> int:
    """Derive customer count from ORTEC filename."""
    m = re.search(r"-n(\d+)-", name, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return max(0, n - 1)
    return 0
