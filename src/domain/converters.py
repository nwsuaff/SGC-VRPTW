"""Data format converters for VRPTW instances."""

from typing import Dict, Any

from src.domain.schema import VRPTWInstance, RouteSolution


def instance_to_dict(instance: VRPTWInstance) -> Dict[str, Any]:
    """Convert VRPTWInstance to dictionary for serialization."""
    return {
        "name": instance.name,
        "source": instance.source,
        "family": instance.family,
        "size": instance.size,
        "depot_id": instance.depot_id,
        "customer_ids": instance.customer_ids,
        "x_coords": instance.x_coords,
        "y_coords": instance.y_coords,
        "demand": instance.demand,
        "service_time": instance.service_time,
        "ready_time": instance.ready_time,
        "due_time": instance.due_time,
        "vehicle_capacity": instance.vehicle_capacity,
        "vehicle_count": instance.vehicle_count,
        "distance_matrix": instance.distance_matrix,
        "duration_matrix": instance.duration_matrix,
        "objective": instance.objective,
        "metadata": instance.metadata,
    }


def dict_to_instance(data: Dict[str, Any]) -> VRPTWInstance:
    """Convert dictionary to VRPTWInstance."""
    return VRPTWInstance(**data)


def solution_to_dict(solution: RouteSolution) -> Dict[str, Any]:
    """Convert RouteSolution to dictionary for serialization."""
    return solution.model_dump()


def dict_to_solution(data: Dict[str, Any]) -> RouteSolution:
    """Convert dictionary to RouteSolution."""
    return RouteSolution(**data)
