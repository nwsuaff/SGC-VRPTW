"""Scene Card Module - Structured Instance Representation for DRoC.

This module implements the Scene Card concept from the paper methodology,
providing a structured representation of VRPTW instances that can be used
to enhance LLM code generation prompts.

The Scene Card captures:
- Instance characteristics (size, capacity, distribution)
- Problem difficulty indicators (time window tightness, capacity utilization)
- Critical customers that require special handling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance


class DistributionType(Enum):
    """Customer spatial distribution type."""
    CLUSTERED = "clustered"
    RANDOM = "random"
    RANDOM_CLUSTERED = "random_clustered"


class TimeWindowTightness(Enum):
    """Time window tightness classification."""
    LOOSE = "loose"      # TW width > 50% of planning horizon
    MEDIUM = "medium"    # TW width 25-50% of planning horizon
    TIGHT = "tight"     # TW width < 25% of planning horizon


@dataclass
class CustomerProfile:
    """Profile of a single customer."""
    customer_id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float
    
    @property
    def tw_width(self) -> float:
        """Time window width."""
        return self.due_time - self.ready_time
    
    @property
    def is_tight_tw(self) -> bool:
        """Whether the time window is tight relative to service time."""
        return self.tw_width < self.service_time * 3


@dataclass
class RouteStatistics:
    """Statistics from an incumbent solution."""
    num_vehicles: int
    total_distance: float
    avg_route_length: float
    max_route_length: float
    min_route_length: float
    avg_load_utilization: float
    route_details: list[list[int]] = field(default_factory=list)


@dataclass
class SceneCard:
    """Structured scene description for LLM prompts.
    
    This class implements the Scene Card concept from the paper,
    providing a comprehensive representation of a VRPTW instance
    that can be used to enhance DRoC-style code generation.
    
    Attributes:
        instance_name: Name of the instance.
        num_customers: Number of customers.
        num_vehicles: Number of available vehicles.
        vehicle_capacity: Vehicle capacity.
        depot_location: (x, y) coordinates of depot.
        distribution_type: Spatial distribution of customers.
        tw_tightness: Overall time window tightness.
        tw_tightness_score: Numeric score (0-1, higher = tighter).
        capacity_utilization: Demand / (vehicles * capacity).
        critical_customers: IDs of customers with tight time windows.
        farthest_customers: IDs of customers farthest from depot.
        nearest_customers: IDs of customers nearest to depot.
        total_demand: Sum of all customer demands.
        avg_tw_width: Average time window width.
        customer_profiles: Detailed profiles of each customer.
        route_stats: Statistics from incumbent solution (if available).
    """
    
    # Instance identification
    instance_name: str
    num_customers: int
    num_vehicles: int
    vehicle_capacity: float
    
    # Location information
    depot_location: tuple[float, float]
    customer_locations: list[tuple[float, float]] = field(default_factory=list)
    
    # Distribution analysis
    distribution_type: DistributionType = DistributionType.RANDOM
    avg_inter_customer_distance: float = 0.0
    
    # Time window analysis
    tw_tightness: TimeWindowTightness = TimeWindowTightness.MEDIUM
    tw_tightness_score: float = 0.5  # 0-1 scale
    avg_tw_width: float = 0.0
    planning_horizon: float = 0.0  # Max due_time in the instance
    
    # Capacity analysis
    capacity_utilization: float = 0.0  # total_demand / (num_vehicles * capacity)
    total_demand: float = 0.0
    
    # Critical customer identification
    critical_customers: list[int] = field(default_factory=list)  # Tight TW customers
    farthest_customers: list[int] = field(default_factory=list)  # Farthest from depot
    nearest_customers: list[int] = field(default_factory=list)   # Nearest to depot
    
    # Detailed profiles
    customer_profiles: list[CustomerProfile] = field(default_factory=list)
    
    # Incumbent solution statistics (optional)
    route_stats: RouteStatistics | None = None
    
    def to_prompt_string(self) -> str:
        """Convert Scene Card to a string suitable for LLM prompts.
        
        Returns:
            Formatted string representation of the scene card.
        """
        lines = [
            f"## Instance: {self.instance_name}",
            "",
            "### Problem Scale",
            f"- Customers: {self.num_customers}",
            f"- Vehicles: {self.num_vehicles} (capacity: {self.vehicle_capacity})",
            f"- Depot location: ({self.depot_location[0]:.1f}, {self.depot_location[1]:.1f})",
            "",
            "### Distribution Characteristics",
            f"- Distribution type: {self.distribution_type.value}",
            f"- Avg inter-customer distance: {self.avg_inter_customer_distance:.1f}",
            "",
            "### Time Window Analysis",
            f"- Tightness level: {self.tw_tightness.value} (score: {self.tw_tightness_score:.2f})",
            f"- Avg time window width: {self.avg_tw_width:.1f}",
            f"- Planning horizon: {self.planning_horizon:.1f}",
        ]
        
        if self.critical_customers:
            lines.extend([
                "",
                "### Critical Customers (tight time windows)",
                f"- Customer IDs: {self.critical_customers[:10]}" +
                (" ..." if len(self.critical_customers) > 10 else ""),
            ])
        
        if self.farthest_customers:
            lines.extend([
                "",
                "### Farthest Customers (from depot)",
                f"- Customer IDs: {self.farthest_customers[:5]}" +
                (" ..." if len(self.farthest_customers) > 5 else ""),
            ])
        
        if self.route_stats:
            lines.extend([
                "",
                "### Incumbent Solution Statistics",
                f"- Vehicles used: {self.route_stats.num_vehicles}",
                f"- Total distance: {self.route_stats.total_distance:.1f}",
                f"- Avg route length: {self.route_stats.avg_route_length:.1f}",
            ])
        
        return "\n".join(lines)


def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def analyze_distribution(
    depot: tuple[float, float],
    customer_locs: list[tuple[float, float]]
) -> tuple[DistributionType, float]:
    """Analyze the spatial distribution of customers.
    
    Uses k-means style analysis to determine if customers are clustered,
    random, or mixed.
    
    Returns:
        Tuple of (distribution_type, avg_inter_customer_distance).
    """
    if not customer_locs:
        return DistributionType.RANDOM, 0.0
    
    # Calculate distances from depot
    depot_distances = [
        _euclidean_distance(depot[0], depot[1], cx, cy)
        for cx, cy in customer_locs
    ]
    avg_dist_from_depot = sum(depot_distances) / len(depot_distances)
    
    # Calculate inter-customer distances
    if len(customer_locs) > 1:
        total_inter_dist = 0.0
        count = 0
        for i in range(len(customer_locs)):
            for j in range(i + 1, len(customer_locs)):
                d = _euclidean_distance(
                    customer_locs[i][0], customer_locs[i][1],
                    customer_locs[j][0], customer_locs[j][1]
                )
                total_inter_dist += d
                count += 1
        avg_inter_dist = total_inter_dist / count if count > 0 else 0.0
    else:
        avg_inter_dist = depot_distances[0] if depot_distances else 0.0
    
    # Simple heuristic: check variance of distances from depot
    if depot_distances:
        mean_d = sum(depot_distances) / len(depot_distances)
        variance = sum((d - mean_d) ** 2 for d in depot_distances) / len(depot_distances)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_d if mean_d > 0 else 0.0  # Coefficient of variation
        
        # High CV suggests clustering
        if cv > 0.5:
            dist_type = DistributionType.CLUSTERED
        elif cv > 0.3:
            dist_type = DistributionType.RANDOM_CLUSTERED
        else:
            dist_type = DistributionType.RANDOM
    else:
        dist_type = DistributionType.RANDOM
    
    return dist_type, avg_inter_dist


def create_scene_card(
    instance: "VRPTWInstance",
    incumbent_routes: list[list[int]] | None = None
) -> SceneCard:
    """Create a Scene Card from a VRPTW instance.
    
    This function analyzes the instance and creates a structured
    representation suitable for enhancing DRoC prompts.
    
    Args:
        instance: The VRPTW instance.
        incumbent_routes: Optional list of routes from incumbent solution.
    
    Returns:
        A SceneCard with comprehensive instance analysis.
    """
    n = instance.size
    depot = (instance.x_coords.get(instance.depot_id, 0.0),
             instance.y_coords.get(instance.depot_id, 0.0))
    
    # Collect customer data
    customer_locs = []
    customer_profiles = []
    demands = []
    tw_widths = []
    distances_from_depot = []
    
    for cid in instance.customer_ids:
        x = instance.x_coords.get(cid, 0.0)
        y = instance.y_coords.get(cid, 0.0)
        d = instance.demand.get(cid, 0.0)
        rt = instance.ready_time.get(cid, 0.0)
        dt = instance.due_time.get(cid, 999999)
        st = instance.service_time.get(cid, 0.0)
        
        customer_locs.append((x, y))
        demands.append(d)
        tw_widths.append(dt - rt)
        
        dist = _euclidean_distance(depot[0], depot[1], x, y)
        distances_from_depot.append((cid, dist))
        
        profile = CustomerProfile(
            customer_id=cid,
            x=x, y=y,
            demand=d,
            ready_time=rt,
            due_time=dt,
            service_time=st
        )
        customer_profiles.append(profile)
    
    # Calculate statistics
    total_demand = sum(demands)
    avg_tw_width = sum(tw_widths) / n if n > 0 else 0.0
    planning_horizon = max(instance.due_time.get(cid, 0) for cid in instance.customer_ids) if instance.customer_ids else 0.0
    
    # Capacity utilization
    num_vehicles = instance.vehicle_count or max(1, int(total_demand / instance.vehicle_capacity) + 1)
    capacity_utilization = total_demand / (num_vehicles * instance.vehicle_capacity) if instance.vehicle_capacity > 0 else 0.0
    
    # Time window tightness
    if planning_horizon > 0:
        tw_ratio = avg_tw_width / planning_horizon if planning_horizon > 0 else 1.0
        tw_tightness_score = 1.0 - tw_ratio  # Higher = tighter
    else:
        tw_tightness_score = 0.5
    
    if tw_tightness_score < 0.25:
        tw_tightness = TimeWindowTightness.LOOSE
    elif tw_tightness_score < 0.5:
        tw_tightness = TimeWindowTightness.MEDIUM
    else:
        tw_tightness = TimeWindowTightness.TIGHT
    
    # Identify critical customers (tight time windows)
    # Tight TW: TW width < 2 * avg service time OR customer is far from depot
    avg_service_time = sum(instance.service_time.get(cid, 0) for cid in instance.customer_ids) / n if n > 0 else 0
    critical = []
    for cid in instance.customer_ids:
        tw_width = instance.due_time.get(cid, 0) - instance.ready_time.get(cid, 0)
        dist = next((d for c_id, d in distances_from_depot if c_id == cid), 0)
        if tw_width < avg_service_time * 2 or (tw_width < planning_horizon * 0.2 and dist > 50):
            critical.append(cid)
    
    # Sort by distance
    distances_from_depot.sort(key=lambda x: x[1], reverse=True)
    farthest = [cid for cid, _ in distances_from_depot[:10]]
    
    distances_from_depot.sort(key=lambda x: x[1])
    nearest = [cid for cid, _ in distances_from_depot[:5]]
    
    # Analyze distribution
    dist_type, avg_inter_dist = analyze_distribution(depot, customer_locs)
    
    # Route statistics from incumbent (if provided)
    route_stats = None
    if incumbent_routes:
        num_veh_used = len(incumbent_routes)
        total_dist = 0.0
        route_lengths = []
        
        for route in incumbent_routes:
            if not route:
                continue
            # Calculate route length
            route_dist = 0.0
            prev = instance.depot_id
            for cust in route:
                curr_dist = instance.dist(prev, cust)
                route_dist += curr_dist
                prev = cust
            route_dist += instance.dist(prev, instance.depot_id)  # Return to depot
            total_dist += route_dist
            route_lengths.append(route_dist)
        
        if route_lengths:
            avg_route_len = sum(route_lengths) / len(route_lengths)
            max_route_len = max(route_lengths)
            min_route_len = min(route_lengths)
            
            route_stats = RouteStatistics(
                num_vehicles=num_veh_used,
                total_distance=total_dist,
                avg_route_length=avg_route_len,
                max_route_length=max_route_len,
                min_route_length=min_route_len,
                avg_load_utilization=0.0,  # Would need route load data
                route_details=incumbent_routes
            )
    
    return SceneCard(
        instance_name=instance.name,
        num_customers=n,
        num_vehicles=num_vehicles,
        vehicle_capacity=instance.vehicle_capacity,
        depot_location=depot,
        customer_locations=customer_locs,
        distribution_type=dist_type,
        avg_inter_customer_distance=avg_inter_dist,
        tw_tightness=tw_tightness,
        tw_tightness_score=tw_tightness_score,
        avg_tw_width=avg_tw_width,
        planning_horizon=planning_horizon,
        capacity_utilization=capacity_utilization,
        total_demand=total_demand,
        critical_customers=critical,
        farthest_customers=farthest,
        nearest_customers=nearest,
        customer_profiles=customer_profiles,
        route_stats=route_stats,
    )


# Convenience function
def scene_card_to_string(instance: "VRPTWInstance") -> str:
    """Create a Scene Card and return its prompt string.
    
    Args:
        instance: The VRPTW instance.
    
    Returns:
        Formatted string for LLM prompts.
    """
    card = create_scene_card(instance)
    return card.to_prompt_string()
