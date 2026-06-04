"""Metrics for comparing VRPTW solutions."""

from src.domain.schema import RouteSolution


def lexicographic_compare(a: RouteSolution, b: RouteSolution) -> int:
    """Compare two solutions lexicographically.
    
    Priority: 1) fewer vehicles, 2) shorter distance, 3) feasible > infeasible
    
    Returns:
        -1 if a is better, 1 if b is better, 0 if equal
    """
    if not a.feasible and not b.feasible:
        return 0
    
    if a.feasible and not b.feasible:
        return -1
    if not a.feasible and b.feasible:
        return 1
    
    if a.vehicles_used != b.vehicles_used:
        return -1 if a.vehicles_used < b.vehicles_used else 1
    
    if abs(a.total_distance - b.total_distance) > 1e-6:
        return -1 if a.total_distance < b.total_distance else 1
    
    return 0


def compute_gap(solution: RouteSolution, bks: float) -> float:
    """Compute gap to best known solution (BKS).
    
    Args:
        solution: Computed solution
        bks: Best known solution distance
        
    Returns:
        Gap percentage, or infinity if BKS is zero
    """
    if bks <= 0:
        return float('inf')
    
    return (solution.total_distance - bks) / bks * 100
