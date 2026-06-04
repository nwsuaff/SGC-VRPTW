"""Problem Decomposer for VRPTW - DRoC Style.

This module decomposes VRPTW instances into constraint keywords,
similar to the DRoC paper's approach. The keywords are used for
retrieving relevant code examples from the RAG system.

Supported constraints:
- Capacitated: Vehicle capacity constraints
- Time Windows: Customer time window constraints
- Multiple Depots: Multiple depot locations
- Service Time: Per-customer service duration
- Duration Limit: Maximum route duration
- Prize Collecting: Customer prizes with optional visits
- Pickup and Delivery: Pairs of pickup/delivery nodes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Enumeration of VRPTW constraint types."""
    
    # Core constraints (always present in VRPTW)
    CAPACITATED = auto()
    TIME_WINDOWS = auto()
    
    # Optional constraints
    MULTIPLE_DEPOTS = auto()
    SERVICE_TIME = auto()
    DURATION_LIMIT = auto()
    PRIZE_COLLECTING = auto()
    PICKUP_DELIVERY = auto()
    RESOURCE_CONSTRAINTS = auto()
    MULTIPLE_TRIPS = auto()
    
    # Derived constraints
    CAPACITY_WITH_TIME = auto()
    CAPACITY_WITH_SERVICE = auto()
    TW_WITH_SERVICE = auto()


@dataclass
class DecomposedProblem:
    """Result of problem decomposition."""
    
    problem_name: str
    keywords: list[str]
    constraint_types: list[ConstraintType]
    metadata: dict
    
    def __str__(self) -> str:
        return f"DecomposedProblem({self.problem_name}, keywords={self.keywords})"
    
    def __repr__(self) -> str:
        return self.__str__()


class VRPTWDecomposer:
    """Decomposes VRPTW instances into constraint keywords.
    
    This is the first step in the DRoC pipeline. It extracts the
    constraint keywords from the problem instance, which are then
    used to retrieve relevant code examples.
    
    Supports two modes:
    1. Rule-based: Fast, no LLM needed, uses pattern matching
    2. LLM-based: More accurate, uses LLM to extract keywords
    
    Example:
        decomposer = VRPTWDecomposer()
        result = decomposer.decompose(instance)
        # result.keywords = ["Capacitated", "Time Windows", "Service Time"]
        
        # Or with LLM for more accurate decomposition
        decomposer_llm = VRPTWDecomposer(use_llm=True, llm_client=client)
        result_llm = decomposer_llm.decompose(instance)
    """
    
    # Keyword mappings for different constraint types
    KEYWORD_MAP = {
        ConstraintType.CAPACITATED: ["capacitated", "capacity", "vehicle capacity", "load"],
        ConstraintType.TIME_WINDOWS: ["time windows", "time window", "tw", "ready time", "due time"],
        ConstraintType.MULTIPLE_DEPOTS: ["multiple depots", "multi-depot", "depot", "multi depot"],
        ConstraintType.SERVICE_TIME: ["service time", "service duration", "handling time"],
        ConstraintType.DURATION_LIMIT: ["duration limit", "max duration", "route duration", "time limit"],
        ConstraintType.PRIZE_COLLECTING: ["prize collecting", "prize", "profit", "optional"],
        ConstraintType.PICKUP_DELIVERY: ["pickup delivery", "pickup", "delivery", "pd", "pdptw"],
        ConstraintType.RESOURCE_CONSTRAINTS: ["resource", "resources", "crew", "driver"],
        ConstraintType.MULTIPLE_TRIPS: ["multiple trips", "vehicle count", "fleet size"],
    }
    
    # Composite constraint keywords
    COMPOSITE_KEYWORDS = {
        ("capacitated", "time windows"): "Capacitated VRP with Time Windows",
        ("capacitated", "service time"): "Capacitated VRP with Service Time",
        ("time windows", "service time"): "VRP with Time Windows and Service Time",
        ("pickup", "delivery", "time windows"): "PDPTW",
    }
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        """Initialize the decomposer.
        
        Args:
            use_llm: Whether to use LLM for decomposition.
            llm_client: LLM client for advanced decomposition (required if use_llm=True).
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        
        if use_llm and llm_client is None:
            logger.warning("use_llm=True but no llm_client provided, falling back to rule-based")
            self.use_llm = False
    
    def decompose(self, instance: "VRPTWInstance") -> DecomposedProblem:
        """Decompose a VRPTW instance into constraint keywords.
        
        Args:
            instance: The VRPTW instance to decompose.
        
        Returns:
            DecomposedProblem containing keywords and metadata.
        """
        logger.debug(f"Decomposing instance: {instance.name}")
        
        # Use LLM-based decomposition if enabled
        if self.use_llm and self.llm_client:
            return self._decompose_with_llm(instance)
        
        # Use rule-based decomposition
        constraint_types = self._detect_constraints(instance)
        
        # Extract keywords from detected constraints
        keywords = self._extract_keywords(constraint_types)
        
        # Build problem name
        problem_name = self._build_problem_name(constraint_types, instance)
        
        # Create metadata
        metadata = self._build_metadata(instance, constraint_types)
        
        result = DecomposedProblem(
            problem_name=problem_name,
            keywords=keywords,
            constraint_types=constraint_types,
            metadata=metadata,
        )
        
        logger.info(f"Decomposed: {result}")
        return result
    
    def _decompose_with_llm(self, instance: "VRPTWInstance") -> DecomposedProblem:
        """Decompose using LLM (matches original DRoC decomposer behavior)."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            # Build prompt following original DRoC pattern
            system = """You will extract the keywords of a vehicle routing problem (VRP) for me. 
            I give you the name of a VRP and you produce the keywords according to its constraints.
            Structure your answer with a list of keywords inside "<>" and use commas to separate different keywords. Do not return other things. 
            For example, the output of "Capacitated Vehicle Routing Problem with Time Windows and Multiple Depots (CVRPTWMD)" should be <Capacitated, Time Windows, Multiple Depots>, 
            and the output of "Prize Collecting Travelling Salesman Problem (PCTSP)" should be <Prize Collecting>."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "Here is the name of the VRP: \n\n {problem}"),
            ])
            
            # Get LLM client
            if hasattr(self.llm_client, 'model'):
                model_name = self.llm_client.model
            else:
                model_name = "unknown"
            
            if model_name.startswith("gpt"):
                llm = ChatOpenAI(model=model_name, temperature=0.0)
            elif model_name.startswith("claude"):
                llm = ChatAnthropic(model=model_name, temperature=0.0, max_tokens=5000)
            else:
                # Try to use the client directly
                llm = self.llm_client
            
            keyword_extractor = prompt | llm
            
            # Build problem name from instance
            problem_name = self._build_problem_name_from_instance(instance)
            
            res = keyword_extractor.invoke({"problem": problem_name})
            keywords_text = res.content if hasattr(res, 'content') else str(res)
            
            # Parse keywords from <...> format
            keywords = self._parse_keywords_from_llm(keywords_text)
            
            # Map keywords to constraint types
            constraint_types = self._keywords_to_constraint_types(keywords)
            
            result = DecomposedProblem(
                problem_name=problem_name,
                keywords=keywords,
                constraint_types=constraint_types,
                metadata={
                    "instance_name": instance.name,
                    "decomposition_method": "llm",
                    "raw_llm_output": keywords_text,
                },
            )
            
            logger.info(f"Decomposed (LLM): {result}")
            return result
            
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}, falling back to rule-based")
            return self.decompose(instance)
    
    def _parse_keywords_from_llm(self, text: str) -> list[str]:
        """Parse keywords from LLM output in <...> format."""
        import re
        # Find content within < >
        match = re.search(r'<(.+?)>', text)
        if match:
            content = match.group(1)
            keywords = [kw.strip() for kw in content.split(',')]
            return keywords
        return []
    
    def _keywords_to_constraint_types(self, keywords: list[str]) -> list[ConstraintType]:
        """Map keyword strings to ConstraintType enums."""
        keyword_to_type = {
            "capacitated": ConstraintType.CAPACITATED,
            "capacity": ConstraintType.CAPACITATED,
            "vehicle capacity": ConstraintType.CAPACITATED,
            "time windows": ConstraintType.TIME_WINDOWS,
            "time window": ConstraintType.TIME_WINDOWS,
            "tw": ConstraintType.TIME_WINDOWS,
            "multiple depots": ConstraintType.MULTIPLE_DEPOTS,
            "multi-depot": ConstraintType.MULTIPLE_DEPOTS,
            "service time": ConstraintType.SERVICE_TIME,
            "service duration": ConstraintType.SERVICE_TIME,
            "duration limit": ConstraintType.DURATION_LIMIT,
            "max duration": ConstraintType.DURATION_LIMIT,
            "prize collecting": ConstraintType.PRIZE_COLLECTING,
            "pickups and deliveries": ConstraintType.PICKUP_DELIVERY,
            "pickup delivery": ConstraintType.PICKUP_DELIVERY,
            "resource constraints": ConstraintType.RESOURCE_CONSTRAINTS,
            "multiple vehicles": ConstraintType.MULTIPLE_TRIPS,
        }
        
        constraint_types = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in keyword_to_type:
                ct = keyword_to_type[kw_lower]
                if ct not in constraint_types:
                    constraint_types.append(ct)
        
        # Always include core constraints for VRPTW
        if ConstraintType.CAPACITATED not in constraint_types:
            constraint_types.append(ConstraintType.CAPACITATED)
        if ConstraintType.TIME_WINDOWS not in constraint_types:
            constraint_types.append(ConstraintType.TIME_WINDOWS)
        
        return constraint_types
    
    def _build_problem_name_from_instance(self, instance: "VRPTWInstance") -> str:
        """Build a problem name string from VRPTWInstance metadata."""
        parts = ["Capacitated Vehicle Routing Problem with Time Windows"]
        
        if self._has_service_times(instance):
            parts.append("and Service Time")
        if self._has_multiple_depots(instance):
            parts.append("and Multiple Depots")
        if self._has_pickup_delivery(instance):
            parts = ["Vehicle Routing Problem with Pickups and Deliveries and Time Windows"]
        
        return " ".join(parts)
    
    def decompose_from_name(self, problem_name: str) -> DecomposedProblem:
        """Decompose a problem from its name string.
        
        This method uses pattern matching on the problem name to extract
        constraints. Useful for problems without a full instance object.
        
        Args:
            problem_name: The problem name, e.g., "CVRPTWMD", "PDPTW".
        
        Returns:
            DecomposedProblem containing keywords and metadata.
        """
        logger.debug(f"Decomposing from name: {problem_name}")
        
        # Detect constraints from name patterns
        constraint_types = self._detect_from_name(problem_name)
        
        # Extract keywords
        keywords = self._extract_keywords(constraint_types)
        
        # Build problem name
        problem_name_clean = self._build_problem_name(constraint_types, None)
        
        result = DecomposedProblem(
            problem_name=problem_name_clean,
            keywords=keywords,
            constraint_types=constraint_types,
            metadata={"source_name": problem_name},
        )
        
        logger.info(f"Decomposed from name: {result}")
        return result
    
    def _detect_constraints(self, instance: "VRPTWInstance") -> list[ConstraintType]:
        """Detect which constraints are present in the instance."""
        constraints = []
        
        # Always present in VRPTW
        constraints.append(ConstraintType.CAPACITATED)
        constraints.append(ConstraintType.TIME_WINDOWS)
        
        # Check vehicle capacity (always present)
        if instance.vehicle_capacity > 0:
            pass  # Already detected
        
        # Check for multiple depots
        if self._has_multiple_depots(instance):
            constraints.append(ConstraintType.MULTIPLE_DEPOTS)
        
        # Check for service times
        if self._has_service_times(instance):
            constraints.append(ConstraintType.SERVICE_TIME)
            constraints.append(ConstraintType.TW_WITH_SERVICE)
        
        # Check for duration limit
        if self._has_duration_limit(instance):
            constraints.append(ConstraintType.DURATION_LIMIT)
        
        # Check for pickup/delivery
        if self._has_pickup_delivery(instance):
            constraints.append(ConstraintType.PICKUP_DELIVERY)
        
        # Check for multiple trips
        if instance.vehicle_count is not None:
            constraints.append(ConstraintType.MULTIPLE_TRIPS)
        
        return constraints
    
    def _detect_from_name(self, name: str) -> list[ConstraintType]:
        """Detect constraints from problem name patterns.
        
        Uses DRoC-style pattern matching based on VRP problem naming conventions.
        Format: [Problem type][Constraint modifiers]
        
        Examples:
            - VRPTW: Vehicle Routing with Time Windows (base CVRPTW)
            - CVRPTW: Capacitated VRPTW
            - PDPTW: Pickup and Delivery with Time Windows
            - CVRPTWMD: Capacitated VRPTW with Multiple Depots
        """
        constraints = []
        name_upper = name.upper()
        name_lower = name.lower()
        
        # Strip file extensions and clean up
        clean_name = name_upper.replace(".PY", "").replace("_", " ").strip()
        
        # Determine base problem type
        is_pickup_delivery = "PDP" in clean_name or "PD" in clean_name
        is_cvrp = "CVRP" in clean_name or clean_name.startswith("VRP")
        
        # Always add core constraints for VRP problems
        if is_cvrp or "VRP" in clean_name or "TSP" in clean_name:
            constraints.append(ConstraintType.CAPACITATED)
            constraints.append(ConstraintType.TIME_WINDOWS)
        
        # Multiple depots
        if "MD" in clean_name or "MULTI" in clean_name:
            constraints.append(ConstraintType.MULTIPLE_DEPOTS)
        
        # Service time - only if explicitly marked with "S" that stands alone
        # PDPTWS = PDPTW with Service Time
        # CVRPTS = CVRP with Time Windows and Service
        if "S" in clean_name.split()[-1] or clean_name.endswith("S"):
            constraints.append(ConstraintType.SERVICE_TIME)
        
        # Duration limit - L suffix
        if clean_name.endswith("L") or "LIMIT" in clean_name:
            constraints.append(ConstraintType.DURATION_LIMIT)
        
        # Pickup and delivery
        if is_pickup_delivery:
            constraints.append(ConstraintType.PICKUP_DELIVERY)
            # Remove core constraints for PDPTW variants (they have their own base)
            if ConstraintType.CAPACITATED in constraints:
                constraints.remove(ConstraintType.CAPACITATED)
            if ConstraintType.TIME_WINDOWS in constraints:
                constraints.remove(ConstraintType.TIME_WINDOWS)
        
        # Resource constraints - R or RC suffix
        if "RC" in clean_name:
            constraints.append(ConstraintType.RESOURCE_CONSTRAINTS)
        
        # If nothing detected, assume basic VRPTW
        if not constraints:
            constraints.append(ConstraintType.CAPACITATED)
            constraints.append(ConstraintType.TIME_WINDOWS)
        
        return constraints
    
    def _has_multiple_depots(self, instance: "VRPTWInstance") -> bool:
        """Check if instance has multiple depots."""
        # Check if there are multiple depot nodes
        # This depends on how the instance encodes multiple depots
        # For now, check if there are nodes with special depot marking
        depot_count = sum(1 for cid, due in instance.due_time.items() 
                         if due == 999999 and cid == 0)
        return depot_count > 1
    
    def _has_service_times(self, instance: "VRPTWInstance") -> bool:
        """Check if instance has non-zero service times."""
        return any(st > 0 for st in instance.service_time.values())
    
    def _has_duration_limit(self, instance: "VRPTWInstance") -> bool:
        """Check if instance has a duration limit."""
        # Check if there's a metadata flag or if max time is set
        return instance.metadata.get("has_duration_limit", False)
    
    def _has_pickup_delivery(self, instance: "VRPTWInstance") -> bool:
        """Check if instance has pickup/delivery pairs."""
        # Check for paired demands (positive = pickup, negative = delivery)
        demands = list(instance.demand.values())
        has_positive = any(d > 0 for d in demands)
        has_negative = any(d < 0 for d in demands)
        return has_positive and has_negative
    
    def _extract_keywords(self, constraint_types: list[ConstraintType]) -> list[str]:
        """Extract human-readable keywords from constraint types."""
        keywords = []
        
        # Core constraints always get keywords
        core_keywords = {
            ConstraintType.CAPACITATED: "Capacitated",
            ConstraintType.TIME_WINDOWS: "Time Windows",
        }
        
        # Optional constraint keywords
        optional_keywords = {
            ConstraintType.MULTIPLE_DEPOTS: "Multiple Depots",
            ConstraintType.SERVICE_TIME: "Service Time",
            ConstraintType.DURATION_LIMIT: "Duration Limit",
            ConstraintType.PRIZE_COLLECTING: "Prize Collecting",
            ConstraintType.PICKUP_DELIVERY: "Pickups and Deliveries",
            ConstraintType.RESOURCE_CONSTRAINTS: "Resource Constraints",
            ConstraintType.MULTIPLE_TRIPS: "Multiple Vehicles",
            ConstraintType.CAPACITY_WITH_TIME: "Capacitated with Time Windows",
            ConstraintType.CAPACITY_WITH_SERVICE: "Capacitated with Service Time",
            ConstraintType.TW_WITH_SERVICE: "Time Windows with Service Time",
        }
        
        # Add core keywords
        for ct in constraint_types:
            if ct in core_keywords:
                keywords.append(core_keywords[ct])
        
        # Add optional keywords
        for ct in constraint_types:
            if ct in optional_keywords:
                kw = optional_keywords[ct]
                if kw not in keywords:
                    keywords.append(kw)
        
        return keywords
    
    def _build_problem_name(
        self, 
        constraint_types: list[ConstraintType],
        instance: "VRPTWInstance | None"
    ) -> str:
        """Build a clean problem name from constraints."""
        parts = []
        
        # Check for standard abbreviations
        has_multiple_depots = ConstraintType.MULTIPLE_DEPOTS in constraint_types
        has_service_time = ConstraintType.SERVICE_TIME in constraint_types
        has_duration_limit = ConstraintType.DURATION_LIMIT in constraint_types
        has_pickup_delivery = ConstraintType.PICKUP_DELIVERY in constraint_types
        has_resource = ConstraintType.RESOURCE_CONSTRAINTS in constraint_types
        
        # Build name based on constraint combination
        if has_pickup_delivery:
            parts.append("PDPTW")
            if has_service_time:
                parts.append("with Service Time")
            if has_duration_limit:
                parts.append("and Duration Limit")
            if has_multiple_depots:
                parts.append("and Multiple Depots")
        else:
            parts.append("CVRPTW")
            if has_multiple_depots:
                parts.append("MD")
            if has_resource:
                parts.append("RC")
            if has_service_time:
                parts.append("S")
            if has_duration_limit:
                parts.append("L")
        
        return " ".join(parts)
    
    def _build_metadata(
        self,
        instance: "VRPTWInstance",
        constraint_types: list[ConstraintType]
    ) -> dict:
        """Build metadata about the decomposition."""
        return {
            "instance_name": instance.name,
            "num_customers": instance.size,
            "vehicle_capacity": instance.vehicle_capacity,
            "vehicle_count": instance.vehicle_count,
            "has_service_time": self._has_service_times(instance),
            "num_constraints": len(constraint_types),
        }


# Convenience function for simple decomposition
def decompose_vrptw(instance: "VRPTWInstance") -> DecomposedProblem:
    """Decompose a VRPTW instance into constraint keywords.
    
    Args:
        instance: The VRPTW instance to decompose.
    
    Returns:
        DecomposedProblem with keywords and metadata.
    """
    decomposer = VRPTWDecomposer()
    return decomposer.decompose(instance)


def decompose_from_name(problem_name: str) -> DecomposedProblem:
    """Decompose a problem from its name.
    
    Args:
        problem_name: The problem name string.
    
    Returns:
        DecomposedProblem with keywords and metadata.
    """
    decomposer = VRPTWDecomposer()
    return decomposer.decompose_from_name(problem_name)
