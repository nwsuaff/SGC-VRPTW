"""Constraint Retriever for DRoC - RAG-based code retrieval.

This module implements the RAG (Retrieval-Augmented Generation) retrieval
for constraint-specific code examples, following the DRoC paper approach.

Workflow:
1. Initialize vector store (Chroma) with code examples
2. For each constraint keyword, retrieve relevant code snippets
3. Return filtered, relevant code for each constraint
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Type alias for code snippet
CodeSnippet = dict[str, Any]


@dataclass
class RetrievedConstraintCodes:
    """Retrieved code snippets for a constraint keyword."""
    
    keyword: str
    snippets: list[CodeSnippet]
    relevance_scores: list[float] | None = None
    
    def __len__(self) -> int:
        return len(self.snippets)
    
    def is_empty(self) -> bool:
        return len(self.snippets) == 0


class ConstraintRetriever:
    """Retrieves relevant code examples for VRPTW constraints.
    
    This retriever uses vector similarity search to find code examples
    that implement specific VRPTW constraints.
    
    Example:
        retriever = ConstraintRetriever(solver="OR-tools")
        result = retriever.retrieve(["Capacitated", "Time Windows"])
        # result["Capacitated"] = [code_snippet_1, code_snippet_2]
    """
    
    def __init__(
        self,
        solver: str = "OR-tools",
        vector_store_path: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 3,
    ):
        """Initialize the constraint retriever.
        
        Args:
            solver: The solver backend ("OR-tools" or "Gurobi").
            vector_store_path: Path to existing vector store. If None, uses default.
            embedding_model: Embedding model for vectorization.
            top_k: Number of top results to retrieve per keyword.
        """
        self.solver = solver
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._retriever = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of the vector store."""
        if self._initialized:
            return
        
        try:
            self._initialize_vector_store()
            self._initialized = True
        except ImportError as e:
            logger.warning(f"Chroma not available, using fallback retrieval: {e}")
            self._initialized = True  # Don't retry
            self._retriever = None
    
    def _initialize_vector_store(self) -> None:
        """Initialize the Chroma vector store."""
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        
        # Determine path based on solver
        if self.vector_store_path:
            base_path = self.vector_store_path
        else:
            # Default paths from DRoC
            if self.solver == "OR-tools":
                base_path = "./chroma_db/code"
            elif self.solver == "Gurobi":
                base_path = "./chroma_db/gurobi"
            else:
                raise ValueError(f"Unknown solver: {self.solver}")
        
        # Check if vector store exists
        if os.path.exists(base_path):
            logger.info(f"Loading existing vector store from {base_path}")
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
            vectorstore = Chroma(
                persist_directory=base_path,
                embedding_function=embeddings,
            )
        else:
            logger.info(f"Vector store not found at {base_path}, will create on demand")
            vectorstore = None
        
        if vectorstore:
            self._retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
    
    def retrieve(
        self,
        keywords: list[str],
        force_refresh: bool = False,
    ) -> dict[str, RetrievedConstraintCodes]:
        """Retrieve relevant code for each keyword.
        
        Args:
            keywords: List of constraint keywords (e.g., ["Capacitated", "Time Windows"]).
            force_refresh: If True, reinitialize the vector store.
        
        Returns:
            Dictionary mapping keywords to RetrievedConstraintCodes.
        """
        if force_refresh:
            self._initialized = False
        
        self._ensure_initialized()
        
        results = {}
        for keyword in keywords:
            logger.debug(f"Retrieving codes for keyword: {keyword}")
            retrieved = self._retrieve_for_keyword(keyword)
            results[keyword] = retrieved
        
        return results
    
    def _retrieve_for_keyword(self, keyword: str) -> RetrievedConstraintCodes:
        """Retrieve code snippets for a single keyword."""
        if self._retriever is None:
            # Fallback to static code snippets
            return self._fallback_retrieval(keyword)
        
        try:
            # Format query
            query = f"Python code of {keyword} for {self.solver} solver"
            
            # Retrieve documents
            docs = self._retriever.invoke(query)
            
            # Convert to CodeSnippet format
            snippets = []
            for doc in docs:
                snippet = {
                    "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                }
                snippets.append(snippet)
            
            return RetrievedConstraintCodes(
                keyword=keyword,
                snippets=snippets,
            )
            
        except Exception as e:
            logger.warning(f"Retrieval failed for {keyword}: {e}, using fallback")
            return self._fallback_retrieval(keyword)
    
    def _fallback_retrieval(self, keyword: str) -> RetrievedConstraintCodes:
        """Provide fallback code snippets when vector store is unavailable.
        
        These are hardcoded examples from DRoC's generated code repository.
        """
        fallback_codes = {
            "Capacitated": [
                {
                    "content": """# Capacity constraint implementation
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    return demands[node]

routing.AddDimensionWithVehicleCapacity(
    routing.RegisterUnaryTransitCallback(demand_callback),
    0,  # null capacity slack
    [vehicle_capacity] * num_vehicle,  # vehicle capacities
    True,  # start cumul at zero
    "Capacity"
)""",
                    "metadata": {"source": "fallback", "constraint": "Capacitated"},
                }
            ],
            "Time Windows": [
                {
                    "content": """# Time window constraint implementation
time_dimension = routing.GetDimensionOrDie("Time")

# Set time window for each location
for location_idx, (ready, due) in enumerate(time_windows):
    if location_idx == depot:
        continue
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(ready, due)

# Set depot time window
depot_index = manager.NodeToIndex(depot)
time_dimension.CumulVar(depot_index).SetRange(
    time_windows[depot][0], time_windows[depot][1]
)""",
                    "metadata": {"source": "fallback", "constraint": "Time Windows"},
                }
            ],
            "Service Time": [
                {
                    "content": """# Service time in transit callback
def transit_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    # Travel time + service time at destination
    return time_matrix[from_node][to_node] + service_times[to_node]

transit_callback_index = routing.RegisterTransitCallback(transit_callback)""",
                    "metadata": {"source": "fallback", "constraint": "Service Time"},
                }
            ],
            "Multiple Depots": [
                {
                    "content": """# Multiple depots implementation
# When using multiple depots, create separate routing models for each depot
# or use depot-specific vehicle assignments

# Example: 2 depots
depots = [0, 10]  # Depot node indices
vehicles_per_depot = [num_vehicle // 2, num_vehicle // 2]

for depot_idx, depot in enumerate(depots):
    manager = pywrapcp.RoutingIndexManager(
        len(time_matrix), 
        vehicles_per_depot[depot_idx], 
        depot
    )
    # Create routing model for this depot...
""",
                    "metadata": {"source": "fallback", "constraint": "Multiple Depots"},
                }
            ],
            "Duration Limit": [
                {
                    "content": """# Duration limit / maximum route time
time_dimension = routing.GetDimensionOrDie("Time")

# Set maximum route duration for each vehicle
for vehicle_id in range(num_vehicle):
    start_index = routing.Start(vehicle_id)
    # Set maximum cumulative time at start (depot)
    time_dimension.CumulVar(start_index).SetRange(0, max_duration)
    # Set maximum cumulative time at end
    end_index = routing.End(vehicle_id)
    time_dimension.CumulVar(end_index).SetUpper(max_duration)""",
                    "metadata": {"source": "fallback", "constraint": "Duration Limit"},
                }
            ],
            "Multiple Vehicles": [
                {
                    "content": """# Multiple vehicles configuration
num_vehicle = 10  # Total number of vehicles
depot = 0  # Single depot

manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicle, depot)
routing = pywrapcp.RoutingModel(manager)

# Or with heterogeneous fleet
vehicle_capacities = [200, 300, 400]  # Different capacities per vehicle
routing.AddDimensionWithVehicleCapacity(
    routing.RegisterUnaryTransitCallback(demand_callback),
    0,
    vehicle_capacities,
    True,
    "Capacity"
)""",
                    "metadata": {"source": "fallback", "constraint": "Multiple Vehicles"},
                }
            ],
        }
        
        # Find matching fallback codes
        matched = []
        keyword_lower = keyword.lower()
        
        for kw, codes in fallback_codes.items():
            if kw.lower() in keyword_lower or keyword_lower in kw.lower():
                matched.extend(codes)
        
        if not matched:
            # Return generic VRPTW code as fallback
            matched = fallback_codes.get("Capacitated", [])
        
        return RetrievedConstraintCodes(
            keyword=keyword,
            snippets=matched[:self.top_k],
        )
    
    def retrieve_single(
        self,
        keyword: str,
        query: str | None = None,
    ) -> RetrievedConstraintCodes:
        """Retrieve code for a single keyword with custom query.
        
        Args:
            keyword: The constraint keyword.
            query: Optional custom search query (overrides default).
        
        Returns:
            RetrievedConstraintCodes for the keyword.
        """
        self._ensure_initialized()
        
        if query is None:
            query = f"Python code of {keyword} for {self.solver}"
        
        if self._retriever is None:
            return self._fallback_retrieval(keyword)
        
        try:
            docs = self._retriever.invoke(query)
            snippets = []
            for doc in docs:
                snippet = {
                    "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                }
                snippets.append(snippet)
            
            return RetrievedConstraintCodes(
                keyword=keyword,
                snippets=snippets,
            )
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return self._fallback_retrieval(keyword)


class EnsembleConstraintRetriever(ConstraintRetriever):
    """Ensemble retriever combining multiple retrieval methods.
    
    Combines:
    - Vector store retrieval (semantic similarity)
    - BM25 retrieval (keyword matching)
    - Static fallback (rule-based)
    """
    
    def __init__(
        self,
        solver: str = "OR-tools",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        fallback_weight: float = 0.2,
        top_k: int = 3,
    ):
        """Initialize ensemble retriever.
        
        Args:
            solver: Solver backend.
            vector_weight: Weight for vector retrieval.
            bm25_weight: Weight for BM25 retrieval.
            fallback_weight: Weight for fallback retrieval.
            top_k: Number of results to retrieve.
        """
        super().__init__(solver=solver, top_k=top_k)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fallback_weight = fallback_weight
    
    def _initialize_bm25(self) -> Any:
        """Initialize BM25 retriever."""
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            import os
            
            # Load documents for BM25
            if self.solver == "OR-tools":
                doc_path = "./data/OR-tools/mds/"
            else:
                doc_path = "./data/Gurobi/"
            
            if os.path.exists(doc_path):
                loader = DirectoryLoader(
                    doc_path,
                    glob="**/*.md",
                    loader_cls=TextLoader,
                )
                docs = loader.load()
                
                retriever = BM25Retriever.from_documents(docs)
                retriever.k = self.top_k
                return retriever
            
        except ImportError:
            logger.warning("BM25 retriever not available")
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
        
        return None
    
    def retrieve(self, keywords: list[str]) -> dict[str, RetrievedConstraintCodes]:
        """Retrieve using ensemble of methods."""
        results = {}
        
        for keyword in keywords:
            # Get results from each method
            vector_results = self._retrieve_for_keyword(keyword)
            
            # Combine with BM25 if available
            if not hasattr(self, "_bm25_retriever"):
                self._bm25_retriever = self._initialize_bm25()
            
            if self._bm25_retriever is not None:
                try:
                    query = f"{keyword} {self.solver} code"
                    bm25_docs = self._bm25_retriever.invoke(query)
                    # Merge results (simplified - just add BM25 results)
                    for doc in bm25_docs[:self.top_k]:
                        snippet = {
                            "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                            "metadata": {"source": "bm25", **doc.metadata} if hasattr(doc, "metadata") else {},
                        }
                        if snippet not in vector_results.snippets:
                            vector_results.snippets.append(snippet)
                except Exception as e:
                    logger.debug(f"BM25 retrieval skipped: {e}")
            
            results[keyword] = vector_results
        
        return results


# Convenience functions
def retrieve_constraint_codes(
    keywords: list[str],
    solver: str = "OR-tools",
    top_k: int = 3,
) -> dict[str, RetrievedConstraintCodes]:
    """Convenience function to retrieve constraint codes.
    
    Args:
        keywords: List of constraint keywords.
        solver: Solver backend.
        top_k: Number of results per keyword.
    
    Returns:
        Dictionary mapping keywords to RetrievedConstraintCodes.
    """
    retriever = ConstraintRetriever(solver=solver, top_k=top_k)
    return retriever.retrieve(keywords)


def build_constraint_context(
    retrieved: dict[str, RetrievedConstraintCodes],
) -> dict[str, str]:
    """Build context string for each constraint from retrieved codes.
    
    Args:
        retrieved: Results from retrieve().
    
    Returns:
        Dictionary mapping keywords to combined context strings.
    """
    context = {}
    
    for keyword, codes in retrieved.items():
        if codes.is_empty():
            context[keyword] = ""
            continue
        
        # Combine all snippets for this keyword
        combined = []
        for i, snippet in enumerate(codes.snippets):
            combined.append(f"--- Example {i+1} for {keyword} ---\n{snippet['content']}")
        
        context[keyword] = "\n\n".join(combined)
    
    return context
