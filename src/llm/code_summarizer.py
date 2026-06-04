"""Code Summarizer for DRoC - Relevance assessment for retrieved code.

This module implements the code summarization step in the DRoC pipeline.
It evaluates whether retrieved code snippets are relevant to a specific
constraint and extracts the key code patterns.

Based on DRoC's summarize_document() function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.llm.constraint_retriever import RetrievedConstraintCodes

logger = logging.getLogger(__name__)


@dataclass
class CodeSummary:
    """Summary of a code snippet's relevance to a constraint."""
    
    relevance: bool
    code_snippet: str
    explanation: str
    constraint_name: str
    quality_score: float = 0.0
    
    def __str__(self) -> str:
        status = "RELEVANT" if self.relevance else "NOT RELEVANT"
        return f"CodeSummary({self.constraint_name}, {status}, score={self.quality_score:.2f})"


@dataclass
class ConstraintContext:
    """Context for a constraint combining multiple code snippets."""
    
    constraint_name: str
    relevant_snippets: list[CodeSummary]
    combined_context: str
    
    @property
    def is_empty(self) -> bool:
        return len(self.relevant_snippets) == 0


class CodeSummarizer:
    """Summarizes and filters retrieved code snippets.
    
    This class evaluates code snippets for relevance to a specific
    constraint keyword and extracts the most useful patterns.
    
    Example:
        summarizer = CodeSummarizer(solver="OR-tools")
        
        # Summarize retrieved codes for "Time Windows"
        retrieved = RetrievedConstraintCodes(
            keyword="Time Windows",
            snippets=[...]
        )
        summary = summarizer.summarize(retrieved, "Time Windows")
        
        # Filter to most relevant snippets
        filtered = summarizer.filter_relevant(retrieved, "Time Windows")
    """
    
    def __init__(
        self,
        solver: str = "OR-tools",
        llm_client: Any = None,
        relevance_threshold: float = 0.5,
    ):
        """Initialize the code summarizer.
        
        Args:
            solver: The solver backend ("OR-tools" or "Gurobi").
            llm_client: Optional LLM client for semantic evaluation.
            relevance_threshold: Minimum quality score for relevance.
        """
        self.solver = solver
        self.llm_client = llm_client
        self.relevance_threshold = relevance_threshold
    
    def summarize(
        self,
        retrieved_codes: "RetrievedConstraintCodes",
        constraint_keyword: str | None = None,
    ) -> CodeSummary:
        """Summarize a single code snippet's relevance.
        
        Args:
            retrieved_codes: The retrieved constraint codes.
            constraint_keyword: The constraint keyword to evaluate against.
        
        Returns:
            CodeSummary with relevance assessment.
        """
        if constraint_keyword is None:
            constraint_keyword = retrieved_codes.keyword
        
        if retrieved_codes.is_empty():
            return CodeSummary(
                relevance=False,
                code_snippet="",
                explanation="No code snippets retrieved",
                constraint_name=constraint_keyword,
                quality_score=0.0,
            )
        
        # If LLM client is available, use it for semantic evaluation
        if self.llm_client is not None:
            return self._summarize_with_llm(
                retrieved_codes.snippets[0],
                constraint_keyword,
            )
        
        # Otherwise, use rule-based evaluation
        return self._summarize_rule_based(
            retrieved_codes.snippets[0],
            constraint_keyword,
        )
    
    def summarize_all(
        self,
        retrieved: dict[str, "RetrievedConstraintCodes"],
    ) -> dict[str, CodeSummary]:
        """Summarize all retrieved codes.
        
        Args:
            retrieved: Dictionary from constraint retriever.
        
        Returns:
            Dictionary mapping keywords to summaries.
        """
        results = {}
        for keyword, codes in retrieved.items():
            results[keyword] = self.summarize(codes, keyword)
        return results
    
    def filter_relevant(
        self,
        retrieved_codes: "RetrievedConstraintCodes",
        constraint_keyword: str | None = None,
        max_snippets: int = 1,
    ) -> list[CodeSummary]:
        """Filter retrieved snippets to most relevant ones.
        
        Args:
            retrieved_codes: Retrieved codes to filter.
            constraint_keyword: The constraint keyword.
            max_snippets: Maximum number of snippets to return.
        
        Returns:
            List of relevant CodeSummary objects, sorted by quality.
        """
        if constraint_keyword is None:
            constraint_keyword = retrieved_codes.keyword
        
        if retrieved_codes.is_empty():
            return []
        
        summaries = []
        for snippet in retrieved_codes.snippets:
            if self.llm_client is not None:
                summary = self._summarize_with_llm(snippet, constraint_keyword)
            else:
                summary = self._summarize_rule_based(snippet, constraint_keyword)
            
            if summary.relevance:
                summaries.append(summary)
        
        # Sort by quality score and return top-k
        summaries.sort(key=lambda s: s.quality_score, reverse=True)
        return summaries[:max_snippets]
    
    def rank_snippets(
        self,
        retrieved_codes: "RetrievedConstraintCodes",
        constraint_keyword: str | None = None,
    ) -> list[tuple[int, float]]:
        """Rank snippets by relevance to constraint.
        
        Args:
            retrieved_codes: Retrieved codes to rank.
            constraint_keyword: The constraint keyword.
        
        Returns:
            List of (index, score) tuples sorted by score descending.
        """
        if constraint_keyword is None:
            constraint_keyword = retrieved_codes.keyword
        
        scores = []
        for i, snippet in enumerate(retrieved_codes.snippets):
            if self.llm_client is not None:
                summary = self._summarize_with_llm(snippet, constraint_keyword)
            else:
                summary = self._summarize_rule_based(snippet, constraint_keyword)
            scores.append((i, summary.quality_score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _summarize_with_llm(
        self,
        snippet: dict[str, Any],
        constraint_keyword: str,
    ) -> CodeSummary:
        """Use LLM to evaluate code relevance."""
        try:
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            # Define output schema
            class summary_schema:
                relevance: str  # 'yes' or 'no'
                code_snippet: str
                explanation: str
            
            # Create prompt
            prompt = PromptTemplate(
                template="""You are an expert in Python programming and {solver} for vehicle routing problems.

I will give you a retrieved document potentially related to {keyword}, and you will firstly assess if the document includes Python code to program {keyword}.

If so, you should explain how the code addresses the constraint of {keyword}.

Here is the retrieved document:
```
{context}
```

If the document contains Python code related to {keyword}, grade it as relevant.
After that, extract the key code snippet in the document related to {keyword}.
Finally, produce an explanation on how to program the constraint of {keyword}.

Structure your answer with:
1. Binary score 'yes' or 'no' for relevance
2. The key code snippet (if relevant)
3. Explanation of how to implement the constraint

If the document is not related, just return 'no' for the binary score.""",
                input_variables=["solver", "keyword", "context"],
            )
            
            # Get LLM response
            content = snippet.get("content", "")
            if hasattr(content, "page_content"):
                content = content.page_content
            
            response = self.llm_client.invoke(
                prompt.format(
                    solver=self.solver,
                    keyword=constraint_keyword,
                    context=content,
                )
            )
            
            # Parse response
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Simple rule-based parsing
            is_relevant = "yes" in response_text.lower()[:10]
            score = 1.0 if is_relevant else 0.0
            
            return CodeSummary(
                relevance=is_relevant,
                code_snippet=self._extract_code_snippet(response_text),
                explanation=response_text,
                constraint_name=constraint_keyword,
                quality_score=score,
            )
            
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using rule-based fallback")
            return self._summarize_rule_based(snippet, constraint_keyword)
    
    def _summarize_rule_based(
        self,
        snippet: dict[str, Any],
        constraint_keyword: str,
    ) -> CodeSummary:
        """Rule-based summarization (no LLM required)."""
        content = snippet.get("content", "")
        if hasattr(content, "page_content"):
            content = content.page_content
        
        content_lower = content.lower()
        keyword_lower = constraint_keyword.lower()
        
        # Check for relevant keywords in code
        relevance_indicators = {
            "Capacitated": ["capacity", "demand", "load", "vehicle_capacity", "AddDimensionWithVehicleCapacity"],
            "Time Windows": ["time_windows", "time window", "CumulVar", "SetRange", "ready_time", "due_time"],
            "Service Time": ["service_time", "service duration", "handling time"],
            "Multiple Depots": ["multiple depot", "multi depot", "depot", "depots"],
            "Duration Limit": ["duration", "max_time", "max_duration", "time_limit"],
            "Multiple Vehicles": ["num_vehicle", "vehicles", "fleet"],
        }
        
        # Get indicators for this keyword
        indicators = relevance_indicators.get(
            constraint_keyword,
            [keyword_lower.replace(" ", "_"), keyword_lower.replace(" ", "")]
        )
        
        # Count matches
        match_count = sum(1 for ind in indicators if ind in content_lower)
        total_indicators = len(indicators)
        
        # Calculate score
        score = match_count / total_indicators if total_indicators > 0 else 0.0
        is_relevant = score >= self.relevance_threshold
        
        # Extract meaningful code snippet (first 500 chars)
        code_snippet = content[:500] if len(content) > 500 else content
        
        explanation = (
            f"Rule-based evaluation found {match_count}/{total_indicators} "
            f"relevance indicators for '{constraint_keyword}'"
        )
        
        return CodeSummary(
            relevance=is_relevant,
            code_snippet=code_snippet,
            explanation=explanation,
            constraint_name=constraint_keyword,
            quality_score=score,
        )
    
    def _extract_code_snippet(self, response_text: str) -> str:
        """Extract code snippet from LLM response."""
        # Look for code between markdown fences
        import re
        
        # Match code blocks
        code_pattern = r"```(?:\w+)?\n?(.*?)```"
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code block, try to extract lines that look like code
        lines = response_text.split("\n")
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in ["def ", "import ", "from ", "class ", "routing.", "manager.", "= ", "//", "#"]):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped == "" or len(code_lines) > 50):
                break
        
        return "\n".join(code_lines[:30])  # Limit to 30 lines


class ConstraintContextBuilder:
    """Builds combined context from multiple constraint summaries."""
    
    def __init__(self, summarizer: CodeSummarizer):
        """Initialize the context builder.
        
        Args:
            summarizer: CodeSummarizer instance for evaluation.
        """
        self.summarizer = summarizer
    
    def build_context(
        self,
        retrieved: dict[str, "RetrievedConstraintCodes"],
        use_llm_ranking: bool = False,
    ) -> dict[str, ConstraintContext]:
        """Build context dictionary from retrieved codes.
        
        Args:
            retrieved: Dictionary from constraint retriever.
            use_llm_ranking: Whether to use LLM for ranking multiple snippets.
        
        Returns:
            Dictionary mapping keywords to ConstraintContext.
        """
        context_dict = {}
        
        for keyword, codes in retrieved.items():
            if codes.is_empty():
                context_dict[keyword] = ConstraintContext(
                    constraint_name=keyword,
                    relevant_snippets=[],
                    combined_context="",
                )
                continue
            
            # Filter to relevant snippets
            relevant = self.summarizer.filter_relevant(codes, keyword, max_snippets=2)
            
            if not relevant:
                # If no relevant snippets, use the best one anyway
                if self.summarizer.llm_client:
                    summary = self.summarizer._summarize_with_llm(
                        codes.snippets[0], keyword
                    )
                else:
                    summary = self.summarizer._summarize_rule_based(
                        codes.snippets[0], keyword
                    )
                relevant = [summary]
            
            # Build combined context
            combined = self._combine_snippets(relevant)
            
            context_dict[keyword] = ConstraintContext(
                constraint_name=keyword,
                relevant_snippets=relevant,
                combined_context=combined,
            )
        
        return context_dict
    
    def _combine_snippets(self, snippets: list[CodeSummary]) -> str:
        """Combine multiple snippets into a single context string."""
        parts = []
        
        for i, snippet in enumerate(snippets):
            if i > 0:
                parts.append("\n" + "=" * 40 + "\n")
            parts.append(f"# Example {i+1}: {snippet.constraint_name}\n")
            parts.append(snippet.code_snippet)
            parts.append("\n")
            parts.append(f"# Explanation: {snippet.explanation}")
        
        return "".join(parts)
    
    def format_for_prompt(
        self,
        context_dict: dict[str, ConstraintContext],
    ) -> str:
        """Format context dictionary for LLM prompt.
        
        Args:
            context_dict: Dictionary from build_context().
        
        Returns:
            Formatted string for use in prompt.
        """
        lines = []
        
        for keyword, ctx in context_dict.items():
            if ctx.is_empty:
                continue
            
            lines.append(f"\n{'=' * 50}")
            lines.append(f"Constraint: {keyword}")
            lines.append(f"{'=' * 50}")
            lines.append(ctx.combined_context)
        
        return "\n".join(lines)


# Convenience functions
def summarize_code(
    code: str,
    constraint_keyword: str,
    solver: str = "OR-tools",
    llm_client: Any = None,
) -> CodeSummary:
    """Summarize a code snippet's relevance to a constraint.
    
    Args:
        code: The code snippet.
        constraint_keyword: The constraint keyword.
        solver: The solver backend.
        llm_client: Optional LLM client.
    
    Returns:
        CodeSummary with relevance assessment.
    """
    summarizer = CodeSummarizer(solver=solver, llm_client=llm_client)
    
    snippet = {"content": code}
    return summarizer._summarize_rule_based(snippet, constraint_keyword)


def filter_relevant_codes(
    retrieved: dict[str, "RetrievedConstraintCodes"],
    solver: str = "OR-tools",
    llm_client: Any = None,
) -> dict[str, ConstraintContext]:
    """Filter retrieved codes to most relevant ones.
    
    Args:
        retrieved: Dictionary from constraint retriever.
        solver: The solver backend.
        llm_client: Optional LLM client.
    
    Returns:
        Dictionary mapping keywords to ConstraintContext.
    """
    summarizer = CodeSummarizer(solver=solver, llm_client=llm_client)
    builder = ConstraintContextBuilder(summarizer)
    return builder.build_context(retrieved)
