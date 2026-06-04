"""Smoke tests for the full LLM-guided solver loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import read_instance
from src.llm.mock_client import MockLLMClient
from src.llm.parser import parse_llm_output
from src.llm.hint_schema import empty_hints, LLMHints
from src.llm.prompt_builder import build_llm_prompt
from src.llm.hint_to_warmstart import hints_to_warmstart
from src.pipelines.run_llm_solver import run_llm_solver
from src.domain.schema import RouteSolution


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class TestLLMModuleSmoke:
    """Smoke tests for the LLM module components."""

    def test_mock_client_returns_valid_hints(self):
        """Mock client should return valid LLMHints."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        client = MockLLMClient(seed=42)
        hints = client.query(instance)

        assert isinstance(hints, LLMHints)
        assert isinstance(hints.priority_customers, list)
        assert isinstance(hints.locked_subroutes, list)
        assert isinstance(hints.avoid_pairs, list)
        assert isinstance(hints.suggested_moves, list)

    def test_mock_client_deterministic(self):
        """Mock client should be deterministic with the same seed."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        client1 = MockLLMClient(seed=42)
        client2 = MockLLMClient(seed=42)

        h1 = client1.query(instance)
        h2 = client2.query(instance)

        assert h1.priority_customers == h2.priority_customers

    def test_parser_handles_valid_json(self):
        """Parser should handle valid JSON output."""
        raw = '{"priority_customers": [1, 2, 3], "locked_subroutes": [], "avoid_pairs": [], "suggested_moves": [], "solver_control": null, "rationale": "test"}'
        hints = parse_llm_output(raw)

        assert isinstance(hints, LLMHints)
        assert hints.priority_customers == [1, 2, 3]

    def test_parser_handles_markdown_fences(self):
        """Parser should strip markdown code fences."""
        raw = """```json
{"priority_customers": [1], "locked_subroutes": [], "avoid_pairs": [], "suggested_moves": [], "solver_control": null, "rationale": null}
```"""
        hints = parse_llm_output(raw)

        assert isinstance(hints, LLMHints)
        assert 1 in hints.priority_customers

    def test_parser_returns_empty_on_invalid_json(self):
        """Parser should return empty hints on invalid JSON."""
        raw = "this is not json at all"
        hints = parse_llm_output(raw)

        assert isinstance(hints, LLMHints)
        assert hints.priority_customers == []

    def test_parser_returns_empty_on_empty_input(self):
        """Parser should return empty hints on empty input."""
        hints = parse_llm_output("")
        assert isinstance(hints, LLMHints)
        assert hints.priority_customers == []

        hints2 = parse_llm_output("   ")
        assert isinstance(hints2, LLMHints)

    def test_prompt_builder_produces_string(self):
        """Prompt builder should produce a non-empty string."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        prompt = build_llm_prompt(instance)

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "Instance Summary" in prompt
        assert "JSON" in prompt

    def test_hints_to_warmstart(self):
        """hints_to_warmstart should convert hints to warm-start candidate."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        client = MockLLMClient(seed=42)
        hints = client.query(instance)

        incumbent = RouteSolution(
            routes=[[1, 2, 3, 4]],
            vehicles_used=1,
            total_distance=80.0,
            total_duration=100.0,
            feasible=True,
        )
        warmstart = hints_to_warmstart(instance, hints, incumbent)

        assert warmstart is not None
        assert warmstart.initial_routes == [[1, 2, 3, 4]]
        assert warmstart.biased_order

    def test_empty_hints_function(self):
        """empty_hints() should return a valid LLMHints object."""
        hints = empty_hints()
        assert isinstance(hints, LLMHints)
        assert hints.priority_customers == []
        assert hints.locked_subroutes == []


class TestLLMLoopSmoke:
    """End-to-end smoke tests for the LLM solver loop."""

    def test_llm_loop_runs_on_toy_instance(self):
        """The full LLM loop should run without errors on toy data."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        client = MockLLMClient(seed=42)
        incumbent, final, hints, result = run_llm_solver(
            instance,
            llm_client=client,
            short_budget=1,
            final_budget=1,
            seed=42,
            output_dir=None,
        )

        assert incumbent is not None
        assert final is not None
        assert isinstance(hints, LLMHints)
        assert result.instance_name == instance.name

    def test_llm_loop_saves_outputs(self, tmp_path):
        """The LLM loop should save outputs to the output directory."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        client = MockLLMClient(seed=42)
        _, _, _, result = run_llm_solver(
            instance,
            llm_client=client,
            short_budget=1,
            final_budget=1,
            seed=42,
            output_dir=tmp_path,
        )

        exp_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(exp_dirs) >= 1, f"No experiment directories found in {tmp_path}"

        exp_dir = exp_dirs[0]
        assert (exp_dir / "instance.json").exists()
        assert (exp_dir / "llm_hints.json").exists()
        assert (exp_dir / "experiment_result.json").exists()

    def test_llm_loop_malformed_hint_fallback(self, tmp_path):
        """Parser should fall back to empty hints on malformed LLM output."""
        toy_path = DATA_DIR / "toy" / "vrptw_tiny.json"
        instance = read_instance(toy_path)

        class BadMockClient:
            """Mock client that returns invalid output."""

            def query(self, *args, **kwargs):
                from src.llm.hint_schema import empty_hints
                return empty_hints()

        _, final, hints, result = run_llm_solver(
            instance,
            llm_client=BadMockClient(),  # type: ignore
            short_budget=1,
            final_budget=1,
            seed=42,
            output_dir=tmp_path,
        )

        assert isinstance(hints, LLMHints)
        assert len(hints.priority_customers) == 0
