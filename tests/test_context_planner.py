"""
Tests for ContextPlanner.

All tests use a mock LLM — no real API calls.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tpca.config import TPCAConfig
from tpca.logging.log_config import LogConfig
from tpca.logging.structured_logger import StructuredLogger
from tpca.models.slice import SliceRequest
from tpca.pass2.context_planner import ContextPlanner
from tpca.llm.client import LLMClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_config():
    return TPCAConfig(
        reader_model="claude-haiku-4-5-20251001",
        max_planner_retries=3,
        model_context_window=8192,
        context_budget_pct=0.70,
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
    )


def make_logger(config):
    return StructuredLogger(config.log)


def make_mock_llm(response_text: str):
    """Create a mock LLMClient that returns a fixed response."""
    mock = MagicMock(spec=LLMClient)
    mock.complete.return_value = response_text
    mock.token_counter = MagicMock()
    mock.token_counter.count.return_value = 100
    mock.token_counter.count_messages.return_value = 200
    return mock


def make_mock_graph(node_ids: list, tiers: dict = None):
    """Create a minimal mock NetworkX graph."""
    import networkx as nx
    g = nx.DiGraph()
    for nid in node_ids:
        tier = (tiers or {}).get(nid, "SUPPORT")
        g.add_node(nid, tier=tier, pagerank=0.1, symbol=None)
    return g


VALID_PLAN = {
    "primary_symbols": [
        "tests/fixtures/sample_codebase/auth.py::Auth.validate_token"
    ],
    "supporting_symbols": [
        "tests/fixtures/sample_codebase/auth.py::Auth.__init__"
    ],
    "rationale": "Need validate_token to understand the auth flow.",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestContextPlannerBasic:

    def test_returns_slice_request_on_valid_response(self):
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(json.dumps(VALID_PLAN))
        node_ids = list(set(
            VALID_PLAN["primary_symbols"] + VALID_PLAN["supporting_symbols"]
        ))
        graph = make_mock_graph(node_ids)

        planner = ContextPlanner(config, logger, mock_llm)
        result = planner.plan(
            task="Explain the token validation flow.",
            compact_index="## auth.py\nclass Auth\n  + validate_token(...)",
            graph=graph,
        )

        assert isinstance(result, SliceRequest)
        assert result.primary_symbols == VALID_PLAN["primary_symbols"]
        assert result.supporting_symbols == VALID_PLAN["supporting_symbols"]
        assert "validate_token" in result.rationale

    def test_single_llm_call_on_valid_response(self):
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(json.dumps(VALID_PLAN))
        node_ids = list(set(
            VALID_PLAN["primary_symbols"] + VALID_PLAN["supporting_symbols"]
        ))
        graph = make_mock_graph(node_ids)

        planner = ContextPlanner(config, logger, mock_llm)
        planner.plan(
            task="Explain token validation.",
            compact_index="index",
            graph=graph,
        )

        assert mock_llm.complete.call_count == 1

    def test_handles_markdown_fenced_json(self):
        """LLM sometimes wraps JSON in markdown fences — should still parse."""
        config = make_config()
        logger = make_logger(config)
        fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
        mock_llm = make_mock_llm(fenced)
        node_ids = list(set(
            VALID_PLAN["primary_symbols"] + VALID_PLAN["supporting_symbols"]
        ))
        graph = make_mock_graph(node_ids)

        planner = ContextPlanner(config, logger, mock_llm)
        result = planner.plan("task", "index", graph)

        assert result.primary_symbols == VALID_PLAN["primary_symbols"]

    def test_empty_response_returns_empty_slice_request(self):
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm("{}")  # empty but valid JSON
        graph = make_mock_graph([])

        planner = ContextPlanner(config, logger, mock_llm)
        result = planner.plan("task", "index", graph)

        assert result.primary_symbols == []
        assert result.supporting_symbols == []


class TestContextPlannerRetry:

    def test_retries_on_invalid_symbol_ids(self):
        """Unknown symbol IDs should trigger a retry."""
        config = make_config()
        logger = make_logger(config)

        bad_plan = {
            "primary_symbols": ["nonexistent::Symbol"],
            "supporting_symbols": [],
            "rationale": "needs this",
        }
        good_plan = {
            "primary_symbols": ["auth.py::Auth.validate_token"],
            "supporting_symbols": [],
            "rationale": "corrected",
        }

        # First call returns bad plan, second returns good plan
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.side_effect = [
            json.dumps(bad_plan),
            json.dumps(good_plan),
        ]
        mock_llm.token_counter = MagicMock()
        mock_llm.token_counter.count.return_value = 10
        mock_llm.token_counter.count_messages.return_value = 50

        graph = make_mock_graph(["auth.py::Auth.validate_token"])
        planner = ContextPlanner(config, logger, mock_llm)
        result = planner.plan("task", "index", graph)

        assert mock_llm.complete.call_count == 2
        assert result.primary_symbols == ["auth.py::Auth.validate_token"]

    def test_fallback_after_max_retries(self):
        """After max_retries, should fall back to top CORE symbols."""
        config = make_config()
        config.max_planner_retries = 2
        logger = make_logger(config)

        always_bad = json.dumps({
            "primary_symbols": ["definitely::NotReal"],
            "supporting_symbols": [],
            "rationale": "wrong",
        })
        mock_llm = make_mock_llm(always_bad)
        mock_llm.complete = MagicMock(return_value=always_bad)
        mock_llm.token_counter = MagicMock()
        mock_llm.token_counter.count.return_value = 10
        mock_llm.token_counter.count_messages.return_value = 50

        core_ids = [f"auth.py::Auth.method{i}" for i in range(5)]
        graph = make_mock_graph(
            core_ids,
            tiers={nid: "CORE" for nid in core_ids}
        )
        # Set pagerank on nodes
        for i, nid in enumerate(core_ids):
            graph.nodes[nid]["pagerank"] = 0.1 * (i + 1)

        planner = ContextPlanner(config, logger, mock_llm)
        result = planner.plan("task", "index", graph)

        # Should have fallen back to CORE symbols
        assert len(result.primary_symbols) > 0
        assert all(s in core_ids for s in result.primary_symbols)

    def test_retry_emits_warn_event(self):
        """Retry should emit a planner_retry log event."""
        config = make_config()
        logger = make_logger(config)

        bad_plan = json.dumps({
            "primary_symbols": ["bad::Symbol"],
            "supporting_symbols": [],
            "rationale": "bad",
        })
        good_plan = json.dumps({
            "primary_symbols": ["real::Symbol"],
            "supporting_symbols": [],
            "rationale": "good",
        })

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.side_effect = [bad_plan, good_plan]
        mock_llm.token_counter = MagicMock()
        mock_llm.token_counter.count.return_value = 10
        mock_llm.token_counter.count_messages.return_value = 50

        graph = make_mock_graph(["real::Symbol"])
        planner = ContextPlanner(config, logger, mock_llm)
        planner.plan("task", "index", graph)

        warn_events = logger.get_events("planner_retry")
        assert len(warn_events) >= 1
        assert warn_events[0]["attempt"] == 1


class TestContextPlannerBudget:

    def test_budget_passed_to_prompt(self):
        """The budget_tokens value should be included in the prompt sent to LLM."""
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(json.dumps(VALID_PLAN))
        mock_llm.complete = MagicMock(return_value=json.dumps(VALID_PLAN))
        mock_llm.token_counter = MagicMock()
        mock_llm.token_counter.count.return_value = 10
        mock_llm.token_counter.count_messages.return_value = 50

        graph = make_mock_graph(list(set(
            VALID_PLAN["primary_symbols"] + VALID_PLAN["supporting_symbols"]
        )))
        planner = ContextPlanner(config, logger, mock_llm)
        planner.plan("task", "index", graph, budget_tokens=1234)

        call_args = mock_llm.complete.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "1234" in prompt_text

    def test_default_budget_derived_from_config(self):
        config = make_config()
        config.model_context_window = 10000
        config.context_budget_pct = 0.5
        logger = make_logger(config)
        mock_llm = make_mock_llm(json.dumps(VALID_PLAN))
        mock_llm.complete = MagicMock(return_value=json.dumps(VALID_PLAN))
        mock_llm.token_counter = MagicMock()
        mock_llm.token_counter.count.return_value = 10
        mock_llm.token_counter.count_messages.return_value = 50

        graph = make_mock_graph(list(set(
            VALID_PLAN["primary_symbols"] + VALID_PLAN["supporting_symbols"]
        )))
        planner = ContextPlanner(config, logger, mock_llm)
        planner.plan("task", "index", graph)

        call_args = mock_llm.complete.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        # Default budget: 10000 * 0.5 = 5000
        assert "5000" in prompt_text
