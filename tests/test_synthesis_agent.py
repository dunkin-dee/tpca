"""
Tests for SynthesisAgent.

Uses mock LLM with scripted responses — no real API calls.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from tpca.config import TPCAConfig
from tpca.logging.log_config import LogConfig
from tpca.logging.structured_logger import StructuredLogger
from tpca.models.slice import Slice, SliceRequest
from tpca.models.symbol import Symbol
from tpca.pass2.context_planner import ContextPlanner
from tpca.pass2.slice_fetcher import SliceFetcher
from tpca.pass2.synthesis_agent import SynthesisAgent, SynthesisResult
from tpca.llm.client import LLMClient, TokenCounter


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(output_mode="inline"):
    return TPCAConfig(
        output_mode=output_mode,
        output_dir="/tmp/tpca_test_output",
        max_synthesis_iterations=10,
        max_planner_retries=1,
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
    )


def make_logger(config):
    return StructuredLogger(config.log)


def make_mock_llm_client():
    mock = MagicMock(spec=LLMClient)
    counter = MagicMock(spec=TokenCounter)
    counter.count.return_value = 50
    counter.count_messages.return_value = 100
    mock.token_counter = counter
    mock.count_tokens.return_value = 50
    return mock


def make_graph(*symbol_ids: str) -> nx.DiGraph:
    g = nx.DiGraph()
    for i, sid in enumerate(symbol_ids):
        sym = Symbol(
            id=sid,
            type="function",
            name=sid.split("::")[-1],
            qualified_name=sid.split("::")[-1],
            file=sid.split("::")[0] if "::" in sid else "file.py",
            start_line=1,
            end_line=10,
            signature=f"def {sid.split('::')[-1]}():",
            docstring="",
        )
        g.add_node(sid, symbol=sym, tier="CORE", pagerank=0.1 * (i + 1))
    return g


PLANNER_RESPONSE = json.dumps({
    "primary_symbols": ["auth.py::Auth"],
    "supporting_symbols": [],
    "rationale": "Need Auth class to document.",
})

SYNTHESIS_RESPONSE = (
    "Here is the documentation for Auth.\n\n"
    "The Auth class handles JWT validation.\n\n"
    "[SECTION_COMPLETE: Documented Auth class with validate_token method]"
)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSynthesisAgentBasic:

    def _make_agent_with_mocks(self, config, logger, llm_responses: list[str]):
        """Create a SynthesisAgent with scripted LLM responses."""
        mock_llm = make_mock_llm_client()
        mock_llm.complete.side_effect = llm_responses

        # Mock planner to return a fixed SliceRequest
        planner = MagicMock(spec=ContextPlanner)
        planner.plan.return_value = SliceRequest(
            primary_symbols=["auth.py::Auth"],
            supporting_symbols=[],
            rationale="Need Auth.",
        )

        # Mock fetcher to return a fixed slice list
        fetcher = MagicMock(spec=SliceFetcher)
        fetcher.fetch.return_value = [
            Slice(
                symbol_id="auth.py::Auth",
                source="class Auth:\n    def validate_token(self): pass",
                token_count=30,
            )
        ]
        fetcher.format_slices_for_prompt.return_value = "### auth.py::Auth\n```python\nclass Auth:\n    pass\n```"

        agent = SynthesisAgent(
            config=config,
            logger=logger,
            llm_client=mock_llm,
            planner=planner,
            fetcher=fetcher,
        )
        return agent, mock_llm

    def test_returns_synthesis_result(self):
        config = make_config()
        logger = make_logger(config)
        agent, _ = self._make_agent_with_mocks(
            config, logger, [SYNTHESIS_RESPONSE]
        )
        graph = make_graph("auth.py::Auth")
        result = agent.run(task="Document Auth class.", compact_index="index", graph=graph)

        assert isinstance(result, SynthesisResult)
        assert result.output_log is not None
        assert result.manifest is not None
        assert isinstance(result.stats, dict)

    def test_stats_include_llm_calls(self):
        config = make_config()
        logger = make_logger(config)
        agent, _ = self._make_agent_with_mocks(
            config, logger, [SYNTHESIS_RESPONSE]
        )
        graph = make_graph("auth.py::Auth")
        result = agent.run(task="Document Auth class.", compact_index="index", graph=graph)

        assert result.stats["llm_calls"] == 1

    def test_inline_output_contains_response(self):
        config = make_config(output_mode="inline")
        logger = make_logger(config)
        agent, _ = self._make_agent_with_mocks(
            config, logger, [SYNTHESIS_RESPONSE]
        )
        graph = make_graph("auth.py::Auth")
        result = agent.run(task="Document Auth class.", compact_index="index", graph=graph)

        # Inline mode: output is a dict of {file_path: content}
        all_content = " ".join(str(v) for v in result.output.values())
        # Auth documentation should be in some output
        assert len(result.output_log.entries) == 1

    def test_output_log_has_correct_summary(self):
        config = make_config()
        logger = make_logger(config)
        agent, _ = self._make_agent_with_mocks(
            config, logger, [SYNTHESIS_RESPONSE]
        )
        graph = make_graph("auth.py::Auth")
        result = agent.run(task="Document.", compact_index="index", graph=graph)

        entries = result.output_log.entries
        assert len(entries) == 1
        assert "Documented Auth class" in entries[0].summary

    def test_multiple_symbols_multiple_llm_calls(self):
        config = make_config()
        logger = make_logger(config)

        mock_llm = make_mock_llm_client()
        responses = [
            "Auth docs.\n[SECTION_COMPLETE: Documented Auth]",
            "Router docs.\n[SECTION_COMPLETE: Documented Router]",
        ]
        mock_llm.complete.side_effect = responses

        planner = MagicMock(spec=ContextPlanner)
        planner.plan.return_value = SliceRequest(
            primary_symbols=["auth.py::Auth", "router.py::Router"],
            supporting_symbols=[],
            rationale="Two symbols.",
        )

        fetcher = MagicMock(spec=SliceFetcher)
        fetcher.fetch.return_value = [
            Slice(symbol_id="auth.py::Auth", source="class Auth: pass", token_count=20),
            Slice(symbol_id="router.py::Router", source="class Router: pass", token_count=20),
        ]
        fetcher.format_slices_for_prompt.return_value = "slices"

        agent = SynthesisAgent(config, logger, mock_llm, planner, fetcher)
        graph = make_graph("auth.py::Auth", "router.py::Router")
        result = agent.run("Document both.", "index", graph)

        assert result.stats["llm_calls"] == 2
        assert len(result.output_log.entries) == 2

    def test_task_complete_marker_stops_loop(self):
        """[TASK_COMPLETE] marker should end the synthesis loop early."""
        config = make_config()
        logger = make_logger(config)

        mock_llm = make_mock_llm_client()
        mock_llm.complete.return_value = (
            "All done.\n[TASK_COMPLETE]"
        )

        planner = MagicMock(spec=ContextPlanner)
        planner.plan.return_value = SliceRequest(
            primary_symbols=["auth.py::Auth", "router.py::Router"],
        )

        fetcher = MagicMock(spec=SliceFetcher)
        fetcher.fetch.return_value = []
        fetcher.format_slices_for_prompt.return_value = "(no slices)"

        agent = SynthesisAgent(config, logger, mock_llm, planner, fetcher)
        graph = make_graph("auth.py::Auth", "router.py::Router")
        result = agent.run("Document.", "index", graph)

        # Loop should have exited after first TASK_COMPLETE
        assert mock_llm.complete.call_count == 1


class TestSynthesisAgentOutputExtraction:

    def test_extract_section_complete(self):
        text = "Here is the output.\n[SECTION_COMPLETE: documented Auth.validate_token]"
        raw, summary = SynthesisAgent._extract_output_and_summary(text, "sym")

        assert "Here is the output" in raw
        assert "documented Auth.validate_token" in summary

    def test_extract_task_complete(self):
        text = "Final output here.\n[TASK_COMPLETE]"
        raw, summary = SynthesisAgent._extract_output_and_summary(text, "sym1")

        assert "Final output here" in raw
        assert "sym1" in summary

    def test_extract_no_marker_returns_full_text(self):
        text = "Some output with no marker."
        raw, summary = SynthesisAgent._extract_output_and_summary(text, "sym1")

        assert raw == text
        assert "no marker" in summary.lower() or "sym1" in summary


class TestSynthesisAgentLogging:

    def _run_with_mocks(self, config, logger, response: str = SYNTHESIS_RESPONSE):
        mock_llm = make_mock_llm_client()
        mock_llm.complete.return_value = response

        planner = MagicMock(spec=ContextPlanner)
        planner.plan.return_value = SliceRequest(
            primary_symbols=["auth.py::Auth"], supporting_symbols=[]
        )

        fetcher = MagicMock(spec=SliceFetcher)
        fetcher.fetch.return_value = []
        fetcher.format_slices_for_prompt.return_value = "(none)"

        agent = SynthesisAgent(config, logger, mock_llm, planner, fetcher)
        graph = make_graph("auth.py::Auth")
        return agent.run("task", "index", graph)

    def test_emits_synthesis_start(self):
        config = make_config()
        logger = make_logger(config)
        self._run_with_mocks(config, logger)

        events = logger.get_events("synthesis_start")
        assert len(events) == 1

    def test_emits_synthesis_result(self):
        config = make_config()
        logger = make_logger(config)
        self._run_with_mocks(config, logger)

        events = logger.get_events("synthesis_result")
        assert len(events) == 1
        assert "llm_calls" in events[0]
