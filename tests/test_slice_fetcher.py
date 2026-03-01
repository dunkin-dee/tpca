"""
Tests for SliceFetcher.

All tests use a mock LLMClient for token counting.
"""
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
from tpca.pass2.slice_fetcher import SliceFetcher
from tpca.llm.client import LLMClient, TokenCounter


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(**kwargs):
    defaults = dict(
        model_context_window=8192,
        context_budget_pct=0.70,
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
    )
    defaults.update(kwargs)
    return TPCAConfig(**defaults)


def make_logger(config):
    return StructuredLogger(config.log)


def make_mock_llm(tokens_per_call=50):
    """Return a mock LLMClient with a fixed token count."""
    mock = MagicMock(spec=LLMClient)
    counter = MagicMock(spec=TokenCounter)
    counter.count.return_value = tokens_per_call
    mock.token_counter = counter
    return mock


def make_symbol(
    sym_id: str,
    file: str = "auth.py",
    start_line: int = 1,
    end_line: int = 10,
    signature: str = "def validate_token(self, token: str) -> bool:",
    docstring: str = "Validates a JWT token.",
) -> Symbol:
    parts = sym_id.split("::")
    name = parts[-1] if parts else sym_id
    return Symbol(
        id=sym_id,
        type="method",
        name=name,
        qualified_name=name,
        file=file,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring,
    )


def make_graph_with_symbols(symbols: list) -> nx.DiGraph:
    g = nx.DiGraph()
    for sym in symbols:
        g.add_node(sym.id, symbol=sym, tier="CORE", pagerank=0.1)
    return g


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSliceFetcherBasic:

    def test_returns_slices_for_primary_symbols(self, tmp_path):
        """Should return a Slice for each valid primary symbol."""
        source_file = tmp_path / "auth.py"
        source_file.write_text("\n".join([f"line{i}" for i in range(20)]))

        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(tokens_per_call=50)

        sym_id = f"{source_file}::Auth.validate_token"
        sym = make_symbol(sym_id, file=str(source_file), start_line=1, end_line=5)
        graph = make_graph_with_symbols([sym])

        request = SliceRequest(primary_symbols=[sym_id], supporting_symbols=[])
        fetcher = SliceFetcher(config, logger, mock_llm)
        slices = fetcher.fetch(request, graph, budget=5000)

        assert len(slices) == 1
        assert slices[0].symbol_id == sym_id
        assert not slices[0].truncated

    def test_missing_symbol_skipped_with_warning(self):
        """Symbols not in the graph should be skipped gracefully."""
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm()

        graph = nx.DiGraph()  # empty graph
        request = SliceRequest(
            primary_symbols=["missing::Symbol"],
            supporting_symbols=[],
        )

        fetcher = SliceFetcher(config, logger, mock_llm)
        slices = fetcher.fetch(request, graph, budget=5000)

        assert slices == []
        warn_events = logger.get_events("slice_symbol_missing")
        assert len(warn_events) == 1

    def test_truncates_primary_when_over_budget(self, tmp_path):
        """Primary symbols that exceed budget should be truncated to signature-only."""
        source_file = tmp_path / "big.py"
        # Write a large file
        source_file.write_text("\n".join([f"# line {i}" for i in range(1000)]))

        config = make_config()
        logger = make_logger(config)

        # Token counter: full source = 9000, signature = 20
        counter = MagicMock()
        call_counts = {"n": 0}
        def count_side_effect(text):
            call_counts["n"] += 1
            # Return large count for first call (full source), small for second (sig)
            if call_counts["n"] == 1:
                return 9000
            return 20
        counter.count.side_effect = count_side_effect
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.token_counter = counter

        sym_id = f"{source_file}::BigClass"
        sym = make_symbol(sym_id, file=str(source_file), start_line=1, end_line=100)
        graph = make_graph_with_symbols([sym])
        request = SliceRequest(primary_symbols=[sym_id], supporting_symbols=[])

        fetcher = SliceFetcher(config, logger, mock_llm)
        slices = fetcher.fetch(request, graph, budget=100)  # tiny budget

        assert len(slices) == 1
        assert slices[0].truncated is True
        warn_events = logger.get_events("slice_truncated")
        assert len(warn_events) == 1

    def test_supporting_symbols_included_greedily(self, tmp_path):
        """Supporting symbols are included while budget permits."""
        source_file = tmp_path / "utils.py"
        source_file.write_text("\n".join([f"line{i}" for i in range(10)]))

        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(tokens_per_call=100)  # each slice = 100 tokens

        syms = [
            make_symbol(f"{source_file}::Func{i}", file=str(source_file))
            for i in range(3)
        ]
        graph = make_graph_with_symbols(syms)

        # Primary: syms[0], Supporting: syms[1], syms[2]
        request = SliceRequest(
            primary_symbols=[syms[0].id],
            supporting_symbols=[syms[1].id, syms[2].id],
        )

        fetcher = SliceFetcher(config, logger, mock_llm)
        # Budget: 350 tokens — room for primary (100) + syms[1] (100) but not syms[2] (100+200 min)
        slices = fetcher.fetch(request, graph, budget=350)

        symbol_ids = [s.symbol_id for s in slices]
        assert syms[0].id in symbol_ids
        assert syms[1].id in symbol_ids
        assert syms[2].id not in symbol_ids

    def test_supporting_symbols_skipped_when_budget_exhausted(self, tmp_path):
        """Supporting symbols should be skipped when budget is exhausted."""
        source_file = tmp_path / "utils.py"
        source_file.write_text("pass\n")

        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(tokens_per_call=500)

        syms = [
            make_symbol(f"{source_file}::Func{i}", file=str(source_file))
            for i in range(2)
        ]
        graph = make_graph_with_symbols(syms)

        request = SliceRequest(
            primary_symbols=[syms[0].id],
            supporting_symbols=[syms[1].id],
        )

        fetcher = SliceFetcher(config, logger, mock_llm)
        slices = fetcher.fetch(request, graph, budget=600)  # only room for primary

        assert len(slices) == 1
        assert slices[0].symbol_id == syms[0].id


class TestSliceFetcherFormatting:

    def test_format_slices_for_prompt_includes_symbol_ids(self, tmp_path):
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm()

        fetcher = SliceFetcher(config, logger, mock_llm)
        slices = [
            Slice(symbol_id="auth.py::Auth", source="class Auth:\n    pass", token_count=10),
            Slice(symbol_id="utils.py::hash_pw", source="def hash_pw(s):\n    pass", token_count=8),
        ]
        formatted = fetcher.format_slices_for_prompt(slices)

        assert "auth.py::Auth" in formatted
        assert "utils.py::hash_pw" in formatted
        assert "class Auth:" in formatted
        assert "def hash_pw" in formatted

    def test_format_truncated_slice_shows_marker(self):
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm()

        fetcher = SliceFetcher(config, logger, mock_llm)
        slices = [
            Slice(
                symbol_id="auth.py::Auth",
                source="def validate_token(self):",
                token_count=10,
                truncated=True,
            )
        ]
        formatted = fetcher.format_slices_for_prompt(slices)
        assert "SIGNATURE ONLY" in formatted or "TRUNCATED" in formatted

    def test_format_empty_slices(self):
        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm()

        fetcher = SliceFetcher(config, logger, mock_llm)
        formatted = fetcher.format_slices_for_prompt([])
        assert "no source slices" in formatted.lower()


class TestSliceFetcherLogging:

    def test_emits_slice_fetch_start_event(self, tmp_path):
        source_file = tmp_path / "auth.py"
        source_file.write_text("class Auth:\n    pass\n")

        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(tokens_per_call=10)

        sym = make_symbol(f"{source_file}::Auth", file=str(source_file))
        graph = make_graph_with_symbols([sym])
        request = SliceRequest(primary_symbols=[sym.id], supporting_symbols=[])

        fetcher = SliceFetcher(config, logger, mock_llm)
        fetcher.fetch(request, graph, budget=5000)

        start_events = logger.get_events("slice_fetch_start")
        assert len(start_events) == 1
        assert start_events[0]["budget_tokens"] == 5000

    def test_emits_slice_fetch_complete_event(self, tmp_path):
        source_file = tmp_path / "auth.py"
        source_file.write_text("class Auth:\n    pass\n")

        config = make_config()
        logger = make_logger(config)
        mock_llm = make_mock_llm(tokens_per_call=10)

        sym = make_symbol(f"{source_file}::Auth", file=str(source_file))
        graph = make_graph_with_symbols([sym])
        request = SliceRequest(primary_symbols=[sym.id], supporting_symbols=[])

        fetcher = SliceFetcher(config, logger, mock_llm)
        fetcher.fetch(request, graph, budget=5000)

        complete_events = logger.get_events("slice_fetch_complete")
        assert len(complete_events) == 1
        assert complete_events[0]["slices_fetched"] == 1
