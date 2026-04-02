"""
Tests for OutputChunker.

Verifies topological ordering, OutputLog management, resumability,
and that completed symbols are never re-processed.
"""
import sys
from pathlib import Path
import pytest
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.config import PRISMConfig
from prism.logging.log_config import LogConfig
from prism.logging.structured_logger import StructuredLogger
from prism.models.output import OutputLog, OutputChunk
from prism.models.slice import SliceRequest
from prism.models.chunk_plan import ChunkPlan
from prism.pass2.output_chunker import OutputChunker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config():
    return PRISMConfig(
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
    )


def make_logger(config):
    return StructuredLogger(config.log)


def make_linear_graph(*node_ids: str) -> nx.DiGraph:
    """A → B → C (dependency chain)."""
    g = nx.DiGraph()
    for nid in node_ids:
        g.add_node(nid, tier="CORE", pagerank=0.1, symbol=None)
    for i in range(len(node_ids) - 1):
        g.add_edge(node_ids[i], node_ids[i + 1], type="calls", weight=1)
    return g


def make_independent_graph(*node_ids: str) -> nx.DiGraph:
    """Nodes with no edges (all independent)."""
    g = nx.DiGraph()
    for i, nid in enumerate(node_ids):
        g.add_node(nid, tier="CORE", pagerank=0.1 * (i + 1), symbol=None)
    return g


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOutputChunkerBasic:

    def test_returns_none_when_empty(self):
        """No symbols → first get_next_chunk call returns None."""
        config = make_config()
        logger = make_logger(config)
        graph = nx.DiGraph()
        chunker = OutputChunker(graph, OutputLog(), config, logger)

        result = chunker.get_next_chunk(compact_index="", slices_by_symbol={})
        assert result is None

    def test_returns_chunk_plan_for_single_symbol(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("auth.py::Auth")

        request = SliceRequest(primary_symbols=["auth.py::Auth"], supporting_symbols=[])
        chunker = OutputChunker(graph, OutputLog(), config, logger, request)

        plan = chunker.get_next_chunk(compact_index="index", slices_by_symbol={})
        assert plan is not None
        assert isinstance(plan, ChunkPlan)
        assert plan.symbol_id == "auth.py::Auth"
        assert plan.chunk_id == 0

    def test_second_call_returns_none_after_completion(self):
        """After recording the chunk, get_next_chunk should return None."""
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("auth.py::Auth")
        request = SliceRequest(primary_symbols=["auth.py::Auth"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        plan = chunker.get_next_chunk(compact_index="idx", slices_by_symbol={})
        assert plan is not None

        chunker.record_chunk(plan.chunk_id, plan.symbol_id, "output", "summary")

        plan2 = chunker.get_next_chunk(compact_index="idx", slices_by_symbol={})
        assert plan2 is None

    def test_chunk_id_increments(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1", "sym2", "sym3")
        request = SliceRequest(primary_symbols=["sym1", "sym2", "sym3"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        ids = []
        for _ in range(3):
            plan = chunker.get_next_chunk("idx", {})
            assert plan is not None
            ids.append(plan.chunk_id)
            chunker.record_chunk(plan.chunk_id, plan.symbol_id, "out", "sum")

        assert ids == [0, 1, 2]


class TestOutputChunkerOrdering:

    def test_processes_symbols_in_topo_order(self):
        """In a dependency chain A → B → C, A should be processed before B."""
        config = make_config()
        logger = make_logger(config)
        graph = make_linear_graph("A", "B", "C")
        request = SliceRequest(primary_symbols=["A", "B", "C"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        processed = []
        for _ in range(3):
            plan = chunker.get_next_chunk("idx", {})
            if plan is None:
                break
            processed.append(plan.symbol_id)
            chunker.record_chunk(plan.chunk_id, plan.symbol_id, "out", "sum")

        # Topological order: A must come before B, B before C
        assert processed.index("A") < processed.index("B")
        assert processed.index("B") < processed.index("C")


class TestOutputChunkerOutputLog:

    def test_output_log_updated_after_record(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1")
        request = SliceRequest(primary_symbols=["sym1"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        plan = chunker.get_next_chunk("idx", {})
        chunker.record_chunk(plan.chunk_id, plan.symbol_id, "output text", "did the thing")

        assert len(log.entries) == 1
        assert log.entries[0].symbol_id == "sym1"
        assert log.entries[0].summary == "did the thing"
        assert log.entries[0].status == "complete"

    def test_prior_log_included_in_subsequent_chunk(self):
        """Second chunk plan should include the prior OutputLog text."""
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1", "sym2")
        request = SliceRequest(primary_symbols=["sym1", "sym2"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        plan1 = chunker.get_next_chunk("idx", {})
        chunker.record_chunk(plan1.chunk_id, plan1.symbol_id, "out1", "summary one")

        plan2 = chunker.get_next_chunk("idx", {})
        assert plan2 is not None
        assert "summary one" in plan2.prior_log

    def test_output_log_render_compact(self):
        log = OutputLog()
        log.add(OutputChunk(
            chunk_id=0,
            symbol_id="auth.py::Auth",
            summary="Documented validate_token and refresh_token.",
            status="complete",
            token_count=200,
        ))
        log.add(OutputChunk(
            chunk_id=1,
            symbol_id="utils.py::hash_pw",
            summary="Documented hash_pw function.",
            status="complete",
            token_count=100,
        ))

        rendered = log.render_compact()
        assert "COMPLETED WORK" in rendered
        assert "auth.py::Auth" in rendered
        assert "Documented validate_token" in rendered
        assert "hash_pw" in rendered

    def test_completed_symbols_returns_set(self):
        log = OutputLog()
        log.add(OutputChunk(0, "sym1", "done", "complete", 50))
        log.add(OutputChunk(1, "sym2", "done", "complete", 50))

        assert log.completed_symbols() == {"sym1", "sym2"}


class TestOutputChunkerResumability:

    def test_already_completed_symbols_skipped(self):
        """Symbols already in the OutputLog should not be re-processed."""
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1", "sym2")
        request = SliceRequest(primary_symbols=["sym1", "sym2"])

        # Pre-populate log as if sym1 was already completed
        log = OutputLog()
        log.add(OutputChunk(0, "sym1", "previously done", "complete", 100))

        chunker = OutputChunker(graph, log, config, logger, request)
        plan = chunker.get_next_chunk("idx", {})

        # sym1 is already complete — should get sym2
        assert plan is not None
        assert plan.symbol_id == "sym2"

    def test_is_complete_false_when_work_remains(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1", "sym2")
        request = SliceRequest(primary_symbols=["sym1", "sym2"])
        chunker = OutputChunker(graph, OutputLog(), config, logger, request)

        assert not chunker.is_complete()

    def test_is_complete_true_after_all_recorded(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1")
        request = SliceRequest(primary_symbols=["sym1"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        plan = chunker.get_next_chunk("idx", {})
        chunker.record_chunk(plan.chunk_id, plan.symbol_id, "out", "sum")

        assert chunker.is_complete()

    def test_total_count_and_completed_count(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1", "sym2", "sym3")
        request = SliceRequest(primary_symbols=["sym1", "sym2", "sym3"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        assert chunker.total_count == 3
        assert chunker.completed_count == 0

        plan = chunker.get_next_chunk("idx", {})
        chunker.record_chunk(plan.chunk_id, plan.symbol_id, "out", "sum")

        assert chunker.completed_count == 1


class TestOutputChunkerLogging:

    def test_emits_output_chunk_start_event(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1")
        request = SliceRequest(primary_symbols=["sym1"])
        chunker = OutputChunker(graph, OutputLog(), config, logger, request)

        chunker.get_next_chunk("idx", {})

        events = logger.get_events("output_chunk_start")
        assert len(events) == 1
        assert events[0]["symbol_id"] == "sym1"

    def test_emits_output_chunk_end_event(self):
        config = make_config()
        logger = make_logger(config)
        graph = make_independent_graph("sym1")
        request = SliceRequest(primary_symbols=["sym1"])
        log = OutputLog()
        chunker = OutputChunker(graph, log, config, logger, request)

        plan = chunker.get_next_chunk("idx", {})
        chunker.record_chunk(plan.chunk_id, plan.symbol_id, "output text", "summary here")

        events = logger.get_events("output_chunk_end")
        assert len(events) == 1
        assert events[0]["symbol_id"] == "sym1"
        assert "summary here" in events[0]["summary_preview"]
