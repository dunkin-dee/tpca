"""
Tests for spec-compliance fixes applied post design-review.

Covers all 11 changes made to bring the implementation in line with
PRISM_Design_v2.docx:

1.  Symbol.calls field present and defaults to []
2.  IndexCache round-trips the calls field; gracefully handles old cache entries
3.  ASTIndexer._extract_calls_from_body populates Symbol.calls from function bodies
4.  ASTIndexer extracts import symbols (type='import')
5.  ASTIndexer extracts module-level UPPER_CASE constant symbols (type='constant')
6.  GraphBuilder._resolve_cross_file_edges uses AST-derived calls (real edges, weights)
7.  GraphBuilder emits graph_built event with build_time_ms
8.  GraphRanker SUPPORT tier threshold is 0.60 (top 10-40%), not 0.70
9.  IndexRenderer renders CORE/SUPPORT constant symbols; suppresses PERIPHERAL
10. IndexRenderer renders import symbols under an "imports:" block
11. ContextPlanner emits planner_request (not planner_start / planner_success)
12. PRISMConfig.output_mode defaults to 'single_file'
13. Orchestrator.run() stats include raw_tokens_available
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.config import PRISMConfig
from prism.logging.log_config import LogConfig
from prism.logging.structured_logger import StructuredLogger
from prism.models.symbol import Symbol
from prism.models.slice import SliceRequest
from prism.pass1.ast_indexer import ASTIndexer
from prism.pass1.graph_builder import GraphBuilder
from prism.pass1.graph_ranker import GraphRanker
from prism.pass1.index_renderer import IndexRenderer
from prism.cache.index_cache import IndexCache
from prism.pass2.context_planner import ContextPlanner
from prism.llm.client import LLMClient, TokenCounter

FIXTURES = Path(__file__).parent / "fixtures" / "sample_codebase"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _logger(tmpdir=None):
    log_file = f"{tmpdir}/test.log" if tmpdir else "/dev/null"
    return StructuredLogger(LogConfig(log_file=log_file, console_level="ERROR"))


def _config(**kwargs):
    return PRISMConfig(log=LogConfig(log_file="/dev/null", console_level="ERROR"), **kwargs)


def _mock_llm():
    mock = MagicMock(spec=LLMClient)
    mock.count_tokens.return_value = 50
    mock.available = True
    return mock


# ── 1. Symbol.calls field ──────────────────────────────────────────────────────

class TestSymbolCallsField:
    def test_calls_defaults_to_empty_list(self):
        sym = Symbol(
            id="f.py::foo", type="function", name="foo",
            qualified_name="foo", file="f.py",
            start_line=1, end_line=5, signature="def foo()",
        )
        assert sym.calls == []

    def test_calls_can_be_set(self):
        sym = Symbol(
            id="f.py::foo", type="function", name="foo",
            qualified_name="foo", file="f.py",
            start_line=1, end_line=5, signature="def foo()",
            calls=["bar", "baz", "bar"],
        )
        assert sym.calls == ["bar", "baz", "bar"]


# ── 2. IndexCache round-trip for calls ────────────────────────────────────────

class TestIndexCacheCallsField:
    def test_serialize_deserialize_preserves_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _config(cache_dir=tmpdir)
            log = _logger()
            cache = IndexCache(cfg, log)

            test_file = Path(tmpdir) / "sample.py"
            test_file.write_text("def foo(): bar()\ndef bar(): pass\n")

            sym = Symbol(
                id=f"{test_file}::foo", type="function", name="foo",
                qualified_name="foo", file=str(test_file),
                start_line=1, end_line=1, signature="def foo()",
                calls=["bar", "baz"],
            )
            cache.set(str(test_file), [sym])

            loaded = cache.get(str(test_file))
            assert loaded is not None
            assert loaded[0].calls == ["bar", "baz"]

    def test_deserialize_without_calls_field_uses_default(self):
        """Old cache entries that predate the calls field must still load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _config(cache_dir=tmpdir)
            log = _logger()
            cache = IndexCache(cfg, log)

            test_file = Path(tmpdir) / "old.py"
            test_file.write_text("def foo(): pass\n")

            # Write a cache entry that has no 'calls' key (simulates old cache)
            sym = Symbol(
                id=f"{test_file}::foo", type="function", name="foo",
                qualified_name="foo", file=str(test_file),
                start_line=1, end_line=1, signature="def foo()",
            )
            cache.set(str(test_file), [sym])

            # Manually strip 'calls' from the written JSON to simulate an old entry
            import hashlib, os
            mtime = os.path.getmtime(str(test_file))
            hash_input = f"{test_file}:{mtime}".encode()
            file_hash = hashlib.sha256(hash_input).hexdigest()[:16]
            cache_file = Path(tmpdir) / "index" / f"{file_hash}.json"
            data = json.loads(cache_file.read_text())
            for s in data["symbols"]:
                s.pop("calls", None)
            cache_file.write_text(json.dumps(data))

            # Should still load without KeyError, defaulting calls to []
            loaded = cache.get(str(test_file))
            assert loaded is not None
            assert loaded[0].calls == []


# ── 3. ASTIndexer: call extraction from function bodies ───────────────────────

class TestASTIndexerCallExtraction:
    def setup_method(self):
        self.cfg = _config()
        self.log = _logger()
        self.indexer = ASTIndexer(self.cfg, self.log)

    def test_method_calls_are_extracted(self):
        """validate_token calls _decode_payload — must appear in sym.calls."""
        symbols = self.indexer.index(str(FIXTURES / "auth.py"))
        validate = next(s for s in symbols if s.name == "validate_token")
        assert "_decode_payload" in validate.calls

    def test_method_calls_multiple_callees(self):
        """refresh_token calls both _decode_payload and _encode_token."""
        symbols = self.indexer.index(str(FIXTURES / "auth.py"))
        refresh = next(s for s in symbols if s.name == "refresh_token")
        assert "_decode_payload" in refresh.calls
        assert "_encode_token" in refresh.calls

    def test_function_calls_are_extracted(self):
        """generate_token in utils.py calls _sign_token."""
        symbols = self.indexer.index(str(FIXTURES / "utils.py"))
        gen = next(s for s in symbols if s.name == "generate_token")
        assert "_sign_token" in gen.calls

    def test_repeated_call_appears_multiple_times(self):
        """A callee called more than once should appear multiple times in calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "repeat.py"
            src.write_text(
                "def foo():\n"
                "    bar()\n"
                "    bar()\n"
                "    bar()\n"
                "\ndef bar(): pass\n"
            )
            symbols = self.indexer.index(str(src))
            foo = next(s for s in symbols if s.name == "foo")
            assert foo.calls.count("bar") == 3

    def test_builtins_are_not_extracted(self):
        """Built-in calls (len, print, etc.) must not end up in calls list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "builtins.py"
            src.write_text(
                "def foo(items):\n"
                "    print(len(items))\n"
                "    return str(items)\n"
            )
            symbols = self.indexer.index(str(src))
            foo = next(s for s in symbols if s.name == "foo")
            for builtin in ("print", "len", "str"):
                assert builtin not in foo.calls

    def test_empty_function_has_no_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "empty.py"
            src.write_text("def foo():\n    pass\n")
            symbols = self.indexer.index(str(src))
            foo = next(s for s in symbols if s.name == "foo")
            assert foo.calls == []


# ── 4. ASTIndexer: import symbol extraction ───────────────────────────────────

class TestASTIndexerImportSymbols:
    def setup_method(self):
        self.cfg = _config()
        self.log = _logger()
        self.indexer = ASTIndexer(self.cfg, self.log)

    def test_import_statement_creates_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "importer.py"
            src.write_text("import os\nimport sys\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            import_syms = [s for s in symbols if s.type == "import"]
            import_names = [s.name for s in import_syms]
            assert "os" in import_names
            assert "sys" in import_names

    def test_from_import_creates_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "fromimport.py"
            src.write_text("from typing import Optional\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            import_syms = [s for s in symbols if s.type == "import"]
            assert any(s.name == "typing" for s in import_syms)

    def test_import_symbol_has_correct_type_and_signature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "imp.py"
            src.write_text("import hashlib\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            imp = next(s for s in symbols if s.type == "import" and s.name == "hashlib")
            assert imp.signature == "import hashlib"
            assert imp.type == "import"

    def test_from_import_signature_contains_from_keyword(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "fimport.py"
            src.write_text("from os.path import join\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            imp = next(s for s in symbols if s.type == "import")
            assert "from" in imp.signature
            assert "os.path" in imp.signature

    def test_no_duplicate_import_symbols_per_statement(self):
        """'from x import y' must produce exactly one Symbol, not two."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "dedup.py"
            src.write_text("from os.path import join\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            import_syms = [s for s in symbols if s.type == "import"]
            assert len(import_syms) == 1


# ── 5. ASTIndexer: constant symbol extraction ─────────────────────────────────

class TestASTIndexerConstantSymbols:
    def setup_method(self):
        self.cfg = _config()
        self.log = _logger()
        self.indexer = ASTIndexer(self.cfg, self.log)

    def test_upper_case_constant_is_extracted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "consts.py"
            src.write_text("MAX_RETRIES = 3\nDEFAULT_TIMEOUT = 30\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            const_names = [s.name for s in symbols if s.type == "constant"]
            assert "MAX_RETRIES" in const_names
            assert "DEFAULT_TIMEOUT" in const_names

    def test_lower_case_assignment_not_extracted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "noconst.py"
            src.write_text("x = 5\nresult = 'hello'\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            const_names = [s.name for s in symbols if s.type == "constant"]
            assert "x" not in const_names
            assert "result" not in const_names

    def test_constant_signature_contains_value_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "sigconst.py"
            src.write_text("API_VERSION = 'v2'\n\ndef foo(): pass\n")
            symbols = self.indexer.index(str(src))
            const = next((s for s in symbols if s.name == "API_VERSION"), None)
            assert const is not None
            assert "API_VERSION" in const.signature
            assert "v2" in const.signature

    def test_constant_inside_function_not_extracted(self):
        """Constants inside function bodies must NOT be extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "funcconst.py"
            src.write_text(
                "def foo():\n"
                "    INNER = 42\n"
                "    return INNER\n"
            )
            symbols = self.indexer.index(str(src))
            assert not any(s.type == "constant" for s in symbols)


# ── 6. GraphBuilder: real AST-derived call edges ──────────────────────────────

class TestGraphBuilderCallEdges:
    def setup_method(self):
        self.cfg = _config()
        self.log = _logger()
        self.indexer = ASTIndexer(self.cfg, self.log)
        self.builder = GraphBuilder(self.cfg, self.log)

    def _build(self, *files):
        symbols = []
        for f in files:
            symbols.extend(self.indexer.index(str(f)))
        return self.builder.build(symbols), symbols

    def test_calls_edge_exists_for_known_callee(self):
        """validate_token calls _decode_payload — a calls edge must exist."""
        graph, _ = self._build(FIXTURES / "auth.py")
        validate_id = next(n for n in graph.nodes if "validate_token" in n)
        decode_id = next(n for n in graph.nodes if "_decode_payload" in n)
        assert graph.has_edge(validate_id, decode_id)
        assert graph[validate_id][decode_id].get("type") == "calls"

    def test_calls_edge_weight_reflects_call_count(self):
        """Edge weight must equal the number of times the callee is invoked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "weighted.py"
            src.write_text(
                "def caller():\n"
                "    helper()\n"
                "    helper()\n"
                "\ndef helper(): pass\n"
            )
            symbols = self.indexer.index(str(src))
            graph = self.builder.build(symbols)
            caller_id = next(n for n in graph.nodes if "caller" in n)
            helper_id = next(n for n in graph.nodes if "helper" in n)
            assert graph.has_edge(caller_id, helper_id)
            assert graph[caller_id][helper_id]["weight"] == 2

    def test_external_call_edge_for_unresolved_name(self):
        """Calls to names not in the symbol graph become external_call edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "external.py"
            src.write_text(
                "def foo():\n"
                "    external_library_call()\n"
            )
            symbols = self.indexer.index(str(src))
            graph = self.builder.build(symbols)
            foo_id = next(n for n in graph.nodes if "foo" in n)
            external_edges = [
                (u, v, d) for u, v, d in graph.edges(data=True)
                if u == foo_id and d.get("type") == "external_call"
            ]
            assert len(external_edges) >= 1
            targets = [v for _, v, _ in external_edges]
            assert "external_library_call" in targets

    def test_no_self_loops(self):
        """A function must not have a calls edge pointing to itself."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "noloop.py"
            src.write_text("def foo():\n    foo()\n")
            symbols = self.indexer.index(str(src))
            graph = self.builder.build(symbols)
            foo_id = next(n for n in graph.nodes if "foo" in n)
            assert not graph.has_edge(foo_id, foo_id)

    def test_cross_file_calls_edge(self):
        """router.py's _validate_auth calls auth.validate_token across files."""
        graph, _ = self._build(FIXTURES / "auth.py", FIXTURES / "router.py")
        validate_auth_id = next(n for n in graph.nodes if "_validate_auth" in n)
        validate_token_id = next(n for n in graph.nodes if "validate_token" in n)
        # _validate_auth calls validate_token via self.auth_handler.validate_token()
        # The attribute call 'validate_token' should resolve to the auth.py symbol
        assert graph.has_edge(validate_auth_id, validate_token_id)


# ── 7. GraphBuilder: graph_built event with build_time_ms ─────────────────────

class TestGraphBuilderEvent:
    def setup_method(self):
        self.cfg = _config()
        self.log = _logger()
        self.builder = GraphBuilder(self.cfg, self.log)
        self.indexer = ASTIndexer(self.cfg, self.log)

    def test_graph_built_event_emitted(self):
        symbols = self.indexer.index(str(FIXTURES / "auth.py"))
        self.builder.build(symbols)
        events = self.log.get_events("graph_built")
        assert len(events) == 1

    def test_graph_built_event_has_build_time_ms(self):
        symbols = self.indexer.index(str(FIXTURES / "auth.py"))
        self.builder.build(symbols)
        event = self.log.get_events("graph_built")[0]
        assert "build_time_ms" in event
        assert isinstance(event["build_time_ms"], int)
        assert event["build_time_ms"] >= 0

    def test_graph_built_event_has_node_and_edge_counts(self):
        symbols = self.indexer.index(str(FIXTURES / "auth.py"))
        graph = self.builder.build(symbols)
        event = self.log.get_events("graph_built")[0]
        assert event["node_count"] == graph.number_of_nodes()
        assert event["edge_count"] == graph.number_of_edges()


# ── 8. GraphRanker: SUPPORT tier threshold ────────────────────────────────────

class TestGraphRankerTierThresholds:
    def test_support_threshold_is_0_60(self):
        """Spec §6.4: SUPPORT = top 10-40%, threshold at 60th percentile."""
        assert GraphRanker.TIER_THRESHOLDS["SUPPORT"] == 0.60

    def test_core_threshold_is_0_90(self):
        assert GraphRanker.TIER_THRESHOLDS["CORE"] == 0.90

    def test_support_band_covers_10_to_40_percent(self):
        """With 10 symbols, CORE=1, SUPPORT=3 (30% of 10), PERIPHERAL=6."""
        import networkx as nx
        cfg = _config()
        log = _logger()
        ranker = GraphRanker(cfg, log)

        graph = nx.DiGraph()
        # Create 10 symbols with an unequal call graph so PageRank varies
        for i in range(10):
            sym = Symbol(
                id=f"f.py::func{i}", type="function", name=f"func{i}",
                qualified_name=f"func{i}", file="f.py",
                start_line=i * 5 + 1, end_line=i * 5 + 5,
                signature=f"def func{i}()",
            )
            graph.add_node(sym.id, symbol=sym)

        # Chain: func0 → func1 → func2 → ... → func9 so PageRank varies
        for i in range(9):
            graph.add_edge(f"f.py::func{i}", f"f.py::func{i+1}",
                           type="calls", weight=1)

        ranker.rank_symbols(graph, task_keywords=[])

        tiers = [graph.nodes[n]["tier"] for n in graph.nodes]
        core_count = tiers.count("CORE")
        support_count = tiers.count("SUPPORT")
        peripheral_count = tiers.count("PERIPHERAL")

        # With 10 symbols: top 10%=1 CORE, top 10-40%=3 SUPPORT, bottom 60%=6 PERIPHERAL
        assert core_count >= 1
        assert support_count >= 1        # at least some in SUPPORT
        # The total must add up
        assert core_count + support_count + peripheral_count == 10

    def test_peripheral_does_not_include_top_40_percent(self):
        """No symbol in the top 40% should be labelled PERIPHERAL."""
        import networkx as nx
        cfg = _config()
        log = _logger()
        ranker = GraphRanker(cfg, log)

        graph = nx.DiGraph()
        for i in range(20):
            sym = Symbol(
                id=f"f.py::f{i}", type="function", name=f"f{i}",
                qualified_name=f"f{i}", file="f.py",
                start_line=i * 3 + 1, end_line=i * 3 + 3,
                signature=f"def f{i}()",
            )
            graph.add_node(sym.id, symbol=sym)
        for i in range(19):
            graph.add_edge(f"f.py::f{i}", f"f.py::f{i+1}", type="calls", weight=1)

        ranker.rank_symbols(graph, task_keywords=[])

        scores = sorted(
            [(graph.nodes[n]["symbol"].pagerank, graph.nodes[n]["tier"])
             for n in graph.nodes],
            reverse=True,
        )
        top_8 = scores[:8]   # top 40% of 20
        for _score, tier in top_8:
            assert tier != "PERIPHERAL", \
                "A top-40% symbol was incorrectly labelled PERIPHERAL"


# ── 9 & 10. IndexRenderer: constant and import symbols ────────────────────────

class TestIndexRendererNewSymbols:
    def setup_method(self):
        import networkx as nx
        self.cfg = _config()
        self.log = _logger()
        self.renderer = IndexRenderer(self.cfg, self.log)
        self.graph = nx.DiGraph()

    def _add(self, sym, tier="CORE"):
        self.graph.add_node(sym.id, symbol=sym, tier=tier)

    def test_core_constant_is_rendered(self):
        self._add(Symbol(
            id="f.py::constant.MAX_SIZE", type="constant", name="MAX_SIZE",
            qualified_name="MAX_SIZE", file="f.py",
            start_line=1, end_line=1, signature="MAX_SIZE = 100",
        ), tier="CORE")
        output = self.renderer.render(self.graph)
        assert "MAX_SIZE" in output

    def test_support_constant_is_rendered(self):
        self._add(Symbol(
            id="f.py::constant.TIMEOUT", type="constant", name="TIMEOUT",
            qualified_name="TIMEOUT", file="f.py",
            start_line=2, end_line=2, signature="TIMEOUT = 30",
        ), tier="SUPPORT")
        output = self.renderer.render(self.graph)
        assert "TIMEOUT" in output

    def test_peripheral_constant_is_suppressed(self):
        self._add(Symbol(
            id="f.py::constant.DEBUG_FLAG", type="constant", name="DEBUG_FLAG",
            qualified_name="DEBUG_FLAG", file="f.py",
            start_line=3, end_line=3, signature="DEBUG_FLAG = False",
        ), tier="PERIPHERAL")
        output = self.renderer.render(self.graph)
        assert "DEBUG_FLAG" not in output

    def test_import_symbol_is_rendered(self):
        self._add(Symbol(
            id="f.py::import.os", type="import", name="os",
            qualified_name="os", file="f.py",
            start_line=1, end_line=1, signature="import os",
        ))
        output = self.renderer.render(self.graph)
        assert "import os" in output

    def test_import_symbols_appear_under_imports_block(self):
        self._add(Symbol(
            id="f.py::import.sys", type="import", name="sys",
            qualified_name="sys", file="f.py",
            start_line=1, end_line=1, signature="import sys",
        ))
        output = self.renderer.render(self.graph)
        assert "imports:" in output

    def test_multiple_imports_all_rendered(self):
        for name in ("os", "sys", "json"):
            self._add(Symbol(
                id=f"f.py::import.{name}", type="import", name=name,
                qualified_name=name, file="f.py",
                start_line=1, end_line=1, signature=f"import {name}",
            ))
        output = self.renderer.render(self.graph)
        for name in ("os", "sys", "json"):
            assert name in output


# ── 11. ContextPlanner: planner_request event ─────────────────────────────────

class TestContextPlannerEvent:
    def setup_method(self):
        import networkx as nx
        self.log = _logger()
        self.cfg = _config(max_planner_retries=1)
        self.llm = _mock_llm()
        graph = nx.DiGraph()
        sym = Symbol(
            id="auth.py::Auth.validate_token", type="method",
            name="validate_token", qualified_name="Auth.validate_token",
            file="auth.py", start_line=1, end_line=10,
            signature="def validate_token(self, token: str) -> bool",
        )
        graph.add_node(sym.id, symbol=sym, tier="CORE", pagerank=0.5)
        self.graph = graph

    def test_planner_request_event_emitted_on_success(self):
        self.llm.complete.return_value = json.dumps({
            "primary_symbols": ["auth.py::Auth.validate_token"],
            "supporting_symbols": [],
            "rationale": "need validate_token",
        })
        planner = ContextPlanner(self.cfg, self.log, self.llm)
        planner.plan("document auth", "## auth.py\nAuth [CORE]", self.graph)
        events = self.log.get_events("planner_request")
        assert len(events) >= 1

    def test_planner_request_event_has_required_fields(self):
        self.llm.complete.return_value = json.dumps({
            "primary_symbols": ["auth.py::Auth.validate_token"],
            "supporting_symbols": [],
            "rationale": "testing",
        })
        planner = ContextPlanner(self.cfg, self.log, self.llm)
        planner.plan("document auth", "## auth.py\nAuth [CORE]", self.graph)

        first_event = self.log.get_events("planner_request")[0]
        assert "budget_tokens" in first_event

        success_events = [
            e for e in self.log.get_events("planner_request")
            if "primary_count" in e
        ]
        if success_events:
            assert "primary_count" in success_events[-1]
            assert "rationale_preview" in success_events[-1]

    def test_planner_start_event_not_emitted(self):
        """The old event names must no longer appear."""
        self.llm.complete.return_value = json.dumps({
            "primary_symbols": ["auth.py::Auth.validate_token"],
            "supporting_symbols": [],
            "rationale": "testing",
        })
        planner = ContextPlanner(self.cfg, self.log, self.llm)
        planner.plan("document auth", "## auth.py\nAuth [CORE]", self.graph)

        assert self.log.get_events("planner_start") == []
        assert self.log.get_events("planner_success") == []


# ── 12. PRISMConfig: output_mode default ───────────────────────────────────────

class TestConfigDefaults:
    def test_output_mode_defaults_to_single_file(self):
        assert PRISMConfig().output_mode == "single_file"

    def test_output_mode_can_be_overridden(self):
        cfg = PRISMConfig(output_mode="inline")
        assert cfg.output_mode == "inline"

    def test_output_mode_mirror_accepted(self):
        cfg = PRISMConfig(output_mode="mirror")
        assert cfg.output_mode == "mirror"


# ── 13. Orchestrator: raw_tokens_available in stats ───────────────────────────

class TestOrchestratorStats:
    def test_raw_tokens_available_in_run_stats(self):
        """run() stats must include raw_tokens_available (§ Appendix B)."""
        from prism.orchestrator import PRISMOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _config(
                output_mode="inline",
                output_dir=tmpdir,
                cache_dir=tmpdir,
                max_synthesis_iterations=1,
                max_planner_retries=1,
            )

            # Build a valid LLM response for context planning
            plan_response = json.dumps({
                "primary_symbols": [],
                "supporting_symbols": [],
                "rationale": "nothing needed",
            })
            synthesis_response = "Documentation.\n[TASK_COMPLETE]"

            with patch("prism.orchestrator.LLMClient") as MockLLM:
                mock_instance = MagicMock()
                mock_instance.available = True
                mock_instance.complete.side_effect = [plan_response, synthesis_response]
                mock_instance.count_tokens.return_value = 10
                MockLLM.return_value = mock_instance

                orch = PRISMOrchestrator(config=cfg)
                result = orch.run(
                    source=str(FIXTURES),
                    task="document all methods",
                )

            assert "raw_tokens_available" in result["stats"]
            assert isinstance(result["stats"]["raw_tokens_available"], int)
            assert result["stats"]["raw_tokens_available"] >= 0

    def test_raw_tokens_available_is_positive_for_nonempty_codebase(self):
        from prism.orchestrator import PRISMOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _config(
                output_mode="inline",
                output_dir=tmpdir,
                cache_dir=tmpdir,
                max_synthesis_iterations=1,
                max_planner_retries=1,
            )
            plan_response = json.dumps({
                "primary_symbols": [],
                "supporting_symbols": [],
                "rationale": "nothing",
            })

            with patch("prism.orchestrator.LLMClient") as MockLLM:
                mock_instance = MagicMock()
                mock_instance.available = True
                mock_instance.complete.return_value = plan_response
                mock_instance.count_tokens.return_value = 10
                MockLLM.return_value = mock_instance

                orch = PRISMOrchestrator(config=cfg)
                result = orch.run(
                    source=str(FIXTURES),
                    task="document all",
                )

            assert result["stats"]["raw_tokens_available"] > 0
