"""
Tests for Phase D — WorkerContextBuilder, templates, and WorkerAgent.

All LLM calls and file I/O are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.config import PRISMConfig
from prism.logging.log_config import LogConfig
from prism.logging.structured_logger import StructuredLogger
from prism.plan.plan_model import PlanSection, SessionPlan, WorkerSummary
from prism.tools.executor import ToolExecutor, ToolResult
from prism.workers.templates import detect_task_type, get_template, TASK_TYPES
from prism.workers.worker_context import WorkerContextBuilder, _extract_file_sections
from prism.workers.worker_agent import WorkerAgent, _TestAwareExecutor


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_config():
    return PRISMConfig(
        provider="anthropic",
        synthesis_model="claude-sonnet-4-6",
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
        max_tool_rounds=5,
    )


def make_section(
    id="s1",
    title="Test section",
    description="Implement the foo function",
    scope_files=None,
    scope_symbols=None,
):
    return PlanSection(
        id=id,
        title=title,
        description=description,
        scope_files=scope_files or ["prism/foo.py"],
        scope_symbols=scope_symbols or ["foo.py::foo"],
    )


def make_plan(sections=None):
    plan = SessionPlan.new(task="Test task", project_root="/tmp/proj")
    plan.global_style_notes = "snake_case, Google docstrings"
    plan.sections = sections or []
    return plan


def make_mock_graph():
    """Build a minimal mock NetworkX graph with CORE/SUPPORT tiers."""
    g = MagicMock()
    sym = MagicMock()
    sym.signature = "def foo(x: int) -> str"
    g.nodes.return_value = [("foo.py::foo", {"tier": "CORE", "pagerank": 0.9, "symbol": sym})]
    g.nodes.__iter__ = lambda self: iter([])
    # nodes(data=True) returns list of (id, data)
    g.nodes.side_effect = None
    data_iter = [("foo.py::foo", {"tier": "CORE", "pagerank": 0.9, "symbol": sym})]
    g.nodes = MagicMock()
    g.nodes.return_value = data_iter
    # Support both g.nodes() and g.nodes(data=True)
    g.nodes.__call__ = lambda data=False: data_iter
    return g


# ── Template tests ─────────────────────────────────────────────────────────────

class TestDetectTaskType:
    def test_new_file_keywords(self):
        assert detect_task_type("create new auth module") == "NEW_FILE"

    def test_test_keywords(self):
        assert detect_task_type("write tests for the router") == "TEST"

    def test_docs_keywords(self):
        assert detect_task_type("add docstrings to public methods") == "DOCS"

    def test_refactor_keywords(self):
        assert detect_task_type("refactor the auth class") == "REFACTOR"

    def test_code_edit_default(self):
        assert detect_task_type("fix the validation logic") == "CODE_EDIT"

    def test_unknown_defaults_to_code_edit(self):
        assert detect_task_type("something completely unspecific") == "CODE_EDIT"


class TestGetTemplate:
    def test_returns_string_for_all_types(self):
        for t in TASK_TYPES:
            result = get_template(t)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_includes_write_summary_reminder(self):
        for t in TASK_TYPES:
            assert "write_summary" in get_template(t)

    def test_naming_conventions_injected(self):
        result = get_template("CODE_EDIT", "snake_case")
        assert "snake_case" in result

    def test_unknown_type_falls_back_to_code_edit(self):
        result = get_template("UNKNOWN_TYPE")
        # Should not raise; falls back to CODE_EDIT template
        assert "write_summary" in result


# ── WorkerContextBuilder tests ────────────────────────────────────────────────

class TestExtractFileSections:
    COMPACT_INDEX = """\
## src/auth.py
class Auth                              [CORE]
  + validate_token(token)               [CORE]

## src/router.py
class Router                            [SUPPORT]
  + add_route(path, handler)            [SUPPORT]

## Cross-file references
router.py → auth.py::Auth
"""

    def test_extracts_matching_section(self):
        result = _extract_file_sections(
            self.COMPACT_INDEX, {"src/auth.py"}
        )
        assert "Auth" in result
        assert "Router" not in result

    def test_returns_empty_for_no_match(self):
        result = _extract_file_sections(self.COMPACT_INDEX, {"nonexistent.py"})
        assert result.strip() == ""

    def test_multiple_scope_files(self):
        result = _extract_file_sections(
            self.COMPACT_INDEX, {"src/auth.py", "src/router.py"}
        )
        assert "Auth" in result
        assert "Router" in result

    def test_basename_match(self):
        result = _extract_file_sections(self.COMPACT_INDEX, {"auth.py"})
        assert "Auth" in result


class TestWorkerContextBuilder:
    def setup_method(self):
        self.config = make_config()
        self.compact_index = (
            "## src/foo.py\n"
            "def foo(x: int) -> str      [CORE]\n"
        )

    def test_build_returns_worker_context(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("def foo(x): return str(x)\n")

        section = make_section(scope_files=["src/foo.py"])
        plan = make_plan()

        builder = WorkerContextBuilder(
            config=self.config,
            graph=None,
            compact_index=self.compact_index,
            project_root=str(tmp_path),
        )
        ctx = builder.build(section, plan, prior_summaries=[])

        assert ctx.section is section
        assert isinstance(ctx.global_prefix, str)
        assert isinstance(ctx.sub_index, str)
        assert isinstance(ctx.source_slices, str)
        assert isinstance(ctx.prior_text, str)

    def test_sub_index_filtered_to_scope(self, tmp_path):
        index = (
            "## src/foo.py\ndef foo()  [CORE]\n"
            "## src/bar.py\ndef bar()  [SUPPORT]\n"
        )
        section = make_section(scope_files=["src/foo.py"])
        plan = make_plan()

        builder = WorkerContextBuilder(
            config=self.config,
            graph=None,
            compact_index=index,
            project_root=str(tmp_path),
        )
        ctx = builder.build(section, plan, prior_summaries=[])

        assert "foo" in ctx.sub_index
        assert "bar" not in ctx.sub_index

    def test_source_slices_reads_file(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("def foo(): pass\n")

        section = make_section(scope_files=["src/foo.py"])
        plan = make_plan()

        builder = WorkerContextBuilder(
            config=self.config,
            graph=None,
            compact_index=self.compact_index,
            project_root=str(tmp_path),
        )
        ctx = builder.build(section, plan, prior_summaries=[])

        assert "def foo" in ctx.source_slices

    def test_prior_text_includes_overlapping_summaries(self, tmp_path):
        prior = WorkerSummary(
            section_id="s0",
            status="COMPLETE",
            brief="Added foo function",
            detail="",
            files_changed=["src/foo.py"],
            symbols_touched=["foo.py::foo"],
        )
        section = make_section(scope_files=["src/foo.py"])
        plan = make_plan()

        builder = WorkerContextBuilder(
            config=self.config,
            graph=None,
            compact_index=self.compact_index,
            project_root=str(tmp_path),
        )
        ctx = builder.build(section, plan, prior_summaries=[prior])

        assert "Added foo function" in ctx.prior_text

    def test_prior_text_excludes_non_overlapping(self, tmp_path):
        prior = WorkerSummary(
            section_id="s0",
            status="COMPLETE",
            brief="Unrelated change",
            detail="",
            files_changed=["src/other.py"],
            symbols_touched=["other.py::other"],
        )
        section = make_section(scope_files=["src/foo.py"])
        plan = make_plan()

        builder = WorkerContextBuilder(
            config=self.config,
            graph=None,
            compact_index=self.compact_index,
            project_root=str(tmp_path),
        )
        ctx = builder.build(section, plan, prior_summaries=[prior])

        assert "Unrelated" not in ctx.prior_text


# ── _TestAwareExecutor tests ──────────────────────────────────────────────────

class TestTestAwareExecutor:
    def _make_executor(self, write_result, test_result):
        inner = MagicMock(spec=ToolExecutor)
        inner.execute.side_effect = lambda tool, args: (
            write_result if tool != "run_tests" else test_result
        )
        inner.last_summary = None
        inner.get_schemas.return_value = []
        inner.get_descriptions.return_value = ""
        return _TestAwareExecutor(inner)

    def test_pass_appended_after_successful_write(self):
        write_ok = ToolResult(output="Wrote 100 chars to foo.py")
        test_pass = ToolResult(output="EXIT 0\n1 passed")
        wrapper = self._make_executor(write_ok, test_pass)

        result = wrapper.execute("write_file", {"path": "foo.py", "content": "x=1"})
        assert "Tests: PASS" in result.output
        assert wrapper.last_test_result == "PASS"

    def test_fail_appended_after_failed_tests(self):
        write_ok = ToolResult(output="Wrote 100 chars to foo.py")
        test_fail = ToolResult(output="EXIT 1\nAssertionError: 1 != 2")
        wrapper = self._make_executor(write_ok, test_fail)

        result = wrapper.execute("write_file", {"path": "foo.py", "content": "x=1"})
        assert "Tests FAILED" in result.output
        assert wrapper.last_test_result.startswith("FAIL:")

    def test_non_mutation_tool_not_tested(self):
        read_result = ToolResult(output="file content")
        inner = MagicMock(spec=ToolExecutor)
        inner.execute.return_value = read_result
        inner.last_summary = None
        wrapper = _TestAwareExecutor(inner)

        result = wrapper.execute("read_file", {"path": "foo.py"})
        assert result.output == "file content"
        # run_tests should not have been called
        calls = [c for c in inner.execute.call_args_list if c[0][0] == "run_tests"]
        assert not calls

    def test_failed_write_not_tested(self):
        write_fail = ToolResult(output="", error="Path traversal")
        inner = MagicMock(spec=ToolExecutor)
        inner.execute.return_value = write_fail
        inner.last_summary = None
        wrapper = _TestAwareExecutor(inner)

        wrapper.execute("write_file", {"path": "../../etc/passwd", "content": ""})
        calls = [c for c in inner.execute.call_args_list if c[0][0] == "run_tests"]
        assert not calls


# ── WorkerAgent tests ─────────────────────────────────────────────────────────

class TestWorkerAgent:
    def _make_llm(self, tool_calls=None, final_text="Done."):
        llm = MagicMock()
        llm.complete_with_tools.return_value = (final_text, tool_calls or [])
        return llm

    def _make_executor_with_summary(self, summary_dict):
        # No spec= — _TestAwareExecutor adds last_test_result which ToolExecutor lacks
        exec_ = MagicMock()
        exec_.last_summary = summary_dict
        exec_.last_test_result = "PASS"
        exec_.get_schemas.return_value = []
        exec_.get_descriptions.return_value = ""
        exec_.execute.return_value = ToolResult(output="ok")
        return exec_

    def test_run_extracts_write_summary_payload(self):
        summary_payload = {
            "section_id": "s1",
            "status": "COMPLETE",
            "brief": "Implemented foo function",
            "detail": "Added foo to auth.py",
            "files_changed": ["auth.py"],
            "symbols_touched": ["auth.py::foo"],
            "interfaces_changed": [],
            "new_symbols": ["auth.py::foo"],
            "assumptions": [],
            "blockers": [],
        }
        config = make_config()
        section = make_section()
        plan = make_plan()

        from prism.workers.worker_context import WorkerContext
        ctx = WorkerContext(
            section=section,
            global_prefix="",
            sub_index="",
            source_slices="",
            prior_text="",
            tool_specs="",
        )

        executor = self._make_executor_with_summary(summary_payload)
        llm = self._make_llm(
            tool_calls=[{"tool_name": "write_summary", "args": summary_payload, "result": "ok"}]
        )

        # Patch _TestAwareExecutor to pass through the mock
        with patch("prism.workers.worker_agent._TestAwareExecutor") as MockWrapper:
            MockWrapper.return_value = executor
            agent = WorkerAgent(section=section, context=ctx, llm=llm,
                                executor=executor, config=config)
            summary = agent.run()

        assert summary.status == "COMPLETE"
        assert summary.brief == "Implemented foo function"
        assert "auth.py::foo" in summary.new_symbols

    def test_run_synthesizes_summary_when_write_summary_missing(self):
        config = make_config()
        section = make_section()
        plan = make_plan()

        from prism.workers.worker_context import WorkerContext
        ctx = WorkerContext(
            section=section,
            global_prefix="",
            sub_index="",
            source_slices="",
            prior_text="",
            tool_specs="",
        )

        executor = self._make_executor_with_summary(None)  # no write_summary
        tool_calls = [
            {"tool_name": "write_file", "args": {"path": "src/foo.py"}, "result": "ok"},
        ]
        llm = self._make_llm(tool_calls=tool_calls, final_text="Work done.")

        with patch("prism.workers.worker_agent._TestAwareExecutor") as MockWrapper:
            MockWrapper.return_value = executor
            agent = WorkerAgent(section=section, context=ctx, llm=llm,
                                executor=executor, config=config)
            summary = agent.run()

        assert summary.status == "PARTIAL"
        assert "src/foo.py" in summary.files_changed
        assert "write_summary not called" in summary.detail
