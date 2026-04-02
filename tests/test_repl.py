"""
Tests for Phase F — REPL plan commands and rendering helpers.

All LLM calls, filesystem writes, and subprocess calls are mocked.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.config import PRISMConfig
from prism.logging.log_config import LogConfig
from prism.plan.plan_model import PlanEvaluation, PlanSection, SessionPlan, WorkerSummary
from prism.plan.plan_store import PlanStore
from prism.cli.main import (
    PRISMRepl,
    _find_section_by_id,
    _render_plan_tree,
    _render_summary_table,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_config():
    return PRISMConfig(
        provider="anthropic",
        synthesis_model="claude-sonnet-4-6",
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
    )


def make_section(
    id="s1",
    title="Test section",
    files=None,
    symbols=None,
    status="PENDING",
    sub_sections=None,
    worker_summary=None,
):
    now = datetime.utcnow().isoformat()
    return PlanSection(
        id=id,
        title=title,
        description=f"Do work for {id}",
        scope_files=files or [],
        scope_symbols=symbols or [],
        estimated_tokens=500,
        status=status,
        created_at=now,
        updated_at=now,
        sub_sections=sub_sections or [],
        worker_summary=worker_summary,
    )


def make_plan(sections=None, status="EXECUTING", task="Test task"):
    plan = SessionPlan.new(task=task, project_root="/tmp/proj")
    plan.sections = sections or []
    plan.status = status
    return plan


def make_worker_summary(
    section_id="s1",
    status="COMPLETE",
    brief="Did the thing",
    files=None,
    test_result="PASS",
):
    return WorkerSummary(
        section_id=section_id,
        status=status,
        brief=brief,
        detail="",
        files_changed=files if files is not None else ["src/foo.py"],
        symbols_touched=[],
        interfaces_changed=[],
        new_symbols=[],
        assumptions=[],
        blockers=[],
        token_cost=0,
        test_result=test_result,
    )


def make_evaluation(recommendation="APPROVE"):
    return PlanEvaluation(
        score=0.9,
        completeness=0.9,
        granularity=1.0,
        consistency=0.9,
        code_correctness=0.0,
        test_coverage=0.0,
        critique="Looks good",
        recommendation=recommendation,
    )


def make_repl(tmp_path) -> PRISMRepl:
    config = make_config()
    repl = PRISMRepl(startup_dir=tmp_path, config=config)
    repl.current_dir = tmp_path
    return repl


# ── _find_section_by_id ───────────────────────────────────────────────────────

class TestFindSectionById:
    def test_finds_top_level_section(self):
        s1 = make_section("s1")
        s2 = make_section("s2")
        result = _find_section_by_id([s1, s2], "s2")
        assert result is s2

    def test_finds_nested_section(self):
        sub = make_section("s1.1")
        parent = make_section("s1", sub_sections=[sub])
        result = _find_section_by_id([parent], "s1.1")
        assert result is sub

    def test_returns_none_when_not_found(self):
        s1 = make_section("s1")
        assert _find_section_by_id([s1], "s99") is None

    def test_finds_deeply_nested_section(self):
        deep = make_section("s1.1.1")
        mid = make_section("s1.1", sub_sections=[deep])
        top = make_section("s1", sub_sections=[mid])
        result = _find_section_by_id([top], "s1.1.1")
        assert result is deep

    def test_empty_list_returns_none(self):
        assert _find_section_by_id([], "s1") is None


# ── _render_plan_tree ─────────────────────────────────────────────────────────

class TestRenderPlanTree:
    def test_renders_session_header(self):
        plan = make_plan(task="Add JWT auth", status="EXECUTING")
        tree = _render_plan_tree(plan)
        assert 'Session: "Add JWT auth" [EXECUTING]' in tree

    def test_renders_pending_section_with_dot_icon(self):
        plan = make_plan(sections=[make_section("s1", status="PENDING")])
        tree = _render_plan_tree(plan)
        assert "[PENDING] ·" in tree

    def test_renders_complete_section_with_checkmark(self):
        ws = make_worker_summary(brief="Did the thing", test_result="PASS")
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        tree = _render_plan_tree(plan)
        assert "[COMPLETE] ✓" in tree
        assert '"Did the thing"' in tree
        assert "tests:PASS" in tree

    def test_renders_blocked_section_with_x_icon(self):
        plan = make_plan(sections=[make_section("s1", status="BLOCKED")])
        tree = _render_plan_tree(plan)
        assert "[BLOCKED] ✗" in tree

    def test_renders_needs_revision_icon(self):
        plan = make_plan(sections=[make_section("s1", status="NEEDS_REVISION")])
        tree = _render_plan_tree(plan)
        assert "[NEEDS_REVISION] ↺" in tree

    def test_last_child_uses_corner_connector(self):
        s1 = make_section("s1")
        s2 = make_section("s2")
        plan = make_plan(sections=[s1, s2])
        tree = _render_plan_tree(plan)
        lines = tree.splitlines()
        last_section_line = [l for l in lines if "s2" in l][0]
        assert "└──" in last_section_line

    def test_intermediate_children_use_tee_connector(self):
        s1 = make_section("s1")
        s2 = make_section("s2")
        plan = make_plan(sections=[s1, s2])
        tree = _render_plan_tree(plan)
        lines = tree.splitlines()
        first_section_line = [l for l in lines if "s1" in l][0]
        assert "├──" in first_section_line

    def test_renders_nested_sub_sections(self):
        sub1 = make_section("s1.1", status="COMPLETE")
        sub2 = make_section("s1.2", status="PENDING")
        parent = make_section("s1", sub_sections=[sub1, sub2])
        plan = make_plan(sections=[parent])
        tree = _render_plan_tree(plan)
        assert "s1.1" in tree
        assert "s1.2" in tree

    def test_renders_test_fail(self):
        ws = make_worker_summary(test_result="FAIL: assertion error")
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        tree = _render_plan_tree(plan)
        assert "tests:FAIL" in tree

    def test_model_preset_in_header_when_present(self):
        plan = make_plan()
        plan.model_preset = "13b-local"
        tree = _render_plan_tree(plan)
        assert "(13b-local)" in tree

    def test_no_model_preset_suffix_when_absent(self):
        plan = make_plan()
        plan.model_preset = None
        tree = _render_plan_tree(plan)
        assert "(" not in tree.splitlines()[0]

    def test_truncates_long_brief(self):
        long_brief = "A" * 50
        ws = make_worker_summary(brief=long_brief)
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        tree = _render_plan_tree(plan)
        # Should have truncated to 40 chars + ellipsis
        assert "…" in tree


# ── _render_summary_table ─────────────────────────────────────────────────────

class TestRenderSummaryTable:
    def test_renders_header_row(self):
        plan = make_plan(sections=[make_section("s1")])
        table = _render_summary_table(plan)
        assert "ID" in table
        assert "STATUS" in table
        assert "BRIEF" in table
        assert "TEST" in table
        assert "CHANGED" in table

    def test_renders_complete_section_with_summary(self):
        ws = make_worker_summary(brief="Added foo", files=["src/foo.py"], test_result="PASS")
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        table = _render_summary_table(plan)
        assert "COMPLETE" in table
        assert "Added foo" in table
        assert "PASS" in table
        assert "src/foo.py" in table

    def test_renders_pending_section_with_dashes(self):
        plan = make_plan(sections=[make_section("s1", status="PENDING")])
        table = _render_summary_table(plan)
        assert "PENDING" in table
        # All data columns show "-"
        lines = [l for l in table.splitlines() if "s1" in l]
        assert len(lines) == 1
        assert lines[0].count("-") >= 3

    def test_truncates_brief_to_44_chars(self):
        long_brief = "B" * 50
        ws = make_worker_summary(brief=long_brief)
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        table = _render_summary_table(plan)
        assert "…" in table

    def test_shows_multiple_files_with_count(self):
        ws = make_worker_summary(files=["a.py", "b.py", "c.py"])
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        table = _render_summary_table(plan)
        assert "(+2)" in table

    def test_fail_test_result_shows_fail_label(self):
        ws = make_worker_summary(test_result="FAIL: something broke")
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        table = _render_summary_table(plan)
        assert "FAIL" in table

    def test_no_files_changed_shows_dash(self):
        ws = make_worker_summary(files=[])
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        table = _render_summary_table(plan)
        # changed column should be "-"
        data_line = [l for l in table.splitlines() if "s1" in l][0]
        assert data_line.endswith("-")


# ── REPL command tests ─────────────────────────────────────────────────────────

class TestReplPlanShow:
    def test_shows_tree_when_plan_exists(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])
        store = PlanStore(project_root=str(tmp_path))
        store.save(plan)

        repl = make_repl(tmp_path)
        repl._repl_plan([])
        out = capsys.readouterr().out
        assert "Session:" in out

    def test_shows_no_plan_message_when_absent(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_plan([])
        out = capsys.readouterr().out
        assert "No plan found" in out


class TestReplPlanNew:
    def test_shows_usage_when_no_task(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_plan(["new"])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_creates_plan_calls_start_session(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])

        repl = make_repl(tmp_path)
        repl._compact_index = "## foo.py\ndef foo() [CORE]\n"  # pre-populated
        repl._graph = None

        with patch("prism.cli.main.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.start_session.return_value = plan
            MockSM.return_value = mock_sm
            with patch.object(repl, "_get_orchestrator") as mock_orch:
                mock_orch.return_value.llm = MagicMock()
                repl._repl_plan(["new", "Add", "JWT", "auth"])

        mock_sm.start_session.assert_called_once_with("Add JWT auth")
        out = capsys.readouterr().out
        assert "Session:" in out

    def test_aborts_when_existing_plan_and_user_says_no(self, tmp_path, capsys):
        plan = make_plan()
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        with patch("builtins.input", return_value="n"):
            repl._repl_plan(["new", "some task"])
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_overwrites_when_user_confirms(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._compact_index = "index"
        repl._graph = None

        with patch("builtins.input", return_value="y"):
            with patch("prism.cli.main.SessionManager") as MockSM:
                mock_sm = MagicMock()
                mock_sm.start_session.return_value = plan
                MockSM.return_value = mock_sm
                with patch.object(repl, "_get_orchestrator") as mock_orch:
                    mock_orch.return_value.llm = MagicMock()
                    repl._repl_plan(["new", "new task"])

        mock_sm.start_session.assert_called_once()


class TestReplPlanClear:
    def test_deletes_plan_after_confirmation(self, tmp_path, capsys):
        plan = make_plan()
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        with patch("builtins.input", return_value="y"):
            repl._repl_plan(["clear"])

        out = capsys.readouterr().out
        assert "deleted" in out.lower()
        assert not PlanStore(project_root=str(tmp_path)).exists()

    def test_aborts_when_user_says_no(self, tmp_path, capsys):
        plan = make_plan()
        store = PlanStore(project_root=str(tmp_path))
        store.save(plan)

        repl = make_repl(tmp_path)
        with patch("builtins.input", return_value="n"):
            repl._repl_plan(["clear"])

        assert store.exists()
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_shows_message_when_no_plan(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_plan(["clear"])
        out = capsys.readouterr().out
        assert "No plan" in out


class TestReplContinue:
    def test_shows_message_when_no_plan(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_continue([])
        out = capsys.readouterr().out
        assert "No saved plan" in out

    def test_shows_message_when_plan_is_complete(self, tmp_path, capsys):
        plan = make_plan(status="COMPLETE")
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_continue([])
        out = capsys.readouterr().out
        assert "already complete" in out

    def test_dispatches_workers_for_executing_plan(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")], status="EXECUTING")
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._compact_index = "index"
        repl._graph = None

        ws = make_worker_summary()

        with patch("prism.cli.main.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.dispatch_workers.return_value = [ws]
            MockSM.return_value = mock_sm
            with patch("prism.cli.main.ToolExecutor"):
                with patch.object(repl, "_get_orchestrator") as mock_orch:
                    mock_orch.return_value.llm = MagicMock()
                    repl._repl_continue([])

        mock_sm.dispatch_workers.assert_called_once()
        out = capsys.readouterr().out
        assert "1 section(s) completed" in out


class TestReplEval:
    def test_shows_usage_when_no_args(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_eval([])
        assert "Usage:" in capsys.readouterr().out

    def test_shows_error_when_no_plan(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_eval(["s1"])
        assert "No plan found" in capsys.readouterr().out

    def test_shows_error_when_section_not_found(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_eval(["s99"])
        assert "not found" in capsys.readouterr().out

    def test_evaluates_section_and_shows_result(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        evaluation = make_evaluation(recommendation="APPROVE")

        with patch("prism.cli.main.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm._evaluator.evaluate_plan_section.return_value = evaluation
            MockSM.return_value = mock_sm
            with patch.object(repl, "_get_orchestrator") as mock_orch:
                mock_orch.return_value.llm = MagicMock()
                repl._repl_eval(["s1"])

        out = capsys.readouterr().out
        assert "APPROVE" in out
        assert "Score" in out


class TestReplRetry:
    def test_shows_usage_when_no_args(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_retry([])
        assert "Usage:" in capsys.readouterr().out

    def test_shows_error_when_no_plan(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_retry(["s1"])
        assert "No plan found" in capsys.readouterr().out

    def test_shows_error_when_section_not_found(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_retry(["s99"])
        assert "not found" in capsys.readouterr().out

    def test_refuses_to_retry_complete_section(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1", status="COMPLETE")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_retry(["s1"])
        assert "Only BLOCKED" in capsys.readouterr().out

    def test_marks_blocked_section_pending_and_dispatches(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1", status="BLOCKED")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._compact_index = "index"
        repl._graph = None

        ws = make_worker_summary()

        with patch("prism.cli.main.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.dispatch_workers.return_value = [ws]
            MockSM.return_value = mock_sm
            with patch("prism.cli.main.ToolExecutor"):
                with patch.object(repl, "_get_orchestrator") as mock_orch:
                    mock_orch.return_value.llm = MagicMock()
                    repl._repl_retry(["s1"])

        mock_sm.dispatch_workers.assert_called_once()
        out = capsys.readouterr().out
        assert "reset to PENDING" in out


class TestReplTools:
    def test_prints_tool_descriptions(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        with patch("prism.cli.main.ToolExecutor") as MockExec:
            mock_exec = MagicMock()
            mock_exec.get_descriptions.return_value = "read_file: read a file\nwrite_file: write a file"
            MockExec.return_value = mock_exec
            repl._repl_tools([])

        out = capsys.readouterr().out
        assert "read_file" in out
        assert "write_file" in out


class TestReplSummary:
    def test_shows_message_when_no_plan(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_summary([])
        assert "No plan found" in capsys.readouterr().out

    def test_renders_summary_table(self, tmp_path, capsys):
        ws = make_worker_summary(brief="Did the thing", test_result="PASS")
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_summary([])
        out = capsys.readouterr().out
        assert "COMPLETE" in out
        assert "Did the thing" in out


class TestReplDiff:
    def test_shows_usage_when_no_args(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_diff([])
        assert "Usage:" in capsys.readouterr().out

    def test_shows_error_when_no_plan(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        repl._repl_diff(["s1"])
        assert "No plan found" in capsys.readouterr().out

    def test_shows_error_when_section_not_found(self, tmp_path, capsys):
        plan = make_plan(sections=[make_section("s1")])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_diff(["s99"])
        assert "not found" in capsys.readouterr().out

    def test_shows_message_when_no_files_changed(self, tmp_path, capsys):
        ws = make_worker_summary(files=[])
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        repl._repl_diff(["s1"])
        assert "no recorded file changes" in capsys.readouterr().out

    def test_calls_git_diff_for_section_files(self, tmp_path, capsys):
        ws = make_worker_summary(files=["src/auth.py"])
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        with patch("prism.cli.main.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1 +1 @@\n",
                stderr="",
            )
            repl._repl_diff(["s1"])

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "git" in args
        assert "diff" in args
        assert "src/auth.py" in args

        out = capsys.readouterr().out
        assert "src/auth.py" in out

    def test_shows_message_when_no_diff_output(self, tmp_path, capsys):
        ws = make_worker_summary(files=["src/auth.py"])
        plan = make_plan(sections=[make_section("s1", status="COMPLETE", worker_summary=ws)])
        PlanStore(project_root=str(tmp_path)).save(plan)

        repl = make_repl(tmp_path)
        with patch("prism.cli.main.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="")
            repl._repl_diff(["s1"])

        assert "no diff" in capsys.readouterr().out.lower()


class TestEnsureIndex:
    def test_runs_pass1_when_compact_index_is_none(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        assert repl._compact_index is None

        with patch.object(repl, "_get_orchestrator") as mock_get_orch:
            mock_orch = MagicMock()
            mock_orch.run_pass1_only.return_value = {
                "index": "## foo.py\n",
                "graph": MagicMock(),
                "stats": {"symbols_indexed": 3},
            }
            mock_get_orch.return_value = mock_orch
            repl._ensure_index()

        assert repl._compact_index == "## foo.py\n"
        assert repl._graph is not None
        mock_orch.run_pass1_only.assert_called_once()

    def test_skips_pass1_when_already_indexed(self, tmp_path):
        repl = make_repl(tmp_path)
        repl._compact_index = "already indexed"
        repl._graph = MagicMock()

        with patch.object(repl, "_get_orchestrator") as mock_get_orch:
            repl._ensure_index()
            mock_get_orch.assert_not_called()

        assert repl._compact_index == "already indexed"


class TestReplIndexUpdatesState:
    def test_stores_graph_and_compact_index_after_index_command(self, tmp_path, capsys):
        repl = make_repl(tmp_path)
        mock_graph = MagicMock()

        with patch.object(repl, "_get_orchestrator") as mock_get_orch:
            mock_orch = MagicMock()
            mock_orch.run_pass1_only.return_value = {
                "index": "## foo.py\ndef foo() [CORE]",
                "graph": mock_graph,
                "stats": {"symbols_indexed": 1},
            }
            mock_get_orch.return_value = mock_orch
            repl._repl_index([])

        assert repl._compact_index == "## foo.py\ndef foo() [CORE]"
        assert repl._graph is mock_graph
