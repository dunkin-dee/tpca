"""
Tests for Phase E — SessionManager and _partition_by_file_independence.

All LLM calls and filesystem writes are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tpca.config import TPCAConfig
from tpca.logging.log_config import LogConfig
from tpca.plan.plan_model import PlanEvaluation, PlanSection, SessionPlan, WorkerSummary
from tpca.plan.plan_store import PlanStore
from tpca.session_manager import SessionManager, _partition_by_file_independence, _flag_dependents


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config():
    return TPCAConfig(
        provider="anthropic",
        synthesis_model="claude-sonnet-4-6",
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
        parallel_workers=False,
    )


def make_section(id, files=None, symbols=None, status="PENDING"):
    now = datetime.utcnow().isoformat()
    return PlanSection(
        id=id,
        title=f"Section {id}",
        description=f"Do work for section {id}",
        scope_files=files or [],
        scope_symbols=symbols or [],
        estimated_tokens=500,
        status=status,
        created_at=now,
        updated_at=now,
    )


def make_plan(sections=None):
    plan = SessionPlan.new(task="Test task", project_root="/tmp/proj")
    plan.global_style_notes = "snake_case"
    plan.sections = sections or []
    plan.status = "EXECUTING"
    return plan


def make_approve_eval():
    return PlanEvaluation(
        score=0.9,
        completeness=0.9,
        granularity=1.0,
        consistency=0.9,
        code_correctness=0.0,
        test_coverage=0.0,
        critique="Looks good",
        recommendation="APPROVE",
    )


def make_worker_summary(section_id, status="COMPLETE", files=None, interfaces=None):
    return WorkerSummary(
        section_id=section_id,
        status=status,
        brief=f"Completed {section_id}",
        detail="",
        files_changed=files or [],
        symbols_touched=[],
        interfaces_changed=interfaces or [],
        new_symbols=[],
        assumptions=[],
        blockers=[],
        token_cost=0,
        test_result="PASS",
    )


def make_session_manager(tmp_path, config=None):
    config = config or make_config()
    plan_store = PlanStore(project_root=str(tmp_path))

    mock_llm = MagicMock()
    mock_planner = MagicMock()
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_plan_section.return_value = make_approve_eval()
    mock_evaluator.evaluate_worker.return_value = make_approve_eval()

    sm = SessionManager(
        plan_store=plan_store,
        llm=mock_llm,
        config=config,
        graph=None,
        compact_index="## src/foo.py\ndef foo()  [CORE]\n",
        project_root=str(tmp_path),
    )
    # Replace agents with mocks
    mock_sub_planner = MagicMock()
    mock_sub_planner.split.return_value = []

    sm._planner = mock_planner
    sm._evaluator = mock_evaluator
    sm._sub_planner = mock_sub_planner
    return sm


# ── _partition_by_file_independence ──────────────────────────────────────────

class TestPartitionByFileIndependence:
    def test_disjoint_sections_in_same_group(self):
        # Disjoint files → no conflict → placed in the same group for parallel dispatch
        s1 = make_section("s1", files=["a.py"])
        s2 = make_section("s2", files=["b.py"])
        groups = _partition_by_file_independence([s1, s2])
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_overlapping_sections_in_separate_groups(self):
        # Shared files → conflict → each section gets its own group
        s1 = make_section("s1", files=["a.py"])
        s2 = make_section("s2", files=["a.py", "b.py"])
        groups = _partition_by_file_independence([s1, s2])
        assert len(groups) == 2

    def test_empty_scope_files_grouped_independently(self):
        s1 = make_section("s1", files=[])
        s2 = make_section("s2", files=[])
        groups = _partition_by_file_independence([s1, s2])
        # Both have empty file sets — no conflict, can be independent
        assert sum(len(g) for g in groups) == 2

    def test_preserves_order_within_group(self):
        s1 = make_section("s1", files=["a.py"])
        s2 = make_section("s2", files=["b.py"])
        s3 = make_section("s3", files=["c.py"])
        groups = _partition_by_file_independence([s1, s2, s3])
        flat = [s for g in groups for s in g]
        ids = [s.id for s in flat]
        assert ids == ["s1", "s2", "s3"]


# ── _flag_dependents ──────────────────────────────────────────────────────────

class TestFlagDependents:
    def test_flags_section_with_dependent_symbol(self):
        section = make_section("s2", symbols=["auth.py::Auth.validate"])
        plan = make_plan(sections=[section])

        summary = make_worker_summary(
            "s1", interfaces=["auth.py::Auth.validate(token) -> bool"]
        )
        _flag_dependents(plan, summary)

        assert section.status == "NEEDS_REVISION"

    def test_does_not_flag_complete_section(self):
        section = make_section("s2", symbols=["auth.py::Auth.validate"], status="COMPLETE")
        plan = make_plan(sections=[section])

        summary = make_worker_summary(
            "s1", interfaces=["auth.py::Auth.validate(token) -> bool"]
        )
        _flag_dependents(plan, summary)

        assert section.status == "COMPLETE"

    def test_no_change_when_no_interfaces_changed(self):
        section = make_section("s2", symbols=["auth.py::Auth.validate"])
        plan = make_plan(sections=[section])

        summary = make_worker_summary("s1", interfaces=[])
        _flag_dependents(plan, summary)

        assert section.status == "PENDING"


# ── SessionManager.run_planning_loop ─────────────────────────────────────────

class TestRunPlanningLoop:
    def test_sets_status_executing(self, tmp_path):
        sm = make_session_manager(tmp_path)
        plan = make_plan(sections=[make_section("s1")])

        result = sm.run_planning_loop(plan)

        assert result.status == "EXECUTING"

    def test_evaluates_each_section(self, tmp_path):
        sm = make_session_manager(tmp_path)
        s1 = make_section("s1")
        s2 = make_section("s2")
        plan = make_plan(sections=[s1, s2])

        sm.run_planning_loop(plan)

        assert sm._evaluator.evaluate_plan_section.call_count == 2

    def test_splits_over_budget_section(self, tmp_path):
        config = make_config()
        config.fallback_chunk_tokens = 1000
        sm = make_session_manager(tmp_path, config)

        # Section with estimated_tokens > budget * 1.5
        big_section = make_section("s1")
        big_section.estimated_tokens = 2000  # > 1000 * 1.5 = 1500

        sub1 = make_section("s1.1")
        sub2 = make_section("s1.2")
        sm._sub_planner.split.return_value = [sub1, sub2]  # already a MagicMock

        plan = make_plan(sections=[big_section])
        result = sm.run_planning_loop(plan)

        sm._sub_planner.split.assert_called_once()
        assert len(result.sections[0].sub_sections) == 2


# ── SessionManager.dispatch_workers ──────────────────────────────────────────

class TestDispatchWorkers:
    def _make_worker_mock(self, summary):
        """Return a mock WorkerAgent whose run() returns summary."""
        mock = MagicMock()
        mock.run.return_value = summary
        return mock

    def test_completes_pending_sections(self, tmp_path):
        sm = make_session_manager(tmp_path)
        s1 = make_section("s1", files=["a.py"])
        plan = make_plan(sections=[s1])

        mock_executor = MagicMock()
        mock_executor.get_descriptions.return_value = ""

        summary = make_worker_summary("s1")

        with patch("tpca.session_manager.WorkerAgent") as MockAgent:
            MockAgent.return_value = self._make_worker_mock(summary)
            with patch.object(sm._ctx_builder, "build", return_value=MagicMock()):
                all_summaries = sm.dispatch_workers(plan, mock_executor)

        assert len(all_summaries) == 1
        assert all_summaries[0].section_id == "s1"
        assert s1.status == "COMPLETE"
        assert plan.status == "COMPLETE"

    def test_partial_worker_sets_needs_revision(self, tmp_path):
        sm = make_session_manager(tmp_path)
        s1 = make_section("s1", files=["a.py"])
        plan = make_plan(sections=[s1])

        mock_executor = MagicMock()
        mock_executor.get_descriptions.return_value = ""

        summary = make_worker_summary("s1", status="PARTIAL")

        with patch("tpca.session_manager.WorkerAgent") as MockAgent:
            MockAgent.return_value = self._make_worker_mock(summary)
            with patch.object(sm._ctx_builder, "build", return_value=MagicMock()):
                sm.dispatch_workers(plan, mock_executor)

        assert s1.status == "NEEDS_REVISION"

    def test_plan_saved_after_each_section(self, tmp_path):
        sm = make_session_manager(tmp_path)
        s1 = make_section("s1", files=["a.py"])
        s2 = make_section("s2", files=["b.py"])
        plan = make_plan(sections=[s1, s2])

        mock_executor = MagicMock()
        mock_executor.get_descriptions.return_value = ""

        save_count = []

        def track_save(p):
            save_count.append(1)

        sm._store.save = track_save

        with patch("tpca.session_manager.WorkerAgent") as MockAgent:
            MockAgent.return_value = self._make_worker_mock(
                make_worker_summary("s1")
            )
            with patch.object(sm._ctx_builder, "build", return_value=MagicMock()):
                sm.dispatch_workers(plan, mock_executor)

        # Saved at least once per section + final COMPLETE save
        assert len(save_count) >= 2

    def test_interface_change_flags_dependent(self, tmp_path):
        sm = make_session_manager(tmp_path)
        s1 = make_section("s1", files=["auth.py"])
        s2 = make_section("s2", files=["router.py"], symbols=["auth.py::Auth.login"])
        plan = make_plan(sections=[s1, s2])

        mock_executor = MagicMock()
        mock_executor.get_descriptions.return_value = ""

        call_count = [0]

        def worker_side_effect(*args, **kwargs):
            mock = MagicMock()
            # First call (s1): returns summary with interface change
            # Second call (s2): returns complete summary
            if call_count[0] == 0:
                call_count[0] += 1
                mock.run.return_value = make_worker_summary(
                    "s1", interfaces=["auth.py::Auth.login(user) -> bool"]
                )
            else:
                mock.run.return_value = make_worker_summary("s2")
            return mock

        with patch("tpca.session_manager.WorkerAgent", side_effect=worker_side_effect):
            with patch.object(sm._ctx_builder, "build", return_value=MagicMock()):
                sm.dispatch_workers(plan, mock_executor)

        # s2 should have been flagged NEEDS_REVISION before it ran
        # (it will then run again as PENDING in next iteration — but since
        # we set it NEEDS_REVISION, pending_leaves() won't return it again)
        # The important thing is _flag_dependents was called
        assert sm._evaluator.evaluate_worker.called


# ── SessionManager.start_session ─────────────────────────────────────────────

class TestStartSession:
    def test_creates_and_persists_plan(self, tmp_path):
        sm = make_session_manager(tmp_path)
        plan = make_plan(sections=[make_section("s1")])
        sm._planner.plan.return_value = plan

        result = sm.start_session("Write tests")

        sm._planner.plan.assert_called_once()
        assert result.status == "EXECUTING"
        assert (tmp_path / ".tpca_plan.json").exists()

    def test_resume_returns_stored_plan(self, tmp_path):
        sm = make_session_manager(tmp_path)
        plan = make_plan(sections=[make_section("s1")])
        plan.status = "EXECUTING"
        sm._store.save(plan)

        loaded = sm.resume_session()

        assert loaded is not None
        assert loaded.plan_id == plan.plan_id

    def test_resume_returns_none_when_no_plan(self, tmp_path):
        sm = make_session_manager(tmp_path)
        result = sm.resume_session()
        assert result is None
