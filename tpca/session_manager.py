"""
SessionManager — orchestrates the full plan → evaluate → dispatch lifecycle.

Responsibilities:
  E1: Run PlannerAgent, evaluate sections, optionally split over-budget ones.
  E2: Extract GlobalContext (CORE symbols + naming conventions) for all workers.
  E3: Dispatch WorkerAgent instances for each pending leaf section, with
      optional parallel execution for file-independent sections.
      Propagate interface-change notifications to dependent sections.
      Persist plan state after every section completes.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import TPCAConfig
from .llm.client import LLMClient
from .plan.plan_model import PlanSection, SessionPlan, WorkerSummary
from .plan.plan_store import PlanStore
from .plan.planner_agent import PlannerAgent
from .plan.sub_planner_agent import SubPlannerAgent
from .plan.evaluator_agent import EvaluatorAgent
from .tools.executor import ToolExecutor
from .workers.worker_context import WorkerContextBuilder
from .workers.worker_agent import WorkerAgent


class SessionManager:
    """
    Manages the full planning + worker dispatch lifecycle for a coding session.

    Args:
        plan_store:    PlanStore for the project root.
        llm:           Initialised LLMClient.
        config:        TPCAConfig.
        graph:         Pass 1 NetworkX DiGraph (optional but strongly recommended).
        compact_index: Pass 1 compact text index string.
        project_root:  Absolute path to project root directory.
    """

    def __init__(
        self,
        plan_store: PlanStore,
        llm: LLMClient,
        config: TPCAConfig,
        graph: Any = None,
        compact_index: str = "",
        project_root: str = ".",
    ) -> None:
        self._store = plan_store
        self._llm = llm
        self._config = config
        self._graph = graph
        self._compact_index = compact_index
        self._root = str(Path(project_root).resolve())

        self._planner = PlannerAgent(config, llm)
        self._sub_planner = SubPlannerAgent(config, llm)
        self._evaluator = EvaluatorAgent(config, llm)
        self._ctx_builder = WorkerContextBuilder(
            config=config,
            graph=graph,
            compact_index=compact_index,
            project_root=self._root,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_session(self, task: str) -> SessionPlan:
        """
        Create a new SessionPlan via the planner, evaluate all sections,
        persist, and return the plan in EXECUTING status.
        """
        plan = self._planner.plan(
            task=task,
            compact_index=self._compact_index,
            project_root=self._root,
        )
        plan = self.run_planning_loop(plan)
        self._store.save(plan)
        return plan

    def resume_session(self) -> Optional[SessionPlan]:
        """Load the persisted plan. Returns None if none found or file is corrupt."""
        return self._store.load()

    def run_planning_loop(self, plan: SessionPlan) -> SessionPlan:
        """
        Evaluate every top-level section; split over-budget ones recursively.

        Sets plan.status = "EXECUTING" on return.
        """
        plan.status = "EVALUATING"
        budget = self._config.fallback_chunk_tokens

        refined: list[PlanSection] = []
        for section in plan.sections:
            refined.append(self._evaluate_and_refine(section, budget, depth=0))
        plan.sections = refined

        plan.status = "EXECUTING"
        plan.updated_at = datetime.utcnow().isoformat()
        return plan

    def dispatch_workers(
        self,
        plan: SessionPlan,
        executor: ToolExecutor,
    ) -> list[WorkerSummary]:
        """
        Execute all pending leaf sections in dependency order.

        After each section:
          - Evaluates worker output with EvaluatorAgent.
          - Propagates interface changes to dependent pending sections.
          - Atomically saves the plan to disk.

        Returns list of all WorkerSummary objects produced (in completion order).
        """
        all_summaries: list[WorkerSummary] = []
        completed_summaries: list[WorkerSummary] = []

        while True:
            pending = plan.pending_leaves()
            if not pending:
                break

            groups = _partition_by_file_independence(pending)

            for group in groups:
                if len(group) > 1 and self._config.parallel_workers:
                    summaries = self._run_parallel(
                        group, plan, completed_summaries, executor
                    )
                else:
                    summaries = [
                        self._run_one(s, plan, completed_summaries, executor)
                        for s in group
                    ]

                for section, summary in zip(group, summaries):
                    section.worker_summary = summary
                    section.status = _section_status(summary)
                    section.updated_at = datetime.utcnow().isoformat()

                    evaluation = self._evaluator.evaluate_worker(section, summary)
                    section.evaluation = evaluation
                    if evaluation.recommendation == "REVISE":
                        section.status = "NEEDS_REVISION"

                    if summary.interfaces_changed:
                        _flag_dependents(plan, summary)

                    completed_summaries.append(summary)
                    all_summaries.append(summary)
                    self._store.save(plan)

        plan.status = "COMPLETE"
        plan.updated_at = datetime.utcnow().isoformat()
        self._store.save(plan)
        return all_summaries

    def build_global_context(self, plan: SessionPlan) -> str:
        """
        Build the 800-token GlobalContext prefix shared by all workers.

        Includes top CORE symbols (by PageRank) and naming conventions.
        """
        parts: list[str] = []

        if plan.global_style_notes:
            parts.append(f"Style: {plan.global_style_notes}")

        if self._graph is not None:
            core_nodes = [
                (nid, d)
                for nid, d in self._graph.nodes(data=True)
                if d.get("tier") == "CORE"
            ]
            core_nodes.sort(key=lambda x: x[1].get("pagerank", 0.0), reverse=True)
            core_lines: list[str] = []
            for nid, data in core_nodes[:10]:
                sym = data.get("symbol")
                if sym:
                    sig = getattr(sym, "signature", nid)
                    core_lines.append(f"  {sig}")
            if core_lines:
                parts.append("Core symbols:\n" + "\n".join(core_lines))

        return "\n\n".join(parts)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _evaluate_and_refine(
        self, section: PlanSection, budget: int, depth: int
    ) -> PlanSection:
        """Recursively evaluate and optionally split a section."""
        if depth >= 3:
            return section

        evaluation = self._evaluator.evaluate_plan_section(section, budget)
        section.evaluation = evaluation

        should_split = (
            evaluation.recommendation == "SPLIT"
            or section.estimated_tokens > budget * 1.5
        )

        if should_split:
            sub_sections = self._sub_planner.split(
                section=section,
                compact_index=self._compact_index,
                depth=depth + 1,
            )
            if len(sub_sections) > 1:
                section.sub_sections = [
                    self._evaluate_and_refine(s, budget, depth + 1)
                    for s in sub_sections
                ]

        return section

    def _run_one(
        self,
        section: PlanSection,
        plan: SessionPlan,
        completed_summaries: list[WorkerSummary],
        executor: ToolExecutor,
    ) -> WorkerSummary:
        """Mark section IN_PROGRESS, build WorkerAgent, run, return summary."""
        section.status = "IN_PROGRESS"
        section.updated_at = datetime.utcnow().isoformat()
        self._store.save(plan)

        context = self._ctx_builder.build(
            section=section,
            plan=plan,
            prior_summaries=completed_summaries,
            tool_specs=executor.get_descriptions(),
        )
        agent = WorkerAgent(
            section=section,
            context=context,
            llm=self._llm,
            executor=executor,
            config=self._config,
        )
        return agent.run()

    def _run_parallel(
        self,
        group: list[PlanSection],
        plan: SessionPlan,
        completed_summaries: list[WorkerSummary],
        executor: ToolExecutor,
    ) -> list[WorkerSummary]:
        """Run a file-independent group of sections in a thread pool."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as pool:
            futures = {
                pool.submit(
                    self._run_one, s, plan, list(completed_summaries), executor
                ): s
                for s in group
            }
            results: dict[str, WorkerSummary] = {}
            for future in concurrent.futures.as_completed(futures):
                section = futures[future]
                try:
                    results[section.id] = future.result()
                except Exception as exc:
                    results[section.id] = _failed_summary(section, str(exc))

        return [results[s.id] for s in group]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _partition_by_file_independence(
    sections: list[PlanSection],
) -> list[list[PlanSection]]:
    """
    Partition sections into groups that touch disjoint sets of files.

    Sections within a group share no scope_files and can execute in parallel.
    Sequential ordering is preserved within each group.
    """
    groups: list[list[PlanSection]] = []
    group_files: list[set[str]] = []

    for section in sections:
        files = set(section.scope_files)
        placed = False
        for i, gfiles in enumerate(group_files):
            if not gfiles & files:
                groups[i].append(section)
                group_files[i] |= files
                placed = True
                break
        if not placed:
            groups.append([section])
            group_files.append(set(files))

    return groups


def _section_status(summary: WorkerSummary) -> str:
    return {
        "COMPLETE": "COMPLETE",
        "PARTIAL":  "NEEDS_REVISION",
        "FAILED":   "BLOCKED",
    }.get(summary.status, "NEEDS_REVISION")


def _flag_dependents(plan: SessionPlan, summary: WorkerSummary) -> None:
    """
    Mark pending sections NEEDS_REVISION if they depend on changed interfaces.

    Scans all non-complete leaf sections for scope_symbols that appear in
    summary.interfaces_changed or summary.new_symbols.
    """
    changed = set(summary.interfaces_changed) | set(summary.new_symbols)
    if not changed:
        return

    for section in plan.all_leaf_sections():
        if section.status in ("COMPLETE", "BLOCKED"):
            continue
        for sym in section.scope_symbols:
            if any(sym in c or c.startswith(sym) for c in changed):
                section.status = "NEEDS_REVISION"
                section.updated_at = datetime.utcnow().isoformat()
                break


def _failed_summary(section: PlanSection, error: str) -> WorkerSummary:
    return WorkerSummary(
        section_id=section.id,
        status="FAILED",
        brief=f"Worker raised an exception: {error[:100]}",
        detail=error[:500],
        files_changed=[],
        symbols_touched=[],
        interfaces_changed=[],
        new_symbols=[],
        assumptions=[],
        blockers=[error],
        token_cost=0,
        test_result=None,
    )
