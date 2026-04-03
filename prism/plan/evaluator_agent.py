"""
EvaluatorAgent — scores plan sections and completed worker output.

Two modes:
  evaluate_plan_section(section, budget) → PlanEvaluation
      Checks structure, granularity, and consistency.

  evaluate_worker(section, summary) → PlanEvaluation
      Reviews actual diff/summary against the section goal.
      Activates code_correctness and test_coverage for 13B+ models.
"""
from __future__ import annotations

import json
from typing import Optional

from ..config import PRISMConfig
from ..llm.client import LLMClient
from .plan_model import PlanEvaluation, PlanSection, WorkerSummary
from .planner_agent import _parse_json

_PLAN_EVAL_SYSTEM = """\
You are a plan evaluator. Score the given section on these dimensions (0.0–1.0):
  completeness  — does the description cover what the task requires?
  granularity   — does estimated_tokens fit within the budget? (1.0 = yes, 0.0 = far too large)
  consistency   — are scope_files and description coherent?
  score         — weighted composite of the above

code_correctness and test_coverage are 0.0 for plan evaluation (not applicable yet).

Output JSON only:
{
  "score": 0.0-1.0,
  "completeness": 0.0-1.0,
  "granularity": 0.0-1.0,
  "consistency": 0.0-1.0,
  "code_correctness": 0.0,
  "test_coverage": 0.0,
  "critique": "<≤200 chars>",
  "recommendation": "APPROVE|REVISE|SPLIT"
}\
"""

_WORKER_EVAL_SYSTEM = """\
You are a worker evaluator. Review the completed section against its goal.
Score each dimension 0.0–1.0:
  completeness     — was all required work done?
  consistency      — are changes consistent with the section goal?
  code_correctness — is the diff/change logically correct? (0.0 if test_result contains FAIL)
  test_coverage    — does new code have tests?
  score            — weighted composite

If test_result contains "FAIL", set code_correctness=0.0 and recommend REVISE.
If interfaces_changed is non-empty, verify consistency with stated section goal.

Output JSON only:
{
  "score": 0.0-1.0,
  "completeness": 0.0-1.0,
  "granularity": 1.0,
  "consistency": 0.0-1.0,
  "code_correctness": 0.0-1.0,
  "test_coverage": 0.0-1.0,
  "critique": "<≤200 chars>",
  "recommendation": "APPROVE|REVISE|SPLIT"
}\
"""

# Safe default returned when all retries fail
_DEFAULT_EVAL = PlanEvaluation(
    score=0.5,
    completeness=0.5,
    granularity=1.0,
    consistency=0.5,
    code_correctness=0.0,
    test_coverage=0.0,
    critique="Evaluation failed — using safe defaults.",
    recommendation="APPROVE",
)


class EvaluatorAgent:
    """
    Calls the reader model to produce a PlanEvaluation.

    Args:
        config: PRISMConfig (uses active_reader_model, max_planner_retries).
        llm:    Initialised LLMClient.
    """

    def __init__(self, config: PRISMConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    def evaluate_plan_section(
        self, section: PlanSection, budget: int
    ) -> PlanEvaluation:
        """Evaluate a plan section for structure and granularity."""
        user = (
            f"Section: {section.id} — {section.title}\n"
            f"Description: {section.description}\n"
            f"Scope files: {section.scope_files}\n"
            f"Scope symbols: {section.scope_symbols}\n"
            f"Estimated tokens: {section.estimated_tokens}\n"
            f"Budget per section: {budget}\n"
        )
        return self._call(_PLAN_EVAL_SYSTEM, user, purpose=f"eval:plan:{section.id}")

    def evaluate_worker(
        self, section: PlanSection, summary: WorkerSummary
    ) -> PlanEvaluation:
        """Evaluate completed worker output against the section goal."""
        user = (
            f"Section goal: {section.description}\n"
            f"Files changed: {summary.files_changed}\n"
            f"Brief: {summary.brief}\n"
            f"Detail: {summary.detail}\n"
            f"Test result: {summary.test_result or 'not run'}\n"
            f"Interfaces changed: {summary.interfaces_changed}\n"
            f"New symbols: {summary.new_symbols}\n"
            f"Assumptions: {summary.assumptions}\n"
            f"Blockers: {summary.blockers}\n"
        )
        return self._call(_WORKER_EVAL_SYSTEM, user, purpose=f"eval:worker:{section.id}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str, purpose: str = "evaluator") -> PlanEvaluation:
        last_err: Optional[Exception] = None
        for attempt in range(self._config.max_planner_retries):
            try:
                raw = self._llm.complete(
                    messages=[{"role": "user", "content": user}],
                    model=self._config.active_reader_model,
                    system=system,
                    max_tokens=512,
                    purpose=purpose,
                )
                data = _parse_json(raw)
                return PlanEvaluation.from_dict(data)

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_err = exc
                if attempt < self._config.max_planner_retries - 1:
                    user = (
                        user
                        + f"\n\nPrevious attempt failed: {exc}\n"
                        "Output valid JSON only."
                    )

        return _DEFAULT_EVAL
