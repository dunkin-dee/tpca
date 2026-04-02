"""
PlannerAgent — breaks a task into PlanSections via the reader model.

Uses the compact index from Pass 1 to assign scope_files and scope_symbols
to each section. Retries on JSON parse failures.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Optional

from ..config import PRISMConfig
from ..llm.client import LLMClient
from .plan_model import PlanSection, SessionPlan


_SYSTEM = """\
You are a planning agent for a coding assistant.
Break the given task into sections that can each be completed within one context window.

Section token budget: {budget} tokens of source code per section.
Map each section to specific files and symbols from the provided index.
Aim for {budget} tokens or fewer per section; split large modules into multiple sections.

Output JSON only — no preamble, no markdown fences:
{{
  "global_style_notes": "<detected naming conventions, test patterns — ≤200 chars>",
  "sections": [
    {{
      "id": "s1",
      "title": "<short title ≤60 chars>",
      "description": "<what to do ≤150 chars>",
      "scope_symbols": ["<symbol_id>", ...],
      "scope_files": ["<relative_file_path>", ...],
      "estimated_tokens": <integer>
    }}
  ]
}}\
"""

_USER = """\
Task: {task}

Codebase index:
{compact_index}\
"""


class PlannerAgent:
    """
    Calls the reader model to produce a SessionPlan for the given task.

    Args:
        config: PRISMConfig (uses active_reader_model, max_planner_retries,
                            fallback_chunk_tokens as section budget).
        llm:    Initialised LLMClient.
    """

    def __init__(self, config: PRISMConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    def plan(
        self,
        task: str,
        compact_index: str,
        project_root: str,
    ) -> SessionPlan:
        """
        Generate a SessionPlan for the given task and codebase index.

        Raises:
            RuntimeError: if all retry attempts fail to produce valid JSON.
        """
        budget = self._config.fallback_chunk_tokens
        system = _SYSTEM.format(budget=budget)
        user = _USER.format(task=task, compact_index=compact_index)

        index_hash = hashlib.md5(compact_index.encode()).hexdigest()[:12]
        plan = SessionPlan.new(
            task=task,
            project_root=project_root,
            index_hash=index_hash,
            preset=self._config.provider,
        )

        last_err: Optional[Exception] = None
        for attempt in range(self._config.max_planner_retries):
            try:
                raw = self._llm.complete(
                    messages=[{"role": "user", "content": user}],
                    model=self._config.active_reader_model,
                    system=system,
                    max_tokens=2048,
                )
                data = _parse_json(raw)
                sections = data.get("sections", [])
                if not sections:
                    raise ValueError("Response contained no sections")
                plan.global_style_notes = str(data.get("global_style_notes", ""))
                plan.sections = [_section_from_dict(s) for s in sections]
                return plan

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_err = exc
                if attempt < self._config.max_planner_retries - 1:
                    # Inject error hint for the next attempt
                    user = (
                        user
                        + f"\n\nPrevious attempt failed: {exc}\n"
                        "Please output valid JSON only, no markdown."
                    )

        raise RuntimeError(
            f"PlannerAgent failed after {self._config.max_planner_retries} "
            f"attempts: {last_err}"
        )


# ── Shared helpers (also used by sub_planner_agent and evaluator_agent) ───────

def _parse_json(text: str) -> dict:
    """
    Extract a JSON object from an LLM response.
    Handles markdown code fences and leading/trailing text.
    """
    text = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return json.loads(m.group(1))

    # Find first bare JSON object
    start = text.find("{")
    if start != -1:
        return json.loads(text[start:])

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def _section_from_dict(d: dict) -> PlanSection:
    now = datetime.utcnow().isoformat()
    return PlanSection(
        id=str(d["id"]),
        title=str(d.get("title", d["id"])),
        description=str(d.get("description", "")),
        scope_symbols=list(d.get("scope_symbols", [])),
        scope_files=list(d.get("scope_files", [])),
        estimated_tokens=int(d.get("estimated_tokens", 0)),
        status="PENDING",
        created_at=now,
        updated_at=now,
    )
