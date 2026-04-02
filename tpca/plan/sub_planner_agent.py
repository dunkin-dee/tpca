"""
SubPlannerAgent — splits an over-budget PlanSection into sub-sections.

Triggered when section.estimated_tokens > budget * 1.5.
Recursion is capped at depth 3.
"""
from __future__ import annotations

import json
from typing import Optional

from ..config import TPCAConfig
from ..llm.client import LLMClient
from .plan_model import PlanSection
from .planner_agent import _parse_json, _section_from_dict

_MAX_DEPTH = 3

_SYSTEM = """\
You are a sub-planning agent. Split the given section into smaller sub-sections.
Each sub-section must fit within {budget} tokens of source code.
Keep sub-section IDs as "{parent_id}.1", "{parent_id}.2", etc.

Output JSON only:
{{
  "sub_sections": [
    {{
      "id": "{parent_id}.1",
      "title": "...",
      "description": "...",
      "scope_symbols": [...],
      "scope_files": [...],
      "estimated_tokens": <integer>
    }}
  ]
}}\
"""


class SubPlannerAgent:
    """
    Splits a single over-budget PlanSection into leaf sub-sections.

    Args:
        config: TPCAConfig (uses active_reader_model, max_planner_retries,
                            fallback_chunk_tokens as target budget).
        llm:    Initialised LLMClient.
    """

    def __init__(self, config: TPCAConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    def split(
        self,
        section: PlanSection,
        compact_index: str,
        depth: int = 1,
    ) -> list[PlanSection]:
        """
        Break section into sub-sections. Returns [section] unchanged if
        splitting fails or recursion depth is exceeded.
        """
        if depth > _MAX_DEPTH:
            return [section]

        budget = self._config.fallback_chunk_tokens
        system = _SYSTEM.format(budget=budget, parent_id=section.id)
        user = (
            f"Parent section: {section.title}\n"
            f"Description: {section.description}\n"
            f"Files in scope: {section.scope_files}\n"
            f"Symbols in scope: {section.scope_symbols}\n\n"
            f"Codebase index:\n{compact_index}"
        )

        last_err: Optional[Exception] = None
        for attempt in range(self._config.max_planner_retries):
            try:
                raw = self._llm.complete(
                    messages=[{"role": "user", "content": user}],
                    model=self._config.active_reader_model,
                    system=system,
                    max_tokens=1024,
                )
                data = _parse_json(raw)
                subs = [
                    _section_from_dict(s)
                    for s in data.get("sub_sections", [])
                ]
                if not subs:
                    return [section]
                return subs

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_err = exc
                if attempt < self._config.max_planner_retries - 1:
                    user = (
                        user
                        + f"\n\nPrevious attempt failed: {exc}\n"
                        "Output valid JSON only."
                    )

        # Splitting failed — return original section unchanged
        return [section]
