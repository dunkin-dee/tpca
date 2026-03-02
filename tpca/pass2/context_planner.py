"""
ContextPlanner: shows the LLM the compact index and asks it to request
exactly the symbols it needs for the task.

The LLM acts as a librarian, not a reader.
"""
from __future__ import annotations

import json
from difflib import get_close_matches
from typing import Optional

from ..models.slice import SliceRequest
from ..llm.client import LLMClient


CONTEXT_PLANNING_PROMPT = """You are a context planning agent for a code task.

Task: {task}

Context budget: {budget_tokens} tokens.

CODEBASE INDEX:
{compact_index}

Request ONLY the symbols you need. Use fully qualified IDs
(e.g. src/auth.py::Auth.validate_token).

Respond with ONLY valid JSON — no markdown fences, no preamble:
{{
  "primary_symbols":    ["<file>::<Symbol.method>"],
  "supporting_symbols": ["<file>::<Symbol.method>"],
  "rationale": "Brief explanation of why these symbols are needed."
}}

Rules:
- primary_symbols: MUST read to complete the task
- supporting_symbols: helpful context, droppable if over budget
- max 15 symbols total
- use fully qualified IDs from the index exactly as shown
- if no supporting symbols are needed, use an empty list
"""

REPLANNER_PROMPT = """Your previous symbol request contained unknown IDs.

Unknown IDs you used: {invalid_ids}

Closest matches found in the index:
{suggestions}

Please revise your request using ONLY IDs from the index.
Respond with ONLY valid JSON (same format as before):
{{
  "primary_symbols":    [...],
  "supporting_symbols": [...],
  "rationale": "..."
}}
"""


class ContextPlanner:
    """
    LLM-driven context planner.

    Given the compact index from Pass 1 and a task description, asks the LLM
    to identify which symbols are required. Validates the response against the
    actual SymbolGraph and retries with suggestions if invalid IDs are used.
    """

    def __init__(self, config, logger: object, llm_client: LLMClient):
        self._config = config
        self._logger = logger
        self._llm = llm_client

    def plan(
        self,
        task: str,
        compact_index: str,
        graph,
        budget_tokens: Optional[int] = None,
    ) -> SliceRequest:
        """
        Generate a SliceRequest for the given task.

        Args:
            task:          The user's task description.
            compact_index: The rendered Pass 1 index string.
            graph:         The NetworkX SymbolGraph (for validation).
            budget_tokens: Available token budget (defaults from config).

        Returns:
            A validated SliceRequest with primary and supporting symbols.
        """
        if budget_tokens is None:
            budget_tokens = int(
                self._config.model_context_window * self._config.context_budget_pct
            )

        valid_ids = set(graph.nodes())
        self._logger.info(
            "planner_start",
            task=task[:80],
            budget_tokens=budget_tokens,
            valid_symbol_count=len(valid_ids),
        )

        # Initial planning call
        prompt = CONTEXT_PLANNING_PROMPT.format(
            task=task,
            budget_tokens=budget_tokens,
            compact_index=compact_index,
        )
        messages = [{"role": "user", "content": prompt}]
        response_text = self._llm.complete(
            messages=messages,
            model=self._config.active_reader_model,
            max_tokens=1024,
        )

        request = self._parse_response(response_text)

        # Validation + retry loop
        max_retries = self._config.max_planner_retries
        for attempt in range(max_retries):
            invalid = [s for s in request.all_symbols if s not in valid_ids]
            if not invalid:
                self._logger.info(
                    "planner_success",
                    primary_count=len(request.primary_symbols),
                    supporting_count=len(request.supporting_symbols),
                )
                return request

            suggestions = {
                s: get_close_matches(s, list(valid_ids), n=3, cutoff=0.4)
                for s in invalid
            }
            self._logger.warn(
                "planner_retry",
                attempt=attempt + 1,
                invalid_symbols=invalid,
            )

            suggestion_text = "\n".join(
                f"  {bad!r}  →  {matches or ['(no close match found)']}"
                for bad, matches in suggestions.items()
            )
            retry_prompt = REPLANNER_PROMPT.format(
                invalid_ids=json.dumps(invalid, indent=2),
                suggestions=suggestion_text,
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": retry_prompt})
            response_text = self._llm.complete(
                messages=messages,
                model=self._config.active_reader_model,
                max_tokens=1024,
            )
            request = self._parse_response(response_text)

        # Fallback: use top-N CORE symbols by PageRank
        self._logger.warn(
            "planner_fallback",
            reason="max_retries_exceeded",
            max_retries=max_retries,
        )
        return self._fallback_request(graph)

    def _parse_response(self, text: str) -> SliceRequest:
        """Parse the LLM JSON response into a SliceRequest."""
        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        try:
            data = json.loads(cleaned)
            return SliceRequest.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            self._logger.error("planner_parse_error", error=str(e), raw=text[:200])
            return SliceRequest()

    def _fallback_request(self, graph) -> SliceRequest:
        """Fall back to top CORE symbols from the graph."""
        core = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("tier") == "CORE"
        ]
        # Sort by pagerank descending
        core.sort(
            key=lambda n: graph.nodes[n].get("pagerank", 0.0), reverse=True
        )
        primary = core[:8]
        supporting = core[8:12]
        self._logger.info(
            "planner_fallback_symbols",
            primary_count=len(primary),
        )
        return SliceRequest(
            primary_symbols=primary,
            supporting_symbols=supporting,
            rationale="Fallback: top CORE symbols by PageRank (planner validation failed).",
        )
