"""
SliceFetcher: retrieves exact source lines for requested symbols,
enforcing a tiktoken-accurate token budget.

Primary symbols always get their full source (or signature-only fallback).
Supporting symbols are included greedily while budget permits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.slice import Slice, SliceRequest
from ..llm.client import LLMClient


class SliceFetcher:
    """
    Fetches source code slices for the symbols identified by ContextPlanner.

    Budget management:
    - Token budget comes from config (model_context_window * context_budget_pct).
    - Primary symbols are always included; truncated to signature-only if they
      individually exceed budget.
    - Supporting symbols are included greedily in order until budget is exhausted.
    - All token counting uses tiktoken (cl100k_base) for accuracy.
    """

    def __init__(self, config, logger, llm_client: LLMClient):
        self._config = config
        self._logger = logger
        self._token_counter = llm_client.token_counter

    def fetch(
        self,
        request: SliceRequest,
        graph,
        budget: Optional[int] = None,
    ) -> list[Slice]:
        """
        Fetch source slices for all symbols in the SliceRequest.

        Args:
            request: The SliceRequest from ContextPlanner.
            graph:   The ranked SymbolGraph (nodes carry Symbol objects).
            budget:  Token budget for slices (defaults from config).

        Returns:
            List of Slice objects, ordered: primaries first, then supporting.
        """
        if budget is None:
            budget = int(
                self._config.model_context_window * self._config.context_budget_pct
            )

        slices: list[Slice] = []
        used = 0

        self._logger.info(
            "slice_fetch_start",
            primary_count=len(request.primary_symbols),
            supporting_count=len(request.supporting_symbols),
            budget_tokens=budget,
        )

        # ── Primary symbols (always include, truncate if needed) ──────────────
        for sym_id in request.primary_symbols:
            if sym_id not in graph.nodes:
                self._logger.warn("slice_symbol_missing", symbol_id=sym_id)
                continue

            source, start, end = self._get_full_source(sym_id, graph)
            tokens = self._token_counter.count(source)
            truncated = False

            if used + tokens > budget:
                # Fall back to signature-only
                sig_source = self._get_signature_only(sym_id, graph)
                sig_tokens = self._token_counter.count(sig_source)

                if used + sig_tokens <= budget:
                    source = sig_source
                    tokens = sig_tokens
                    truncated = True
                    self._logger.warn(
                        "slice_truncated",
                        symbol_id=sym_id,
                        full_tokens=self._token_counter.count(source),
                        sig_tokens=sig_tokens,
                    )
                else:
                    # Even signature doesn't fit — skip with a warning
                    self._logger.warn(
                        "slice_skipped_over_budget",
                        symbol_id=sym_id,
                        available=budget - used,
                    )
                    continue

            slices.append(
                Slice(
                    symbol_id=sym_id,
                    source=source,
                    token_count=tokens,
                    truncated=truncated,
                    start_line=start,
                    end_line=end,
                )
            )
            used += tokens
            self._logger.info(
                "slice_fetched",
                symbol_id=sym_id,
                token_count=tokens,
                truncated=truncated,
                budget_used=used,
            )

        # ── Supporting symbols (greedy, best-effort) ──────────────────────────
        MIN_SUPPORTING_BUDGET = 200  # Don't try if less than this remains
        for sym_id in request.supporting_symbols:
            if budget - used < MIN_SUPPORTING_BUDGET:
                self._logger.debug(
                    "supporting_budget_exhausted",
                    remaining=budget - used,
                )
                break

            if sym_id not in graph.nodes:
                self._logger.warn("slice_symbol_missing", symbol_id=sym_id)
                continue

            source, start, end = self._get_full_source(sym_id, graph)
            tokens = self._token_counter.count(source)

            if used + tokens <= budget:
                slices.append(
                    Slice(
                        symbol_id=sym_id,
                        source=source,
                        token_count=tokens,
                        truncated=False,
                        start_line=start,
                        end_line=end,
                    )
                )
                used += tokens
                self._logger.info(
                    "slice_fetched",
                    symbol_id=sym_id,
                    token_count=tokens,
                    truncated=False,
                    budget_used=used,
                )
            else:
                self._logger.debug(
                    "supporting_slice_skipped",
                    symbol_id=sym_id,
                    needed=tokens,
                    remaining=budget - used,
                )

        self._logger.info(
            "slice_fetch_complete",
            slices_fetched=len(slices),
            total_tokens_used=used,
            budget_tokens=budget,
        )
        return slices

    # ── Source Retrieval ──────────────────────────────────────────────────────

    def _get_full_source(
        self, sym_id: str, graph
    ) -> tuple[str, int, int]:
        """Read the full source lines for a symbol from disk."""
        sym_data = graph.nodes[sym_id]
        symbol = sym_data.get("symbol")

        if symbol is None:
            return self._get_signature_only(sym_id, graph), 0, 0

        file_path = symbol.file
        start_line = symbol.start_line
        end_line = symbol.end_line

        try:
            path = Path(file_path)
            if not path.exists():
                self._logger.warn("slice_file_missing", file=file_path, symbol_id=sym_id)
                return self._get_signature_only(sym_id, graph), 0, 0

            lines = path.read_text(encoding="utf-8").splitlines()
            # Lines are 1-indexed in Symbol
            extracted = lines[max(0, start_line - 1): end_line]
            source = "\n".join(extracted)
            return source, start_line, end_line

        except Exception as e:
            self._logger.warn(
                "slice_read_error", file=file_path, symbol_id=sym_id, error=str(e)
            )
            return self._get_signature_only(sym_id, graph), start_line, end_line

    def _get_signature_only(self, sym_id: str, graph) -> str:
        """Return just the signature line (from Symbol.signature) as a fallback."""
        sym_data = graph.nodes.get(sym_id, {})
        symbol = sym_data.get("symbol")
        if symbol is None:
            return f"# {sym_id} (source unavailable)"
        sig = symbol.signature or f"# {symbol.name} (no signature)"
        doc = f'    """{symbol.docstring}"""' if symbol.docstring else ""
        parts = [f"# Source: {symbol.file}:{symbol.start_line}", sig]
        if doc:
            parts.append(doc)
        return "\n".join(parts)

    def format_slices_for_prompt(self, slices: list[Slice]) -> str:
        """Format fetched slices into a readable context block for the synthesis prompt."""
        if not slices:
            return "(no source slices available)"

        sections = []
        for s in slices:
            header = f"### {s.symbol_id}"
            if s.truncated:
                header += "  [SIGNATURE ONLY — full source exceeded budget]"
            sections.append(f"{header}\n```python\n{s.source}\n```")
        return "\n\n".join(sections)
