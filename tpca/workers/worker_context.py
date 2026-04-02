"""
WorkerContextBuilder — assembles the per-section context bundle for WorkerAgent.

Token budget targets (13B primary, 16K context):
  Global prefix:    800 tokens  (CORE symbols + naming conventions)
  Sub-index:       2000 tokens  (section-scoped compact index)
  Source slices:   3500 tokens  (actual source for scope files, token-budgeted)
  Prior summaries:  800 tokens  (brief + interfaces + new_symbols for relevant sections)
  ─────────────────────────────────────────────────────────────────────────────
  Total context:   7100 tokens  → leaves ~4000 tokens for tool specs + response
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import TPCAConfig
from ..llm.client import TokenCounter
from ..plan.plan_model import PlanSection, SessionPlan, WorkerSummary


@dataclass
class WorkerContext:
    section: PlanSection
    global_prefix: str      # top CORE symbols, naming conventions (~800 tok)
    sub_index: str          # section-scoped compact index (~2000 tok)
    source_slices: str      # actual source for scope_files (~3500 tok)
    prior_text: str         # prior section summaries (~800 tok)
    tool_specs: str         # compact tool descriptions (~500 tok)


class WorkerContextBuilder:
    """
    Builds a WorkerContext for a PlanSection.

    Reads source files directly from the filesystem and filters the compact
    Pass-1 index to produce a context that fits within the model's token budget.

    Args:
        config:        TPCAConfig (uses tokenizer, fallback_chunk_tokens).
        graph:         Pass 1 NetworkX DiGraph (for CORE symbol extraction).
        compact_index: Pass 1 compact text index string.
        project_root:  Absolute path to project root (for file reading).
        token_counter: Optional pre-built TokenCounter; one is created if None.
    """

    GLOBAL_BUDGET = 800
    SUBINDEX_BUDGET = 2000
    SLICES_BUDGET = 3500
    PRIOR_BUDGET = 800

    def __init__(
        self,
        config: TPCAConfig,
        graph: Any,
        compact_index: str,
        project_root: str,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self._config = config
        self._graph = graph
        self._compact_index = compact_index
        self._root = Path(project_root).resolve()
        self._tc = token_counter or TokenCounter(config.tokenizer)

    def build(
        self,
        section: PlanSection,
        plan: SessionPlan,
        prior_summaries: list[WorkerSummary],
        tool_specs: str = "",
    ) -> WorkerContext:
        """Assemble all context components for the given section."""
        return WorkerContext(
            section=section,
            global_prefix=self._build_global_prefix(plan),
            sub_index=self._build_sub_index(section),
            source_slices=self._build_source_slices(section),
            prior_text=self._build_prior_text(section, prior_summaries),
            tool_specs=tool_specs,
        )

    # ── Component builders ────────────────────────────────────────────────────

    def _build_global_prefix(self, plan: SessionPlan) -> str:
        """800-token global context prefix: style notes + top CORE symbols."""
        parts: list[str] = []

        if plan.global_style_notes:
            parts.append(f"# Style conventions\n{plan.global_style_notes}")

        if self._graph is not None:
            core_nodes = [
                (nid, data)
                for nid, data in self._graph.nodes(data=True)
                if data.get("tier") == "CORE"
            ]
            core_nodes.sort(
                key=lambda x: x[1].get("pagerank", 0.0), reverse=True
            )
            core_lines: list[str] = []
            for nid, data in core_nodes[:10]:
                sym = data.get("symbol")
                if sym:
                    sig = getattr(sym, "signature", nid)
                    core_lines.append(f"  {sig}  [{nid}]")
            if core_lines:
                parts.append("# Core symbols\n" + "\n".join(core_lines))

        return self._truncate("\n\n".join(parts), self.GLOBAL_BUDGET)

    def _build_sub_index(self, section: PlanSection) -> str:
        """
        2000-token sub-index: compact index filtered to section scope_files.

        Falls back to the full index (truncated) when scope_files is empty.
        """
        if not section.scope_files:
            return self._truncate(self._compact_index, self.SUBINDEX_BUDGET)

        scope_set = set(section.scope_files)
        filtered = _extract_file_sections(self._compact_index, scope_set)
        if not filtered.strip():
            filtered = self._compact_index

        return self._truncate(filtered, self.SUBINDEX_BUDGET)

    def _build_source_slices(self, section: PlanSection) -> str:
        """
        3500-token source slice: actual file contents for scope_files.

        Files are read in order; content is truncated once the budget runs out.
        """
        if not section.scope_files:
            return ""

        budget_remaining = self.SLICES_BUDGET
        parts: list[str] = []

        for rel_path in section.scope_files:
            if budget_remaining <= 50:
                break

            abs_path = self._root / rel_path
            if not abs_path.is_file():
                # Try resolving by filename only
                candidates = list(self._root.rglob(Path(rel_path).name))
                if candidates:
                    abs_path = candidates[0]
                else:
                    parts.append(f"# {rel_path}\n(file not found)")
                    continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                parts.append(f"# {rel_path}\n(read error: {exc})")
                continue

            header = f"# {rel_path}\n"
            header_tokens = self._tc.count(header)
            available = budget_remaining - header_tokens
            if available <= 0:
                break

            content_tokens = self._tc.count(content)
            if content_tokens > available:
                # Approximate character truncation (4 chars ≈ 1 token)
                content = content[: available * 4] + "\n... (truncated)"

            entry = header + content
            parts.append(entry)
            budget_remaining -= self._tc.count(entry)

        return "\n\n".join(parts)

    def _build_prior_text(
        self, section: PlanSection, prior_summaries: list[WorkerSummary]
    ) -> str:
        """
        800-token prior summaries: brief + interfaces_changed + new_symbols for
        completed sections whose scope overlaps with this section.
        """
        if not prior_summaries:
            return ""

        scope_files = set(section.scope_files)
        scope_symbols = set(section.scope_symbols)

        relevant: list[str] = []
        for s in prior_summaries:
            overlaps_files = bool(set(s.files_changed) & scope_files)
            overlaps_symbols = bool(set(s.symbols_touched) & scope_symbols)
            has_interface_info = bool(s.interfaces_changed or s.new_symbols)

            if overlaps_files or overlaps_symbols or has_interface_info:
                lines = [f"[{s.section_id}] {s.brief}"]
                if s.interfaces_changed:
                    lines.append(
                        "  interfaces_changed: " + "; ".join(s.interfaces_changed)
                    )
                if s.new_symbols:
                    lines.append("  new_symbols: " + "; ".join(s.new_symbols))
                relevant.append("\n".join(lines))

        return self._truncate("\n".join(relevant), self.PRIOR_BUDGET)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _truncate(self, text: str, budget_tokens: int) -> str:
        if self._tc.count(text) <= budget_tokens:
            return text
        # Approximate: 4 chars per token
        return text[: budget_tokens * 4] + "\n... (truncated)"


def _extract_file_sections(compact_index: str, scope_set: set[str]) -> str:
    """
    Extract file-section blocks from a compact index for the given scope_set.

    File sections start with '## <path>' and end at the next '## ' line.
    """
    lines = compact_index.splitlines(keepends=True)
    file_sections: dict[str, list[str]] = {}
    current_file: Optional[str] = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_file is not None:
                file_sections[current_file] = current_lines
            current_file = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_file is not None:
        file_sections[current_file] = current_lines

    result_parts: list[str] = []
    for file_path, content_lines in file_sections.items():
        for scope_file in scope_set:
            if (
                file_path == scope_file
                or file_path.endswith(scope_file)
                or scope_file.endswith(file_path)
                or Path(file_path).name == Path(scope_file).name
            ):
                result_parts.append("".join(content_lines))
                break

    return "\n".join(result_parts)
