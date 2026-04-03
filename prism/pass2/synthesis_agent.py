"""
SynthesisAgent: assembles the full context package from Pass 1 outputs
and ContextPlanner decisions, then drives the OutputChunker synthesis loop.

The synthesis prompt structure follows Section 8.5 of the design document:
  - compact index (Pass 1 output)
  - targeted source slices (SliceFetcher output)
  - planning rationale (ContextPlanner output)
  - prior OutputLog (bounded working memory)
  - current symbol to process
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..models.slice import SliceRequest, Slice
from ..models.output import OutputLog, OutputManifest
from ..models.chunk_plan import ChunkPlan
from ..pass2.context_planner import ContextPlanner
from ..pass2.slice_fetcher import SliceFetcher
from ..pass2.output_chunker import OutputChunker
from ..pass2.output_writer import OutputWriter
from ..llm.client import LLMClient


SYNTHESIS_PROMPT = """You are a synthesis agent working from a targeted context package.
The full source is NOT available — work only from what is provided below.

TASK: {task}

## CODEBASE INDEX:
{compact_index}

## TARGETED SOURCE SLICES:
{slices}

## PLANNING RATIONALE:
{rationale}

{prior_output_log}
Now produce output for: **{current_symbol}**

End your response with exactly one of these markers on its own line:
- [SECTION_COMPLETE: <one-line summary of what you just wrote>]
- [TASK_COMPLETE] (only if this is the final symbol and all work is done)
"""


@dataclass
class SynthesisResult:
    """The result of a complete synthesis run."""
    output: dict                     # {symbol_id: raw_output} or manifest path
    output_log: OutputLog
    manifest: Optional[OutputManifest] = None
    stats: dict = field(default_factory=dict)


class SynthesisAgent:
    """
    Orchestrates the full Pass 2 synthesis pipeline.

    1. ContextPlanner → SliceRequest (which symbols to read)
    2. SliceFetcher   → list[Slice]  (actual source code)
    3. OutputChunker  → ChunkPlan   (per-symbol work orders)
    4. LLM calls      → raw output   (one call per chunk)
    5. OutputWriter   → files or inline buffer

    All calls are bounded by config.max_synthesis_iterations.
    """

    def __init__(
        self,
        config,
        logger,
        llm_client: LLMClient,
        planner: Optional[ContextPlanner] = None,
        fetcher: Optional[SliceFetcher] = None,
        fallback=None,  # Phase 3: ChunkedFallback instance, or None
    ):
        self._config = config
        self._logger = logger
        self._llm = llm_client
        self._planner = planner or ContextPlanner(config, logger, llm_client)
        self._fetcher = fetcher or SliceFetcher(config, logger, llm_client)
        self._fallback = fallback  # Phase 3

    def run(
        self,
        task: str,
        compact_index: str,
        graph,
        source_root: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        prior_log: Optional[OutputLog] = None,     # Phase 3: rehydrated from manifest
        skip_symbols: Optional[set] = None,        # Phase 3: already-complete IDs
    ) -> SynthesisResult:
        """
        Execute the full Pass 2 pipeline for a task.

        Args:
            task:          The user's task description.
            compact_index: The Pass 1 compact index string.
            graph:         The ranked SymbolGraph from Pass 1.
            source_root:   Root directory of the source (for mirror mode paths).
            budget_tokens: Token budget for slices.

        Returns:
            SynthesisResult with output, log, manifest, and stats.
        """
        import time
        t0 = time.time()

        self._logger.info("synthesis_start", task=task[:80])

        # ── Step 1: Plan — ask the LLM what it needs ──────────────────────────
        slice_request = self._planner.plan(
            task=task,
            compact_index=compact_index,
            graph=graph,
            budget_tokens=budget_tokens,
        )

        # ── Step 2: Fetch — retrieve exact source lines ───────────────────────
        slices = self._fetcher.fetch(
            request=slice_request,
            graph=graph,
            budget=budget_tokens,
        )
        slices_by_symbol = {s.symbol_id: s for s in slices}
        formatted_slices = self._fetcher.format_slices_for_prompt(slices)

        # ── Step 3: Chunk — process symbols in dependency order ───────────────
        # Phase 3: seed the log from a prior run so completed symbols are
        # already in completed_symbols() and OutputChunker skips them.
        output_log = prior_log or OutputLog()
        chunker = OutputChunker(
            graph=graph,
            output_log=output_log,
            config=self._config,
            logger=self._logger,
            slice_request=slice_request,
        )
        writer = OutputWriter(
            config=self._config,
            logger=self._logger,
            source_root=source_root,
            task=task,  # Phase 3: stored in manifest
        )

        llm_calls = 0
        raw_outputs: dict[str, str] = {}
        max_iterations = self._config.max_synthesis_iterations

        try:
            for iteration in range(max_iterations):
                plan = chunker.get_next_chunk(
                    compact_index=compact_index,
                    slices_by_symbol=slices_by_symbol,
                    rationale=slice_request.rationale,
                )

                if plan is None:
                    self._logger.info("synthesis_complete", iterations=iteration)
                    break

                # ── Build the synthesis prompt ─────────────────────────────────
                prior_log_section = ""
                if plan.has_prior_work():
                    prior_log_section = plan.prior_log + "\n\n"

                # Format context slices for this chunk
                chunk_slices = self._fetcher.format_slices_for_prompt(
                    plan.context_pkg.get("slices", [])
                )

                prompt = SYNTHESIS_PROMPT.format(
                    task=task,
                    compact_index=plan.context_pkg.get("index", compact_index),
                    slices=chunk_slices or formatted_slices,
                    rationale=plan.context_pkg.get("rationale", slice_request.rationale),
                    prior_output_log=prior_log_section,
                    current_symbol=plan.symbol_id,
                )

                # ── LLM call ───────────────────────────────────────────────────
                response_text = self._llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._config.active_synthesis_model,
                    max_tokens=4096,
                    purpose=f"synthesis:{plan.symbol_id}",
                )
                llm_calls += 1

                # ── Extract output and summary ─────────────────────────────────
                raw_output, summary = self._extract_output_and_summary(
                    response_text, plan.symbol_id
                )

                # ── Record and write ───────────────────────────────────────────
                chunker.record_chunk(
                    chunk_id=plan.chunk_id,
                    symbol_id=plan.symbol_id,
                    raw_output=raw_output,
                    summary=summary,
                )
                writer.write(plan.symbol_id, raw_output)
                raw_outputs[plan.symbol_id] = raw_output

                # Check for TASK_COMPLETE marker
                if "[TASK_COMPLETE]" in response_text:
                    self._logger.info(
                        "synthesis_task_complete_marker",
                        iteration=iteration,
                    )
                    break
            else:
                self._logger.warn(
                    "synthesis_max_iterations",
                    max_iterations=max_iterations,
                    completed=chunker.completed_count,
                    total=chunker.total_count,
                )

            # Clean exit — mark manifest complete and persist
            writer.mark_all_complete()
            writer.finalize()  # Phase 3

        except KeyboardInterrupt:
            # Interrupted — save partial progress so the run can be resumed
            self._logger.warn("synthesis_interrupted", llm_calls=llm_calls)
            writer.save_partial()  # Phase 3
            raise

        # ── Build manifest and stats ───────────────────────────────────────
        manifest = writer.manifest

        elapsed_ms = int((time.time() - t0) * 1000)
        total_slice_tokens = sum(s.token_count for s in slices)
        stats = {
            "elapsed_ms": elapsed_ms,
            "llm_calls": llm_calls,
            "symbols_requested": len(slice_request.all_symbols),
            "slices_fetched": len(slices),
            "tokens_sent_to_llm": total_slice_tokens,
            "output_chunks": len(output_log.entries),
            "output_log_tokens": self._llm.count_tokens(output_log.render_compact()),
            "fallback_used": False,
        }

        self._logger.info(
            "synthesis_result",
            **{k: v for k, v in stats.items() if isinstance(v, (int, float, bool))},
        )

        # For inline mode, output is the in-memory buffer; else per-file status
        if self._config.output_mode == "inline":
            output = writer.get_output()  # Phase 3: replaces flush_inline()
        else:
            output = {e.output_file: e.status for e in manifest.files}

        return SynthesisResult(
            output=output,
            output_log=output_log,
            manifest=manifest,
            stats=stats,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_output_and_summary(
        response_text: str, symbol_id: str
    ) -> tuple[str, str]:
        """
        Split the LLM response into (raw_output, summary).

        Looks for the [SECTION_COMPLETE: ...] or [TASK_COMPLETE] markers
        and splits accordingly.
        """
        section_pattern = re.compile(
            r"\[SECTION_COMPLETE:\s*(.+?)\]", re.IGNORECASE
        )
        task_pattern = re.compile(r"\[TASK_COMPLETE\]", re.IGNORECASE)

        # Try SECTION_COMPLETE marker
        match = section_pattern.search(response_text)           
        if match:
            summary = match.group(1).strip()
            raw_output = response_text[: match.start()].strip()
            return raw_output, summary

        # Try TASK_COMPLETE marker
        match = task_pattern.search(response_text)
        if match:
            raw_output = response_text[: match.start()].strip()
            return raw_output, f"Completed {symbol_id}"

        # No marker found — use full response as output with default summary
        return response_text.strip(), f"Completed {symbol_id} (no marker found)"