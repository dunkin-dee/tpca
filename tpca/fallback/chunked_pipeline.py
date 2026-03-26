"""
ChunkedFallback — Fallback pipeline for over-budget subgraphs.

Activates when SliceFetcher signals that the relevant subgraph for a task
exceeds the context budget even after primary/supporting truncation. At
that point the orchestrator calls ChunkedFallback instead of the normal
SynthesisAgent path.

Crucially, this only chunks the *relevant* subgraph identified by Pass 1
— not the full codebase — so even the fallback benefits from PageRank
filtering.

Design note (§10.1):
    "The ChunkedFallback activates when the SliceFetcher determines that
    the relevant subgraph for a given task exceeds the context budget even
    after all budget management measures. Critically, it chunks only the
    relevant subgraph — not the full codebase."

Flow:
    1. Receive the relevant subgraph (list of Symbol objects + source slices)
    2. Partition into chunks of ≤ fallback_chunk_tokens with overlap
    3. For each chunk, create an ephemeral ReaderAgent and call .read()
    4. Accumulate extractions in AgentMemoryStore
    5. Return the AgentMemoryStore for the SynthesisAgent to use as context
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from tpca.config import TPCAConfig
from tpca.logging.structured_logger import StructuredLogger
from tpca.models.symbol import Symbol
from tpca.fallback.reader_agent import ReaderAgent
from tpca.fallback.memory_store import AgentMemoryStore


class ChunkedFallback:
    """
    Orchestrates the reader-agent loop for over-budget subgraphs.

    Parameters
    ----------
    config:  Shared TPCAConfig (uses fallback_chunk_tokens, fallback_overlap_tokens).
    logger:  Shared StructuredLogger.
    llm:     LLMClient instance.
    """

    def __init__(
        self,
        config: TPCAConfig,
        logger: StructuredLogger,
        llm,  # LLMClient — avoid circular import at module level
    ) -> None:
        self._config = config
        self._logger = logger
        self._llm = llm

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbols: list[Symbol],
        source_slices: dict[str, str],
        task: str,
        memory_store: Optional[AgentMemoryStore] = None,
    ) -> AgentMemoryStore:
        """
        Run the fallback reader loop over *symbols* and return a populated
        AgentMemoryStore for the synthesis agent.

        Parameters
        ----------
        symbols:       All Symbol objects in the relevant subgraph.
        source_slices: Mapping of symbol_id → source text (from SliceFetcher).
        task:          Top-level task description.
        memory_store:  Existing store to resume from (None = fresh run).

        Returns
        -------
        AgentMemoryStore populated with extractions from every chunk.
        """
        store = memory_store or AgentMemoryStore()

        # Build ordered list of (symbol, source) pairs for chunking
        pairs = self._order_pairs(symbols, source_slices)

        # Partition into chunks
        chunks = self._partition(pairs)
        total = len(chunks)

        self._logger.info(
            'fallback_activated',
            reason='budget_exceeded',
            subgraph_size=len(symbols),
            chunks=total,
            chunk_tokens=self._config.fallback_chunk_tokens,
        )

        # Identify already-processed chunk indices for resume
        completed_ids = {c.chunk_id for c in store.chunks}

        t0 = time.monotonic()
        for i, chunk_pairs in enumerate(chunks):
            if i in completed_ids:
                self._logger.debug('fallback_chunk_skip', chunk_index=i, reason='already_complete')
                continue

            chunk_symbols = [p[0] for p in chunk_pairs]
            source_text = self._format_chunk(chunk_pairs)

            agent = ReaderAgent(
                config=self._config,
                logger=self._logger,
                llm=self._llm,
                chunk_index=i,
                total_chunks=total,
            )
            extraction, token_count = agent.read(
                symbols=chunk_symbols,
                source_text=source_text,
                task=task,
            )

            store.add_extraction(
                chunk_id=i,
                symbol_ids=[s.id for s in chunk_symbols],
                summary=extraction,
                token_count=token_count,
            )

        store.total_source_tokens = sum(
            self._llm.count_tokens(src) for _, src in pairs
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        self._logger.info(
            'fallback_complete',
            chunks_processed=store.chunks_processed,
            total_source_tokens=store.total_source_tokens,
            elapsed_ms=elapsed_ms,
        )

        return store

    # ── Chunking helpers ──────────────────────────────────────────────────────

    def _order_pairs(
        self,
        symbols: list[Symbol],
        source_slices: dict[str, str],
    ) -> list[tuple[Symbol, str]]:
        """
        Return (Symbol, source_text) pairs ordered by file then start_line.
        Symbols without a source slice get signature-only text.
        """
        pairs = []
        for sym in sorted(symbols, key=lambda s: (s.file, s.start_line)):
            src = source_slices.get(sym.id, sym.signature or f'# {sym.id}')
            pairs.append((sym, src))
        return pairs

    def _partition(
        self,
        pairs: list[tuple[Symbol, str]],
    ) -> list[list[tuple[Symbol, str]]]:
        """
        Greedily partition pairs into chunks of ≤ fallback_chunk_tokens.

        Each chunk overlaps with the next by fallback_overlap_tokens worth
        of content from the end of the previous chunk — this preserves
        cross-symbol context at boundaries.
        """
        budget = self._config.fallback_chunk_tokens
        overlap = self._config.fallback_overlap_tokens

        chunks: list[list[tuple[Symbol, str]]] = []
        current: list[tuple[Symbol, str]] = []
        current_tokens = 0

        for sym, src in pairs:
            tok = self._llm.count_tokens(src)
            # If a single symbol exceeds the budget, add it alone (truncated)
            if tok > budget:
                src = self._truncate_to_budget(src, budget)
                tok = budget

            if current_tokens + tok > budget and current:
                chunks.append(current)
                # Carry over overlap: last N tokens of the previous chunk
                current, current_tokens = self._build_overlap(current, overlap)

            current.append((sym, src))
            current_tokens += tok

        if current:
            chunks.append(current)

        return chunks

    def _build_overlap(
        self,
        chunk: list[tuple[Symbol, str]],
        overlap_tokens: int,
    ) -> tuple[list[tuple[Symbol, str]], int]:
        """
        Return the tail of *chunk* that fits within *overlap_tokens*,
        together with its total token count.
        """
        carried: list[tuple[Symbol, str]] = []
        tokens = 0
        for sym, src in reversed(chunk):
            t = self._llm.count_tokens(src)
            if tokens + t > overlap_tokens:
                break
            carried.insert(0, (sym, src))
            tokens += t
        return carried, tokens

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """
        Truncate *text* to approximately *budget* tokens.
        Uses 4-chars/token as a fast approximation for the truncation only.
        """
        approx_chars = budget * 4
        if len(text) <= approx_chars:
            return text
        return text[:approx_chars] + '\n# [truncated — source exceeds chunk budget]'

    @staticmethod
    def _format_chunk(pairs: list[tuple[Symbol, str]]) -> str:
        """Format a list of (Symbol, source) pairs into a readable text block."""
        parts = []
        for sym, src in pairs:
            header = f'### {sym.id}'
            parts.append(f'{header}\n```\n{src}\n```')
        return '\n\n'.join(parts)
