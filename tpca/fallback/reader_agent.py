"""
ReaderAgent — Ephemeral, single-chunk reader for the fallback pipeline.

Each ReaderAgent is created, run once, and discarded. It processes one
chunk of the relevant subgraph (a slice of source code within the fallback
token budget) and extracts high-signal content: signatures, docstrings,
and key cross-symbol relationships.

Design note (§10.2):
    "The fallback pipeline uses ephemeral ReaderAgents that process one
    chunk of the relevant subgraph at a time, write to a persistent
    AgentMemoryStore, and are discarded."
"""
from __future__ import annotations

import time
from typing import Optional

from tpca.config import TPCAConfig
from tpca.logging.structured_logger import StructuredLogger
from tpca.models.symbol import Symbol


READER_PROMPT = """You are a code reader extracting high-signal information for a downstream synthesis agent.

TASK CONTEXT: {task}

SOURCE CHUNK ({chunk_index} of {total_chunks}):
{source_text}

Extract the following in a compact paragraph (≤ 150 words):
1. All public function/method signatures with their parameter types and return types
2. What each function/method does (1 sentence each)
3. Key dependencies — what does this code call or import that matters?
4. Any important constants, type aliases, or patterns a synthesis agent would need

Do NOT reproduce full function bodies. Signatures and behaviour descriptions only.
End with: [EXTRACTION_COMPLETE]
"""


class ReaderAgent:
    """
    Ephemeral reader that processes one subgraph chunk and returns a
    high-signal extraction string for the AgentMemoryStore.

    Parameters
    ----------
    config:       Shared TPCAConfig.
    logger:       Shared StructuredLogger.
    llm:          LLMClient instance.
    chunk_index:  0-based index of this chunk within the fallback run.
    total_chunks: Total number of chunks in this fallback run (for prompt context).
    """

    def __init__(
        self,
        config: TPCAConfig,
        logger: StructuredLogger,
        llm,  # LLMClient — avoid circular import
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> None:
        self._config = config
        self._logger = logger
        self._llm = llm
        self._chunk_index = chunk_index
        self._total_chunks = total_chunks

    def read(
        self,
        symbols: list[Symbol],
        source_text: str,
        task: str,
    ) -> tuple[str, int]:
        """
        Read a source chunk and return (extraction_text, token_count).

        Parameters
        ----------
        symbols:     Symbol objects covered by this chunk (for logging).
        source_text: Concatenated source lines for all symbols in the chunk.
        task:        The top-level task description for context.

        Returns
        -------
        extraction:  Compact high-signal text suitable for AgentMemoryStore.
        token_count: Token count of `source_text` (before extraction).
        """
        token_count = self._llm.count_tokens(source_text)
        symbol_ids = [s.id for s in symbols]

        self._logger.info(
            'reader_agent_start',
            chunk_index=self._chunk_index,
            total_chunks=self._total_chunks,
            symbols=len(symbols),
            source_tokens=token_count,
        )

        prompt = READER_PROMPT.format(
            task=task,
            chunk_index=self._chunk_index + 1,
            total_chunks=self._total_chunks,
            source_text=source_text,
        )

        t0 = time.monotonic()
        try:
            raw = self._llm.complete(
                prompt=prompt,
                model=self._config.active_reader_model,
                max_tokens=512,
            )
        except Exception as exc:
            self._logger.error(
                'reader_agent_error',
                chunk_index=self._chunk_index,
                error=str(exc),
            )
            # Graceful degradation: return truncated signatures only
            raw = self._fallback_extraction(symbols)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Strip the completion marker if present
        extraction = raw.replace('[EXTRACTION_COMPLETE]', '').strip()

        self._logger.info(
            'reader_agent_complete',
            chunk_index=self._chunk_index,
            elapsed_ms=elapsed_ms,
            extraction_len=len(extraction),
        )

        return extraction, token_count

    # ── Fallback extraction (no LLM) ─────────────────────────────────────────

    def _fallback_extraction(self, symbols: list[Symbol]) -> str:
        """
        Build a signature-only extraction without an LLM call.
        Used when the LLM call fails so the pipeline can continue.
        """
        lines = ['Signatures only (reader LLM call failed):']
        for sym in symbols:
            if sym.signature:
                doc = f' — {sym.docstring[:60]}' if sym.docstring else ''
                lines.append(f'  {sym.signature}{doc}')
        return '\n'.join(lines)
