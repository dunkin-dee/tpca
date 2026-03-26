"""
AgentMemoryStore — Fallback pipeline working memory.

Reuses the OutputLog dataclass so the SynthesisAgent uses an identical
interface whether or not the fallback was triggered. The memory store
accumulates high-signal extractions (signatures, docstrings, key
relationships) from each ReaderAgent chunk.

Design note (§10.2):
    "The AgentMemoryStore in the fallback path uses the same OutputLog
    dataclass as the normal path. This means the synthesis agent uses an
    identical interface regardless of whether the fallback was triggered."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tpca.models.output import OutputLog, OutputChunk


class AgentMemoryStore(OutputLog):
    """
    Persistent, append-only memory store for the ChunkedFallback pipeline.

    Extends OutputLog so that SynthesisAgent.run() can receive it directly
    in place of a normal OutputLog — no interface changes needed downstream.

    Extra capabilities over plain OutputLog:
      - save(path)   — persist the store to disk as JSON
      - load(path)   — classmethod; rehydrate a prior store for resume
      - add_extraction(chunk_id, symbol_ids, summary, token_count)
                     — convenience wrapper used by ReaderAgent
    """

    def __init__(self) -> None:
        super().__init__()
        # Metadata added by ChunkedFallback
        self.fallback_used: bool = True
        self.total_source_tokens: int = 0
        self.chunks_processed: int = 0

    # ── Reader-agent API ──────────────────────────────────────────────────────

    def add_extraction(
        self,
        chunk_id: int,
        symbol_ids: list[str],
        summary: str,
        token_count: int = 0,
    ) -> None:
        """
        Record a completed ReaderAgent extraction.

        Parameters
        ----------
        chunk_id:    Sequential chunk index (0-based).
        symbol_ids:  Symbol IDs covered by this chunk.
        summary:     High-signal one-paragraph extraction from the reader.
        token_count: Approximate token count of the raw source that was read.
        """
        label = ', '.join(s.split('::')[-1] for s in symbol_ids[:3])
        if len(symbol_ids) > 3:
            label += f' (+{len(symbol_ids) - 3} more)'

        chunk = OutputChunk(
            chunk_id=chunk_id,
            symbol_id=label,
            summary=summary,
            status='extracted',
            token_count=token_count,
        )
        self.add(chunk)
        self.chunks_processed += 1

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write the memory store to a JSON file."""
        data = {
            'fallback_used': self.fallback_used,
            'total_source_tokens': self.total_source_tokens,
            'chunks_processed': self.chunks_processed,
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'symbol_id': c.symbol_id,
                    'summary': c.summary,
                    'status': c.status,
                    'token_count': c.token_count,
                }
                for c in self.chunks
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> 'AgentMemoryStore':
        """Rehydrate a memory store saved by a prior run."""
        data = json.loads(Path(path).read_text())
        store = cls()
        store.fallback_used = data.get('fallback_used', True)
        store.total_source_tokens = data.get('total_source_tokens', 0)
        store.chunks_processed = data.get('chunks_processed', 0)
        for c in data.get('chunks', []):
            store.add(OutputChunk(
                chunk_id=c['chunk_id'],
                symbol_id=c['symbol_id'],
                summary=c['summary'],
                status=c.get('status', 'extracted'),
                token_count=c.get('token_count', 0),
            ))
        return store

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_compact(self) -> str:
        """
        Render the memory store as a compact text block for prompt injection.

        Format is identical to OutputLog.render_compact() so downstream
        callers need no changes.
        """
        if not self.chunks:
            return ''
        lines = ['## READER EXTRACTIONS (fallback context — do not repeat)']
        for c in self.chunks:
            lines.append(f'[{c.chunk_id}] {c.symbol_id} → {c.summary}')
        return '\n'.join(lines)
