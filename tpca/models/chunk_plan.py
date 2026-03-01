"""
ChunkPlan: the work order passed to each synthesis call by the OutputChunker.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkPlan:
    """
    Describes a single unit of synthesis work: one symbol (or symbol group)
    to be processed in the current LLM call.
    """
    chunk_id: int
    symbol_id: str
    prior_log: str          # OutputLog.render_compact() — bounded context
    context_pkg: dict       # {'index': str, 'slices': list[Slice], 'rationale': str}

    def has_prior_work(self) -> bool:
        return bool(self.prior_log.strip())
