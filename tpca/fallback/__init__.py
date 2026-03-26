"""
TPCA Fallback Pipeline (Phase 3)

Activates when the relevant subgraph for a task exceeds the context budget
even after all normal budget management. Processes only the relevant
subgraph — not the full codebase — so even the fallback benefits from
Pass 1 filtering.
"""
from .chunked_pipeline import ChunkedFallback
from .reader_agent import ReaderAgent
from .memory_store import AgentMemoryStore

__all__ = ['ChunkedFallback', 'ReaderAgent', 'AgentMemoryStore']
