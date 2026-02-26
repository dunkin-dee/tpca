"""
Two-Pass Context Agent (TPCA)
AST-Driven, Graph-Ranked Context Retrieval for Limited-Window LLMs

Phase 1 Implementation:
- StructuredLogger: Structured logging with file, console, and ring buffer
- ASTIndexer: Multi-file Python AST parsing with Tree-sitter
- GraphBuilder: Symbol graph construction with cross-file edge resolution  
- GraphRanker: Task-biased PageRank for symbol importance
- IndexRenderer: Compact text index generation
- IndexCache: Per-file caching with invalidation
"""
from .config import TPCAConfig
from .logging import LogConfig, StructuredLogger
from .models import Symbol, SymbolGraph
from .pass1 import ASTIndexer, GraphBuilder, GraphRanker, IndexRenderer
from .cache import IndexCache

__version__ = '0.1.0.dev1'

__all__ = [
    'TPCAConfig',
    'LogConfig',
    'StructuredLogger',
    'Symbol',
    'SymbolGraph',
    'ASTIndexer',
    'GraphBuilder',
    'GraphRanker',
    'IndexRenderer',
    'IndexCache',
]
