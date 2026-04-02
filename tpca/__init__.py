"""
TPCA — Two-Pass Context Agent

Phase 1: AST-driven indexing and ranking (zero LLM).
Phase 2: LLM-driven context planning and synthesis.
"""

# Config & Logging
from .config import TPCAConfig
from .logging.log_config import LogConfig
from .logging.structured_logger import StructuredLogger

# Data models
from .models.symbol import Symbol
from .models.slice import Slice, SliceRequest
from .models.output import OutputLog, OutputChunk, OutputManifest, ManifestEntry
from .models.chunk_plan import ChunkPlan

# Pass 1
from .pass1.ast_indexer import ASTIndexer
from .pass1.graph_builder import GraphBuilder
from .pass1.graph_ranker import GraphRanker
from .pass1.index_renderer import IndexRenderer
from .cache.index_cache import IndexCache

# LLM client
from .llm.client import LLMClient, TokenCounter

# Pass 2
from .pass2.context_planner import ContextPlanner
from .pass2.slice_fetcher import SliceFetcher
from .pass2.output_chunker import OutputChunker
from .pass2.output_writer import OutputWriter
from .pass2.synthesis_agent import SynthesisAgent, SynthesisResult

# Phase 3 — Fallback pipeline
from .fallback.chunked_pipeline import ChunkedFallback
from .fallback.reader_agent import ReaderAgent
from .fallback.memory_store import AgentMemoryStore

# Orchestrator
from .orchestrator import TPCAOrchestrator

# Phase D — Worker agents
from .workers.worker_context import WorkerContext, WorkerContextBuilder
from .workers.worker_agent import WorkerAgent
from .workers.templates import TASK_TYPES, detect_task_type, get_template

# Phase E — Session management
from .session_manager import SessionManager
from .plan.plan_store import PlanStore
from .plan.plan_model import (
    PlanEvaluation,
    PlanSection,
    SessionPlan,
    WorkerSummary,
)
from .plan.planner_agent import PlannerAgent
from .plan.sub_planner_agent import SubPlannerAgent
from .plan.evaluator_agent import EvaluatorAgent

__version__ = "2.0.0"

__all__ = [
    # Config
    "TPCAConfig",
    "LogConfig",
    "StructuredLogger",
    # Models
    "Symbol",
    "Slice",
    "SliceRequest",
    "OutputLog",
    "OutputChunk",
    "OutputManifest",
    "ManifestEntry",
    "ChunkPlan",
    # Pass 1
    "ASTIndexer",
    "GraphBuilder",
    "GraphRanker",
    "IndexRenderer",
    "IndexCache",
    # LLM
    "LLMClient",
    "TokenCounter",
    # Pass 2
    "ContextPlanner",
    "SliceFetcher",
    "OutputChunker",
    "OutputWriter",
    "SynthesisAgent",
    "SynthesisResult",
    # Phase 3 — Fallback
    "ChunkedFallback",
    "ReaderAgent",
    "AgentMemoryStore",
    # Orchestrator
    "TPCAOrchestrator",
    # Phase D — Worker agents
    "WorkerContext",
    "WorkerContextBuilder",
    "WorkerAgent",
    "TASK_TYPES",
    "detect_task_type",
    "get_template",
    # Phase E — Session management
    "SessionManager",
    "PlanStore",
    "PlanEvaluation",
    "PlanSection",
    "SessionPlan",
    "WorkerSummary",
    "PlannerAgent",
    "SubPlannerAgent",
    "EvaluatorAgent",
]

__version__ = "3.0.0"