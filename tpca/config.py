"""
TPCAConfig — unified configuration for all TPCA phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .logging.log_config import LogConfig


@dataclass
class TPCAConfig:
    # ── Pass 1 — AST Indexing ─────────────────────────────────────────────────
    languages: list[str] = field(default_factory=lambda: ["python"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "__pycache__", ".git", "node_modules", "dist", ".venv"
        ]
    )
    cache_dir: str = ".tpca_cache"
    cache_enabled: bool = True

    # ── Graph Ranking ──────────────────────────────────────────────────────────
    pagerank_alpha: float = 0.85
    top_n_symbols: int = 50

    # ── Token Counting ─────────────────────────────────────────────────────────
    tokenizer: str = "cl100k_base"       # tiktoken encoding
    model_context_window: int = 8192
    context_budget_pct: float = 0.70     # fraction of context window for slices

    # ── LLM ───────────────────────────────────────────────────────────────────
    provider: str = "anthropic"          # anthropic | ollama
    reader_model: str = "claude-haiku-4-5-20251001"    # lightweight planning model
    synthesis_model: str = "claude-sonnet-4-6"          # powerful synthesis model
    max_planner_retries: int = 3

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_reader_model: str = "llama3.2"      # lightweight model for planning
    ollama_synthesis_model: str = "llama3.2"   # swap for a larger model if available

    # ── Output ────────────────────────────────────────────────────────────────
    output_mode: str = "inline"          # inline | single_file | mirror | per_symbol
    output_dir: str = "./tpca_output"
    max_synthesis_iterations: int = 20

    # ── Fallback (Phase 3) ────────────────────────────────────────────────────
    fallback_chunk_tokens: int = 1800
    fallback_overlap_tokens: int = 150

    # ── Logging ───────────────────────────────────────────────────────────────
    log: LogConfig = field(default_factory=LogConfig)

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def active_reader_model(self) -> str:
        """Returns the correct reader model for the active provider."""
        return self.ollama_reader_model if self.provider == "ollama" else self.reader_model

    @property
    def active_synthesis_model(self) -> str:
        """Returns the correct synthesis model for the active provider."""
        return self.ollama_synthesis_model if self.provider == "ollama" else self.synthesis_model