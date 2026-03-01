"""
TPCAOrchestrator: top-level entry point that wires together the entire
two-pass pipeline.

  Pass 1 (deterministic, zero LLM):
    ASTIndexer → GraphBuilder → GraphRanker → IndexRenderer → IndexCache

  Pass 2 (LLM-driven synthesis):
    ContextPlanner → SliceFetcher → SynthesisAgent (OutputChunker loop)
    → OutputWriter → OutputManifest

Usage:
    from tpca import TPCAOrchestrator, TPCAConfig, LogConfig

    config = TPCAConfig(
        synthesis_model='claude-sonnet-4-6',
        reader_model='claude-haiku-4-5-20251001',
        output_mode='inline',
    )
    orchestrator = TPCAOrchestrator(config=config)
    result = orchestrator.run(
        source='./my_project/src',
        task='Document every public method with parameter types and return values.',
    )
    print(result['stats'])
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .config import TPCAConfig
from .logging.structured_logger import StructuredLogger
from .pass1.ast_indexer import ASTIndexer
from .pass1.graph_builder import GraphBuilder
from .pass1.graph_ranker import GraphRanker
from .pass1.index_renderer import IndexRenderer
from .cache.index_cache import IndexCache
from .llm.client import LLMClient
from .pass2.context_planner import ContextPlanner
from .pass2.slice_fetcher import SliceFetcher
from .pass2.synthesis_agent import SynthesisAgent, SynthesisResult


class TPCAOrchestrator:
    """
    End-to-end orchestrator for the Two-Pass Context Agent.

    Instantiate once and call .run() for each task. Pass 1 results are
    cached to disk and reused when source files are unchanged.
    """

    def __init__(self, config: Optional[TPCAConfig] = None):
        self._config = config or TPCAConfig()
        self._logger = StructuredLogger(self._config.log)
        self._cache = IndexCache(self._config, self._logger)
        self._llm = LLMClient(self._config, self._logger)

        # Pass 1
        self._indexer = ASTIndexer(self._config, self._logger, self._cache)
        self._builder = GraphBuilder(self._config, self._logger)
        self._ranker = GraphRanker(self._config, self._logger)
        self._renderer = IndexRenderer(self._config, self._logger)

        # Pass 2
        self._planner = ContextPlanner(self._config, self._logger, self._llm)
        self._fetcher = SliceFetcher(self._config, self._logger, self._llm)
        self._synthesis = SynthesisAgent(
            self._config,
            self._logger,
            self._llm,
            planner=self._planner,
            fetcher=self._fetcher,
        )

        self._logger.info(
            "orchestrator_init",
            output_mode=self._config.output_mode,
            synthesis_model=self._config.synthesis_model,
            llm_available=self._llm.available,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        source: str,
        task: str,
        task_keywords: Optional[list[str]] = None,
        budget_tokens: Optional[int] = None,
    ) -> dict:
        """
        Execute the full TPCA pipeline for a given task.

        Args:
            source:        Path to source directory, file, or list of files.
            task:          Natural language task description.
            task_keywords: Optional keywords for Pass 1 PageRank bias.
                           If not given, extracted from the task string.
            budget_tokens: Override the token budget for context slices.

        Returns:
            dict with keys:
              - 'output':  dict of {file: content} or {file: status}
              - 'stats':   performance and token-efficiency metrics
              - 'index':   the compact Pass 1 index string (for inspection)
              - 'log':     the OutputLog render (for inspection)
        """
        t_total = time.time()
        self._logger.info("orchestrator_run_start", source=str(source), task=task[:80])

        # ── Pass 1 ─────────────────────────────────────────────────────────────
        t_pass1 = time.time()

        symbols = self._indexer.index(source)
        graph = self._builder.build(symbols)

        # Derive task keywords from task text if not explicitly given
        keywords = task_keywords or self._extract_keywords(task)
        graph = self._ranker.rank_symbols(graph, keywords)

        compact_index = self._renderer.render(graph)
        pass1_ms = int((time.time() - t_pass1) * 1000)

        # Estimate raw tokens available (chars in all source files / 4)
        raw_token_estimate = sum(
            len(graph.nodes[n].get("symbol").signature or "")
            for n in graph.nodes
            if graph.nodes[n].get("symbol")
        ) * 10  # rough estimate

        self._logger.info(
            "pass1_complete",
            symbols_indexed=len(symbols),
            graph_nodes=graph.number_of_nodes(),
            graph_edges=graph.number_of_edges(),
            pass1_ms=pass1_ms,
        )

        # Check if LLM is available before attempting Pass 2
        if not self._llm.available:
            self._logger.warn(
                "pass2_skipped",
                reason="LLM client unavailable — returning Pass 1 index only",
            )
            return {
                "output": {},
                "stats": {
                    "pass1_time_ms": pass1_ms,
                    "files_indexed": len({
                        graph.nodes[n]["symbol"].file
                        for n in graph.nodes
                        if graph.nodes[n].get("symbol")
                    }),
                    "symbols_indexed": len(symbols),
                    "pass2_skipped": True,
                    "reason": "LLM unavailable",
                },
                "index": compact_index,
                "log": "",
            }

        # ── Pass 2 ─────────────────────────────────────────────────────────────
        source_root = str(Path(source)) if Path(source).is_dir() else str(Path(source).parent)

        result: SynthesisResult = self._synthesis.run(
            task=task,
            compact_index=compact_index,
            graph=graph,
            source_root=source_root,
            budget_tokens=budget_tokens,
        )

        total_ms = int((time.time() - t_total) * 1000)

        # Compute compression ratio
        tokens_sent = result.stats.get("tokens_sent_to_llm", 1)
        compression_ratio = raw_token_estimate / max(tokens_sent, 1)

        stats = {
            "pass1_time_ms": pass1_ms,
            "total_time_ms": total_ms,
            "files_indexed": len({
                graph.nodes[n]["symbol"].file
                for n in graph.nodes
                if graph.nodes[n].get("symbol")
            }),
            "symbols_indexed": len(symbols),
            "compression_ratio": round(compression_ratio, 1),
            **result.stats,
        }

        self._logger.info("orchestrator_run_complete", total_ms=total_ms)

        return {
            "output": result.output,
            "stats": stats,
            "index": compact_index,
            "log": result.output_log.render_compact(),
            "manifest": result.manifest,
        }

    def run_pass1_only(
        self,
        source: str,
        task_keywords: Optional[list[str]] = None,
    ) -> dict:
        """
        Run only Pass 1 (indexing + ranking). Useful for inspection or
        when you want to use the index without LLM synthesis.

        Returns:
            dict with 'graph', 'index', 'symbols', and 'stats'.
        """
        symbols = self._indexer.index(source)
        graph = self._builder.build(symbols)
        keywords = task_keywords or []
        if keywords:
            graph = self._ranker.rank_symbols(graph, keywords)
        compact_index = self._renderer.render(graph)
        return {
            "graph": graph,
            "index": compact_index,
            "symbols": symbols,
            "stats": {
                "symbols_indexed": len(symbols),
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
            },
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(task: str) -> list[str]:
        """
        Extract candidate keywords from the task string.
        Simple heuristic: words longer than 4 characters, excluding stopwords.
        """
        stopwords = {
            "every", "their", "with", "that", "from", "this", "have",
            "will", "should", "each", "into", "about", "which", "them",
            "also", "when", "where", "what", "document", "write", "create",
        }
        words = task.lower().replace(",", " ").replace(".", " ").split()
        return [w for w in words if len(w) > 4 and w not in stopwords][:10]
