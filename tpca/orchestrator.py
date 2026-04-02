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
from .watch.file_watcher import FileWatcher


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

        # ── Phase 3 — ChunkedFallback (optional) ──────────────────────────────
        self._fallback = None
        if getattr(self._config, "fallback_enabled", False):
            try:
                from .fallback.chunked_pipeline import ChunkedFallback
                self._fallback = ChunkedFallback(self._config, self._logger, self._llm)
            except ImportError:
                pass  # fallback module not yet available — non-fatal

        # File watcher (started on demand via start_watching())
        self._watcher = FileWatcher(
            exclude_patterns=self._config.exclude_patterns,
        )

        self._logger.info(
            "orchestrator_init",
            output_mode=self._config.output_mode,
            synthesis_model=self._config.synthesis_model,
            llm_available=self._llm.available,
            watcher_available=self._watcher.available,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        source: str,
        task: str,
        task_keywords: Optional[list[str]] = None,
        budget_tokens: Optional[int] = None,
        resume_manifest: Optional[str] = None,
    ) -> dict:
        """
        Execute the full TPCA pipeline for a given task.

        Args:
            source:          Path to source directory, file, or list of files.
            task:            Natural language task description.
            task_keywords:   Optional keywords for Pass 1 PageRank bias.
                             If not given, extracted from the task string.
            budget_tokens:   Override the token budget for context slices.
            resume_manifest: Path to a manifest.json from a prior interrupted
                             run. When set (or via config.resume_manifest),
                             already-complete symbols are skipped.  [Phase 3]

        Returns:
            dict with keys:
              - 'output':  dict of {file: content} or {file: status}
              - 'stats':   performance and token-efficiency metrics
              - 'index':   the compact Pass 1 index string (for inspection)
              - 'log':     the OutputLog render (for inspection)
        """
        t_total = time.time()
        self._logger.info("orchestrator_run_start", source=str(source), task=task[:80])

        # ── Phase 3 — Resume: resolve manifest path and load prior state ───────
        resume_path = resume_manifest or getattr(self._config, "resume_manifest", None)
        _, prior_log, skip_symbols = self._load_resume_state(resume_path)

        # ── Pass 1 ─────────────────────────────────────────────────────────────
        t_pass1 = time.time()

        symbols = self._indexer.index(source)
        graph = self._builder.build(symbols)

        # Derive task keywords from task text if not explicitly given
        keywords = task_keywords or self._extract_keywords(task)
        graph = self._ranker.rank_symbols(graph, keywords)

        compact_index = self._renderer.render(graph)
        pass1_ms = int((time.time() - t_pass1) * 1000)

        # Estimate raw tokens available: sum of all source line ranges * avg tokens/line
        raw_token_estimate = sum(
            max((graph.nodes[n]["symbol"].end_line - graph.nodes[n]["symbol"].start_line + 1), 1) * 8
            for n in graph.nodes
            if graph.nodes[n].get("symbol")
        )

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
            prior_log=prior_log,               # Phase 3: rehydrated OutputLog or None
            skip_symbols=skip_symbols,         # Phase 3: set of already-complete IDs
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
            "raw_tokens_available": raw_token_estimate,
            "compression_ratio": round(compression_ratio, 1),
            "fallback_used": getattr(result, "fallback_used", False),  # Phase 3
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

    def run_coding_session(
        self,
        source: str,
        task: str,
        resume: bool = False,
    ) -> dict:
        """
        Execute the full coding-assistant pipeline for a task.

        Pass 1 runs first (deterministic indexing + ranking), then a
        SessionManager orchestrates planning → evaluation → worker dispatch.

        Args:
            source:  Path to source directory or file.
            task:    Natural language task description.
            resume:  If True, resume the existing plan from .tpca_plan.json
                     instead of creating a new one.

        Returns:
            dict with keys:
              - 'plan':      Completed SessionPlan dataclass.
              - 'summaries': list[WorkerSummary] in completion order.
              - 'stats':     Performance and section-completion metrics.
              - 'index':     Compact Pass 1 index string.
        """
        import time as _time
        from .tools.executor import ToolExecutor
        from .plan.plan_store import PlanStore
        from .session_manager import SessionManager

        t_total = _time.time()
        source_path = Path(source)
        source_root = str(source_path if source_path.is_dir() else source_path.parent)

        # ── Pass 1 ─────────────────────────────────────────────────────────────
        t_pass1 = _time.time()
        symbols = self._indexer.index(source)
        graph = self._builder.build(symbols)
        keywords = self._extract_keywords(task)
        graph = self._ranker.rank_symbols(graph, keywords)
        compact_index = self._renderer.render(graph)
        pass1_ms = int((_time.time() - t_pass1) * 1000)

        self._logger.info(
            "coding_session_pass1_complete",
            symbols=len(symbols),
            pass1_ms=pass1_ms,
        )

        # ── Session setup ───────────────────────────────────────────────────────
        executor = ToolExecutor(
            project_root=source_root,
            graph=graph,
            index_text=compact_index,
        )
        plan_store = PlanStore(project_root=source_root)
        session = SessionManager(
            plan_store=plan_store,
            llm=self._llm,
            config=self._config,
            graph=graph,
            compact_index=compact_index,
            project_root=source_root,
        )

        # ── Planning ────────────────────────────────────────────────────────────
        if resume:
            plan = session.resume_session()
            if plan is None:
                self._logger.warn(
                    "coding_session_no_plan_to_resume",
                    hint="No .tpca_plan.json found — starting fresh session",
                )
                plan = session.start_session(task)
        else:
            plan = session.start_session(task)

        # ── Worker dispatch ─────────────────────────────────────────────────────
        summaries = session.dispatch_workers(plan, executor)
        total_ms = int((_time.time() - t_total) * 1000)

        leaf_sections = plan.all_leaf_sections()
        return {
            "plan": plan,
            "summaries": summaries,
            "stats": {
                "pass1_time_ms": pass1_ms,
                "total_time_ms": total_ms,
                "symbols_indexed": len(symbols),
                "sections_total": len(leaf_sections),
                "sections_complete": sum(
                    1 for s in leaf_sections if s.status == "COMPLETE"
                ),
                "sections_failed": sum(
                    1 for s in leaf_sections if s.status == "BLOCKED"
                ),
            },
            "index": compact_index,
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

    # ── File watching ──────────────────────────────────────────────────────────

    def start_watching(self, source: str) -> bool:
        """
        Start watching *source* for file changes.

        When a source file is modified the file's AST cache entry is
        invalidated so the next run() / run_pass1_only() call re-parses it.

        Args:
            source: Path to the directory (or a file inside it) to watch.

        Returns:
            True if the watcher started successfully, False if watchdog is
            not installed or the path does not exist.
        """
        source_path = Path(source)
        watch_dir = source_path if source_path.is_dir() else source_path.parent
        if not watch_dir.exists():
            self._logger.warn("watcher_bad_path", path=str(watch_dir))
            return False
        started = self._watcher.start(str(watch_dir), self._on_file_changed)
        if started:
            self._logger.info("watcher_started", directory=str(watch_dir))
        else:
            self._logger.warn(
                "watcher_unavailable",
                hint="pip install watchdog",
            )
        return started

    def stop_watching(self) -> None:
        """Stop the file watcher if running."""
        if self._watcher.running:
            self._watcher.stop()
            self._logger.info("watcher_stopped")

    @property
    def watcher(self) -> FileWatcher:
        """Expose the FileWatcher instance for REPL status queries."""
        return self._watcher

    @property
    def llm(self) -> LLMClient:
        """Expose the LLMClient for external use (e.g. SessionManager construction)."""
        return self._llm

    def _on_file_changed(self, path: str) -> None:
        """Callback invoked by FileWatcher on every source file change."""
        self._cache.invalidate(path)
        self._logger.info("file_changed", path=path, action="cache_invalidated")

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

    # ── Phase 3 — Resume helpers ───────────────────────────────────────────────

    def _load_resume_state(self, manifest_path: Optional[str]):
        """
        Load a prior manifest and rehydrate its OutputLog.

        Returns (manifest, output_log, skip_symbols) where skip_symbols is
        the set of fully-qualified symbol IDs that are already complete.
        Returns (None, None, set()) when manifest_path is None or unreadable.
        """
        if not manifest_path:
            return None, None, set()

        try:
            from .models.output import OutputManifest, OutputLog
            manifest = OutputManifest.load(manifest_path)
            output_log = OutputLog.from_manifest(manifest)
            skip_symbols = self._complete_symbols(manifest)
            self._logger.info(
                "resume_manifest_loaded",
                path=manifest_path,
                files_total=len(manifest.files),
                files_complete=sum(1 for e in manifest.files if e.status == "complete"),
                symbols_skipping=len(skip_symbols),
            )
            return manifest, output_log, skip_symbols
        except FileNotFoundError:
            self._logger.warn("resume_manifest_not_found", path=manifest_path)
            return None, None, set()
        except Exception as exc:
            self._logger.warn("resume_manifest_failed", path=manifest_path, error=str(exc))
            return None, None, set()

    @staticmethod
    def _complete_symbols(manifest) -> set:
        """Return the set of symbol IDs already processed in complete entries."""
        if manifest is None:
            return set()
        completed: set = set()
        for entry in manifest.files:
            if entry.status == "complete":
                completed.update(entry.symbols_processed)
        return completed