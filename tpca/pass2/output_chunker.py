"""
OutputChunker: splits large synthesis tasks at logical code boundaries
(class or top-level function) and maintains a compact OutputLog as
working memory across synthesis calls.

Key invariant: each subsequent synthesis call receives the OutputLog
(~50-100 tokens/entry), NOT the full prior raw output.
This keeps context bounded regardless of total output size.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

from ..models.output import OutputLog, OutputChunk
from ..models.chunk_plan import ChunkPlan
from ..models.slice import SliceRequest


class OutputChunker:
    """
    Manages the synthesis loop for tasks that may produce large outputs.

    Boundaries are derived from the SymbolGraph (not raw output text).
    Symbols are processed in dependency order (topological sort on the
    relevant subgraph). Each chunk corresponds to one top-level symbol
    (a class with all its methods, or a module-level function).
    """

    def __init__(
        self,
        graph,
        output_log: OutputLog,
        config,
        logger,
        slice_request: Optional[SliceRequest] = None,
    ):
        self._graph = graph
        self._log = output_log
        self._config = config
        self._logger = logger
        self._topo_order = self._compute_topo_order(slice_request)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_next_chunk(
        self,
        compact_index: str,
        slices_by_symbol: dict,
        rationale: str = "",
    ) -> Optional[ChunkPlan]:
        """
        Return the next symbol group to synthesise, or None if complete.

        Args:
            compact_index:     The Pass 1 rendered index string.
            slices_by_symbol:  {symbol_id -> Slice} for context.
            rationale:         The ContextPlanner's rationale string.

        Returns:
            ChunkPlan if work remains, None when all symbols are done.
        """
        completed = self._log.completed_symbols()
        pending = [s for s in self._topo_order if s not in completed]

        if not pending:
            self._logger.info("chunker_all_complete", total=len(self._topo_order))
            return None

        symbol_id = pending[0]
        chunk_id = len(self._log.entries)

        # Gather relevant slices for this symbol and its related symbols
        context_slices = self._build_context_slices(symbol_id, slices_by_symbol)

        self._logger.info(
            "output_chunk_start",
            chunk_id=chunk_id,
            symbol_id=symbol_id,
            pending_remaining=len(pending),
        )

        return ChunkPlan(
            chunk_id=chunk_id,
            symbol_id=symbol_id,
            prior_log=self._log.render_compact(),
            context_pkg={
                "index": compact_index,
                "slices": context_slices,
                "rationale": rationale,
            },
        )

    def record_chunk(
        self,
        chunk_id: int,
        symbol_id: str,
        raw_output: str,
        summary: str,
        status: str = "complete",
    ) -> OutputChunk:
        """
        Called after each synthesis call to record the completed chunk.

        Args:
            chunk_id:   The chunk ID from ChunkPlan.
            symbol_id:  The symbol that was processed.
            raw_output: The raw synthesis output (used only for token counting).
            summary:    One-line summary extracted from the LLM response.
            status:     'complete' | 'partial' | 'failed'.

        Returns:
            The recorded OutputChunk.
        """
        from ..llm.client import TokenCounter
        token_count = TokenCounter().count(raw_output)

        chunk = OutputChunk(
            chunk_id=chunk_id,
            symbol_id=symbol_id,
            summary=summary,
            status=status,
            token_count=token_count,
        )
        self._log.add(chunk)

        self._logger.info(
            "output_chunk_end",
            chunk_id=chunk_id,
            symbol_id=symbol_id,
            summary_preview=summary[:80],
            status=status,
            token_count=token_count,
        )
        return chunk

    def is_complete(self) -> bool:
        """Return True when all symbols in topo order have been completed."""
        completed = self._log.completed_symbols()
        return all(s in completed for s in self._topo_order)

    @property
    def output_log(self) -> OutputLog:
        return self._log

    @property
    def completed_count(self) -> int:
        return len(self._log.completed_symbols())

    @property
    def total_count(self) -> int:
        return len(self._topo_order)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_topo_order(
        self, slice_request: Optional[SliceRequest]
    ) -> list[str]:
        """
        Compute topological processing order for the relevant symbols.

        Uses only the symbols from the SliceRequest if provided;
        falls back to all graph nodes otherwise.
        """
        if slice_request is not None:
            relevant = set(slice_request.all_symbols)
            # Include members (methods) of requested classes
            for node in list(relevant):
                if node in self._graph.nodes:
                    for nbr in self._graph.successors(node):
                        edge_data = self._graph.edges[node, nbr]
                        if edge_data.get("type") == "member_of":
                            relevant.add(nbr)
            subgraph = self._graph.subgraph(relevant)
        else:
            subgraph = self._graph

        # Topological sort; fall back to PageRank order if graph has cycles
        try:
            order = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            self._logger.warn(
                "chunker_cycle_detected",
                hint="Falling back to PageRank order",
            )
            order = sorted(
                subgraph.nodes,
                key=lambda n: subgraph.nodes[n].get("pagerank", 0.0),
                reverse=True,
            )

        # Filter to only nodes that are "top-level" (classes or module functions)
        # Methods (member_of edges) are grouped with their class
        top_level = []
        member_symbols = set()
        for node in subgraph.nodes:
            for nbr in subgraph.predecessors(node):
                if subgraph.edges.get((nbr, node), {}).get("type") == "member_of":
                    member_symbols.add(node)

        for node in order:
            if node not in member_symbols:
                top_level.append(node)

        # If filtering removed everything (e.g. all are methods), use full order
        return top_level if top_level else order

    def _build_context_slices(
        self, symbol_id: str, slices_by_symbol: dict
    ) -> list:
        """Return relevant slices for the current symbol."""
        result = []
        # Include this symbol's slice
        if symbol_id in slices_by_symbol:
            result.append(slices_by_symbol[symbol_id])
        # Include member slices (methods of this class)
        for nbr in self._graph.successors(symbol_id):
            edge = self._graph.edges.get((symbol_id, nbr), {})
            if edge.get("type") == "member_of" and nbr in slices_by_symbol:
                result.append(slices_by_symbol[nbr])
        return result
