"""
Graph Builder for constructing symbol relationship graphs.
Resolves cross-file edges in a second pass after all files are indexed.
"""
import networkx as nx
from collections import defaultdict
from typing import Optional

from ..config import TPCAConfig
from ..logging import StructuredLogger
from ..models import Symbol, SymbolGraph, PendingEdge


class GraphBuilder:
    """
    Builds a directed graph of symbol relationships.
    
    Edge types:
    - calls: (caller_id, callee_id, type='calls', weight=call_count)
    - inherits: (subclass_id, superclass_id, type='inherits', weight=2.0)
    - imports: (module_id, symbol_id, type='imports', weight=1.0)
    - member_of: (method_id, class_id, type='member_of', weight=1.0)
    - external_call: (caller_id, target_name, type='external_call', weight=0.1)
    """
    
    def __init__(self, config: TPCAConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self._graph: Optional[SymbolGraph] = None
        self._pending_edges: list[PendingEdge] = []
        self._symbols_by_file: dict[str, list[Symbol]] = defaultdict(list)
    
    def build(self, symbols: list[Symbol]) -> SymbolGraph:
        """
        Build a symbol graph from a list of symbols.

        Args:
            symbols: List of Symbol objects from ASTIndexer

        Returns:
            NetworkX DiGraph with symbols as nodes and relationships as edges
        """
        import time
        t0 = time.time()

        self._graph = nx.DiGraph()
        self._pending_edges = []
        self._symbols_by_file = defaultdict(list)

        # Add all symbols as nodes
        for sym in symbols:
            self._graph.add_node(sym.id, symbol=sym)
            self._symbols_by_file[sym.file].append(sym)

        self.logger.info('graph_nodes_added', count=len(symbols))

        # Add structural edges (class membership, inheritance)
        self._add_structural_edges(symbols)

        # Resolve call edges from AST-extracted call lists
        self._resolve_cross_file_edges(symbols)

        build_time_ms = int((time.time() - t0) * 1000)
        self.logger.info('graph_built',
                        node_count=self._graph.number_of_nodes(),
                        edge_count=self._graph.number_of_edges(),
                        build_time_ms=build_time_ms)

        return self._graph
    
    def _add_structural_edges(self, symbols: list[Symbol]):
        """
        Add structural edges (inheritance, class membership).
        These are resolved directly from symbol attributes.
        """
        # Build quick lookup: name -> symbol_id
        name_to_id = {}
        for sym in symbols:
            name_to_id[sym.qualified_name] = sym.id
            name_to_id[sym.name] = sym.id  # Also map simple name
        
        for sym in symbols:
            # Add member_of edges for methods
            if sym.type == 'method' and sym.parent_class:
                # Find the class symbol
                parent_qualified = sym.parent_class
                parent_id = name_to_id.get(parent_qualified)
                
                # Try file-scoped lookup
                if not parent_id:
                    parent_id = f"{sym.file}::{sym.parent_class}"
                    if parent_id not in self._graph:
                        parent_id = None
                
                if parent_id and self._graph.has_node(parent_id):
                    self._graph.add_edge(sym.id, parent_id,
                                       type='member_of', weight=1.0)
                    self.logger.debug('edge_added', type='member_of',
                                    source=sym.id, target=parent_id)
            
            # Add inheritance edges
            if sym.type == 'class' and sym.bases:
                for base_name in sym.bases:
                    # Try to resolve base class
                    base_id = self._resolve_symbol_name(base_name, sym.file, symbols)
                    if base_id:
                        self._graph.add_edge(sym.id, base_id,
                                           type='inherits', weight=2.0)
                        self.logger.debug('edge_added', type='inherits',
                                        source=sym.id, target=base_id)
                    else:
                        # External base class (not in our codebase)
                        self._graph.add_edge(sym.id, base_name,
                                           type='external_inherit', weight=0.1)
                        self.logger.debug('edge_external', type='inherits',
                                        source=sym.id, target=base_name)
    
    def _resolve_cross_file_edges(self, symbols: list[Symbol]):
        """
        Build call edges from the AST-extracted call lists stored on each Symbol.

        Resolution order:
        1. Qualified name match (e.g. 'Auth.validate_token')
        2. Simple name match — prefer same-file candidate, then first found
        3. Unresolvable names become external_call edges (weight 0.1)

        Edge weight = number of call sites (duplicate entries in Symbol.calls).
        """
        # Build lookup tables: simple name → [symbol_id, ...] and
        # qualified name → symbol_id
        by_simple: dict[str, list[str]] = defaultdict(list)
        by_qualified: dict[str, str] = {}

        for sym in symbols:
            by_simple[sym.name].append(sym.id)
            by_qualified[sym.qualified_name] = sym.id

        for sym in symbols:
            if not sym.calls:
                continue

            # Count how many times each callee name appears (= call weight)
            call_counts: dict[str, int] = {}
            for callee in sym.calls:
                call_counts[callee] = call_counts.get(callee, 0) + 1

            for callee_name, count in call_counts.items():
                # 1. Qualified name match
                target_id = by_qualified.get(callee_name)

                # 2. Simple name match
                if not target_id:
                    candidates = by_simple.get(callee_name, [])
                    if candidates:
                        # Prefer same-file symbol to reduce false cross-file edges
                        same_file = [c for c in candidates if c.startswith(sym.file + '::')]
                        target_id = same_file[0] if same_file else candidates[0]

                if target_id and target_id != sym.id and self._graph.has_node(target_id):
                    # Accumulate weight if edge already exists
                    if self._graph.has_edge(sym.id, target_id):
                        self._graph[sym.id][target_id]['weight'] += count
                    else:
                        self._graph.add_edge(sym.id, target_id,
                                             type='calls', weight=count)
                    self.logger.debug('edge_resolved',
                                      source=sym.id, target=target_id, weight=count)
                elif not target_id:
                    self._graph.add_edge(sym.id, callee_name,
                                         type='external_call', weight=0.1)
                    self.logger.debug('edge_external',
                                      source=sym.id, target=callee_name)
    
    def _resolve_symbol_name(self, name: str, current_file: str,
                            symbols: list[Symbol]) -> Optional[str]:
        """
        Resolve a symbol name to its full ID, trying file-local first.
        
        Args:
            name: Symbol name to resolve
            current_file: File where the reference occurs
            symbols: All symbols
        
        Returns:
            Full symbol ID or None
        """
        # Try file-local first
        file_scoped_id = f"{current_file}::{name}"
        if self._graph.has_node(file_scoped_id):
            return file_scoped_id
        
        # Try global lookup
        for sym in symbols:
            if sym.name == name or sym.qualified_name == name:
                return sym.id
        
        return None
    
    def get_symbol(self, symbol_id: str) -> Optional[Symbol]:
        """
        Get a Symbol object by ID.
        
        Args:
            symbol_id: Full symbol ID
        
        Returns:
            Symbol object or None
        """
        if not self._graph or not self._graph.has_node(symbol_id):
            return None
        return self._graph.nodes[symbol_id].get('symbol')
    
    def get_edges_by_type(self, edge_type: str) -> list[tuple]:
        """
        Get all edges of a specific type.
        
        Args:
            edge_type: Edge type (calls, inherits, member_of, etc.)
        
        Returns:
            List of (source, target, data) tuples
        """
        if not self._graph:
            return []
        
        return [(u, v, d) for u, v, d in self._graph.edges(data=True)
                if d.get('type') == edge_type]
