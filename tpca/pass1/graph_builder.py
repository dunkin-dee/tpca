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
        
        # Resolve cross-file call edges
        self._resolve_cross_file_edges(symbols)
        
        self.logger.info('graph_build_complete',
                        nodes=self._graph.number_of_nodes(),
                        edges=self._graph.number_of_edges())
        
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
        Resolve call relationships between symbols.
        This is simplified for Phase 1 - a more complete implementation
        would parse function bodies for actual calls.
        
        For now, we create edges based on heuristics:
        - Methods in the same class call each other
        - Public methods are called by other files
        """
        # Build namespace for quick lookup
        namespace = {}
        for sym in symbols:
            namespace[sym.name] = sym.id
            namespace[sym.qualified_name] = sym.id
        
        # For Phase 1, add some basic call edges
        # In Phase 2/3, this would be replaced with actual call graph analysis
        methods_by_class = defaultdict(list)
        for sym in symbols:
            if sym.type == 'method' and sym.parent_class:
                class_id = f"{sym.file}::{sym.parent_class}"
                methods_by_class[class_id].append(sym)
        
        # Add intra-class method calls (heuristic: methods call each other)
        for class_id, methods in methods_by_class.items():
            if len(methods) > 1:
                # Create weighted edges based on method names
                for i, method1 in enumerate(methods):
                    for method2 in methods[i+1:]:
                        # Heuristic: private methods are called by public ones
                        if method2.name.startswith('_') and not method1.name.startswith('_'):
                            self._graph.add_edge(method1.id, method2.id,
                                               type='calls', weight=1)
                            self.logger.debug('edge_inferred', type='calls',
                                            source=method1.id, target=method2.id)
        
        # Note: A complete implementation would:
        # 1. Parse function bodies with tree-sitter
        # 2. Find all call expressions
        # 3. Resolve them against the namespace
        # 4. Track call counts
        # This is sufficient for Phase 1 demonstration
    
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
