"""
Index Renderer - generates compact text representation of symbol graphs.
"""
from collections import defaultdict
from ..config import TPCAConfig
from ..logging import StructuredLogger
from ..models import SymbolGraph


class IndexRenderer:
    """
    Renders a SymbolGraph to compact text format for LLM consumption.
    
    Output includes:
    - Symbols organized by file
    - Rank tier annotations [CORE], [SUPPORT], [PERIPHERAL]
    - Method signatures with truncated docstrings
    - Cross-file import relationships
    
    Typical output: 1,000-3,000 tokens for a 10,000-line codebase.
    """
    
    def __init__(self, config: TPCAConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
    
    def render(self, graph: SymbolGraph) -> str:
        """
        Render graph to compact text format.
        
        Args:
            graph: Symbol graph with PageRank scores and tiers
        
        Returns:
            Compact text index
        """
        if graph.number_of_nodes() == 0:
            return "# Empty codebase - no symbols found\n"
        
        # Group symbols by file
        symbols_by_file = defaultdict(list)
        for node_id in graph.nodes():
            symbol = graph.nodes[node_id].get('symbol')
            if symbol:
                symbols_by_file[symbol.file].append((node_id, symbol))
        
        # Sort files alphabetically
        sorted_files = sorted(symbols_by_file.keys())
        
        # Render each file section
        sections = []
        for filepath in sorted_files:
            section = self._render_file_section(filepath,
                                              symbols_by_file[filepath],
                                              graph)
            sections.append(section)
        
        # Add cross-file imports section
        imports_section = self._render_cross_file_imports(graph)
        if imports_section:
            sections.append(imports_section)
        
        result = '\n'.join(sections)
        
        # Log statistics
        token_estimate = len(result) // 4  # Rough estimate
        self.logger.info('index_rendered',
                        files=len(sorted_files),
                        symbols=graph.number_of_nodes(),
                        char_count=len(result),
                        token_estimate=token_estimate)
        
        return result
    
    def _render_file_section(self, filepath: str,
                            symbols: list[tuple[str, 'Symbol']],
                            graph: SymbolGraph) -> str:
        """
        Render symbols for a single file.
        
        Format:
        ## filepath
        class ClassName(BaseClass)           [CORE]
          + public_method(args) -> return    # docstring
          - _private_method()                [PERIPHERAL]
        function_name(args)                  [SUPPORT]
        """
        lines = [f"## {filepath}"]
        
        # Sort symbols: classes first, then functions, by PageRank within each group
        classes = [(nid, sym) for nid, sym in symbols if sym.type == 'class']
        functions = [(nid, sym) for nid, sym in symbols
                    if sym.type == 'function']
        methods = [(nid, sym) for nid, sym in symbols if sym.type == 'method']
        
        # Sort by PageRank (descending)
        classes.sort(key=lambda x: x[1].pagerank, reverse=True)
        functions.sort(key=lambda x: x[1].pagerank, reverse=True)
        
        # Render classes with their methods
        for class_id, class_sym in classes:
            tier = graph.nodes[class_id].get('tier', '')
            tier_label = f"[{tier}]" if tier else ""
            
            # Class signature
            bases_str = f"({', '.join(class_sym.bases)})" if class_sym.bases else ""
            lines.append(f"class {class_sym.name}{bases_str}".ljust(40) + f" {tier_label}")
            
            # Find methods of this class
            class_methods = [
                (nid, sym) for nid, sym in methods
                if sym.parent_class == class_sym.name
            ]
            class_methods.sort(key=lambda x: x[1].pagerank, reverse=True)
            
            for method_id, method_sym in class_methods:
                method_tier = graph.nodes[method_id].get('tier', '')
                method_tier_label = f"[{method_tier}]" if method_tier else ""
                
                # Method visibility prefix
                prefix = "  - " if method_sym.name.startswith('_') else "  + "
                
                # Method signature (simplified)
                sig = self._simplify_signature(method_sym.signature)
                
                # Docstring preview
                doc_preview = ""
                if method_sym.docstring:
                    doc_preview = f"  # {method_sym.docstring[:60]}"
                
                method_line = f"{prefix}{sig}".ljust(40) + f" {method_tier_label}"
                lines.append(method_line)
                if doc_preview:
                    lines.append(doc_preview)
        
        # Render standalone functions
        for func_id, func_sym in functions:
            tier = graph.nodes[func_id].get('tier', '')
            tier_label = f"[{tier}]" if tier else ""
            
            sig = self._simplify_signature(func_sym.signature)
            func_line = sig.ljust(40) + f" {tier_label}"
            lines.append(func_line)
            
            if func_sym.docstring:
                lines.append(f"  # {func_sym.docstring[:60]}")
        
        return '\n'.join(lines)
    
    def _simplify_signature(self, signature: str) -> str:
        """
        Simplify a function signature for compact display.
        
        Removes 'def ' prefix and truncates long parameter lists.
        """
        # Remove 'def ' prefix
        sig = signature.replace('def ', '')
        
        # Truncate if too long
        if len(sig) > 50:
            # Try to keep just the name and first few params
            if '(' in sig:
                name_part, rest = sig.split('(', 1)
                params = rest.split(')')[0]
                if len(params) > 30:
                    params = params[:27] + '...'
                sig = f"{name_part}({params})"
        
        return sig
    
    def _render_cross_file_imports(self, graph: SymbolGraph) -> str:
        """
        Render cross-file import/call relationships.
        
        Format:
        ## Cross-file references
        router.py → auth.py::Auth.validate_token
        auth.py → utils.py::generate_token
        """
        # Find cross-file edges
        cross_file_edges = []
        
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('type')
            if edge_type in ['calls', 'imports', 'inherits']:
                # Get symbols
                source_sym = graph.nodes[u].get('symbol') if graph.has_node(u) else None
                target_sym = graph.nodes[v].get('symbol') if graph.has_node(v) else None
                
                if source_sym and target_sym:
                    # Check if cross-file
                    if source_sym.file != target_sym.file:
                        cross_file_edges.append((
                            source_sym.file,
                            v,  # target symbol ID
                            edge_type
                        ))
        
        if not cross_file_edges:
            return ""
        
        lines = ["", "## Cross-file references"]
        
        # Group by source file
        by_source = defaultdict(list)
        for src_file, tgt_id, edge_type in cross_file_edges:
            by_source[src_file].append((tgt_id, edge_type))
        
        for src_file in sorted(by_source.keys()):
            for tgt_id, edge_type in by_source[src_file]:
                # Format: short_filename → target_id
                src_short = src_file.split('/')[-1]
                lines.append(f"{src_short} → {tgt_id}")
        
        return '\n'.join(lines)
    
    def render_compact_summary(self, graph: SymbolGraph) -> str:
        """
        Render a very compact summary (for logging or quick display).
        
        Returns:
            One-line summary like "5 files, 42 symbols (8 CORE, 15 SUPPORT, 19 PERIPHERAL)"
        """
        tier_counts = defaultdict(int)
        files = set()
        
        for node_id in graph.nodes():
            symbol = graph.nodes[node_id].get('symbol')
            if symbol:
                files.add(symbol.file)
                tier = graph.nodes[node_id].get('tier', 'UNKNOWN')
                tier_counts[tier] += 1
        
        return (f"{len(files)} files, {graph.number_of_nodes()} symbols "
                f"({tier_counts['CORE']} CORE, {tier_counts['SUPPORT']} SUPPORT, "
                f"{tier_counts['PERIPHERAL']} PERIPHERAL)")
