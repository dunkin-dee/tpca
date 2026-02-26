"""
AST Indexer for extracting symbols from source code using Tree-sitter.
Supports multi-file input (single file, list of files, or directory).
"""
import os
from pathlib import Path
from typing import Union
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from ..config import TPCAConfig
from ..logging import StructuredLogger
from ..models import Symbol


class ASTIndexer:
    """
    Parses source files using Tree-sitter to extract structural symbols.
    
    Runs once per source, caches to disk, costs zero LLM tokens.
    Accepts three input forms:
    - Single file path (str)
    - List of file paths
    - Directory path (walks recursively)
    """
    
    LANGUAGE_EXTENSIONS = {
        'python': ['.py'],
        'javascript': ['.js', '.jsx'],
        'typescript': ['.ts', '.tsx'],
    }
    
    def __init__(self, config: TPCAConfig, logger: StructuredLogger, cache=None):
        self.config = config
        self.logger = logger
        self.cache = cache
        
        # Initialize Tree-sitter parser for Python
        PY_LANGUAGE = Language(tspython.language(), 'python')
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)        
        # Load query for Python
        query_file = Path(__file__).parent / 'queries' / 'python.scm'
        with open(query_file, 'r') as f:
            query_text = f.read()
        self.query = PY_LANGUAGE.query(query_text)
    
    def index(self, source: Union[str, list[str]]) -> list[Symbol]:
        """
        Index source files and return list of symbols.
        
        Args:
            source: Single file path, list of paths, or directory path
        
        Returns:
            List of Symbol objects
        """
        paths = self._resolve_paths(source)
        symbols: list[Symbol] = []
        
        for path in paths:
            lang = self._detect_language(path)
            if lang not in self.config.languages:
                self.logger.debug('skip_file', file=path, reason='language_unsupported')
                continue
            
            # Check cache
            if self.cache and self.config.cache_enabled:
                cached = self.cache.get(path)
                if cached is not None:
                    self.logger.debug('cache_hit', file=path)
                    symbols.extend(cached)
                    continue
            
            # Parse file
            file_symbols = self._parse_file(path, lang)
            
            # Update cache
            if self.cache and self.config.cache_enabled:
                self.cache.set(path, file_symbols)
            
            symbols.extend(file_symbols)
            self.logger.info('file_indexed', file=path, symbol_count=len(file_symbols))
        
        return symbols
    
    def _resolve_paths(self, source: Union[str, list[str]]) -> list[str]:
        """
        Resolve input to list of file paths.
        
        Args:
            source: Single path, list of paths, or directory
        
        Returns:
            List of absolute file paths
        """
        if isinstance(source, list):
            return [str(Path(p).resolve()) for p in source]
        
        path = Path(source)
        if not path.exists():
            self.logger.error('path_not_found', path=source)
            return []
        
        if path.is_file():
            return [str(path.resolve())]
        
        # Directory - walk recursively
        if path.is_dir():
            files = []
            for root, dirs, filenames in os.walk(path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude(d)]
                
                for filename in filenames:
                    if self._should_exclude(filename):
                        continue
                    filepath = Path(root) / filename
                    if self._detect_language(str(filepath)) in self.config.languages:
                        files.append(str(filepath.resolve()))
            
            self.logger.info('directory_walk', path=source, files_found=len(files))
            return files
        
        return []
    
    def _should_exclude(self, name: str) -> bool:
        """Check if a path component should be excluded."""
        for pattern in self.config.exclude_patterns:
            if pattern.startswith('*'):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in name:
                return True
        return False
    
    def _detect_language(self, path: str) -> str:
        """
        Detect language from file extension.
        
        Args:
            path: File path
        
        Returns:
            Language name (e.g., 'python') or 'unknown'
        """
        ext = Path(path).suffix
        for lang, exts in self.LANGUAGE_EXTENSIONS.items():
            if ext in exts:
                return lang
        return 'unknown'
    
    def _parse_file(self, path: str, language: str) -> list[Symbol]:
        """
        Parse a single file and extract symbols.
        
        Args:
            path: File path
            language: Language name
        
        Returns:
            List of Symbol objects
        """
        try:
            with open(path, 'rb') as f:
                source_code = f.read()
            
            tree = self.parser.parse(source_code)
            
            if language == 'python':
                return self._extract_python_symbols(path, source_code, tree)
            else:
                self.logger.warn('unsupported_language', file=path, language=language)
                return []
        
        except Exception as e:
            self.logger.error('parse_error', file=path, error=str(e))
            return []
    
    def _extract_python_symbols(self, filepath: str, source_code: bytes,
                                tree) -> list[Symbol]:
        """
        Extract symbols from Python AST.
        
        Args:
            filepath: Path to source file
            source_code: Raw bytes of source
            tree: Tree-sitter parse tree
        
        Returns:
            List of Symbol objects
        """
        symbols = []
        rel_path = self._make_relative_path(filepath)
        
        # Get source lines for extracting text
        source_lines = source_code.decode('utf-8').split('\n')
        
        # Query the tree — returns [(node, capture_name), ...] in 0.21.x
        captures = self.query.captures(tree.root_node)
        
        # Track context for methods
        current_class = None
        class_stack = []
        
        for node, capture_name in captures:
            try:
                if capture_name == 'class.def':
                    sym = self._extract_class(node, rel_path, source_lines)
                    if sym:
                        symbols.append(sym)
                        current_class = sym.name
                        class_stack.append(sym.name)
                
                elif capture_name == 'function.def':
                    # Determine if it's a method or standalone function
                    parent_class = class_stack[-1] if class_stack else None
                    sym = self._extract_function(node, rel_path, source_lines, parent_class)
                    if sym:
                        symbols.append(sym)
            
            except Exception as e:
                self.logger.debug('symbol_extraction_error',
                                file=filepath, capture=capture_name, error=str(e))
        
        return symbols
        
    def _extract_class(self, node, filepath: str, source_lines: list[str]) -> Symbol:
        """Extract a class symbol."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._node_text(name_node, source_lines)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # Extract bases
        bases = []
        superclasses_node = node.child_by_field_name('superclasses')
        if superclasses_node:
            for child in superclasses_node.children:
                if child.type == 'identifier':
                    bases.append(self._node_text(child, source_lines))
        
        # Extract docstring
        docstring = self._extract_docstring(node, source_lines)
        
        # Build signature
        bases_str = f"({', '.join(bases)})" if bases else ""
        signature = f"class {name}{bases_str}"
        
        symbol_id = f"{filepath}::{name}"
        
        return Symbol(
            id=symbol_id,
            type='class',
            name=name,
            qualified_name=name,
            file=filepath,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            bases=bases
        )
    
    def _extract_function(self, node, filepath: str, source_lines: list[str],
                         parent_class: str = None) -> Symbol:
        """Extract a function or method symbol."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._node_text(name_node, source_lines)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # Extract parameters
        params_node = node.child_by_field_name('parameters')
        params_text = self._node_text(params_node, source_lines) if params_node else "()"
        
        # Extract return type
        return_type_node = node.child_by_field_name('return_type')
        return_type = ""
        if return_type_node:
            return_type = f" -> {self._node_text(return_type_node, source_lines)}"
        
        # Extract docstring
        docstring = self._extract_docstring(node, source_lines)
        
        # Build signature
        signature = f"def {name}{params_text}{return_type}"
        
        # Determine type and qualified name
        if parent_class:
            sym_type = 'method'
            qualified_name = f"{parent_class}.{name}"
            symbol_id = f"{filepath}::{qualified_name}"
        else:
            sym_type = 'function'
            qualified_name = name
            symbol_id = f"{filepath}::{name}"
        
        return Symbol(
            id=symbol_id,
            type=sym_type,
            name=name,
            qualified_name=qualified_name,
            file=filepath,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            parent_class=parent_class
        )
    
    def _extract_docstring(self, node, source_lines: list[str]) -> str:
        """Extract docstring from function or class body."""
        body_node = node.child_by_field_name('body')
        if not body_node:
            return ""
        
        # Docstring is first expression_statement with a string
        for child in body_node.children:
            if child.type == 'expression_statement':
                for expr_child in child.children:
                    if expr_child.type == 'string':
                        docstring = self._node_text(expr_child, source_lines)
                        # Remove quotes and limit to 120 chars
                        docstring = docstring.strip('"\'').strip()
                        if len(docstring) > 120:
                            docstring = docstring[:117] + '...'
                        return docstring
        return ""
    
    def _node_text(self, node, source_lines: list[str]) -> str:
        """Extract text from a node."""
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        
        if start_row == end_row:
            return source_lines[start_row][start_col:end_col]
        
        # Multi-line
        lines = [source_lines[start_row][start_col:]]
        for row in range(start_row + 1, end_row):
            lines.append(source_lines[row])
        lines.append(source_lines[end_row][:end_col])
        
        return '\n'.join(lines)
    
    def _make_relative_path(self, filepath: str) -> str:
        """
        Make path relative to current working directory.
        
        Args:
            filepath: Absolute file path
        
        Returns:
            Relative path
        """
        try:
            return str(Path(filepath).relative_to(Path.cwd()))
        except ValueError:
            return filepath