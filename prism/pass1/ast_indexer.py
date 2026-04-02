"""
AST Indexer for extracting symbols from source code using Tree-sitter.
Supports multi-file input (single file, list of files, or directory).
"""
import os
from pathlib import Path
from typing import Union
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query, QueryCursor

from ..config import PRISMConfig
from ..logging import StructuredLogger
from ..models import Symbol

# ── Phase 3 — Optional JS/TS parsers ─────────────────────────────────────────
# These are graceful: Python-only installs work fine without them.
try:
    import tree_sitter_javascript as _tsjs
    _JS_LANGUAGE = Language(_tsjs.language())
    _JS_AVAILABLE = True
except Exception as e:
    _JS_LANGUAGE = None
    _JS_AVAILABLE = False

try:
    import tree_sitter_typescript as _tsts
    _TS_LANGUAGE  = Language(_tsts.language_typescript())
    _TSX_LANGUAGE = Language(_tsts.language_tsx())
    _TSTYPE_AVAILABLE = True
except Exception as e:
    _TS_LANGUAGE  = None
    _TSX_LANGUAGE = None
    _TSTYPE_AVAILABLE = False


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
    
    def __init__(self, config: PRISMConfig, logger: StructuredLogger, cache=None):
        self.config = config
        self.logger = logger
        self.cache = cache
        
        # Initialize Tree-sitter parser for Python
        PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(PY_LANGUAGE)   
        # Load query for Python
        query_file = Path(__file__).parent / 'queries' / 'python.scm'
        with open(query_file, 'r') as f:
            query_text = f.read()
        self.query = Query(PY_LANGUAGE, query_text)
    
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
    
    def _walk_directory(self, directory: str) -> list[str]:
        """Walk a directory and return list of file paths, respecting exclusions."""
        return self._resolve_paths(directory)

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
            
            self.logger.info('directory_walk', path=str(source), files_found=len(files))
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

        Delegates to config.detect_language() for Phase 3 extension mapping
        (which respects the configured languages list). Falls back to the
        class-level LANGUAGE_EXTENSIONS dict for backward compatibility.

        Returns language name (e.g. 'python') or 'unknown'.
        """
        # Use config's detect_language if it supports the new Phase 3 API
        if hasattr(self.config, "detect_language"):
            result = self.config.detect_language(path)
            if result is not None:
                return result
            # detect_language returns None when lang not in config.languages —
            # return 'unknown' so the caller skips this file.
            ext = Path(path).suffix
            for lang, exts in self.LANGUAGE_EXTENSIONS.items():
                if ext in exts:
                    return lang  # recognised extension but language not enabled
            return "unknown"

        # Legacy fallback (config without detect_language)
        ext = Path(path).suffix
        for lang, exts in self.LANGUAGE_EXTENSIONS.items():
            if ext in exts:
                return lang
        return "unknown"
    
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
            elif language == 'javascript':                  # Phase 3
                return self._parse_javascript(path, source_code)
            elif language in ('typescript', 'tsx'):         # Phase 3
                return self._parse_typescript(path, source_code, tsx=(language == 'tsx'))
            else:
                self.logger.warn('unsupported_language', file=path, language=language)
                return []
        
        except Exception as e:
            self.logger.error('parse_error', file=path, error=str(e))
            return []

    # ── Phase 3 — JavaScript parsing ─────────────────────────────────────────

    def _parse_javascript(self, path: str, source: bytes) -> list[Symbol]:
        if not _JS_AVAILABLE or _JS_LANGUAGE is None:
            print(_JS_LANGUAGE)
            print(_JS_AVAILABLE)
            self.logger.warn(
                'js_parser_unavailable', file=path,
                hint='pip install tree-sitter-javascript',
            )
            return []
        parser = Parser(_JS_LANGUAGE)
        tree = parser.parse(source)
        query = self._get_js_query('javascript', _JS_LANGUAGE)
        return self._extract_js_symbols(path, tree, query)

    def _parse_typescript(self, path: str, source: bytes, tsx: bool = False) -> list[Symbol]:
        lang_obj = _TSX_LANGUAGE if tsx else _TS_LANGUAGE
        lang_key = 'tsx' if tsx else 'typescript'
        if not _TSTYPE_AVAILABLE or lang_obj is None:
            self.logger.warn(
                'ts_parser_unavailable', file=path,
                hint='pip install tree-sitter-typescript',
            )
            return []
        parser = Parser(lang_obj)
        tree = parser.parse(source)
        query = self._get_js_query(lang_key, lang_obj)
        return self._extract_js_symbols(path, tree, query)

    def _extract_js_symbols(self, path: str, tree, query) -> list[Symbol]:
        """Extract classes, methods, functions (and TS interfaces/enums) from a JS/TS tree."""
        symbols: list[Symbol] = []
        rel_path = self._make_relative_path(path)
        # captures = self.query.captures(tree.root_node)

        by_name: dict[str, list] = {}
        for node, cap_name in self._run_query(query, tree.root_node):
            by_name.setdefault(cap_name, []).append(node)

        jsdoc_map = self._build_jsdoc_map(by_name.get('docstring.candidate', []))
        seen_ids: set[str] = set()

        def add(sym):
            if sym and sym.id not in seen_ids:
                seen_ids.add(sym.id)
                symbols.append(sym)

        for n in by_name.get('class.name', []):
            add(self._build_js_class(rel_path, n, jsdoc_map))
        for n in by_name.get('interface.name', []):
            add(self._build_js_interface(rel_path, n))
        for n in by_name.get('type.name', []):
            add(self._build_js_type_alias(rel_path, n))
        for n in by_name.get('enum.name', []):
            add(self._build_js_enum(rel_path, n))
        for n in by_name.get('function.name', []):
            add(self._build_js_function(rel_path, n, jsdoc_map))
        for n in by_name.get('method.name', []):
            add(self._build_js_method(rel_path, n, jsdoc_map))

        return symbols

    def _build_js_class(self, path, name_node, jsdoc_map):
        name = name_node.text.decode('utf-8')
        cls_node = self._find_ancestor(name_node, {'class_declaration', 'class'})
        if not cls_node:
            return None
        superclass = self._extract_js_superclass(cls_node)
        sig = f'class {name}' + (f' extends {superclass}' if superclass else '')
        sym_id = f'{path}::{name}'
        return Symbol(
            id=sym_id, type='class', name=name, qualified_name=name, file=path,
            start_line=cls_node.start_point[0], end_line=cls_node.end_point[0],
            signature=sig, docstring=jsdoc_map.get(cls_node.id, '')[:120],
            bases=[superclass] if superclass else [],
        )

    def _build_js_interface(self, path, name_node):
        name = name_node.text.decode('utf-8')
        node = self._find_ancestor(name_node, {'interface_declaration'})
        if not node:
            return None
        sym_id = f'{path}::{name}'
        return Symbol(
            id=sym_id, type='class', name=name, qualified_name=name, file=path,
            start_line=node.start_point[0], end_line=node.end_point[0],
            signature=f'interface {name}', docstring='',
        )

    def _build_js_type_alias(self, path, name_node):
        name = name_node.text.decode('utf-8')
        node = self._find_ancestor(name_node, {'type_alias_declaration'})
        if not node:
            return None
        sym_id = f'{path}::{name}'
        return Symbol(
            id=sym_id, type='constant', name=name, qualified_name=name, file=path,
            start_line=node.start_point[0], end_line=node.end_point[0],
            signature=f'type {name}', docstring='',
        )

    def _build_js_enum(self, path, name_node):
        name = name_node.text.decode('utf-8')
        node = self._find_ancestor(name_node, {'enum_declaration'})
        if not node:
            return None
        sym_id = f'{path}::{name}'
        return Symbol(
            id=sym_id, type='class', name=name, qualified_name=name, file=path,
            start_line=node.start_point[0], end_line=node.end_point[0],
            signature=f'enum {name}', docstring='',
        )

    def _build_js_function(self, path, name_node, jsdoc_map):
        name = name_node.text.decode('utf-8')
        ancestor_types = {
            'function_declaration', 'lexical_declaration',
            'variable_declarator', 'export_statement',
        }
        func_node = self._find_ancestor(name_node, ancestor_types)
        if not func_node:
            return None
        actual = self._find_descendant(func_node, {'function_declaration', 'arrow_function', 'function_expression'})
        target = actual or func_node
        params = self._extract_js_params(target)
        ret = self._extract_js_return_type(target)
        sig = f'function {name}({params})' + (f': {ret}' if ret else '')
        sym_id = f'{path}::{name}'
        return Symbol(
            id=sym_id, type='function', name=name, qualified_name=name, file=path,
            start_line=target.start_point[0], end_line=target.end_point[0],
            signature=sig, docstring=jsdoc_map.get(func_node.id, '')[:120],
        )

    def _build_js_method(self, path, name_node, jsdoc_map):
        name = name_node.text.decode('utf-8')
        method_node = self._find_ancestor(name_node, {'method_definition', 'abstract_method_signature'})
        if not method_node:
            return None
        class_node = self._find_ancestor(method_node, {'class_declaration', 'class', 'interface_declaration'})
        parent_name = None
        if class_node:
            for child in class_node.children:
                if child.type in ('identifier', 'type_identifier'):
                    parent_name = child.text.decode('utf-8')
                    break
        params = self._extract_js_params(method_node)
        ret = self._extract_js_return_type(method_node)
        qualified = f'{parent_name}.{name}' if parent_name else name
        sig = f'{qualified}({params})' + (f': {ret}' if ret else '')
        sym_id = f'{path}::{qualified}'
        return Symbol(
            id=sym_id, type='method', name=name, qualified_name=qualified, file=path,
            start_line=method_node.start_point[0], end_line=method_node.end_point[0],
            signature=sig, docstring=jsdoc_map.get(method_node.id, '')[:120],
            parent_class=parent_name,
        )

    # ── JS/TS AST helpers ─────────────────────────────────────────────────────

    def _extract_js_superclass(self, class_node):
        for child in class_node.children:
            if child.type == 'class_heritage':
                for sub in child.children:
                    if sub.type == 'extends_clause':
                        for s in sub.children:
                            if s.type in ('identifier', 'member_expression'):
                                return s.text.decode('utf-8')
                    if sub.type == 'identifier':
                        return sub.text.decode('utf-8')
        return None

    def _extract_js_params(self, node) -> str:
        for child in node.children:
            if child.type == 'formal_parameters':
                inner = child.text.decode('utf-8').strip('()')
                return inner[:120]
        return ''

    def _extract_js_return_type(self, node) -> str:
        for child in node.children:
            if child.type == 'type_annotation':
                return child.text.decode('utf-8').lstrip(': ').strip()[:60]
        return ''

    def _build_jsdoc_map(self, comment_nodes: list) -> dict:
        import re
        result = {}
        for node in comment_nodes:
            raw = node.text.decode('utf-8').strip()
            if raw.startswith('/**'):
                inner = raw[3:-2] if raw.endswith('*/') else raw[3:]
                cleaned = re.sub(r'^\s*\*\s?', '', inner, flags=re.MULTILINE).strip()
                result[node.id] = cleaned[:120]
        return result

    @staticmethod
    def _find_ancestor(node, types: set):
        current = node.parent
        while current:
            if current.type in types:
                return current
            current = current.parent
        return None

    @staticmethod
    def _find_descendant(node, types: set):
        queue = list(node.children)
        while queue:
            child = queue.pop(0)
            if child.type in types:
                return child
            queue.extend(child.children)
        return None

    @staticmethod
    def _run_query(query, node) -> list[tuple]:
        """Execute a Query against a node using QueryCursor (tree-sitter 0.22+).
        Returns a list of (node, capture_name) tuples."""
        cursor = QueryCursor(query)
        result = []
        for _pattern_idx, match in cursor.matches(node):
            for cap_name, cap_nodes in match.items():
                nodes = cap_nodes if isinstance(cap_nodes, list) else [cap_nodes]
                for n in nodes:
                    result.append((n, cap_name))
        return result

    def _get_js_query(self, lang: str, language_obj):
        """Load and cache the Tree-sitter query for a JS/TS language."""
        cache_key = f'_js_query_{lang}'
        if not hasattr(self, cache_key):
            query_file = Path(__file__).parent / 'queries' / f'{lang}.scm'
            if not query_file.exists() and lang == 'tsx':
                query_file = Path(__file__).parent / 'queries' / 'typescript.scm'
            if query_file.exists():
                q = Query(language_obj, query_file.read_text())
            else:
                self.logger.warn('js_query_file_missing', language=lang, path=str(query_file))
                q = Query(language_obj, '(identifier) @noop')
            setattr(self, cache_key, q)
        return getattr(self, cache_key)
    
    def _extract_calls_from_body(self, body_node) -> list[str]:
        """
        Recursively walk a function/method body node and return all called names.

        Simple calls (foo()) yield the function name.
        Attribute calls (self.bar(), obj.method()) yield the method name.
        Built-in names are excluded — they will never resolve to project symbols.
        """
        _BUILTINS = frozenset({
            'print', 'len', 'range', 'isinstance', 'type', 'str', 'int',
            'float', 'list', 'dict', 'set', 'tuple', 'bool', 'super',
            'hasattr', 'getattr', 'setattr', 'vars', 'enumerate', 'zip',
            'map', 'filter', 'sorted', 'reversed', 'sum', 'min', 'max',
            'open', 'iter', 'next', 'repr', 'format', 'any', 'all',
        })

        calls: list[str] = []

        def walk(node):
            if node.type == 'call':
                func = node.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        name = func.text.decode('utf-8')
                        if name not in _BUILTINS:
                            calls.append(name)
                    elif func.type == 'attribute':
                        attr = func.child_by_field_name('attribute')
                        if attr:
                            calls.append(attr.text.decode('utf-8'))
            for child in node.children:
                walk(child)

        if body_node:
            walk(body_node)
        return calls

    def _extract_python_symbols(self, filepath: str, source_code: bytes,
                                tree) -> list[Symbol]:
        """
        Extract symbols from Python AST.

        Extracts: classes, functions/methods (with call lists),
        module-level import statements, and module-level UPPER_CASE constants.
        """
        symbols = []
        rel_path = self._make_relative_path(filepath)
        source_lines = source_code.decode('utf-8').split('\n')

        class_stack: list[str] = []
        processed_import_nodes: set[int] = set()
        processed_constant_nodes: set[int] = set()

        all_captures = sorted(
            self._run_query(self.query, tree.root_node),
            key=lambda x: x[0].start_point,
        )

        for node, capture_name in all_captures:
            try:
                if capture_name == 'class.def':
                    sym = self._extract_class(node, rel_path, source_lines)
                    if sym:
                        symbols.append(sym)
                        class_stack.append(sym.name)

                elif capture_name == 'function.def':
                    parent_class = class_stack[-1] if class_stack else None
                    sym = self._extract_function(node, rel_path, source_lines, parent_class)
                    if sym:
                        symbols.append(sym)

                elif capture_name == 'import.module':
                    # import X  or  import X as Y
                    parent = node.parent  # import_statement node
                    if parent and parent.id not in processed_import_nodes:
                        processed_import_nodes.add(parent.id)
                        module_name = node.text.decode('utf-8')
                        sym_id = f"{rel_path}::import.{module_name}"
                        symbols.append(Symbol(
                            id=sym_id,
                            type='import',
                            name=module_name,
                            qualified_name=module_name,
                            file=rel_path,
                            start_line=parent.start_point[0] + 1,
                            end_line=parent.end_point[0] + 1,
                            signature=f"import {module_name}",
                            docstring='',
                        ))

                elif capture_name == 'import.from':
                    # from X import Y  (one Symbol per from-import statement)
                    parent = node.parent  # import_from_statement node
                    if parent and parent.id not in processed_import_nodes:
                        processed_import_nodes.add(parent.id)
                        from_module = node.text.decode('utf-8')
                        # Use the raw source line(s) for a readable signature
                        stmt_lines = source_lines[
                            parent.start_point[0]:parent.end_point[0] + 1
                        ]
                        signature = ' '.join(l.strip() for l in stmt_lines)[:80]
                        sym_id = f"{rel_path}::import.{from_module}"
                        symbols.append(Symbol(
                            id=sym_id,
                            type='import',
                            name=from_module,
                            qualified_name=from_module,
                            file=rel_path,
                            start_line=parent.start_point[0] + 1,
                            end_line=parent.end_point[0] + 1,
                            signature=signature,
                            docstring='',
                        ))

                elif capture_name == 'constant.name':
                    # Module-level assignment — only keep UPPER_CASE names
                    name = node.text.decode('utf-8')
                    if not (name.isupper() or (name[0].isupper() and '_' in name and name == name.upper())):
                        continue
                    # Locate the assignment node to get the value preview
                    assign_node = node.parent
                    while assign_node and assign_node.type not in ('assignment', 'augmented_assignment'):
                        assign_node = assign_node.parent
                    if not assign_node or assign_node.id in processed_constant_nodes:
                        continue
                    processed_constant_nodes.add(assign_node.id)
                    line_text = source_lines[assign_node.start_point[0]] if assign_node.start_point[0] < len(source_lines) else ''
                    value_preview = line_text.split('=', 1)[-1].strip()[:60] if '=' in line_text else ''
                    sym_id = f"{rel_path}::constant.{name}"
                    symbols.append(Symbol(
                        id=sym_id,
                        type='constant',
                        name=name,
                        qualified_name=name,
                        file=rel_path,
                        start_line=assign_node.start_point[0] + 1,
                        end_line=assign_node.end_point[0] + 1,
                        signature=f"{name} = {value_preview}",
                        docstring='',
                    ))

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
        
        # Extract all call names from the function body for the call graph
        body_node = node.child_by_field_name('body')
        call_names = self._extract_calls_from_body(body_node)

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
            parent_class=parent_class,
            calls=call_names,
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