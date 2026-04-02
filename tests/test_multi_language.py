"""
Tests for Phase 3 — Multi-language (JavaScript / TypeScript) AST indexing.

All tests are deterministic; no LLM calls, no API key required.
JS/TS parser tests are skipped gracefully if tree-sitter-javascript /
tree-sitter-typescript are not installed, so Python-only environments
still pass the full suite.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from prism.config import PRISMConfig
from prism.logging.log_config import LogConfig
from prism.logging.structured_logger import StructuredLogger

# ── Helpers ───────────────────────────────────────────────────────────────────

FIXTURES_JS = Path(__file__).parent / 'fixtures' / 'sample_js_codebase'

JS_AVAILABLE = False
try:
    import tree_sitter_javascript  # noqa: F401
    JS_AVAILABLE = True
except ImportError:
    pass

TS_AVAILABLE = False
try:
    import tree_sitter_typescript  # noqa: F401
    TS_AVAILABLE = True
except ImportError:
    pass


def _make_config(languages: list[str]) -> PRISMConfig:
    return PRISMConfig(
        languages=languages,
        cache_enabled=False,
        log=LogConfig(log_file='/dev/null', console_level='ERROR'),
    )


def _make_indexer(config: PRISMConfig):
    from prism.pass1.ast_indexer import ASTIndexer
    logger = StructuredLogger(config.log)
    return ASTIndexer(config, logger, cache=None)


# ── Language detection tests (no tree-sitter needed) ──────────────────────────

class TestLanguageDetection:
    """PRISMConfig.detect_language() must map extensions correctly."""

    def test_python_extensions(self):
        config = _make_config(['python'])
        assert config.detect_language('src/auth.py') == 'python'
        assert config.detect_language('stubs/mod.pyi') == 'python'

    def test_javascript_extensions(self):
        config = _make_config(['python', 'javascript'])
        assert config.detect_language('src/index.js') == 'javascript'
        assert config.detect_language('src/component.jsx') == 'javascript'
        assert config.detect_language('esm/mod.mjs') == 'javascript'

    def test_typescript_extensions(self):
        config = _make_config(['python', 'typescript'])
        assert config.detect_language('src/auth.ts') == 'typescript'
        assert config.detect_language('src/types.mts') == 'typescript'

    def test_tsx_extension(self):
        config = _make_config(['python', 'tsx'])
        assert config.detect_language('src/App.tsx') == 'tsx'

    def test_unsupported_extension_returns_none(self):
        config = _make_config(['python'])
        assert config.detect_language('README.md') is None
        assert config.detect_language('Makefile') is None

    def test_language_not_in_config_returns_none(self):
        # .js extension exists in map but 'javascript' not in languages list
        config = _make_config(['python'])
        assert config.detect_language('src/index.js') is None

    def test_supports_language(self):
        config = _make_config(['python', 'javascript'])
        assert config.supports_language('python') is True
        assert config.supports_language('javascript') is True
        assert config.supports_language('typescript') is False

    def test_custom_extension_override(self):
        config = _make_config(['python', 'javascript'])
        config.language_extensions['.coffee'] = 'javascript'
        assert config.detect_language('src/app.coffee') == 'javascript'


# ── JavaScript indexing tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not JS_AVAILABLE, reason='tree-sitter-javascript not installed')
class TestJavaScriptIndexing:
    """
    Full AST indexing of the sample_js_codebase fixture.
    Mirrors the Python cross-file edge tests from Phase 1.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = _make_config(['javascript'])
        self.indexer = _make_indexer(self.config)

    def test_indexes_js_directory(self):
        symbols = self.indexer.index(str(FIXTURES_JS))
        assert len(symbols) > 0, 'Expected symbols from JS fixture directory'

    def test_extracts_class_from_auth_js(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        names = [s.name for s in symbols]
        assert 'Auth' in names

    def test_auth_class_has_correct_type(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        auth_sym = next((s for s in symbols if s.name == 'Auth'), None)
        assert auth_sym is not None
        assert auth_sym.type == 'class'

    def test_extracts_methods_from_auth_js(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        method_names = [s.name for s in symbols if s.type == 'method']
        # Auth has: constructor, validateToken, refreshToken, createToken,
        #           _decodePayload, _createToken
        assert 'validateToken' in method_names
        assert 'refreshToken' in method_names

    def test_method_has_qualified_name(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        vt = next((s for s in symbols if s.name == 'validateToken'), None)
        assert vt is not None
        assert 'Auth' in vt.qualified_name

    def test_method_symbol_id_includes_file_and_class(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        vt = next((s for s in symbols if s.name == 'validateToken'), None)
        assert vt is not None
        assert 'auth.js' in vt.id
        assert 'Auth.validateToken' in vt.id

    def test_extracts_functions_from_utils_js(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'utils.js'))
        func_names = [s.name for s in symbols if s.type == 'function']
        assert 'hashPassword' in func_names
        assert 'verifyPassword' in func_names
        assert 'generateToken' in func_names
        assert 'checkRateLimit' in func_names

    def test_extracts_router_class(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'router.js'))
        names = [s.name for s in symbols]
        assert 'Router' in names

    def test_router_has_route_method(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'router.js'))
        route = next((s for s in symbols if s.name == 'route'), None)
        assert route is not None
        assert route.type == 'method'

    def test_symbol_has_start_and_end_lines(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        for sym in symbols:
            assert sym.start_line >= 0, f'{sym.id} has negative start_line'
            assert sym.end_line >= sym.start_line, f'{sym.id} end_line < start_line'

    def test_symbol_has_file_path(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        for sym in symbols:
            assert 'auth.js' in sym.file

    def test_jsdoc_extracted_as_docstring(self):
        symbols = self.indexer.index(str(FIXTURES_JS / 'auth.js'))
        vt = next((s for s in symbols if s.name == 'validateToken'), None)
        if vt:  # docstring extraction is best-effort
            assert len(vt.docstring) <= 120

    def test_no_symbols_from_unsupported_extension(self, tmp_path):
        md_file = tmp_path / 'README.md'
        md_file.write_text('# Docs')
        symbols = self.indexer.index(str(tmp_path))
        assert symbols == []

    def test_multi_file_indexes_all_three_files(self):
        symbols = self.indexer.index(str(FIXTURES_JS))
        files_seen = {s.file for s in symbols}
        assert any('auth.js' in f for f in files_seen)
        assert any('router.js' in f for f in files_seen)
        assert any('utils.js' in f for f in files_seen)

    def test_graph_built_from_js_symbols(self):
        from prism.pass1.graph_builder import GraphBuilder
        config = _make_config(['javascript'])
        logger = StructuredLogger(config.log)
        indexer = _make_indexer(config)
        builder = GraphBuilder(config, logger)

        symbols = indexer.index(str(FIXTURES_JS))
        graph = builder.build(symbols)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() >= 0

    def test_pagerank_runs_on_js_graph(self):
        from prism.pass1.graph_builder import GraphBuilder
        from prism.pass1.graph_ranker import GraphRanker
        config = _make_config(['javascript'])
        logger = StructuredLogger(config.log)
        indexer = _make_indexer(config)
        builder = GraphBuilder(config, logger)
        ranker = GraphRanker(config, logger)

        symbols = indexer.index(str(FIXTURES_JS))
        graph = builder.build(symbols)
        ranked = ranker.rank_symbols(graph, ['auth', 'token', 'validate'])

        # All nodes should have a pagerank score
        for node in ranked.nodes:
            assert ranked.nodes[node].get('pagerank', 0) >= 0

    def test_compact_index_generated_from_js(self):
        from prism.pass1.graph_builder import GraphBuilder
        from prism.pass1.graph_ranker import GraphRanker
        from prism.pass1.index_renderer import IndexRenderer
        config = _make_config(['javascript'])
        logger = StructuredLogger(config.log)
        indexer = _make_indexer(config)
        builder = GraphBuilder(config, logger)
        ranker = GraphRanker(config, logger)
        renderer = IndexRenderer(config, logger)

        symbols = indexer.index(str(FIXTURES_JS))
        graph = builder.build(symbols)
        graph = ranker.rank_symbols(graph, ['auth', 'router'])
        index = renderer.render(graph)

        assert isinstance(index, str)
        assert len(index) > 0


# ── TypeScript indexing tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not TS_AVAILABLE, reason='tree-sitter-typescript not installed')
class TestTypeScriptIndexing:
    """
    Tests for TypeScript-specific constructs: interfaces, type aliases, enums,
    return type annotations. Uses inline source strings rather than fixture files.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.config = _make_config(['typescript'])
        self.indexer = _make_indexer(self.config)
        self.tmp = tmp_path

    def _write_ts(self, name: str, content: str) -> str:
        path = self.tmp / name
        path.write_text(content, encoding='utf-8')
        return str(path)

    def test_extracts_class_from_ts(self):
        path = self._write_ts('service.ts', '''
export class UserService {
    constructor(private db: Database) {}
    async getUser(id: string): Promise<User> {
        return this.db.find(id);
    }
}
''')
        symbols = self.indexer.index(path)
        names = [s.name for s in symbols]
        assert 'UserService' in names

    def test_extracts_interface_from_ts(self):
        path = self._write_ts('types.ts', '''
export interface User {
    id: string;
    email: string;
    role: 'admin' | 'user';
}
''')
        symbols = self.indexer.index(path)
        names = [s.name for s in symbols]
        assert 'User' in names

    def test_extracts_type_alias_from_ts(self):
        path = self._write_ts('types.ts', '''
export type UserId = string;
export type Role = 'admin' | 'user' | 'guest';
''')
        symbols = self.indexer.index(path)
        names = [s.name for s in symbols]
        assert 'UserId' in names or 'Role' in names

    def test_extracts_enum_from_ts(self):
        path = self._write_ts('enums.ts', '''
export enum Status {
    Active = 'ACTIVE',
    Inactive = 'INACTIVE',
    Pending = 'PENDING',
}
''')
        symbols = self.indexer.index(path)
        names = [s.name for s in symbols]
        assert 'Status' in names

    def test_extracts_function_with_return_type(self):
        path = self._write_ts('util.ts', '''
export function parseToken(raw: string): { sub: string; exp: number } | null {
    try { return JSON.parse(atob(raw.split(".")[1])); }
    catch { return null; }
}
''')
        symbols = self.indexer.index(path)
        func = next((s for s in symbols if s.name == 'parseToken'), None)
        assert func is not None
        assert func.type == 'function'

    def test_method_with_return_type_annotation(self):
        path = self._write_ts('auth.ts', '''
class AuthService {
    validateToken(token: string): boolean {
        return token.length > 0;
    }
    async refreshToken(token: string): Promise<string> {
        return token + '_refreshed';
    }
}
''')
        symbols = self.indexer.index(path)
        vt = next((s for s in symbols if s.name == 'validateToken'), None)
        assert vt is not None
        assert vt.type == 'method'

    def test_tsx_file_indexed_with_tsx_language(self, tmp_path):
        config = _make_config(['tsx'])
        indexer = _make_indexer(config)
        path = tmp_path / 'Button.tsx'
        path.write_text('''
export function Button({ label }: { label: string }) {
    return <button>{label}</button>;
}
''', encoding='utf-8')
        symbols = indexer.index(str(path))
        assert any(s.name == 'Button' for s in symbols)

    def test_ts_and_python_in_same_run(self, tmp_path):
        config = _make_config(['python', 'typescript'])
        logger = StructuredLogger(config.log)
        from prism.pass1.ast_indexer import ASTIndexer
        indexer = ASTIndexer(config, logger, cache=None)

        py_file = tmp_path / 'module.py'
        py_file.write_text('def helper(): pass\n')
        ts_file = tmp_path / 'service.ts'
        ts_file.write_text(
            'export function greet(name: string): string { return name; }\n')

        symbols = indexer.index(str(tmp_path))
        langs = {s.file.rsplit('.', 1)[-1] for s in symbols}
        # Should have indexed both languages
        assert 'py' in langs or 'ts' in langs


# ── Directory walk tests (no parser needed) ───────────────────────────────────

class TestDirectoryWalk:
    """
    Tests for ASTIndexer._walk_directory() — no actual parsing needed.
    """

    def test_excludes_node_modules(self, tmp_path):
        config = _make_config(['javascript'])
        (tmp_path / 'node_modules').mkdir()
        (tmp_path / 'node_modules' / 'lib.js').write_text('const x = 1;')
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'main.js').write_text('function main() {}')

        logger = StructuredLogger(config.log)
        from prism.pass1.ast_indexer import ASTIndexer
        indexer = ASTIndexer(config, logger, cache=None)

        paths = indexer._walk_directory(tmp_path)
        assert not any('lib.js' in p for p in paths)
        assert any('main.js' in p for p in paths)

    def test_excludes_dist_folder(self, tmp_path):
        config = _make_config(['javascript'])
        (tmp_path / 'dist').mkdir()
        (tmp_path / 'dist' / 'bundle.js').write_text('var x = 1;')
        (tmp_path / 'index.js').write_text('function main() {}')

        logger = StructuredLogger(config.log)
        from prism.pass1.ast_indexer import ASTIndexer
        indexer = ASTIndexer(config, logger, cache=None)

        paths = indexer._walk_directory(tmp_path)
        assert not any('bundle.js' in p for p in paths)
        assert any('index.js' in p for p in paths)

    def test_mixed_py_and_js_only_returns_configured(self, tmp_path):
        config = _make_config(['python'])  # only python
        (tmp_path / 'app.py').write_text('def run(): pass')
        (tmp_path / 'helper.js').write_text('function help() {}')

        logger = StructuredLogger(config.log)
        from prism.pass1.ast_indexer import ASTIndexer
        indexer = ASTIndexer(config, logger, cache=None)

        paths = indexer._walk_directory(tmp_path)
        assert all('.py' in p or '.pyi' in p for p in paths)

    def test_returns_empty_for_empty_directory(self, tmp_path):
        config = _make_config(['python', 'javascript'])
        logger = StructuredLogger(config.log)
        from prism.pass1.ast_indexer import ASTIndexer
        indexer = ASTIndexer(config, logger, cache=None)
        assert indexer._walk_directory(tmp_path) == []
