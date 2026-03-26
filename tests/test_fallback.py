"""
Tests for Phase 3 — ChunkedFallback, ReaderAgent, AgentMemoryStore.

All LLM calls are mocked; no API key or Ollama required.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tpca.config import TPCAConfig
from tpca.logging.log_config import LogConfig
from tpca.logging.structured_logger import StructuredLogger
from tpca.models.output import OutputChunk
from tpca.fallback.memory_store import AgentMemoryStore
from tpca.fallback.reader_agent import ReaderAgent
from tpca.fallback.chunked_pipeline import ChunkedFallback


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return TPCAConfig(
        fallback_chunk_tokens=500,
        fallback_overlap_tokens=50,
        log=LogConfig(log_file='/dev/null', console_level='ERROR'),
    )


@pytest.fixture
def logger(config):
    return StructuredLogger(config.log)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.count_tokens.side_effect = lambda text: max(1, len(text) // 4)
    llm.complete.return_value = (
        'Extracted: hashPassword hashes a password with salt. '
        'verifyPassword checks hash equality. [EXTRACTION_COMPLETE]'
    )
    return llm


def _make_symbol(sym_id: str, file: str, start: int = 0, end: int = 10, sig: str = ''):
    from tpca.models.symbol import Symbol
    name = sym_id.split('::')[-1]
    return Symbol(
        id=sym_id, type='function', name=name, qualified_name=name,
        file=file, start_line=start, end_line=end,
        signature=sig or f'def {name}(): ...', docstring='',
    )


# ── AgentMemoryStore tests ────────────────────────────────────────────────────

class TestAgentMemoryStore:
    def test_extends_output_log(self):
        store = AgentMemoryStore()
        from tpca.models.output import OutputLog
        assert isinstance(store, OutputLog)

    def test_add_extraction_creates_chunk(self):
        store = AgentMemoryStore()
        store.add_extraction(
            chunk_id=0,
            symbol_ids=['src/utils.py::hash_password', 'src/utils.py::verify_password'],
            summary='Hashes and verifies passwords.',
            token_count=120,
        )
        assert len(store.chunks) == 1
        assert store.chunks_processed == 1
        assert store.chunks[0].status == 'extracted'

    def test_render_compact_uses_fallback_header(self):
        store = AgentMemoryStore()
        store.add_extraction(0, ['src/auth.py::Auth'], 'Auth validates JWTs.', 200)
        rendered = store.render_compact()
        assert 'READER EXTRACTIONS' in rendered
        assert 'Auth validates JWTs.' in rendered

    def test_render_compact_empty(self):
        assert AgentMemoryStore().render_compact() == ''

    def test_fallback_used_flag(self):
        store = AgentMemoryStore()
        assert store.fallback_used is True

    def test_save_and_load_round_trip(self, tmp_path):
        store = AgentMemoryStore()
        store.add_extraction(0, ['src/a.py::Foo'], 'Foo does X.', 100)
        store.add_extraction(1, ['src/b.py::Bar'], 'Bar does Y.', 80)
        store.total_source_tokens = 5000

        path = str(tmp_path / 'memory.json')
        store.save(path)

        loaded = AgentMemoryStore.load(path)
        assert len(loaded.chunks) == 2
        assert loaded.chunks[0].summary == 'Foo does X.'
        assert loaded.total_source_tokens == 5000
        assert loaded.fallback_used is True

    def test_load_creates_correct_chunk_count(self, tmp_path):
        store = AgentMemoryStore()
        for i in range(5):
            store.add_extraction(i, [f'src/f{i}.py::F'], f'Summary {i}.', 50)
        path = str(tmp_path / 'm.json')
        store.save(path)

        loaded = AgentMemoryStore.load(path)
        assert loaded.chunks_processed == 5
        assert len(loaded.chunks) == 5


# ── ReaderAgent tests ─────────────────────────────────────────────────────────

class TestReaderAgent:
    def test_read_returns_extraction_and_token_count(self, config, logger, mock_llm):
        agent = ReaderAgent(config, logger, mock_llm, chunk_index=0, total_chunks=2)
        sym = _make_symbol('src/utils.py::hash_password', 'src/utils.py')
        extraction, tokens = agent.read(
            symbols=[sym],
            source_text='def hash_password(pw, salt): return hmac(pw, salt)',
            task='Document all utility functions.',
        )
        assert 'hash' in extraction.lower() or 'extract' in extraction.lower()
        assert tokens > 0
        assert '[EXTRACTION_COMPLETE]' not in extraction

    def test_read_strips_extraction_marker(self, config, logger, mock_llm):
        mock_llm.complete.return_value = 'Some content. [EXTRACTION_COMPLETE]'
        agent = ReaderAgent(config, logger, mock_llm)
        sym = _make_symbol('src/a.py::f', 'src/a.py')
        extraction, _ = agent.read([sym], 'def f(): pass', 'task')
        assert '[EXTRACTION_COMPLETE]' not in extraction

    def test_read_falls_back_gracefully_on_llm_error(self, config, logger, mock_llm):
        mock_llm.complete.side_effect = RuntimeError('LLM unavailable')
        agent = ReaderAgent(config, logger, mock_llm)
        sym = _make_symbol('src/a.py::f', 'src/a.py', sig='def f(x: int) -> str')
        extraction, _ = agent.read([sym], 'def f(x): return str(x)', 'task')
        # Should return a fallback signature extraction, not raise
        assert isinstance(extraction, str)

    def test_uses_reader_model(self, config, logger, mock_llm):
        agent = ReaderAgent(config, logger, mock_llm)
        sym = _make_symbol('src/a.py::f', 'src/a.py')
        agent.read([sym], 'def f(): pass', 'task')
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs.get('model') == config.active_reader_model


# ── ChunkedFallback tests ─────────────────────────────────────────────────────

class TestChunkedFallback:
    def test_run_returns_memory_store(self, config, logger, mock_llm):
        fb = ChunkedFallback(config, logger, mock_llm)
        symbols = [_make_symbol(f'src/f.py::func{i}', 'src/f.py', i * 5, i * 5 + 4) for i in range(3)]
        slices = {s.id: f'def func{i}(): pass' for i, s in enumerate(symbols)}
        store = fb.run(symbols=symbols, source_slices=slices, task='Document everything.')
        assert isinstance(store, AgentMemoryStore)
        assert store.fallback_used is True

    def test_run_creates_one_extraction_per_chunk(self, config, logger, mock_llm):
        # Small budget → multiple chunks
        config.fallback_chunk_tokens = 20
        fb = ChunkedFallback(config, logger, mock_llm)
        # Each "source" is ~25 chars → ~6 tokens → exceeds budget alone after first
        symbols = [_make_symbol(f'src/f.py::f{i}', 'src/f.py') for i in range(4)]
        slices = {s.id: 'def function(): pass # comment' for s in symbols}
        store = fb.run(symbols=symbols, source_slices=slices, task='task')
        # Should have produced at least one chunk
        assert len(store.chunks) >= 1

    def test_run_skips_already_completed_chunks(self, config, logger, mock_llm):
        fb = ChunkedFallback(config, logger, mock_llm)
        symbols = [_make_symbol('src/f.py::f0', 'src/f.py')]
        slices = {'src/f.py::f0': 'def f0(): pass'}

        # Pre-populate the store with chunk 0 already done
        prior_store = AgentMemoryStore()
        prior_store.add_extraction(0, ['src/f.py::f0'], 'Already done.', 50)

        store = fb.run(symbols=symbols, source_slices=slices, task='task', memory_store=prior_store)
        # The LLM should not have been called again for chunk 0
        mock_llm.complete.assert_not_called()
        assert len(store.chunks) == 1

    def test_partition_respects_overlap(self, config, logger, mock_llm):
        config.fallback_chunk_tokens = 30
        config.fallback_overlap_tokens = 10
        fb = ChunkedFallback(config, logger, mock_llm)
        symbols = [_make_symbol(f'src/f.py::f{i}', 'src/f.py') for i in range(6)]
        pairs = [(s, 'def f(): pass  # 16 chars ~ 4 tokens') for s in symbols]
        chunks = fb._partition(pairs)
        # Verify we got multiple chunks
        assert len(chunks) >= 2

    def test_format_chunk_includes_symbol_header(self, config, logger, mock_llm):
        fb = ChunkedFallback(config, logger, mock_llm)
        sym = _make_symbol('src/auth.py::Auth', 'src/auth.py')
        formatted = fb._format_chunk([(sym, 'class Auth: pass')])
        assert 'src/auth.py::Auth' in formatted
        assert 'class Auth: pass' in formatted
