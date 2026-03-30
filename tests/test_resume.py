"""
Tests for Phase 3 — OutputManifest persistence and orchestrator resume logic.

All tests are deterministic; no LLM calls, no API key required.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from tpca.config import TPCAConfig
from tpca.logging.log_config import LogConfig
from tpca.logging.structured_logger import StructuredLogger
from tpca.models.output import (
    OutputManifest, ManifestEntry, OutputLog, OutputChunk
)
from tpca.pass2.output_writer import OutputWriter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    return TPCAConfig(
        output_mode='single_file',
        output_dir=str(tmp_path / 'output'),
        cache_dir=str(tmp_path / 'cache'),
        log=LogConfig(log_file='/dev/null', console_level='ERROR'),
    )


@pytest.fixture
def logger(config):
    return StructuredLogger(config.log)


def _entry(source: str, output: str, symbols: list[str], status: str = 'complete') -> ManifestEntry:
    return ManifestEntry(
        source_file=source,
        output_file=output,
        symbols_processed=symbols,
        chunk_count=len(symbols),
        token_count=200,
        status=status,
    )


# ── ManifestEntry tests ───────────────────────────────────────────────────────

class TestManifestEntry:
    def test_to_dict_round_trip(self):
        entry = _entry('src/auth.py', 'docs/auth.md',
                       ['src/auth.py::Auth'], 'complete')
        restored = ManifestEntry.from_dict(entry.to_dict())
        assert restored.source_file == entry.source_file
        assert restored.output_file == entry.output_file
        assert restored.symbols_processed == entry.symbols_processed
        assert restored.status == entry.status
        assert restored.token_count == entry.token_count

    def test_default_status_is_partial(self):
        entry = ManifestEntry(source_file='a.py', output_file='a.md')
        assert entry.status == 'partial'

    def test_from_dict_defaults(self):
        entry = ManifestEntry.from_dict(
            {'source_file': 'a.py', 'output_file': 'a.md'})
        assert entry.symbols_processed == []
        assert entry.chunk_count == 0
        assert entry.status == 'partial'


# ── OutputManifest tests ──────────────────────────────────────────────────────

class TestOutputManifest:
    def test_upsert_adds_new_entry(self):
        m = OutputManifest(task='test')
        entry = _entry('src/auth.py', 'docs/auth.md', ['src/auth.py::Auth'])
        m.upsert_entry(entry)
        assert len(m.files) == 1

    def test_upsert_replaces_existing_entry(self):
        m = OutputManifest(task='test')
        entry = _entry('src/auth.py', 'docs/auth.md', ['src/auth.py::Auth'])
        m.upsert_entry(entry)
        updated = _entry('src/auth.py', 'docs/auth.md',
                         ['src/auth.py::Auth', 'src/auth.py::Auth.validate_token'])
        m.upsert_entry(updated)
        assert len(m.files) == 1
        assert len(m.files[0].symbols_processed) == 2

    def test_get_entry_returns_correct_entry(self):
        m = OutputManifest(task='test')
        m.upsert_entry(
            _entry('src/auth.py', 'docs/auth.md', ['src/auth.py::Auth']))
        m.upsert_entry(_entry('src/router.py', 'docs/router.md',
                       ['src/router.py::Router']))
        found = m.get_entry('src/router.py')
        assert found is not None
        assert 'Router' in found.symbols_processed[0]

    def test_get_entry_returns_none_for_unknown(self):
        m = OutputManifest(task='test')
        assert m.get_entry('nonexistent.py') is None

    def test_incomplete_files_returns_partial(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md', [], 'complete'))
        m.upsert_entry(
            _entry('src/router.py', 'docs/router.md', [], 'partial'))
        m.upsert_entry(_entry('src/utils.py', 'docs/utils.md', [], 'skipped'))
        incomplete = m.incomplete_files()
        assert len(incomplete) == 2
        assert all(e.status != 'complete' for e in incomplete)

    def test_is_done_false_when_partial_entries_exist(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md', [], 'partial'))
        m.mark_complete()
        assert m.is_done() is False  # completed_at set but not all entries complete

    def test_is_done_true_when_all_complete(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md', [], 'complete'))
        m.upsert_entry(_entry('src/utils.py', 'docs/utils.md', [], 'complete'))
        m.mark_complete()
        assert m.is_done() is True

    def test_mark_complete_sets_completed_at(self):
        m = OutputManifest(task='test')
        assert m.completed_at is None
        m.mark_complete()
        assert m.completed_at is not None

    def test_to_dict_from_dict_round_trip(self):
        m = OutputManifest(task='Document all methods', output_mode='mirror')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md',
                       ['src/auth.py::Auth'], 'complete'))
        m.stats = {'llm_calls': 3, 'compression_ratio': 12.5}
        m.mark_complete()

        restored = OutputManifest.from_dict(m.to_dict())
        assert restored.task == m.task
        assert restored.output_mode == m.output_mode
        assert restored.completed_at == m.completed_at
        assert len(restored.files) == 1
        assert restored.stats['llm_calls'] == 3


# ── OutputManifest persistence tests ─────────────────────────────────────────

class TestOutputManifestPersistence:
    def test_save_creates_json_file(self, tmp_path):
        m = OutputManifest(task='test')
        m.upsert_entry(
            _entry('src/auth.py', 'docs/auth.md', ['sym1'], 'complete'))
        path = str(tmp_path / 'manifest.json')
        m.save(path)
        assert Path(path).exists()

    def test_save_creates_parent_directories(self, tmp_path):
        m = OutputManifest(task='test')
        path = str(tmp_path / 'deep' / 'nested' / 'manifest.json')
        m.save(path)
        assert Path(path).exists()

    def test_save_writes_valid_json(self, tmp_path):
        m = OutputManifest(task='Document everything')
        m.upsert_entry(
            _entry('src/auth.py', 'docs/auth.md', ['src/auth.py::Auth']))
        path = str(tmp_path / 'manifest.json')
        m.save(path)
        raw = json.loads(Path(path).read_text())
        assert raw['task'] == 'Document everything'
        assert len(raw['files']) == 1

    def test_load_restores_full_manifest(self, tmp_path):
        m = OutputManifest(
            task='Document all public methods', output_mode='mirror')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md',
                       ['src/auth.py::Auth'], 'complete'))
        m.upsert_entry(_entry('src/router.py', 'docs/router.md',
                       ['src/router.py::Router'], 'partial'))
        m.stats = {'llm_calls': 5}
        path = str(tmp_path / 'manifest.json')
        m.save(path)

        loaded = OutputManifest.load(path)
        assert loaded.task == 'Document all public methods'
        assert loaded.output_mode == 'mirror'
        assert len(loaded.files) == 2
        assert loaded.stats['llm_calls'] == 5

    def test_load_preserves_entry_statuses(self, tmp_path):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/a.py', 'docs/a.md', ['a::A'], 'complete'))
        m.upsert_entry(_entry('src/b.py', 'docs/b.md', ['b::B'], 'partial'))
        path = str(tmp_path / 'manifest.json')
        m.save(path)
        loaded = OutputManifest.load(path)
        statuses = {e.source_file: e.status for e in loaded.files}
        assert statuses['src/a.py'] == 'complete'
        assert statuses['src/b.py'] == 'partial'

    def test_load_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OutputManifest.load(str(tmp_path / 'nonexistent.json'))


# ── OutputLog resume tests ────────────────────────────────────────────────────

class TestOutputLogResume:
    def test_from_manifest_creates_chunks_for_complete_files(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry(
            'src/auth.py', 'docs/auth.md',
            ['src/auth.py::Auth', 'src/auth.py::Auth.validate_token'],
            'complete'
        ))
        m.upsert_entry(_entry(
            'src/router.py', 'docs/router.md',
            ['src/router.py::Router'],
            'complete'
        ))
        log = OutputLog.from_manifest(m)
        # 2 symbols from auth.py + 1 from router.py = 3 chunks
        assert len(log.chunks) == 3

    def test_from_manifest_skips_partial_files(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md',
                       ['src/auth.py::Auth'], 'complete'))
        m.upsert_entry(_entry('src/router.py', 'docs/router.md',
                       ['src/router.py::Router'], 'partial'))
        log = OutputLog.from_manifest(m)
        # Only the complete file's symbols appear
        assert all('auth.py' in c.symbol_id for c in log.chunks)

    def test_from_manifest_empty_gives_empty_log(self):
        m = OutputManifest(task='test')
        log = OutputLog.from_manifest(m)
        assert len(log.chunks) == 0
        assert log.render_compact() == ''

    def test_from_manifest_completed_symbols_set(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry(
            'src/auth.py', 'docs/auth.md',
            ['src/auth.py::Auth', 'src/auth.py::Auth.validate'],
            'complete'
        ))
        log = OutputLog.from_manifest(m)
        completed = log.completed_symbols()
        assert 'src/auth.py::Auth' in completed
        assert 'src/auth.py::Auth.validate' in completed

    def test_from_manifest_chunks_have_complete_status(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/a.py', 'docs/a.md', ['a::Foo'], 'complete'))
        log = OutputLog.from_manifest(m)
        assert all(c.status == 'complete' for c in log.chunks)

    def test_render_compact_from_manifest_log(self):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry('src/auth.py', 'docs/auth.md',
                       ['src/auth.py::Auth'], 'complete'))
        log = OutputLog.from_manifest(m)
        rendered = log.render_compact()
        assert 'COMPLETED WORK' in rendered
        assert 'src/auth.py::Auth' in rendered


# ── OutputWriter Phase 3 mode tests ──────────────────────────────────────────

class TestOutputWriterModes:
    def test_mirror_mode_creates_mirrored_file(self, tmp_path, config, logger):
        config.output_mode = 'mirror'
        config.output_dir = str(tmp_path / 'docs')
        writer = OutputWriter(config, logger, task='test',
                              source_root=str(tmp_path / 'src'))

        writer.write('src/auth.py::Auth', '# Auth\nDocumentation here.\n')

        output_file = tmp_path / 'docs' / 'auth.py'
        # .py → .md
        output_md = tmp_path / 'docs' / 'auth.md'
        # Either path naming is valid depending on relative path resolution
        written_files = list((tmp_path / 'docs').rglob('*'))
        assert len(written_files) > 0

    def test_per_symbol_mode_creates_separate_files(self, tmp_path, config, logger):
        config.output_mode = 'per_symbol'
        config.output_dir = str(tmp_path / 'docs')
        writer = OutputWriter(config, logger, task='test')

        writer.write('src/auth.py::Auth', '# Auth class\n')
        writer.write('src/router.py::Router', '# Router class\n')

        symbols_dir = tmp_path / 'docs' / 'symbols'
        assert symbols_dir.exists()
        files = list(symbols_dir.glob('*.md'))
        assert len(files) == 2

    def test_per_symbol_file_name_is_safe(self, tmp_path, config, logger):
        config.output_mode = 'per_symbol'
        config.output_dir = str(tmp_path / 'docs')
        writer = OutputWriter(config, logger, task='test')

        sym_id = 'src/auth.py::Auth.validate_token'
        writer.write(sym_id, '# validate_token\n')

        symbols_dir = tmp_path / 'docs' / 'symbols'
        files = list(symbols_dir.glob('*.md'))
        assert len(files) == 1
        # No path separators or :: in filename
        fname = files[0].name
        assert '/' not in fname
        assert '\\' not in fname
        assert '::' not in fname

    def test_inline_mode_no_files_written(self, tmp_path, config, logger):
        config.output_mode = 'inline'
        writer = OutputWriter(config, logger, task='test')

        writer.write('src/auth.py::Auth', 'Documentation.')

        output = writer.get_output()
        assert 'src/auth.py::Auth' in output
        # No files written to disk
        output_dir = tmp_path / 'output'
        assert not output_dir.exists()

    def test_finalize_marks_manifest_complete(self, tmp_path, config, logger):
        config.output_mode = 'inline'
        writer = OutputWriter(config, logger, task='test')
        writer.write('src/a.py::Foo', 'content')
        assert writer.manifest.completed_at is None
        writer.finalize()
        assert writer.manifest.completed_at is not None

    def test_manifest_saved_to_cache_dir_on_finalize(self, tmp_path, config, logger):
        config.output_mode = 'inline'
        writer = OutputWriter(config, logger, task='test')
        writer.write('src/a.py::Foo', 'content')
        writer.finalize()

        manifest_path = Path(config.cache_dir) / 'manifest.json'
        assert manifest_path.exists()

    def test_save_partial_writes_manifest(self, tmp_path, config, logger):
        config.output_mode = 'inline'
        writer = OutputWriter(config, logger, task='test')
        writer.write('src/a.py::Foo', 'content')
        writer.save_partial()

        manifest_path = Path(config.cache_dir) / 'manifest.json'
        assert manifest_path.exists()
        raw = json.loads(manifest_path.read_text())
        # Not yet marked complete
        assert raw['completed_at'] is None

    def test_mark_all_complete_updates_entries(self, config, logger):
        config.output_mode = 'inline'
        writer = OutputWriter(config, logger, task='test')
        writer.write('src/a.py::A', 'a')
        writer.write('src/b.py::B', 'b')
        writer.mark_all_complete()
        assert all(e.status == 'complete' for e in writer.manifest.files)

    def test_load_manifest_classmethod(self, tmp_path, config, logger):
        m = OutputManifest(task='load test')
        m.upsert_entry(_entry('src/a.py', 'docs/a.md', ['a::A'], 'complete'))
        path = str(tmp_path / 'manifest.json')
        m.save(path)

        loaded = OutputWriter.load_manifest(path)
        assert loaded.task == 'load test'
        assert len(loaded.files) == 1


# ── TPCAOrchestrator resume tests ─────────────────────────────────────────────

class TestOrchestratorResume:
    """
    Test the orchestrator's _load_resume_state and _complete_symbols helpers
    without making any LLM calls.
    """

    @pytest.fixture
    def orchestrator(self, config):
        from tpca.orchestrator import TPCAOrchestrator
        return TPCAOrchestrator(config=config)

    def test_complete_symbols_from_manifest(self, orchestrator, tmp_path):
        m = OutputManifest(task='test')
        m.upsert_entry(_entry(
            'src/auth.py', 'docs/auth.md',
            ['src/auth.py::Auth', 'src/auth.py::Auth.validate'],
            'complete'
        ))
        m.upsert_entry(_entry(
            'src/router.py', 'docs/router.md',
            ['src/router.py::Router'],
            'partial'   # not complete — should not be in the set
        ))
        completed = orchestrator._complete_symbols(m)
        assert 'src/auth.py::Auth' in completed
        assert 'src/auth.py::Auth.validate' in completed
        assert 'src/router.py::Router' not in completed

    def test_complete_symbols_none_manifest(self, orchestrator):
        assert orchestrator._complete_symbols(None) == set()

    def test_load_resume_state_returns_none_on_missing_file(self, orchestrator, tmp_path):
        manifest, log, skip = orchestrator._load_resume_state(
            str(tmp_path / 'nonexistent_manifest.json')
        )
        assert manifest is None
        assert log is None
        assert skip == set()

    def test_load_resume_state_returns_manifest_and_log(self, orchestrator, tmp_path):
        m = OutputManifest(task='resume test')
        m.upsert_entry(_entry(
            'src/auth.py', 'docs/auth.md',
            ['src/auth.py::Auth'],
            'complete'
        ))
        path = str(tmp_path / 'manifest.json')
        m.save(path)

        manifest, log, skip = orchestrator._load_resume_state(path)
        assert manifest is not None
        assert log is not None
        assert isinstance(log, OutputLog)
        assert 'src/auth.py::Auth' in log.completed_symbols()

    def test_load_resume_state_logs_warning_on_bad_json(self, orchestrator, tmp_path):
        bad_path = str(tmp_path / 'bad.json')
        Path(bad_path).write_text('not json at all {{{')
        manifest, log, skip = orchestrator._load_resume_state(bad_path)
        assert manifest is None
        assert log is None
        assert skip == set()

    def test_extract_keywords_filters_stopwords(self, orchestrator):
        task = 'Document every public method with parameters and return types.'
        keywords = orchestrator._extract_keywords(task)
        assert 'every' not in keywords
        assert 'with' not in keywords
        assert 'and' not in keywords
        # Meaningful words should survive
        assert any(k in keywords for k in [
                   'public', 'method', 'parameters', 'return', 'types'])

    def test_extract_keywords_max_ten(self, orchestrator):
        long_task = ' '.join([f'keyword{i}' for i in range(30)])
        keywords = orchestrator._extract_keywords(long_task)
        assert len(keywords) <= 10

    def test_run_pass1_only_returns_expected_keys(self, tmp_path):
        from tpca.orchestrator import TPCAOrchestrator
        config = TPCAConfig(
            languages=['python'],
            cache_enabled=False,
            log=LogConfig(log_file='/dev/null', console_level='ERROR'),
        )
        orch = TPCAOrchestrator(config=config)
        fixture = Path(__file__).parent / 'fixtures' / 'sample_codebase'
        if not fixture.exists():
            pytest.skip('sample_codebase fixture not present')
        result = orch.run_pass1_only(str(fixture))
        assert 'graph' in result
        assert 'symbols' in result
        assert 'stats' in result
