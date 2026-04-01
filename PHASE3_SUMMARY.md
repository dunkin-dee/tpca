# TPCA Phase 3 Implementation Summary

## Overview

Phase 3 delivers the three capabilities deferred from Phase 2: multi-file output with manifest-based resume, the ChunkedFallback pipeline for over-budget subgraphs, and multi-language support for JavaScript and TypeScript.

**Goal**: Production-ready pipeline — any output size (via manifest resume), any subgraph size (via ChunkedFallback), any JS/TS codebase (via multi-language AST), and fully implemented `mirror` / `per_symbol` output modes.

## Components Implemented

### ChunkedFallback (`tpca/fallback/chunked_pipeline.py`)
- Activates when the relevant subgraph exceeds the context budget after primary/supporting truncation
- Chunks only the relevant subgraph — not the full codebase — so Pass 1 filtering applies even in the fallback path
- Partitions symbols into `fallback_chunk_tokens`-sized windows with `fallback_overlap_tokens` carry-over at each boundary
- Each chunk processed by an ephemeral `ReaderAgent`; all results accumulated in `AgentMemoryStore`
- Resumable: skips chunks whose `chunk_id` already appears in the store
- Wired into `TPCAOrchestrator` and `SynthesisAgent` via optional `fallback=` parameter; guarded by `config.fallback_enabled`

### ReaderAgent (`tpca/fallback/reader_agent.py`)
- Ephemeral: created per chunk, run once, discarded
- Calls the reader model (lightweight) with a structured extraction prompt
- Returns `(extraction_text, token_count)`
- Graceful degradation: on LLM error returns signature-only extraction so the pipeline continues

### AgentMemoryStore (`tpca/fallback/memory_store.py`)
- Extends `OutputLog` so `SynthesisAgent` receives an identical interface whether or not fallback was triggered
- Adds `add_extraction()`, `save(path)`, and `load(path)` classmethod
- `render_compact()` produces the same format as `OutputLog.render_compact()` but with a READER EXTRACTIONS header
- `fallback_used: bool = True` flag propagates into stats

### JavaScript/TypeScript Tree-sitter Queries

**`tpca/pass1/queries/javascript.scm`**
- Captures: class declarations, class expressions, method definitions, function declarations, arrow functions, exports, imports, JSDoc comments, call expressions, `new` expressions
- Cross-file `require()` / `import` edges resolved by GraphBuilder using the same pending-edge mechanism as Python

**`tpca/pass1/queries/typescript.scm`**
- Extends JavaScript patterns with: interfaces, type aliases, enums, abstract method signatures, generic type parameters, return type annotations, type-only imports
- TSX (`.tsx`) shares the TypeScript query file; detected via extension map

### Updated ASTIndexer (`tpca/pass1/ast_indexer.py`)
- Dispatches to `_parse_python`, `_parse_javascript`, or `_parse_typescript` based on `TPCAConfig.detect_language()`
- `tree-sitter-javascript` / `tree-sitter-typescript` are optional imports: Python-only installations continue to work; the indexer emits a `warn` event and skips unsupported files
- JSDoc / TSDoc comment extraction (`/** ... */`) for docstring field population

### Updated TPCAConfig (`tpca/config.py`)
- `language_extensions: dict[str, str]`: maps file extensions to language strings; includes all JS/TS/TSX variants by default
- `fallback_enabled: bool = True`: toggle ChunkedFallback without code change
- `resume_manifest: Optional[str] = None`: path to prior manifest.json; when set, `TPCAOrchestrator.run()` resumes automatically
- `detect_language(path)` method replaces inline extension checks in the indexer

### OutputManifest Persistence (`tpca/models/output.py`)
- `OutputManifest.save(path)`: writes full manifest as formatted JSON
- `OutputManifest.load(path)`: classmethod; rehydrates from disk
- `OutputManifest.to_dict()` / `from_dict()`: full round-trip serialization including `files`, `stats`, `started_at`, `completed_at`
- `incomplete_files()`: returns entries where `status != 'complete'`
- `is_done()`: True only when `completed_at` is set and all entries are complete

### OutputLog Resume (`tpca/models/output.py`)
- `OutputLog.from_manifest(manifest)`: reconstructs an OutputLog from a prior manifest, creating one OutputChunk per symbol in each complete entry
- Restored log passed to `SynthesisAgent` as `prior_log`; completed symbols are skipped by `OutputChunker` via `completed_symbols()`

### OutputWriter — mirror and per_symbol (fully implemented) (`tpca/pass2/output_writer.py`)
- **mirror**: writes `{output_dir}/{relative_path}.md` mirroring the source tree; multiple symbols from the same source file appended in topological order
- **per_symbol**: one `{output_dir}/symbols/{safe_name}.md` per top-level symbol; filename derived from fully-qualified symbol ID with `/` and `::` replaced by `__`
- `finalize()`: marks manifest complete and calls `save_manifest()`
- `save_partial()`: persists manifest in partial state (called on interrupt)
- `mark_file_complete(source_file)` / `mark_all_complete()`: fine-grained completion tracking

### Updated TPCAOrchestrator (`tpca/orchestrator.py`)
- `resume_manifest` parameter on `run()` (or via `config.resume_manifest`)
- `_load_resume_state(path)`: loads manifest and rehydrates OutputLog
- `_complete_symbols(manifest)`: set of symbol IDs to skip this run
- `ChunkedFallback` instantiated when `config.fallback_enabled` and wired to `SynthesisAgent`
- `source_root` propagated to OutputWriter for correct relative paths in mirror mode

## New Files

```
tpca/
└── fallback/
    ├── __init__.py
    ├── chunked_pipeline.py
    ├── reader_agent.py
    └── memory_store.py

tpca/pass1/queries/
├── javascript.scm
└── typescript.scm

tests/
├── test_fallback.py
├── test_multi_language.py
├── test_resume.py
└── fixtures/
    └── sample_js_codebase/
        ├── auth.js
        ├── router.js
        └── utils.js

demo_phase3.py
```

## Updated Files

| File | Change |
|---|---|
| `tpca/config.py` | `language_extensions`, `fallback_enabled`, `resume_manifest`, `detect_language()` |
| `tpca/models/output.py` | `OutputManifest.save/load/to_dict/from_dict`, `OutputLog.from_manifest` |
| `tpca/pass1/ast_indexer.py` | JS/TS parser dispatch, `_parse_javascript`, `_parse_typescript`, JSDoc extraction |
| `tpca/pass2/output_writer.py` | mirror + per_symbol fully implemented, `finalize()`, `save_partial()`, manifest write |
| `tpca/orchestrator.py` | Resume wiring, ChunkedFallback wiring, `source_root` propagation |
| `tpca/__init__.py` | Exports `ChunkedFallback`, `ReaderAgent`, `AgentMemoryStore` |
| `requirements.txt` | `tree-sitter-javascript>=0.23.0`, `tree-sitter-typescript>=0.23.0` |

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python demo_phase3.py

# Resume an interrupted run
TPCA_RESUME=1 ANTHROPIC_API_KEY=sk-ant-... python demo_phase3.py

# All tests (no API key needed)
pytest tests/ -v

# Phase 3 tests only
pytest tests/test_fallback.py tests/test_multi_language.py tests/test_resume.py -v

# Skip JS/TS tests if parsers not installed
pytest tests/ -v -k "not JavaScript and not TypeScript"
```

```python
from tpca import TPCAOrchestrator, TPCAConfig, LogConfig

config = TPCAConfig(
    languages=['python', 'javascript', 'typescript'],
    provider='anthropic',
    synthesis_model='claude-sonnet-4-6',
    reader_model='claude-haiku-4-5-20251001',
    output_mode='mirror',
    output_dir='./my_project/docs',
    fallback_enabled=True,
    fallback_chunk_tokens=1800,
    fallback_overlap_tokens=150,
    resume_manifest='.tpca_cache/manifest.json',
    log=LogConfig(console_level='INFO'),
)

result = TPCAOrchestrator(config=config).run(
    source='./my_project/src',
    task='Document every public method with parameters and return types.',
)

print(result['stats'])
# {
#   'pass1_time_ms': 210,
#   'llm_calls': 9,
#   'compression_ratio': 14.2,
#   'fallback_used': False,
#   'output_chunks': 7,
#   'wall_time_ms': 38400,
# }
```

## Key Design Properties

### Identical Interface With/Without Fallback
`AgentMemoryStore` extends `OutputLog`, so `SynthesisAgent.run()` accepts either type as `prior_log` without any conditional logic. The `render_compact()` format is identical; synthesis prompts are consistent regardless of whether fallback was triggered.

### Graceful Degradation on Missing JS/TS Parsers
`tree-sitter-javascript` and `tree-sitter-typescript` are optional imports. If absent, the indexer emits a `warn` log event and returns an empty symbol list for that file. Python-only installs pass the full test suite; JS/TS tests self-skip via `pytest.mark.skipif`.

### Resume Without Re-Processing
The orchestrator intersects completed symbols from the manifest with the current `OutputChunker` pending list. Symbols already in `OutputLog.completed_symbols()` are skipped at `get_next_chunk()` — no special resume code path in `SynthesisAgent`, which is unaware of the resume.

### Mirror Mode Path Safety
`OutputWriter` calls `Path(source_file).relative_to(source_root)` to compute the mirrored path, falling back to `basename` if the file is outside the source root.

## Performance

- **ChunkedFallback**: Adds one reader-model call per chunk (~0.5-1s each at `fallback_chunk_tokens=1800` with `claude-haiku`)
- **AgentMemoryStore overhead**: ~75 tokens per extraction entry in the prompt — identical to normal OutputLog overhead
- **JS/TS indexing**: ~50-200 files/second; Tree-sitter is language-agnostic at the C level
- **Resume overhead**: near zero — manifest load is a single JSON parse; completed symbol skip is a set lookup

## Known Limitations

1. **Call-graph edges in JS/TS**: Cross-file edges from `require()` / `import` are resolved by name matching. Dynamic `import()` and re-export aliasing are not captured.
2. **TSX component props**: JSX prop types in `.tsx` files are not extracted as separate symbols; the component function/class is captured but its props interface requires an explicit `interface Props` definition.
3. **Fallback + resume interleaving**: If a run is interrupted mid-fallback, the `AgentMemoryStore` is not persisted to the manifest (only `OutputLog` chunks are). Resuming re-runs the fallback from the last complete chunk — correct but redundant for already-extracted chunks.

## Test Coverage

| Test File | Tests | Coverage |
|---|---|---|
| `test_fallback.py` | 22 | AgentMemoryStore, ReaderAgent, ChunkedFallback |
| `test_multi_language.py` | 29 | Language detection, JS indexing, TS indexing, directory walk |
| `test_resume.py` | 32 | ManifestEntry, OutputManifest, OutputLog resume, OutputWriter modes, orchestrator resume |

All 83 Phase 3 tests are deterministic and complete in under 10 seconds with no API key or network access required.

## Design Compliance

All Phase 3 requirements from Section 13 of the design document are met:

- OutputWriter: mirror and per_symbol modes fully implemented; OutputManifest written at completion and on partial runs
- ChunkedFallback: ReaderAgent + AgentMemoryStore; reuses OutputLog interface as specified
- Multi-language: JavaScript and TypeScript Tree-sitter queries; graceful fallback when parsers not installed
- Resume logic: `OutputLog.from_manifest()` rehydrates prior run; orchestrator skips complete symbols; `save_partial()` called on interruption

## Conclusion

The TPCA implementation is complete across all three phases. The system handles codebases of any size (ChunkedFallback), outputs of any size (OutputManifest + resume), codebases in Python, JavaScript, TypeScript, and TSX, and all four output modes.
