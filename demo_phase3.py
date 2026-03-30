"""
TPCA Phase 3 Demo
Demonstrates Phase 3 capabilities on top of the two-pass pipeline:
  1. Multi-language indexing (JavaScript alongside Python)
  2. mirror output mode (per-file .md output under docs/)
  3. OutputManifest write + resume from an interrupted run
  4. ChunkedFallback pipeline (simulated over-budget scenario)

Requirements:
    pip install tree-sitter-python tree-sitter-javascript networkx anthropic tiktoken

Usage:
    # Full demo with real LLM synthesis (mirror mode):
    ANTHROPIC_API_KEY=sk-ant-... python demo_phase3.py

    # Pass 1 only (no API key):
    python demo_phase3.py

    # Override task:
    TPCA_TASK="Explain every class" python demo_phase3.py

    # Demo resume from prior run:
    TPCA_RESUME=1 python demo_phase3.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tpca import (
    TPCAConfig, LogConfig, StructuredLogger,
    ASTIndexer, GraphBuilder, GraphRanker, IndexRenderer, IndexCache,
    LLMClient, ChunkedFallback, AgentMemoryStore, TPCAOrchestrator,
)
from tpca.models.output import OutputManifest


def banner(title: str, char: str = '=') -> None:
    width = 65
    print(char * width)
    print(f'  {title}')
    print(char * width)
    print()


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f'  {title}')
    print(f"{'─' * 65}")


def main():
    banner('TPCA Phase 3 Demo — Multi-Language, Mirror Mode, Fallback & Resume')

    task = os.environ.get(
        'TPCA_TASK',
        'Document every public method and class with parameters, '
        'return types, and a 1–2 sentence description.',
    )
    do_resume = os.environ.get('TPCA_RESUME', '').strip() == '1'
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    print(f'📋 Task:        {task[:75]}{"..." if len(task) > 75 else ""}')
    print(f'🔑 LLM:         {"✅ Anthropic" if api_key else "✅ Ollama (qwen2.5-coder:14B)"}')
    print(f'🔁 Resume mode: {"✅ Yes" if do_resume else "❌ No (fresh run)"}')
    print()

    # ── Configuration ─────────────────────────────────────────────────────────
    cache_dir = '.tpca_cache_phase3'
    output_dir = './tpca_phase3_output'
    manifest_path = f'{cache_dir}/manifest.json'

    config = TPCAConfig(
        languages=['python', 'javascript'],   # Phase 3: multi-language
        cache_enabled=True,
        cache_dir=cache_dir,
        pagerank_alpha=0.85,
        model_context_window=8192,
        context_budget_pct=0.65,
        provider='anthropic' if api_key else 'ollama',
        ollama_reader_model='qwen2.5-coder:14B',
        ollama_synthesis_model='qwen2.5-coder:14B',
        output_mode='mirror',                 # Phase 3: mirror mode
        output_dir=output_dir,
        fallback_enabled=True,                # Phase 3: fallback active
        fallback_chunk_tokens=1800,
        fallback_overlap_tokens=150,
        resume_manifest=manifest_path if do_resume else None,
        log=LogConfig(
            console_level='INFO',
            log_file=f'{cache_dir}/demo_phase3.log',
        ),
    )

    logger = StructuredLogger(config.log)

    # ── SECTION 1: Multi-Language Pass 1 ─────────────────────────────────────
    section('PASS 1 — Multi-Language AST Indexing (Python + JavaScript)')

    cache = IndexCache(config, logger)
    indexer = ASTIndexer(config, logger, cache)
    builder = GraphBuilder(config, logger)
    ranker = GraphRanker(config, logger)
    renderer = IndexRenderer(config, logger)

    # Index both the Python and JavaScript fixture codebases
    py_dir = Path(__file__).parent / 'tests' / 'fixtures' / 'sample_codebase'
    js_dir = Path(__file__).parent / 'tests' / 'fixtures' / 'sample_js_codebase'

    sources_found = []
    for d, label in [(py_dir, 'Python'), (js_dir, 'JavaScript')]:
        if d.exists():
            sources_found.append((str(d), label))
            print(f'  📂 Found {label} fixture: {d}')
        else:
            print(f'  ⚠️  {label} fixture not found at {d}')
    print()

    if not sources_found:
        print('❌ No fixture directories found. Run from the project root.')
        return

    all_symbols = []
    for source_dir, label in sources_found:
        print(f'  Step: Indexing {label} files with Tree-sitter AST...')
        try:
            symbols = indexer.index(source_dir)
            print(f'    ✓ {label}: {len(symbols)} symbols')
            all_symbols.extend(symbols)
        except Exception as exc:
            print(f'    ⚠️  {label} indexing skipped: {exc}')

    if not all_symbols:
        print('\n❌ No symbols indexed.')
        return

    print(f'\n  Total symbols across all languages: {len(all_symbols)}')

    # Build combined graph
    print('\n  Building cross-file symbol graph...')
    graph = builder.build(all_symbols)
    print(f'  ✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')

    print('\n  Ranking with task-biased PageRank...')
    keywords = ['auth', 'validate', 'token', 'router', 'hash', 'user', 'password']
    graph = ranker.rank_symbols(graph, keywords)
    print(f'  ✓ Ranked (keywords: {keywords})')

    compact_index = renderer.render(graph)
    print(f'  ✓ Index: {len(compact_index)} chars (~{len(compact_index)//4} tokens)')

    # Show top symbols across all languages
    section('Top 12 Ranked Symbols (All Languages)')
    top = ranker.get_top_symbols(graph, n=12)
    for i, (sym_id, score) in enumerate(top, 1):
        tier = graph.nodes[sym_id].get('tier', '')
        # Shorten the ID for display
        parts = sym_id.split('/')
        short_id = '/'.join(parts[-2:]) if len(parts) >= 2 else sym_id
        lang = 'JS' if '.js' in sym_id else 'PY'
        print(f'  {i:2}. [{tier:11}] [{lang}] {score:.5f}  {short_id}')
    print()

    # ── SECTION 2: Mirror Output Mode ─────────────────────────────────────────
    section('PASS 2 — Mirror Output Mode  (output/ mirrors src/)')

    has_llm = bool(api_key) or config.provider == "ollama"

    if not has_llm:
        print('  ⚠️  No ANTHROPIC_API_KEY — demonstrating mirror mode structure only.')
        _demo_mirror_structure(config, output_dir)
        _demo_manifest_structure(manifest_path)
        _demo_fallback_structure(config, logger, all_symbols)
        return

    # ── Full pipeline with real LLM ───────────────────────────────────────────
    resume_manifest = manifest_path if do_resume and Path(manifest_path).exists() else None
    if resume_manifest:
        print(f'  🔁 Resuming from manifest: {resume_manifest}')
        prior = OutputManifest.load(resume_manifest)
        complete = sum(1 for e in prior.files if e.status == 'complete')
        total = len(prior.files)
        print(f'     Prior run: {complete}/{total} files complete')
    print()

    orchestrator = TPCAOrchestrator(config=config)
    print(f'  🚀 Running full pipeline (output mode: mirror → {output_dir})...')

    try:
        result = orchestrator.run(
            source=sources_found[0][0],   # primary Python fixture
            task=task,
            resume_manifest=resume_manifest,
        )
    except KeyboardInterrupt:
        print('\n\n  ⚠️  Interrupted — partial manifest saved to', manifest_path)
        return

    # ── Results ────────────────────────────────────────────────────────────────
    section('Mirror Output — Files Written')
    manifest = result.get('manifest')
    if manifest:
        for entry in manifest.files:
            status_icon = '✅' if entry.status == 'complete' else '🔲'
            print(f'  {status_icon} {entry.source_file}')
            print(f'       → {entry.output_file}')
            print(f'       Symbols: {len(entry.symbols_processed)}, '
                  f'Chunks: {entry.chunk_count}, Status: {entry.status}')
    else:
        print('  (inline mode — no files written)')
        for key, content in result.get('output', {}).items():
            print(f'\n  ── {key} ──')
            print(content[:400])
            if len(content) > 400:
                print(f'  ... [{len(content)-400} more chars]')

    section('Performance Stats')
    stats = result.get('stats', {})
    items = [
        ('Symbols indexed',      stats.get('symbols_indexed', len(all_symbols))),
        ('LLM calls',            stats.get('llm_calls', '?')),
        ('Compression ratio',    f"{stats.get('compression_ratio', '?')}x"),
        ('Fallback used',        stats.get('fallback_used', False)),
        ('Output chunks',        stats.get('output_chunks', '?')),
        ('Total time',           f"{stats.get('wall_time_ms', '?')}ms"),
    ]
    for label, value in items:
        print(f'  {label:<25} {value}')

    section('Output Log (Working Memory)')
    log_text = result.get('log', '')
    if log_text:
        print(log_text)
    else:
        print('  (empty)')

    print()
    print('✅ Phase 3 demo complete!')
    print(f'   Logs:     {config.log.log_file}')
    print(f'   Output:   {output_dir}/')
    print(f'   Manifest: {manifest_path}')
    if not do_resume:
        print()
        print('  💡 To resume an interrupted run:  TPCA_RESUME=1 python demo_phase3.py')


# ── No-LLM structure demonstrations ──────────────────────────────────────────

def _demo_mirror_structure(config: TPCAConfig, output_dir: str) -> None:
    section('Mirror Mode Output Structure (preview)')
    print('  With a real LLM, the mirror output would look like:')
    print()
    print(f'  {output_dir}/')
    print(f'  ├── auth.md          ← mirrored from auth.py')
    print(f'  ├── router.md        ← mirrored from router.py')
    print(f'  └── utils.md         ← mirrored from utils.py')
    print()
    print(f'  {config.cache_dir}/')
    print(f'  └── manifest.json    ← completion record (enables resume)')
    print()

    # Show what a manifest looks like
    example_manifest = OutputManifest(
        task='Document every public method.',
        output_mode='mirror',
    )
    from tpca.models.output import ManifestEntry
    example_manifest.upsert_entry(ManifestEntry(
        source_file='tests/fixtures/sample_codebase/auth.py',
        output_file=f'{output_dir}/auth.md',
        symbols_processed=[
            'tests/fixtures/sample_codebase/auth.py::Auth',
            'tests/fixtures/sample_codebase/auth.py::Auth.validate_token',
        ],
        chunk_count=2, token_count=480, status='complete',
    ))
    example_manifest.upsert_entry(ManifestEntry(
        source_file='tests/fixtures/sample_codebase/router.py',
        output_file=f'{output_dir}/router.md',
        symbols_processed=['tests/fixtures/sample_codebase/router.py::Router'],
        chunk_count=1, token_count=210, status='partial',  # interrupted
    ))
    print('  Example manifest.json:')
    print('  ' + json.dumps(example_manifest.to_dict(), indent=2).replace('\n', '\n  '))
    print()


def _demo_manifest_structure(manifest_path: str) -> None:
    section('Resume Capability')
    print('  The manifest.json written at the end of each run enables resume:')
    print()
    print('  # If a run is interrupted, restart with:')
    print('  TPCA_RESUME=1 python demo_phase3.py')
    print()
    print('  The orchestrator will:')
    print('    1. Load manifest.json from .tpca_cache_phase3/')
    print('    2. Identify files with status != "complete"')
    print('    3. Rehydrate the OutputLog for cross-file context')
    print('    4. Skip already-complete symbols and resume from where it left off')
    print()


def _demo_fallback_structure(
    config: TPCAConfig,
    logger: StructuredLogger,
    symbols: list,
) -> None:
    section('ChunkedFallback — Over-Budget Subgraph Handling')
    print('  The ChunkedFallback activates when the relevant subgraph exceeds')
    print('  the context budget even after primary/supporting truncation.')
    print()
    print('  Flow:')
    print('    SliceFetcher detects budget overflow')
    print('    → ChunkedFallback.run(relevant_symbols, source_slices, task)')
    print('    → Partitions subgraph into chunks of ≤ 1800 tokens')
    print('    → Each chunk processed by an ephemeral ReaderAgent (reader model)')
    print('    → Extractions accumulated in AgentMemoryStore')
    print('    → AgentMemoryStore passed to SynthesisAgent instead of raw slices')
    print()
    print('  Key property: Even in fallback, only the RELEVANT subgraph is')
    print('  chunked — not the full codebase. Pass 1 filtering still applies.')
    print()

    # Show AgentMemoryStore render_compact() format
    store = AgentMemoryStore()
    store.add_extraction(
        chunk_id=0,
        symbol_ids=['tests/fixtures/sample_codebase/auth.py::Auth'],
        summary=(
            'Auth.__init__ accepts config dict with secret and expirySeconds. '
            'validate_token checks JWT expiry, returns bool; raises ValueError on '
            'tampered token. refresh_token returns new JWT; requires valid input.'
        ),
        token_count=320,
    )
    store.add_extraction(
        chunk_id=1,
        symbol_ids=['tests/fixtures/sample_codebase/utils.py::hash_password',
                    'tests/fixtures/sample_codebase/utils.py::generate_token'],
        summary=(
            'hash_password uses bcrypt; returns hex digest. '
            'generate_token returns URL-safe base64 string of byteLength bytes.'
        ),
        token_count=180,
    )

    llm = LLMClient(config, logger)
    store_tokens = llm.count_tokens(store.render_compact())

    print('  Example AgentMemoryStore.render_compact() output:')
    print()
    for line in store.render_compact().splitlines():
        print(f'  {line}')
    print()
    print(f'  ↑ Store = ~{store_tokens} tokens '
          f'(vs full source which could be thousands)')
    print()
    print(f'  fallback_used flag: {store.fallback_used}')
    print(f'  chunks_processed:   {store.chunks_processed}')
    print()


if __name__ == '__main__':
    try:
        main()
    except ImportError as exc:
        print(f'❌ Missing dependency: {exc}')
        print('\nInstall required packages:')
        print('  pip install tree-sitter-python tree-sitter-javascript networkx anthropic tiktoken')
    except KeyboardInterrupt:
        print('\n\nInterrupted.')
    except Exception as exc:
        import traceback
        print(f'❌ Error: {exc}')
        traceback.print_exc()
