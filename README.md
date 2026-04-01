# TPCA - Two-Pass Context Agent

AST-driven, graph-ranked context retrieval for limited-window LLMs. TPCA implements a complete three-phase pipeline: Pass 1 (zero-LLM indexing), Pass 2 (LLM-driven synthesis), and Phase 3 (multi-language, fallback, resume).

## What This Does

TPCA analyzes Python, JavaScript, and TypeScript codebases and generates documentation or other task-driven outputs using a bounded-context approach:

**Pass 1 (Deterministic, Zero LLM)**: Creates a compact, ranked index of all code symbols without any LLM calls.

**Pass 2 (LLM-Driven Synthesis)**: Uses the compact index to guide an LLM in generating detailed outputs, with bounded context regardless of output size.

**Phase 3 (Advanced)**: Multi-language support, fallback pipeline for over-budget subgraphs, and manifest-based resume.

**Features:**
- Multi-file parsing for Python, JavaScript, TypeScript, and TSX via Tree-sitter AST
- Cross-file symbol relationship graph with edge types: calls, inherits, member_of
- Task-biased PageRank for symbol importance ranking
- Compact text index generation (1,000-3,000 tokens for 10K lines of code)
- Provider-agnostic LLM client (Anthropic Claude or Ollama)
- tiktoken-accurate token counting and budget management
- Structured logging with file, console, and ring buffer outputs
- Per-file caching with automatic hash-based invalidation
- Four output modes: inline, single_file, mirror, per_symbol
- ChunkedFallback pipeline for subgraphs that exceed context budget
- OutputManifest-based resume for interrupted runs

## Installation

**Requirements:** Python 3.10+, and either an Anthropic API key or a running Ollama instance.

```bash
pip install -r requirements.txt
```

**Minimal install (Python-only, no LLM synthesis):**
```bash
pip install tree-sitter-python networkx
```

**Without JS/TS support:**
```bash
pip install tree-sitter-python networkx anthropic tiktoken openai click prompt_toolkit
# JS/TS files are silently skipped if tree-sitter-javascript/typescript are absent
```

### LLM Setup

**Anthropic Claude:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Ollama (local):**
```bash
# Install Ollama, then pull a model
ollama pull qwen2.5-coder:14B
# TPCA auto-detects Ollama if ANTHROPIC_API_KEY is not set
```

## Quick Start

**Pass 1 demo (no API key needed):**
```bash
python demo_phase1.py
```

**Full two-pass pipeline:**
```bash
ANTHROPIC_API_KEY=sk-ant-... python demo_phase2.py
```

**Multi-language, mirror mode, resume:**
```bash
ANTHROPIC_API_KEY=sk-ant-... python demo_phase3.py
```

**Use as a library:**
```python
from tpca import TPCAOrchestrator, TPCAConfig

config = TPCAConfig(
    provider="anthropic",
    reader_model="claude-haiku-4-5-20251001",
    synthesis_model="claude-sonnet-4-6",
    output_mode="mirror",
    output_dir="./docs",
)

orchestrator = TPCAOrchestrator(config=config)
result = orchestrator.run(
    source="./my_project/src",
    task="Document every public method with parameters and return types.",
)

print(result['stats'])
# {'pass1_time_ms': 187, 'llm_calls': 4, 'compression_ratio': 12.5, ...}

print(result['output'])  # dict of {file: content} in inline mode
```

## Project Structure

```
tpca/
├── __init__.py                 # Package exports
├── config.py                   # TPCAConfig (unified for all phases)
├── orchestrator.py             # TPCAOrchestrator (top-level entry point)
├── logging/
│   ├── structured_logger.py    # File (JSON-lines) + console + ring buffer
│   ├── console_handler.py      # Human-readable console formatter
│   └── log_config.py           # LogConfig dataclass
├── models/
│   ├── symbol.py               # Symbol, SymbolGraph, PendingEdge
│   ├── slice.py                # Slice, SliceRequest
│   ├── output.py               # OutputLog, OutputChunk, OutputManifest
│   └── chunk_plan.py           # ChunkPlan
├── pass1/                      # Pass 1: zero-LLM indexing
│   ├── ast_indexer.py          # Multi-language Tree-sitter parser
│   ├── graph_builder.py        # Symbol relationship graph
│   ├── graph_ranker.py         # Task-biased PageRank ranking
│   ├── index_renderer.py       # Compact text index generation
│   └── queries/
│       ├── python.scm          # Tree-sitter query for Python
│       ├── javascript.scm      # Tree-sitter query for JavaScript
│       └── typescript.scm      # Tree-sitter query for TypeScript + TSX
├── pass2/                      # Pass 2: LLM-driven synthesis
│   ├── context_planner.py      # LLM symbol selection with validation/retry
│   ├── slice_fetcher.py        # Token-budgeted source slice retrieval
│   ├── output_chunker.py       # Synthesis loop, topological ordering, OutputLog
│   ├── output_writer.py        # Multi-mode output + manifest persistence
│   └── synthesis_agent.py      # Synthesis loop orchestration
├── llm/
│   └── client.py               # Provider-agnostic LLM client + TokenCounter
├── cache/
│   └── index_cache.py          # Per-file symbol cache with hash invalidation
└── fallback/                   # Phase 3: over-budget fallback
    ├── chunked_pipeline.py     # Partition subgraph into chunks
    ├── reader_agent.py         # Lightweight reader model per chunk
    └── memory_store.py         # Aggregate extractions into compact context

tests/
├── test_phase1.py
├── test_context_planner.py
├── test_slice_fetcher.py
├── test_output_chunker.py
├── test_synthesis_agent.py
├── test_llm_client.py
├── test_fallback.py            # Phase 3: ChunkedFallback + ReaderAgent
├── test_multi_language.py      # Phase 3: JS/TS indexing
├── test_resume.py              # Phase 3: manifest-based resume
└── fixtures/
    ├── sample_codebase/        # Python: auth.py, router.py, utils.py
    └── sample_js_codebase/     # JavaScript: auth.js, router.js, utils.js

demo_phase1.py                  # Pass 1 demo (no LLM)
demo_phase2.py                  # Full two-pass pipeline demo
demo_phase3.py                  # Multi-language, mirror mode, resume demo
```

## Configuration

```python
config = TPCAConfig(
    # Languages (Phase 3: JS/TS support)
    languages=['python', 'javascript', 'typescript'],

    # Paths to skip during directory walk
    exclude_patterns=['__pycache__', '.git', 'node_modules', 'dist', '.venv'],

    # Cache
    cache_dir='.tpca_cache',
    cache_enabled=True,

    # Graph ranking
    pagerank_alpha=0.85,
    top_n_symbols=50,

    # LLM
    provider='anthropic',                         # 'anthropic' or 'ollama'
    reader_model='claude-haiku-4-5-20251001',     # lightweight: planning, extraction
    synthesis_model='claude-sonnet-4-6',          # powerful: synthesis output

    # Ollama-specific
    ollama_base_url='http://localhost:11434/v1',
    ollama_reader_model='qwen2.5-coder:14B',
    ollama_synthesis_model='qwen2.5-coder:14B',

    # Token budget
    model_context_window=8192,
    context_budget_pct=0.70,
    max_planner_retries=3,

    # Output
    output_mode='mirror',          # 'inline' | 'single_file' | 'mirror' | 'per_symbol'
    output_dir='./docs',
    max_synthesis_iterations=20,

    # Phase 3: fallback
    fallback_enabled=True,
    fallback_chunk_tokens=1800,
    fallback_overlap_tokens=150,

    # Phase 3: resume
    resume_manifest='.tpca_cache/manifest.json',  # omit to start fresh

    # Logging
    log=LogConfig(
        log_file='.tpca_cache/tpca.log',
        console_level='INFO',    # DEBUG | INFO | WARN | ERROR
        file_level='DEBUG',
        ring_buffer_size=1000,
    ),
)
```

## Architecture

### Data Flow

```
Source Files (Python / JavaScript / TypeScript)
    |
    v  PASS 1 — DETERMINISTIC (ZERO LLM)
ASTIndexer -> GraphBuilder -> GraphRanker -> IndexRenderer
(Symbol[])    (DiGraph)       (ranked)       (compact text ~1-3K tokens)
    |
    v  PASS 2 — LLM-DRIVEN SYNTHESIS
ContextPlanner -> SliceFetcher -> SynthesisAgent -> OutputWriter
(LLM->SliceReq)  (Slice[])       (OutputLog loop)   (files + manifest)
                                      |
                             ChunkedFallback (if over budget)
                             ReaderAgent per chunk -> AgentMemoryStore
```

### Pass 1 Components

1. **ASTIndexer**: Parses Python/JS/TS with Tree-sitter; extracts symbols with signatures, docstrings, and line ranges; integrates with IndexCache.
2. **GraphBuilder**: Builds a NetworkX DiGraph with call, inheritance, and membership edges; resolves cross-file references.
3. **GraphRanker**: Task-biased PageRank; assigns `CORE`, `SUPPORT`, `PERIPHERAL` tier labels.
4. **IndexRenderer**: Renders the ranked subgraph as a compact, hierarchical text index.
5. **IndexCache**: Per-file symbol cache with SHA-based invalidation; avoids re-parsing unchanged files.

### Pass 2 Components

6. **LLMClient**: Provider-agnostic wrapper (Anthropic or Ollama); `TokenCounter` uses tiktoken (cl100k_base).
7. **ContextPlanner**: Sends compact index to the reader model; validates returned symbol IDs; retries with suggestions on invalid IDs; falls back to top CORE symbols.
8. **SliceFetcher**: Retrieves exact source lines per symbol; enforces tiktoken-accurate token budget; primary symbols always included.
9. **OutputChunker**: Topological processing order; maintains bounded `OutputLog` (~50-100 tokens/entry).
10. **SynthesisAgent**: Orchestrates the full synthesis loop; delegates to ChunkedFallback when a subgraph exceeds budget.
11. **OutputWriter**: Writes output in the configured mode; persists `OutputManifest` for resume capability.
12. **TPCAOrchestrator**: Top-level entry point; wires all components; handles resume from prior manifest.

### Phase 3 Components

13. **ChunkedFallback**: Partitions the relevant subgraph (not the full codebase) into overlapping token windows; dispatches each to a `ReaderAgent`.
14. **ReaderAgent**: Ephemeral, one per chunk; calls the reader model for structured extraction; gracefully returns signature-only on LLM error.
15. **AgentMemoryStore**: Extends `OutputLog`; aggregates extractions; `render_compact()` produces ~500 tokens that replace raw slices in the synthesis step.

### Key Design Properties

- **Bounded context**: `OutputLog` is O(chunks), not O(total output size).
- **Token accuracy**: All budget enforcement uses tiktoken; character-count approximation is a fallback only.
- **Resumability**: `OutputManifest` tracks completed symbols; `OutputLog.from_manifest()` reconstructs state; already-complete symbols are skipped at `get_next_chunk()`.
- **Graceful degradation**: No LLM available — Pass 1 runs fully, Pass 2 skipped with a warning. JS/TS parsers absent — those files skipped, Python unaffected. LLM returns bad symbol IDs — retried with suggestions, then falls back to top CORE by PageRank.
- **Identical interface with/without fallback**: `AgentMemoryStore` extends `OutputLog`, so `SynthesisAgent` needs no conditional logic.

## Testing

```bash
# All unit tests (no API key needed — all LLM calls mocked)
pytest tests/ -v

# Phase-specific
pytest tests/test_phase1.py -v
pytest tests/test_fallback.py tests/test_multi_language.py tests/test_resume.py -v

# Skip JS/TS tests if parsers not installed
pytest tests/ -v -k "not JavaScript and not TypeScript"

# Integration tests (requires ANTHROPIC_API_KEY or Ollama)
TPCA_RUN_INTEGRATION=1 pytest tests/ -v -m integration
```

## Performance

**Pass 1:**
- Indexing: ~50-200 files/second (Python or JS/TS — Tree-sitter is language-agnostic at the C level)
- Graph build + PageRank: ~100ms for 1,000 symbols
- Total: under 5 seconds for a 50K-line codebase

**Pass 2:**
- Context planning: 1-3 reader-model calls
- Synthesis loop: 1 synthesis-model call per top-level symbol
- Total LLM calls: approximately 2 + N (where N = symbols in scope)
- Typical compression ratio: 10-20x (raw source to compact index)

**ChunkedFallback (Phase 3):**
- Adds one reader-model call per chunk (~0.5-1s each at `fallback_chunk_tokens=1800`)
- AgentMemoryStore overhead: ~75 tokens per extraction entry in the prompt

## Design Document

See `TPCA_Design_v2.docx` for the complete technical specification.

## Status

Pass 1: complete | Pass 2: complete | Phase 3: complete
