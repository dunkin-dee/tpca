# TPCA — Two-Pass Context Agent

## Project Overview

TPCA is a Python library and CLI tool that analyzes and documents codebases within bounded LLM context windows. It solves the "too much code, too little context" problem via a two-pass pipeline:

- **Pass 1 (Deterministic)**: Tree-sitter AST parsing → NetworkX symbol graph → task-biased PageRank → compact text index. Zero LLM calls.
- **Pass 2 (LLM-Driven)**: LLM acts as librarian to request symbols → token-budgeted slices fetched → synthesis loop with bounded working memory.
- **Phase 3 (Advanced)**: Multi-language support (JS/TS), ChunkedFallback for over-budget subgraphs, resume from manifest.

Current version: **3.0.0** (despite `__version__` showing 2.0.0 in `__init__.py` — the last assignment wins at line 84).

---

## Repository Structure

```
tpca/                        # Main package
├── config.py                # TPCAConfig (single dataclass for all settings)
├── orchestrator.py          # TPCAOrchestrator — top-level entry point
├── __init__.py              # Package exports
│
├── pass1/                   # Phase 1: AST indexing (zero LLM)
│   ├── ast_indexer.py       # Tree-sitter multi-language parser
│   ├── graph_builder.py     # Symbol relationship graph
│   ├── graph_ranker.py      # Task-biased PageRank ranking
│   ├── index_renderer.py    # Compact text index generator
│   └── queries/python.scm   # Tree-sitter S-expression query
│
├── pass2/                   # Phase 2: LLM synthesis
│   ├── context_planner.py   # LLM selects symbols (SliceRequest)
│   ├── slice_fetcher.py     # Token-budgeted source retrieval
│   ├── output_chunker.py    # Synthesis loop & topological ordering
│   ├── synthesis_agent.py   # Orchestrates full synthesis
│   └── output_writer.py     # Writes output (inline/file/mirror/per_symbol)
│
├── llm/client.py            # Provider-agnostic LLM (Anthropic / Ollama)
├── cache/index_cache.py     # Per-file symbol cache with hash invalidation
│
├── fallback/                # Phase 3: Over-budget fallback
│   ├── chunked_pipeline.py  # Partition subgraph → chunks
│   ├── reader_agent.py      # Lightweight reader model per chunk
│   └── memory_store.py      # Aggregates extractions → compact context
│
├── models/                  # Core data structures
│   ├── symbol.py            # Symbol, SymbolGraph
│   ├── slice.py             # Slice, SliceRequest
│   ├── output.py            # OutputLog, OutputChunk, OutputManifest
│   └── chunk_plan.py        # ChunkPlan
│
├── logging/                 # Structured logging
│   ├── structured_logger.py # File (JSON-lines) + console + ring buffer
│   ├── console_handler.py   # Human-readable console formatter
│   └── log_config.py        # LogConfig dataclass
│
└── cli/main.py              # Click CLI + prompt_toolkit REPL

tests/                       # Pytest test suite (all LLM calls mocked)
├── fixtures/
│   ├── sample_codebase/     # Python: auth.py, router.py, utils.py
│   └── sample_js_codebase/  # JavaScript fixtures (Phase 3)
├── test_phase1.py
├── test_context_planner.py
├── test_slice_fetcher.py
├── test_synthesis_agent.py
├── test_output_chunker.py
├── test_llm_client.py
├── test_fallback.py
├── test_multi_language.py
└── test_resume.py

demo_phase1.py               # Demo: Pass 1 only (no LLM needed)
demo_phase2.py               # Demo: Full two-pass pipeline
demo_phase3.py               # Demo: Multi-language, mirror mode, resume
```

---

## Tech Stack

| Concern | Library |
|---------|---------|
| AST Parsing | `tree-sitter>=0.25.0`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript` |
| Symbol Graphs | `networkx>=3.0` |
| PageRank numerics | `numpy`, `scipy` |
| Token counting | `tiktoken>=0.6.0` (cl100k_base) |
| LLM (primary) | `anthropic>=0.25.0` |
| LLM (alternate) | `openai>=1.0.0` (Ollama via OpenAI-compatible API) |
| CLI | `click>=8.0.0`, `prompt_toolkit>=3.0` |
| Tests | `pytest>=7.0.0`, `pytest-asyncio>=0.21.0` |

---

## Setup & Running

```bash
# Install
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run Pass 1 demo (no API key needed)
python demo_phase1.py

# Run full pipeline
export ANTHROPIC_API_KEY=sk-ant-...
python demo_phase2.py
python demo_phase3.py

# Run tests (all mocked — no API key needed)
pytest tests/ -v

# Run integration tests (requires API key)
TPCA_RUN_INTEGRATION=1 pytest tests/ -m integration -v

# CLI
tpca run "document all public methods" --source ./src
tpca index ./src
tpca shell   # interactive REPL
```

---

## Key Design Decisions

### Bounded Context
`OutputLog` keeps working memory O(chunks), not O(total output size). Each synthesis call sees only the compact index + current slices + prior `OutputLog` entries — never all prior output.

### Token Accuracy
All token budgets use `tiktoken` (cl100k_base) with a 4-char-per-token fallback for robustness. Never use character counts for budget logic.

### Two Models
- **Reader model** (`claude-haiku-4-5-20251001` by default): lightweight, used for context planning and fallback chunk reading.
- **Synthesis model** (`claude-sonnet-4-6` by default): powerful, used for actual synthesis.

### Output Modes
- `inline`: dict of `{file: content}` in memory
- `single_file`: everything in one output file
- `mirror`: output directory mirrors source structure (e.g., `src/auth.py` → `docs/auth.md`)
- `per_symbol`: one file per symbol

### Resumability
`OutputManifest` (JSON) tracks which files/symbols are complete. Pass `resume_manifest=path` to `TPCAConfig` to skip already-complete work.

### Fallback Pipeline
When a relevant subgraph exceeds the context budget, `ChunkedFallback` is activated:
1. Subgraph partitioned into ~1800-token chunks (150-token overlap)
2. Each chunk processed by an ephemeral `ReaderAgent` (reader model)
3. Extractions aggregated in `AgentMemoryStore` → compacted to ~500 tokens
4. Compact memory passed to `SynthesisAgent` instead of raw slices

---

## Configuration Reference (TPCAConfig)

All configuration lives in `TPCAConfig` (`tpca/config.py`). Key fields:

```python
TPCAConfig(
    # Languages
    languages=['python'],               # 'python' | 'javascript' | 'typescript'

    # LLM
    provider='anthropic',               # 'anthropic' | 'ollama'
    reader_model='claude-haiku-4-5-20251001',
    synthesis_model='claude-sonnet-4-6',

    # Token budget
    model_context_window=8192,
    context_budget_pct=0.70,            # 70% of window allocated to slices

    # Indexing
    top_n_symbols=50,
    pagerank_alpha=0.85,
    cache_enabled=True,
    cache_dir='.tpca_cache',
    exclude_patterns=['__pycache__', '.git', 'venv'],

    # Output
    output_mode='inline',               # inline | single_file | mirror | per_symbol
    output_dir='./tpca_output',
    max_synthesis_iterations=20,

    # Phase 3 fallback
    fallback_enabled=True,
    fallback_chunk_tokens=1800,
    fallback_overlap_tokens=150,

    # Resume
    resume_manifest=None,               # path to prior manifest.json
)
```

---

## Data Flow

```
Source Files (Python / JS / TS)
    │
    ▼ Pass 1 (deterministic)
ASTIndexer ──► GraphBuilder ──► GraphRanker ──► IndexRenderer
(Symbol[])     (DiGraph)        (ranked DiGraph)  (compact text ~1-3K tokens)
    │
    ▼ Pass 2 (LLM-driven)
ContextPlanner ──► SliceFetcher ──► SynthesisAgent (loop) ──► OutputWriter
(LLM→SliceReq)    (Slice[])        (OutputLog bounded mem)    (files/inline)
    │                                        │
    └── ChunkedFallback (if over budget) ────┘
        (ReaderAgent per chunk → AgentMemoryStore)
```

---

## Testing Conventions

- All LLM calls are **mocked** in unit tests — no API key needed for `pytest tests/ -v`.
- Integration tests are gated by `TPCA_RUN_INTEGRATION=1` env var.
- Test fixtures live in `tests/fixtures/` — do not modify them without updating affected tests.
- Use `pytest-asyncio` for any async tests; mark with `@pytest.mark.asyncio`.

---

## Important Notes

- `__version__` is assigned twice in `tpca/__init__.py` (lines 7 and 84). The effective version is `"3.0.0"` (last assignment). This is a known inconsistency.
- Tree-sitter JS/TS parsers are optional — install `tree-sitter-javascript` and `tree-sitter-typescript` for Phase 3 multi-language support.
- The `.tpca_cache/` directory is auto-created at runtime; it is git-ignored.
- `ANTHROPIC_API_KEY` env var is required for Anthropic provider; Ollama needs a running Ollama server.
