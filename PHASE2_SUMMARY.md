# TPCA Phase 2 Implementation Summary

## Overview

Phase 2 adds the LLM-driven synthesis pipeline on top of the Phase 1 foundation. The system runs end-to-end: source files -> compact index -> targeted symbol selection -> source slice retrieval -> chunked synthesis -> output.

**Goal**: End-to-end pipeline for tasks that fit in a single or multi-call synthesis session, with bounded context regardless of output size.

## Components Implemented

### LLMClient (`tpca/llm/client.py`)
- Provider-agnostic wrapper: Anthropic Claude or Ollama (via OpenAI-compatible API)
- `TokenCounter`: tiktoken-accurate counting (cl100k_base) with 4-char/token fallback
- Exponential backoff retry logic
- Structured logging of all API calls (prompt tokens, output tokens, latency)

### ContextPlanner (`tpca/pass2/context_planner.py`)
- Sends compact Pass 1 index + task to the reader model
- Parses JSON `SliceRequest` from LLM response (strips markdown fences)
- Validates all symbol IDs against the actual SymbolGraph
- Retries with edit-distance suggestions for unknown symbol IDs
- Falls back to top CORE symbols by PageRank after `max_planner_retries`

### SliceFetcher (`tpca/pass2/slice_fetcher.py`)
- Reads exact source lines for each requested symbol using file paths from Symbol
- tiktoken-accurate budget enforcement; respects `context_budget_pct * model_context_window`
- Primary symbols always included (truncated to signature-only if over budget)
- Supporting symbols included greedily while budget permits

### OutputChunker (`tpca/pass2/output_chunker.py`)
- Derives processing order from SymbolGraph topological sort; falls back to PageRank order on cycles
- Maintains `OutputLog` as bounded working memory (~50-100 tokens/entry)
- Each `ChunkPlan` carries only the OutputLog, not full prior output — context stays O(chunks)
- Already-completed symbols (from prior OutputLog) are skipped at `get_next_chunk()`

### OutputWriter (`tpca/pass2/output_writer.py`)
- Four output modes: `inline`, `single_file`, `mirror`, `per_symbol`
- Builds `OutputManifest` with per-file status tracking

### SynthesisAgent (`tpca/pass2/synthesis_agent.py`)
- Drives the full synthesis loop: plan -> fetch -> chunk -> synthesize -> write
- Bounded by `max_synthesis_iterations` (default 20)
- Returns `SynthesisResult` with output dict, `OutputLog`, manifest, and stats

### TPCAOrchestrator (`tpca/orchestrator.py`)
- Single entry point wiring all Phase 1 and Phase 2 components
- `run(source, task)`: full two-pass pipeline
- `run_pass1_only(source)`: indexing only, zero LLM
- Gracefully skips Pass 2 and warns when LLM is unavailable

### New Data Models

- **`Slice` / `SliceRequest`** (`tpca/models/slice.py`): fetched code block with token count and truncation flag; structured planner output.
- **`OutputChunk` / `OutputLog` / `OutputManifest`** (`tpca/models/output.py`): bounded working memory with `render_compact()` for prompts; completion record with per-file entries.
- **`ChunkPlan`** (`tpca/models/chunk_plan.py`): work order for one synthesis call.

## New Files

```
tpca/
├── orchestrator.py
├── llm/
│   └── client.py
├── pass2/
│   ├── context_planner.py
│   ├── slice_fetcher.py
│   ├── output_chunker.py
│   ├── output_writer.py
│   └── synthesis_agent.py
└── models/
    ├── slice.py
    ├── output.py
    └── chunk_plan.py

tests/
├── test_llm_client.py
├── test_context_planner.py
├── test_slice_fetcher.py
├── test_output_chunker.py
└── test_synthesis_agent.py

demo_phase2.py
```

## Usage

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python demo_phase2.py

# Ollama
ollama pull qwen2.5-coder:14B
python demo_phase2.py

# All tests (LLM calls mocked — no API key needed)
pytest tests/ -v
```

```python
from tpca import TPCAOrchestrator, TPCAConfig, LogConfig

config = TPCAConfig(
    synthesis_model='claude-sonnet-4-6',
    reader_model='claude-haiku-4-5-20251001',
    output_mode='inline',
    output_dir='./docs',
    log=LogConfig(console_level='INFO'),
)

result = TPCAOrchestrator(config=config).run(
    source='./my_project/src',
    task='Document every public method with parameters and return types.',
)

print(result['stats'])
# {'pass1_time_ms': 187, 'llm_calls': 4, 'compression_ratio': 12.5, ...}
```

## Key Design Properties

### Bounded Context
`OutputChunker` passes `OutputLog.render_compact()` (~50-100 tokens/entry) to each subsequent synthesis call, not the full prior output. Context size stays O(chunks), not O(output_size).

### Token Accuracy
All budget management uses tiktoken (cl100k_base). The 4-chars/token approximation is used only as a fallback when tiktoken is unavailable.

### Graceful Degradation
- No LLM available: Pass 1 runs fully; Pass 2 is skipped with a warning.
- LLM returns unknown symbol IDs: ContextPlanner retries with suggestions; falls back to top CORE symbols.
- Source file missing: SliceFetcher returns a signature-only slice.

## Performance

- Pass 1: unchanged (<5 seconds for 50K-line codebase)
- ContextPlanner: 1-3 reader-model calls
- SliceFetcher: disk I/O only, <100ms for typical slice sets
- Synthesis loop: 1 synthesis-model call per top-level symbol
- Total LLM calls: approximately 2 + N (N = top-level symbols in scope)

## Design Compliance

All Phase 2 requirements from Section 13 of the design document are met:

- LLMClient: tiktoken token counting, provider-agnostic, retry logic
- ContextPlanner: planning prompt, JSON validation, symbol validation with retry
- SliceFetcher: tiktoken-accurate budget management, truncation logic
- OutputChunker: logical boundary detection, OutputLog, synthesis loop
- OutputWriter: all four modes (inline, single_file, mirror, per_symbol)
- SynthesisAgent: assembles context package, runs chunker loop, returns SynthesisResult
- TPCAOrchestrator: wires all components, handles LLM unavailability gracefully

## Limitations Addressed in Phase 3

The following were intentional deferrals from Phase 2, resolved in Phase 3:

- **Python only**: JS/TS support added via Tree-sitter queries
- **No ChunkedFallback**: fallback pipeline added for over-budget subgraphs
- **No manifest resume**: `OutputManifest.save/load` and `OutputLog.from_manifest` added
- **mirror/per_symbol modes**: fully implemented with path safety and manifest writes
