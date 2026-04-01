# tpca/pass2 — LLM-Driven Synthesis

Pass 2 takes the compact index from Pass 1 and uses an LLM to plan, fetch, and synthesize documentation or analysis for the codebase.

## Pipeline

```
Compact index (from Pass 1)
    ↓
ContextPlanner   — LLM call #1: "Which symbols do you need?" → SliceRequest
    ↓
SliceFetcher     — Fetch exact source lines for requested symbols (token-budgeted)
    ↓
SynthesisAgent   — Loop: for each symbol in topological order:
                       LLM call: index + slices + OutputLog → synthesis output
                       Update OutputLog (bounded working memory)
    ↓
OutputWriter     — Route output to files (inline/single_file/mirror/per_symbol)
```

## Components

### `context_planner.py` — ContextPlanner
- Sends the compact index + user task to the **reader model** (lightweight).
- LLM responds with a JSON `SliceRequest`: lists of primary and supporting symbol IDs plus rationale.
- Validates that all returned symbol IDs exist in the index; retries with error feedback if invalid IDs are returned (up to a configurable limit).
- Output feeds directly into `SliceFetcher`.

### `slice_fetcher.py` — SliceFetcher
- Given a `SliceRequest`, reads exact source lines for each symbol from disk.
- Enforces a tiktoken-accurate token budget (`context_budget_pct * model_context_window`).
- **Primary symbols** are always included; truncated to signature + docstring if needed.
- **Supporting symbols** are added greedily (largest-rank-first) until budget is exhausted.
- Returns `Slice[]` objects with `symbol_id`, `source`, `token_count`, `truncated` flag.

### `output_chunker.py` — OutputChunker
- Determines topological processing order for symbols (dependencies processed first).
- Manages the `OutputLog` — a bounded list of `OutputChunk` entries (completed/partial/failed).
- The OutputLog is the "working memory" passed between synthesis LLM calls; it grows O(chunks), not O(total output).
- Tracks per-symbol status for resume capability.

### `synthesis_agent.py` — SynthesisAgent
- Orchestrates the full synthesis loop using `OutputChunker`.
- Each iteration: compact index + relevant slices + prior `OutputLog` → LLM call (synthesis model) → new `OutputChunk`.
- Returns `SynthesisResult` with the full output dict, stats (LLM calls, tokens, compression ratio), and final `OutputLog`.
- If `ChunkedFallback` is enabled and a symbol's subgraph exceeds budget, delegates to the fallback pipeline.

### `output_writer.py` — OutputWriter
- Routes synthesis output to the correct files based on `output_mode`:
  - `inline`: returns dict `{file_path: content}` in memory
  - `single_file`: appends all output to one file
  - `mirror`: mirrors source structure under `output_dir` (e.g., `src/auth.py` → `docs/auth.md`)
  - `per_symbol`: one file per symbol
- Updates `OutputManifest` after each file is written (enables resume).

## Key Constraints

- **Bounded working memory**: `OutputLog` must never grow unbounded. If adding all prior outputs would overflow the context window, older entries are evicted (oldest first).
- **Token accuracy**: always use `TokenCounter` (tiktoken) for budget calculations, not character counts.
- **Reader vs. synthesis model**: `ContextPlanner` and `ReaderAgent` use the cheaper reader model; `SynthesisAgent` uses the more capable synthesis model.
- **Do not change topological order** in `OutputChunker` without understanding that downstream synthesis calls depend on prior OutputLog entries from dependencies.
