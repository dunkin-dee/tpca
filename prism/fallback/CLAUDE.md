# prism/fallback — ChunkedFallback Pipeline (Phase 3)

The fallback pipeline activates when a relevant symbol subgraph exceeds the available context budget. It uses a divide-and-conquer approach: split the subgraph into small chunks, process each with a lightweight reader model, then aggregate into a compact memory that replaces raw slices in the synthesis step.

## When It Activates

`SynthesisAgent` checks whether the token count of the requested slices exceeds `context_budget_pct * model_context_window`. If it does and `fallback_enabled=True`, it calls `ChunkedFallback` instead of proceeding with the over-budget context.

## Pipeline

```
Over-budget symbol subgraph
    ↓
ChunkedFallback.partition()
    — Splits ONLY the relevant subgraph (not full codebase) into chunks
    — Each chunk: ~fallback_chunk_tokens tokens with fallback_overlap_tokens overlap
    ↓
ReaderAgent (one per chunk, ephemeral)
    — Lightweight reader model processes each chunk
    — Extracts key information relevant to the synthesis task
    — Returns structured extraction dict
    ↓
AgentMemoryStore.aggregate()
    — Collects all extractions
    — render_compact() → ~500 tokens of compressed context
    ↓
SynthesisAgent receives compact memory instead of raw slices
```

## Components

### `chunked_pipeline.py` — ChunkedFallback
- `partition(slices, chunk_tokens, overlap_tokens)`: splits `Slice[]` into overlapping chunks respecting token boundaries.
- Chunks only the **relevant subgraph** selected by Pass 1, not the entire codebase.
- Default chunk size: 1800 tokens; overlap: 150 tokens.
- Returns a `ChunkPlan` describing chunk boundaries and total count.

### `reader_agent.py` — ReaderAgent
- Instantiated once per chunk; processes that chunk and is discarded.
- Uses the **reader model** (lightweight, e.g., `claude-haiku-4-5-20251001`) to minimize cost.
- System prompt focuses the model on extracting information relevant to the task rather than generating full documentation.
- Returns a structured extraction dict to `AgentMemoryStore`.

### `memory_store.py` — AgentMemoryStore
- Collects extraction dicts from all `ReaderAgent` calls.
- `render_compact()`: de-duplicates, merges, and compresses extractions to ~500 tokens.
- The compact memory replaces the `Slice[]` that would normally be passed to `SynthesisAgent`.

## Key Constraints

- Chunk boundaries should respect symbol boundaries where possible (don't split a function body across chunks unless necessary).
- Overlap is intentional — it ensures context near chunk boundaries is seen by both adjacent reader agents.
- `AgentMemoryStore` output must fit within the remaining context budget after the compact index is included.
- Reader agents are **stateless** — each sees only its chunk and the task description, not prior agents' outputs.
