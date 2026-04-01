# tpca/pass1 — Deterministic AST Indexing

Pass 1 is entirely deterministic (zero LLM calls). It converts source files into a compact, ranked text index that fits comfortably in an LLM context window.

## Pipeline

```
Source files
    ↓
ASTIndexer       — Tree-sitter parse → Symbol[] (with source line ranges)
    ↓
GraphBuilder     — Symbol[] → NetworkX DiGraph (calls/inherits/member_of edges)
    ↓
GraphRanker      — DiGraph + task keywords → Task-biased PageRank → Ranked DiGraph
    ↓
IndexRenderer    — Ranked DiGraph → compact text index (~1,000–3,000 tokens)
```

## Components

### `ast_indexer.py` — ASTIndexer
- Uses Tree-sitter to parse Python (and optionally JS/TS).
- Extracts: classes, methods, functions with name, signature, docstring, line range, file path.
- Assigns stable symbol IDs: `{file_stem}:{qualified_name}` (e.g., `auth:Auth.validate_token`).
- Integrates with `IndexCache` — skips unchanged files (hash-based invalidation).
- Tree-sitter queries are in `queries/python.scm`.

### `graph_builder.py` — GraphBuilder
- Takes a flat `Symbol[]` and builds a `NetworkX DiGraph`.
- Edge types: `calls`, `inherits`, `member_of`, `external_call`.
- Cross-file relationships are tracked; call edges are weighted by call count.

### `graph_ranker.py` — GraphRanker
- Runs task-biased PageRank: personalization vector boosts symbols whose name/docstring match the task keywords.
- `pagerank_alpha` (default 0.85) controls damping factor.
- Returns top-N symbols labeled `CORE`, `SUPPORT`, or `PERIPHERAL` by rank tier.
- `top_n_symbols` (default 50) caps the ranked set passed to Pass 2.

### `index_renderer.py` — IndexRenderer
- Renders the ranked subgraph as a compact human-readable text index.
- Format: hierarchical `file → class → method` with tier labels and signatures.
- Targets ~1,000–3,000 tokens for a 10K-line codebase (10–20x compression).
- This index is the sole input to the ContextPlanner in Pass 2.

### `queries/python.scm`
- Tree-sitter S-expression query for Python symbol extraction.
- Captures: `class_definition`, `function_definition`, `decorated_definition`.
- Do not modify without verifying that `ASTIndexer` capture names still match.

## Key Constraints
- No LLM calls — must remain deterministic and fast (<5s for 50K-line codebases).
- Symbol IDs must be stable across runs for cache and resume to work correctly.
- Token counts in the index are informational; actual budgeting happens in Pass 2.
