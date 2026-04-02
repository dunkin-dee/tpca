# TPCA — Two-Pass Context Agent

AST-driven, graph-ranked context management for limited-window LLMs. TPCA solves the "too much code, too little context" problem and works in two modes:

- **Documentation mode** — generates documentation, summaries, or any text-based output for any codebase
- **Coding assistant mode** — an interactive session that plans, executes, and tracks code changes across a project

Both modes work against local models (Ollama) by default, with cloud (Anthropic) available via a flag.

---

## How It Works

### Pass 1 — Deterministic Indexing (zero LLM)

Every run starts here. TPCA parses your source files with Tree-sitter, builds a symbol relationship graph, runs task-biased PageRank, and renders a compact text index (~1–3K tokens for a 10K-line codebase). No LLM is involved; the result is fully deterministic and cached per file.

### Pass 2 — LLM Synthesis

The compact index is handed to an LLM that selects which symbols to read in detail. Token-budgeted slices of actual source are fetched and fed into a bounded synthesis loop. Working memory stays O(chunks), not O(total output size), so the context window never blows up regardless of project size.

### Coding Assistant

On top of the two passes, TPCA has a full coding-session pipeline:

1. A **PlannerAgent** breaks the task into scoped sections (one section ≈ one context window of work)
2. An **EvaluatorAgent** scores and optionally splits over-budget sections
3. **WorkerAgents** run a tool-call loop per section — reading files, writing diffs, running tests, and emitting a structured summary
4. A **SessionManager** orchestrates the whole lifecycle, saves state after every section, and propagates interface-change notifications to downstream sections

Sessions survive interruption: `.tpca_plan.json` in the project root captures the full plan state.

---

## Installation

**Requirements:** Python 3.10+, and either a running [Ollama](https://ollama.com) instance or an Anthropic API key.

```bash
git clone <repo>
cd tpca
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**JS/TS support** (optional — Python is always available):
```bash
pip install tree-sitter-javascript tree-sitter-typescript
```

---

## LLM Setup

### Ollama (default, local)

```bash
ollama pull qwen2.5-coder:14b   # recommended for coding tasks
# TPCA defaults to Ollama on http://localhost:11434/v1
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Use the `--preset cloud` flag or set `provider="anthropic"` in config to switch.

---

## Quick Start

### Interactive REPL

```bash
tpca repl                       # Ollama default
tpca repl --preset cloud        # Anthropic Claude
tpca repl --preset 13b-local    # explicit 13B local config
```

The REPL is the primary interface. Commands:

```
File browsing:
  ls [path]          List directory contents
  tree [path]        Directory tree (depth 3)
  cat <file>         Print file contents (max 500 lines)
  pwd / cd <path>    Navigate (sandboxed to startup directory)

Documentation:
  run <task>         Run full pipeline on current directory
  index [path]       Run Pass 1 only and show compact index
  stats              Show stats from the last operation

Coding sessions:
  plan new <task>    Start a new coding session (plans, does not execute)
  plan               Show tree view of current plan
  plan clear         Delete the current plan
  continue           Resume execution from last saved state
  retry <id>         Re-run a blocked or needs-revision section
  eval <id>          Re-evaluate a specific section
  summary            Compact table of all section summaries
  diff <id>          Show git diff for files changed by a section
  tools              List all available worker tools

Other:
  watch              File watcher status
  config             Show current configuration
  set <key> <value>  Change a config value live
  !<command>         Shell passthrough (e.g. !git status)
```

### Documentation — CLI

```bash
# Document all public methods in the current directory
tpca run "Document every public method with parameters and return types."

# Index only (no LLM, useful for inspection)
tpca index ./src

# Mirror mode: outputs to docs/ mirroring the source tree
tpca run "Summarise each module" --output-mode mirror --output-dir ./docs

# Resume an interrupted run
tpca run "..." --resume .tpca_cache/manifest.json
```

### Coding Assistant — Library

```python
from tpca import TPCAOrchestrator, TPCAConfig

config = TPCAConfig.from_preset("13b-local")   # or "cloud", "7b-local"
orchestrator = TPCAOrchestrator(config=config)

result = orchestrator.run_coding_session(
    source="./my_project/src",
    task="Add input validation to every public API endpoint.",
)

print(result["stats"])
# {'sections_total': 4, 'sections_complete': 4, 'total_time_ms': 41200, ...}

# Resume after an interruption
result = orchestrator.run_coding_session(
    source="./my_project/src",
    task="Add input validation to every public API endpoint.",
    resume=True,
)
```

### Documentation — Library

```python
from tpca import TPCAOrchestrator, TPCAConfig

config = TPCAConfig(
    provider="anthropic",
    synthesis_model="claude-sonnet-4-6",
    output_mode="mirror",
    output_dir="./docs",
)
orchestrator = TPCAOrchestrator(config=config)
result = orchestrator.run(
    source="./my_project/src",
    task="Document every public method with parameters and return types.",
)
print(result["stats"])  # compression_ratio, llm_calls, total_time_ms, ...
```

---

## Configuration

All configuration lives in `TPCAConfig`. The easiest entry point is a named preset:

```python
from tpca import TPCAConfig

config = TPCAConfig.from_preset("13b-local")   # Ollama, qwen2.5-coder:14b, 16K context
config = TPCAConfig.from_preset("7b-local")    # Ollama, qwen2.5-coder:7b, 4K context
config = TPCAConfig.from_preset("cloud")       # Anthropic, claude-sonnet-4-6, 32K context
```

Full reference:

```python
TPCAConfig(
    # Languages
    languages=["python"],                       # "python" | "javascript" | "typescript"

    # LLM provider
    provider="ollama",                          # "ollama" | "anthropic"
    reader_model="claude-haiku-4-5-20251001",  # lightweight: planning and extraction
    synthesis_model="claude-sonnet-4-6",       # powerful: synthesis output
    ollama_base_url="http://localhost:11434/v1",
    ollama_reader_model="qwen2.5-coder:14b",
    ollama_synthesis_model="qwen2.5-coder:14b",

    # Token budget
    model_context_window=16384,
    context_budget_pct=0.75,
    max_tool_rounds=20,                        # max LLM tool-call rounds per worker

    # Indexing
    top_n_symbols=50,
    pagerank_alpha=0.85,
    cache_enabled=True,
    cache_dir=".tpca_cache",
    exclude_patterns=["__pycache__", ".git", "node_modules", "venv"],

    # Output (documentation mode)
    output_mode="inline",                      # inline | single_file | mirror | per_symbol
    output_dir="./tpca_output",
    max_synthesis_iterations=20,

    # Coding sessions
    fallback_chunk_tokens=3500,               # target tokens per plan section
    fallback_overlap_tokens=150,
    parallel_workers=False,                   # True: file-independent sections run concurrently

    # Resume (documentation mode)
    resume_manifest=None,                     # path to prior manifest.json

    # Logging
    log=LogConfig(
        log_file=".tpca_cache/tpca.log",
        console_level="WARN",                 # DEBUG | INFO | WARN | ERROR
    ),
)
```

---

## Coding Session Details

### Plan lifecycle

A session moves through these states:

```
PLANNING → EVALUATING → EXECUTING → COMPLETE
```

Each section in the plan follows:

```
PENDING → IN_PROGRESS → COMPLETE
                      ↘ NEEDS_REVISION  (evaluator flagged it, or an interface changed)
                      ↘ BLOCKED         (worker raised an exception)
```

### Worker tools

Each worker agent has access to 10 tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file content, optionally line-ranged |
| `write_file` | Write or overwrite a file |
| `apply_diff` | Apply a unified diff (preferred over write_file) |
| `patch_file` | Exact old→new text replacement (fallback for models that can't produce diffs) |
| `list_dir` | List files in a directory |
| `grep_symbol` | Find a symbol definition in the AST index |
| `query_graph` | Get callers/callees of a symbol |
| `run_shell` | Run an allowlisted shell command |
| `run_tests` | Run tests scoped to specified files |
| `write_summary` | Emit a structured work summary (required before finishing) |

`run_shell` and `run_tests` are restricted to the project root and a command allowlist (`pytest`, `python -m pytest`, `git status`, `git diff`, `npm test`, `cargo test`). No destructive shell commands are permitted.

After every file-mutating tool call (`write_file`, `apply_diff`, `patch_file`), TPCA automatically runs the test suite on the changed files and appends the result to the tool output, giving the model an immediate self-correction signal.

### Plan persistence

The active plan is saved to `.tpca_plan.json` in the project root after every section completes. Add it to `.gitignore` if you don't want to commit session state.

---

## Project Structure

```
tpca/
├── config.py                   # TPCAConfig — all settings in one dataclass
├── orchestrator.py             # TPCAOrchestrator — top-level entry point
├── session_manager.py          # SessionManager — coding session lifecycle
│
├── pass1/                      # Deterministic indexing
│   ├── ast_indexer.py          # Tree-sitter multi-language parser
│   ├── graph_builder.py        # Symbol relationship graph (NetworkX)
│   ├── graph_ranker.py         # Task-biased PageRank
│   ├── index_renderer.py       # Compact text index renderer
│   └── queries/                # Tree-sitter S-expression queries
│       ├── python.scm
│       ├── javascript.scm
│       └── typescript.scm
│
├── pass2/                      # LLM synthesis
│   ├── context_planner.py      # LLM selects symbols to read
│   ├── slice_fetcher.py        # Token-budgeted source retrieval
│   ├── output_chunker.py       # Synthesis loop with bounded OutputLog
│   ├── synthesis_agent.py      # Orchestrates full synthesis
│   └── output_writer.py        # Writes output + manifest
│
├── plan/                       # Coding session planning
│   ├── plan_model.py           # PlanSection, SessionPlan, WorkerSummary
│   ├── plan_store.py           # Atomic JSON persistence (.tpca_plan.json)
│   ├── planner_agent.py        # LLM-driven plan generation
│   ├── sub_planner_agent.py    # Splits over-budget sections (max depth 3)
│   └── evaluator_agent.py      # Scores sections and worker output
│
├── workers/                    # Coding session execution
│   ├── worker_agent.py         # WorkerAgent — tool-call loop per section
│   ├── worker_context.py       # WorkerContextBuilder — bounded context assembly
│   └── templates.py            # Per-task-type system prompt templates
│
├── tools/                      # Worker tool system
│   ├── registry.py             # Tool definitions and JSON schemas
│   ├── executor.py             # ToolExecutor — executes and sandboxes tools
│   └── specs.py                # Compact tool description text
│
├── llm/
│   └── client.py               # LLMClient — Anthropic + Ollama, native tool calls
│
├── fallback/                   # Over-budget subgraph fallback
│   ├── chunked_pipeline.py     # Partition subgraph into overlapping chunks
│   ├── reader_agent.py         # Lightweight reader model per chunk
│   └── memory_store.py         # Aggregate extractions → compact context
│
├── cache/
│   └── index_cache.py          # Per-file symbol cache with hash invalidation
│
├── watch/
│   └── file_watcher.py         # watchdog-based background file watcher
│
└── cli/
    └── main.py                 # Click CLI + prompt_toolkit REPL

tests/                          # 300+ tests — all LLM calls mocked
```

---

## Architecture Notes

### Bounded context

`OutputLog` keeps working memory O(chunks), not O(total output size). Each synthesis call sees only the compact index + current source slices + prior log entries — never all prior output.

### Token accuracy

All budget enforcement uses `tiktoken` (cl100k_base) with a 4-char/token fallback. Character counts are never used for budget decisions.

### Two model roles

- **Reader model** (default: `qwen2.5-coder:14b` / `claude-haiku-4-5-20251001`): lightweight, used for context planning, section evaluation, and fallback chunk reading
- **Synthesis model** (default: same / `claude-sonnet-4-6`): used for actual synthesis output and worker tool loops

### Graceful degradation

- No LLM available → Pass 1 runs fully; Pass 2 skipped with a warning
- JS/TS parsers absent → those files silently skipped; Python unaffected
- LLM returns bad symbol IDs → retried with suggestions, falls back to top CORE by PageRank
- Worker doesn't call `write_summary` → PARTIAL summary synthesised from tool call history

---

## Testing

```bash
# All unit tests (no API key needed — all LLM calls mocked)
pytest tests/ -v

# Run specific test modules
pytest tests/test_phase1.py tests/test_repl.py -v

# Skip JS/TS tests if parsers not installed
pytest tests/ -v -k "not JavaScript and not TypeScript"

# Integration tests (requires API key or Ollama)
TPCA_RUN_INTEGRATION=1 pytest tests/ -v -m integration
```

---

## Performance

**Pass 1 (indexing):**
- ~50–200 files/second (Tree-sitter is fast at the C level)
- Graph build + PageRank: ~100ms for 1,000 symbols
- Under 5 seconds for a 50K-line codebase

**Pass 2 (documentation synthesis):**
- ~2 + N LLM calls (where N = symbols in scope)
- Typical compression ratio: 10–20× (raw source tokens to index tokens)

**Coding sessions:**
- One planning call + one evaluation call per section
- One tool-loop call per section (multiple rounds within the loop)
- Parallel dispatch available for file-independent sections (`parallel_workers=True`)
