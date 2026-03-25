# TPCA - Two-Pass Context Agent

**Two-Pass Context Agent (TPCA)** - Complete Implementation

AST-driven, graph-ranked context retrieval for limited-window LLMs. TPCA implements a complete two-pass pipeline: Pass 1 (zero-LLM indexing) + Pass 2 (LLM-driven synthesis).

## 🎯 What This Does

TPCA analyzes Python codebases and generates comprehensive documentation or other outputs using a two-pass approach:

**Pass 1 (Deterministic, Zero LLM)**: Creates a compact, ranked index of all code symbols without using any LLM calls.

**Pass 2 (LLM-Driven Synthesis)**: Uses the compact index to guide an LLM in generating detailed outputs, with bounded context regardless of output size.

**Key Features:**
- ✅ Multi-file Python parsing with Tree-sitter AST
- ✅ Cross-file symbol relationship graph
- ✅ Task-biased PageRank for symbol importance
- ✅ Compact text index generation (1,000-3,000 tokens for 10K lines)
- ✅ Provider-agnostic LLM client (Anthropic Claude or Ollama)
- ✅ tiktoken-accurate token counting and budget management
- ✅ Structured logging with file, console, and ring buffer
- ✅ Per-file caching with automatic invalidation
- ✅ Multiple output modes (inline, single_file, mirror, per_symbol)
- ✅ Resumable synthesis with OutputLog working memory

## 📦 Installation

### Requirements
- Python 3.10+
- pip
- **LLM Provider**: Either Anthropic API key OR Ollama running locally

### Install Dependencies

```bash
pip install tree-sitter-python networkx anthropic tiktoken openai
```

### LLM Setup

**Option 1: Anthropic Claude (Cloud)**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option 2: Ollama (Local)**
```bash
# Install Ollama from https://ollama.ai/
ollama pull qwen2.5-coder:14B  # or another model
# TPCA will auto-detect and use Ollama if no ANTHROPIC_API_KEY
```

## 🚀 Quick Start

### Run Phase 1 Demo (Indexing Only)

```bash
python demo_phase1.py
```

This analyzes the sample codebase and creates a compact index without any LLM calls.

### Run Phase 2 Demo (Full Pipeline)

```bash
# With Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-... python demo_phase2.py

# With Ollama (default if no API key)
python demo_phase2.py
```

This runs the complete two-pass pipeline: indexing → LLM planning → synthesis → output.

### Use as a Library

```python
from tpca import TPCAOrchestrator, TPCAConfig

# Configure for Ollama
config = TPCAConfig(
    provider="ollama",
    ollama_reader_model="qwen2.5-coder:14B",
    ollama_synthesis_model="qwen2.5-coder:14B",
    output_mode="inline",
)

# Or for Anthropic
config = TPCAConfig(
    provider="anthropic",
    reader_model="claude-haiku-4-5-20251001",
    synthesis_model="claude-sonnet-4-6",
    output_mode="single_file",
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

## 📁 Project Structure

```
tpca/
├── __init__.py                 # Main exports (all phases)
├── config.py                   # TPCAConfig (unified for both phases)
├── orchestrator.py             # TPCAOrchestrator (top-level entry point)
├── logging/                    # Structured logging system
│   ├── structured_logger.py    # Main logger with file/console/buffer
│   ├── console_handler.py      # Human-readable console output
│   └── log_config.py           # LogConfig dataclass
├── models/                     # Core data models
│   ├── symbol.py              # Symbol, SymbolGraph, PendingEdge
│   ├── slice.py               # Slice, SliceRequest
│   ├── output.py              # OutputLog, OutputChunk, OutputManifest
│   └── chunk_plan.py          # ChunkPlan
├── pass1/                      # Pass 1 components (no LLM)
│   ├── ast_indexer.py         # Multi-file AST parsing
│   ├── graph_builder.py       # Symbol graph construction
│   ├── graph_ranker.py        # PageRank-based ranking
│   ├── index_renderer.py      # Compact index generation
│   └── queries/
│       └── python.scm         # Tree-sitter query for Python
├── pass2/                      # Pass 2 components (LLM-driven)
│   ├── context_planner.py     # LLM planning with validation/retry
│   ├── slice_fetcher.py       # Token-budgeted source slice retrieval
│   ├── output_chunker.py      # Logical boundary detection, OutputLog
│   ├── output_writer.py       # Multi-mode output (inline/mirror/etc.)
│   └── synthesis_agent.py     # Synthesis loop orchestration
├── llm/                        # LLM client abstraction
│   └── client.py               # Provider-agnostic client (Anthropic/Ollama)
└── cache/                      # Caching system
    └── index_cache.py         # Per-file symbol cache
```

## 🔧 Configuration

### TPCAConfig Options

```python
config = TPCAConfig(
    # Pass 1 - AST Indexing
    languages=['python'],           # Supported: python (more in Phase 3)
    exclude_patterns=[              # Paths to skip
        '__pycache__', '.git', 
        'node_modules', 'dist', '.venv'
    ],
    cache_dir='.tpca_cache',       # Cache location
    cache_enabled=True,
    
    # Graph Ranking
    pagerank_alpha=0.85,           # PageRank damping factor
    top_n_symbols=50,              # Number of top symbols to track
    
    # LLM Configuration
    provider='ollama',             # 'anthropic' or 'ollama'
    reader_model='claude-haiku-4-5-20251001',  # Lightweight planning model
    synthesis_model='claude-sonnet-4-6',       # Powerful synthesis model
    
    # Ollama-specific (when provider='ollama')
    ollama_base_url='http://localhost:11434/v1',
    ollama_reader_model='qwen2.5-coder:14B',
    ollama_synthesis_model='qwen2.5-coder:14B',
    
    # Token Management
    model_context_window=8192,     # Context window size
    context_budget_pct=0.70,       # Fraction of context for slices
    max_planner_retries=3,         # Retry on invalid symbol IDs
    
    # Output Configuration
    output_mode='inline',          # 'inline', 'single_file', 'mirror', 'per_symbol'
    output_dir='./tpca_output',    # Output directory
    max_synthesis_iterations=20,   # Max synthesis loop iterations
    
    # Logging
    log=LogConfig(
        log_file='.tpca_cache/tpca.log',
        console_level='INFO',       # DEBUG|INFO|WARN|ERROR
        file_level='DEBUG',
        ring_buffer_size=1000
    )
)
```

## 📊 Output Example

```
## auth.py
class Auth(BaseAuth)                           [CORE]
  + __init__(self, config: dict) -> None
  + validate_token(self, token: str) -> bool   # Validates a JWT token.
  + refresh_token(self, token: str) -> str     [SUPPORT]
  - _decode_payload(self, token: str) -> dict  [PERIPHERAL]

## router.py
class Router                                   [CORE]
  + route(self, request: Request) -> Response
  + register(self, path: str, handler: Callable) -> None
  - _validate_auth(self, token: str) -> bool   [SUPPORT]

## Cross-file references
router.py → auth.py::Auth.validate_token
auth.py → utils.py::hash_password
```

## 🏗️ Architecture

### Data Flow

```
Source Files (directory or list)
    │
    ▼  PASS 1 — DETERMINISTIC INDEXING (NO LLM)
ASTIndexer → GraphBuilder → GraphRanker → IndexRenderer
    │           │              │              │
    Symbol[]    DiGraph        Ranked Graph   Compact Index (~1-3K tokens)
    │
    ▼  PASS 2 — LLM-DRIVEN SYNTHESIS
ContextPlanner → SliceFetcher → SynthesisAgent → OutputWriter
    │              │              │              │
SliceRequest    Source Slices   Output Chunks   Final Output (any size)
```

### Pass 1 Components

1. **ASTIndexer**: Parses Python files using Tree-sitter, extracts symbols with signatures and docstrings
2. **GraphBuilder**: Creates cross-file symbol relationship graph
3. **GraphRanker**: Applies task-biased PageRank to rank symbols by importance
4. **IndexRenderer**: Generates compact text index for LLM consumption
5. **IndexCache**: Per-file caching with automatic invalidation

### Pass 2 Components

6. **LLMClient**: Provider-agnostic client (Anthropic Claude or Ollama) with tiktoken counting
7. **ContextPlanner**: Shows compact index to LLM, gets structured SliceRequest with validation/retry
8. **SliceFetcher**: Retrieves exact source code slices with token budget enforcement
9. **OutputChunker**: Manages synthesis loop with OutputLog working memory (bounded context)
10. **SynthesisAgent**: Orchestrates the full synthesis pipeline
11. **OutputWriter**: Supports multiple output modes (inline, single_file, mirror, per_symbol)
12. **TPCAOrchestrator**: Top-level coordinator wiring all components

### Key Design Properties

- **Bounded Context**: OutputLog keeps context O(chunks) not O(output_size)
- **Token Accuracy**: All budget management uses tiktoken (cl100k_base)
- **Resumability**: Synthesis can restart from partial OutputLog
- **Graceful Degradation**: Works with Pass 1 only if LLM unavailable
- **Provider Agnostic**: Supports both Anthropic and Ollama seamlessly

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

All tests are mocked and don't require API keys or Ollama running.

### Test Files

The `tests/fixtures/sample_codebase/` contains a multi-file Python project:
- `auth.py` - Authentication with JWT validation
- `router.py` - HTTP routing with auth integration  
- `utils.py` - Utility functions

### Integration Tests

```bash
# Requires ANTHROPIC_API_KEY or Ollama running
TPCA_RUN_INTEGRATION=1 pytest tests/ -v -m integration
```

## 📈 Performance

### Pass 1 (Deterministic Indexing)
- **Indexing**: ~50-200 files/second (Python)
- **Graph Building**: Linear in number of symbols  
- **PageRank**: ~100ms for 1,000 symbols
- **Rendering**: ~10ms for typical index
- **Total**: <5 seconds for a 50K-line codebase

### Pass 2 (LLM Synthesis)
- **Context Planning**: 1-3 LLM calls (reader model, lightweight)
- **Slice Fetching**: Disk I/O only, <100ms for typical slices
- **Synthesis Loop**: 1 LLM call per symbol (synthesis model)
- **Total LLM Calls**: ~2 + N (where N = symbols in scope)
- **Context Efficiency**: 10-20x compression ratio typical

## 🔮 Future Phases

### Phase 3 (Planned)
- Multi-language support (JavaScript, TypeScript)
- Chunked fallback pipeline for large outputs
- Enhanced call graph analysis
- Output resume from partial completion

## 📝 Design Document

See `TPCA_Design_v2.docx` for the complete technical specification.

## 🤝 Contributing

This is Phase 1 of a multi-phase implementation. Key areas for contribution:
- Additional Tree-sitter language support
- Enhanced call graph analysis
- Performance optimizations
- Test coverage

## 📄 License

MIT License - See LICENSE file for details

## 🔗 References

- Design Document: `TPCA_Design_v2.docx`
- Tree-sitter: https://tree-sitter.github.io/
- NetworkX: https://networkx.org/

---

**Status**: Phase 1 Complete ✅ | Phase 2 Complete ✅ | Phase 3 Planned 🚧
