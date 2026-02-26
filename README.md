# TPCA Phase 1 - AST-Driven Context Indexing

**Two-Pass Context Agent (TPCA)** - Phase 1 Implementation

AST-driven, graph-ranked context retrieval for limited-window LLMs. Phase 1 implements the complete Pass 1 pipeline: zero-LLM, deterministic indexing and ranking of code symbols.

## 🎯 What This Does

TPCA Phase 1 analyzes Python codebases and creates a **compact, ranked index** of all code symbols (classes, functions, methods) without using any LLM calls. This index can then be used by LLMs to efficiently request only the code they need.

**Key Features:**
- ✅ Multi-file Python parsing with Tree-sitter AST
- ✅ Cross-file symbol relationship graph
- ✅ Task-biased PageRank for symbol importance
- ✅ Compact text index generation (1,000-3,000 tokens for 10K lines)
- ✅ Structured logging with file, console, and ring buffer
- ✅ Per-file caching with automatic invalidation

## 📦 Installation

### Requirements
- Python 3.10+
- pip

### Install Dependencies

```bash
pip install tree-sitter-python networkx
```

## 🚀 Quick Start

### Run the Demo

```bash
python demo_phase1.py
```

This will:
1. Parse all Python files in `tpca/tests/fixtures/sample_codebase/`
2. Build a symbol relationship graph with cross-file edges
3. Rank symbols using task-biased PageRank
4. Generate a compact text index
5. Display top-ranked symbols and statistics

### Use as a Library

```python
from tpca import (
    TPCAConfig,
    StructuredLogger,
    ASTIndexer,
    GraphBuilder,
    GraphRanker,
    IndexRenderer,
    IndexCache
)

# Configure
config = TPCAConfig(
    languages=['python'],
    cache_enabled=True
)

logger = StructuredLogger(config.log)
cache = IndexCache(config, logger)

# Initialize pipeline
indexer = ASTIndexer(config, logger, cache)
builder = GraphBuilder(config, logger)
ranker = GraphRanker(config, logger)
renderer = IndexRenderer(config, logger)

# Run Pass 1
symbols = indexer.index('./my_project/src')
graph = builder.build(symbols)
graph = ranker.rank_symbols(graph, task_keywords=['auth', 'user'])
compact_index = renderer.render(graph)

print(compact_index)
```

## 📁 Project Structure

```
tpca/
├── __init__.py                 # Main exports
├── config.py                   # TPCAConfig
├── logging/                    # Structured logging system
│   ├── structured_logger.py    # Main logger with file/console/buffer
│   ├── console_handler.py      # Human-readable console output
│   └── log_config.py           # LogConfig dataclass
├── models/                     # Core data models
│   └── symbol.py              # Symbol, SymbolGraph, PendingEdge
├── pass1/                      # Pass 1 components (no LLM)
│   ├── ast_indexer.py         # Multi-file AST parsing
│   ├── graph_builder.py       # Symbol graph construction
│   ├── graph_ranker.py        # PageRank-based ranking
│   ├── index_renderer.py      # Compact index generation
│   └── queries/
│       └── python.scm         # Tree-sitter query for Python
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
    ▼  PASS 1 — NO LLM CALLS
ASTIndexer → GraphBuilder → GraphRanker → IndexRenderer
    │           │              │              │
    Symbol[]    DiGraph        Ranked Graph   Compact Index
```

### Key Components

1. **ASTIndexer**: Parses Python files using Tree-sitter, extracts:
   - Classes (with inheritance)
   - Functions and methods (with signatures)
   - Docstrings
   - File locations

2. **GraphBuilder**: Creates a directed graph where:
   - Nodes = symbols
   - Edges = relationships (calls, inherits, member_of)
   - Resolves cross-file references

3. **GraphRanker**: Applies PageRank to identify:
   - Architecturally central symbols (many callers)
   - Task-relevant symbols (lexical matching)
   - Assigns tiers: CORE, SUPPORT, PERIPHERAL

4. **IndexRenderer**: Generates compact text (~1-3K tokens) showing:
   - Symbols grouped by file
   - Rank tier annotations
   - Cross-file relationships

5. **IndexCache**: Caches parsed symbols per-file with:
   - Automatic invalidation on file changes
   - JSON serialization
   - Fast lookup

6. **StructuredLogger**: Logs all events to:
   - Rotating JSON-lines file
   - Human-readable console
   - In-memory ring buffer (for tests)

## 🧪 Testing

### Run Tests

```bash
pytest tpca/tests/
```

### Test Files

The `tpca/tests/fixtures/sample_codebase/` contains a sample Python project:
- `auth.py` - Authentication with JWT validation
- `router.py` - HTTP routing with auth integration
- `utils.py` - Utility functions

These demonstrate cross-file relationships and various symbol types.

## 📈 Performance

Phase 1 is designed to be fast and deterministic:

- **Indexing**: ~50-200 files/second (Python)
- **Graph Building**: Linear in number of symbols
- **PageRank**: ~100ms for 1,000 symbols
- **Rendering**: ~10ms for typical index

Total: **<5 seconds** for a 50K-line codebase.

## 🔮 Future Phases

### Phase 2 (Coming Next)
- Context Planner (LLM-driven symbol selection)
- Slice Fetcher (tiktoken-accurate budget management)
- Synthesis Agent (output generation)
- Output Chunker (large output handling)

### Phase 3 (Planned)
- Multi-language support (JavaScript, TypeScript)
- Chunked fallback pipeline
- Multi-file output modes (mirror, per_symbol)
- Resume from partial completion

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

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🚧
