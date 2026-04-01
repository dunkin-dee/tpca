# tests/ — Test Suite

## Running Tests

```bash
# All unit tests (no API key needed — all LLM calls mocked)
pytest tests/ -v

# Specific component
pytest tests/test_phase1.py -v
pytest tests/test_fallback.py -v

# Integration tests (requires ANTHROPIC_API_KEY)
TPCA_RUN_INTEGRATION=1 pytest tests/ -m integration -v
```

## Test Files

| File | What It Tests |
|------|--------------|
| `test_phase1.py` | ASTIndexer, GraphBuilder, GraphRanker, IndexRenderer, IndexCache |
| `test_context_planner.py` | ContextPlanner: LLM planning, validation, retry on invalid symbol IDs |
| `test_slice_fetcher.py` | SliceFetcher: token budget enforcement, primary vs supporting symbol handling |
| `test_synthesis_agent.py` | SynthesisAgent: full synthesis loop, OutputLog management |
| `test_output_chunker.py` | OutputChunker: topological ordering, OutputLog eviction |
| `test_llm_client.py` | LLMClient: provider abstraction, TokenCounter accuracy |
| `test_fallback.py` | ChunkedFallback: partitioning, ReaderAgent, AgentMemoryStore |
| `test_multi_language.py` | ASTIndexer with JavaScript/TypeScript sources (Phase 3) |
| `test_resume.py` | OutputManifest-based resume: skip completed files, resume partial |

## Fixtures

```
tests/fixtures/
├── sample_codebase/        # Python fixtures used by most tests
│   ├── auth.py             # Auth class with JWT token handling
│   ├── router.py           # HTTP router with route registration
│   └── utils.py            # Utility functions
└── sample_js_codebase/     # JavaScript fixtures for multi-language tests (Phase 3)
```

**Do not modify fixture files** without updating the tests that depend on their symbol names and line counts.

## Conventions

- **Mock LLM calls** at the `LLMClient.complete()` level. Tests should not hit real APIs.
- **Integration tests** are marked `@pytest.mark.integration` and gated by `TPCA_RUN_INTEGRATION=1`.
- Use `@pytest.mark.asyncio` for any async test functions.
- Token counts in tests use the same `TokenCounter` as production code — do not hardcode character-based estimates.
- When adding a new Pass 1 feature, add tests to `test_phase1.py`. For Pass 2, add to the appropriate `test_*.py` file.
