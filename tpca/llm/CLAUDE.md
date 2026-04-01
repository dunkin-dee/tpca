# tpca/llm — LLM Abstraction Layer

Provider-agnostic LLM client used by all components that make LLM calls.

## Components

### `client.py` — LLMClient & TokenCounter

**LLMClient**
- Supports two providers:
  - `anthropic`: uses the `anthropic` SDK; requires `ANTHROPIC_API_KEY` env var.
  - `ollama`: uses the `openai` SDK pointed at a local Ollama server (OpenAI-compatible API).
- Provider is selected via `TPCAConfig.provider`.
- Handles retries and surfaces provider errors with clear messages.
- All LLM calls are synchronous; async wrappers exist for test compatibility.

**TokenCounter**
- Uses `tiktoken` with `cl100k_base` encoding.
- Fallback: `len(text) // 4` characters-per-token estimate when tiktoken fails.
- Used by `SliceFetcher`, `OutputChunker`, and `AgentMemoryStore` for budget enforcement.
- **Always use `TokenCounter` for budget math** — never raw character counts.

## Provider Selection

```python
# Anthropic (default)
config = TPCAConfig(provider='anthropic', synthesis_model='claude-sonnet-4-6')

# Ollama (local)
config = TPCAConfig(provider='ollama', synthesis_model='llama3.2')
# Requires Ollama running at localhost:11434
```

## Model Roles

| Role | Default Model | Used By |
|------|--------------|---------|
| Reader | `claude-haiku-4-5-20251001` | ContextPlanner, ReaderAgent |
| Synthesis | `claude-sonnet-4-6` | SynthesisAgent |

The reader model handles lightweight tasks (planning, extraction) to minimize cost. The synthesis model handles the actual documentation/analysis generation.

## Notes

- When mocking in tests, mock `LLMClient.complete()` (the core method) rather than provider-specific internals.
- Token counts returned by `TokenCounter` are estimates for Ollama models (uses cl100k_base regardless of actual model tokenizer).
