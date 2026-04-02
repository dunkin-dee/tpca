"""
Model capability detection for TPCA.

Maintains a registry of known local and cloud models with their capabilities,
and provides a runtime probe fallback for unknown models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelCapabilities:
    supports_tool_calls: bool   # native function/tool calling API
    supports_json_mode: bool    # reliable structured JSON output
    context_window: int         # approximate token context window
    tier: str                   # "small" (<10B), "medium" (10-30B), "large" (>30B)


# ── Known model registry ──────────────────────────────────────────────────────
# Keyed by lowercase name prefix. Longer/more-specific prefixes take priority.
# Add entries here as new models become relevant.
_REGISTRY: list[tuple[str, ModelCapabilities]] = [
    # ── qwen2.5-coder ──────────────────────────────────────────────────────
    ("qwen2.5-coder:32b",   ModelCapabilities(True,  True,  32768, "large")),
    ("qwen2.5-coder:14B",   ModelCapabilities(True,  True,  32768, "medium")),
    ("qwen2.5-coder:7b",    ModelCapabilities(True,  True,   8192, "small")),
    ("qwen2.5-coder:3b",    ModelCapabilities(False, True,   4096, "small")),
    ("qwen2.5-coder:1.5b",  ModelCapabilities(False, False,  4096, "small")),
    # ── qwen2.5 (general) ──────────────────────────────────────────────────
    ("qwen2.5:72b",         ModelCapabilities(True,  True,  32768, "large")),
    ("qwen2.5:32b",         ModelCapabilities(True,  True,  32768, "large")),
    ("qwen2.5:14b",         ModelCapabilities(True,  True,  32768, "medium")),
    ("qwen2.5:7b",          ModelCapabilities(True,  True,   8192, "small")),
    # ── deepseek-coder-v2 ──────────────────────────────────────────────────
    ("deepseek-coder-v2:236b", ModelCapabilities(True, True, 32768, "large")),
    ("deepseek-coder-v2:16b",  ModelCapabilities(True, True, 16384, "medium")),
    # ── deepseek-r1 ────────────────────────────────────────────────────────
    ("deepseek-r1:70b",     ModelCapabilities(True,  True,  32768, "large")),
    ("deepseek-r1:32b",     ModelCapabilities(True,  True,  32768, "large")),
    ("deepseek-r1:14b",     ModelCapabilities(True,  True,  16384, "medium")),
    ("deepseek-r1:8b",      ModelCapabilities(False, True,   8192, "small")),
    # ── codestral ──────────────────────────────────────────────────────────
    ("codestral:22b",       ModelCapabilities(True,  True,  32768, "medium")),
    # ── llama3.3 ───────────────────────────────────────────────────────────
    ("llama3.3:70b",        ModelCapabilities(True,  True,  32768, "large")),
    # ── llama3.2 ───────────────────────────────────────────────────────────
    ("llama3.2:3b",         ModelCapabilities(False, True,   4096, "small")),
    ("llama3.2:1b",         ModelCapabilities(False, False,  4096, "small")),
    ("llama3.2",            ModelCapabilities(False, True,   4096, "small")),
    # ── llama3.1 ───────────────────────────────────────────────────────────
    ("llama3.1:405b",       ModelCapabilities(True,  True, 131072, "large")),
    ("llama3.1:70b",        ModelCapabilities(True,  True,  32768, "large")),
    ("llama3.1:8b",         ModelCapabilities(True,  True,   8192, "small")),
    # ── mistral ────────────────────────────────────────────────────────────
    ("mixtral:8x22b",       ModelCapabilities(True,  True,  32768, "large")),
    ("mixtral:8x7b",        ModelCapabilities(True,  True,  32768, "medium")),
    ("mistral-nemo",        ModelCapabilities(True,  True,  32768, "medium")),
    ("mistral:7b",          ModelCapabilities(False, True,   8192, "small")),
    ("mistral",             ModelCapabilities(False, True,   8192, "small")),
    # ── phi ────────────────────────────────────────────────────────────────
    ("phi4:14b",            ModelCapabilities(True,  True,  16384, "medium")),
    ("phi3.5",              ModelCapabilities(False, True,   4096, "small")),
    ("phi3",                ModelCapabilities(False, True,   4096, "small")),
    # ── gemma ──────────────────────────────────────────────────────────────
    ("gemma2:27b",          ModelCapabilities(False, True,   8192, "medium")),
    ("gemma2:9b",           ModelCapabilities(False, True,   8192, "small")),
    # ── anthropic (always full capability) ─────────────────────────────────
    ("claude-opus",         ModelCapabilities(True,  True, 200000, "large")),
    ("claude-sonnet",       ModelCapabilities(True,  True, 200000, "large")),
    ("claude-haiku",        ModelCapabilities(True,  True, 200000, "medium")),
]

# Cached probe results: model_name → bool (supports tool calls)
_probe_cache: dict[str, bool] = {}


def detect_capabilities(
    model_name: str,
    provider: str = "ollama",
    ollama_base_url: str = "http://localhost:11434/v1",
    probe: bool = False,
) -> ModelCapabilities:
    """
    Return capabilities for a model.

    Lookup order:
      1. Known registry (longest prefix match, case-insensitive)
      2. Runtime probe via Ollama API (only if probe=True and provider='ollama')
      3. Conservative defaults (no tool calls, 8K context)

    Args:
        model_name:      The model identifier (e.g. "qwen2.5-coder:14B").
        provider:        "ollama" or "anthropic".
        ollama_base_url: Base URL for Ollama OpenAI-compat API.
        probe:           If True, attempt a live tool-call test for unknown models.
    """
    if provider == "anthropic":
        return ModelCapabilities(True, True, 200000, "large")

    name_lower = model_name.lower()

    # Registry lookup — longest matching prefix wins
    best: Optional[ModelCapabilities] = None
    best_len = 0
    for pattern, caps in _REGISTRY:
        if name_lower == pattern or name_lower.startswith(pattern):
            if len(pattern) > best_len:
                best = caps
                best_len = len(pattern)

    if best is not None:
        return best

    # Runtime probe for unknown models
    if probe:
        tool_support = _probe_tool_calls(model_name, ollama_base_url)
        return ModelCapabilities(
            supports_tool_calls=tool_support,
            supports_json_mode=True,   # assume JSON works; conservative on tools
            context_window=8192,
            tier="medium",
        )

    # Conservative defaults
    return ModelCapabilities(False, True, 8192, "small")


def _probe_tool_calls(model_name: str, ollama_base_url: str) -> bool:
    """
    Send a minimal tool-call request to Ollama and check if the response
    contains a tool call block. Result is cached per model name.
    """
    if model_name in _probe_cache:
        return _probe_cache[model_name]

    try:
        import openai  # type: ignore
    except ImportError:
        return False

    try:
        client = openai.OpenAI(base_url=ollama_base_url, api_key="ollama")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "What is 2+2? Use the calculate tool."}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }],
            tool_choice="auto",
            max_tokens=64,
            timeout=10,
        )
        result = bool(
            resp.choices
            and resp.choices[0].message.tool_calls
        )
    except Exception:
        result = False

    _probe_cache[model_name] = result
    return result
