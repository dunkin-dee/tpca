"""
Tool spec text export — compact, token-efficient description of all tools.

get_tool_spec_text() is cached on first call. Pass an executor instance to
regenerate from a live registry (e.g. after registering custom tools).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .executor import ToolExecutor

_cache: Optional[str] = None


def get_tool_spec_text(executor: "Optional[ToolExecutor]" = None) -> str:
    """
    Return a compact, token-efficient description of all standard tools.

    If executor is provided its descriptions are returned directly (not cached).
    Otherwise a default ToolExecutor is constructed once and the result cached.
    """
    global _cache

    if executor is not None:
        return executor.get_descriptions()

    if _cache is None:
        import tempfile
        from .executor import ToolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            _cache = ToolExecutor(tmp).get_descriptions()

    return _cache
