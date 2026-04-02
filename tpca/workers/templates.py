"""
Consistency templates for WorkerAgent system prompts.

Each task type (CODE_EDIT, NEW_FILE, REFACTOR, TEST, DOCS) has a short template
(≤300 tokens) that frames the worker's output format, naming conventions, and
the write_summary requirement.

Auto-detection maps a section description to a task type via keyword matching.
"""
from __future__ import annotations

TASK_TYPES = ("CODE_EDIT", "NEW_FILE", "REFACTOR", "TEST", "DOCS")

# Keywords that signal each task type (checked against lowercase description)
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "NEW_FILE": [
        "create new", "add new file", "new file", "implement new",
        "write new", "create a new", "add a new",
    ],
    "TEST": [
        "test", "tests", "unittest", "pytest", "coverage",
        "write tests", "add tests",
    ],
    "DOCS": [
        "document", "docstring", "documentation", "readme",
        "add comments", "write docs",
    ],
    "REFACTOR": [
        "refactor", "rename", "restructure", "extract", "reorganize",
        "move", "clean up", "clean-up",
    ],
    # CODE_EDIT is the catch-all default
    "CODE_EDIT": [
        "fix", "update", "modify", "change", "edit", "implement", "add",
    ],
}

_TEMPLATES: dict[str, str] = {
    "CODE_EDIT": """\
Edit existing code to fulfil the section goal.
- Read the relevant files first with read_file.
- Prefer apply_diff for precise, minimal edits; use patch_file as fallback.
- Keep changes minimal and focused on the section goal.
- You MUST call write_summary when done.\
""",
    "NEW_FILE": """\
Create new source file(s) to fulfil the section goal.
- Use write_file to create each new file.
- Follow the naming conventions shown in the style notes.
- Ensure imports and module structure match existing code patterns.
- List every added symbol in write_summary.new_symbols.
- You MUST call write_summary when done.\
""",
    "REFACTOR": """\
Refactor code while preserving observable behaviour.
- Read all affected files before making any changes.
- Apply changes with apply_diff for traceability.
- Use query_graph to find callers and update them.
- List every changed signature in write_summary.interfaces_changed.
- You MUST call write_summary when done.\
""",
    "TEST": """\
Write or update tests for the section goal.
- Use read_file to understand the code under test.
- Follow the test naming pattern from the style notes (e.g. test_*.py).
- Use write_file to create or update test files.
- Run tests with run_tests after writing to verify they pass.
- Set write_summary.test_result to "PASS" or "FAIL: <reason>".
- You MUST call write_summary when done.\
""",
    "DOCS": """\
Add or update documentation and docstrings.
- Read existing code before editing.
- Use patch_file or apply_diff to insert docstrings.
- Keep each docstring concise (≤3 sentences).
- You MUST call write_summary when done.\
""",
}


def detect_task_type(description: str) -> str:
    """
    Classify a section description into one of the TASK_TYPES.

    Checks task-specific keywords first (most specific first), then falls back
    to CODE_EDIT when ambiguous.
    """
    lower = description.lower()
    for task_type in ("NEW_FILE", "TEST", "DOCS", "REFACTOR", "CODE_EDIT"):
        if any(kw in lower for kw in _TYPE_KEYWORDS[task_type]):
            return task_type
    return "CODE_EDIT"


def get_template(task_type: str, naming_conventions: str = "") -> str:
    """
    Build the per-task-type template block for a worker system prompt.

    Args:
        task_type:          One of the TASK_TYPES strings.
        naming_conventions: Short style-notes string from global_style_notes.

    Returns:
        A system-prompt fragment (≤300 tokens).
    """
    body = _TEMPLATES.get(task_type, _TEMPLATES["CODE_EDIT"])
    header = f"# Task type: {task_type}"
    if naming_conventions:
        header = f"# Style notes\n{naming_conventions}\n\n" + header
    return f"{header}\n{body}"
