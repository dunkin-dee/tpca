"""
ToolExecutor — executes tool calls on behalf of worker agents.

Provides read_file, write_file, apply_diff, patch_file, list_dir,
grep_symbol, query_graph, run_shell, run_tests, and write_summary.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .registry import Tool, ToolRegistry


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    output: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_str(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return self.output


# ── Security constants ────────────────────────────────────────────────────────

# Shell commands permitted (prefix match against stripped command)
_SHELL_ALLOWED_PREFIXES = (
    "pytest",
    "python -m pytest",
    "git status",
    "git diff",
    "npm test",
    "cargo test",
)

# Shell injection characters that are never allowed
_SHELL_DANGEROUS_CHARS = (";", "&&", "||", "|", ">", "<", "`", "$(")

# Max bytes captured from subprocess before truncation (~500 tokens)
_MAX_SHELL_OUTPUT_BYTES = 8000


# ── Path safety ───────────────────────────────────────────────────────────────

def _resolve_safe(
    path_str: str, project_root: str
) -> tuple[Optional[Path], Optional[str]]:
    """
    Resolve path_str relative to project_root.
    Returns (resolved_path, None) on success, (None, error) on traversal attempt.
    """
    root = Path(project_root).resolve()
    try:
        if os.path.isabs(path_str):
            resolved = Path(path_str).resolve()
        else:
            resolved = (root / path_str).resolve()
        resolved.relative_to(root)  # raises ValueError if outside root
        return resolved, None
    except (ValueError, OSError) as exc:
        return None, f"Path '{path_str}' is outside project root: {exc}"


# ── Unified diff applier ──────────────────────────────────────────────────────

def _apply_unified_diff(
    original: str, diff: str
) -> tuple[str, Optional[str]]:
    """
    Apply a unified diff to original text.
    Returns (patched_text, None) on success, (original, error_message) on failure.
    """
    lines = original.splitlines(keepends=True)
    diff_lines = diff.splitlines()

    # Skip file header lines (--- / +++)
    di = 0
    while di < len(diff_lines) and not diff_lines[di].startswith("@@"):
        di += 1

    result = list(lines)
    offset = 0  # cumulative offset from prior hunks

    while di < len(diff_lines):
        line = diff_lines[di]
        if not line.startswith("@@"):
            di += 1
            continue

        m = re.match(r"@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
        if not m:
            di += 1
            continue

        orig_start = int(m.group(1)) - 1       # 0-indexed
        orig_count = int(m.group(2)) if m.group(2) is not None else 1
        di += 1

        old_lines: list[str] = []
        new_lines: list[str] = []

        while di < len(diff_lines) and not diff_lines[di].startswith("@@"):
            hl = diff_lines[di]
            di += 1
            if hl == "\\ No newline at end of file":
                continue
            if hl.startswith("-"):
                old_lines.append(hl[1:])
            elif hl.startswith("+"):
                new_lines.append(hl[1:])
            else:
                ctx = hl[1:] if hl.startswith(" ") else hl
                old_lines.append(ctx)
                new_lines.append(ctx)

        # Ensure each replacement line ends with a newline
        new_with_nl = [
            (ln if ln.endswith("\n") else ln + "\n") for ln in new_lines
        ]

        actual_start = orig_start + offset
        if actual_start < 0 or actual_start > len(result):
            return original, (
                f"Hunk at line {orig_start + 1} is out of range "
                f"(file has {len(result)} lines)"
            )

        result[actual_start : actual_start + orig_count] = new_with_nl
        offset += len(new_with_nl) - orig_count

    return "".join(result), None


# ── Shell helpers ─────────────────────────────────────────────────────────────

def _is_allowed_shell_command(cmd: str) -> bool:
    cmd = cmd.strip()
    for prefix in _SHELL_ALLOWED_PREFIXES:
        if cmd == prefix or cmd.startswith(prefix + " "):
            return True
    return False


def _has_dangerous_chars(cmd: str) -> bool:
    return any(c in cmd for c in _SHELL_DANGEROUS_CHARS)


# ── ToolExecutor ──────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes the 10 standard PRISM tools for worker agents.

    Args:
        project_root: Absolute path to the project directory. All file
                      operations are confined to this directory.
        graph:        Optional NetworkX DiGraph from Pass 1, used by
                      grep_symbol and query_graph.
        index_text:   Optional compact index text, searched by grep_symbol.
    """

    def __init__(
        self,
        project_root: str,
        graph: Any = None,
        index_text: str = "",
    ) -> None:
        self._root = str(Path(project_root).resolve())
        self._graph = graph
        self._index_text = index_text
        self._last_summary: Optional[dict] = None
        self._registry = self._build_registry()

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a named tool with args dict. Returns ToolResult."""
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult("", f"Unknown tool '{tool_name}'")
        try:
            output = tool.handler(args)
            return ToolResult(output)
        except Exception as exc:
            return ToolResult("", str(exc))

    def get_schemas(self) -> list[dict]:
        """
        Return tool schemas in generic format:
        [{"name": ..., "description": ..., "parameters": <JSON Schema>}]
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._registry.all()
        ]

    def get_descriptions(self) -> str:
        """Compact text description of all tools for JSON-fallback models."""
        lines = [
            'Available tools — call with: {"tool": "<name>", "args": {...}}\n'
        ]
        for t in self._registry.all():
            props = t.parameters.get("properties", {})
            required = t.parameters.get("required", [])
            param_parts = []
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                suffix = "" if pname in required else "?"
                param_parts.append(f"{pname}{suffix}:{ptype}")
            params = ", ".join(param_parts)
            lines.append(f"  {t.name}({params}) — {t.description}")
        return "\n".join(lines)

    @property
    def last_summary(self) -> Optional[dict]:
        """Most recent write_summary payload, or None if not yet called."""
        return self._last_summary

    # ── Registry builder ──────────────────────────────────────────────────────

    def _build_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        tools = [
            Tool(
                name="read_file",
                description=(
                    "Read file content. Optionally specify start_line and "
                    "end_line (1-indexed, inclusive) to read a slice."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line to read (1-indexed)",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Last line to read (inclusive)",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._handle_read_file,
            ),
            Tool(
                name="write_file",
                description=(
                    "Write or overwrite a file with the given content. "
                    "Creates parent directories if needed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full file content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._handle_write_file,
            ),
            Tool(
                name="apply_diff",
                description=(
                    "Apply a unified diff (output of `diff -u`) to an existing "
                    "file. Preferred over patch_file for complex edits."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root",
                        },
                        "unified_diff": {
                            "type": "string",
                            "description": "Unified diff string to apply",
                        },
                    },
                    "required": ["path", "unified_diff"],
                },
                handler=self._handle_apply_diff,
            ),
            Tool(
                name="patch_file",
                description=(
                    "Replace exact old_text with new_text in a file. "
                    "Use when you cannot produce a unified diff."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Exact text to find and replace",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
                handler=self._handle_patch_file,
            ),
            Tool(
                name="list_dir",
                description=(
                    "List files in a directory. "
                    "Optionally filter by glob pattern (e.g. '*.py')."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to project root",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to filter results",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._handle_list_dir,
            ),
            Tool(
                name="grep_symbol",
                description=(
                    "Find where a symbol (function, class, variable) is defined "
                    "in the AST index or dependency graph."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Symbol name to search for",
                        },
                    },
                    "required": ["name"],
                },
                handler=self._handle_grep_symbol,
            ),
            Tool(
                name="query_graph",
                description=(
                    "Get callers ('in') or callees ('out') of a symbol "
                    "from the Pass 1 dependency graph."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol_id": {
                            "type": "string",
                            "description": (
                                "Symbol ID in the graph "
                                "(e.g. 'auth.py::Auth.validate_token')"
                            ),
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["in", "out"],
                            "description": "'in' for callers, 'out' for callees",
                        },
                    },
                    "required": ["symbol_id", "direction"],
                },
                handler=self._handle_query_graph,
            ),
            Tool(
                name="run_shell",
                description=(
                    "Run an allowlisted shell command in the project root. "
                    "Allowed: pytest, python -m pytest, git status, git diff, "
                    "npm test, cargo test."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to run",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory relative to project root",
                        },
                    },
                    "required": ["command"],
                },
                handler=self._handle_run_shell,
            ),
            Tool(
                name="run_tests",
                description=(
                    "Run pytest on specified file or directory paths. "
                    "Output is truncated to ~500 tokens."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Test file or directory paths "
                                "relative to project root"
                            ),
                        },
                    },
                    "required": ["paths"],
                },
                handler=self._handle_run_tests,
            ),
            Tool(
                name="write_summary",
                description=(
                    "Required final call. Emit a structured summary of work "
                    "completed for this section. Must be called before finishing."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["COMPLETE", "PARTIAL", "FAILED"],
                        },
                        "brief": {
                            "type": "string",
                            "description": "≤150 chars, past tense",
                        },
                        "detail": {
                            "type": "string",
                            "description": "≤500 chars, decisions made and why",
                        },
                        "files_changed": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "symbols_touched": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "interfaces_changed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Signatures that changed — "
                                "'file::Class.method(args) -> return_type'"
                            ),
                        },
                        "new_symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Symbols added — 'file::SymbolName'",
                        },
                        "assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "section_id", "status", "brief", "detail",
                        "files_changed", "symbols_touched",
                    ],
                },
                handler=self._handle_write_summary,
            ),
        ]
        for tool in tools:
            reg.register(tool)
        return reg

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_read_file(self, args: dict) -> str:
        path, err = _resolve_safe(args["path"], self._root)
        if err:
            raise ValueError(err)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {args['path']}")

        text = path.read_text(encoding="utf-8", errors="replace")
        start = args.get("start_line")
        end = args.get("end_line")
        if start is not None or end is not None:
            all_lines = text.splitlines(keepends=True)
            s = max(0, (start or 1) - 1)
            e = end or len(all_lines)
            text = "".join(all_lines[s:e])
        return text

    def _handle_write_file(self, args: dict) -> str:
        path, err = _resolve_safe(args["path"], self._root)
        if err:
            raise ValueError(err)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} chars to {args['path']}"

    def _handle_apply_diff(self, args: dict) -> str:
        path, err = _resolve_safe(args["path"], self._root)
        if err:
            raise ValueError(err)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {args['path']}")

        original = path.read_text(encoding="utf-8", errors="replace")
        patched, patch_err = _apply_unified_diff(original, args["unified_diff"])
        if patch_err:
            raise ValueError(f"Failed to apply diff: {patch_err}")
        path.write_text(patched, encoding="utf-8")
        return f"Applied diff to {args['path']}"

    def _handle_patch_file(self, args: dict) -> str:
        path, err = _resolve_safe(args["path"], self._root)
        if err:
            raise ValueError(err)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {args['path']}")

        content = path.read_text(encoding="utf-8", errors="replace")
        if args["old_text"] not in content:
            raise ValueError(
                f"old_text not found in {args['path']}. "
                "Ensure exact whitespace and newline match."
            )
        patched = content.replace(args["old_text"], args["new_text"], 1)
        path.write_text(patched, encoding="utf-8")
        return f"Patched {args['path']}"

    def _handle_list_dir(self, args: dict) -> str:
        path, err = _resolve_safe(args["path"], self._root)
        if err:
            raise ValueError(err)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {args['path']}")

        pattern = args.get("pattern", "*")
        matched = sorted(path.glob(pattern))
        rel_paths = [
            str(p.relative_to(self._root)).replace("\\", "/")
            for p in matched
        ]
        if not rel_paths:
            return f"No files matching '{pattern}' in {args['path']}"
        return "\n".join(rel_paths)

    def _handle_grep_symbol(self, args: dict) -> str:
        name = args["name"]
        results: list[str] = []

        # Search AST graph nodes
        if self._graph is not None:
            for node_id, data in self._graph.nodes(data=True):
                sym_name = data.get("name", "") or node_id.split("::")[-1]
                if name.lower() in sym_name.lower():
                    file_path = data.get("file", "?")
                    line = data.get("line", "?")
                    kind = data.get("kind", "symbol")
                    results.append(f"{file_path}:{line}  [{kind}] {node_id}")

        # Search compact index text for the name
        if self._index_text and name in self._index_text:
            for i, line in enumerate(self._index_text.splitlines(), 1):
                if name in line:
                    results.append(f"index:{i}  {line.strip()}")

        if not results:
            return f"Symbol '{name}' not found in index or graph."
        return "\n".join(results[:50])

    def _handle_query_graph(self, args: dict) -> str:
        symbol_id = args["symbol_id"]
        direction = args["direction"]

        if self._graph is None:
            return "Graph not available (Pass 1 not yet run)."

        if symbol_id not in self._graph:
            candidates = [n for n in self._graph.nodes() if symbol_id in n]
            if not candidates:
                return f"Symbol '{symbol_id}' not found in graph."
            symbol_id = candidates[0]

        if direction == "in":
            neighbors = list(self._graph.predecessors(symbol_id))
            label = f"callers of '{symbol_id}'"
        else:
            neighbors = list(self._graph.successors(symbol_id))
            label = f"callees of '{symbol_id}'"

        if not neighbors:
            return f"No {label}."
        return f"{label}:\n" + "\n".join(f"  {n}" for n in neighbors[:30])

    def _handle_run_shell(self, args: dict) -> str:
        cmd = args["command"].strip()
        if _has_dangerous_chars(cmd):
            raise ValueError(
                f"Command contains disallowed characters: {cmd!r}"
            )
        if not _is_allowed_shell_command(cmd):
            raise ValueError(
                f"Command not in allowlist: {cmd!r}\n"
                "Allowed prefixes: pytest, python -m pytest, git status, "
                "git diff, npm test, cargo test"
            )

        cwd = self._root
        if args.get("cwd"):
            cwd_path, err = _resolve_safe(args["cwd"], self._root)
            if err:
                raise ValueError(err)
            cwd = str(cwd_path)

        return self._run_subprocess(cmd, cwd)

    def _handle_run_tests(self, args: dict) -> str:
        paths = args.get("paths", [])
        if not paths:
            raise ValueError("paths list is empty")

        safe_paths: list[str] = []
        for p in paths:
            resolved, err = _resolve_safe(p, self._root)
            if err:
                raise ValueError(err)
            safe_paths.append(
                str(resolved.relative_to(self._root)).replace("\\", "/")
            )

        cmd = "pytest " + " ".join(safe_paths)
        return self._run_subprocess(cmd, self._root)

    def _handle_write_summary(self, args: dict) -> str:
        required = [
            "section_id", "status", "brief", "detail",
            "files_changed", "symbols_touched",
        ]
        missing = [f for f in required if f not in args]
        if missing:
            raise ValueError(
                f"write_summary missing required fields: {missing}"
            )

        status = args["status"]
        if status not in ("COMPLETE", "PARTIAL", "FAILED"):
            raise ValueError(
                f"status must be COMPLETE|PARTIAL|FAILED, got {status!r}"
            )

        brief = args["brief"]
        if len(brief) > 150:
            raise ValueError(
                f"brief exceeds 150 chars (got {len(brief)}). Shorten it."
            )

        detail = args.get("detail", "")
        if len(detail) > 500:
            raise ValueError(
                f"detail exceeds 500 chars (got {len(detail)}). Shorten it."
            )

        self._last_summary = {
            "section_id": args["section_id"],
            "status": status,
            "brief": brief,
            "detail": detail,
            "files_changed": list(args.get("files_changed", [])),
            "symbols_touched": list(args.get("symbols_touched", [])),
            "interfaces_changed": list(args.get("interfaces_changed", [])),
            "new_symbols": list(args.get("new_symbols", [])),
            "assumptions": list(args.get("assumptions", [])),
            "blockers": list(args.get("blockers", [])),
        }
        return f"Summary recorded: {status} — {brief}"

    # ── Subprocess helper ─────────────────────────────────────────────────────

    def _run_subprocess(self, cmd: str, cwd: str) -> str:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=120,
            )
            output = result.stdout + result.stderr
            if len(output.encode()) > _MAX_SHELL_OUTPUT_BYTES:
                output = (
                    output.encode()[:_MAX_SHELL_OUTPUT_BYTES].decode(
                        "utf-8", errors="replace"
                    )
                    + "\n... (truncated)"
                )
            prefix = "EXIT 0\n" if result.returncode == 0 else f"EXIT {result.returncode}\n"
            return prefix + output
        except subprocess.TimeoutExpired:
            return "EXIT TIMEOUT\nCommand exceeded 120s timeout."
        except Exception as exc:
            return f"EXIT ERROR\n{exc}"
