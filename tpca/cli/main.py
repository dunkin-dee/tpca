"""
TPCA command-line interface.

Usage:
    tpca run "document all public methods"
    tpca index [path]
    tpca repl
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Optional

import click
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

from tpca import TPCAOrchestrator, TPCAConfig
from tpca.logging.log_config import LogConfig

TPCA_VERSION = "3.0.0"

_NOISE_DIRS = {".git", "__pycache__", "node_modules", ".tpca_cache", "dist", ".venv", "venv"}

_COMMANDS = [
    "ls", "tree", "cat", "pwd", "cd",
    "run", "index", "stats",
    "config", "set",
    "shell", "help", "version", "exit", "quit",
]
_PATH_CMDS = {"ls", "tree", "cat", "cd", "index"}

REPL_HELP = """\
TPCA Interactive Session
========================

File browsing  (sandboxed to startup directory):
  ls [path]          List directory contents
  tree [path]        Directory tree (depth 3)
  cat <file>         Print file contents (max 500 lines)
  pwd                Print current directory
  cd <path>          Change directory

TPCA operations:
  run <task>         Run full pipeline on current directory
  index [path]       Run Pass 1 only and show compact index
  stats              Show stats from the last operation

Configuration:
  config             Show current configuration
  set <key> <value>  Modify a config value (use 'set budget N' for token budget)

Shell passthrough:
  !<command>         Run a shell command  (e.g. !git status)
  shell <command>    Alias for shell passthrough

Other:
  help               Show this message
  version            Show TPCA version
  exit / quit        Exit  (also Ctrl+D)
"""


# ── Config factory ─────────────────────────────────────────────────────────────

def _build_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    reader_model: Optional[str] = None,
    output_mode: Optional[str] = None,
    output_dir: Optional[str] = None,
    languages: Optional[str] = None,
    no_cache: bool = False,
    verbose: bool = False,
) -> TPCAConfig:
    """Map CLI options to TPCAConfig."""
    kwargs: dict = {}

    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["synthesis_model"] = model
    if reader_model:
        kwargs["reader_model"] = reader_model
    if output_mode:
        kwargs["output_mode"] = output_mode
    if output_dir:
        kwargs["output_dir"] = output_dir
    if languages:
        kwargs["languages"] = [l.strip() for l in languages.split(",")]
    if no_cache:
        kwargs["cache_enabled"] = False

    log_level = "DEBUG" if verbose else "WARN"
    kwargs["log"] = LogConfig(console_level=log_level)

    return TPCAConfig(**kwargs)


# ── Output formatters ──────────────────────────────────────────────────────────

_STAT_LABELS = {
    "symbols_indexed":    "Symbols indexed",
    "files_indexed":      "Files indexed",
    "pass1_time_ms":      "Pass 1 time (ms)",
    "total_time_ms":      "Total time (ms)",
    "compression_ratio":  "Compression ratio",
    "llm_calls":          "LLM calls",
    "output_chunks":      "Output chunks",
    "tokens_sent_to_llm": "Tokens sent to LLM",
    "fallback_used":      "Fallback used",
    "pass2_skipped":      "Pass 2 skipped",
    "reason":             "Skip reason",
}


def _print_stats(stats: dict) -> None:
    for key, label in _STAT_LABELS.items():
        if key in stats:
            val = stats[key]
            if key == "compression_ratio":
                val = f"{val}x"
            click.echo(f"  {label:<28} {val}")


def _print_run_result(result: dict) -> None:
    output = result.get("output", {})
    stats = result.get("stats", {})
    manifest = result.get("manifest")

    if output:
        click.echo("=" * 60)
        click.echo("OUTPUT")
        click.echo("=" * 60)
        for key, content in output.items():
            click.echo(f"\n-- {key} --")
            click.echo(str(content) if not isinstance(content, str) else content)

    if manifest and getattr(manifest, "files", None):
        click.echo("\n" + "=" * 60)
        click.echo("FILES WRITTEN")
        click.echo("=" * 60)
        for entry in manifest.files:
            marker = "OK" if getattr(entry, "status", "") == "complete" else "PARTIAL"
            click.echo(f"  [{marker}] {getattr(entry, 'output_file', entry)}")

    click.echo("\n" + "=" * 60)
    click.echo("STATS")
    click.echo("=" * 60)
    _print_stats(stats)


# ── Click CLI ─────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.version_option(TPCA_VERSION, prog_name="tpca")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Two-Pass Context Agent — AST-driven code synthesis for small LLMs."""
    if ctx.invoked_subcommand is None:
        _launch_repl(TPCAConfig())


@main.command()
@click.argument("task")
@click.option("--source", default=".", show_default=True, help="Source path to index.")
@click.option("--provider", type=click.Choice(["anthropic", "ollama"]), help="LLM provider.")
@click.option("--model", help="Synthesis model name.")
@click.option("--reader-model", help="Reader/planner model name.")
@click.option(
    "--output-mode",
    type=click.Choice(["inline", "single_file", "mirror", "per_symbol"]),
    help="Output mode.",
)
@click.option("--output-dir", help="Output directory (for non-inline modes).")
@click.option("--budget", type=int, default=None, help="Token budget for context slices.")
@click.option("--languages", help="Comma-separated languages to index (e.g. python,javascript).")
@click.option("--resume", default=None, help="Path to a manifest.json from a prior interrupted run.")
@click.option("--no-cache", is_flag=True, help="Disable the AST index cache.")
@click.option("--verbose", is_flag=True, help="Show detailed orchestrator logs.")
def run(
    task: str,
    source: str,
    provider: Optional[str],
    model: Optional[str],
    reader_model: Optional[str],
    output_mode: Optional[str],
    output_dir: Optional[str],
    budget: Optional[int],
    languages: Optional[str],
    resume: Optional[str],
    no_cache: bool,
    verbose: bool,
) -> None:
    """Run the full TPCA pipeline for TASK on SOURCE."""
    config = _build_config(
        provider=provider,
        model=model,
        reader_model=reader_model,
        output_mode=output_mode,
        output_dir=output_dir,
        languages=languages,
        no_cache=no_cache,
        verbose=verbose,
    )
    orchestrator = TPCAOrchestrator(config=config)
    click.echo(f"Source : {source}")
    click.echo(f"Task   : {task}")
    click.echo()
    try:
        result = orchestrator.run(
            source=source,
            task=task,
            budget_tokens=budget,
            resume_manifest=resume,
        )
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        sys.exit(1)
    _print_run_result(result)


@main.command("index")
@click.argument("path", default=".", required=False)
@click.option("--keywords", help="Comma-separated task keywords to bias ranking.")
@click.option("--languages", help="Comma-separated languages to index.")
@click.option("--no-cache", is_flag=True, help="Disable the AST index cache.")
@click.option("--verbose", is_flag=True, help="Show detailed orchestrator logs.")
def index_cmd(
    path: str,
    keywords: Optional[str],
    languages: Optional[str],
    no_cache: bool,
    verbose: bool,
) -> None:
    """Run Pass 1 only (AST index + ranking) and print the compact index."""
    config = _build_config(languages=languages, no_cache=no_cache, verbose=verbose)
    orchestrator = TPCAOrchestrator(config=config)
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    result = orchestrator.run_pass1_only(source=path, task_keywords=kw_list)
    click.echo(result["index"])
    click.echo()
    click.echo("─" * 40)
    _print_stats(result["stats"])


@main.command()
def repl() -> None:
    """Start an interactive TPCA session."""
    _launch_repl(TPCAConfig())


# ── REPL ──────────────────────────────────────────────────────────────────────

def _launch_repl(config: TPCAConfig) -> None:
    startup_dir = Path(os.getcwd()).resolve()
    click.echo(f"TPCA {TPCA_VERSION}  —  interactive session")
    click.echo(f"Root: {startup_dir}")
    click.echo('Type "help" for commands, Ctrl+D or "exit" to quit.\n')
    session = TPCARepl(startup_dir=startup_dir, config=config)
    session.cmdloop()


# ── Tab completer ──────────────────────────────────────────────────────────────

class _ReplCompleter(Completer):
    """prompt_toolkit completer for the TPCA REPL."""

    def __init__(self, session: "TPCARepl") -> None:
        self._s = session

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # No completion inside shell passthrough lines
        if text.lstrip().startswith("!"):
            return

        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        ends_with_space = text.endswith(" ")

        if ends_with_space:
            prior_tokens = tokens
            word = ""
        elif len(tokens) <= 1:
            # Still typing the command itself
            prior_tokens = []
            word = tokens[0] if tokens else ""
        else:
            prior_tokens = tokens[:-1]
            word = tokens[-1]

        if not prior_tokens:
            for cmd in _COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))

        elif prior_tokens[0] in _PATH_CMDS:
            for path in self._s._path_completions(word):
                yield Completion(path, start_position=-len(word))

        elif prior_tokens[0] == "set" and len(prior_tokens) == 1:
            keys = [f.name for f in dc_fields(self._s.config)] + ["budget", "resume"]
            for k in sorted(keys):
                if k.startswith(word):
                    yield Completion(k, start_position=-len(word))


# ── REPL class ────────────────────────────────────────────────────────────────

class TPCARepl:
    def __init__(self, startup_dir: Path, config: TPCAConfig) -> None:
        self.startup_dir = startup_dir
        self.current_dir = startup_dir
        self.config = config
        self._orchestrator: Optional[TPCAOrchestrator] = None
        self._last_stats: Optional[dict] = None
        # Extra run-time opts not stored in TPCAConfig
        self._budget: Optional[int] = None
        self._resume: Optional[str] = None

    # ── main loop ──────────────────────────────────────────────────────────

    def cmdloop(self) -> None:
        history = FileHistory(str(Path.home() / ".tpca_history"))
        completer = _ReplCompleter(self)

        while True:
            try:
                line = pt_prompt(
                    "tpca> ",
                    completer=completer,
                    history=history,
                    complete_while_typing=False,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye.")
                break
            if not line:
                continue
            try:
                self._dispatch(line)
            except SystemExit:
                break
            except Exception as exc:
                click.echo(f"Error: {exc}")

    def _dispatch(self, line: str) -> None:
        if line.startswith("!"):
            self._run_shell(line[1:].strip())
            return

        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            click.echo(f"Parse error: {exc}")
            return

        if not tokens:
            return

        cmd, *rest = tokens
        handlers = {
            "ls":      self._repl_ls,
            "tree":    self._repl_tree,
            "cat":     self._repl_cat,
            "pwd":     self._repl_pwd,
            "cd":      self._repl_cd,
            "run":     self._repl_run,
            "index":   self._repl_index,
            "stats":   self._repl_stats,
            "config":  self._repl_config,
            "set":     self._repl_set,
            "shell":   self._repl_shell_cmd,
            "help":    self._repl_help,
            "version": self._repl_version,
            "exit":    self._repl_exit,
            "quit":    self._repl_exit,
        }
        handler = handlers.get(cmd)
        if handler is None:
            click.echo(f"Unknown command: '{cmd}'. Type 'help' for available commands.")
            return
        handler(rest)

    # ── sandbox ────────────────────────────────────────────────────────────

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to current_dir, enforce it stays within startup_dir."""
        raw = Path(path_str)
        candidate = (raw if raw.is_absolute() else self.current_dir / raw).resolve()
        try:
            candidate.relative_to(self.startup_dir)
        except ValueError:
            raise ValueError(
                f"Access denied: '{path_str}' is outside the root directory ({self.startup_dir})"
            )
        return candidate

    # ── file browsing ──────────────────────────────────────────────────────

    def _repl_ls(self, args: list[str]) -> None:
        target = self._resolve_path(args[0] if args else ".")
        if not target.exists():
            click.echo(f"Not found: {target}")
            return
        if not target.is_dir():
            click.echo(f"Not a directory: {target}")
            return
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for entry in entries:
            if entry.is_dir():
                click.echo(f"  d  {entry.name}/")
            else:
                size = entry.stat().st_size
                click.echo(f"  f  {entry.name:<40} {size:>10,} B")

    def _repl_tree(self, args: list[str]) -> None:
        target = self._resolve_path(args[0] if args else ".")
        if not target.is_dir():
            click.echo(f"Not a directory: {target}")
            return
        click.echo(str(target))
        for line in _tree_lines(target, prefix="", max_depth=3):
            click.echo(line)

    def _repl_cat(self, args: list[str]) -> None:
        if not args:
            click.echo("Usage: cat <file>")
            return
        target = self._resolve_path(args[0])
        if not target.exists():
            click.echo(f"Not found: {target}")
            return
        if not target.is_file():
            click.echo(f"Not a file: {target}")
            return
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            click.echo(f"Cannot read file: {exc}")
            return
        truncated = len(lines) > 500
        for i, line in enumerate(lines[:500], 1):
            click.echo(f"{i:4}  {line}")
        if truncated:
            click.echo(f"\n  ... ({len(lines) - 500} more lines not shown)")

    def _repl_pwd(self, _args: list[str]) -> None:
        click.echo(str(self.current_dir))

    def _repl_cd(self, args: list[str]) -> None:
        if not args:
            self.current_dir = self.startup_dir
            click.echo(str(self.current_dir))
            return
        try:
            target = self._resolve_path(args[0])
        except ValueError as exc:
            click.echo(str(exc))
            return
        if not target.exists():
            click.echo(f"Not found: {target}")
            return
        if not target.is_dir():
            click.echo(f"Not a directory: {target}")
            return
        self.current_dir = target
        click.echo(str(self.current_dir))

    # ── TPCA ops ───────────────────────────────────────────────────────────

    def _repl_run(self, args: list[str]) -> None:
        if not args:
            click.echo("Usage: run <task description>")
            return
        task = " ".join(args)
        orch = self._get_orchestrator()
        click.echo(f"Running: {task}\n")
        try:
            result = orch.run(
                source=str(self.current_dir),
                task=task,
                budget_tokens=self._budget,
                resume_manifest=self._resume,
            )
        except KeyboardInterrupt:
            click.echo("\nInterrupted.")
            return
        self._last_stats = result.get("stats")
        _print_run_result(result)

    def _repl_index(self, args: list[str]) -> None:
        path_str = args[0] if args else "."
        try:
            target = self._resolve_path(path_str)
        except ValueError as exc:
            click.echo(str(exc))
            return
        orch = self._get_orchestrator()
        result = orch.run_pass1_only(source=str(target), task_keywords=None)
        self._last_stats = result.get("stats")
        click.echo(result["index"])
        click.echo()
        click.echo("─" * 40)
        _print_stats(result["stats"])

    def _repl_stats(self, args: list[str]) -> None:
        if self._last_stats is None:
            click.echo("No stats yet — run 'run' or 'index' first.")
            return
        _print_stats(self._last_stats)

    # ── config ─────────────────────────────────────────────────────────────

    def _repl_config(self, args: list[str]) -> None:
        click.echo("Current configuration:")
        for f in dc_fields(self.config):
            if f.name == "log":
                continue
            click.echo(f"  {f.name:<35} = {getattr(self.config, f.name)!r}")
        click.echo(f"  {'budget':<35} = {self._budget!r}")
        click.echo(f"  {'resume':<35} = {self._resume!r}")

    def _repl_set(self, args: list[str]) -> None:
        if len(args) < 2:
            click.echo("Usage: set <key> <value>")
            return
        key, raw = args[0], " ".join(args[1:])

        if key == "budget":
            try:
                self._budget = int(raw)
                click.echo(f"Set budget = {self._budget}")
            except ValueError:
                click.echo(f"Invalid integer: {raw!r}")
            return
        if key == "resume":
            self._resume = raw
            click.echo(f"Set resume = {self._resume!r}")
            return

        valid = {f.name: f for f in dc_fields(self.config)}
        if key not in valid:
            click.echo(f"Unknown key: '{key}'")
            click.echo(f"Valid keys: {', '.join(sorted(valid))} + budget, resume")
            return

        try:
            value = _coerce_value(raw, valid[key])
        except (ValueError, TypeError) as exc:
            click.echo(f"Invalid value for '{key}': {exc}")
            return

        setattr(self.config, key, value)
        self._orchestrator = None  # invalidate so next op picks up new config
        click.echo(f"Set {key} = {value!r}")

    # ── shell passthrough ──────────────────────────────────────────────────

    def _run_shell(self, cmd_str: str) -> None:
        click.echo(f"[shell] {cmd_str}")
        result = subprocess.run(cmd_str, shell=True, cwd=str(self.current_dir))
        if result.returncode != 0:
            click.echo(f"[exit code {result.returncode}]")

    def _repl_shell_cmd(self, args: list[str]) -> None:
        if not args:
            click.echo("Usage: shell <command>")
            return
        self._run_shell(" ".join(args))

    # ── meta ───────────────────────────────────────────────────────────────

    def _repl_help(self, args: list[str]) -> None:
        click.echo(REPL_HELP)

    def _repl_version(self, args: list[str]) -> None:
        click.echo(f"TPCA {TPCA_VERSION}")

    def _repl_exit(self, args: list[str]) -> None:
        click.echo("Goodbye.")
        raise SystemExit(0)

    # ── path completion (used by _ReplCompleter) ───────────────────────────

    def _path_completions(self, text: str) -> list[str]:
        """Return matching filesystem paths relative to current_dir, sandboxed."""
        if text.endswith("/") or text.endswith(os.sep):
            search_dir = self.current_dir / text
            dir_prefix = text
            name_prefix = ""
        else:
            partial = Path(text)
            parent_str = str(partial.parent)
            search_dir = self.current_dir / partial.parent
            dir_prefix = "" if parent_str == "." else parent_str + "/"
            name_prefix = partial.name

        try:
            search_dir.resolve().relative_to(self.startup_dir)
        except ValueError:
            return []

        if not search_dir.is_dir():
            return []

        results = []
        try:
            for entry in sorted(search_dir.iterdir(), key=lambda p: p.name.lower()):
                if entry.name.startswith(name_prefix):
                    completion = dir_prefix + entry.name
                    if entry.is_dir():
                        completion += "/"
                    results.append(completion)
        except PermissionError:
            pass
        return results

    # ── lazy orchestrator ──────────────────────────────────────────────────

    def _get_orchestrator(self) -> TPCAOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = TPCAOrchestrator(config=self.config)
        return self._orchestrator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tree_lines(directory: Path, prefix: str, depth: int, max_depth: int = 3) -> list[str]:
    if depth >= max_depth:
        return []
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return []
    entries = [e for e in entries if e.name not in _NOISE_DIRS]
    lines = []
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(_tree_lines(entry, prefix + extension, depth + 1, max_depth))
    return lines


def _coerce_value(raw: str, field_def) -> object:
    """Best-effort type coercion for `set` command."""
    import dataclasses
    default = None
    if field_def.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        default = field_def.default_factory()

    if isinstance(default, dict):
        raise TypeError("dict fields cannot be set via the CLI")
    if isinstance(default, list):
        return [v.strip() for v in raw.split(",")]
    if isinstance(default, bool) or raw.lower() in ("true", "false", "1", "0"):
        return raw.lower() in ("true", "1")
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


if __name__ == "__main__":
    main()
