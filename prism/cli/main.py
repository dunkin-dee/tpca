"""
PRISM command-line interface.

Usage:
    prism run "document all public methods"
    prism index [path]
    prism repl
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

from prism import PRISMOrchestrator, PRISMConfig
from prism.logging.log_config import LogConfig
from prism.plan.plan_model import PlanSection, SessionPlan
from prism.plan.plan_store import PlanStore
from prism.session_manager import SessionManager
from prism.tools.executor import ToolExecutor

PRISM_VERSION = "3.0.0"

_NOISE_DIRS = {".git", "__pycache__", "node_modules", ".prism_cache", "dist", ".venv", "venv"}

_COMMANDS = [
    "ls", "tree", "cat", "pwd", "cd",
    "run", "index", "stats",
    "plan", "continue", "eval", "retry",
    "tools", "summary", "diff",
    "config", "set",
    "watch",
    "shell", "help", "version", "exit", "quit",
]
_PATH_CMDS = {"ls", "tree", "cat", "cd", "index"}

REPL_HELP = """\
PRISM Interactive Session
========================

File browsing  (sandboxed to startup directory):
  ls [path]          List directory contents
  tree [path]        Directory tree (depth 3)
  cat <file>         Print file contents (max 500 lines)
  pwd                Print current directory
  cd <path>          Change directory

PRISM operations:
  run <task>         Run full pipeline on current directory
  index [path]       Run Pass 1 only and show compact index
  stats              Show stats from the last operation
  watch              Show file watcher status

Plan management (coding sessions):
  plan               Show tree view of current plan
  plan new <task>    Start a new coding session
  plan clear         Delete the current plan (asks for confirmation)
  continue           Resume execution from last saved state
  eval <section_id>  Re-evaluate a specific section
  retry <section_id> Re-run a BLOCKED or NEEDS_REVISION section
  tools              List all available worker tools
  summary            Compact table of all worker section summaries
  diff <section_id>  Show git diff for files changed by a section

Configuration:
  config             Show current configuration
  set <key> <value>  Modify a config value (use 'set budget N' for token budget)

Shell passthrough:
  !<command>         Run a shell command  (e.g. !git status)
  shell <command>    Alias for shell passthrough

Other:
  help               Show this message
  version            Show PRISM version
  exit / quit        Exit  (also Ctrl+D)
"""


# ── Config factory ─────────────────────────────────────────────────────────────

def _build_config(
    preset: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    reader_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    output_mode: Optional[str] = None,
    output_dir: Optional[str] = None,
    languages: Optional[str] = None,
    no_cache: bool = False,
    verbose: bool = False,
    debug_llm: bool = False,
) -> PRISMConfig:
    """Map CLI options to PRISMConfig. Preset is applied first; explicit flags override it."""
    # Start from preset or bare defaults
    if preset:
        try:
            config = PRISMConfig.from_preset(preset)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--preset")
        kwargs: dict = {}
    else:
        config = None
        kwargs = {}

    if provider:
        kwargs["provider"] = provider
    if model:
        if config and config.provider == "ollama":
            kwargs["ollama_synthesis_model"] = model
        else:
            kwargs["synthesis_model"] = model
    if reader_model:
        if config and config.provider == "ollama":
            kwargs["ollama_reader_model"] = reader_model
        else:
            kwargs["reader_model"] = reader_model
    if ollama_url:
        kwargs["ollama_base_url"] = ollama_url
    if output_mode:
        kwargs["output_mode"] = output_mode
    if output_dir:
        kwargs["output_dir"] = output_dir
    if languages:
        kwargs["languages"] = [l.strip() for l in languages.split(",")]
    if no_cache:
        kwargs["cache_enabled"] = False

    log_level = "DEBUG" if verbose else "WARN"
    kwargs["log"] = LogConfig(console_level=log_level, include_prompt_text=debug_llm)

    if config is not None:
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config
    return PRISMConfig(**kwargs)


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
@click.version_option(PRISM_VERSION, prog_name="prism")
@click.pass_context
def main(ctx: click.Context) -> None:
    """PageRank-indexed, Symbol-aware Model — AST-driven code synthesis for small LLMs."""
    if ctx.invoked_subcommand is None:
        _launch_repl(PRISMConfig())


@main.command()
@click.argument("task")
@click.option("--source", default=".", show_default=True, help="Source path to index.")
@click.option(
    "--preset",
    type=click.Choice(["13b-local", "7b-local", "cloud"]),
    help="Config preset (13b-local, 7b-local, cloud).",
)
@click.option("--provider", type=click.Choice(["anthropic", "ollama"]), help="LLM provider.")
@click.option("--model", help="Synthesis model name.")
@click.option("--reader-model", help="Reader/planner model name.")
@click.option("--ollama-url", help="Ollama base URL (default: http://localhost:11434/v1).")
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
@click.option("--debug-llm", is_flag=True, help="Print full LLM input/output to stderr.")
def run(
    task: str,
    source: str,
    preset: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    reader_model: Optional[str],
    ollama_url: Optional[str],
    output_mode: Optional[str],
    output_dir: Optional[str],
    budget: Optional[int],
    languages: Optional[str],
    resume: Optional[str],
    no_cache: bool,
    verbose: bool,
    debug_llm: bool,
) -> None:
    """Run the full PRISM pipeline for TASK on SOURCE."""
    config = _build_config(
        preset=preset,
        provider=provider,
        model=model,
        reader_model=reader_model,
        ollama_url=ollama_url,
        output_mode=output_mode,
        output_dir=output_dir,
        languages=languages,
        no_cache=no_cache,
        verbose=verbose,
        debug_llm=debug_llm,
    )
    orchestrator = PRISMOrchestrator(config=config)
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
    orchestrator = PRISMOrchestrator(config=config)
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    result = orchestrator.run_pass1_only(source=path, task_keywords=kw_list)
    click.echo(result["index"])
    click.echo()
    click.echo("─" * 40)
    _print_stats(result["stats"])


@main.command()
@click.option(
    "--preset",
    type=click.Choice(["13b-local", "7b-local", "cloud"]),
    help="Config preset (13b-local, 7b-local, cloud).",
)
@click.option("--ollama-url", help="Ollama base URL (default: http://localhost:11434/v1).")
def repl(preset: Optional[str], ollama_url: Optional[str]) -> None:
    """Start an interactive PRISM session."""
    config = _build_config(preset=preset, ollama_url=ollama_url)
    _launch_repl(config)


# ── REPL ──────────────────────────────────────────────────────────────────────

def _launch_repl(config: PRISMConfig) -> None:
    startup_dir = Path(os.getcwd()).resolve()
    click.echo(f"PRISM {PRISM_VERSION}  —  interactive session")
    click.echo(f"Root: {startup_dir}")
    click.echo(f"Preset: {config.provider} / {config.active_synthesis_model}")
    click.echo('Type "help" for commands, Ctrl+D or "exit" to quit.')
    session = PRISMRepl(startup_dir=startup_dir, config=config)
    # Probe Ollama connectivity when provider is ollama
    if config.provider == "ollama":
        session._check_ollama()
    click.echo()
    session.cmdloop()


# ── Tab completer ──────────────────────────────────────────────────────────────

class _ReplCompleter(Completer):
    """prompt_toolkit completer for the PRISM REPL."""

    def __init__(self, session: "PRISMRepl") -> None:
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

class PRISMRepl:
    def __init__(self, startup_dir: Path, config: PRISMConfig) -> None:
        self.startup_dir = startup_dir
        self.current_dir = startup_dir
        self.config = config
        self._orchestrator: Optional[PRISMOrchestrator] = None
        self._last_stats: Optional[dict] = None
        # Extra run-time opts not stored in PRISMConfig
        self._budget: Optional[int] = None
        self._resume: Optional[str] = None
        # Phase F — plan/session state
        self._compact_index: Optional[str] = None
        self._graph = None

    # ── main loop ──────────────────────────────────────────────────────────

    def cmdloop(self) -> None:
        history = FileHistory(str(Path.home() / ".prism_history"))
        completer = _ReplCompleter(self)

        while True:
            try:
                line = pt_prompt(
                    "prism> ",
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
            "ls":       self._repl_ls,
            "tree":     self._repl_tree,
            "cat":      self._repl_cat,
            "pwd":      self._repl_pwd,
            "cd":       self._repl_cd,
            "run":      self._repl_run,
            "index":    self._repl_index,
            "stats":    self._repl_stats,
            "watch":    self._repl_watch,
            "plan":     self._repl_plan,
            "continue": self._repl_continue,
            "eval":     self._repl_eval,
            "retry":    self._repl_retry,
            "tools":    self._repl_tools,
            "summary":  self._repl_summary,
            "diff":     self._repl_diff,
            "config":   self._repl_config,
            "set":      self._repl_set,
            "shell":    self._repl_shell_cmd,
            "help":     self._repl_help,
            "version":  self._repl_version,
            "exit":     self._repl_exit,
            "quit":     self._repl_exit,
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

    # ── PRISM ops ───────────────────────────────────────────────────────────

    def _repl_run(self, args: list[str]) -> None:
        if not args:
            click.echo("Usage: run <task description>")
            return
        task = " ".join(args)
        orch = self._get_orchestrator()
        orch.start_watching(str(self.current_dir))
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
        orch.start_watching(str(target))
        result = orch.run_pass1_only(source=str(target), task_keywords=None)
        self._last_stats = result.get("stats")
        self._compact_index = result.get("index")
        self._graph = result.get("graph")
        click.echo(result["index"])
        click.echo()
        click.echo("─" * 40)
        _print_stats(result["stats"])

    def _repl_stats(self, args: list[str]) -> None:
        if self._last_stats is None:
            click.echo("No stats yet — run 'run' or 'index' first.")
            return
        _print_stats(self._last_stats)

    # ── watch ──────────────────────────────────────────────────────────────

    def _repl_watch(self, _args: list[str]) -> None:
        """Show file watcher status."""
        if self._orchestrator is None:
            click.echo("Watcher: not started  (run 'index' or 'run' to activate)")
            return
        click.echo(f"Watcher: {self._orchestrator.watcher.status_line()}")

    # ── Ollama health check ────────────────────────────────────────────────

    def _check_ollama(self) -> None:
        """Probe Ollama connectivity and list available models."""
        import json
        import urllib.request
        import urllib.error

        base = self.config.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = base + "/api/tags"

        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            click.echo(f"Ollama: connected — {len(models)} model(s) available")
            if models:
                shown = models[:6]
                suffix = f"  (+{len(models) - 6} more)" if len(models) > 6 else ""
                click.echo("  " + "  ".join(shown) + suffix)
            current = self.config.active_synthesis_model
            if models and current not in models:
                click.echo(
                    f"  [!] configured model '{current}' not found locally"
                    f" — run: ollama pull {current}"
                )
        except urllib.error.URLError:
            click.echo(f"Ollama: not reachable at {base}")
            click.echo(
                "  Start Ollama, or change the URL with:  "
                "set ollama_base_url http://<host>:11434/v1"
            )
        except Exception as exc:
            click.echo(f"Ollama: probe failed ({exc})")

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
        if key == "debug_llm":
            val = raw.lower() in ("1", "true", "yes", "on")
            self.config.log.include_prompt_text = val
            click.echo(f"Set debug_llm = {val}")
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
        if self._orchestrator is not None:
            self._orchestrator.stop_watching()
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
        click.echo(f"PRISM {PRISM_VERSION}")

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

    # ── Phase F — plan/session helpers ────────────────────────────────────────

    def _get_plan_store(self) -> PlanStore:
        return PlanStore(project_root=str(self.current_dir))

    def _get_session_manager(self) -> SessionManager:
        orch = self._get_orchestrator()
        return SessionManager(
            plan_store=self._get_plan_store(),
            llm=orch.llm,
            config=self.config,
            graph=self._graph,
            compact_index=self._compact_index or "",
            project_root=str(self.current_dir),
        )

    def _ensure_index(self) -> None:
        """Run Pass 1 if not already done this session."""
        if self._compact_index is None:
            click.echo("Running Pass 1 index…")
            orch = self._get_orchestrator()
            result = orch.run_pass1_only(source=str(self.current_dir))
            self._compact_index = result["index"]
            self._graph = result["graph"]
            self._last_stats = result.get("stats")
            click.echo(
                f"Index ready ({result['stats']['symbols_indexed']} symbols)."
            )

    # ── Phase F — plan commands ────────────────────────────────────────────────

    def _repl_plan(self, args: list[str]) -> None:
        """plan / plan new <task> / plan clear."""
        store = self._get_plan_store()

        if not args:
            plan = store.load()
            if plan is None:
                click.echo(
                    "No plan found. Use 'plan new <task>' to start a session."
                )
                return
            click.echo(_render_plan_tree(plan))
            return

        sub = args[0]

        if sub == "new":
            task_parts = args[1:]
            if not task_parts:
                click.echo("Usage: plan new <task description>")
                return
            task = " ".join(task_parts)

            if store.exists():
                click.echo(
                    "Warning: a plan already exists for this directory."
                )
                try:
                    answer = input("Overwrite? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    click.echo("\nAborted.")
                    return
                if answer != "y":
                    click.echo("Aborted.")
                    return

            self._ensure_index()
            sm = self._get_session_manager()
            click.echo(f"Planning session: {task}")
            try:
                plan = sm.start_session(task)
            except Exception as exc:
                click.echo(f"Planning failed: {exc}")
                return
            click.echo(_render_plan_tree(plan))

        elif sub == "clear":
            if not store.exists():
                click.echo("No plan to clear.")
                return
            try:
                answer = input("Delete current plan? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nAborted.")
                return
            if answer != "y":
                click.echo("Aborted.")
                return
            store.clear()
            click.echo("Plan deleted.")

        else:
            click.echo(
                f"Unknown plan subcommand: '{sub}'. "
                "Try: plan, plan new <task>, plan clear"
            )

    def _repl_continue(self, args: list[str]) -> None:
        """Resume from last saved state and dispatch pending workers."""
        store = self._get_plan_store()
        plan = store.load()

        if plan is None:
            click.echo(
                "No saved plan found. Use 'plan new <task>' to start a session."
            )
            return

        if plan.status == "COMPLETE":
            click.echo("Plan is already complete. Use 'plan' to view results.")
            return

        self._ensure_index()
        sm = self._get_session_manager()
        executor = ToolExecutor(
            project_root=str(self.current_dir),
            graph=self._graph,
            index_text=self._compact_index or "",
        )

        click.echo(f"Resuming: {plan.task!r}")
        try:
            summaries = sm.dispatch_workers(plan, executor)
        except KeyboardInterrupt:
            click.echo("\nInterrupted.")
            return
        click.echo(f"\nDone. {len(summaries)} section(s) completed.")
        click.echo(_render_plan_tree(plan))

    def _repl_eval(self, args: list[str]) -> None:
        """Re-evaluate a specific section: eval <section_id>."""
        if not args:
            click.echo("Usage: eval <section_id>")
            return
        section_id = args[0]

        store = self._get_plan_store()
        plan = store.load()
        if plan is None:
            click.echo("No plan found.")
            return

        section = _find_section_by_id(plan.sections, section_id)
        if section is None:
            click.echo(f"Section '{section_id}' not found.")
            return

        sm = self._get_session_manager()
        budget = self.config.fallback_chunk_tokens
        try:
            evaluation = sm._evaluator.evaluate_plan_section(section, budget)
        except Exception as exc:
            click.echo(f"Evaluation failed: {exc}")
            return

        section.evaluation = evaluation
        store.save(plan)

        click.echo(f"Section {section_id}: {evaluation.recommendation}")
        click.echo(f"  Score        : {evaluation.score:.2f}")
        click.echo(f"  Completeness : {evaluation.completeness:.2f}")
        click.echo(f"  Granularity  : {evaluation.granularity:.2f}")
        click.echo(f"  Consistency  : {evaluation.consistency:.2f}")
        click.echo(f"  Critique     : {evaluation.critique}")

    def _repl_retry(self, args: list[str]) -> None:
        """Mark a BLOCKED/NEEDS_REVISION section PENDING and re-dispatch: retry <section_id>."""
        if not args:
            click.echo("Usage: retry <section_id>")
            return
        section_id = args[0]

        store = self._get_plan_store()
        plan = store.load()
        if plan is None:
            click.echo("No plan found.")
            return

        section = _find_section_by_id(plan.sections, section_id)
        if section is None:
            click.echo(f"Section '{section_id}' not found.")
            return

        if section.status not in ("BLOCKED", "NEEDS_REVISION", "FAILED"):
            click.echo(
                f"Section '{section_id}' has status '{section.status}'. "
                "Only BLOCKED / NEEDS_REVISION sections can be retried."
            )
            return

        section.status = "PENDING"
        store.save(plan)
        click.echo(f"Section '{section_id}' reset to PENDING.")

        self._ensure_index()
        sm = self._get_session_manager()
        executor = ToolExecutor(
            project_root=str(self.current_dir),
            graph=self._graph,
            index_text=self._compact_index or "",
        )
        try:
            summaries = sm.dispatch_workers(plan, executor)
        except KeyboardInterrupt:
            click.echo("\nInterrupted.")
            return
        click.echo(f"\nDone. {len(summaries)} section(s) run.")
        click.echo(_render_plan_tree(plan))

    def _repl_tools(self, args: list[str]) -> None:
        """List all available worker tools with descriptions."""
        executor = ToolExecutor(
            project_root=str(self.current_dir),
            graph=self._graph,
            index_text=self._compact_index or "",
        )
        click.echo(executor.get_descriptions())

    def _repl_summary(self, args: list[str]) -> None:
        """Show a compact table of all WorkerSummary entries."""
        store = self._get_plan_store()
        plan = store.load()
        if plan is None:
            click.echo(
                "No plan found. Use 'plan new <task>' to start a session."
            )
            return
        click.echo(_render_summary_table(plan))

    def _repl_diff(self, args: list[str]) -> None:
        """Show git diff for files changed by a section: diff <section_id>."""
        if not args:
            click.echo("Usage: diff <section_id>")
            return
        section_id = args[0]

        store = self._get_plan_store()
        plan = store.load()
        if plan is None:
            click.echo("No plan found.")
            return

        section = _find_section_by_id(plan.sections, section_id)
        if section is None:
            click.echo(f"Section '{section_id}' not found.")
            return

        if section.worker_summary is None or not section.worker_summary.files_changed:
            click.echo(f"Section '{section_id}' has no recorded file changes.")
            return

        files = section.worker_summary.files_changed
        result = subprocess.run(
            ["git", "diff", "--"] + files,
            capture_output=True,
            text=True,
            cwd=str(self.current_dir),
        )
        output = result.stdout or result.stderr
        if output:
            click.echo(output)
        else:
            click.echo(
                "(no diff — files may be untracked or unchanged in git)"
            )

    # ── lazy orchestrator ──────────────────────────────────────────────────

    def _get_orchestrator(self) -> PRISMOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = PRISMOrchestrator(config=self.config)
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


# ── Phase F — plan rendering helpers ─────────────────────────────────────────

_STATUS_GLYPHS = {
    "COMPLETE":       "✓",
    "IN_PROGRESS":    "⟳",
    "PENDING":        "·",
    "NEEDS_REVISION": "↺",
    "BLOCKED":        "✗",
    "EVALUATING":     "~",
}


def _find_section_by_id(
    sections: list[PlanSection], section_id: str
) -> Optional[PlanSection]:
    """Recursively search plan sections by id. Returns None if not found."""
    for section in sections:
        if section.id == section_id:
            return section
        if section.sub_sections:
            found = _find_section_by_id(section.sub_sections, section_id)
            if found is not None:
                return found
    return None


def _render_plan_tree(plan: SessionPlan) -> str:
    """
    Render a SessionPlan as a Unicode tree string.

    Example output:
      Session: "Add JWT auth" [EXECUTING] (13b-local)
      ├── s1  Auth module             [COMPLETE] ✓  "Added validate_token"  tests:PASS
      └── s2  Router integration      [PENDING] ·
    """
    lines: list[str] = []
    preset = f" ({plan.model_preset})" if getattr(plan, "model_preset", None) else ""
    lines.append(f'Session: "{plan.task}" [{plan.status}]{preset}')

    def _section_line(section: PlanSection) -> str:
        icon = _STATUS_GLYPHS.get(section.status, "?")
        annotation = ""
        if section.worker_summary:
            ws = section.worker_summary
            brief = ws.brief[:40] + "…" if len(ws.brief) > 40 else ws.brief
            test_tag = ""
            if ws.test_result:
                label = "PASS" if ws.test_result == "PASS" else "FAIL"
                test_tag = f"  tests:{label}"
            annotation = f'  "{brief}"{test_tag}'
        title = section.title[:32]
        return (
            f"{section.id:<8}  {title:<32}"
            f"  [{section.status}] {icon}{annotation}"
        )

    def _render_sections(sections: list[PlanSection], prefix: str) -> None:
        for i, section in enumerate(sections):
            is_last = i == len(sections) - 1
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + _section_line(section))
            if section.sub_sections:
                extension = "    " if is_last else "│   "
                _render_sections(section.sub_sections, prefix + extension)

    _render_sections(plan.sections, "")
    return "\n".join(lines)


def _render_summary_table(plan: SessionPlan) -> str:
    """
    Render all leaf section WorkerSummary entries as a compact table.

    Example output:
      ID        STATUS    BRIEF                                      TEST   CHANGED
      s1        COMPLETE  Added validate_token to Auth class         PASS   src/auth.py
      s2        PENDING   -                                          -      -
    """
    header = f"{'ID':<10}  {'STATUS':<14}  {'BRIEF':<44}  {'TEST':<6}  CHANGED"
    sep = "─" * (len(header) + 6)
    rows: list[str] = [header, sep]

    for section in plan.all_leaf_sections():
        sid = section.id
        status = section.status
        if section.worker_summary:
            ws = section.worker_summary
            brief = ws.brief[:44] if len(ws.brief) <= 44 else ws.brief[:41] + "…"
            test = "PASS" if ws.test_result == "PASS" else (
                "FAIL" if ws.test_result else "-"
            )
            if not ws.files_changed:
                changed = "-"
            elif len(ws.files_changed) == 1:
                changed = ws.files_changed[0]
            else:
                changed = f"{ws.files_changed[0]} (+{len(ws.files_changed) - 1})"
        else:
            brief = "-"
            test = "-"
            changed = "-"
        rows.append(
            f"{sid:<10}  {status:<14}  {brief:<44}  {test:<6}  {changed}"
        )

    return "\n".join(rows)


if __name__ == "__main__":
    main()
