"""
WorkerAgent — executes a PlanSection via an LLM tool-call loop.

Lifecycle:
1. Build system + user messages from WorkerContext.
2. Run complete_with_tools() with a _TestAwareExecutor wrapper.
3. _TestAwareExecutor automatically runs tests after write_file / apply_diff /
   patch_file and appends the result to the tool output — the LLM sees failures
   and self-corrects within the tool loop (bounded by max_tool_rounds).
4. Extract WorkerSummary from executor.last_summary (the write_summary call),
   or synthesize a PARTIAL summary from tool call history if write_summary was
   never called.
"""
from __future__ import annotations

from typing import Optional

from ..config import PRISMConfig
from ..llm.client import LLMClient
from ..plan.plan_model import PlanSection, WorkerSummary
from ..tools.executor import ToolExecutor, ToolResult
from .templates import detect_task_type, get_template
from .worker_context import WorkerContext


# ── System / user prompt templates ───────────────────────────────────────────

_SYSTEM_PREFIX = """\
You are a worker agent in a coding assistant pipeline.
Your job is to complete exactly one section of a larger plan.

Rules:
- Work only within the scope_files and scope_symbols listed in the section.
- Make the minimum change needed to fulfil the section goal.
- Read files before editing them (use read_file).
- After every file edit, tests are run automatically — fix failures before continuing.
- You MUST call write_summary before finishing to record your work.
"""

_SECTION_HEADER = """\
# Section {section_id}: {title}
Goal: {description}
Scope files: {scope_files}
Scope symbols: {scope_symbols}\
"""

_USER_TEMPLATE = """\
{section_header}

## Global context
{global_prefix}

## Codebase index (section scope)
{sub_index}

## Source files
{source_slices}

## Prior completed sections (relevant to this section)
{prior_text}

Begin work. Use tools to read, edit, and verify files. Call write_summary when done.\
"""


class WorkerAgent:
    """
    Executes a single PlanSection via an LLM tool-call loop.

    Args:
        section:  The PlanSection to execute.
        context:  Assembled WorkerContext (from WorkerContextBuilder.build).
        llm:      Initialised LLMClient.
        executor: ToolExecutor bound to the project root.
        config:   PRISMConfig (uses active_synthesis_model, max_tool_rounds).
    """

    def __init__(
        self,
        section: PlanSection,
        context: WorkerContext,
        llm: LLMClient,
        executor: ToolExecutor,
        config: PRISMConfig,
    ) -> None:
        self._section = section
        self._context = context
        self._llm = llm
        self._executor = _TestAwareExecutor(executor)
        self._config = config

    def run(self) -> WorkerSummary:
        """Execute the section and return a WorkerSummary."""
        task_type = detect_task_type(self._section.description)
        # Use first 200 chars of global_prefix as style-note hint for template
        style_hint = self._context.global_prefix[:200].replace("\n", " ")
        template = get_template(task_type, style_hint)

        system = _SYSTEM_PREFIX + "\n\n" + template
        if self._context.tool_specs:
            system += "\n\n" + self._context.tool_specs

        section_header = _SECTION_HEADER.format(
            section_id=self._section.id,
            title=self._section.title,
            description=self._section.description,
            scope_files=self._section.scope_files or [],
            scope_symbols=self._section.scope_symbols or [],
        )

        user = _USER_TEMPLATE.format(
            section_header=section_header,
            global_prefix=self._context.global_prefix or "(none)",
            sub_index=self._context.sub_index or "(none)",
            source_slices=self._context.source_slices or "(none)",
            prior_text=self._context.prior_text or "(none)",
        )

        _final_text, all_tool_calls = self._llm.complete_with_tools(
            messages=[{"role": "user", "content": user}],
            tools=self._executor.get_schemas(),
            executor=self._executor,
            model=self._config.active_synthesis_model,
            system=system,
            max_tokens=4096,
            max_tool_rounds=self._config.max_tool_rounds,
            purpose=f"worker:{self._section.id} {self._section.title}",
        )

        return self._extract_summary(all_tool_calls, _final_text)

    # ── Summary extraction ────────────────────────────────────────────────────

    def _extract_summary(
        self, tool_calls: list[dict], final_text: str
    ) -> WorkerSummary:
        """
        Build WorkerSummary from write_summary payload, or synthesize if absent.
        """
        last = self._executor.last_summary
        if last is not None:
            return WorkerSummary(
                section_id=last.get("section_id", self._section.id),
                status=last.get("status", "COMPLETE"),
                brief=last.get("brief", "Work completed.")[:150],
                detail=last.get("detail", "")[:500],
                files_changed=list(last.get("files_changed", [])),
                symbols_touched=list(last.get("symbols_touched", [])),
                interfaces_changed=list(last.get("interfaces_changed", [])),
                new_symbols=list(last.get("new_symbols", [])),
                assumptions=list(last.get("assumptions", [])),
                blockers=list(last.get("blockers", [])),
                token_cost=0,
                test_result=self._executor.last_test_result,
            )

        # Synthesize from tool call history — write_summary was not called
        files_changed: list[str] = []
        for call in tool_calls:
            if call["tool_name"] in ("write_file", "apply_diff", "patch_file"):
                path = call.get("args", {}).get("path", "")
                if path and path not in files_changed:
                    files_changed.append(path)

        return WorkerSummary(
            section_id=self._section.id,
            status="PARTIAL",
            brief=_synthesize_brief(tool_calls, final_text)[:150],
            detail=(
                f"write_summary not called. Synthesized from "
                f"{len(tool_calls)} tool calls."
            )[:500],
            files_changed=files_changed,
            symbols_touched=[],
            interfaces_changed=[],
            new_symbols=[],
            assumptions=["write_summary not called — summary may be incomplete"],
            blockers=[],
            token_cost=0,
            test_result=self._executor.last_test_result,
        )


def _synthesize_brief(tool_calls: list[dict], final_text: str) -> str:
    """Generate a best-effort brief from tool call history."""
    write_calls = [
        c for c in tool_calls
        if c["tool_name"] in ("write_file", "apply_diff", "patch_file")
    ]
    if write_calls:
        paths = list(dict.fromkeys(
            c.get("args", {}).get("path", "?") for c in write_calls
        ))
        return f"Modified {', '.join(paths[:3])}"
    if final_text:
        first = final_text.strip().split(".")[0]
        return first[:150] if first else "Work completed."
    return "Work completed (no file changes detected)."


# ── _TestAwareExecutor ────────────────────────────────────────────────────────

class _TestAwareExecutor:
    """
    Wraps ToolExecutor to run tests automatically after file mutations.

    When write_file, apply_diff, or patch_file succeeds, this wrapper
    immediately calls run_tests on the modified file and appends the result
    to the tool call output. The LLM sees the test outcome and can
    self-correct within the tool loop (up to max_tool_rounds).
    """

    _MUTATION_TOOLS = frozenset({"write_file", "apply_diff", "patch_file"})

    def __init__(self, executor: ToolExecutor) -> None:
        self._exec = executor
        self._last_test_result: Optional[str] = None

    def execute(self, tool_name: str, args: dict) -> ToolResult:
        result = self._exec.execute(tool_name, args)

        if tool_name in self._MUTATION_TOOLS and result.success:
            path = args.get("path")
            if path:
                test_result = self._exec.execute("run_tests", {"paths": [path]})
                test_out = test_result.to_str()

                if test_out.startswith("EXIT 0"):
                    self._last_test_result = "PASS"
                    suffix = f"\n\nTests: PASS\n{test_out[:400]}"
                else:
                    self._last_test_result = f"FAIL: {test_out[:250]}"
                    suffix = (
                        f"\n\nTests FAILED:\n{test_out[:500]}\n"
                        "Fix the issue and re-run tests before calling write_summary."
                    )
                result = ToolResult(output=result.output + suffix, error=result.error)

        return result

    def get_schemas(self) -> list[dict]:
        return self._exec.get_schemas()

    def get_descriptions(self) -> str:
        return self._exec.get_descriptions()

    @property
    def last_summary(self) -> Optional[dict]:
        return self._exec.last_summary

    @property
    def last_test_result(self) -> Optional[str]:
        return self._last_test_result
