"""
Provider-agnostic LLM client with tiktoken token counting and retry logic.

Supports Anthropic Claude by default. Set ANTHROPIC_API_KEY in environment.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..tools.executor import ToolExecutor

# tiktoken for accurate token counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# Anthropic client
try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class TokenCounter:
    """
    Tiktoken-accurate token counter using cl100k_base encoding.
    Falls back to 4-chars/token approximation if tiktoken is unavailable.
    """

    def __init__(self, encoding: str = "cl100k_base"):
        self._encoding = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(encoding)
            except Exception:
                pass

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # Fallback: ~4 chars per token
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self.count(block.get("text", ""))
        return total


class LLMClient:
    """
    Provider-agnostic LLM client.

    Usage:
        client = LLMClient(config, logger)
        response = client.complete(
            model='claude-haiku-4-5-20251001',
            messages=[{'role': 'user', 'content': '...'}],
            system='You are ...',
            max_tokens=2048
        )
    """

    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self.token_counter = TokenCounter(config.tokenizer)
        self._client = self._build_client()

    def _build_client(self):
        provider = getattr(self._config, "provider", "anthropic")

        if provider == "ollama":
            if not _OPENAI_AVAILABLE:
                self._logger.warn("openai_unavailable", hint="pip install openai")
                return None
            return openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required by the client but ignored by Ollama
            )

        # Default: Anthropic
        if not _ANTHROPIC_AVAILABLE:
            self._logger.warn("anthropic_unavailable", hint="pip install anthropic")
            return None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._logger.warn("api_key_missing", hint="Set ANTHROPIC_API_KEY")
            return None
        return anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Call the LLM and return the text response.
        Retries on transient errors with exponential backoff.
        """
        model = model or self._config.active_synthesis_model
        provider = getattr(self._config, "provider", "anthropic")

        if self._client is None:
            raise RuntimeError(
                "LLM client not available. Check ANTHROPIC_API_KEY and "
                "install the 'anthropic' package."
            )

        prompt_tokens = self.token_counter.count_messages(messages)
        self._logger.info(
            "llm_call_start",
            model=model,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                t0 = time.time()

                if provider == "ollama":
                    all_messages = messages
                    if system:
                        all_messages = [{"role": "system", "content": system}] + messages
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=all_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    text = response.choices[0].message.content

                else:  # anthropic
                    kwargs = dict(model=model, max_tokens=max_tokens,
                                temperature=temperature, messages=messages)
                    if system:
                        kwargs["system"] = system
                    response = self._client.messages.create(**kwargs)
                    text = response.content[0].text

                self._logger.info(
                    "llm_call_complete",
                    model=model,
                    prompt_tokens=prompt_tokens
                )
                return text

            except Exception as e:
                last_error = e
                wait = retry_delay * (2 ** attempt)
                self._logger.warn(
                    "llm_call_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    wait_s=wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        self._logger.error(
            "llm_call_failed",
            model=model,
            error=str(last_error),
            attempts=max_retries,
        )
        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts: {last_error}"
        ) from last_error

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        executor: "ToolExecutor",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        max_tool_rounds: int = 20,
    ) -> tuple[str, list[dict]]:
        """
        Tool-calling loop: call LLM → execute tool calls → inject results → repeat.

        Args:
            messages:        Initial conversation messages.
            tools:           Generic tool schemas:
                             [{"name": str, "description": str, "parameters": dict}]
            executor:        ToolExecutor instance to execute tool calls.
            model:           Model override (defaults to active_synthesis_model).
            system:          System prompt.
            max_tokens:      Max tokens per LLM call.
            max_tool_rounds: Hard cap on tool-call iterations.

        Returns:
            (final_text, all_tool_calls) where all_tool_calls is a list of
            {"tool_name": str, "args": dict, "result": str} dicts.
        """
        from .capability import detect_capabilities

        model = model or self._config.active_synthesis_model
        provider = getattr(self._config, "provider", "anthropic")
        caps = detect_capabilities(model, provider)

        if caps.supports_tool_calls:
            if provider == "anthropic":
                return self._tool_loop_anthropic(
                    messages, tools, executor, model, system,
                    max_tokens, max_tool_rounds,
                )
            else:
                return self._tool_loop_openai(
                    messages, tools, executor, model, system,
                    max_tokens, max_tool_rounds,
                )
        else:
            return self._tool_loop_json_fallback(
                messages, tools, executor, model, system,
                max_tokens, max_tool_rounds,
            )

    # ── Native tool-call loops ────────────────────────────────────────────────

    def _tool_loop_anthropic(
        self,
        messages: list[dict],
        tools: list[dict],
        executor: "ToolExecutor",
        model: str,
        system: Optional[str],
        max_tokens: int,
        max_rounds: int,
    ) -> tuple[str, list[dict]]:
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

        current_messages = list(messages)
        all_tool_calls: list[dict] = []

        for _ in range(max_rounds):
            kwargs: dict = dict(
                model=model,
                max_tokens=max_tokens,
                messages=current_messages,
                tools=anthropic_tools,
            )
            if system:
                kwargs["system"] = system

            response = self._client.messages.create(**kwargs)

            tool_use_blocks = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use"
            ]
            text_blocks = [
                b for b in response.content
                if getattr(b, "type", None) == "text"
            ]

            if not tool_use_blocks:
                final_text = text_blocks[0].text if text_blocks else ""
                return final_text, all_tool_calls

            # Append assistant message (preserve all content blocks as dicts)
            current_messages.append({
                "role": "assistant",
                "content": [_content_block_to_dict(b) for b in response.content],
            })

            # Execute each tool call and collect results
            tool_results = []
            for block in tool_use_blocks:
                result = executor.execute(block.name, block.input)
                result_text = result.to_str()
                all_tool_calls.append({
                    "tool_name": block.name,
                    "args": block.input,
                    "result": result_text,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            current_messages.append({"role": "user", "content": tool_results})

        return f"(stopped after {max_rounds} tool rounds)", all_tool_calls

    def _tool_loop_openai(
        self,
        messages: list[dict],
        tools: list[dict],
        executor: "ToolExecutor",
        model: str,
        system: Optional[str],
        max_tokens: int,
        max_rounds: int,
    ) -> tuple[str, list[dict]]:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

        all_messages = list(messages)
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        all_tool_calls: list[dict] = []

        for _ in range(max_rounds):
            response = self._client.chat.completions.create(
                model=model,
                messages=all_messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=max_tokens,
            )

            msg = response.choices[0].message
            tc = msg.tool_calls

            if not tc:
                return msg.content or "", all_tool_calls

            # Append assistant message with tool_calls
            all_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in tc
                ],
            })

            # Execute each tool call
            for call in tc:
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result = executor.execute(call.function.name, args)
                result_text = result.to_str()
                all_tool_calls.append({
                    "tool_name": call.function.name,
                    "args": args,
                    "result": result_text,
                })
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_text,
                })

        return f"(stopped after {max_rounds} tool rounds)", all_tool_calls

    # ── JSON fallback loop (models without native tool calls) ─────────────────

    def _tool_loop_json_fallback(
        self,
        messages: list[dict],
        tools: list[dict],
        executor: "ToolExecutor",
        model: str,
        system: Optional[str],
        max_tokens: int,
        max_rounds: int,
    ) -> tuple[str, list[dict]]:
        tool_desc = _build_tool_descriptions(tools)
        fallback_system = (
            (system.rstrip() + "\n\n" if system else "")
            + tool_desc
            + "\n\n"
            "To use a tool, output ONLY a JSON block (no preamble):\n"
            '{"tool": "<name>", "args": {<params>}}\n'
            "After seeing the result, continue your work. "
            "Call write_summary when done."
        )

        current_messages = list(messages)
        all_tool_calls: list[dict] = []

        for _ in range(max_rounds):
            text = self.complete(
                messages=current_messages,
                model=model,
                system=fallback_system,
                max_tokens=max_tokens,
            )

            call = _extract_json_tool_call(text)
            if call is None:
                return text, all_tool_calls

            tool_name = call.get("tool", "")
            args = call.get("args", {})

            result = executor.execute(tool_name, args)
            result_text = result.to_str()
            all_tool_calls.append({
                "tool_name": tool_name,
                "args": args,
                "result": result_text,
            })

            current_messages.append({"role": "assistant", "content": text})
            current_messages.append({
                "role": "user",
                "content": f"Tool result:\n{result_text}",
            })

        return f"(stopped after {max_rounds} tool rounds)", all_tool_calls

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    @property
    def available(self) -> bool:
        return self._client is not None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _content_block_to_dict(block: Any) -> dict:
    """Convert an Anthropic SDK content block object to a plain dict."""
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {"type": btype or "unknown"}


def _build_tool_descriptions(tools: list[dict]) -> str:
    """Build compact text description of tools for JSON-fallback models."""
    lines = ['Available tools — call with: {"tool": "<name>", "args": {...}}\n']
    for t in tools:
        props = t.get("parameters", {}).get("properties", {})
        required = t.get("parameters", {}).get("required", [])
        param_parts = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any")
            suffix = "" if pname in required else "?"
            param_parts.append(f"{pname}{suffix}:{ptype}")
        params = ", ".join(param_parts)
        lines.append(f"  {t['name']}({params}) — {t['description']}")
    return "\n".join(lines)


def _extract_json_tool_call(text: str) -> Optional[dict]:
    """
    Extract {"tool": ..., "args": ...} from an LLM response.
    Handles markdown code fences and bare JSON objects.
    """
    # Try ```json ... ``` fence first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON object containing "tool" key
    m = re.search(r'\{\s*"tool"\s*:[\s\S]+?\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None
