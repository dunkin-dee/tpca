"""
Provider-agnostic LLM client with tiktoken token counting and retry logic.

Supports Anthropic Claude by default. Set ANTHROPIC_API_KEY in environment.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Any

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
        """Build the underlying API client."""
        if not _ANTHROPIC_AVAILABLE:
            self._logger.warn(
                "anthropic_unavailable",
                hint="Install with: pip install anthropic",
            )
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._logger.warn(
                "api_key_missing",
                hint="Set ANTHROPIC_API_KEY environment variable",
            )
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
        model = model or self._config.synthesis_model

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
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                t0 = time.time()
                response = self._client.messages.create(**kwargs)
                elapsed_ms = int((time.time() - t0) * 1000)

                text = response.content[0].text
                output_tokens = self.token_counter.count(text)

                self._logger.info(
                    "llm_call_complete",
                    model=model,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    elapsed_ms=elapsed_ms,
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

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    @property
    def available(self) -> bool:
        return self._client is not None
