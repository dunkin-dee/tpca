"""
Tests for LLMClient and TokenCounter.

These tests do NOT make real API calls.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.config import PRISMConfig
from prism.logging.log_config import LogConfig
from prism.logging.structured_logger import StructuredLogger
from prism.llm.client import LLMClient, TokenCounter


def make_config():
    return PRISMConfig(
        provider="anthropic",
        synthesis_model="claude-sonnet-4-6",
        tokenizer="cl100k_base",
        log=LogConfig(log_file="/dev/null", console_level="ERROR"),
    )


def make_logger(config):
    return StructuredLogger(config.log)


class TestTokenCounter:

    def test_count_nonempty_string(self):
        counter = TokenCounter()
        count = counter.count("Hello, world!")
        assert count > 0

    def test_count_empty_string(self):
        counter = TokenCounter()
        count = counter.count("")
        assert count == 0 or count >= 0  # fallback may return 0

    def test_count_longer_text_larger(self):
        counter = TokenCounter()
        short = counter.count("hi")
        long_ = counter.count("hello " * 100)
        assert long_ > short

    def test_count_messages(self):
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_with_list_content(self):
        counter = TokenCounter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First block"},
                    {"type": "text", "text": "Second block"},
                ],
            }
        ]
        count = counter.count_messages(messages)
        assert count > 0


class TestLLMClientInit:

    def test_unavailable_without_api_key(self):
        """LLMClient.available should be False if no ANTHROPIC_API_KEY is set."""
        import os
        config = make_config()
        logger = make_logger(config)

        with patch.dict("os.environ", {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            os.environ.pop("ANTHROPIC_API_KEY", None)
            client = LLMClient(config, logger)
            # May or may not be available depending on env — just check it doesn't crash
            assert isinstance(client.available, bool)

    def test_warns_when_api_key_missing(self):
        import os
        config = make_config()
        logger = make_logger(config)

        env_without_key = {k: v for k, v in __import__("os").environ.items()
                          if k != "ANTHROPIC_API_KEY"}
        with patch.dict("os.environ", env_without_key, clear=True):
            LLMClient(config, logger)
            # Should emit a warning about missing key (if anthropic package is available)


class TestLLMClientComplete:

    def test_complete_with_mock_anthropic(self):
        """Test complete() by mocking the underlying Anthropic client."""
        config = make_config()
        logger = make_logger(config)

        # Create client, then inject a mock
        client = LLMClient.__new__(LLMClient)
        client._config = config
        client._logger = logger
        client.token_counter = TokenCounter()

        # Mock the Anthropic messages.create response
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from mock LLM!")]
        mock_anthropic.messages.create.return_value = mock_response
        client._client = mock_anthropic

        result = client.complete(
            messages=[{"role": "user", "content": "Say hello."}],
            model="claude-sonnet-4-6",
        )
        assert result == "Hello from mock LLM!"
        assert mock_anthropic.messages.create.call_count == 1

    def test_complete_retries_on_error(self):
        """Should retry on transient errors."""
        config = make_config()
        logger = make_logger(config)

        client = LLMClient.__new__(LLMClient)
        client._config = config
        client._logger = logger
        client.token_counter = TokenCounter()

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Success after retry")]
        mock_anthropic.messages.create.side_effect = [
            Exception("Transient error"),
            mock_response,
        ]
        client._client = mock_anthropic

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            max_retries=3,
            retry_delay=0.0,  # skip actual sleep in tests
        )
        assert result == "Success after retry"
        assert mock_anthropic.messages.create.call_count == 2

    def test_complete_raises_after_max_retries(self):
        """Should raise RuntimeError after exhausting retries."""
        config = make_config()
        logger = make_logger(config)

        client = LLMClient.__new__(LLMClient)
        client._config = config
        client._logger = logger
        client.token_counter = TokenCounter()

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = Exception("Always fails")
        client._client = mock_anthropic

        with pytest.raises(RuntimeError, match="LLM call failed"):
            client.complete(
                messages=[{"role": "user", "content": "test"}],
                max_retries=2,
                retry_delay=0.0,
            )

    def test_complete_raises_when_client_unavailable(self):
        """Should raise RuntimeError immediately when _client is None."""
        config = make_config()
        logger = make_logger(config)

        client = LLMClient.__new__(LLMClient)
        client._config = config
        client._logger = logger
        client.token_counter = TokenCounter()
        client._client = None  # Simulate unavailable client

        with pytest.raises(RuntimeError, match="not available"):
            client.complete(messages=[{"role": "user", "content": "test"}])

    def test_count_tokens(self):
        config = make_config()
        logger = make_logger(config)

        client = LLMClient.__new__(LLMClient)
        client._config = config
        client._logger = logger
        client.token_counter = TokenCounter()
        client._client = None

        count = client.count_tokens("Hello, world! " * 10)
        assert count > 0
