"""
Test suite for call_api.py (All-in-One API Caller)
Based on template tests from tests/test_template.py

Run from the project folder:
    pytest tests/test_call_api.py -v
or
    python -m unittest tests/test_call_api.py

All external API calls are mocked — no real API keys or internet connection required.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Locate directories relative to this test file
CALL_API_PATH = "call_api.py"


def _load_module(path: str, unique_name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target file not found at: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Dynamically load call_api module
_m = _load_module(CALL_API_PATH, "call_api")

call_openai = getattr(_m, "call_openai")
call_gemini = getattr(_m, "call_gemini")
call_anthropic = getattr(_m, "call_anthropic")
stream_openai = getattr(_m, "stream_openai")
streaming_chatbot = getattr(_m, "streaming_chatbot")


# ---------------------------------------------------------------------------
# Mock Generators
# ---------------------------------------------------------------------------
def _make_openai_response(text: str = "Hello from OpenAI"):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 25
    resp.usage_metadata = usage
    return resp


def _make_anthropic_response(text: str = "Hello from Anthropic"):
    resp = MagicMock()
    content_part = MagicMock()
    content_part.text = text
    resp.content = [content_part]
    usage = MagicMock()
    usage.input_tokens = 15
    usage.output_tokens = 30
    resp.usage = usage
    return resp


def _make_openai_stream_chunks(text: str = "Hello from Stream"):
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = text
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = []
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 10
    chunk2.usage = usage

    return [chunk1, chunk2]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestCallOpenAI(unittest.TestCase):

    def test_openai_returns_tuple_structure(self):
        with patch.object(_m, "OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("OpenAI Mock Response")

            text, latency, usage = call_openai("Hello")

            self.assertEqual(text, "OpenAI Mock Response")
            self.assertIsInstance(latency, float)
            self.assertGreaterEqual(latency, 0.0)
            self.assertEqual(usage["input_tokens"], 10)
            self.assertEqual(usage["output_tokens"], 20)


class TestCallGemini(unittest.TestCase):

    def test_gemini_returns_tuple_structure(self):
        with patch.object(_m.genai, "Client") as MockGenAIClient:
            mock_client = MagicMock()
            MockGenAIClient.return_value = mock_client
            mock_client.models.generate_content.return_value = _make_gemini_response("Gemini Mock Response")

            text, latency, usage = call_gemini("Hello")

            self.assertEqual(text, "Gemini Mock Response")
            self.assertIsInstance(latency, float)
            self.assertGreaterEqual(latency, 0.0)
            self.assertEqual(usage["input_tokens"], 12)
            self.assertEqual(usage["output_tokens"], 25)


class TestCallAnthropic(unittest.TestCase):

    def test_anthropic_returns_tuple_structure(self):
        with patch.object(_m.anthropic, "Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.return_value = _make_anthropic_response("Claude Mock Response")

            text, latency, usage = call_anthropic("Hello")

            self.assertEqual(text, "Claude Mock Response")
            self.assertIsInstance(latency, float)
            self.assertGreaterEqual(latency, 0.0)
            self.assertEqual(usage["input_tokens"], 15)
            self.assertEqual(usage["output_tokens"], 30)


class TestStreamOpenAI(unittest.TestCase):

    def test_stream_openai_returns_tuple_structure(self):
        with patch.object(_m, "OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_stream_chunks("Streaming Mock Response")

            text, latency, usage, speed = stream_openai("Hello")

            self.assertEqual(text, "Streaming Mock Response")
            self.assertIsInstance(latency, float)
            self.assertGreaterEqual(latency, 0.0)
            self.assertIn("input_tokens", usage)
            self.assertIn("output_tokens", usage)
            self.assertIsInstance(speed, float)
            self.assertGreaterEqual(speed, 0.0)


class TestStreamingChatbot(unittest.TestCase):

    @patch("builtins.input", side_effect=["quit"])
    def test_exits_cleanly(self, mock_input):
        # Ensure it quits cleanly without entering infinite loop
        try:
            streaming_chatbot()
        except SystemExit:
            pass


if __name__ == "__main__":
    unittest.main()
