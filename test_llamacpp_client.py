from __future__ import annotations

import unittest

from llamacpp_client import transcription_text
from llamacpp_config import llama_server_command


class FakeResponse:
    def __init__(self, text, content_type="text/plain", payload=None):
        self.text = text
        self.headers = {"Content-Type": content_type}
        self.payload = payload

    def json(self):
        if self.payload is None:
            raise ValueError("not JSON")
        return self.payload


class LlamaCppClientTests(unittest.TestCase):
    def test_plain_text_transcription(self):
        response = FakeResponse("open chrome")
        self.assertEqual(transcription_text(response), "open chrome")

    def test_json_text_transcription(self):
        response = FakeResponse(
            '{"text":"open chrome"}',
            "application/json",
            {"text": "open chrome"},
        )
        self.assertEqual(transcription_text(response), "open chrome")

    def test_openai_choice_transcription(self):
        response = FakeResponse(
            "{}",
            "application/json",
            {"choices": [{"message": {"content": "open chrome"}}]},
        )
        self.assertEqual(transcription_text(response), "open chrome")

    def test_server_command_uses_model_and_projector(self):
        command = llama_server_command()
        self.assertIn("--model", command)
        self.assertIn("--mmproj", command)
        self.assertIn("--reasoning", command)
        self.assertIn("off", command)


if __name__ == "__main__":
    unittest.main()
