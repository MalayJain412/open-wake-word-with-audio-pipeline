"""OpenAI-compatible audio transcription client for a local llama.cpp server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from llamacpp_config import (
    LLAMA_ASR_PROMPT,
    LLAMA_BASE_URL,
    LLAMA_MODEL_NAME,
    LLAMA_TRANSCRIPTIONS_URL,
)


class LlamaCppError(RuntimeError):
    pass


def server_is_ready(timeout: float = 2.0) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            response = requests.get(f"{LLAMA_BASE_URL}{path}", timeout=timeout)
            if response.ok:
                return True
        except requests.RequestException:
            continue
    return False


def transcribe_wav(path: Path, timeout: float = 120.0) -> str:
    if not path.is_file():
        raise LlamaCppError(f"Audio file does not exist: {path}")

    try:
        with path.open("rb") as audio_file:
            response = requests.post(
                LLAMA_TRANSCRIPTIONS_URL,
                files={"file": ("command.wav", audio_file, "audio/wav")},
                data={
                    "model": LLAMA_MODEL_NAME,
                    "prompt": LLAMA_ASR_PROMPT,
                    "response_format": "json",
                },
                timeout=timeout,
            )
        response.raise_for_status()
    except requests.RequestException as error:
        raise LlamaCppError(f"llama.cpp transcription request failed: {error}") from error

    transcription = transcription_text(response)
    if not transcription:
        raise LlamaCppError("llama.cpp returned an empty transcription.")
    return transcription


def transcription_text(response: Any) -> str:
    content_type = str(response.headers.get("Content-Type", "")).lower()
    body = response.text.strip()
    payload: Any = None

    if "json" in content_type or body.startswith(("{", "[")):
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            payload = None

    if isinstance(payload, dict):
        for key in ("text", "transcript", "transcription"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            first = choices[0]
            value = first.get("text")
            if isinstance(value, str):
                return value.strip()
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()

    if isinstance(payload, str):
        return payload.strip()
    return body
