"""Portable llama.cpp paths and runtime settings for the Sentry audio pipeline."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


LLAMA_CPP_DIR = environment_path("SENTRY_LLAMA_CPP_DIR", ROOT / "llama.cpp")
LLAMA_SERVER_EXE = environment_path(
    "SENTRY_LLAMA_SERVER_EXE",
    LLAMA_CPP_DIR / "llama-server.exe",
)
LLAMA_MODEL_PATH = environment_path(
    "SENTRY_LLAMA_MODEL",
    ROOT / "models" / "LFM3-Omni-2B-Q4_K_M.gguf",
)
LLAMA_MMPROJ_PATH = environment_path(
    "SENTRY_LLAMA_MMPROJ",
    ROOT / "models" / "mmproj-LFM3-Omni-2B-Q4_K_M.gguf",
)

LLAMA_HOST = os.environ.get("SENTRY_LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("SENTRY_LLAMA_PORT", "8081"))
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
LLAMA_TRANSCRIPTIONS_URL = f"{LLAMA_BASE_URL}/v1/audio/transcriptions"
LLAMA_MODEL_NAME = os.environ.get("SENTRY_LLAMA_MODEL_NAME", "LFM3-Omni-2B")
LLAMA_ASR_PROMPT = os.environ.get("SENTRY_LLAMA_ASR_PROMPT", "Perform ASR in English.")

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = int(os.environ.get("SENTRY_FRONTEND_PORT", "8765"))
FRONTEND_ORIGIN = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"

WAKE_WORD = os.environ.get("SENTRY_WAKE_WORD", "alexa")
WAKE_THRESHOLD = float(os.environ.get("SENTRY_WAKE_THRESHOLD", "0.5"))
SILENCE_DURATION = float(os.environ.get("SENTRY_SILENCE_DURATION", "2.0"))
SILENCE_RMS_THRESHOLD = float(os.environ.get("SENTRY_SILENCE_RMS_THRESHOLD", "450"))
MAX_RECORDING_SECONDS = float(os.environ.get("SENTRY_MAX_RECORDING_SECONDS", "30"))

OPENWAKEWORD_MODELS_DIR = environment_path(
    "SENTRY_OPENWAKEWORD_MODELS_DIR",
    ROOT / "models" / "openwakeword",
)
WAKEWORD_MODEL_PATH = OPENWAKEWORD_MODELS_DIR / f"{WAKE_WORD}_v0.1.onnx"
WAKEWORD_PREDICTION_KEY = WAKEWORD_MODEL_PATH.stem
MELSPECTROGRAM_MODEL_PATH = OPENWAKEWORD_MODELS_DIR / "melspectrogram.onnx"
EMBEDDING_MODEL_PATH = OPENWAKEWORD_MODELS_DIR / "embedding_model.onnx"


def llama_server_command() -> list[str]:
    return [
        str(LLAMA_SERVER_EXE),
        "--model",
        str(LLAMA_MODEL_PATH),
        "--mmproj",
        str(LLAMA_MMPROJ_PATH),
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--reasoning",
        "off",
        "--temp",
        "0",
    ]


def missing_runtime_files() -> list[Path]:
    return [
        path
        for path in (LLAMA_SERVER_EXE, LLAMA_MODEL_PATH, LLAMA_MMPROJ_PATH)
        if not path.is_file()
    ]


def missing_wakeword_files() -> list[Path]:
    return [
        path
        for path in (
            WAKEWORD_MODEL_PATH,
            MELSPECTROGRAM_MODEL_PATH,
            EMBEDDING_MODEL_PATH,
        )
        if not path.is_file()
    ]
