"""Start the complete portable llama.cpp wake-word application."""

from __future__ import annotations

import importlib.util
import sys


REQUIRED_MODULES = (
    "numpy",
    "scipy",
    "sounddevice",
    "requests",
    "openwakeword",
    "onnxruntime",
)


def missing_dependencies() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def ensure_wakeword_models() -> None:
    from llamacpp_config import (
        OPENWAKEWORD_MODELS_DIR,
        WAKE_WORD,
        missing_wakeword_files,
    )

    missing = missing_wakeword_files()
    if not missing:
        return

    print("Downloading the openWakeWord model files for first-time setup...", flush=True)
    try:
        from openwakeword.utils import download_models

        download_models(
            model_names=[WAKE_WORD],
            target_directory=str(OPENWAKEWORD_MODELS_DIR),
        )
    except Exception as error:
        raise SystemExit(
            "Could not download the openWakeWord model files. "
            "Connect to the internet and run this command again.\n"
            f"Details: {error}"
        ) from error

    missing = missing_wakeword_files()
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"The openWakeWord setup is incomplete:\n{paths}")

    print("openWakeWord model files are ready for offline use.", flush=True)


def main() -> None:
    print(f"Using Python: {sys.executable}", flush=True)

    missing = missing_dependencies()
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"Missing Python dependencies: {names}\n"
            "Install them into this Python environment with:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt'
        )

    ensure_wakeword_models()

    from llamacpp_launcher import main as launch

    launch()


if __name__ == "__main__":
    main()
