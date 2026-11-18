"""Start the portable llama.cpp audio server and Sentry test console together."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO

from llamacpp_client import server_is_ready
from llamacpp_config import (
    LLAMA_BASE_URL,
    ROOT,
    llama_server_command,
    missing_runtime_files,
)


STARTUP_TIMEOUT_SECONDS = float(os.environ.get("SENTRY_LLAMA_STARTUP_TIMEOUT", "180"))


def start_llama_server() -> tuple[subprocess.Popen[bytes] | None, BinaryIO | None]:
    if server_is_ready():
        print(f"Using the existing llama.cpp server at {LLAMA_BASE_URL}.", flush=True)
        return None, None

    missing = missing_runtime_files()
    if missing:
        print("The portable llama.cpp runtime is incomplete:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print("See README-LLAMACPP.md for the required files.", file=sys.stderr)
        raise SystemExit(1)

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "llamacpp-server.log"
    log_file = log_path.open("ab", buffering=0)
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    print(f"Starting llama.cpp audio server at {LLAMA_BASE_URL}...", flush=True)
    process = subprocess.Popen(
        llama_server_command(),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.close()
            raise SystemExit(
                f"llama.cpp exited during startup with code {process.returncode}. "
                f"Check {log_path}."
            )
        if server_is_ready():
            print("llama.cpp audio server is ready.", flush=True)
            return process, log_file
        time.sleep(0.5)

    stop_process(process)
    log_file.close()
    raise SystemExit(f"llama.cpp did not become ready. Check {log_path}.")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def main() -> None:
    server_process, log_file = start_llama_server()
    try:
        from llamacpp_frontend import main as run_frontend

        run_frontend()
    finally:
        if server_process is not None:
            print("Stopping llama.cpp audio server...", flush=True)
            stop_process(server_process)
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
