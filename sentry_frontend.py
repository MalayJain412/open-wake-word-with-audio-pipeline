"""Serve the Sentry test console on the local laptop."""

import json
import signal
import subprocess
import sys
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

from audio.device_manager import discover_microphones, system_default_microphone


ROOT = Path(__file__).resolve().parent
ADDRESS = "127.0.0.1"
PORT = 8765
ORIGIN = f"http://{ADDRESS}:{PORT}"


class Listener:
    def __init__(self):
        self.lock = Lock()
        self.process = None
        self.lines = deque(maxlen=100)
        self.exit_code = None

    def snapshot(self):
        with self.lock:
            return {"running": self.process is not None, "lines": list(self.lines), "exit_code": self.exit_code}

    def start(self, index):
        with self.lock:
            if self.process is not None:
                raise RuntimeError("The listener is already running.")
            args = [sys.executable, "-X", "utf8", "-u", str(ROOT / "script.py")]
            if index == "system":
                args.append("--system-default-microphone")
            elif index is not None:
                args.extend(["--microphone-index", str(index)])
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       creationflags=creation_flags,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            self.lines.clear()
            self.exit_code = None
            self.process = process
            Thread(target=self.read_output, args=(process,), daemon=True).start()

    def read_output(self, process):
        with process.stdout:
            for output in process.stdout:
                line = output.rstrip()
                if line:
                    with self.lock:
                        self.lines.append(line)
        code = process.wait()
        with self.lock:
            self.process = None
            if self.exit_code is None:
                self.exit_code = code

    def stop(self):
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return
            self.exit_code = 0
        try:
            interrupt = signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT
            process.send_signal(interrupt)
            process.wait(timeout=2)
        except (KeyboardInterrupt, OSError, subprocess.TimeoutExpired):
            try:
                process.terminate()
            except OSError:
                if process.poll() is None:
                    raise


listener = Listener()


def device_snapshot():
    devices = discover_microphones()
    default = system_default_microphone(devices)
    return {"devices": [asdict(device) for device in devices],
            "default_index": default.index if default else None}


class Handler(BaseHTTPRequestHandler):
    def send_content(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data, status=200):
        self.send_content(json.dumps(data).encode("utf-8"), "application/json", status)

    def do_GET(self):
        if self.headers.get("Host") != f"{ADDRESS}:{PORT}":
            self.send_error(403)
            return
        if self.path == "/":
            self.send_content((ROOT / "frontend" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self.send_json(listener.snapshot())
        elif self.path == "/api/devices":
            from script import OLLAMA_MODEL, WAKEWORD

            try:
                # PortAudio enumerates devices when the process loads it. A fresh
                # process sees USB hot-plugs and Windows default changes.
                output = subprocess.check_output([sys.executable, str(Path(__file__)), "--devices"],
                                                 cwd=ROOT, encoding="utf-8", timeout=10)
                snapshot = json.loads(output)
            except subprocess.CalledProcessError as error:
                self.send_json({"error": error.output.strip() or "Could not read microphone inputs."}, 503)
                return
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                self.send_json({"error": str(error)}, 503)
                return
            self.send_json({**snapshot, "wake_word": WAKEWORD, "model": OLLAMA_MODEL})
        else:
            self.send_error(404)

    def do_POST(self):
        if (self.headers.get("Host") != f"{ADDRESS}:{PORT}"
                or self.headers.get("Origin") != ORIGIN
                or self.headers.get_content_type() != "application/json"):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 1 <= length <= 256:
                raise ValueError("Invalid request length.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a request object.")
            if self.path == "/api/start":
                index = data.get("index")
                if index not in (None, "system") and (type(index) is not int or index < 0):
                    raise ValueError("Invalid microphone selection.")
                listener.start(index)
            elif self.path == "/api/stop":
                listener.stop()
            else:
                self.send_error(404)
                return
        except (ValueError, RuntimeError, OSError) as error:
            self.send_json({"error": str(error)}, 400)
            return
        self.send_json(listener.snapshot())

    def log_request(self, code="-", size="-"):
        if str(code) != "200":
            super().log_request(code, size)


if __name__ == "__main__":
    if sys.argv[1:] == ["--devices"]:
        try:
            print(json.dumps(device_snapshot()))
        except (RuntimeError, OSError) as error:
            print(str(error))
            sys.exit(1)
        sys.exit(0)
    with ThreadingHTTPServer((ADDRESS, PORT), Handler) as server:
        print(f"Sentry test console: {ORIGIN}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
