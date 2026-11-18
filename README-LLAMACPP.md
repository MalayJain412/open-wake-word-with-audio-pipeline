# Sentry with portable llama.cpp transcription

This is a parallel runtime for the existing wake-word pipeline. It keeps the Ollama scripts unchanged and reuses the same microphone selection, wake-word detection, recording behavior, and `frontend/index.html` console.

## Runtime layout

Place a current Windows llama.cpp release and the LFM3-Omni model files as follows:

```text
open-wake-word-with-audio-pipeline
|-- llama.cpp
|   |-- llama-server.exe
|   `-- all DLL files shipped beside llama-server.exe
|-- models
|   |-- LFM3-Omni-2B-Q4_K_M.gguf
|   `-- mmproj-LFM3-Omni-2B-Q4_K_M.gguf
|-- run_llamacpp.py
|-- START-LLAMACPP.bat
|-- llamacpp_launcher.py
|-- llamacpp_frontend.py
|-- llamacpp_listener.py
|-- llamacpp_client.py
`-- llamacpp_config.py
```

Use llama.cpp v0.8.0 or newer. The multimodal projector is required for audio input.

Model files:

- [LiquidAI/LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/tree/main)
- `LFM3-Omni-2B-Q4_K_M.gguf`
- `mmproj-LFM3-Omni-2B-Q4_K_M.gguf`

Download a Windows llama.cpp release from the [official llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases). Copy the entire matching release directory, not only `llama-server.exe`, because the executable may require the accompanying backend DLLs.

## Python dependencies

Install the listener dependencies into this project's `.venv`, or into the parent PortableAI runtime environment:

```powershell
uv pip install -r requirements-llamacpp.txt
```

The launcher checks these Python locations in order:

1. `.venv\Scripts\python.exe`
2. `..\runtime\venv\Scripts\python.exe`

## Run

Activate the Python environment where `requirements.txt` was installed, then run:

```powershell
python run_llamacpp.py
```

`START-LLAMACPP.bat` remains available as an optional double-click launcher.

On the first run, `run_llamacpp.py` downloads the openWakeWord Alexa and feature
models into `models\openwakeword`. They remain there for later offline runs.

The launcher:

1. Validates `llama-server.exe`, the model, and the multimodal projector.
2. Starts the audio model on `127.0.0.1:8081`.
3. Waits for the llama.cpp health endpoint.
4. Starts the existing browser console at `http://127.0.0.1:8765`.
5. Stops the llama.cpp process when the console exits.

Open [the Sentry audio test console](http://127.0.0.1:8765), select a microphone, and click **Start listening**. Say “Alexa,” wait for the recording state, then speak the command.

The flow is:

```text
microphone -> openWakeWord -> command WAV -> llama.cpp transcription -> browser transcript
```

## Direct listener

When a compatible llama.cpp server is already running on port `8081`, the listener can be started directly:

```powershell
.\.venv\Scripts\python.exe llamacpp_listener.py
```

The client submits the recorded WAV file to:

```text
POST http://127.0.0.1:8081/v1/audio/transcriptions
```

It requests JSON output and accepts plain-text, `{ "text": "..." }`, and common OpenAI-compatible response shapes.

## Configuration

These optional environment variables override the portable defaults:

| Variable | Default |
| --- | --- |
| `SENTRY_LLAMA_SERVER_EXE` | `llama.cpp\llama-server.exe` |
| `SENTRY_LLAMA_MODEL` | `models\LFM3-Omni-2B-Q4_K_M.gguf` |
| `SENTRY_LLAMA_MMPROJ` | `models\mmproj-LFM3-Omni-2B-Q4_K_M.gguf` |
| `SENTRY_LLAMA_HOST` | `127.0.0.1` |
| `SENTRY_LLAMA_PORT` | `8081` |
| `SENTRY_LLAMA_ASR_PROMPT` | `Perform ASR in English.` |
| `SENTRY_FRONTEND_PORT` | `8765` |
| `SENTRY_WAKE_WORD` | `alexa` |
| `SENTRY_WAKE_THRESHOLD` | `0.5` |

Example:

```powershell
$env:SENTRY_LLAMA_ASR_PROMPT = "Perform ASR."
python run_llamacpp.py
```

Server output is written to `logs\llamacpp-server.log`.

## Existing Ollama runtime

The original files remain available and continue to use Ollama:

- `script.py`
- `sentry_frontend.py`

The llama.cpp files do not import, replace, or modify that transcription path.
