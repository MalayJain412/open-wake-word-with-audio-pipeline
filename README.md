# Sentry

Sentry listens for “Alexa,” records your voice command, and prints the transcription from a local Ollama server. Recording stops after two seconds of silence once speech begins, or after 30 seconds.

## 1. Install Ollama and download the model

1. Download and run the Windows installer. Open a new PowerShell window once the installation finishes. [Ollama for Windows](https://ollama.com/download/windows)
2. Open Ollama from the Start menu. If you need to start the server manually, run `ollama serve` in a separate PowerShell window and leave it open.
3. Download the model and confirm that `lfm3-omni:2b` appears in the list.

   ```powershell
   ollama pull lfm3-omni:2b
   ollama list
   ```

   [LFM3 Omni model](https://ollama.com/library/lfm3-omni:2b) · [Ollama CLI reference](https://docs.ollama.com/cli)

## 2. Set up Python

With Python installed, run these commands from the project folder in PowerShell. The final command downloads the pretrained wake-word models used by the scripts. [openWakeWord documentation](https://github.com/dscripka/openWakeWord#usage)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy sounddevice requests openwakeword onnxruntime
.\.venv\Scripts\python.exe -c "from openwakeword.utils import download_models; download_models()"
```

## 3. Select your microphone

```powershell
.\.venv\Scripts\python.exe -m sounddevice
```

Find your microphone’s input device index and replace `DEVICE = 11` in `script.py` with that number. The script expects a microphone that supports mono input at 48 kHz. If you use `openwakeword-script.py`, set its `DEVICE` value too.

## 4. Run Sentry

```powershell
.\.venv\Scripts\python.exe script.py
```

Say “Alexa,” wait for `[RECORDING]`, then speak. The transcript appears in the terminal. Press `Ctrl+C` to exit.

To test wake-word detection on its own, run the script below and say “Hey Jarvis.”

```powershell
.\.venv\Scripts\python.exe openwakeword-script.py
```

## Troubleshooting

- If the microphone fails to open, check the `DEVICE` index and confirm that Windows allows desktop apps to access the microphone.
- If Ollama refuses the connection, check that it is running on `localhost:11434` and that `ollama list` shows `lfm3-omni:2b`.
- If `/v1/audio/transcriptions` returns a 404 even though the model is installed, update Ollama and restart it. [Ollama transcription endpoint](https://docs.ollama.com/api/openai-compatibility#v1audiotranscriptions)
