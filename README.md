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

On the first interactive run, Sentry asks how to pick a microphone.

1. **Automatic** opens and tests the available inputs. Stay quiet, then speak when prompted so Sentry can compare speech against background noise. If there’s no clear result, it uses the system default when available or offers manual selection.
2. **Manual** lists active inputs that Windows classifies as microphones, headsets, or handsets, with their default sample rates and maximum channel counts. Choose a numbered input and Sentry will check that it can capture audio. [Windows endpoint classifications](https://learn.microsoft.com/en-us/windows/win32/api/mmdeviceapi/ne-mmdeviceapi-endpointformfactor)
3. **System default** uses the current laptop’s default input, which you can change in Windows Sound settings.

Both scripts share the `config.json` saved beside the scripts on the pendrive. Sentry looks up the saved device by name and host API every time it starts. It falls back to automatic selection if that device is missing or unusable. Sentry negotiates the capture format and converts input to 16 kHz mono for wake-word detection.

Run these commands to list inputs or change your saved choice without starting the listener.

```powershell
.\.venv\Scripts\python.exe -m audio.device_manager --list-microphones
.\.venv\Scripts\python.exe -m audio.device_manager --configure-microphone
```

Each laptop needs a working Python installation and the dependencies from step 2. Run that setup on the new laptop if the virtual environment from the previous laptop cannot start. Ollama and its model also need to be available on the laptop running Sentry.

## 4. Run Sentry

```powershell
.\.venv\Scripts\python.exe script.py
```

Say “Alexa,” wait for `[RECORDING]`, then speak. The transcript appears in the terminal. Press `Ctrl+C` to exit.

To test wake-word detection on its own, run the script below and say “Hey Jarvis.”

```powershell
.\.venv\Scripts\python.exe openwakeword-script.py
```

## Browser test console

With Ollama running, start the local test console from the project folder.

```powershell
.\.venv\Scripts\python.exe sentry_frontend.py
```

Open [the Sentry test console](http://127.0.0.1:8765). It starts with **Windows default microphone** selected. Choose another input or your saved Sentry preference if needed. Click **Start listening**, then say the displayed wake word. The console shows the latest transcript and recent listener output. A microphone chosen in the browser applies to the current run. While idle, the console checks for input changes every four seconds. Use **Refresh inputs** to check immediately. A Bluetooth headset appears once Windows exposes its microphone under **Sound > Input**. Click **Stop** to release the microphone, or press `Ctrl+C` in PowerShell to shut down the console.

## Troubleshooting

- If the microphone fails to open, confirm that Windows allows desktop apps to access the microphone, then run `--configure-microphone` to try another input.
- If Ollama refuses the connection, check that it is running on `localhost:11434` and that `ollama list` shows `lfm3-omni:2b`.
- If `/v1/audio/transcriptions` returns a 404 even though the model is installed, update Ollama and restart it. [Ollama transcription endpoint](https://docs.ollama.com/api/openai-compatibility#v1audiotranscriptions)
