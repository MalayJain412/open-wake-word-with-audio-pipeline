"""Wake-word listener that transcribes commands with a portable llama.cpp server."""

from __future__ import annotations

import os
import queue
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
from openwakeword.model import Model
from scipy.io.wavfile import write

from audio.audio_processor import to_mono, wake_word_audio
from audio.device_manager import microphone_from_cli
from llamacpp_client import LlamaCppError, server_is_ready, transcribe_wav
from llamacpp_config import (
    LLAMA_MODEL_NAME,
    LLAMA_TRANSCRIPTIONS_URL,
    MAX_RECORDING_SECONDS,
    EMBEDDING_MODEL_PATH,
    MELSPECTROGRAM_MODEL_PATH,
    SILENCE_DURATION,
    SILENCE_RMS_THRESHOLD,
    WAKE_THRESHOLD,
    WAKE_WORD,
    WAKEWORD_MODEL_PATH,
    WAKEWORD_PREDICTION_KEY,
    missing_wakeword_files,
)


WAKEWORD = WAKE_WORD
TRANSCRIPTION_MODEL = LLAMA_MODEL_NAME
AUDIO_QUEUE_SIZE = 200


def audio_callback(audio_queue: queue.Queue[np.ndarray]):
    def callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status, flush=True)
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    return callback


def get_audio_chunk(audio_queue: queue.Queue[np.ndarray]) -> np.ndarray:
    return to_mono(audio_queue.get())


def audio_rms(audio: np.ndarray) -> float:
    samples = audio.astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def clear_audio_queue(audio_queue: queue.Queue[np.ndarray]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            return


def record_command(audio_queue: queue.Queue[np.ndarray]) -> np.ndarray | None:
    print("\n[RECORDING] Speak your command...", flush=True)
    recorded_chunks: list[np.ndarray] = []
    speech_started = False
    silence_start: float | None = None
    recording_start = time.monotonic()

    while True:
        audio = get_audio_chunk(audio_queue)
        recorded_chunks.append(audio.copy())
        level = audio_rms(audio)
        now = time.monotonic()

        if level >= SILENCE_RMS_THRESHOLD:
            speech_started = True
            silence_start = None
        elif speech_started:
            if silence_start is None:
                silence_start = now
            elif now - silence_start >= SILENCE_DURATION:
                print(f"[RECORDING STOPPED] {SILENCE_DURATION:g} seconds of silence detected.", flush=True)
                break

        if now - recording_start >= MAX_RECORDING_SECONDS:
            print("[RECORDING STOPPED] Maximum recording duration reached.", flush=True)
            break

    if not speech_started:
        print("[NO SPEECH] No command detected.", flush=True)
        return None
    return np.concatenate(recorded_chunks, axis=0)


def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> str | None:
    print("\n[LLAMA.CPP] Sending audio for transcription...", flush=True)
    temporary_path: Path | None = None
    try:
        samples = np.clip(audio_data, -32768, 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        write(temporary_path, sample_rate, samples)
        transcription = transcribe_wav(temporary_path)
        print("\n========== TRANSCRIPTION ==========", flush=True)
        print(transcription, flush=True)
        print("===================================", flush=True)
        return transcription
    except (LlamaCppError, OSError, ValueError) as error:
        print("[LLAMA.CPP ERROR]", error, flush=True)
        return None
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass


def main() -> None:
    if not server_is_ready():
        raise SystemExit(
            "llama.cpp is not ready at "
            f"{LLAMA_TRANSCRIPTIONS_URL}. Run START-LLAMACPP.bat first."
        )
    missing_models = missing_wakeword_files()
    if missing_models:
        paths = "\n".join(f"  - {path}" for path in missing_models)
        raise SystemExit(
            "The openWakeWord model files are missing. "
            "Run 'python run_llamacpp.py' to download them.\n"
            f"{paths}"
        )
    microphone = microphone_from_cli("Sentry voice transcription through llama.cpp.")
    if microphone is None:
        return

    print("Loading openWakeWord model...", flush=True)
    wake_model = Model(
        wakeword_models=[str(WAKEWORD_MODEL_PATH)],
        inference_framework="onnx",
        melspec_model_path=str(MELSPECTROGRAM_MODEL_PATH),
        embedding_model_path=str(EMBEDDING_MODEL_PATH),
    )
    print("Wake-word model loaded.", flush=True)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
    last_detection = 0.0
    print("\nStarting microphone...", flush=True)
    print(f"Wake word: {WAKEWORD}", flush=True)
    print(f"Transcription model: {TRANSCRIPTION_MODEL}", flush=True)
    print("Say the wake word to start recording.", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    try:
        with microphone.open(audio_callback(audio_queue)):
            while True:
                audio = get_audio_chunk(audio_queue)
                predictions = wake_model.predict(wake_word_audio(audio, microphone.sample_rate))
                score = float(predictions.get(WAKEWORD_PREDICTION_KEY, 0.0))
                now = time.monotonic()
                if score < WAKE_THRESHOLD or now - last_detection < 2.0:
                    continue

                last_detection = now
                print(f"\n[WAKE DETECTED] {WAKEWORD} | Score: {score:.4f}", flush=True)
                wake_model.reset()
                command_audio = record_command(audio_queue)
                if command_audio is not None:
                    transcribe_audio(command_audio, microphone.sample_rate)
                clear_audio_queue(audio_queue)
                wake_model.reset()
                print("\n[READY] Waiting for wake word again...", flush=True)
    except KeyboardInterrupt:
        print("\nStopping Sentry...", flush=True)
    finally:
        print("Sentry stopped.", flush=True)


if __name__ == "__main__":
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    main()
