
import time
import queue
import tempfile
import os

import numpy as np
import sounddevice as sd
import requests

from scipy.signal import resample_poly
from scipy.io.wavfile import write
from openwakeword.model import Model


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = 11
SAMPLE_RATE = 48000
CHANNELS = 1
CHUNK_SIZE = 3840          # 80 ms at 48 kHz

WAKEWORD = "alexa"         # Change to "hey_jarvis" if needed
WAKE_THRESHOLD = 0.5

SILENCE_DURATION = 2.0
SILENCE_RMS_THRESHOLD = 450

MAX_RECORDING_SECONDS = 30

OLLAMA_URL = "http://localhost:11434/v1/audio/transcriptions"
OLLAMA_MODEL = "lfm3-omni:2b"

audio_queue = queue.Queue(maxsize=200)


# --------------------------------------------------
# LOAD WAKE WORD MODEL
# --------------------------------------------------

print("Loading openWakeWord model...")

wake_model = Model(
    wakeword_models=[WAKEWORD],
    inference_framework="onnx"
)

print("Wake-word model loaded.")


# --------------------------------------------------
# MICROPHONE CALLBACK
# --------------------------------------------------

def audio_callback(indata, frames, time_info, status):

    if status:
        print("Audio status:", status)

    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass


# --------------------------------------------------
# AUDIO HELPERS
# --------------------------------------------------

def get_audio_chunk():

    return audio_queue.get()


def get_rms(audio):

    audio = audio.astype(np.float32)

    return float(np.sqrt(np.mean(audio ** 2)))


def clear_audio_queue():

    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# --------------------------------------------------
# RECORD COMMAND UNTIL 2 SECONDS OF SILENCE
# --------------------------------------------------

def record_command():

    print("\n[RECORDING] Speak your command...")

    recorded_chunks = []

    speech_started = False
    silence_start = None

    recording_start = time.monotonic()

    while True:

        audio = get_audio_chunk()

        recorded_chunks.append(audio.copy())

        rms = get_rms(audio)

        now = time.monotonic()

        # Detect the beginning of speech
        if rms >= SILENCE_RMS_THRESHOLD:

            speech_started = True
            silence_start = None

        # Start counting silence only after speech begins
        elif speech_started:

            if silence_start is None:
                silence_start = now

            elif now - silence_start >= SILENCE_DURATION:

                print("[RECORDING STOPPED] 2 seconds of silence detected.")
                break

        # Safety limit
        if now - recording_start >= MAX_RECORDING_SECONDS:

            print("[RECORDING STOPPED] Maximum recording duration reached.")
            break

    if not speech_started:

        print("[NO SPEECH] No command detected.")
        return None

    audio_data = np.concatenate(recorded_chunks, axis=0)

    return audio_data


# --------------------------------------------------
# SEND AUDIO TO OLLAMA
# --------------------------------------------------

def transcribe_audio(audio_data):

    print("\n[OLLAMA] Sending audio for transcription...")

    temp_path = None

    try:

        audio_data = np.clip(
            audio_data,
            -32768,
            32767
        ).astype(np.int16)

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_path = temp_file.name
        temp_file.close()

        write(
            temp_path,
            SAMPLE_RATE,
            audio_data
        )

        with open(temp_path, "rb") as audio_file:

            response = requests.post(
                OLLAMA_URL,
                files={
                    "file": (
                        "command.wav",
                        audio_file,
                        "audio/wav"
                    )
                },
                data={
                    "model": OLLAMA_MODEL,
                    "response_format": "text"
                },
                timeout=120
            )

        response.raise_for_status()

        transcription = response.text.strip()

        print("\n========== TRANSCRIPTION ==========")
        print(transcription)
        print("===================================")

        return transcription

    except requests.RequestException as error:

        print("[OLLAMA ERROR]", error)

    except Exception as error:

        print("[ERROR]", error)

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return None


# --------------------------------------------------
# MAIN LISTENER
# --------------------------------------------------

def main():

    last_detection = 0

    print("\nStarting microphone...")
    print(f"Wake word: {WAKEWORD}")
    print("Say the wake word to start recording.")
    print("Press Ctrl+C to stop.\n")

    try:

        with sd.InputStream(
            device=DEVICE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):

            while True:

                audio = get_audio_chunk()

                # Resample 48 kHz to 16 kHz for openWakeWord
                audio_16k = resample_poly(
                    audio.reshape(-1),
                    1,
                    3
                )

                audio_16k = np.clip(
                    audio_16k,
                    -32768,
                    32767
                ).astype(np.int16)

                predictions = wake_model.predict(audio_16k)

                score = float(
                    predictions.get(WAKEWORD, 0.0)
                )

                now = time.monotonic()

                if (
                    score >= WAKE_THRESHOLD
                    and now - last_detection >= 2.0
                ):

                    last_detection = now

                    print(
                        f"\n[WAKE DETECTED] "
                        f"{WAKEWORD} | Score: {score:.4f}"
                    )

                    # Reset wake-word model state
                    wake_model.reset()

                    # Record the user's command
                    audio_data = record_command()

                    if audio_data is not None:

                        transcribe_audio(audio_data)

                    # Remove any queued audio captured while
                    # transcription was being processed
                    clear_audio_queue()

                    # Reset model before listening again
                    wake_model.reset()

                    print("\n[READY] Waiting for wake word again...")

    except KeyboardInterrupt:

        print("\nStopping Sentry...")

    finally:

        print("Sentry stopped.")


if __name__ == "__main__":
    main()
