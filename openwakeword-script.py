
import time
import queue
import numpy as np
import sounddevice as sd

from scipy.signal import resample_poly
from openwakeword.model import Model

# ---------------- CONFIGURATION ----------------

DEVICE = 11
SAMPLE_RATE = 48000
CHANNELS = 1

CHUNK_SIZE = 3840       # 80 ms at 48 kHz
WAKE_THRESHOLD = 0.3
COOLDOWN_SECONDS = 2.0

# ------------------------------------------------

audio_queue = queue.Queue(maxsize=100)

print("Loading openWakeWord model...")

model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

print("Model loaded successfully.")

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass

print("\nStarting microphone...")
print("Say hey_jarvis to test wake-word detection.")
print("Press Ctrl+C to stop.\n")

last_detection = 0
max_score = 0

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
            # Get next audio block
            audio = audio_queue.get()

            # Convert 48 kHz to 16 kHz
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

            # Run wake-word detection
            predictions = model.predict(audio_16k)

            score = float(predictions.get("hey_jarvis", 0.0))
            max_score = max(max_score, score)

            now = time.monotonic()

            if score >= WAKE_THRESHOLD:
                if now - last_detection >= COOLDOWN_SECONDS:
                    print(
                        f"\n[WAKE DETECTED] hey_jarvis | "
                        f"Score: {score:.4f}"
                    )

                    last_detection = now

            # Periodically display detection status
            if int(now) % 5 == 0:
                pass

except KeyboardInterrupt:
    print("\nStopping Sentry listener...")

finally:
    print("Listener stopped.")
