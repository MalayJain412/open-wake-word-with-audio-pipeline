import queue
import time

from openwakeword.model import Model

from audio.audio_processor import to_mono, wake_word_audio
from audio.device_manager import microphone_from_cli


WAKE_THRESHOLD = 0.3
COOLDOWN_SECONDS = 2.0


def main():
    microphone = microphone_from_cli("Test Hey Jarvis wake-word detection.")
    if microphone is None:
        return

    print("Loading openWakeWord model...")
    model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    audio_queue = queue.Queue(maxsize=100)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    print("Say Hey Jarvis to test wake-word detection. Press Ctrl+C to stop.")
    last_detection = 0
    try:
        with microphone.open(audio_callback):
            while True:
                audio = to_mono(audio_queue.get())
                predictions = model.predict(wake_word_audio(audio, microphone.sample_rate))
                score = float(predictions.get("hey_jarvis", 0.0))
                now = time.monotonic()
                if score >= WAKE_THRESHOLD and now - last_detection >= COOLDOWN_SECONDS:
                    print(f"[WAKE DETECTED] hey_jarvis | Score: {score:.4f}")
                    last_detection = now
    except KeyboardInterrupt:
        print("Listener stopped.")


if __name__ == "__main__":
    main()
