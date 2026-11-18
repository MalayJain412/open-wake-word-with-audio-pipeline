
import time
import numpy as np
import sounddevice as sd

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------

QUIET_SECONDS = 3
SPEAK_SECONDS = 5

# Ignore virtual/system playback inputs
IGNORED_NAMES = [
    "microsoft sound mapper",
    "primary sound capture driver",
    "stereo mix",
    "pc speaker"
]

results = {}
streams = []
start_time = None


# ---------------------------------------
# AUDIO CALLBACK
# ---------------------------------------

def create_callback(device_index):

    def callback(indata, frames, time_info, status):

        if status:
            return

        now = time.monotonic()
        elapsed = now - start_time

        rms = float(
            np.sqrt(np.mean(indata.astype(np.float32) ** 2))
        )

        if device_index not in results:
            return

        if elapsed <= QUIET_SECONDS:
            results[device_index]["quiet"].append(rms)

        elif elapsed <= QUIET_SECONDS + SPEAK_SECONDS:
            results[device_index]["speech"].append(rms)

    return callback


# ---------------------------------------
# DETECT INPUT DEVICES
# ---------------------------------------

def detect_active_microphone():

    global start_time

    devices = sd.query_devices()
    default_input = sd.default.device[0]

    candidates = []

    print("\n========== SCANNING INPUT DEVICES ==========\n")

    for index, device in enumerate(devices):

        if device["max_input_channels"] <= 0:
            continue

        name = device["name"]

        if any(
            ignored in name.lower()
            for ignored in IGNORED_NAMES
        ):
            continue

        candidates.append((index, device))

        print(
            f"[{index}] {name} | "
            f"{device['default_samplerate']} Hz"
        )

    if not candidates:
        raise RuntimeError("No usable microphones found.")

    # ---------------------------------------
    # OPEN ALL MICROPHONES
    # ---------------------------------------

    print("\nOpening microphone streams...\n")

    for index, device in candidates:

        results[index] = {
            "name": device["name"],
            "quiet": [],
            "speech": []
        }

        try:

            stream = sd.InputStream(
                device=index,
                samplerate=int(device["default_samplerate"]),
                channels=1,
                dtype="int16",
                callback=create_callback(index),
                blocksize=1024
            )

            stream.start()
            streams.append(stream)

            print(f"[OPENED] {index}: {device['name']}")

        except Exception as error:

            print(f"[FAILED] {index}: {error}")
            results.pop(index, None)

    if not streams:
        raise RuntimeError("Could not open any microphone.")

    # ---------------------------------------
    # RECORD QUIET BASELINE
    # ---------------------------------------

    print("\n---------------------------------------")
    print("KEEP QUIET FOR 3 SECONDS...")
    print("---------------------------------------")

    start_time = time.monotonic()

    time.sleep(QUIET_SECONDS)

    # ---------------------------------------
    # RECORD SPEECH
    # ---------------------------------------

    print("\n---------------------------------------")
    print("NOW SPEAK CLEARLY FOR 5 SECONDS!")
    print("---------------------------------------")

    time.sleep(SPEAK_SECONDS)

    # ---------------------------------------
    # STOP STREAMS
    # ---------------------------------------

    for stream in streams:
        stream.stop()
        stream.close()

    # ---------------------------------------
    # COMPARE MICROPHONES
    # ---------------------------------------

    print("\n========== MICROPHONE RESULTS ==========\n")

    rankings = []

    for index, data in results.items():

        quiet = data["quiet"]
        speech = data["speech"]

        if not quiet or not speech:
            continue

        quiet_rms = float(np.median(quiet))
        speech_rms = float(np.median(speech))

        gain = speech_rms - quiet_rms

        rankings.append((
            gain,
            index,
            quiet_rms,
            speech_rms
        ))

    rankings.sort(reverse=True)

    for gain, index, quiet, speech in rankings:

        marker = " [WINDOWS DEFAULT]" if index == default_input else ""

        print(f"Device: {index}{marker}")
        print(f"Name: {results[index]['name']}")
        print(f"Quiet RMS: {quiet:.2f}")
        print(f"Speech RMS: {speech:.2f}")
        print(f"Speech Gain: {gain:.2f}")
        print("-" * 45)

    if not rankings:
        print("No usable audio measurements.")
        return None

    best = rankings[0]

    if best[3] < 1 or best[0] < 1:
        print("\nNo clear speech signal detected.")
        return None

    print("\n========== BEST CANDIDATE ==========")
    print("Device Index:", best[1])
    print("Device Name:", results[best[1]]["name"])
    print("====================================")

    return best[1]


# ---------------------------------------
# MAIN
# ---------------------------------------

if __name__ == "__main__":

    try:
        selected_device = detect_active_microphone()

        if selected_device is not None:
            print("\nSelected microphone:", selected_device)

    except KeyboardInterrupt:
        print("\nDetection interrupted.")

    except Exception as error:
        print("\nERROR:", error)

    finally:
        for stream in streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
