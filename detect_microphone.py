
import time
import numpy as np
import sounddevice as sd

QUIET_SECONDS = 3
SPEAK_SECONDS = 5

IGNORED_NAMES = [
    "microsoft sound mapper",
    "primary sound capture driver",
    "stereo mix",
    "pc speaker"
]

phase = "ignore"
results = {}
streams = []


def create_callback(record):

    def callback(indata, frames, time_info, status):

        if phase == "ignore":
            return

        rms = float(
            np.sqrt(np.mean(indata.astype(np.float32) ** 2))
        )

        if phase == "quiet":
            record["quiet"].append(rms)

        elif phase == "speech":
            record["speech"].append(rms)

    return callback


def detect_active_microphone():

    global phase

    devices = sd.query_devices()
    default_input = sd.default.device[0]

    print("\n========== SCANNING INPUT DEVICES ==========\n")

    for index, device in enumerate(devices):

        if device["max_input_channels"] <= 0:
            continue

        name = device["name"]

        if any(x in name.lower() for x in IGNORED_NAMES):
            continue

        record = {
            "name": name,
            "sample_rate": int(device["default_samplerate"]),
            "channels": None,
            "quiet": [],
            "speech": []
        }

        print(
            f"[{index}] {name} | "
            f"{record['sample_rate']} Hz | "
            f"{device['max_input_channels']} channels"
        )

        callback = create_callback(record)

        # Try mono first, then the device's maximum channels.
        channel_options = [1]

        max_channels = int(device["max_input_channels"])

        if max_channels > 1:
            channel_options.append(max_channels)

        opened = False

        for channel_count in channel_options:

            try:
                stream = sd.InputStream(
                    device=index,
                    samplerate=record["sample_rate"],
                    channels=channel_count,
                    dtype="int16",
                    blocksize=0,
                    callback=callback
                )

                stream.start()

                record["channels"] = channel_count
                results[index] = record
                streams.append(stream)

                print(
                    f"[OPENED] Device {index} "
                    f"using {channel_count} channel(s)"
                )

                opened = True
                break

            except Exception as error:

                print(
                    f"[FAILED] Device {index}, "
                    f"{channel_count} channel(s): {error}"
                )

                try:
                    stream.close()
                except Exception:
                    pass

        if not opened:
            print(f"[SKIPPED] Device {index}")

    if not streams:
        raise RuntimeError("No microphone streams could be opened.")

    # Ignore audio captured while streams were opening.
    for record in results.values():
        record["quiet"].clear()
        record["speech"].clear()

    print("\n======================================")
    print("KEEP QUIET FOR 3 SECONDS...")
    print("======================================")

    phase = "quiet"
    time.sleep(QUIET_SECONDS)

    print("\n======================================")
    print("SPEAK CLEARLY FOR 5 SECONDS...")
    print("======================================")

    phase = "speech"
    time.sleep(SPEAK_SECONDS)

    phase = "ignore"

    print("\nStopping microphone streams...")

    for stream in streams:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    streams.clear()

    # Compare results.
    rankings = []

    print("\n========== MICROPHONE RESULTS ==========\n")

    for index, record in results.items():

        quiet = record["quiet"]
        speech = record["speech"]

        if not quiet or not speech:
            print(f"[NO DATA] Device {index}: {record['name']}")
            continue

        quiet_rms = float(np.median(quiet))
        speech_rms = float(np.median(speech))

        gain_db = 20 * np.log10(
            (speech_rms + 1) / (quiet_rms + 1)
        )

        rankings.append((
            gain_db,
            index,
            quiet_rms,
            speech_rms
        ))

    rankings.sort(reverse=True)

    for gain_db, index, quiet, speech in rankings:

        marker = (
            " [WINDOWS DEFAULT]"
            if index == default_input
            else ""
        )

        print(f"Device: {index}{marker}")
        print(f"Name: {results[index]['name']}")
        print(f"Sample Rate: {results[index]['sample_rate']} Hz")
        print(f"Channels: {results[index]['channels']}")
        print(f"Quiet RMS: {quiet:.2f}")
        print(f"Speech RMS: {speech:.2f}")
        print(f"Speech Gain: {gain_db:.2f} dB")
        print("-" * 45)

    if not rankings:
        print("No usable audio measurements.")
        return None

    best = rankings[0]

    if best[3] < 1 or best[0] < 1:
        print("\nNo clear speech signal detected.")
        return None

    selected = results[best[1]]

    print("\n========== BEST CANDIDATE ==========")
    print("Device Index:", best[1])
    print("Device Name:", selected["name"])
    print("Sample Rate:", selected["sample_rate"])
    print("Channels:", selected["channels"])
    print("====================================")

    return best[1], selected["sample_rate"], selected["channels"]


if __name__ == "__main__":

    try:
        selected = detect_active_microphone()

        if selected:
            print("\nSelected microphone:", selected)

    except KeyboardInterrupt:
        print("\nDetection interrupted.")

    except Exception as error:
        print("\nERROR:", error)

    finally:
        phase = "ignore"

        for stream in streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
