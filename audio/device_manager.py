"""Discover, test, and remember portable microphone preferences."""

import json
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
PLAYBACK_INPUT_PREFIXES = ("stereo mix", "pc speaker")
WASAPI = "Windows WASAPI"
MICROPHONE_FORM_FACTORS = {4, 5, 6}  # Microphone, Headset, Handset


@dataclass(frozen=True)
class Microphone:
    index: int
    name: str
    host_api: str
    native_rate: int
    max_channels: int

    @property
    def identity(self):
        return {"name": self.name, "host_api": self.host_api}

    @property
    def label(self):
        return f"{self.name} ({self.host_api}) | {self.native_rate} Hz | {self.max_channels} channels"


@dataclass(frozen=True)
class InputFormat:
    microphone: Microphone
    sample_rate: int
    channels: int

    def open(self, callback=None):
        return sd.InputStream(
            device=self.microphone.index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.sample_rate * 2 // 25,
            callback=callback,
        )


def windows_microphones(devices):
    from ctypes import CDLL, c_int

    # Use the same loaded PortAudio library that supplies the device indexes.
    library = CDLL(sd._libname)
    try:
        device_role = library.PaWasapi_GetDeviceRole
    except AttributeError as error:
        raise RuntimeError("This PortAudio build cannot classify Windows microphone endpoints.") from error
    device_role.argtypes = [c_int]
    device_role.restype = c_int

    is_loopback = getattr(library, "PaWasapi_IsLoopback", None)
    if is_loopback is not None:
        is_loopback.argtypes = [c_int]
        is_loopback.restype = c_int

    return [
        device for device in devices
        if device.host_api == WASAPI
        and device_role(device.index) in MICROPHONE_FORM_FACTORS
        and (is_loopback is None or is_loopback(device.index) == 0)
    ]


def discover_microphones():
    host_apis = sd.query_hostapis()
    devices = [
        Microphone(index, device["name"], host_apis[device["hostapi"]]["name"],
                   round(device["default_samplerate"]), device["max_input_channels"])
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]
    if sys.platform != "win32":
        return [device for device in devices if not device.name.casefold().startswith(PLAYBACK_INPUT_PREFIXES)]

    return windows_microphones(devices)


def system_default_microphone(devices):
    if any(device.host_api == WASAPI for device in devices):
        index = next(api["default_input_device"] for api in sd.query_hostapis() if api["name"] == WASAPI)
    else:
        index = sd.default.device[0]
    return next((device for device in devices if device.index == index), None)


def input_formats(microphone):
    rates = dict.fromkeys((microphone.native_rate, 48000, 44100, 16000))
    channels = dict.fromkeys((1, min(2, microphone.max_channels), microphone.max_channels))
    for rate in rates:
        # An 80 ms input block must resample to exactly 1280 wake-word samples.
        if rate <= 0 or rate % 25:
            continue
        for count in channels:
            yield InputFormat(microphone, rate, count)


def validate(microphone):
    for settings in input_formats(microphone):
        try:
            with settings.open() as stream:
                stream.read(settings.sample_rate * 2 // 25)
            return settings
        except (sd.PortAudioError, ValueError):
            continue
    return None


def validate_identity(devices, identity):
    matches = [device for device in devices if device.identity == identity]
    return validate(matches[0]) if len(matches) == 1 else None


def scan_activity(candidates):
    """Compare speech against ambient level on endpoints that can run together."""
    phase = None
    active = []

    def measure(levels):
        def callback(indata, frames, time_info, status):
            current_phase = phase
            if current_phase is not None and not status:
                samples = indata.astype(np.float32)
                levels[current_phase].append(float(np.sqrt(np.mean(samples ** 2))))
        return callback

    with ExitStack() as streams:
        for settings in candidates:
            levels = ([], [])
            try:
                streams.enter_context(settings.open(measure(levels)))
            except (sd.PortAudioError, ValueError):
                continue
            active.append((settings, levels))
        if not active:
            return None
        print("Stay quiet for one second, then speak when prompted.", flush=True)
        phase = 0
        time.sleep(1)
        phase = 1
        print("Speak normally for two seconds.", flush=True)
        time.sleep(2)

    ranked = []
    for settings, (quiet, speech) in active:
        if quiet and speech:
            baseline = float(np.median(quiet))
            level = float(np.percentile(speech, 85))
            if level > max(100, baseline * 1.5):
                ranked.append(((level - baseline) / max(baseline, 100), settings))
    return max(ranked, key=lambda result: result[0])[1] if ranked else None


def manual_selection(devices):
    print("\nAvailable microphones")
    for number, microphone in enumerate(devices, 1):
        print(f"{number}. {microphone.label}")
    while True:
        choice = input(f"Select microphone (1-{len(devices)}, or q to cancel): ").strip()
        if choice.lower() == "q":
            raise RuntimeError("Microphone selection cancelled.")
        if choice.isdecimal() and 1 <= int(choice) <= len(devices):
            settings = validate(devices[int(choice) - 1])
            if settings:
                return settings
            print("That input could not be opened. Choose another microphone.")


def load_config(path):
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(config, dict) and isinstance(config.get("microphone", {}), dict):
            return config
        raise ValueError("expected a microphone configuration object")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        print(f"Could not read {path.name}: {error}. Using microphone defaults.")
        return {}


def save_microphone(path, config, preference):
    if config.get("microphone") == preference:
        return
    config["microphone"] = preference
    temporary_path = path.with_suffix(".tmp")
    try:
        temporary_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        temporary_path.replace(path)
    except OSError as error:
        print(f"Could not save microphone preference to {path}: {error}")


def select_microphone(configure=False, path=CONFIG_PATH):
    devices = discover_microphones()
    if not devices:
        raise RuntimeError("No input devices found. Connect a microphone and try again.")
    default = system_default_microphone(devices)
    default_index = default.index if default else None
    config = load_config(path)
    preference = config.get("microphone", {})
    mode = preference.get("mode")
    interactive = sys.stdin.isatty()
    if configure and not interactive:
        raise RuntimeError("Run --configure-microphone in an interactive terminal.")

    if configure or mode not in ("auto", "manual", "default"):
        mode = "auto"
        preference = {}
        if interactive:
            print("\nSentry microphone configuration\n1. Automatic detection\n"
                  "2. Select manually\n3. Use the system default microphone")
            while True:
                choice = input("Enter choice (1-3, default 1): ").strip() or "1"
                if choice in ("1", "2", "3"):
                    mode = {"1": "auto", "2": "manual", "3": "default"}[choice]
                    break

    settings = None
    if mode == "default":
        settings = validate(default) if default else None
        if not settings:
            print("The system default microphone is unavailable.")
    elif preference.get("preferred_device"):
        settings = validate_identity(devices, preference["preferred_device"])
        if not settings:
            print("The saved microphone is unavailable or ambiguous.")
    elif mode == "manual" and interactive:
        settings = manual_selection(devices)

    if not settings:
        if preference.get("fallback_device"):
            settings = validate_identity(devices, preference["fallback_device"])
            if settings:
                print("Using the previously selected fallback microphone.")
                return settings
        if preference:
            print("Trying automatic detection.")
        ordered = sorted(devices, key=lambda device: device.index != default_index)
        candidates = [settings for device in ordered if (settings := validate(device))]
        if not candidates:
            raise RuntimeError("No microphone could be opened. Check microphone access and try again.")
        microphones = [item for item in candidates if not item.microphone.name.casefold().startswith(
            ("microsoft sound mapper", "primary sound capture driver"))]
        settings = scan_activity(microphones or candidates) if interactive else None
        if not settings:
            settings = next((item for item in candidates if item.microphone.index == default_index), None)
            if settings:
                print("Using the system default microphone.")
            elif interactive:
                settings = manual_selection([item.microphone for item in candidates])
            else:
                settings = candidates[0]
        # Retain an explicit manual/default preference when borrowing a fallback.
        if mode != "auto":
            save_microphone(path, config, {
                "mode": mode,
                "preferred_device": preference.get("preferred_device"),
                "fallback_device": settings.microphone.identity,
            })
            return settings

    save_microphone(path, config, {
        "mode": mode,
        "preferred_device": settings.microphone.identity if mode != "default" else None,
    })
    return settings


def microphone_from_cli(description):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--list-microphones", action="store_true", help="list discovered inputs and exit")
    options.add_argument("--configure-microphone", action="store_true", help="change the saved microphone preference")
    options.add_argument("--microphone-index", type=int, help="use a discovered input for this run")
    options.add_argument("--system-default-microphone", action="store_true", help="use the current system default for this run")
    args = parser.parse_args()
    try:
        if args.list_microphones:
            for microphone in discover_microphones():
                print(microphone.label)
            return None
        if args.system_default_microphone:
            device = system_default_microphone(discover_microphones())
            settings = validate(device) if device else None
            if not settings:
                raise RuntimeError("The system default microphone could not be opened. Check Windows Sound settings or choose an input manually.")
        elif args.microphone_index is None:
            settings = select_microphone(args.configure_microphone)
        else:
            device = next((item for item in discover_microphones() if item.index == args.microphone_index), None)
            settings = validate(device) if device else None
            if not settings:
                raise RuntimeError("The selected microphone could not be opened. Refresh the input list and retry.")
    except (RuntimeError, sd.PortAudioError, EOFError, OSError) as error:
        parser.exit(1, f"Microphone setup failed: {error}\n")
    print(f"Using {settings.microphone.name} ({settings.microphone.host_api}) "
          f"at {settings.sample_rate} Hz, {settings.channels} capture channel(s).")
    return settings


if __name__ == "__main__":
    microphone_from_cli("Check Sentry microphone configuration.")
