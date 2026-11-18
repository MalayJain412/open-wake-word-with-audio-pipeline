"""Convert microphone audio to the format expected by the wake-word model."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def to_mono(audio):
    if audio.shape[1] == 1:
        return audio[:, 0]
    return np.rint(audio.mean(axis=1)).astype(np.int16)


def wake_word_audio(audio, sample_rate):
    if sample_rate == 16000:
        return audio
    divisor = gcd(16000, sample_rate)
    resampled = resample_poly(audio.astype(np.float32), 16000 // divisor, sample_rate // divisor)
    return np.clip(resampled, -32768, 32767).astype(np.int16)
