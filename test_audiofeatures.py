import sys
from pydub import AudioSegment, silence
import os
from pathlib import Path

import numpy as np
import ddsp
import ddsp.training


audio_file_path = sys.argv[1]
sample_rate = 16000

### ###


print(f"Loading & modifying the file: {audio_file_path}")

sound = AudioSegment.from_wav(audio_file_path)

print(f"Number of channels: {sound.channels}")
print(f"Duration in seconds: {sound.duration_seconds}")
print(f"frame_rate: {sound.frame_rate}")


sound = sound.set_channels(1).set_frame_rate(sample_rate)

print(f"new frame_rate: {sound.frame_rate}")


# Convert to numpy array.
channel_asegs = sound.split_to_mono()
samples = [s.get_array_of_samples() for s in channel_asegs]
audio = np.array(samples).astype(np.float32)
audio /= np.iinfo(samples[0].typecode).max

# If only 1 channel, remove extra dim.
if audio.shape[0] == 1:
  audio = audio[0]


if len(audio.shape) == 1:
  audio = audio[np.newaxis, :]

print('\nExtracting audio features...')

ddsp.spectral_ops.reset_crepe()

audio_features = ddsp.training.metrics.compute_audio_features(audio)
