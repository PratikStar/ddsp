import sys
from pydub import AudioSegment, silence
import os
from pathlib import Path

import numpy as np
from ddsp.colab.colab_utils import audio_bytes_to_np

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
fp_arr = np.array(samples).astype(np.float32)
fp_arr /= np.iinfo(samples[0].typecode).max

# If only 1 channel, remove extra dim.
if fp_arr.shape[0] == 1:
  fp_arr = fp_arr[0]


if len(fp_arr.shape) == 1:
  sound = sound[np.newaxis, :]

print('\nExtracting audio features...')

