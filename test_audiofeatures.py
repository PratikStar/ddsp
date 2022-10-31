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

sound = AudioSegment.from_wav(audio_file_path).set_channels(1).set_frame_rate(sample_rate).get_array_of_samples()

print(f"Number of channels: {sound.channels}")
print(f"Duration in seconds: {sound.duration_seconds}")
print(f"frame_rate: {sound.frame_rate}")

if len(sound.shape) == 1:
  sound = sound[np.newaxis, :]

print('\nExtracting audio features...')

print(sound.shape)