import sys

import numpy as np
from ddsp.colab.colab_utils import audio_bytes_to_np

audio_file_path = sys.argv[1]
sample_rate = 16000

print(f"Loading file: {audio_file_path}")


audio = audio_bytes_to_np(audio_file_path,
                                 sample_rate=sample_rate,
                                 normalize_db=None)


if len(audio.shape) == 1:
  audio = audio[np.newaxis, :]

print('\nExtracting audio features...')

print(audio.shape)