import sys
from pydub import AudioSegment, silence
import os, re
from pathlib import Path

import numpy as np
# import ddsp.training


sample_rate = 16000
directory = "/Users/pratik/data/timbre/monophonic/"
regex = "^13B"
### ###
allsound = None

for filename in os.listdir(directory):
  if re.search(regex, filename) == None:
    continue
  print(filename)
  audio_file_path = os.path.join(directory, filename)

  print(f"\nLoading & modifying the file: {audio_file_path}")

  sound = AudioSegment.from_wav(audio_file_path)
  print(f"Frames in audio: {sound.frame_count()}")

  fr = silence.detect_silence(sound)[0][1]
  sound = sound[fr:]

  if allsound is None:
    allsound = sound
  else:
    allsound.append(sound,  crossfade=0)

print(f"Frames in audio: {allsound.frame_count()}")
print(silence.detect_silence(allsound))

allsound.export(directory + "13B - all.mp3", format="mp3")

  # print(f"Number of channels: {sound.channels}")
  # print(f"Duration in seconds: {sound.duration_seconds}")



# Convert to numpy array.
# channel_asegs = sound.split_to_mono()
# samples = [s.get_array_of_samples() for s in channel_asegs]
# audio = np.array(samples).astype(np.float32)
# audio /= np.iinfo(samples[0].typecode).max
#
# # If only 1 channel, remove extra dim.
# if audio.shape[0] == 1:
#   audio = audio[0]
#
#
# if len(audio.shape) == 1:
#   audio = audio[np.newaxis, :]

# print('\nExtracting audio features...')
#
# ddsp.spectral_ops.reset_crepe()
#
# audio_features = ddsp.training.metrics.compute_audio_features(audio)
