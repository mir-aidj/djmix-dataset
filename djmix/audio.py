from __future__ import annotations

import numpy as np
import io
import librosa
import scipy.io.wavfile
from typing import TYPE_CHECKING
from functools import lru_cache
from pydub import AudioSegment
from madmom.audio.signal import Signal
from . import config

if TYPE_CHECKING:
  from numpy.typing import NDArray

try:
  # Load the audio using smart open. It supports streaming large audio files at remote storages such as AWS S3.
  from smart_open import open
except ImportError as e:
  # Use the builtin open.
  pass


@lru_cache(maxsize=config.get_audio_cache_size())
def load_audio(path, sr=None, mono=True, normalize=False, format='numpy'):
  assert format in ['numpy', 'pydub', 'madmom']
  
  with open(path, 'rb') as file_obj:
    pydub_audio = AudioSegment.from_file(file_obj)
    
    # Resample if `sample_rate` is given.
    if sr:
      pydub_audio = pydub_audio.set_frame_rate(sr)
    else:
      sr = pydub_audio.frame_rate
    
    # Set the number of channel (default: mono)
    if mono:
      pydub_audio = pydub_audio.set_channels(1)
    else:
      pydub_audio = pydub_audio.set_channels(2)
    
    # Normalize audio gain (dB) if `normalize` is `True`.
    if normalize:
      # headroom is how close to the maximum volume to boost the signal up to (specified in dB)
      pydub_audio = pydub_audio.normalize(headroom=0.1)
    
    if format == 'numpy':
      nparray = pydub_to_numpy(pydub_audio)
      return nparray, sr
    elif format == 'pydub':
      return pydub_audio, sr
    elif format == 'madmom':
      nparray = pydub_to_numpy(pydub_audio)
      signal = numpy_to_madmom(nparray, sr)
      return signal, sr


def pydub_to_numpy(pydub_audio: AudioSegment) -> NDArray:
  # Ref: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
  channel_sounds = pydub_audio.split_to_mono()
  samples = [s.get_array_of_samples() for s in channel_sounds]
  np_array = np.array(samples).T.astype('float32')
  np_array /= np.iinfo(samples[0].typecode).max
  np_array = np_array.squeeze()
  np_array = np_array.T  # Transpose to fit to librosa format
  return np_array


def numpy_to_pydub(np_array: NDArray, sr: int) -> AudioSegment:
  # Ref: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
  
  # Transpose if it's a stereo audio and librosa format numpy array.
  # The second dimension should be stereo channel.
  if (len(np_array.shape) > 1) and (np_array.shape[0] == 2):
    np_array = np_array.T
  
  wav_io = io.BytesIO()
  scipy.io.wavfile.write(wav_io, sr, np_array)
  wav_io.seek(0)
  sound = AudioSegment.from_wav(wav_io)
  return sound


def numpy_to_madmom(nparray: NDArray, sr: int) -> Signal:
  return Signal(nparray, sample_rate=sr, num_channels=len(nparray.shape))


def export_pydub(pydub_audio: AudioSegment, uri: str, format='mp3'):
  f_mem = io.BytesIO()
  pydub_audio.export(f_mem, format=format)
  with open(uri, 'wb') as f_out:
    f_out.write(f_mem.getvalue())


def metronomify(audio, out_path, beats=None, downbeats=None, sr=None, tags=None):
  if beats is None and downbeats is None:
    raise ValueError('One of beats or downbeats should be given.')
  
  if isinstance(audio, NDArray):
    assert sr is not None, 'sr must be given if the audio is a numpy array.'
    audio = audio
  else:
    audio, sr = load_audio(audio, mono=False)
  
  if downbeats is not None:
    beats, downbeats = downbeats.T
    
    downbeat_clicks = librosa.clicks(times=beats[downbeats == 1], sr=sr, click_freq=3000, length=audio.shape[1])
    beat_clicks = librosa.clicks(times=beats[downbeats != 1], sr=sr, click_freq=1500, length=audio.shape[1])
    clicks = downbeat_clicks + beat_clicks
  
  if beats is not None:
    beats = beats
    
    clicks = librosa.clicks(times=beats, sr=sr, click_freq=1500, length=audio.shape[1])
  
  metronome_mix = audio + clicks.reshape(1, -1)
  metronome_mix = np.clip(metronome_mix, -1, 1)
  
  sound = numpy_to_pydub(metronome_mix, sr)
  sound.export(out_path, format='mp3', bitrate='192', tags=tags)
