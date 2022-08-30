import numpy as np
import librosa

from typing import Optional
from numpy.typing import ArrayLike
from .eq3 import eq3_filters
from djmix.audio import numpy_to_pydub
from scipy.signal import sosfilt


def reconstruct_volume(
  audio_prev,
  audio_next,
  power_prev,
  power_next,
  hop_size,
  sr,
):
  mixed_prev_np = apply_volumes(audio_prev, power_prev, hop_size)
  mixed_next_np = apply_volumes(audio_next, power_next, hop_size)
  
  mixed_prev = numpy_to_pydub(mixed_prev_np, sr)
  mixed_next = numpy_to_pydub(mixed_next_np, sr)
  
  mixed = mixed_prev.overlay(mixed_next)
  
  return mixed, mixed_prev, mixed_next


def apply_volumes(
  audio,
  power,
  hop_size,
):
  audio_shape = audio.shape
  num_samples = audio.shape[1]
  num_segments = power.shape[0]
  window_size = 2 * hop_size
  
  audio = audio.astype('float32')
  audio = librosa.util.pad_center(audio, num_samples + window_size)
  segments = librosa.util.frame(audio, window_size, hop_size)
  window = librosa.filters.get_window('hann', window_size)
  
  assert segments.shape[-1] == num_segments
  
  filtered = np.zeros_like(audio)
  for i_seg in range(num_segments):
    segment = segments[:, :, i_seg]
    out = segment * np.sqrt(power[i_seg])
    filtered[:, i_seg * hop_size:i_seg * hop_size + window_size] += window * out
  
  lpad = window_size // 2
  rpad = window_size // 2 + window_size % 2
  filtered = filtered[:, lpad:-(rpad)]
  
  assert audio_shape == filtered.shape
  
  return filtered


def reconstruct(
  curves,
  audio_prev,
  audio_next,
  cutoff_low,
  center_mid,
  cutoff_high,
  sr,
  hop_size,
):
  mixed_prev_np = apply_eqs(
    audio=audio_prev,
    gains_fader=curves.get('fader_prev'),
    gains_low=curves.get('eq_prev_low_db'),
    gains_mid=curves.get('eq_prev_mid_db'),
    gains_high=curves.get('eq_prev_high_db'),
    cutoff_low=cutoff_low,
    center_mid=center_mid,
    cutoff_high=cutoff_high,
    hop_size=hop_size,
    sr=sr,
  )
  mixed_next_np = apply_eqs(
    audio=audio_next,
    gains_fader=curves.get('fader_next'),
    gains_low=curves.get('eq_next_low_db'),
    gains_mid=curves.get('eq_next_mid_db'),
    gains_high=curves.get('eq_next_high_db'),
    cutoff_low=cutoff_low,
    center_mid=center_mid,
    cutoff_high=cutoff_high,
    hop_size=hop_size,
    sr=sr,
  )
  
  mixed_prev = numpy_to_pydub(mixed_prev_np, sr)
  mixed_next = numpy_to_pydub(mixed_next_np, sr)
  
  mixed = mixed_prev.overlay(mixed_next)
  
  # TODO: is clipping needed...?
  
  return mixed, mixed_prev, mixed_next


def apply_eqs(
  audio,
  gains_fader: Optional[ArrayLike],
  gains_low: Optional[ArrayLike],
  gains_mid: Optional[ArrayLike],
  gains_high: Optional[ArrayLike],
  cutoff_low,
  center_mid,
  cutoff_high,
  hop_size,
  sr,
):
  audio_shape = audio.shape
  num_samples = audio.shape[1]
  num_segments = gains_fader.shape[0] if gains_low is None else gains_low.shape[0]
  window_size = 2 * hop_size
  
  audio = audio.astype('float32')
  audio = librosa.util.pad_center(audio, size=num_samples + window_size)
  segments = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_size)
  window = librosa.filters.get_window('hann', window_size)
  
  assert segments.shape[-1] == num_segments
  
  filtered = np.zeros_like(audio)
  for i_seg in range(num_segments):
    segment = segments[:, :, i_seg]
    out = segment.copy()
    
    if gains_low is not None:
      gain_low = gains_low[i_seg]
      gain_mid = gains_mid[i_seg]
      gain_high = gains_high[i_seg]
      
      sos_low, sos_mid, sos_high = eq3_filters(
        low_db_gain=gain_low,
        mid_db_gain=gain_mid,
        high_db_gain=gain_high,
        cutoff_low=cutoff_low,
        center_mid=center_mid,
        cutoff_high=cutoff_high,
        sr=sr,
      )
      
      out = sosfilt(sos_low, out)
      out = sosfilt(sos_mid, out)
      out = sosfilt(sos_high, out)
    
    if gains_fader is not None:
      gain_fader = gains_fader[i_seg]
      out *= gain_fader
    
    # TODO: gain 작으면 그냥 꺼버리기
    filtered[:, i_seg * hop_size:i_seg * hop_size + window_size] += window * out
  
  lpad = window_size // 2
  rpad = window_size // 2 + window_size % 2
  filtered = filtered[:, lpad:-(rpad)]
  
  assert audio_shape == filtered.shape
  
  return filtered


def reconstruct_v1(
  audio_prev,
  audio_next,
  curves_prev,
  curves_next,
  low_cutoff_freq,
  mid_cufoff_freq,
  high_cutoff_freq,
  order_lowhigh,
  order_mid,
  sr,
  hop_size,
):
  filter_low, filter_mid, filter_high = create_filters(
    sr,
    low_cutoff_freq,
    high_cutoff_freq,
    mid_cufoff_freq,
    order_lowhigh,
    order_mid,
  )
  
  mixed_prev_np = apply_eqs(audio_prev, curves_prev, filter_low, filter_mid, filter_high, hop_size)
  mixed_next_np = apply_eqs(audio_next, curves_next, filter_low, filter_mid, filter_high, hop_size)
  
  mixed_prev = numpy_to_pydub(mixed_prev_np, sr)
  mixed_next = numpy_to_pydub(mixed_next_np, sr)
  
  mixed = mixed_prev.overlay(mixed_next)
  
  return mixed, mixed_prev, mixed_next


def reconstruct_v0(
  audio_prev,
  audio_next,
  curves_prev,
  curves_next,
  low_cutoff_freq,
  mid_cufoff_freq,
  high_cutoff_freq,
  order_lowhigh,
  order_mid,
  sr,
  hop_size,
):
  filter_low, filter_mid, filter_high = create_filters(
    sr,
    low_cutoff_freq,
    high_cutoff_freq,
    mid_cufoff_freq,
    order_lowhigh,
    order_mid,
  )
  
  mixed_prev_np = apply_eqs(audio_prev, curves_prev, filter_low, filter_mid, filter_high, hop_size)
  mixed_next_np = apply_eqs(audio_next, curves_next, filter_low, filter_mid, filter_high, hop_size)
  
  mixed_prev = numpy_to_pydub(mixed_prev_np, sr)
  mixed_next = numpy_to_pydub(mixed_next_np, sr)
  
  mixed = mixed_prev.overlay(mixed_next)
  
  return mixed, mixed_prev, mixed_next
