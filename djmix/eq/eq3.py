import numpy as np
import librosa
from numpy.typing import ArrayLike
from scipy import signal
from . import biquad_filters


def eq3_filters(
  cutoff_low: int,
  center_mid: int,
  cutoff_high: int,
  sr: int,
  low_db_gain: float = -80,
  mid_db_gain: float = -27,
  high_db_gain: float = -80,
  mid_Q: int = 3,
):
  nyq = 0.5 * sr
  sos_low = biquad_filters.shelf(cutoff_low / nyq, dBgain=low_db_gain, btype='low', ftype='inner', output='sos')
  sos_mid = biquad_filters.peaking(center_mid / nyq, dBgain=mid_db_gain, Q=mid_Q, type='constantq', output='sos')
  sos_high = biquad_filters.shelf(cutoff_high / nyq, dBgain=high_db_gain, btype='high', ftype='inner', output='sos')

  return sos_low, sos_mid, sos_high


def bin_gains(
  filter: ArrayLike,
  bin_frequencies: ArrayLike,
  sr: int,
):
  w, h = signal.sosfreqz(filter, worN=8192)
  filt_freqs = (sr * 0.5 / np.pi) * w
  filt_gains = np.abs(h)

  # Find the closest filter frequency of each bin frequency.
  dist = np.abs(filt_freqs - bin_frequencies.reshape(-1, 1))
  i_closest = dist.argmin(axis=1)
  bin_gains = filt_gains[i_closest]

  return bin_gains


def bin_frequencies(
  spec_type: str,
  num_bins: int,
  fmin: int,
  fmax: int,
):
  cqt_freqs = librosa.cqt_frequencies(num_bins, fmin=fmin, bins_per_octave=12)
  mel_freqs = librosa.mel_frequencies(num_bins, fmin=fmin, fmax=fmax)

  freqs_map = {
    'mel': mel_freqs,
    'cqt': cqt_freqs,
  }
  bin_freqs = freqs_map[spec_type]

  return bin_freqs
