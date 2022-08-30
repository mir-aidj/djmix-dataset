from __future__ import annotations

import numpy as np
import librosa
from djmix import models
from djmix.audio import load_audio

from .cache import memory
from .beats import corrected_beats, beat_aggregate

SAMPLE_RATE = 44100
HOP = 1024
N_FFT = 4096
N_MFCC = 20
FMIN = 20
FMAX = 20000
MEL_BINS = 128
CQT_BINS = 120
CQT_BINS_PER_OCTAVE = 12


@memory.cache
def melspectrogram(audio: models.Audio, **kwargs):
  """Amplitude mel-spectrogram"""
  y, sr = load_audio(audio.path)
  melspectrogram_ = librosa.feature.melspectrogram(
    y=y,
    sr=kwargs.get('sr', sr),
    n_fft=kwargs.get('n_fft', N_FFT),
    hop_length=kwargs.get('hop_length', HOP),
    n_mels=kwargs.get('n_mels', MEL_BINS),
    fmin=kwargs.get('fmin', FMIN),
    fmax=kwargs.get('fmax', FMAX),
    power=kwargs.get('power', 1),
  )
  return melspectrogram_


@memory.cache
def mfcc(audio: models.Audio):
  melspectrogram_ = melspectrogram(audio)
  mfcc_ = librosa.feature.mfcc(
    S=librosa.amplitude_to_db(melspectrogram_),
    n_mfcc=N_MFCC,
  )
  return mfcc_


@memory.cache
def cqt(audio: models.Audio, **kwargs):
  y, sr = load_audio(audio.path)
  cqt_ = librosa.cqt(
    y,
    sr=kwargs.get('sr', sr),
    hop_length=kwargs.get('hop_length', HOP),
    fmin=kwargs.get('fmin', FMIN),
    n_bins=kwargs.get('n_bins', CQT_BINS),
    bins_per_octave=kwargs.get('bins_per_octave', CQT_BINS_PER_OCTAVE),
    filter_scale=kwargs.get('filter_scale', 1),
  )
  cqt_ = np.abs(cqt_)
  return cqt_


@memory.cache
def chroma(audio: models.Audio):
  cqt_ = cqt(audio)
  chroma_ = librosa.feature.chroma_cens(
    C=cqt_,
    hop_length=HOP,
    fmin=FMIN,
    n_chroma=12,
    n_octaves=7,
    bins_per_octave=CQT_BINS_PER_OCTAVE,
  )
  return chroma_


@memory.cache
def halfbeat_mfcc(audio: models.Audio):
  mfcc_ = mfcc(audio)
  beats_ = corrected_beats(audio)
  
  halfbeat_mfcc_ = beat_aggregate(mfcc_, beats_, SAMPLE_RATE, HOP, frames_per_beat=2)
  return halfbeat_mfcc_


@memory.cache
def halfbeat_chroma(audio: models.Audio):
  chroma_ = chroma(audio)
  beats_ = corrected_beats(audio)
  
  halfbeat_chroma_ = beat_aggregate(chroma_, beats_, SAMPLE_RATE, HOP, frames_per_beat=2)
  return halfbeat_chroma_
