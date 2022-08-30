from __future__ import annotations

import numpy as np
import os
import librosa
import skimage.transform

from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from djmix import models
from .cache import memory


@memory.cache
def downbeat_activations(audio: models.Audio):
  signal, _ = audio.madmom(sr=44100, mono=True)
  beat_activations_ = RNNDownBeatProcessor()(signal)
  return beat_activations_


@memory.cache
def downbeats(audio: models.Audio):
  beat_processor = DBNDownBeatTrackingProcessor(
    beats_per_bar=4,
    transition_lambda=300,
    observation_lambda=128,
    fps=100,
    num_threads=os.cpu_count(),
  )
  beat_activations_ = downbeat_activations(audio)
  beats_ = beat_processor(beat_activations_)
  return beats_


@memory.cache
def corrected_beats(audio: models.Audio, num_beats_aside=33):
  beats = downbeats(audio)[:, 0]
  intvls = np.diff(beats).round(2)
  frames = librosa.util.frame(
    librosa.util.pad_center(
      intvls,
      size=len(intvls) + num_beats_aside - 1,
      mode='reflect'
    ),
    frame_length=num_beats_aside,
    hop_length=1
  )
  
  absdev = np.abs(frames - np.median(frames, axis=0))  # absolute deviations
  scores = absdev[num_beats_aside // 2] / 0.01
  scores = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
  outliers = np.flatnonzero(scores > 2)
  
  if outliers.size == 0:
    beats_crt = beats
  else:
    # Group consecutive wrong beat indices.
    outlier_groups = np.split(outliers, np.flatnonzero(np.diff(outliers) != 1) + 1)
    
    total_beats_inserted = 0
    total_beats_deleted = 0
    segments = []
    i_working = 0
    for inds in outlier_groups:
      first = inds[0] - 1
      last = inds[-1] + 1
      
      if (first - 1) < 0 or (last + 2) > len(beats):
        continue
      
      l_intvl = np.median(intvls[max(first - num_beats_aside // 2, 0):first])
      r_intvl = np.median(intvls[last:min(last + num_beats_aside // 2, len(intvls))])
      c_intvl = (l_intvl + r_intvl) / 2
      
      segment_intvls = intvls[first:last + 1]
      num_beats_crt = round(segment_intvls.sum() / c_intvl)
      segment_beats_crt = np.linspace(beats[first], beats[last], num_beats_crt)
      segment_beats_crt = segment_beats_crt[:-1]  # exclude the last beat
      
      num_beats_raw = len(inds) + 2
      if num_beats_raw < num_beats_crt:
        num_beats_inserted = num_beats_crt - num_beats_raw
        total_beats_inserted += num_beats_inserted
      elif num_beats_raw > num_beats_crt:
        num_beats_deleted = num_beats_raw - num_beats_crt
        total_beats_deleted += num_beats_deleted
      
      segments.append(beats[i_working:first])
      segments.append(segment_beats_crt)
      i_working = last
    
    # Append the last segment.
    segments.append(beats[i_working:])
    
    # Concatenate segments.
    beats_crt = np.concatenate(segments)
  
  return beats_crt


def beat_aggregate(feature, beats, sr, hop_length, frames_per_beat=None):
  max_frame = feature.shape[1]
  beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)
  beat_frames = beat_frames[beat_frames < max_frame]
  beat_feature = np.split(feature, beat_frames, axis=1)
  # Average for each beat.
  beat_feature = beat_feature[1:-1]  # only use chroma features between beats. not before or after beat
  if frames_per_beat is not None:
    beat_feature = [skimage.transform.resize(f, (f.shape[0], frames_per_beat)) for f in beat_feature]
    beat_feature = np.concatenate(beat_feature, axis=1)
  else:
    beat_feature = [f.mean(axis=1) for f in beat_feature]  # average chroma features for each beat
    beat_feature = np.array(beat_feature).T
  return beat_feature
