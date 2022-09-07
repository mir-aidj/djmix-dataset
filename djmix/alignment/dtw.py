from __future__ import annotations

import os
import traceback

import librosa

from scipy.spatial.distance import cdist
from djmix import models, utils, config
from .helpers import *

NUM_FORGIVABLE_WARP_POINTS = 2
WARP_POINTS_PER_BEAT = 2
MIN_WP_BEATS = 16


def align(mix: models.Mix, use_all_distances=False):
  root = config.get_root()
  result_path = utils.mkpath(root, f'results/alignment/{mix.id}.pkl')
  if os.path.isfile(result_path):
    return pd.read_pickle(result_path)
  
  mix_chroma = mix.halfbeat_chroma()
  mix_mfcc = mix.halfbeat_mfcc()
  mix_beats = mix.beats()
  
  # Align each track.
  data = []
  for i_track, track in enumerate(mix.tracklist):
    track.id = track.id
    if track.id is None:
      continue
    
    try:
      track_chroma = track.halfbeat_chroma()
      track_mfcc = track.halfbeat_mfcc()
      track_beats = track.beats()
      
      if len(track_beats) < 16:
        # There are a few tracks of which beats are not detected at all,
        # because they only have ambient sounds. Skip them.
        continue
      
      track_mfcc_means = track_mfcc.mean(axis=1, keepdims=True)
      track_mfcc_stds = track_mfcc.std(axis=1, keepdims=True)
      track_mfcc_scaled = (track_mfcc - track_mfcc_means) / track_mfcc_stds
      mix_mfcc_scaled = (mix_mfcc - track_mfcc_means) / track_mfcc_stds
      track_mfcc_scaled = np.nan_to_num(track_mfcc_scaled, nan=0.0, posinf=0.0, neginf=0.0)
      mix_mfcc_scaled = np.nan_to_num(mix_mfcc_scaled, nan=0.0, posinf=0.0, neginf=0.0)
      
      track_chroma_means = track_chroma.mean(axis=1, keepdims=True)
      track_chroma_stds = track_chroma.std(axis=1, keepdims=True)
      track_chroma_scaled = (track_chroma - track_chroma_means) / track_chroma_stds
      mix_chroma_scaled = (mix_chroma - track_chroma_means) / track_chroma_stds
      track_chroma_scaled = np.nan_to_num(track_chroma_scaled, nan=0.0, posinf=0.0, neginf=0.0)
      mix_chroma_scaled = np.nan_to_num(mix_chroma_scaled, nan=0.0, posinf=0.0, neginf=0.0)
      
      track_all = np.concatenate([track_chroma, track_mfcc])
      track_all_scaled = np.concatenate([track_chroma_scaled, track_mfcc_scaled])
      mix_all = np.concatenate([mix_chroma, mix_mfcc])
      mix_all_scaled = np.concatenate([mix_chroma_scaled, mix_mfcc_scaled])
      
      # if track_all.size == 0:
      #   # There can be no beat in a track if the track only contains ambient sounds.
      #   print(f'Track {track.id} has an empty feature. It will be ignored.')
      #   continue
      if use_all_distances:
        features = {
          'all_scaled': (track_all_scaled, mix_all_scaled),
          'all': (track_all, mix_all),
          'mfcc_scaled': (track_mfcc_scaled, mix_mfcc_scaled),
          'mfcc': (track_mfcc, mix_mfcc),
          'chroma_scaled': (track_chroma_scaled, mix_chroma_scaled),
          'chroma': (track_chroma, mix_chroma),
        }
        metrics = {
          'cos': 'cosine',
          'eucld': 'euclidean',
        }
      else:
        features = {
          'all_scaled': (track_all_scaled, mix_all_scaled),
        }
        metrics = {
          'cos': 'cosine',
        }
      
      for feature_name, (track_feat, mix_feat) in features.items():
        pitch_shifts = [0] if 'mfcc' in feature_name else np.arange(12)
        
        for pitch_shift in pitch_shifts:
          # Transpose if chroma is used.
          if ('mfcc' in feature_name) or (pitch_shift == 0):
            X, Y = track_feat, mix_feat
          else:
            X, Y = track_feat.copy(), mix_feat.copy()
            X[:12] = np.roll(X[:12], pitch_shift, axis=0)  # circular pitch shifting
          
          for metric_abbr, metric in metrics.items():
            case_name = f'{feature_name}_{metric_abbr}'
            print(f'=> Aligning {mix.id} {i_track:02} {track.id:11}: key={pitch_shift:02} case={case_name}')
            
            C = cdist(X.T, Y.T, metric=metric)
            # There are some frames of which values are all zeros,
            # which makes cosine similarity produces nan.
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
            D, wp_raw = librosa.sequence.dtw(C=C, subseq=True)
            
            # Compute the cost and keep the results if they are the best.
            matching_function = D[-1, :] / wp_raw.shape[0]
            cost = matching_function.min()
            
            # Select consecutive warp points. Non-consecutive points will be forgiven
            # if it's in 1 beat (2 warp points) difference.
            wp = correct_wp(wp_raw, NUM_FORGIVABLE_WARP_POINTS)
            
            # Group adjacent warp paths and select the longest one.
            # Include other possible warp paths if they lies on the same line with the longest wp.
            if wp.size > 0:
              wp = drop_weird_wp_segments(wp, NUM_FORGIVABLE_WARP_POINTS, MIN_WP_BEATS, WARP_POINTS_PER_BEAT)
            
            # Compute match rate for raw wp.
            total_beats = len(track_beats)
            wp_mix_raw = wp_raw[:, 1][::-1]
            wp_trk_raw = wp_raw[:, 0][::-1]
            dydx_raw = np.diff(wp_trk_raw) / np.diff(wp_mix_raw)
            matched_beats_raw = (dydx_raw == 1).sum() // WARP_POINTS_PER_BEAT
            match_rate_raw = matched_beats_raw / total_beats
            
            # Compute match rate using wp.
            if (wp is not None) and (wp.size > 0):
              wp_mix = wp[:, 1][::-1]
              wp_trk = wp[:, 0][::-1]
              matched_beats = (wp_trk[-1] - wp_trk[0] + 1) // WARP_POINTS_PER_BEAT
              match_rate = matched_beats / total_beats
              
              mix_cue_in_beat = wp_mix[0] // 2
              mix_cue_out_beat = wp_mix[-1] // 2
              track_cue_in_beat = wp_trk[0] // 2
              track_cue_out_beat = wp_trk[-1] // 2
              
              mix_cue_in_time = mix_beats[mix_cue_in_beat]
              mix_cue_out_time = mix_beats[mix_cue_out_beat]
              track_cue_in_time = track_beats[track_cue_in_beat]
              track_cue_out_time = track_beats[track_cue_out_beat]
              
              matched_time_mix = mix_beats[mix_cue_out_beat] - mix_beats[mix_cue_in_beat]
              matched_time_track = track_beats[track_cue_out_beat] - track_beats[track_cue_in_beat]
            else:
              matched_beats = 0
              match_rate = 0.0
              
              mix_cue_in_beat = None
              mix_cue_out_beat = None
              track_cue_in_beat = None
              track_cue_out_beat = None
              
              mix_cue_in_time = None
              mix_cue_out_time = None
              track_cue_in_time = None
              track_cue_out_time = None
              
              matched_time_mix = 0.0
              matched_time_track = 0.0
            
            data.append((
              mix.id, track.id, case_name, feature_name, metric, pitch_shift,
              match_rate, match_rate_raw, matched_beats, matched_beats_raw,
              matched_time_mix, matched_time_track,
              mix_cue_in_beat, mix_cue_out_beat, track_cue_in_beat, track_cue_out_beat,
              mix_cue_in_time, mix_cue_out_time, track_cue_in_time, track_cue_out_time,
              cost, wp, wp_raw,
            ))
    except Exception as e:
      traceback.print_exc()
  
  # Create result DF.
  df_result = pd.DataFrame(data, columns=[
    'mix_id', 'track_id', 'case_name', 'feature', 'metric', 'key_change',
    'match_rate', 'match_rate_raw', 'matched_beats', 'matched_beats_raw',
    'matched_time_mix', 'matched_time_track',
    'mix_cue_in_beat', 'mix_cue_out_beat', 'track_cue_in_beat', 'track_cue_out_beat',
    'mix_cue_in_time', 'mix_cue_out_time', 'track_cue_in_time', 'track_cue_out_time',
    'cost', 'wp', 'wp_raw',
  ])
  
  os.makedirs(os.path.dirname(result_path), exist_ok=True)
  df_result.to_pickle(result_path)
  
  return df_result
