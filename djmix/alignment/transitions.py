from __future__ import annotations

import numpy as np
import pandas as pd
import os

from djmix import models, utils, config


def transitions(mix: models.Mix):
  root = config.get_root()
  result_path = utils.mkpath(root, f'results/transitions/{mix.id}.pkl')
  if os.path.isfile(result_path):
    return pd.read_pickle(result_path)
  
  # Create the tracklist DataFrame.
  df_tlist = pd.DataFrame([track.dict() for track in mix.tracklist])
  df_tlist = df_tlist.rename(columns={'id': 'track_id'})
  df_tlist['i_track'] = np.arange(len(df_tlist))
  
  # Create the transition DataFrame.
  df_prev = df_tlist.copy()
  df_prev = df_prev.rename(columns={'i_track': 'i_track_prev'})
  df_prev['i_track_next'] = df_prev.i_track_prev + 1
  
  df_next = df_tlist.copy()
  df_next = df_next.rename(columns={'i_track': 'i_track_next'})
  df_tran = df_prev.merge(df_next, on='i_track_next', suffixes=('_prev', '_next'))
  df_tran['i_tran'] = np.arange(len(df_tran))
  df_tran = df_tran[['i_tran', 'i_track_prev', 'i_track_next', 'track_id_prev', 'track_id_next']]
  
  # Load the alignment result
  df_align = mix.align()
  
  # Select the best alignments for each track.
  df_align = df_align.sort_values(
    'match_rate', ascending=False
  ).drop_duplicates(
    ['mix_id', 'track_id']
  )
  
  df_mix_align_prev = df_align.copy()
  df_mix_align_prev.columns = df_mix_align_prev.columns + '_prev'
  df_mix_align_next = df_align.copy()
  df_mix_align_next.columns = df_mix_align_next.columns + '_next'
  df = df_tran.merge(df_mix_align_prev).merge(df_mix_align_next)
  df['mix_id'] = mix.id
  df = df.sort_values('i_tran').reset_index(drop=True)  # just to feel safe
  
  data = []
  for _, r in df.iterrows():
    last_wpt_prev = r.wp_raw_prev[0, 0]
    last_wpt_next = r.wp_raw_next[0, 0]
    total_wpt_prev = last_wpt_prev + 1
    total_wpt_next = last_wpt_next + 1
    
    track_cue_out_wpt_prev = r.wp_prev[0, 0]
    extra_wpts_prev = total_wpt_prev - track_cue_out_wpt_prev
    extra_wpts_next = r.wp_next[-1, 0]
    extra_beats_prev = extra_wpts_prev / 2
    extra_beats_next = extra_wpts_next / 2
    
    mix_cue_out_wpt_prev = r.wp_prev[0, 1]
    mix_cue_in_wpt_next = r.wp_next[-1, 1]
    tran_wpts = abs(mix_cue_out_wpt_prev - mix_cue_in_wpt_next)
    
    if mix_cue_out_wpt_prev > mix_cue_in_wpt_next:
      overlap_wpts = tran_wpts + extra_wpts_prev + extra_wpts_next
    else:
      overlap_wpts = (mix_cue_out_wpt_prev + extra_wpts_prev) - (mix_cue_in_wpt_next - extra_wpts_next)
    overlap_beats = overlap_wpts / 2
    
    data.append({
      'tran_wpts': tran_wpts,
      'overlap_beats': overlap_beats,
      'overlap_wpts': overlap_wpts,
      
      'extra_wpts_prev': extra_wpts_prev,
      'extra_wpts_next': extra_wpts_next,
      'extra_beats_prev': extra_beats_prev,
      'extra_beats_next': extra_beats_next,
      
      'last_wpt_prev': last_wpt_prev,
      'last_wpt_next': last_wpt_next,
      'total_wpt_prev': total_wpt_prev,
      'total_wpt_next': total_wpt_next,
    })
  
  df = pd.concat([df, pd.DataFrame(data)], axis=1)
  
  # Create unique IDs.
  df['tran_id'] = df['mix_id'] + '-' + df['i_tran'].map(lambda i: f'{i:02}')
  df = df.set_index('tran_id')
  
  # Select columns I want.
  df = df[[
    'mix_id',
    'i_tran',
    'i_track_prev',
    'i_track_next',
    'track_id_prev',
    'track_id_next',
    
    'match_rate_prev',
    'match_rate_next',
    'matched_beats_prev',
    'matched_beats_next',
    
    'overlap_wpts',
    'overlap_beats',
    'tran_wpts',
    'extra_wpts_prev',
    'extra_wpts_next',
    'extra_beats_prev',
    'extra_beats_next',
    'last_wpt_prev',
    'last_wpt_next',
    'total_wpt_prev',
    'total_wpt_next',
    
    'matched_time_mix_prev',
    'matched_time_mix_next',
    'matched_time_track_prev',
    'matched_time_track_next',
    
    'case_name_prev',
    'case_name_next',
    'feature_prev',
    'feature_next',
    'metric_prev',
    'metric_next',
    'key_change_prev',
    'key_change_next',
    'mix_cue_in_beat_prev',
    'mix_cue_in_beat_next',
    'mix_cue_out_beat_prev',
    'mix_cue_out_beat_next',
    'track_cue_in_beat_prev',
    'track_cue_in_beat_next',
    'track_cue_out_beat_prev',
    'track_cue_out_beat_next',
    'mix_cue_in_time_prev',
    'mix_cue_in_time_next',
    'mix_cue_out_time_prev',
    'mix_cue_out_time_next',
    'track_cue_in_time_prev',
    'track_cue_in_time_next',
    'track_cue_out_time_prev',
    'track_cue_out_time_next',
    'cost_prev',
    'cost_next',
    'wp_prev',
    'wp_next',
    'wp_raw_prev',
    'wp_raw_next',
  ]]
  
  os.makedirs(os.path.dirname(result_path), exist_ok=True)
  df.to_pickle(result_path)
  
  return df
