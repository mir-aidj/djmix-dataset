from __future__ import annotations

import numpy as np
import pandas as pd
import os
import librosa
import traceback

from typing import Literal, Tuple
from djmix import models, utils, config, alignment, eq, audio
from .optimizers import *
from pydantic import BaseModel


class Config(BaseModel):
  mixer: Literal['fader', 'eq3', 'eq3fader', 'eq3nime', 'xfader', 'sum']
  spectrogram: Literal['mel', 'cqt']
  sr: int = 44100
  pad_beats: int = 8
  # STFT:
  hop: int = 4096
  fft: int = 8192
  fmin: int = 50
  fmax: int = 15000
  num_cqt_bins: int = 100
  num_mel_bins: int = 100
  # CQT:
  cqt_bins_per_octave: int = 12
  cqt_filter_scale: int = 2
  # EQ:
  cutoff_low: int = 180
  center_mid: int = 1000
  cutoff_high: int = 3000
  
  mid_opt_range: Tuple[int, int] = (800, 1400)


def optimize(mix: models.Mix, mixer: str = 'eq3fader', spectrogram: str = 'cqt'):
  c = Config(mixer=mixer, spectrogram=spectrogram)
  df = mix.transitions()
  
  records = []
  for tran_id, tr in df.iterrows():
    try:
      if tr.overlap_beats < 0:
        print(f'=> Skipping: tran_id={tran_id} overlap_beats={tr.overlap_beats}')
        continue
      
      # TODO: what if overlapping region is too short?
      # TODO: overlapping region can be larger than previous track length: mix17612
      if tr.total_wpt_prev < tr.overlap_wpts:
        print('=> Overlapping region is larger than prev track. '
              f'Skipping: tran_id={tran_id} overlap_beats={tr.overlap_beats}')
        continue
      
      record = _cvxopt_transition(mix, tran_id, tr, c)
      records.append(record)
    except Exception as e:
      print(f'=> Error at: {tran_id}')
      traceback.print_exc()
  
  return pd.DataFrame(records)


def _cvxopt_transition(mix: models.Mix, tran_id, tr, c: Config):
  from djmix import tracks
  
  root = config.get_root()
  case_id = f'{tran_id}-{c.mixer}-{c.spectrogram}'
  result_dir = utils.mkpath(root, f'results/cvxopt/{mix.id}/{tran_id}')
  result_path = utils.mkpath(result_dir, f'{case_id}.pkl')
  if os.path.isfile(result_path):
    return pd.read_pickle(result_path)
  
  print(f'=> Loading {tr.track_id_prev} and {tr.track_id_next} tracks')
  
  track_prev = tracks[tr.track_id_prev]
  track_next = tracks[tr.track_id_next]
  
  beats_mix = mix.beats()
  beats_prev = track_prev.beats()
  beats_next = track_next.beats()
  
  track_audio_prev, _ = track_prev.numpy(sr=c.sr, mono=False)
  track_audio_next, _ = track_next.numpy(sr=c.sr, mono=False)
  
  print(f'=> Aligning tracks using TSM')
  tran_dj, tran_prev, tran_next = alignment.aligned_tsm(
    wp_prev=tr.wp_prev,
    wp_next=tr.wp_next,
    total_wpt_prev=tr.total_wpt_prev,
    total_wpt_next=tr.total_wpt_next,
    mix=mix.numpy(sr=c.sr, mono=False)[0],
    track_prev=track_audio_prev,
    track_next=track_audio_next,
    beats_mix=beats_mix,
    beats_prev=beats_prev,
    beats_next=beats_next,
    pad_beats=c.pad_beats,
    sr=c.sr,
  )
  
  def extract_specs(x):
    x_mono = librosa.to_mono(x)
    
    mel = librosa.feature.melspectrogram(
      y=x_mono,
      sr=c.sr,
      n_fft=c.fft,
      hop_length=c.hop,
      n_mels=c.num_mel_bins,
      fmin=c.fmin,
      fmax=c.fmax,
      power=1,
    )
    
    cqt = np.abs(
      librosa.cqt(
        y=x_mono,
        sr=c.sr,
        hop_length=c.hop,
        fmin=c.fmin,
        n_bins=c.num_cqt_bins,
        bins_per_octave=12,
        filter_scale=c.cqt_filter_scale,
      )
    )
    
    return mel, cqt
  
  print(f'=> Extracting spectrograms')
  S_dj_mel, S_dj_cqt = extract_specs(tran_dj)
  S_prev_mel, S_prev_cqt = extract_specs(tran_prev)
  S_next_mel, S_next_cqt = extract_specs(tran_next)
  
  specs = {
    'mel': (S_dj_mel, S_prev_mel, S_next_mel),
    'cqt': (S_dj_cqt, S_prev_cqt, S_next_cqt),
  }
  
  print(f'=> Start case: {c}')
  S_dj, S_prev, S_next = specs[c.spectrogram]
  
  print(f'=> Running convex optimization')
  # TODO: different optimizers
  optimizer_map = {
    'fader': FaderOptimizer,
    'eq3': EQ3Optimizer,
    'eq3fader': EQ3FaderOptimizer,
  }
  Optimizer = optimizer_map[c.mixer]
  optimizer = Optimizer(c)
  results = optimizer.optimize(
    S_dj=S_dj,
    S_prev=S_prev,
    S_next=S_next,
    verbose=True,
  )
  
  est_mix, est_prev, est_next = eq.reconstruct(
    curves=results,
    audio_prev=tran_prev,
    audio_next=tran_next,
    cutoff_low=c.cutoff_low,
    center_mid=c.center_mid,
    cutoff_high=c.cutoff_high,
    sr=c.sr,
    hop_size=c.hop,
  )
  dj_mix = audio.numpy_to_pydub(tran_dj, c.sr)
  # Make gains same.
  est_mix = est_mix.apply_gain(dj_mix.dBFS - est_mix.dBFS)
  
  est_mix_np = audio.pydub_to_numpy(est_mix)
  est_prev_np = audio.pydub_to_numpy(est_prev)
  est_next_np = audio.pydub_to_numpy(est_next)
  
  est_mix_mel, est_mix_cqt = extract_specs(est_mix_np)
  est_prev_mel, est_prev_cqt = extract_specs(est_prev_np)
  est_next_mel, est_next_cqt = extract_specs(est_next_np)
  
  est_mix_cqt_db = librosa.amplitude_to_db(est_mix_cqt)
  est_mix_mel_db = librosa.amplitude_to_db(est_mix_mel)
  est_prev_cqt_db = librosa.amplitude_to_db(est_prev_cqt)
  est_prev_mel_db = librosa.amplitude_to_db(est_prev_mel)
  est_next_cqt_db = librosa.amplitude_to_db(est_next_cqt)
  est_next_mel_db = librosa.amplitude_to_db(est_next_mel)
  
  S_dj_cqt_db = librosa.amplitude_to_db(S_dj_cqt)
  S_dj_mel_db = librosa.amplitude_to_db(S_dj_mel)
  
  # Create a record for the current case.
  r = dict(
    mix_id=mix.id,
    tran_id=tran_id,
    case_id=case_id,
    spectrogram=c.spectrogram,
    mixer=c.mixer,
  )
  
  # Compute MAEs.
  bin_freqs_map = {
    'mel': librosa.mel_frequencies(c.num_mel_bins, fmin=c.fmin, fmax=c.fmax),
    'cqt': librosa.cqt_frequencies(n_bins=c.num_cqt_bins, fmin=c.fmin, bins_per_octave=12),
  }
  bin_freqs = bin_freqs_map[c.spectrogram]
  bins = dict(
    low=bin_freqs <= c.cutoff_low,
    mid=(c.cutoff_low < bin_freqs) & (bin_freqs < c.cutoff_high),
    high=c.cutoff_high <= bin_freqs,
  )
  for band in ['low', 'mid', 'high']:
    mae_cqt_band = np.abs(est_mix_cqt_db[bins[band]] - S_dj_cqt_db[bins[band]]).mean()
    mae_mel_band = np.abs(est_mix_mel_db[bins[band]] - S_dj_mel_db[bins[band]]).mean()
    r[f'mae_cqt_{band}'] = mae_cqt_band
    r[f'mae_mel_{band}'] = mae_mel_band
  
  mae_cqt = np.abs(est_mix_cqt_db - S_dj_cqt_db).mean()
  mae_mel = np.abs(est_mix_mel_db - S_dj_mel_db).mean()
  r['mae_cqt'] = mae_cqt
  r['mae_mel'] = mae_mel
  
  if 'fader_prev' in results:
    r['fader_prev'] = results['fader_prev']
    r['fader_next'] = results['fader_next']
  
  if 'eq_prev_low' in results:
    r['eq_prev_low'] = results['eq_prev_low']
    r['eq_prev_mid'] = results['eq_prev_mid']
    r['eq_prev_high'] = results['eq_prev_high']
    r['eq_next_low'] = results['eq_next_low']
    r['eq_next_mid'] = results['eq_next_mid']
    r['eq_next_high'] = results['eq_next_high']
  
  fig = optimizer.plot(
    S_est_mix_db=est_mix_cqt_db if c.spectrogram == 'cqt' else est_mix_mel_db,
    S_est_prev_db=est_prev_cqt_db if c.spectrogram == 'cqt' else est_prev_mel_db,
    S_est_next_db=est_next_cqt_db if c.spectrogram == 'cqt' else est_next_mel_db,
    title=f'{case_id}  {tr.track_id_prev}  {tr.track_id_next}  {mae_cqt=}  {mae_mel=}',
  )
  fig.show()
  
  os.makedirs(result_dir, exist_ok=True)
  
  with open(utils.mkpath(result_dir, f'{case_id}.pdf'), 'wb') as f:
    fig.savefig(f, format='pdf')
  
  audio.export_pydub(dj_mix, utils.mkpath(result_dir, f'{case_id}_dj.mp3'))
  audio.export_pydub(est_mix, utils.mkpath(result_dir, f'{case_id}_est_mix.mp3'))
  audio.export_pydub(est_prev, utils.mkpath(result_dir, f'{case_id}_est_prev.mp3'))
  audio.export_pydub(est_next, utils.mkpath(result_dir, f'{case_id}_est_next.mp3'))
  
  results = pd.Series(r)
  results.to_pickle(result_path)
  
  return results
