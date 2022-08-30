import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from numpy.typing import ArrayLike
from ... import eq
from .optimizer import Optimizer


# from ..config import Config


class EQ3Optimizer(Optimizer):
  def __init__(self, config):
    self.config = config
    c = config
    
    self.filter_low, self.filter_mid, self.filter_high = eq.eq3_filters(
      cutoff_low=c.cutoff_low,
      center_mid=c.center_mid,
      cutoff_high=c.cutoff_high,
      sr=c.sr,
    )
    
    if c.spec_type == 'cqt':
      self.bin_freqs = librosa.cqt_frequencies(n_bins=c.num_cqt_bins, fmin=c.fmin, bins_per_octave=12)
    elif c.spec_type == 'mel':
      self.bin_freqs = librosa.mel_frequencies(c.num_mel_bins, fmin=c.fmin, fmax=c.fmax)
    else:
      raise ValueError(f'Unknown spec_type: {c.spec_type}')
    
    self.bin_gains = dict(
      low=eq.bin_gains(self.filter_low, self.bin_freqs, c.sr),
      mid=eq.bin_gains(self.filter_mid, self.bin_freqs, c.sr),
      high=eq.bin_gains(self.filter_high, self.bin_freqs, c.sr),
    )
    
    self.bins = dict(
      low=self.bin_freqs <= c.cutoff_low,
      mid=(c.mid_opt_range[0] < self.bin_freqs) & (self.bin_freqs < c.mid_opt_range[1]),
      high=c.cutoff_high <= self.bin_freqs,
    )
    
    self.band_gains = dict(
      low=self.bin_gains['low'][self.bins['low']],
      mid=self.bin_gains['mid'][self.bins['mid']],
      high=self.bin_gains['high'][self.bins['high']],
    )
    
    self.data = {}
    self.results = {}
    self.prob = None
  
  def optimize(
    self,
    S_dj: ArrayLike,
    S_prev: ArrayLike,
    S_next: ArrayLike,
    verbose=True
  ):
    c = self.config
    
    self.data['S_dj'] = S_dj
    self.data['S_prev'] = S_prev
    self.data['S_next'] = S_next
    
    num_frames = S_dj.shape[1]
    
    data = self.data
    losses = {}
    constraints = []
    
    for band in ['low', 'mid', 'high']:
      beta_prev = cp.Variable(shape=num_frames, pos=True, name=f'beta_prev_{band}')
      beta_next = cp.Variable(shape=num_frames, pos=True, name=f'beta_next_{band}')
      
      constraints += [
        beta_prev <= 1,
        beta_next <= 1,
        cp.diff(beta_prev) <= 0,
        cp.diff(beta_next) >= 0,
      ]
      
      Sband_dj = S_dj[self.bins[band]]
      Sband_prev = S_prev[self.bins[band]]
      Sband_next = S_next[self.bins[band]]
      
      # Reshape variables for broadcasting.
      beta_prev_ = cp.reshape(beta_prev, shape=(1, num_frames))
      beta_next_ = cp.reshape(beta_next, shape=(1, num_frames))
      
      H_min = self.band_gains[band].reshape(-1, 1)
      # H_inv = 1 - H_min
      H_prev = beta_prev_ + cp.multiply(1 - beta_prev_, H_min)
      H_next = beta_next_ + cp.multiply(1 - beta_next_, H_min)
      Y_prev = cp.multiply(H_prev, Sband_prev)
      Y_next = cp.multiply(H_next, Sband_next)
      Y = Y_prev + Y_next
      Y_true = Sband_dj
      
      # Mean absolute error. MSE makes errors from small signals insignificant.
      loss = cp.sum(cp.abs(Y - Y_true)) / np.prod(Sband_dj.shape) / Sband_dj.mean()
      losses[band] = loss
      data[band] = dict(
        Y_prev=Y_prev,
        Y_next=Y_next,
        Y=Y,
      )
    
    loss = cp.sum(list(losses.values()))
    objective = cp.Minimize(loss)
    self.prob = cp.Problem(objective, constraints)
    self.prob.solve(solver='ECOS', verbose=verbose)
    
    for deck in ['prev', 'next']:
      for band, min_db in zip(
        ['low', 'mid', 'high'],
        [-80, -27, -80]
      ):
        beta = self.prob.var_dict[f'beta_{deck}_{band}'].value
        db_gain = 20 * np.log10(beta + librosa.db_to_amplitude(min_db))
        self.results[f'eq_{deck}_{band}'] = beta
        self.results[f'eq_{deck}_{band}_db'] = db_gain
    
    return self.results
  
  def plot(
    self,
    S_est_mix_db,
    S_est_prev_db,
    S_est_next_db,
    title=None,
    true_curves=None,
    S_prev=None,
    S_next=None,
  ):
    from matplotlib.patches import Patch
    
    S_prev = self.data['S_prev'] if S_prev is None else S_prev
    S_next = self.data['S_next'] if S_next is None else S_next
    
    linestyle = '--'
    linewidth_eq = 2
    cmap = 'magma'
    colors = dict(
      low='#00FFFF',
      mid='#FF00FF',
      high='#FFFF00',
    )
    c = self.config
    y_axis = 'cqt_hz' if c.spec_type == 'cqt' else 'mel'
    
    S_prev_db = librosa.amplitude_to_db(S_prev)
    S_next_db = librosa.amplitude_to_db(S_next)
    S_dj_db = librosa.amplitude_to_db(self.data['S_dj'])
    
    vmin = np.min([
      S_est_prev_db.min(), S_est_next_db.min(), S_est_mix_db.min(),
      S_prev_db.min(), S_next_db.min(), S_dj_db.min(),
    ])
    vmax = np.max([
      S_est_prev_db.max(), S_est_next_db.max(), S_est_mix_db.max(),
      S_prev_db.max(), S_next_db.max(), S_dj_db.max(),
    ])
    
    def plot_spec(S_db, ax):
      librosa.display.specshow(
        S_db,
        sr=c.sr, hop_length=c.hop,
        x_axis='frames', y_axis=y_axis,
        # x_axis='frames', y_axis='mel',
        fmin=c.fmin, fmax=c.fmax,
        bins_per_octave=c.cqt_bins_per_octave,
        cmap=cmap, ax=ax,
        # Let value-to-color mapping equal for all plots:
        vmin=vmin, vmax=vmax,
      )
      ax.set_xlabel(None)
      ax.set_xticks([])
      if c.spec_type == 'cqt':
        ax.set_yticks([2 ** i for i in range(6, 14)])
    
    def plot_curves(deck, ax):
      axeq = ax.twinx()  # instantiate a second axes that shares the same x-axis
      for band in ['low', 'mid', 'high']:
        eq_db = self.results[f'eq_{deck}_{band}_db']
        axeq.plot(eq_db, color=colors[band], linestyle=linestyle, linewidth=linewidth_eq)
      
      if true_curves is not None:
        for band in ['low', 'mid', 'high']:
          true_eq_db = true_curves[f'eq_{deck}_{band}_db']
          axeq.plot(true_eq_db, color=colors[band], linestyle='-', linewidth=linewidth_eq)
      
      axeq.set_ylim(-85, 5)
      axeq.set_yticks(np.arange(-80, 1, 20))
      axeq.set_ylabel('Gain (dB)')
    
    def plot_name(name, ax):
      fig.text(
        0.01, 0.45, name,
        transform=ax.transAxes, fontsize=18, color='white',
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='black')
      )
    
    fig, axes = plt.subplots(6, 1, figsize=(16, 9))
    # Previous ---------------------------------------------------------
    # The original track:
    ax = axes[0]
    plot_spec(S_prev_db, ax)
    plot_name('Raw prev', ax)
    # Legends:
    patches = [
      Patch(color=color, label=name)
      for name, color in colors.items()
    ]
    ax.legend(loc='upper right', handles=patches)
    # The optimized track:
    ax = axes[1]
    plot_spec(S_est_prev_db, ax)
    plot_name('Estimated prev', ax)
    plot_curves('prev', ax)
    
    # Mix --------------------------------------------------------------
    # The DJ mix:
    ax = axes[2]
    plot_spec(S_dj_db, ax)
    plot_name('DJ mix', ax)
    # The optimized mix:
    ax = axes[3]
    plot_spec(S_est_mix_db, ax)
    plot_name('Estimated mix', ax)
    
    # Next -------------------------------------------------------------
    # The original track:
    ax = axes[4]
    plot_spec(S_next_db, ax)
    plot_name('Raw next', ax)
    # The optimized track:
    ax = axes[5]
    plot_spec(S_est_next_db, ax)
    plot_name('Estimated next', ax)
    plot_curves('next', ax)
    
    # X-axis
    tick_frames = np.arange(0, S_dj_db.shape[1], S_dj_db.shape[1] // 10)
    tick_times = librosa.frames_to_time(tick_frames, sr=c.sr, hop_length=c.hop).round().astype(int)
    tick_labels = [f'{sec // 60:02}:{sec % 60:02}' for sec in tick_times]
    ax.set_xticks(tick_frames)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Time')
    
    if title:
      fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.05)
    
    return fig
