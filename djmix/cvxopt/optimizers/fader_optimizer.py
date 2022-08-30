import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from numpy.typing import ArrayLike
from .optimizer import Optimizer


class FaderOptimizer(Optimizer):
  def __init__(self, config):
    self.config = config
    c = config
    
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
    self.data['S_dj'] = S_dj
    self.data['S_prev'] = S_prev
    self.data['S_next'] = S_next
    
    num_frames = S_dj.shape[1]
    
    fader_prev = cp.Variable(shape=num_frames, pos=True, name=f'fader_prev')
    fader_next = cp.Variable(shape=num_frames, pos=True, name=f'fader_next')
    
    constraints = [
      fader_prev <= 2,
      fader_next <= 2,
      cp.diff(fader_prev) <= 0,
      cp.diff(fader_next) >= 0,
    ]
    
    # Reshape variables for broadcasting.
    fader_prev_ = cp.reshape(fader_prev, shape=(1, num_frames))
    fader_next_ = cp.reshape(fader_next, shape=(1, num_frames))
    
    Y_prev = cp.multiply(fader_prev_, S_prev)
    Y_next = cp.multiply(fader_next_, S_next)
    Y = Y_prev + Y_next
    Y_true = S_dj
    
    # Mean absolute error. MSE makes errors from small signals insignificant.
    loss = cp.sum(cp.abs(Y - Y_true)) / np.prod(S_dj.shape) / S_dj.mean()
    objective = cp.Minimize(loss)
    self.prob = cp.Problem(objective, constraints)
    self.prob.solve(solver='ECOS', verbose=verbose)
    
    for deck in ['prev', 'next']:
      fader = self.prob.var_dict[f'fader_{deck}'].value
      db_gain = librosa.amplitude_to_db(fader)
      self.results[f'fader_{deck}'] = fader
      self.results[f'fader_{deck}_db'] = db_gain
    
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
      fader='white',
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
      fader = self.results[f'fader_{deck}']
      fader_db = librosa.amplitude_to_db(fader)
      axeq.plot(fader_db, color=colors['fader'], linestyle=linestyle, linewidth=linewidth_eq)
      
      if true_curves is not None:
        true_fader_db = librosa.amplitude_to_db(true_curves[f'fader_{deck}'])
        axeq.plot(true_fader_db, color=colors['fader'], linestyle='-', linewidth=linewidth_eq)
      
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
