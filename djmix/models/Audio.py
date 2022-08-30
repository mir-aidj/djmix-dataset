from __future__ import annotations

import os
import abc
from typing import Optional, Tuple, TYPE_CHECKING
from pydantic import BaseModel
from ..audio import *
from djmix import features

if TYPE_CHECKING:
  from numpy.typing import NDArray
  from pydub import AudioSegment
  from madmom.audio.signal import Signal


class Audio(BaseModel, abc.ABC):
  
  def beats(self, **kwargs):
    return features.corrected_beats(self, **kwargs)
  
  def halfbeat_chroma(self, **kwargs):
    return features.halfbeat_chroma(self, **kwargs)
  
  def halfbeat_mfcc(self, **kwargs):
    return features.halfbeat_mfcc(self, **kwargs)
  
  def melspectrogram(self, **kwargs):
    return features.melspectrogram(self, **kwargs)
  
  def cqt(self, **kwargs):
    return features.cqt(self, **kwargs)
  
  def numpy(self, sr=None, mono=True, normalize=False) -> Tuple[NDArray, int]:
    return load_audio(self.path, sr=sr, mono=mono, normalize=normalize, format='numpy')
  
  def pydub(self, sr=None, mono=True, normalize=False) -> Tuple[AudioSegment, int]:
    return load_audio(self.path, sr=sr, mono=mono, normalize=normalize, format='pydub')
  
  def madmom(self, sr=None, mono=True, normalize=False) -> Tuple[Signal, int]:
    return load_audio(self.path, sr=sr, mono=mono, normalize=normalize, format='madmom')
  
  def assert_exists(self):
    assert self.exists(), f'The audio file does not exist: {self.path}'
  
  def exists(self):
    if self.path is None:
      return False
    
    if os.path.isfile(self.path):
      return True
    
    return False
  
  @property
  @abc.abstractmethod
  def path(self) -> str:
    pass
  
  @abc.abstractmethod
  def download(self):
    pass
