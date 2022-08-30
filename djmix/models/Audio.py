import os
import abc
from pydantic import BaseModel
from typing import Optional
from ..audio import *


class Audio(BaseModel, abc.ABC):
  path: Optional[str]
  
  def numpy(self, sr=None, mono=True, normalize=False):
    return load_audio(self.path, sr=sr, mono=mono, normalize=normalize, to_numpy=True)
  
  def pydub(self, sr=None, mono=True, normalize=False):
    return load_audio(self.path, sr=sr, mono=mono, normalize=normalize, to_numpy=False)
  
  def assert_exists(self):
    assert self.exists(), f'The audio file does not exist: {self.path}'
  
  def exists(self):
    if self.path is None:
      return False
    
    if os.path.isfile(self.path):
      return True
    
    return False
  
  @abc.abstractmethod
  def download(self):
    pass
