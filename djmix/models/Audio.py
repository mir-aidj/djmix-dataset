import os
import abc
from pydantic import BaseModel
from typing import Optional
from djmix import config, utils


class Audio(BaseModel, abc.ABC):
  path: Optional[str]
  
  def exists(self):
    if self.path is None:
      return False
    
    if os.path.isfile(self.path):
      return True
    
    return False
  
  @abc.abstractmethod
  def download(self):
    pass
