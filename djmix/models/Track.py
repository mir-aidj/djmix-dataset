from .Audio import Audio
from djmix import config, utils
from ..download import download_track
from typing import Optional


class Track(Audio):
  id: Optional[str]
  title: str
  
  def __repr__(self):
    return f'Track(id={self.id}, title="{self.title}")'
  
  def download(self):
    download_track(self)
  
  @property
  def path(self) -> Optional[str]:
    if self.id is None:
      return None
    return utils.mkpath(config.get_root(), 'tracks', self.id[0], f'{self.id}.mp3')
