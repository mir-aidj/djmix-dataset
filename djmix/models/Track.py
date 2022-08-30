from .Audio import Audio
from .download import download_track
from typing import Optional


class Track(Audio):
  id: Optional[str]
  title: str
  
  def __repr__(self):
    return f'Track(id={self.id}, title="{self.title}")'
  
  def download(self):
    download_track(self)
