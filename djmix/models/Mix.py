import logging
import traceback
from djmix import utils, config, alignment, cvxopt
from .Audio import Audio
from .Track import Track
from ..download import download_mix
from typing import List


class Mix(Audio):
  id: str
  title: str
  url: str
  audio_source: str
  audio_url: str
  num_identified_tracks: int
  num_total_tracks: int
  num_available_transitions: int
  num_timestamps: int
  tracklist: List[Track]
  tags: List
  
  def __init__(self, data_dir: str, **data):
    tracks = []
    for track in data['tracklist']:
      tracks.append(Track(**track))
    data['tracklist'] = tracks
    super().__init__(**data)
  
  def download(self):
    download_mix(self)
    
    for track in self.tracklist:
      try:
        if track.id is None:
          continue
        track.download()
      except Exception as e:
        logging.error(f'Failed to download mix: https://www.youtube.com/watch?v={track.id}')
        traceback.print_exc()
  
  def align(self, **kwargs):
    return alignment.align(self, **kwargs)
  
  def transitions(self):
    return alignment.transitions(self)
  
  def cvxopt(self, **kwargs):
    return cvxopt.optimize(self, **kwargs)
  
  @property
  def path(self) -> str:
    return utils.mkpath(config.get_root(), 'mixes', f'{self.id}.mp3')
  
  def __repr__(self):
    return f'Mix(id={self.id}, title="{self.title}")'
