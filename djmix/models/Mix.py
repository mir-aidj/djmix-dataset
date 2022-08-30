import logging
import traceback
from djmix import utils
from .Audio import Audio
from .Track import Track
from .download import download_mix
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
    mix_id = data['id']
    data['path'] = utils.mkpath(data_dir, 'mixes', f'{mix_id}.mp3')
    
    tracks = []
    for track in data['tracklist']:
      track_id = track['id']
      if track_id is None:
        track['path'] = None
      else:
        track['path'] = utils.mkpath(data_dir, 'tracks', track_id[0], f'{track_id}.mp3')
      tracks.append(Track(**track))
    data['tracklist'] = tracks
    
    super().__init__(**data)
  
  def __repr__(self):
    return f'Mix(id={self.id}, title="{self.title}")'
  
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
