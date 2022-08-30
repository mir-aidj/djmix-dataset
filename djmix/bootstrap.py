import os
import json
import logging

from djmix import config, utils
from .models import Mix


def bootstrap():
  logging.basicConfig(level=logging.INFO)
  
  metadata_path = utils.mkpath(os.path.dirname(__file__), '../dataset/djmix-dataset.json')
  assert os.path.isfile(metadata_path), f'djmix-dataset.json does not exist at: {metadata_path}'
  
  json_mixes = json.load(open(metadata_path))
  
  data_dir = config.get_root()
  mixes = [
    Mix(data_dir, **json_mix)
    for json_mix in json_mixes
  ]
  
  tracks = {}
  for mix in mixes:
    for track in mix.tracklist:
      if track.id is None:
        continue
      tracks[track.id] = track
  
  return mixes, tracks
