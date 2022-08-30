from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING
from yt_dlp import YoutubeDL
from djmix import config, utils

if TYPE_CHECKING:
  from djmix.models import Mix, Track


def download_mix(mix: Mix):
  logging.info(f'=> Start downloading {mix.id}')
  root = config.get_root()
  download_audio(
    url=mix.audio_url,
    path=utils.mkpath(root, 'mixes', f'{mix.id}.mp3')
  )


def download_track(track: Track):
  url = f'https://www.youtube.com/watch?v={track.id}'
  logging.info(f'=> Start downloading track {url}')
  root = config.get_root()
  track_path = utils.mkpath(root, 'tracks', track.id[0], f'{track.id}.mp3')
  download_audio(
    url=url,
    path=track_path,
  )


def download_audio(url, path):
  if os.path.isfile(path):
    logging.info(f'{path} already exists. Skip downloading.')
    return
  
  params = {
    'format': 'bestaudio',
    'outtmpl': path,
    'postprocessors': [{  # Extract audio using ffmpeg
      'key': 'FFmpegExtractAudio',
      'preferredcodec': 'mp3',
    }]
  }
  with YoutubeDL(params) as ydl:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ydl.download(url)
