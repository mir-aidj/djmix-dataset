import os
import logging
import json
import traceback
from yt_dlp import YoutubeDL


def mkpath(*paths):
  return os.path.realpath(os.path.join(*paths))


DATASET_DIR = './dataset'
MIX_DIR = mkpath(DATASET_DIR, 'mixes')
TRACK_DIR = mkpath(DATASET_DIR, 'tracks')


def main():
  logging.basicConfig(level=logging.INFO)
  
  os.makedirs(MIX_DIR, exist_ok=True)
  os.makedirs(TRACK_DIR, exist_ok=True)
  
  mixes = json.load(open(mkpath(DATASET_DIR, 'djmix-dataset.json')))
  
  for mix in mixes:
    try:
      download_mix(mix)
    except Exception as e:
      logging.error(f'Failed to download {mix["id"]}')
      traceback.print_exc()
    
    for track in mix['tracklist']:
      try:
        if track["id"] is None:
          continue
        download_track(track)
      except Exception as e:
        logging.error(f'Failed to download track: https://www.youtube.com/watch?v={track["id"]}')
        traceback.print_exc()


def download_mix(mix):
  logging.info(f'=> Start downloading {mix["id"]}')
  
  download_audio(
    url=mix['audio_url'],
    path=mkpath(MIX_DIR, f'{mix["id"]}.mp3')
  )


def download_track(track):
  url = f'https://www.youtube.com/watch?v={track["id"]}'
  logging.info(f'=> Start downloading track {url}')
  
  track_path = mkpath(TRACK_DIR, track["id"][0], f'{track["id"]}.mp3')
  os.makedirs(os.path.dirname(track_path), exist_ok=True)
  
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
    ydl.download(url)


if __name__ == '__main__':
  main()
