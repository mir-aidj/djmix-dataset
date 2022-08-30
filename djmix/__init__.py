from djmix.bootstrap import bootstrap
from djmix.download import download
from djmix.config import *

__all__ = [
  'mixes',
  'tracks',
  'get_root',
  'set_root',
  'download',
]

mixes, tracks = bootstrap()
