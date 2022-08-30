import configparser
from typing import Any

from djmix import utils

__all__ = [
  'get_root',
  'set_root',
]

_config_path = utils.mkpath('~/.djmix.ini')
_config = configparser.ConfigParser()
_config.read(_config_path)
if not _config.has_section('djmix'):
  _config.add_section('djmix')
_section = _config['djmix']


def get_root():
  return utils.mkpath(_section.get('root', '~/djmix'))


def set_root(path: str):
  _set('root', path)


def _set(key: str, value: Any):
  _section[key] = value
  _save()


def _save():
  with open(_config_path, 'w') as f:
    _config.write(f)
