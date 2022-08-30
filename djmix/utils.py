import os


def mkpath(*paths):
  return os.path.realpath(os.path.expanduser(os.path.join(*paths)))
