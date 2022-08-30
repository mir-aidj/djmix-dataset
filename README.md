# The DJ Mix Dataset

The DJ Mix dataset contains metadata of DJ mixes played by human DJs and played tracks in the mixes.

## How to install `djmix` Python package

### Install FFmpeg

```shell
# For Debian/Ubuntu:
sudo apt-get install ffmpeg

# For OSX:
brew install ffmpeg
```

### Install the Package

```shell
pip install -U pip
pip install numpy cython  # should be installed beforehand for madmom
pip install djmix
```

## Downloading audio files

In Python console:

```python
import djmix as dj

# To download all mix and track audio files:
dj.download()

# To download a specific mix and played tracks in the mix:
dj.mixes[1234].download()

# To download a track:
dj.tracks['Uow3dMA5m14'].download()
```