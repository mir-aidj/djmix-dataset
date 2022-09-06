# The DJ Mix Dataset

The DJ Mix dataset contains metadata of DJ mixes played by human DJs and played tracks in the mixes.

## Installation instructions
Tested on Python 3.9.

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
By default, it creates `djmix` directory at your home directory (i.e. `~/djmix`) and download audio files there.
If you want to change the directory:
```python
dj.set_root('~/my/custom/path/to/data/dir')
```
It will save a configuration file at `~/djmix.ini`.


## Tutorial
Please check [this tutorial jupyter note](notes/tutorial.ipynb).


## Citing
If you want to cite this dataset, please cite the paper below which introduced this dataset:
```
@inproceedings{taejun2022djmix,
  title={Joint Estimation of Fader and Equalizer Gains of DJ Mixers using Convex Optimization},
  author={Kim, Taejun and Yang, Yi-Hsuan and Nam, Juhan},
  booktitle={International Conference on Digital Audio Effects ({DAFx})},
  pages={312--319},
  year={2022}
}
```
