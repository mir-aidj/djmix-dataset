from joblib import Memory
from djmix import config, utils

memory = Memory(utils.mkpath(config.get_root(), 'cache'), verbose=1)
