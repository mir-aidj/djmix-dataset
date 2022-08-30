import abc


class Optimizer:

  @abc.abstractmethod
  def optimize(self, S_dj, S_prev, S_next, verbose=True):
    raise NotImplementedError

  @abc.abstractmethod
  def plot(
    self,
    S_est_prev_db,
    S_est_next_db,
    S_est_mix_db,
    title=None,
    true_curves=None,
    S_prev=None,
    S_next=None,
  ):
    raise NotImplementedError
