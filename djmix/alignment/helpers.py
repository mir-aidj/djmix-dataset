import numpy as np
import pandas as pd


def diff(a, n=1):
  return a[n:] - a[:-n]


def correct_wp(wp, num_forgivable_points=2):
  diff_n = num_forgivable_points + 1

  wp_mix_fwd = wp[:, 1][::-1]
  wp_trk_fwd = wp[:, 0][::-1]
  wp_mix_bwd = wp[:, 1]
  wp_trk_bwd = wp[:, 0]

  dmix_fwd = diff(wp_mix_fwd, diff_n)
  dtrk_fwd = diff(wp_trk_fwd, diff_n)
  dmix_bwd = diff(wp_mix_bwd, diff_n)
  dtrk_bwd = diff(wp_trk_bwd, diff_n)

  crt_fwd = np.logical_and(dmix_fwd, dtrk_fwd)
  crt_bwd = np.logical_and(dmix_bwd, dtrk_bwd)[::-1]
  crt_both = np.logical_and(
    np.concatenate([crt_fwd, [True] * diff_n]),
    np.concatenate([[True] * diff_n, crt_bwd]),
  )
  wp_crt = wp[crt_both[::-1]]

  # Drop points of which x or y coordinate is duplicated.
  duplicated = (
    pd.Series(wp_crt[:, 0]).duplicated()
    | pd.Series(wp_crt[:, 1]).duplicated()
  )
  wp_crt_unique = wp_crt[~duplicated]

  return wp_crt_unique


def drop_weird_wp_segments(wp, num_forgivable_points=2, min_wp_beats=16, warp_points_per_beat=2):
  wp_diff = np.abs(np.concatenate([wp[1:], [wp[-1]]]) - wp).mean(axis=1)
  borders = np.flatnonzero(wp_diff > num_forgivable_points) + 1
  wp_groups = np.split(wp, borders)
  min_warp_points = min_wp_beats * warp_points_per_beat
  wp_long_enough = [wg for wg in wp_groups if len(wg) > min_warp_points]

  if len(wp_long_enough) == 0:
    # There is no wp survived.
    return None

  # Compute length, perform linear regression without outliers
  # to get intercepts (`b` from `y = ax + b`)
  wp_lengths, wp_intercepts = [], []
  for wp_seg in wp_long_enough:
    y, x = wp_seg[::-1].T
    dx, dy = np.diff(x), np.diff(y)
    slopes = dy / dx
    intercepts = y[:-1] - slopes * x[:-1]
    median_intercept = np.median(intercepts)
    wp_lengths.append(len(wp_seg))
    wp_intercepts.append(median_intercept)

  # Select warp path segments which lies down on the same line with the longest wp.
  wp_longest_intercept = wp_intercepts[np.argmax(wp_lengths)]
  wp_verified = [wp_seg for wp_seg, b in zip(wp_long_enough, wp_intercepts)
                 if abs(b - wp_longest_intercept) <= num_forgivable_points]
  wp_verified = np.concatenate(wp_verified)

  return wp_verified


def find_cue(wp, cue_in=False, num_diag=32):
  """
  Args:
    wp
    cue_in: if True, then output cue-in points, otherwise outputs cue-out points
    num_diag
  Returns:
    (cue point in beats on mix, cue point in beats on track)
  """
  if num_diag == 0:
    if cue_in:
      return wp[-1, 1], wp[-1, 0]
    else:
      return wp[0, 1], wp[0, 0]

  x, y = wp[::-1, 1], wp[::-1, 0]
  dx, dy = np.diff(x), np.diff(y)

  with np.errstate(divide='ignore'):
    slope = dy / dx
  slope[np.isinf(slope)] = 0

  if cue_in:
    slope = slope[::-1].cumsum()
    slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
    slope = slope[::-1]
    i_diag = np.nonzero(slope == num_diag)[0]
    if len(i_diag) == 0:
      return find_cue(wp, cue_in, num_diag // 2)
    else:
      i = i_diag[0]
      return x[i], y[i]
  else:
    slope = slope.cumsum()
    slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
    i_diag = np.nonzero(slope == num_diag)[0]
    if len(i_diag) == 0:
      return find_cue(wp, cue_in, num_diag // 2)
    else:
      i = i_diag[-1]
    return x[i] + 1, y[i] + 1


def extend_wp(wp, total_wpts):
  num_pad_befor = wp[-1, 0]
  num_pad_after = total_wpts - wp[0, 0] - 1

  pad_befor = np.stack([
    np.arange(0, num_pad_befor),
    np.arange(wp[-1, 1] - num_pad_befor, wp[-1, 1])
  ], axis=1)[::-1]

  pad_after = np.stack([
    np.arange(wp[0, 0] + 1, wp[0, 0] + 1 + num_pad_after),
    np.arange(wp[0, 1] + 1, wp[0, 1] + 1 + num_pad_after),
  ], axis=1)[::-1]

  wp_ext = np.concatenate([pad_after, wp, pad_befor])

  return wp_ext


def anchors(wp, start_wpt_mix, last_wpt_mix, halfbeats_mix, halfbeats_trk):
  mix2trk_wpt = {wpt_mix: wpt_trk for wpt_trk, wpt_mix, in wp}

  start_wpt = mix2trk_wpt[start_wpt_mix]
  last_wpt = mix2trk_wpt[last_wpt_mix]  # TODO: KeyError

  wpt_trk, wpt_mix = wp[::-1].T
  i_start_wpt, i_last_wpt = np.searchsorted(wpt_trk, [start_wpt, last_wpt])

  i_anchors_trk = wpt_trk[i_start_wpt:i_last_wpt + 1]
  i_anchors_mix = wpt_mix[i_start_wpt:i_last_wpt + 1]

  anchors_trk = halfbeats_trk[i_anchors_trk]
  anchors_mix = halfbeats_mix[i_anchors_mix]

  # Include the end beat.
  anchors_trk = np.concatenate([anchors_trk, [halfbeats_trk[i_anchors_trk[-1] + 1]]])
  anchors_mix = np.concatenate([anchors_mix, [halfbeats_mix[i_anchors_mix[-1] + 1]]])

  return anchors_trk, anchors_mix


def project_wp_raw(wp, track_cue_in, track_cue_out, mix_cue_in, mix_cue_out):
  pad_stop_befor = track_cue_in
  pad_befor = np.stack([
    np.arange(0, pad_stop_befor),
    np.arange(mix_cue_in - pad_stop_befor, mix_cue_in),
  ], axis=1)[::-1]

  track_total_beats = wp[0, 0] + 1
  pad_stop_after = track_total_beats - track_cue_out
  pad_after = np.stack([
    np.arange(track_cue_out + 1, track_cue_out + pad_stop_after),
    np.arange(mix_cue_out + 1, mix_cue_out + pad_stop_after),
  ], axis=1)[::-1]

  wp_ext = wp[(track_cue_in <= wp[:, 0]) & (wp[:, 0] <= track_cue_out)]
  wp_ext = np.concatenate([pad_after, wp_ext, pad_befor])

  return wp_ext
