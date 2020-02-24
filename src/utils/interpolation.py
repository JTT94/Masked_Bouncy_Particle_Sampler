import numpy as np
import pandas as pd


def interp(x, t, v, num_intervals=10000):
    fixed_intervals = np.linspace(t[0], t[-1], num_intervals)
    max_t = np.max(t)
    fixed_intervals = [t for t in fixed_intervals if t < max_t]
    data_df = pd.DataFrame({'x': x, 'v': v, 't': t, 'original_t': t})

    t_df = pd.DataFrame({'t': fixed_intervals})
    df = pd.merge_asof(t_df, data_df, on='t', direction='backward')

    df.eval('new_x = x+(t-original_t)*v', inplace=True)
    return df.new_x.values


def get_xtv(res, coord, final_n=None):
    if final_n is None:
        x = res[:, 0, coord]
        v = res[:, 1, coord]
        t = res[:, 2, coord]

    else:
        x = res[-final_n:, 0, coord]
        v = res[-final_n:, 1, coord]
        t = res[-final_n:, 2, coord]

    return x, v, t


def get_xs(res, i, num_intervals):
    x1, v1, t1 = get_xtv(res, i)
    x = interp(x1, t1, v1, num_intervals)
    return x

