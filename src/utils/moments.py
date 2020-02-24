import numpy as np

def get_first_moment(x1, v1, t1):
    niter = len(x1)
    lag_time = t1[1:] - t1[:niter-1]
    return np.sum(lag_time*x1[:niter-1] + v1[:niter-1]*lag_time**2/2)/t1[-1]


def get_second_moment(x1,v1,t1):
    niter = len(x1)
    lag_time = t1[1:] - t1[:niter-1]
    return np.sum(lag_time*x1[:niter-1]**2 + x1[:niter-1]*v1[:niter-1]*lag_time**2 + v1[:niter-1]**2*lag_time**3/3)/t1[-1]


def get_mean(x1, v1, t1):
    return get_first_moment(x1, v1, t1)


def get_var(x1, v1, t1):
    return get_second_moment(x1,v1,t1) - get_first_moment(x1, v1, t1)**2
