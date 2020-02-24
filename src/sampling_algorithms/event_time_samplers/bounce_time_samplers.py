import numpy as np

def gaussian_bounce(mu, Sig):

    inv_Sig = np.linalg.inv(Sig)

    def func(x,v):
        epsilon = np.random.rand()
        vv = 0.5*v.T.dot(inv_Sig).dot(v)
        xv = 0.5*v.T.dot(inv_Sig).dot(x-mu)
        if xv <0:
            return (-xv + np.sqrt(-vv*np.log(epsilon)))/ vv
        else:
            return (-xv + np.sqrt(xv**2 - vv*np.log(epsilon)))/vv
    return func


def zigzag_gaussian_bounce(mu, Sig):

    inv_Sig = np.linalg.inv(Sig)

    def func(i, x,v):
        epsilon = np.random.rand()

        b = v[i] * (x-mu).dot(inv_Sig[i,:])
        a = v[i] * v.dot(inv_Sig[i, :])
        discrim = np.max([0., b])**2 - 2*a*np.log(epsilon)

        if discrim > 0:
            return (-b + np.sqrt(discrim)) / a
        else:
            return None
    return func

def gaussian_bounce1d(mu, sig):

    inv_sig = 1/sig


    def func(x,v):
        epsilon = np.random.rand()
        vv = 0.5*v*v*inv_sig
        xv = 0.5*v*inv_sig*(x-mu)
        if xv < 0:
            return (-xv + np.sqrt(-vv*np.log(epsilon))) / vv
        else:
            return (-xv + np.sqrt(xv**2 - vv*np.log(epsilon))) / vv
    return func


def chain_bounce(x1, x2, v1, v2, mu1, mu2, inv_sig, transform):
    """
    :param x1:
    :param x2:
    :param v1:
    :param v2:
    :param mu1:
    :param mu2:
    :param inv_sig:
    :param transform:
    :return:
    """
    def inner_product(a, b):
        return 0.5*a.T.dot(inv_sig).dot(b)

    v_diff = v1 - transform.dot(v2)
    x_diff = x1 - mu1 - transform.dot(x2 - mu2)

    a = inner_product(v_diff, v_diff)
    b = inner_product(x_diff, v_diff)

    epsilon = np.random.random()
    if b < 0:
        out = -b + np.sqrt(-a * np.log(epsilon))
    else:
        out = -b + np.sqrt(b ** 2 - a * np.log(epsilon))
    return out / a


def chain_bounce_fn(mu1, mu2, Sig1, Sig2, Sig12):
    """
    :param mu1:
    :param mu2:
    :param Sig1:
    :param Sig2:
    :param Sig12:
    :return:
    """
    inv_sig2 = np.linalg.inv(Sig2)
    sig_bar = Sig1 - Sig12.dot(inv_sig2).dot(Sig12.T)
    inv_sig = np.linalg.pinv(sig_bar)
    transform = Sig12.dot(inv_sig2)

    def func(x, v):
        x1 = x[1:]
        v1 = v[1:]

        if len(x) == 2:
            x1 = np.expand_dims(x1, 0)
            v1 = np.expand_dims(v1, 0)
        x2 = x[0]
        x2 = np.expand_dims(x2, 0)
        v2 = v[0]
        v2 = np.expand_dims(v2, 0)

        return chain_bounce(x1, x2, v1, v2, mu1, mu2, inv_sig, transform)

    return func
