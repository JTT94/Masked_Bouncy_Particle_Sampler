import numpy as np


def gaussian_grad_potential_fn(mu, Sig):
    inv_sig = np.linalg.pinv(Sig)
    def grad_func(x):
        delta = x - mu
        return inv_sig.dot(delta)
    return grad_func


def gaussian_grad_potential_fn1d(mu, Sig):
    def grad_func(x):
        inv_sig = 1/Sig
        delta = x - mu
        return inv_sig * delta
    return grad_func


def gaussian_chain_grad_potential_fn(mu1, mu2, Sig1, Sig12, Sig2):
    inv_sig2 = np.linalg.pinv(Sig2)
    transform = np.dot(Sig12, inv_sig2)
    sig_bar = Sig1 - np.dot(transform, Sig12.T)
    inv_sig = np.linalg.inv(sig_bar)
    dim_x1 = len(mu1)

    mu = np.concatenate([mu1, mu2], 0)
    t2 = np.concatenate([-transform, np.diag(np.repeat(1., dim_x1))], 1)

    def grad_func(x):
        tx = t2.dot(x-mu)
        return t2.T.dot(inv_sig.dot(tx))

    return grad_func
