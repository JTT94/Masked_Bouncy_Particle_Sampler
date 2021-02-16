from .gaussian import (gaussian_bounce, gaussian_bounce1d, zigzag_gaussian_bounce, chain_bounce_fn, chain_bounce, 
                       gaussian_chain_grad_potential_fn, gaussian_grad_potential_fn, gaussian_grad_potential_fn1d
                       )

from .logistic import (logistic, generate_logistic_bounce, alias_sample, lambda_r, grad_logistic, lambda_bound)


def aggregate_bounce(bounce_fns):
    def bounce_fn(x,v):
        return np.min([fn(x,v)[0] for fn in bounce_fns]), 'B'
    return bounce_fn