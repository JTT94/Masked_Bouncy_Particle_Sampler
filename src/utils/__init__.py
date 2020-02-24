from .normal_helpers import gaussian_chain_grad_potential_fn, gaussian_grad_potential_fn, gaussian_grad_potential_fn1d
from .moments import get_first_moment, get_second_moment, get_var
from .params import Params, hash_dict
from .interpolation import interp, get_xtv, get_xs


def print_verbose(string, verbose = True):
    if verbose:
        print(string)
