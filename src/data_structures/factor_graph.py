import numpy as np

class FactorGraph(object):
    def __init__(self, dim_x, factor_indices, factor_potential_fns, grad_factor_potential_fns):
        self.dim_x = dim_x
        self.factors = list(range(len(factor_indices)))
        self.factor_indices = factor_indices
        self.factor_potential_fns = factor_potential_fns
        self.grad_factor_potential_fns = grad_factor_potential_fns
        self.neighbour_map = self._neighbour_map()

    def potential(self, x):
        return np.sum(self.factor_potential_fns[f](x[ind])
                      for f, ind in enumerate(self.factor_indices))

    def factor_potential(self, f, x):
        return self.factor_potential_fns[f](x)

    def grad_potential(self, x):
        return np.sum(self.grad_factor_potential_fns[f](x[ind])
                      for f, ind in enumerate(self.factor_indices))

    def grad_factor_potential(self, f, x):
        return self.grad_factor_potential_fns[f](x)

    def _neighbour_map(self):
        neighbour_map = {}
        for f in self.factors:
            f_ind = self.factor_indices[f]
            neighbour_map[f] = [
                f_prime for (f_prime, ind) in enumerate(self.factor_indices)
                if len(np.intersect1d(f_ind, ind)) > 0]
        return neighbour_map
