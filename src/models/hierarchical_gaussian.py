
from src.data_structures.factor_graph import FactorGraph
import numpy as np
from src.sampling_algorithms.event_time_samplers import gaussian_bounce1d, chain_bounce_fn
from src.utils.normal_helpers import gaussian_grad_potential_fn1d, gaussian_chain_grad_potential_fn
from ..utils import interp


class Hierarchical_Gaussian(FactorGraph):

    def __init__(self, params):

        # TODO: check params have right keys


        # define factor graph, and bounce fns
        # --------------------------------------------------------------
        # num local
        num_local = params.num_local

        # build graph
        #
        factor_indices_level_0 = [np.array([0])]
        factor_indices_level_1 = [np.array([0, i]) for i in range(1, params.num_local)]
        # factor_indices_level_2 = [np.array([i+1])
        #                           for i in range(num_local)]

        factor_indices = np.array(factor_indices_level_0 + factor_indices_level_1)

        num_param = len(np.unique([x for y in factor_indices for x in y]))

        prior_entropy = lambda x: x
        level_1_entropy = lambda x: x

        factor_potential_fns = [prior_entropy] + [
            level_1_entropy for _ in range(1, num_local)]

        # level 0 factors
        grad_potential_fn = gaussian_grad_potential_fn1d(mu=params.global_mu, Sig=1/params.global_prec)
        gaus_bounce_fn = gaussian_bounce1d(mu=params.global_mu, sig=1/params.global_prec)

        bounce_fns = [gaus_bounce_fn]
        grad_potential_fns = [grad_potential_fn]

        # conditional gaussian bounces
        rho = params.rho
        local_prec = params.local_prec**2
        mu = params.local_mu

        prec = np.array([[local_prec, rho*local_prec], [rho*local_prec, local_prec]])
        Sig = np.linalg.pinv(prec)

        for _ in factor_indices[1:num_local]:
            mu2 = np.array([mu])
            mu1 = np.array([mu])
            Sig2 = np.array([[Sig[1, 1]]])
            Sig1 = np.array([[Sig[0, 0]]])
            Sig21 = np.array([[Sig[0, 1]]])
            Sig12 = Sig21.T

            chain_grad_potential = gaussian_chain_grad_potential_fn(mu1, mu2, Sig1, Sig12, Sig2)
            chain_bounce_sampler = chain_bounce_fn(mu1, mu2, Sig1, Sig2, Sig12)

            bounce_fns.append(chain_bounce_sampler)
            grad_potential_fns.append(chain_grad_potential)

        # set methods and attributes
        self.bounce_fns = bounce_fns
        self.num_params = num_param
        self.params = params

        # init factor graph
        super().__init__(dim_x=num_param,
                         factor_indices=factor_indices,
                         factor_potential_fns=factor_potential_fns,
                         grad_factor_potential_fns=grad_potential_fns)


    # masked sampler
    # -----------------------------------------------------------------------------
    def sample_mask(self, num_cuts, num_param):
        mask = np.repeat(1, num_param)

        if np.random.rand(1)[0] < self.params.switch_prob:
            mask[0] = 0
        # else:
        # cuts = np.random.choice(np.arange(1,num_param),num_cuts)
        # mask[cuts]=0
        return mask


    def split_mask_into_groups(self, factor_indices, mask):
        blocks = []
        if mask[0] == 0:
            blocks = [[i] for i in np.arange(1, len(factor_indices))]
        else:
            blocks = [[i for i in np.arange(0, len(factor_indices))]]
        return blocks


    def get_which_group(self, i, groups, factor_indices):
        locs = []
        for g in range(np.shape(groups)[0]):
            for num, group in enumerate(groups[g]):
                if any(i in factor_indices[f] for f in group):
                    locs.append(num)
                    break
        return locs


    def masked_get_xs(self, results, s, i, g):
        res_temp = results[s][g]
        x1, v1, t1, mask = np.array(res_temp)[:, :, i].T
        x = interp(x1, t1, v1 * mask, num_intervals=2 * len(x1))
        return x


    def masked_get_x(self, i, results, factor_indices, groups):
        group_loc = self.get_which_group(i, groups, factor_indices)
        x = np.concatenate([self.masked_get_xs(results, s, i, group_loc[s]) for s in range(len(results))])
        return x

