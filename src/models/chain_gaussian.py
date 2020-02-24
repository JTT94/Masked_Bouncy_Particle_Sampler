
from src.data_structures.factor_graph import FactorGraph
import numpy as np
from src.sampling_algorithms.event_time_samplers.bounce_time_samplers import gaussian_bounce, chain_bounce_fn
from src.utils.normal_helpers import gaussian_grad_potential_fn, gaussian_chain_grad_potential_fn
from src.utils import interp
from scipy.stats import norm

class Chain_Gaussian(FactorGraph):

    def __init__(self, params):

        # TODO: check params have right keys


        # define factor graph, and bounce fns
        # --------------------------------------------------------------

        # build graph
        self.factor_dim = params.factor_dim
        self.overlap = params.overlap
        self.num_factors = params.num_factors

        self.chain_length = self.factor_dim + (self.factor_dim - self.overlap) * (self.num_factors - 1)
        factor_indices = np.array([np.array([i + j for j in range(self.factor_dim)])
                                   for i in range(0, (self.factor_dim - self.overlap) * self.num_factors, self.factor_dim - self.overlap)])

        num_param = len(np.unique([x for y in factor_indices for x in y]))

        mus = np.repeat(params.mu, self.chain_length)
        prec = np.repeat(params.prec, self.chain_length)
        rho = np.repeat(params.rho, self.chain_length - 1)

        # multivariate gaussian bounce
        ind = factor_indices[0]
        Precs = [[prec[i] * prec[j] for j in ind] for i in ind]
        rhos = [[1 if i == j else rho[i] if j == i + 1 else rho[j] if i == j + 1 else 0 for j in ind] for i in ind]
        Prec = np.array(rhos) * np.array(Precs)
        self.Sig = np.linalg.pinv(Prec)

        grad_potential_fn = gaussian_grad_potential_fn(mu=mus[ind], Sig=self.Sig)
        gaus_bounce_fn = gaussian_bounce(mu=mus[ind], Sig=self.Sig)
        factor_potential_fns = [lambda x: -np.sum(norm.logpdf(x)) for _ in range(self.num_factors)]
        bounce_fns = [gaus_bounce_fn]
        grad_potential_fns = [grad_potential_fn]

        for ind in factor_indices[1:]:
            # extract params
            Precs = [[prec[i] * prec[j] for j in ind] for i in ind]
            rhos = [[1 if i == j else rho[i] if j == i + 1 else rho[j] if i == j + 1 else 0 for j in ind] for i in ind]
            Prec = np.array(rhos) * np.array(Precs)
            mu = mus[ind]
            num_steps = self.factor_dim - self.overlap
            Sig = np.linalg.pinv(Prec)

            mu2 = mu[:-num_steps]
            mu1 = mu[-num_steps:]
            Sig2 = Sig[:-num_steps, :-num_steps]
            Sig1 = Sig[-num_steps:, -num_steps:]
            Sig21 = Sig[:-num_steps, -num_steps:]
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
    def sample_mask(self, num_cuts):
        mask = np.repeat(1, self.chain_length)
        cuts = np.random.choice(list(range(0, self.chain_length, self.factor_dim - self.overlap)),num_cuts)
        mask[cuts]=0
        mask[-1] = 1
        mask[0] = 1
        return mask

    def split_mask_into_groups(self, factor_indices, mask):
        blocks = []
        current_block = []
        current_block.append(0)
        num_factors = len(factor_indices)
        for f in range(1, num_factors):
            prev_f = f - 1
            if np.sum(mask[np.intersect1d(factor_indices[prev_f], factor_indices[f])]) > 0:
                current_block.append(f)
            else:
                blocks.append(current_block)
                current_block = []
                current_block.append(f)
        blocks.append(current_block)
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
        x = interp(x1, t1, v1 * mask, num_intervals = 2*len(x1))
        return x


    def masked_get_x(self, i, results, factor_indices, groups):
        group_loc = self.get_which_group(i, groups, factor_indices)
        x = np.concatenate([self.masked_get_xs(results, s, i, group_loc[s]) for s in range(len(results))])
        return x