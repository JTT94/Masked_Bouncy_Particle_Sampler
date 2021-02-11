import numpy as np
from scipy.stats import norm, expon
from src.data_structures.factor_graph import FactorGraph
from src.data_structures import PriorityQueue
import time
from src.utils import interp, print_verbose
from src.utils.serialize import pickle_obj, unpickle_obj, load_json, save_json
from src.sampling_algorithms.linear_pdmcmc import LinearPDMCMC
import ray
import os


@ray.remote
class SubMaskedLocalBPS(LinearPDMCMC):

    def __init__(self, handles, factor_group = None):
        x_handle, v_handle, t_handle, mask_handle, factor_graph_handle, bounce_fns_handle = handles
        
        # set initial values
        init_x = ray.get(x_handle)
        init_v = ray.get(v_handle)
        init_t = ray.get(t_handle)
        self.mask = ray.get(mask_handle)

        super().__init__(init_x.copy(), init_v.copy())
        self.t = init_t


        # load globals
        self.global_factor_graph = ray.get(factor_graph_handle)
        self.global_bounce_fns = ray.get(bounce_fns_handle)
        
        if factor_group is None:
            self.factor_graph_set = False
        else:
            self.re_init(handles, factor_group)
        
    
    def re_init(self, handles, factor_group):
        x_handle, v_handle, t_handle, mask_handle, factor_graph_handle, bounce_fns_handle = handles
        # set initial values
        self.x = ray.get(x_handle).copy()
        self.v = ray.get(v_handle).copy()
        self.t = ray.get(t_handle).copy()
        self.mask = ray.get(mask_handle).copy()
        
        # set factor graph
        subset_indices = [self.global_factor_graph.factor_indices[factor] for factor in factor_group]
        self.factor_graph = FactorGraph(dim_x=self.d,
                                factor_indices=subset_indices,
                                factor_potential_fns=[self.global_factor_graph.factor_potential_fns[factor] 
                                                      for factor in factor_group],
                                grad_factor_potential_fns=[self.global_factor_graph.grad_factor_potential_fns[factor] 
                                                           for factor in factor_group])
        self.bounce_fns = [self.global_bounce_fns[factor] for factor in factor_group]
        self.factor_group = factor_group
        self.factor_graph_set = True
        
        # queue
        self.pq = PriorityQueue()
        start_t = np.max(self.t)
        self.refresh_queue(t=start_t)
                  
        
    def propagate_factor_x(self, f, new_t):
        factor_ind = self.factor_graph.factor_indices[f]
        effective_v = self.v * self.mask
        x_new = self.x.copy()
        t_new = self.t.copy()

        x_new[factor_ind] = self.x[factor_ind].copy() + (new_t - self.t[factor_ind].copy()) * effective_v[factor_ind].copy()
        t_new[factor_ind] = new_t

        self.t = t_new
        self.x = x_new

    def propagate_x(self, new_t):
        effective_v = self.v * self.mask
        self.x = self.x + (new_t - self.t) * effective_v
        self.t = np.repeat(new_t, self.factor_graph.dim_x)

    def propagate_neighbours(self, f, t):
        f_to_update = self.factor_graph.neighbour_map[f]

        for f_prime in f_to_update:
            self.propagate_factor_x(f_prime, t)

    def update_queue(self, f, t):
        f_to_update = self.factor_graph.neighbour_map[f]

        effective_v = self.v * self.mask
        for f_prime in f_to_update:
            factor_ind = self.factor_graph.factor_indices[f_prime]
            x = self.x[factor_ind].copy()
            v = effective_v[factor_ind].copy()
            if np.sum(v) != 0.:
                bounce_time = self.bounce_fns[f_prime](x, v)
                self.pq.add_item(f_prime, t + bounce_time)

    def bounce_factor(self, f):
        factor_ind = self.factor_graph.factor_indices[f]
        x = self.x[factor_ind].copy()
        v = self.v[factor_ind].copy()
        mask = self.mask[factor_ind].copy()

        grad_fx = self.factor_graph.grad_factor_potential(f, x) * mask
        v = v - 2. * grad_fx.dot(v) * grad_fx / grad_fx.dot(grad_fx)
        self.v[factor_ind] = v.copy()

    def next_event(self, f, bounce_time):

        self.propagate_neighbours(f, bounce_time)
        self.bounce_factor(f)
        self.update_queue(f, bounce_time)

    def refresh_queue(self, t):
        for f in self.factor_graph.factors:
            factor_ind = self.factor_graph.factor_indices[f]
            x = self.x[factor_ind].copy()
            v = self.v[factor_ind].copy()
            mask = self.mask[factor_ind].copy()
            effective_v = v*mask
            if np.sum(effective_v) != 0.:
                bounce_time = self.bounce_fns[f](x, effective_v)
                self.pq.add_item(f, t + bounce_time)

    def get_state(self):
        return self.x.copy(), self.v.copy(), self.t.copy(), self.mask.copy()

    def simulate_up_to_T(self, max_t=10**3):
        (x, v, t, mask) = self.get_state()
        latest_t = np.max(t)

        results = list()
        results.append(np.array([x, v, t, mask]))

        keep_going = True
        while keep_going:# latest_t < max_t:
            f, bounce_time = self.pq.pop_task()
            if bounce_time < max_t:
                self.next_event(f, bounce_time)
                (x, v, t, mask) = self.get_state()
                results.append(np.array([x, v, t, mask]))
            else:
                self.propagate_x(max_t)
                keep_going = False
        
        factor_ind = np.unique(np.concatenate(self.factor_graph.factor_indices, axis = 0 ))
        sub_res = np.array(results)[:,:,factor_ind]
        
        return sub_res
    
    def run_to_T(self, handles, factor_group, max_t, output_fp):
        self.re_init(handles, factor_group)
        sub_res = self.simulate_up_to_T(max_t)
        pickle_obj(sub_res, output_fp)
        return self.x
    
    def terminate(self):
        ray.actor.exit_actor()


class MaskedLocalBPS(object):
    def __init__(self, init_x, 
                 init_v, 
                 init_mask, 
                 factor_graph, 
                 bounce_fns, 
                 refresh_rate, 
                 split_mask_fn, 
                 sample_mask_fn, 
                 max_number_sub_samplers):
                           
        # set initial values
        self.mask = init_mask
        self.d = len(init_x)
        self.t = np.repeat(0., self.d)
        self.x = init_x
        self.v = init_v
        self.share_params()
        self.max_number_sub_samplers = max_number_sub_samplers
        

        # refresh events
        self.refresh_rate = refresh_rate
        self.refresh_time = self.new_refresh_time()

        if len(self.v) != len(self.x):
            raise Exception(
                'x and v should be same length. \n Currently of length x: {0} and v: {1} respectively'.format(
                    len(self.x), len(self.v)))

        # factor graph
        self.factor_graph = factor_graph
        self.bounce_fns = bounce_fns
        self.factor_graph_handle = ray.put(factor_graph)
        self.bounce_fns_handle = ray.put(bounce_fns)

        # mask fns
        self.sample_mask_fn = sample_mask_fn
        self.split_mask_fn = split_mask_fn
        
        # init sub samplers
        self.sub_samplers = [SubMaskedLocalBPS.remote(handles = self.get_handles()) 
                             for _ in range(max_number_sub_samplers)]
         
    
    def share_params(self):
        self.x_handle = ray.put(self.x)
        self.t_handle = ray.put(self.t)
        self.v_handle = ray.put(self.v)
        self.mask_handle = ray.put(self.mask)
    
    def get_handles(self):
        return self.x_handle, self.v_handle, self.t_handle, self.mask_handle, self.factor_graph_handle, self.bounce_fns_handle

    def get_state(self):
        return self.x.copy(), self.v.copy(), self.t.copy(), self.mask.copy()

    def new_refresh_time(self):
        return expon.rvs(size=1, scale=1 / self.refresh_rate)[0]

    def refresh_event(self):
        self.v = norm.rvs(size=self.factor_graph.dim_x)
        self.refresh_time = self.refresh_time + self.new_refresh_time()
        self.mask = self.sample_mask_fn()

    def aggregate_subs(self, sub_x, factor_group):
        for f in factor_group:
            ind = self.factor_graph.factor_indices[f]
            self.x[ind] = sub_x[ind]

    def run_iteration(self, iteration = 0, output_dir = "./"):
        # current state operations
        self.share_params()
        state = self.get_state()
        pickle_obj(state, os.path.join(output_dir, "states_iteration{0}.pickle".format(iteration)))
        
        factor_groups = self.split_mask_fn(self.factor_graph.factor_indices, self.mask)
        handles = self.get_handles() 

        # distributed operations
        start_time = time.time()
        new_x = np.array(ray.get([
            self.sub_samplers[i].run_to_T.remote(handles, 
                                                 factor_groups[i], 
                                                 self.refresh_time,
                                                os.path.join(output_dir, 
                                                             "sub_results_iter{0}_group{1}.pickle".format(iteration, i))) 
                                    for i in range(len(factor_groups))]))
        
        time_delta = time.time() - start_time
        
        for i in range(len(factor_groups)):
            x = new_x[i]
            factor_group = factor_groups[i]
            self.aggregate_subs(x, factor_group)
        self.t = np.repeat(self.refresh_time, self.d)

        self.refresh_event()
        return time_delta
        #return time_delta

    def simulate_for_time(self, time_limit=60, output_dir = "./"):
        cumulative_time = 0.
        states = list()
        masks = list()
        groups = list()
        iteration = 0

        while cumulative_time < time_limit:
            states.append(self.get_state())
            groups.append(self.split_mask_fn(self.factor_graph.factor_indices, self.mask))
            
           # iteration_res, 
            time_delta = self.run_iteration(iteration, output_dir)
            cumulative_time += time_delta
            
            iteration += 1
        return states, groups, masks

    def get_which_group(self, i, groups, factor_indices):
        locs = []
        for g in range(np.shape(groups)[0]):
            for num, group in enumerate(groups[g]):
                if any(i in factor_indices[f] for f in group):
                    locs.append(num)
                    break
        return locs

    def masked_get_xs(self, results, s, i, g, num_draws):
        x1, v1, t1, mask = np.array(results[s][g])[:, :, i].T
        num_draws = np.max([num_draws, len(x1)])
        x = interp(x1, t1, v1*mask, num_intervals=num_draws)
        return x

    def g(self, i, results, factor_indices, groups, num_draws):
        group_loc = self.get_which_group(i, groups, factor_indices)
        x = np.concatenate([self.masked_get_xs(results, s, i, group_loc[s], num_draws) for s in range(len(results))])
        return x
