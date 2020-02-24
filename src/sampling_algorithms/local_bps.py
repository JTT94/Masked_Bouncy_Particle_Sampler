import numpy as np
from scipy.stats import norm, expon
from src.data_structures import PriorityQueue
from src.sampling_algorithms.linear_pdmcmc import LinearPDMCMC

class LocalBPS(LinearPDMCMC):

    def __init__(self, init_x, init_v, factor_graph, bounce_fns, refresh_rate):
        super().__init__(init_x, init_v)
        # set initial values
        self.bounce_fns = bounce_fns
        self.factor_graph = factor_graph
        self.pq = PriorityQueue()

        # refresh events
        self.refresh_rate = refresh_rate
        self.refresh_time = self.new_refresh_time()

        # init queue
        self.refresh_queue(t=0.)

    def propagate_factor_x(self, f, new_t):
        factor_ind = self.factor_graph.factor_indices[f]
        prev_x = self.x[factor_ind].copy()
        t_delta = new_t - self.t[factor_ind].copy()
        self.x[factor_ind] = prev_x + t_delta * self.v[factor_ind].copy()
        self.t[factor_ind] = new_t

    def propagate_neighbours(self, f, t):
        f_to_update = self.factor_graph.neighbour_map[f]
        for f_prime in f_to_update:
            self.propagate_factor_x(f_prime, t)

    def update_queue(self, f, t):
        f_to_update = self.factor_graph.neighbour_map[f]

        for f_prime in f_to_update:
            factor_ind = self.factor_graph.factor_indices[f_prime]
            x = self.x[factor_ind].copy()
            v = self.v[factor_ind].copy()
            bounce_time = self.bounce_fns[f_prime](x, v)
            self.pq.add_item(f_prime, t + bounce_time)

    def bounce_factor(self, f):
        factor_ind = self.factor_graph.factor_indices[f]
        x = self.x[factor_ind].copy()
        v = self.v[factor_ind].copy()
        grad_fx = self.factor_graph.grad_factor_potential(f, x)
        new_v = v - 2. * grad_fx.dot(v) * grad_fx / grad_fx.dot(grad_fx)
        self.v[factor_ind] = new_v

    def new_refresh_time(self):
        return expon.rvs(size=1, scale=1 / self.refresh_rate)[0]

    def refresh_queue(self, t):
        for f in self.factor_graph.factors:
            factor_ind = self.factor_graph.factor_indices[f]
            x = self.x[factor_ind].copy()
            v = self.v[factor_ind].copy()
            bounce_time = self.bounce_fns[f](x, v)
            self.pq.add_item(f, t + bounce_time)

    def refresh_event(self):
        self.v = norm.rvs(size=self.factor_graph.dim_x)
        self.refresh_queue(self.refresh_time)
        self.refresh_time = self.refresh_time + self.new_refresh_time()

    def next_event(self):
        f, bounce_time = self.pq.pop_task()
        if bounce_time < self.refresh_time:
            self.propagate_neighbours(f, bounce_time)
            self.bounce_factor(f)
            self.update_queue(f, bounce_time)
        else:
            self.propagate_x(self.refresh_time- self.t)
            self.refresh_event()



