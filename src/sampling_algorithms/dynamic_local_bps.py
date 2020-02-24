import numpy as np
from scipy.stats import norm
from src.sampling_algorithms.local_bps import LocalBPS
from src.data_structures import PriorityQueue


class DynamicLocalBPS(LocalBPS):

    def __init__(self, init_x, init_v, factor_graphs, graph_bounce_fns, refresh_rate, verbose=False):

        # extra properties for multiple factor graphs
        self.factor_graphs = factor_graphs
        self.num_factorisations = len(factor_graphs)
        self.graph_bounce_fns = graph_bounce_fns

        l = self.sample_factorisation_idx()
        super(DynamicLocalBPS, self).__init__(init_x, init_v, factor_graphs[0], self.graph_bounce_fns[0],
                                              refresh_rate, verbose)

    def sample_factorisation_idx(self):
        graph_index = np.random.choice(self.num_factorisations)
        return graph_index

    def restart_factor_graph(self, t):

        # set initial values

        self.pq = PriorityQueue()

        l = self.sample_factorisation_idx()
        self.factor_graph = self.factor_graphs[l]
        self.bounce_fns = self.graph_bounce_fns[l]

        # init queue
        self.refresh_queue(t)

    def refresh_event(self):
        self.v = norm.rvs(size=self.factor_graph.dim_x)
        self.restart_factor_graph(self.refresh_time)
        self.refresh_time = self.refresh_time + self.new_refresh_time()

