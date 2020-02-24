import numpy as np
from .linear_pdmcmc import LinearPDMCMC

class CoordinateSampler(LinearPDMCMC):

    def __init__(self, init_x, grad_potential_fn, bounce_fn, refresh_rate):
        # set initial values
        self.d = len(init_x)
        v = np.repeat(0,self.d)
        super().__init__(init_x, v)

        # refresh events
        self.refresh_rate = refresh_rate

        self.bounce_fn = bounce_fn
        self.grad_potential_fn = grad_potential_fn
        self.sample_v()

    def sample_v(self):
        grad = self.grad_potential_fn(self.x)
        vs = np.array([self.v_i(i) if i < self.d else -self.v_i(i % self.d) for i in range(2 * self.d)])
        probs = np.array([np.max([0,grad.dot(v)]) for v in vs]) + self.refresh_rate
        probs = probs / np.sum(probs)
        i = np.random.choice(range(len(vs)), size=1,p=probs)
        self.v = -vs[i].flatten()

    def simulate_bounce_time(self):
        return self.bounce_fn(self.x, self.v)

    def v_i(self, i):
        v = np.repeat(0, self.d)
        v[i] = 1
        return v

    def next_event(self):
        t = self.simulate_bounce_time()
        self.propagate_x(t)
        self.sample_v()

