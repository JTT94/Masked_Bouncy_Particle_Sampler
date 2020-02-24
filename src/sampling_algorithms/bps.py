from scipy.stats import norm, expon
from .linear_pdmcmc import LinearPDMCMC

class BPS(LinearPDMCMC):

    def __init__(self, init_x, init_v, bounce_fn, grad_entropy_fn, refresh_rate):
        # set initial values
        super().__init__(init_x, init_v)

        # refresh events
        self.refresh_rate = refresh_rate
        self.refresh_time = self.new_refresh_time()

        self.bounce_fn = bounce_fn
        self.grad_entropy_fn = grad_entropy_fn

    def simulate_bounce_time(self):
        return self.bounce_fn(self.x, self.v)

    def bounce_v(self):
        x = self.x.copy()
        v = self.v.copy()
        grad_fx = self.grad_entropy_fn(x)
        v = v - 2. * grad_fx.dot(v) * grad_fx / grad_fx.dot(grad_fx)
        self.v = v

    def new_refresh_time(self):
        if self.refresh_rate == 0:
            refresh_rate = 1/10**10
        else:
            refresh_rate = self.refresh_rate
        return expon.rvs(size=1, scale=1 / refresh_rate)[0]

    def refresh_event(self):
        self.v = norm.rvs(size=self.d)

    def next_event(self):
        bounce_time = self.simulate_bounce_time()
        self.refresh_time = self.new_refresh_time()
        if bounce_time < self.refresh_time:
            self.propagate_x(bounce_time)
            self.bounce_v()
        else:
            self.propagate_x(self.refresh_time)
            self.refresh_event()
