import numpy as np
import time
from abc import ABC, abstractmethod
import tqdm

class LinearPDMCMC(ABC):
    def __init__(self, init_x, init_v):
        # set initial values
        self.d = len(init_x)
        self.t = np.repeat(0., self.d)

        if len(init_x) != len(init_v):
            raise Exception(
                'x and v should be same length. \n Currently of length x: {0} and v: {1} respectively'.format(
                    len(init_x), len(init_v)))

        self.x = init_x
        self.v = init_v


    def propagate_x(self, time_delta):
        self.x += time_delta * self.v
        self.t += time_delta

    def get_state(self):
        return self.x.copy(), self.v.copy(), self.t.copy()

    @abstractmethod
    def next_event(self):
        pass

    # simulate methods
    def simulate(self, num_events=1, burn_in_steps=10):
        results = []
        results.append(self.get_state())
        for _ in range(burn_in_steps):
            self.next_event()
        for _ in tqdm.tqdm(range(num_events)):
            self.next_event()
            results.append(self.get_state())
        return np.array(results)

    def simulate_up_to_T(self, max_T=10**3):
        (x,t,v) = self.get_state()
        latest_t = np.max(t)
        results = list()
        results.append([x, v, t])

        while latest_t < max_T:
            self.next_event()
            (x, t, v) = self.get_state()
            results.append([x, v, t])
            latest_t = np.max(t)
        return np.array(results)

    def simulate_for_time(self, time_limit=60):
        start = time.time()
        (x, v, t) = self.get_state()

        results = list()
        results.append([x, v, t])

        while time.time() - start < time_limit:
            self.next_event()
            (x, v, t) = self.get_state()
            results.append([x, v, t])
        return np.array(results)

