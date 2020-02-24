import numpy as np
import time


class ZigZag(object):

    def __init__(self, init_x, init_v, bounce_fns, refresh_rate):
        # set initial values
        self.d = len(init_x)
        self.t = np.repeat(0., self.d)
        self.x = init_x
        self.v = init_v

        # refresh events
        self.refresh_rate = refresh_rate

        if len(self.v) != len(self.x):
            raise Exception(
                'x and v should be same length. \n Currently of length x: {0} and v: {1} respectively'.format(
                    len(self.x), len(self.v)))

        self.bounce_fns = bounce_fns

    def simulate_bounce_time(self):
        # refresh times
        refresh_times = np.array(np.random.exponential(1/self.refresh_rate, self.d))

        # bounce proposals
        bounce_times = [self.bounce_fns[i](i, self.x, self.v) for i in range(self.d)]
        bounce_times = [t if t is not None else refresh_times[i] for i,t in enumerate(bounce_times)]

        event_times = np.minimum(bounce_times, refresh_times)

        i = np.argmin(event_times)
        t = event_times[i]

        return i, t

    def propagate_x(self, new_t):
        self.x = self.x + new_t * self.v
        self.t += new_t

    def bounce_v(self, i):
        v = self.v.copy()
        v[i] = -v[i]
        self.v = v

    def next_event(self):
        i, bounce_time = self.simulate_bounce_time()
        self.propagate_x(bounce_time)
        self.bounce_v(i)


    def get_state(self):
        return self.x.copy(), self.t.copy(), self.v.copy()

    def simulate(self, num_events=1, burn_in_steps=10):
        results = []
        results.append(self.get_state())
        for _ in range(burn_in_steps):
            self.next_event()
        for _ in range(num_events):
            self.next_event()
            results.append(self.get_state())
        return np.array(results)

    def simulate_up_to_T(self, max_T=10 ** 3):
        (x, t, v) = self.get_state()
        latest_t = np.max(t)

        results = list()
        results.append((x, t, v))

        while latest_t < max_T:
            self.next_event()
            (x, t, v) = self.get_state()
            results.append((x, t, v))
            latest_t = np.max(t)
        return np.array(results)

    def simulate_for_time(self, time_limit=60):
        start = time.time()
        (x,t,v) = self.get_state()

        results = list()
        results.append((x, t, v))

        while time.time() - start < time_limit:
            self.next_event()
            (x, t, v) = self.get_state()
            results.append((x, t, v))
        return np.array(results)
