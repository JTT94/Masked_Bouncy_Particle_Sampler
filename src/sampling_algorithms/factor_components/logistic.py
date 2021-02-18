import numpy as np

def logistic(x):
    return 1/(1+np.exp(-x))

def lambda_r(v,x,t,covariates, y):
    e = np.exp(covariates.dot(x+t*v))
    return np.maximum(0.,(covariates*(e/(1+e)-y)).dot(v))

def alias_sample(i, cov_n, cov_p, covs, sign, observations):
    
    s = sign[i]
    cp = cov_p[i]
    cn = cov_n[i]
    y = observations[i]
    cov = covs[i]
    
    def bounce_fn(x,v):
        pos = (v*s)>=0
        neg = (v*s)<0
        ms = np.abs(v) * (pos*cp + neg*cn) + 1e-12
        
        m_i = np.sum(ms,0)
        m = np.sum(m_i)

        p = m_i/m

        k, = np.random.multinomial(1,p, size=1).ravel().nonzero()
        p = ms[:,k].flatten() / m_i[k]
        r, = np.random.multinomial(1,p, size=1).ravel().nonzero()
        
        u = np.random.rand()
        t = -np.log(u)/m
        if np.random.rand() < lambda_r(v,x,t,cov[r], y[r])/m:
            token = 'B'
        else:
            token = 'N'
        return t, token, r.item()
    return bounce_fn


def lambda_bound(v):
    return np.sum(np.abs(v))

def grad_logistic(covariates, obs):
    ys = obs
    covs = covariates
    def grad_fn(x, thin_factor=None):
        cov = covs[thin_factor]
        y = ys[thin_factor]
        
       
        e = np.exp(cov.dot(x))
        out = cov*(e/(1+e)-y)
        return out
    return grad_fn


def generate_logistic_bounce(covariates, y, limit=100, debug=False):
    def logistic_bounce(x,v):
        accepted = False
        count = 0
        while not accepted:
            count += 1
            lambda_sim = lambda_bound(v)
            u1 = np.random.rand()
            u2 = np.random.rand()
            t = -np.log(u1)/lambda_sim
            acc = lambda_r(v,x,t, covariates, y)/lambda_sim
            if debug:
                print(acc)
            if u2 < acc:
                accepted = True
            if count > limit:
                return float('inf'), 'B', None
        return t, 'B'
    return logistic_bounce

