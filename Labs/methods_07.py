import pyro
import pyro.distributions as dist

def conditioned_scale_file(obs, guess=8.5): 
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    measurement = pyro.sample("measurement", dist.Normal(weight, 1.), obs=obs)
    return measurement

def eight_school_file(J, sigma, y=None):
    mu = pyro.sample("mu", dist.Normal(0., 5.))
    tau = pyro.sample("tau", dist.HalfCauchy(5.))
    with pyro.plate("schools", J):
        theta = pyro.sample("theta", dist.Normal(mu, tau))
        obs = pyro.sample("obs", dist.Normal(theta, sigma), obs=y)
    


def eight_schools_noncentered_file(J, sigma, y=None):
    mu = pyro.sample("mu", dist.Normal(0., 5.))
    tau = pyro.sample("tau", dist.HalfCauchy(5.))
    with pyro.plate("schools", J):
        nu = pyro.sample("nu", dist.Normal(0., 1.))
        theta = mu + tau * nu
        obs = pyro.sample("obs", dist.Normal(theta, sigma), obs=y)
