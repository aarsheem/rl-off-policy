import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate(mu, sigma, size):
    x = np.random.normal(mu, sigma, size)
    return x

def estimate_WIS(x, q, p, f):
    px = p(x)
    qx = q(x)
    fx = f(x)
    out = px/qx*fx
    den =  px/qx
    out = np.cumsum(out, 1)/np.cumsum(den, 1)
    return out


if __name__ == "__main__":
    np.random.seed(2)
    q_mu = 0
    q_sigma = 1

    p_mu = 1
    p_sigma = 1

    p = norm(p_mu, p_sigma).pdf
    q = norm(q_mu, q_sigma).pdf
    f = lambda x:np.sin(x)

    size = [1000, 1000]
    x = generate(q_mu, q_sigma, size)

    actual = np.mean(f(generate(p_mu, p_sigma, 1000)))
    estimates = estimate_WIS(x, q, p, f)

    mean = np.mean(estimates, 0)
    std = np.std(estimates, 0)
    min_ = np.min(estimates, 0)
    max_ = np.max(estimates, 0)

    print("Actual: {},  Estimate: {}".format(actual, mean[-1]))

    axis = np.arange(mean.shape[0])
    plt.plot(axis, mean, label="mean")
    plt.fill_between(axis, mean-std, mean+std, alpha=0.3, label="std")
    plt.plot(axis, mean.shape[0]*[actual], label="actual")
    plt.xlabel("number of samples")
    plt.ylabel("values")
    plt.legend()
    plt.show()

