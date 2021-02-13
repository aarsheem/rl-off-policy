import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def generate(a, b, size):
    x = np.random.beta(a, b, size)
    return x

def estimate(x, q, p, f):
    px = p(x)
    qx = q(x)
    fx = f(x)
    out = px/qx*fx
    length = out.shape[1]
    out = np.cumsum(out, 1)/np.arange(1, length+1)
    return out


if __name__ == "__main__":
    np.random.seed(1)
    q_a = 2
    q_b = 2

    p_a = 2
    p_b = 5

    p = beta(p_a, p_b).pdf
    q = beta(q_a, q_b).pdf
    f = lambda x: x

    size = [1000, 1000]
    x = generate(q_a, q_b, size)

    actual = p_a/(p_a+p_b)
    estimates = estimate(x, q, p, f)

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
