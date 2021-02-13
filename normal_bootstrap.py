import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def bootstrap(z, B, g):
    unbiased = np.zeros(z.shape)
    bias = np.zeros(z.shape)
    biased = np.zeros(z.shape)
    for i in range(z.shape[0]):
        z_ = np.mean(z[:i+1])
        zb = np.random.choice(z[:i+1], size=[B, i+1])
        zb_ = np.mean(zb, 1)
        theta_cap = g(z_)
        theta_b = g(zb_)
        bias_star = np.mean(theta_b) - theta_cap
        theta = theta_cap - bias_star
        unbiased[i] = theta
        biased[i] = theta_cap
        bias[i] = bias_star
    return unbiased, bias, biased

if __name__ == "__main__":
    mu = 1
    sigma = 1
    size = 100
    iterations = 20
    B = 100
    g = lambda x: 7*x**3 - 5*x*x - 1

    z = np.random.normal(mu, sigma, size=[iterations, size])
    unbiased = np.zeros([iterations, size])
    bias = np.zeros([iterations, size])
    biased = np.zeros([iterations, size])

    for i in tqdm(range(iterations)):
        unbiased[i], bias[i], biased[i] = bootstrap(z[i], B, g)

    unbiased_mean = np.mean(unbiased, 0)
    unbiased_std = np.std(unbiased, 0)
    bias_mean = np.mean(bias, 0)
    bias_std = np.std(bias, 0)
    biased_mean = np.mean(biased, 0)
    biased_std = np.std(biased, 0)

    x = np.arange(size)
    plt.plot(x, size*[g(mu)], 'k', label="actual")
    plt.plot(x, unbiased_mean, 'b', label="unbiased mean")
    plt.plot(x, bias_mean, 'y', label="bias mean")
    plt.plot(x, biased_mean, 'r', label="biased mean")
    plt.fill_between(x, unbiased_mean-unbiased_std, unbiased_mean+unbiased_std, color='b', alpha=0.3, label="unbiased_std")
    plt.fill_between(x, bias_mean-bias_std, bias_mean+bias_std, color='y', alpha=0.3, label="bias_std")
    plt.fill_between(x, biased_mean-biased_std, biased_mean+biased_std, color='r', alpha=0.3, label="biased_std")
    plt.xlabel("samples")
    plt.legend()
    plt.show()

