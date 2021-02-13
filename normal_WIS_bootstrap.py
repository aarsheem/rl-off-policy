import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

def generate(mu, sigma, size):
    x = np.random.normal(mu, sigma, size)
    return x

def estimate_WIS(x, q, p, f, P=1):
    px = p(x) * P
    qx = q(x)
    fx = f(x)
    num = px/qx*fx
    den =  px/qx
    if x.ndim == 2:
        out = np.cumsum(num, 1)/np.cumsum(den, 1)
    else:
        out = np.cumsum(num)/np.cumsum(den)
    return out

def bootstrap(z, B, g, samples):
    bias_star = np.zeros(z.shape)
    theta_cap = g(z)
    for i in range(z.shape[0]):
        bias_star[i] = np.mean(g(z[samples[i]])[:,-1]) - theta_cap[i]
    theta = theta_cap - bias_star
    unbiased = theta
    biased = theta_cap
    bias = bias_star
    return unbiased, bias, biased

#chapter 10 of bootstrap book
def bootstrap2(z, B, g, samples, P):
    bias_star = np.zeros(z.shape)
    for i in range(z.shape[0]):
        theta_cap = g(z[:i+1], P[i])[-1]
        bias_star[i] = np.mean(g(z[samples[i]])[:,-1]) - theta_cap
    theta = theta_cap - bias_star
    unbiased = theta
    biased = theta_cap
    bias = bias_star
    return unbiased, bias, biased

def bias_std_error(actual, estimates):
    bias = np.mean(estimates - actual, 0)
    std = np.std(estimates, 0)
    avg_error = np.mean(np.abs(estimates - actual), 0)
    return bias, std, avg_error

def plot_mean(bWIS, WIS, text, zero=True, line=""):
    x = np.arange(WIS.shape[0])
    plt.plot(x, WIS, 'r'+line, label="WIS "+text)
    plt.plot(x, bWIS, 'b'+line, label="bWIS "+text)
    if zero:
        plt.plot(x, np.zeros(x.shape), 'k')
    plt.xlabel("samples")
    plt.legend()

if __name__ == "__main__":
    np.random.seed(2)
    q_mu = 0
    q_sigma = 1

    p_mu = 0.3
    p_sigma = 1

    p = norm(p_mu, p_sigma).pdf
    q = norm(q_mu, q_sigma).pdf
    f = lambda x: x*x - x

    size = 30
    iterations = 500
    B = 100

    def g(x, P=1):
        return estimate_WIS(x, q, p, f, P)

    z = generate(q_mu, q_sigma, size=[iterations, size])
    unbiased = np.zeros([iterations, size])
    bias = np.zeros([iterations, size])
    biased = np.zeros([iterations, size])

    WIS = estimate_WIS(z, q, p, f)

    samples = [[] for i in range(size)]
    P = [[] for i in range(size)]
    for i in range(size):
        samples[i] = np.random.choice(i+1, size=[B, i+1])
        P[i] = np.bincount(samples[i].flatten(), minlength=i+1)/B/(i+1)

    for i in tqdm(range(iterations)):
       unbiased[i], bias[i], biased[i] = bootstrap2(z[i], B, g, samples, P)

    actual = np.mean(f(generate(p_mu, p_sigma, 100000)))
    bWIS_bias, bWIS_std, bWIS_err = bias_std_error(actual, unbiased)
    WIS_bias, WIS_std, WIS_err = bias_std_error(actual, WIS)

    plot_mean(bWIS_bias, WIS_bias, "bias")
    #plt.figure()
    #plot_mean(bWIS_std, WIS_std, "std")
    #plt.figure()
    plot_mean(bWIS_err, WIS_err, "error", False, '--')
    plt.show()



