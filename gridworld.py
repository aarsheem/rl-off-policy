import numpy as np
import matplotlib.pyplot as plt
from systemrl.environments.gridworld import Gridworld
from tqdm import tqdm
import pickle
import os
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import sys
from sklearn.linear_model import LinearRegression


env = Gridworld()
num_states = 25
num_actions = 4
(u,d,l,r) = (0,1,2,3)
optimal_policy = np.array([
        r, r, r, r, d,
        u, r, r, r, d,
        u, d, l, r, d,
        u, l, r, d, d,
        u, l, r, r, d])

#------------------------

def set_policy(randomness):
    def policy(state):
        if np.random.random() > randomness:
            return optimal_policy[state]
        return np.random.choice(num_actions)
    return policy

def pi_action_state(actions, states, randomness):
    optimal_actions = optimal_policy[states]
    probability =  np.ones(len(states)) * randomness/num_actions
    probability[optimal_actions == actions] += 1 - randomness
    return probability

def IS(states, actions, rewards, run, evaluate):
    num_episodes = len(states)
    estimate = np.zeros(num_episodes)
    for i in range(num_episodes):
        pi_run = pi_action_state(actions[i], states[i], run)
        pi_evaluate = pi_action_state(actions[i], states[i], evaluate)
        G = np.sum(rewards[i])
        estimate[i] = np.exp(np.sum(np.log(pi_evaluate)) - np.sum(np.log(pi_run))) * G
    return np.cumsum(estimate)/np.arange(1, num_episodes+1)

def WIS(states, actions, rewards, run, evaluate):
    num_episodes = len(states)
    numerator = np.zeros(num_episodes)
    denominator = np.zeros(num_episodes)
    for i in range(num_episodes):
        pi_run = pi_action_state(actions[i], states[i], run)
        pi_evaluate = pi_action_state(actions[i], states[i], evaluate)
        G = np.sum(rewards[i])
        denominator[i] = np.exp(np.sum(np.log(pi_evaluate)) - np.sum(np.log(pi_run)))
        numerator[i] = denominator[i] * G
    return np.cumsum(numerator)/np.cumsum(denominator)

#bootstrap estimate of WIS
def bWIS(states, actions, rewards, run, evaluate, B=100):
    num_episodes = len(states)
    numerator = np.zeros(num_episodes)
    denominator = np.zeros(num_episodes)
    unbiased_wis = np.zeros(num_episodes)
    sampled_std = np.zeros(num_episodes)
    for i in range(num_episodes):
        pi_run = pi_action_state(actions[i], states[i], run)
        pi_evaluate = pi_action_state(actions[i], states[i], evaluate)
        G = np.sum(rewards[i])
        denominator[i] = np.exp(np.sum(np.log(pi_evaluate)) - np.sum(np.log(pi_run)))
        numerator[i] = denominator[i] * G
        wis = np.sum(numerator[:i+1])/np.sum(denominator[:i+1])
        sample_indices = np.random.choice(i+1, size=[B, i+1])
        sampled_wis = np.sum(numerator[sample_indices], 1)/np.sum(denominator[sample_indices], 1)
        #Standard deviation of samples
        sampled_std[i] = np.std(sampled_wis)
        bias = np.mean(sampled_wis) - wis
        unbiased_wis[i] = wis - bias
    return unbiased_wis, sampled_std

#improved estimate of bias(Section 10.4)
def bWIS2(states, actions, rewards, run, evaluate, B=100):
    num_episodes = len(states)
    numerator = np.zeros(num_episodes)
    denominator = np.zeros(num_episodes)
    unbiased_wis = np.zeros(num_episodes)
    estimate = np.zeros(num_episodes)
    for i in range(num_episodes):
        pi_run = pi_action_state(actions[i], states[i], run)
        pi_evaluate = pi_action_state(actions[i], states[i], evaluate)
        G = np.sum(rewards[i])
        denominator[i] = np.exp(np.sum(np.log(pi_evaluate)) - np.sum(np.log(pi_run)))
        numerator[i] = denominator[i] * G
        sample_indices = np.random.choice(i+1, size=[B, i+1])
        P = np.bincount(sample_indices.flatten(), minlength=i+1)/B/(i+1)
        wis2 = np.sum(numerator[:i+1] * P)/np.sum(denominator[:i+1] * P)
        wis = np.sum(numerator[:i+1])/np.sum(denominator[:i+1])
        estimate[i] = np.mean(np.sum(numerator[sample_indices], 1)/np.sum(denominator[sample_indices], 1))
        bias = estimate[i] - wis2
        unbiased_wis[i] = wis - bias
    return unbiased_wis, estimate

#jackknife
def jackWIS(states, actions, rewards, run, evaluate):
    num_episodes = len(states)
    numerator = np.zeros(num_episodes)
    denominator = np.zeros(num_episodes)
    cum_denominator = np.zeros(num_episodes)
    cum_numerator = np.zeros(num_episodes)
    unbiased_wis = np.zeros(num_episodes)
    for i in range(num_episodes):
        pi_run = pi_action_state(actions[i], states[i], run)
        pi_evaluate = pi_action_state(actions[i], states[i], evaluate)
        G = np.sum(rewards[i])
        denominator[i] = np.exp(np.sum(np.log(pi_evaluate)) - np.sum(np.log(pi_run)))
        numerator[i] = denominator[i] * G
        cum_numerator[i] = numerator[0] + cum_numerator[0]
        cum_denominator[i] = denominator[0] + cum_denominator[0]
        cum_numerator[:i] += numerator[i]
        cum_denominator[:i] += denominator[i]
        wis = (cum_numerator[i] + numerator[i])/(cum_denominator[i] + denominator[i])
        bias = i * (np.mean(cum_numerator[:i+1]/cum_denominator[:i+1]) - wis)
        unbiased_wis[i] = wis - bias
    return unbiased_wis

def controlWIS(states, actions, rewards, run, evaluate, B=100):
    num_episodes = len(states)
    numerator = np.zeros(num_episodes)
    denominator = np.zeros(num_episodes)
    estimate = np.zeros(num_episodes)
    unbiased_wis = np.zeros(num_episodes)
    for i in range(num_episodes):
        pi_run = pi_action_state(actions[i], states[i], run)
        pi_evaluate = pi_action_state(actions[i], states[i], evaluate)
        G = np.sum(rewards[i])
        denominator[i] = np.exp(np.sum(np.log(pi_evaluate)) - np.sum(np.log(pi_run)))
        numerator[i] = denominator[i] * G
        sample_indices = np.random.choice(i+1, size=[B, i+1])
        P = np.array([np.bincount(sample,minlength=i+1) for sample in sample_indices])/(i+1)
        T = np.sum(P * numerator[:i+1], 1)/np.sum(P * denominator[:i+1], 1)
        reg = LinearRegression().fit(P, T)
        estimate[i] = reg.predict([np.ones(i+1)/(i+1)]) + np.mean(T) - reg.predict([np.mean(P, 0)])
        wis = np.sum(numerator[:i+1])/np.sum(denominator[:i+1])
        bias = estimate[i] - wis
        unbiased_wis[i] = wis - bias
    return unbiased_wis, estimate

def history(env, policy, num_episodes=1000, max_steps=1000):
    history_state = [[] for i in range(num_episodes)]
    history_action = [[] for i in range(num_episodes)]
    history_reward = [[] for i in range(num_episodes)]
    for episode in range(num_episodes):
        env.reset()
        state = env.state
        is_end = False
        count = 0
        while not is_end and count < max_steps:
            count += 1
            action = policy(state)
            next_state, reward, is_end = env.step(action)
            history_state[episode].append(state)
            history_action[episode].append(action)
            history_reward[episode].append(reward)
            state = next_state
    return history_state, history_action, history_reward

def performance(history_reward):
    return np.mean([np.sum(episode) for episode in history_reward])

def save_data(filename, data):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)

def load_data(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    return data

def save_history(filename, states, actions, rewards):
    save_data(filename+"_state.txt", states)
    save_data(filename+"_action.txt", actions)
    save_data(filename+"_reward.txt", rewards)

def load_history(filename):
    states = load_data(filename+"_state.txt")
    actions = load_data(filename+"_action.txt")
    rewards = load_data(filename+"_reward.txt")
    return states, actions, rewards

def generate_data(env, run, num_episodes, max_steps):
    policy_run = set_policy(run)
    states, actions, rewards = history(env, policy_run, num_episodes, max_steps)
    #save_history(filename, states, actions, rewards)
    return states, actions, rewards

def write_results(filename, IS_error, WIS_error, bWIS_error, IS_error_std, WIS_error_std, bWIS_error_std):
    try:
        os.remove("results/"+filename+".txt")
    except OSError:
        pass
    f = open("results/"+filename+".txt", "a")
    f.write("IS avg error +/- avg std: " + str(np.mean(IS_error)) +  " +/-" + str(np.mean(IS_error_std)) + "\n")
    f.write("WIS avg error +/- avg std: " + str(np.mean(WIS_error)) +  " +/-" + str(np.mean(WIS_error_std)) + "\n")
    f.write("bWIS error +/- avg std: " + str(np.mean(bWIS_error)) +  " +/-" + str(np.mean(bWIS_error_std)) + "\n")
    f.close()

def plot_mean(estimate, label, color, zero=False, linestyle='-'):
    x = 1+np.arange(estimate.shape[0])
    plt.plot(x, estimate, color, label=label, linestyle=linestyle)
    if zero:
        plt.plot(x, np.zeros(x.shape), 'k')
    plt.xlabel("episodes")
    plt.legend()

def plot_std(mean, lower_bound, upper_bound, label, color):
    x = 1+np.arange(lower_bound.shape[0])
    plt.fill_between(x, mean-lower_bound, mean+upper_bound, color=color, alpha=0.2, label=label)
    plt.xlabel("episodes")
    plt.legend()

def err_bias_std(actual, estimates):
    err = np.mean(np.abs(estimates - actual), 0)
    bias = np.mean(estimates - actual, 0)
    std = np.std(estimates, 0)
    return err, bias, std

def boot(data):
    print(data.shape)
    val = np.zeros(data.shape[1])
    low = np.zeros(data.shape[1])
    up = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        res = bs.bootstrap(data[:,i], stat_func=bs_stats.mean)
        val[i] = res.value
        low[i] = res.lower_bound
        up[i] = res.upper_bound
    return val, low, up

#------------------------

run = 0.5
evaluate = 0.75
num_episodes = 10
max_steps = 1000
B = 1000
iterations = 1000
filename = "run_" + str(run) + "_eval_" + str(evaluate) + "_episodes_" + str(num_episodes) +\
    "_steps_" + str(max_steps) + "_" + "_B_" + str(B) + "_iter_" + str(iterations)

IS_estimates    = np.zeros((iterations, num_episodes))
WIS_estimates   = np.zeros((iterations, num_episodes))
bWIS_estimates  = np.zeros((iterations, num_episodes))
bWIS_sample_std = np.zeros((iterations, num_episodes))
bWIS2_estimates = np.zeros((iterations, num_episodes))
bWIS2_biased = np.zeros((iterations, num_episodes))
jackWIS_estimates = np.zeros((iterations, num_episodes))
controlWIS_estimates = np.zeros((iterations, num_episodes))
controlWIS_biased = np.zeros((iterations, num_episodes))


#states, actions, rewards = generate_data(env, run, 5, 10)
#controlWIS(states, actions, rewards, run, evaluate, 2)
#sys.exit(0)

for i in tqdm(range(iterations)):
    states, actions, rewards = generate_data(env, run, num_episodes, max_steps)

    IS_estimates[i]     = IS(states, actions, rewards, run, evaluate)
    WIS_estimates[i]    = WIS(states, actions, rewards, run, evaluate)
    bWIS_estimates[i],_ = bWIS(states, actions, rewards, run, evaluate, B)
    bWIS2_estimates[i], bWIS2_biased[i]  = bWIS2(states, actions, rewards, run, evaluate, B)
    jackWIS_estimates[i] = jackWIS(states, actions, rewards, run, evaluate)
    controlWIS_estimates[i], controlWIS_biased[i] = controlWIS(states, actions, rewards, run, evaluate, B)

actual = performance(generate_data(env, evaluate, 10000, max_steps)[2])
print("actual: ", actual)

IS_err, IS_bias, IS_std         = err_bias_std(actual, IS_estimates)
WIS_err, WIS_bias, WIS_std       = err_bias_std(actual, WIS_estimates)
bWIS_err, bWIS_bias, bWIS_std     = err_bias_std(actual, bWIS_estimates)
bWIS2_err, bWIS2_bias, bWIS2_std   = err_bias_std(actual, bWIS2_estimates)
bWIS2_biased_err, bWIS2_biased_bias, bWIS2_biased_std   = err_bias_std(actual, bWIS2_biased)
jackWIS_err, jackWIS_bias, jackWIS_std  = err_bias_std(actual, jackWIS_estimates)
controlWIS_err, controlWIS_bias, controlWIS_std  = err_bias_std(actual, controlWIS_estimates)
controlWIS_biased_err, controlWIS_biased_bias, controlWIS_biased_std   = err_bias_std(actual, controlWIS_biased)

#bias plots
plot_mean(IS_bias, 'IS_bias', 'gray', True)
plot_mean(WIS_bias, 'WIS_bias', 'darkorange')
plot_mean(bWIS_bias, 'bWIS_bias', 'seagreen')
plot_mean(bWIS2_biased_bias, 'bWIS2_biased_bias', 'cornflowerblue', linestyle="-.")
plot_mean(controlWIS_bias, 'control_bias', 'yellow')

#std plots
plot_std(IS_bias, IS_std, IS_std, 'IS_std', 'gray')
plot_std(WIS_bias, WIS_std, WIS_std, 'WIS_std', 'red')
plot_std(bWIS_bias, bWIS_std, bWIS_std, 'bWIS_std', 'green')
plot_std(bWIS2_biased_bias, bWIS2_biased_std, bWIS2_biased_std, 'bWIS2_biased_std', 'blue')
#plot_std(controlWIS_bias, controlWIS_std, controlWIS_std, 'controlWIS_std', 'yellow')
plt.figure()

#error plots
plot_mean(IS_err, 'IS_err', 'gray', True)
plot_mean(WIS_err, 'WIS_err', 'darkorange')
plot_mean(bWIS_err, 'bWIS_err', 'seagreen')
plot_mean(bWIS2_err, 'bWIS2_err', 'cornflowerblue')
plot_mean(bWIS2_biased_err, 'bWIS2_biased_err', 'cornflowerblue', linestyle='-.')
plot_mean(jackWIS_err, 'jackWIS_err', 'red')
plot_mean(controlWIS_err, 'controlWIS_err', 'yellow')
plot_mean(controlWIS_biased_err, 'controlWIS_biased_err', 'yellow', linestyle='-.')
plt.show()
#-----------------------
#Yash's idea
sys.exit(0)
num_episodes = 1000
IS_estimates    = np.zeros((iterations, num_episodes))
for i in tqdm(range(iterations)):
    states, actions, rewards = generate_data(env, run, num_episodes, max_steps)
    IS_estimates[i]     = IS(states, actions, rewards, run, evaluate)

IS_boot, IS_lower_bound, IS_upper_bound = boot(IS_estimates-actual)
plot_mean(IS_boot, "IS_boot", 'gray', True)
plot_std(IS_boot, IS_boot-IS_lower_bound, -IS_boot+IS_upper_bound, "IS_bounds", "gray")
plt.show()
