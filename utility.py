# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:40:57 2022

@author: cheng164
"""

import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import math


## citation: https://github.com/yahsiuhsieh/frozen-lake/tree/main/src
##           https://machinelearningjourney.com/index.php/2020/07/02/frozenlake/
##           https://www.kaggle.com/code/karthikcs1/openai-frozen-lake-problem/notebook

def run_episode(env, policy, gamma, render = True):
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done , _ = env.step(int(policy[obs]))
		total_reward += (gamma ** step_idx * reward)
		step_idx += 1
		if done:
			break
	return total_reward


def evaluate_policy(env, policy, gamma , n = 100):
	scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)



def extract_policy(env,v, gamma):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
		policy[s] = np.argmax(q_sa)
	return policy



def compute_policy_v(env, policy, gamma):
	v = np.zeros(env.nS)
	eps = 1e-5
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
		if (np.sum((np.fabs(prev_v - v))) <= eps):
			break
	return v



def policy_iteration(env, gamma, iter_plot=False):
    
    m = int(math.sqrt(env.nS))
    policy = np.random.choice(env.nA, size=(env.nS))  
    max_iters = 200000
    desc = env.unwrapped.desc
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env,old_policy_v, gamma) 
        
        if iter_plot is True and i<=5:
            title = 'Frozen Lake (Policy Iteration)-- '+ 'Gamma = ' + str(gamma)+ ',' + 'iter='+str(i)
            plot = plot_policy_map(title, desc,colors_lake(),directions_lake(), policy= policy.reshape(m,m), values= old_policy_v.reshape(m,m))
           
        if (np.all(policy == new_policy)):
            k=i+1
            break
        policy = new_policy
    
    return policy, k, old_policy_v



def value_iteration(env, gamma, iter_plot=False):
    
    m = int(math.sqrt(env.nS))
    v = np.zeros(env.nS)  # initialize value-function
    max_iters = 100000
    eps = 1e-08
    desc = env.unwrapped.desc
    for i in range(max_iters):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)

        if iter_plot is True and i%50 ==1:
            title = 'Frozen Lake (Value Iteration)-- '+ 'Gamma = ' + str(gamma)+ ',' + 'iter='+ str(i)
            plot = plot_policy_map(title, desc, colors_lake(), directions_lake(), policy= None, values = v.reshape(m,m))            

        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k=i+1
            break
    return v,k



def plot_policy_map(title, map_desc, color_map, direction_map, policy=None, values=None):
    if policy is not None:
        m = policy
    else:
        m = values
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, m.shape[1]), ylim=(0, m.shape[0]))
    font_size = 'x-large'
    if m.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            y = m.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i,j]])
            ax.add_patch(p)

            if policy is not None: 
                text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                               horizontalalignment='center', verticalalignment='center', color='w')	
		          
            # Add value number text in the box
            if values is not None:
                text = ax.text(x+0.5, y+0.2, round(values[i, j],6), size=font_size,
                               horizontalalignment='center', verticalalignment='center', color='Black')	
            
    plt.axis('off')
    plt.xlim((0, m.shape[1]))
    plt.ylim((0, m.shape[0]))
    plt.tight_layout()

    return (plt)




def PI_VI_plot(title, gamma_arr, time_array, list_scores, iters):
    
    plt.figure()	
    plt.plot(gamma_arr, time_array, '-o')
    plt.xlabel('Gammas')
    plt.title(title + ' - Running Time vs.Gamma')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(gamma_arr,list_scores, '-o')
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title(title +' - Average Reward vs.Gamma')
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(gamma_arr,iters, '-o')
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title(title + ' - Iterations to Converge vs.Gamma')
    plt.grid()
    plt.show()



def Q_learner(env, episodes, epsilon, alpha, gamma):

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    alpha_0 = alpha
    decay_rate= 0.99996
    epsilon_0 = epsilon

    for episode in range(episodes):
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = 1000000

        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.rand() < epsilon_0:
                action = env.action_space.sample()
            else:
                action= np.argmax(Q[current, :])
    				
            state, reward, done, info = env.step(action)
            t_reward += reward
            Q[current, action] += alpha_0 * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
             
            # # Decaying learnig rate
            
        # epsilon=(1-2.71**(-episode/1000))
        alpha_0 = decay_rate**(i)* alpha_0 
        epsilon_0=decay_rate** i * epsilon_0
        rewards.append(t_reward)
        iters.append(i)
        
        
    return Q, rewards, iters
   


def Q_learning_plot_frozen_lake(episodes, time_array, averages_array, epsilon_ls, Q_array):
    
    plt.figure()
    for i in range(0,len(averages_array)):
        plt.plot(range(0, episodes, 1000), averages_array[i], '-o', label='epsilon_init='+str(epsilon_ls[i]))
        
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.grid()
    plt.title('Frozen Lake (Q Learning) - Average Reward per thousand episodes')
    plt.ylabel('Average Reward')
    plt.show()
        
       
    plt.figure()
    plt.plot(epsilon_ls, time_array, '-o')
    plt.xlabel('Initial Epsilon Values')
    plt.grid()
    plt.title('Frozen Lake (Q Learning)-- Running Time vs. epsilon_init')
    plt.ylabel('Execution Time (s)')
    plt.show()

    n = len(averages_array) 
    m = Q_array[0].shape[1]
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, ax in enumerate(axes.flatten(), start=1):
        im = ax.imshow(np.transpose(Q_array[i-1]), cmap='cool')
        ax.set_title(f'Epsilon= {epsilon_ls[i-1]}', fontsize='8') 
        ax.set_yticks(range(m))
        ax.set_yticklabels(range(m), fontsize='7')
    
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.xlabel("State")
    fig.text(0.36, 0.5, 'Action', va='center', rotation='vertical')
    fig.text(0.45, 1, 'Q Tables', ha='center', )
    plt.show()





def Q_learning_plot_forest(rewards_ls, running_time_ls, Q_ls, gamma_ls, episodes_ls):
    
    plt.figure()
    for i in range(0,len(gamma_ls)):
        plt.plot(episodes_ls, rewards_ls[i], '-o', label='gamma='+str(gamma_ls[i]))
        
    plt.legend()
    plt.xlabel('Episodes')
    plt.grid()
    plt.title('FOREST (Q Learning) --Average Reward per 1000 Episodes')
    plt.ylabel('Reward')
    plt.show()
        
       
    plt.figure()
    for i in range(0,len(gamma_ls)):
        plt.plot(episodes_ls, running_time_ls[i], '-o', label='gamma='+str(gamma_ls[i]))
        
    plt.legend()
    plt.xlabel('Episodes')
    plt.grid()
    plt.title('Frozen Lake (Q Learning)-- Running Time')
    plt.ylabel('Execution Time (s)')
    plt.show()


    
    n = len(Q_ls) 
    m = Q_ls[-1][-1].shape[1]
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, ax in enumerate(axes.flatten(), start=1):
        im = ax.imshow(np.transpose(Q_ls[i-1][-1]), cmap='cool')
        ax.set_title(f'Gamma= {gamma_ls[i-1]}', fontsize='8') 
        ax.set_yticks(range(m))
        ax.set_yticklabels(range(m), fontsize='7')
    
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.xlabel("State")
    fig.text(0.5, 0.5, 'Action', va='center', rotation='vertical')
    fig.text(0.65, 1, f'Q Tables when p = 0.1', ha='center')
    plt.show()
    



def Q_learning_hyperpara_plot(alpha_0=0.85, epsilon_0=0.95, episodes=30000):
    
    decay_rate= 0.99996
    i = range(0,episodes)
    alpha_array = list(map(lambda x: decay_rate**(x)* alpha_0, i))
    epsilon_array = list(map(lambda x: decay_rate**(x)* epsilon_0, i))   
    
    plt.figure()
    plt.plot(i, alpha_array)
    plt.xlabel('Iterations')
    plt.grid()
    plt.title('Frozen Lake (Q Learning)-- Learning rate vs. iteration')
    plt.ylabel('Learning Rate')
    plt.show()
    
    plt.figure()
    plt.plot(i, epsilon_array)
    plt.xlabel('Iterations')
    plt.grid()
    plt.title('Frozen Lake (Q Learning)-- Epsilon vs. iteration')
    plt.ylabel('Epsilon')
    plt.show()    



def colors_lake():
	return {
		b'S': 'green',
		b'F': 'skyblue',
		b'H': 'grey',
		b'G': 'red',
	}



def directions_lake():
	return {
		3: '⬆',
		2: '➡',
		1: '⬇',
		0: '⬅'
	}