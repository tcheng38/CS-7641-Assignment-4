# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:18:53 2022

@author: cheng164
"""
import os
import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
from utility import*
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv


## citation: https://github.com/yahsiuhsieh/frozen-lake/tree/main/src
##           https://machinelearningjourney.com/index.php/2020/07/02/frozenlake/
##           https://www.kaggle.com/code/karthikcs1/openai-frozen-lake-problem/notebook

cur_dir = os.getcwd()
final_dir = os.path.join(cur_dir, r'results')
if not os.path.exists(final_dir):
    os.makedirs(final_dir)


np.random.seed(1)
map_large = generate_random_map(25)


def run_Frozen_Lake(grid_size, num_iter, plot_flag=False):
    # 0 = left; 1 = down; 2 = right;  3 = up

    if grid_size == "4x4":
        env = gym.make('FrozenLake-v0')
        size = 4
    else:
        env = FrozenLakeEnv(desc=map_large)
        size = 25
    
    env = env.unwrapped
    desc = env.unwrapped.desc
    
    runing_time_ls=[0]*num_iter
    gamma_ls=[0]*num_iter
    iters=[0]*num_iter
    scores_ls=[0]*num_iter
    

    
    ### POLICY ITERATION ####
    print('POLICY ITERATION WITH FROZEN LAKE')
    for i in range(0,num_iter):
        st=time.time()
        gamma = (i+0.8)/num_iter
        opt_policy, final_iter, opt_values = policy_iteration(env, gamma, iter_plot=False)    
        scores = evaluate_policy(env, opt_policy, gamma)
        end=time.time()
        
        title = 'Frozen Lake ' + grid_size + ' Policy Map: ' +' (Policy Iteration)-- ' + 'gamma ='+ str(gamma)
        plot = plot_policy_map(title, desc, colors_lake(), directions_lake(), policy=opt_policy.reshape(size,size), values=opt_values.reshape(size,size))
        plot.savefig(f"results/frozen_lake_PI_gamma{gamma}.png")
        
        gamma_ls[i]=gamma
        scores_ls[i]=np.mean(scores)
        iters[i] = final_iter
        runing_time_ls[i]=end-st
    	
    PI_VI_plot('Frozen Lake (Policy Iteration)', gamma_ls, runing_time_ls, scores_ls, iters)
    
    # # choose a middle value of gamma and plot the map of policy and values by iteration. 
    if plot_flag is True: 
        print('gamma =', gamma_ls[-1] )
        best_policy_, final_iter_, opt_values_ = policy_iteration(env, gamma=gamma_ls[-1], iter_plot=True)
        print('final iterstion =', final_iter_)
    
    
    ### VALUE ITERATION ###
    print('VALUE ITERATION WITH FROZEN LAKE')
    best_vals=[0]*num_iter
    for i in range(0,num_iter):
        st=time.time()
        gamma = (i+0.5)/num_iter
        opt_value, final_iter = value_iteration(env, gamma, iter_plot=False)
        policy = extract_policy(env,opt_value, gamma)
        policy_score = evaluate_policy(env, policy, gamma, n=1000)
        end=time.time()
        
        title = 'Frozen Lake ' + grid_size + ' Policy Map: ' +' (Value Iteration)-- ' + 'gamma ='+ str(gamma)
        plot = plot_policy_map(title, desc, colors_lake(), directions_lake(), policy=policy.reshape(size,size), values=opt_value.reshape(size,size))
        plot.savefig(f"results/frozen_lake_VI_gamma{gamma}.png")
        
        gamma_ls[i]=gamma
        iters[i]= final_iter
        best_vals[i] = opt_value
        scores_ls[i]=np.mean(policy_score)
        runing_time_ls[i]=end-st
    
    PI_VI_plot('Frozen Lake (Value Iteration)', gamma_ls, runing_time_ls, scores_ls, iters)
    
    # # choose a middle value of gamma and plot the map of policy/values by iteration. 
    if plot_flag is True: 
           
        best_policy_, final_iter_= value_iteration(env, gamma=gamma_ls[-1], iter_plot=True)
    
    
    
    
    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')

    reward_array = []
    iter_array = []
    size_array = []
    averages_array = []
    time_array = []
    Q_array = []
    iter_array = []
    episodes = 30000
    epsilon_ls = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    for epsilon in epsilon_ls:
        st = time.time()
        optimal=[0]*env.observation_space.n
        avg_rewards =[]
        alpha = 0.85
        gamma = 0.95
    	
        Q, rewards, iters = Q_learner(env, episodes, epsilon, alpha, gamma)
        end=time.time()
        
        # # plot the policy map
        policy = np.argmax(Q, axis=1)
        # print('epsilon=', epsilon , 'policy:', policy)
        title = 'Frozen Lake ' + grid_size + ' Policy Map: ' +' (Q-learning)-- ' + 'epsilon_init ='+ str(epsilon)
        plot_policy_map(title, desc, colors_lake(), directions_lake(), policy=policy.reshape(size,size), values=None)
        
        for k in range(env.observation_space.n):
            optimal[k]=np.argmax(Q[k, :])
    
        # # calculate Mean reward per thousand episodes 
        for i in range(0, episodes//1000):
            avg_reward = np.mean(rewards[1000*i:1000*(i+1)])
            avg_rewards.append(avg_reward)
    
        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)
        averages_array.append(avg_rewards)
        
        env.close()
        time_array.append(end-st)
        
        
    Q_learning_plot_frozen_lake(episodes, time_array, averages_array, epsilon_ls, Q_array)
    


    
    
print('STARTING FROZEN LAKE')
run_Frozen_Lake("4x4", num_iter=10, plot_flag=False)
# Q_learning_hyperpara_plot(alpha_0=0.85, epsilon_0=0.95, episodes = 30000)