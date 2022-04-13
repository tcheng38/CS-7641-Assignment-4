# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:25:56 2022

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
import mdptoolbox, mdptoolbox.example
from hiive.mdptoolbox.mdp import QLearning



def run_forest(size, num_iter, r1, r2, p):
	
    
    print('FOREST - POLICY ITERATION')
    P, R = mdptoolbox.example.forest(S=size, r1=r1, r2=r2, p=p)
    values_ls = [0]*num_iter
    policy = [0]*num_iter
    iters = [0]*num_iter
    runing_time_ls = [0]*num_iter
    gamma_ls = [0] * num_iter
    
    
    for i in range(0,num_iter):
        gamma = (i+0.5)/num_iter
        pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        pi.run()
        gamma_ls[i]= gamma
        values_ls[i] = np.sum(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        runing_time_ls[i] = pi.time

        #print('gamma=',gamma, 'policy:', pi.policy)
        #print('gamma=',gamma, 'value:', pi.V)
        
    PI_VI_plot('Forest (Policy Iteration)', gamma_ls, runing_time_ls, values_ls, iters)



    print('FOREST - VALUE ITERATION')
    P, R = mdptoolbox.example.forest(S=size, r1=r1, r2=r2, p=p)
    values_ls = [0]*num_iter
    policy = [0]*num_iter
    iters = [0]*num_iter
    runing_time_ls = [0]*num_iter
    gamma_ls = [0] * num_iter
    
    for i in range(0,num_iter):
        gamma = (i+0.5)/num_iter
        pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        pi.run()
        gamma_ls[i]= gamma
        values_ls[i] = np.sum(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        runing_time_ls[i] = pi.time
        
        #print('gamma=',gamma, 'policy:', pi.policy)
        #print('gamma=',gamma, 'value:', pi.V)

    PI_VI_plot('Forest (Value Iteration)', gamma_ls, runing_time_ls, values_ls, iters)
    

    
    
    print('FOREST - Q LEARNING')
    P, R = mdptoolbox.example.forest(S=size, r1=r1, r2=r2, p=p)
    rewards_ls = []
    policy_ls = []
    runing_time_ls = []
    Q_ls = []
    episodes_ls = range(10000,40000,10000)
    gamma_ls = [0.1, 0.3, 0.5, 0.7, 0.9] 
    
    for gamma in gamma_ls:
        rewards =[]
        policies=[]
        running_times = []
        Q_tables = []
        for episode in episodes_ls:                      
            st = time.time()
            pi = QLearning(P,R, gamma= gamma, alpha=0.7, alpha_decay=0.9996, alpha_min=0.001,\
                           epsilon=0.85, epsilon_min=0.1, epsilon_decay=0.9996, n_iter=episode)
            pi.run()
            end = time.time()
            
            rewards.append(np.sum(pi.V))
            policies.append(pi.policy)
            running_times.append(end-st)
            Q_tables.append(pi.Q)
        
            #print('gamma=',gamma, 'episode=', episode, ' policy:', pi.policy)
            #print('gamma=',gamma, 'episode=', episode, ' value:', pi.V)
        
        rewards_ls.append(rewards)
        policy_ls.append(policies)
        runing_time_ls.append(running_times)
        Q_ls.append(Q_tables)
    
    Q_learning_plot_forest(rewards_ls, runing_time_ls, Q_ls, gamma_ls, episodes_ls)     
    
    
    
print('STARTING FOREST')
run_forest(size=5, num_iter=5, r1=4, r2=2, p=0.1)