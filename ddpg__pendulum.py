#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:50:50 2020

@author: ikaya
"""

from DDPG import *


# Learning rate for actor-critic models
total_episodes   = 100
ep_reward_list   = [] # To store reward history of each episode
avg_reward_list  = [] # To store average reward history of last few episodes

myagent = Agent(environment='Pendulum-v0')

for eps in range(total_episodes):

    state           = myagent.sim.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action but not in a python notebook.
        myagent.sim.render()
        action                        = myagent.policy(state.reshape(1,myagent.numberOfStates))
        nextstate, reward, done, info = myagent.sim.step(action)
        myagent.observation           = (state,action,reward,nextstate)
        myagent.record_buffer()
        episodic_reward               = episodic_reward + reward
        myagent.learn()
        # End this episode when `done` is True
        if done:
            break

        state = nextstate

    ep_reward_list.append(episodic_reward)
    # Mean of last 10 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(eps, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()