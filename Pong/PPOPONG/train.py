
# coding: utf-8

# In[1]:


# PONG pygame


import numpy as np
import pygame
from pygame.locals import *
import pickle
import warnings

import tensorflow as tf
import numpy as np
import os

from Pong.PPOPONG.env import Env
from Pong.PPOPONG.ppo import PPO

# In[5]:


env = Env(game_length=1)

#set some params
iters = 300                      # sampling iterations
episode_per_iter = 100           # number of episodes to run for each iteration
max_episode_steps = 300          # maximum steps allowed per episode
minibatch = 64                   # minibatch size
train_iters = 3                  # n of training loops per iteration

# start ppo
ppo = PPO()
try:
    for i in range(iters):
        rewards = []      # all rewards this iter
        dones = []        # all dones this iter
        actions = []      # all actions this iter
        states = []       # all states this iter

        e_lengths = []    # keeps track of this iteration's episode lengths
        e_rewards = []    # keep track of this iteration's episode rewards
        for e in range(episode_per_iter):
            # run episode
            state = env.reset().reshape(1,-1)  #reset
            done = False
            el = 0                                                    #total episode length
            er = 0                                                    #total episode reward
            while not done:
                el +=1
                action = ppo.get_action(state)                        # sample action from ppo net
                next_state, reward, done, _ = env.step(1 if action==1 else -1)        # run a step in the env
                er += reward
                states.append(state)                                  # record old state
                state = next_state.reshape(1,-1)                      # set new state as state
                actions.append(action)                                # record action taken
                rewards.append(reward)                                # record reward obtained
                if el > max_episode_steps:
                    done = True                                       # if we exceed max allowed steps: stop
                dones.append(done)
            e_lengths.append(el)
            e_rewards.append(er)



        states = np.squeeze(np.array(states))
        actions = np.squeeze(np.array(actions))

        #train
        vloss,ploss = ppo.train(states, actions, rewards, dones, train_iters, minibatch)

        #print progress
        print('iter: {0} | Ploss: {1:.2f} | Vloss: {2:.2f} | avg R: {3:.2f} | Best R: {4:.2f} | Avg episode Length: {5:.1f}'.format(
            i,
            np.nan if not ploss else ploss,
            np.nan if not vloss else vloss,
            np.mean(e_rewards),
            np.max(e_rewards),
            np.mean(e_lengths)

        ))
except KeyboardInterrupt as ex:
    print("training stopped, saving model.")
finally:
    ppo.save()



