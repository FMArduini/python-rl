
# coding: utf-8

# In[1]:


# PONG pygame


import numpy as np
import pygame
from pygame.locals import *
import pickle
import warnings

import tensorflow as tf

print(tf)
import numpy as np
import os

from Pong.PPOPONG.env import Env

        

e = Env()


sdim = e.get_state().shape[1] #state dimension
adim = 2 #action space dimension
LR = 0.00025         #learning rate
CLIP = 0.2           #clip parameter
GAMMA = 0.99         #discount gamma
LAYERSIZE = 64       #layer size of networks
vscale = 0.5         #scale constant for value function loss
escale = 0.03        #scale constant for entropy loss
print('action',adim, ' state',sdim)

class PPO:
    def __init__(self):
        
        #----------------------------------------------------------––#
        # input tensors
        tf.reset_default_graph()
        
        self.session = tf.Session()
        
        self.state = tf.placeholder(dtype=tf.float32, shape = [None, sdim], name='state')
        
        self.action = tf.placeholder(dtype=tf.float32, shape = [None], name='actions')
        
        self.advantage = tf.placeholder(dtype=tf.float32, shape = [None], name='advantages')
        
        self.reward = tf.placeholder(dtype=tf.float32, shape = [None], name='values')
        
        self.old_logprob = tf.placeholder(dtype=tf.float32, shape = [None], name='old_action_logprobs')
        
        #----------------------------------------------------------––#
        # neural networks
        with tf.variable_scope("V_net"):
            # neural network for value function estimation
            h1 = tf.layers.dense(self.state,LAYERSIZE,activation=tf.nn.tanh)
            h2 = tf.layers.dense(h1,LAYERSIZE,activation=tf.nn.tanh)
            self.v = tf.squeeze(tf.layers.dense(h2, 1))
        
        with tf.variable_scope('policy_net'):
            # neural network for policy distribution and output
            h1 = tf.layers.dense(self.state,LAYERSIZE,activation=tf.nn.tanh)
            h2 = tf.layers.dense(h1,LAYERSIZE,activation=tf.nn.tanh)
            self.p_out = tf.nn.softmax(tf.layers.dense(h2, adim),axis=-1)  #output policy probabilities
            self.p_dist = tf.distributions.Categorical(probs=self.p_out)   #converts these into a distribution type
            self.p_action = tf.squeeze(self.p_dist.sample(1))              #samples action from distribution
        
        #given an action and the current network params, what is its probability?
        self.a_logprob = self.p_dist._log_prob(self.action) 

        #----------------------------------------------------------––#
        # PPO loss
        
        self.diff = self.a_logprob - self.old_logprob
        self.ratio = tf.exp(self.diff)
        self.unclipped = self.ratio * self.advantage
        self.clipped = tf.clip_by_value(self.ratio, 1. - CLIP, 1. + CLIP) * self.advantage
        self.minclip = tf.minimum(self.clipped,self.unclipped)
        
        self.ploss = -1 * self.minclip                         # L_clip (note sign)

        self.eloss = -1 * self.p_dist.entropy() * escale       # L_entropy (note sign)
        
        self.verr = tf.square(self.reward - self.v)            # L_value
        self.vloss = self.verr * vscale
        
        self.policy_loss = self.ploss + self.eloss
        
        self.loss = self.ploss + self.eloss + self.vloss       # L = L_clip + L_entropy + L_value
        
        self.policy_train_op = tf.train.AdamOptimizer(LR).minimize(self.policy_loss)      # train policy
        
        self.value_train_op = tf.train.AdamOptimizer(LR).minimize(self.vloss)             # train value
        
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)                    # train everything
        
        self.session.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()

        
    def check(self,states,actions,rewards,advantages,old_logprobs):
        """
        Used to check some outputs of the networks
        """
        return self.session.run([
            self.p_out,self.v,self.verr,self.minclip
        ],
                                {
                                    self.state:states,
                                     self.reward:rewards,
                                     self.advantage:advantages,
                                     self.action:actions,
                                     self.old_logprob:old_logprobs
        })

    def get_action(self,states):
        """
        Given a state (or collection of states) and current state of networks, output an action sampled from the
        resulting distribution
        """
        return self.session.run(self.p_action, {self.state:states})
    
    def get_action_prob(self,states, actions):
        """
        Returns the probability of taking the action given the state.
        """
        return self.session.run(self.a_logprob, {self.state:states, self.action:actions})
    
    def get_value(self,states):
        """
        Returns the estimated value of a state
        """
        return self.session.run(self.v, {self.state:states})
    
    def __train_policy(self, states, actions, advantages, old_logprobs):
        """
        Train the policy only
        """
        self.session.run(self.policy_train, {self.state:states, self.action:actions, self.advantage:advantages,
                                            self.old_logprob:old_logprobs})
        
    def __train_value(self, states, rewards):
        """
        Train the value estimator only
        """
        self.session.run(self.value_train, {self.state:states, self.reward:rewards})
        
    def __train(self, states, actions, rewards, advantages, old_logprobs):
        """
        Train the ppo loss
        L_clip + L_entropy + L_value
        """
        self.session.run(self.train_op, {
            self.state:states,
            self.action:actions,
            self.advantage:advantages,
            self.old_logprob:old_logprobs,
            self.reward:rewards
        })
        
    def get_policy_loss(self,states,actions,advantages,old_logprobs):
        """
        Given a bunch of states, actions, advantages and old_action probabilities, 
        returns the policy loss L_clip+L_entropy
        """
        return self.session.run(self.policy_loss, {self.state:states,self.action:actions,
                                           self.advantage:advantages, self.old_logprob:old_logprobs})
        
    def get_value_loss(self,states, rewards):
        """
        Given states and rewards, returns the value loss (L_value)
        """
        return self.session.run(self.vloss, {self.state:states,self.reward:rewards})
    
    @staticmethod
    def discount(rewards, dones, norm=True):
        """
        Static method: given some rewards, and dones (a list of true/falses indicating when an episode ended)
        it returns the discounted rewards according to GAMMA
        """
        drs = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            drs.insert(0, discounted_reward)
        drs = np.squeeze(np.array(drs))
        if norm:
            drs = (drs - np.mean(drs)) / (np.std(drs) + 1e-8)
        return drs
    
    @staticmethod
    def batcher(iterable, n=1):
        """static method, creates batches out of lists"""
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
            
    def save(self, path='/tmp', name='model.ckpt', verbose=False):
        """
        Save the model in target path/folder
        """
        if '.ckpt' not in name:
            name = '{}.ckpt'.format(name)
        self.saver.save(self.session, os.path.join(path, name))
        if verbose:
            print('saved in {}/{}'.format(path, name))

    def restore(self, path='/tmp', name='model.ckpt', verbose=False):
        """
        Restore model from target path
        """
        self.saver.restore(self.session, os.path.join(path, name))
        if verbose:
            print('model restored from {}/{}'.format(path, name))

        
    def train(self, states, actions, rewards, dones, train_iters, minibatch):
        """
        Main method:
        Trains the ppo loss.
        
        Requires:
            - train_iters: number of training loops
            - minibatch (commented out): minibatches to run adam with.
            - states, actions, rewards and dones (for loss)
        
        Before training we do the following:
            1) Using rewards and dones we determine the discounted rewards.
            2) Using states and actions we get the log probabilities of the action given the state 
               given the network BEFORE IT TRAINS. These are our 'old' probabilities
        While we train we do:
            1) get estimated values given states
            2) use discounted rewards and these estiamted values to get an advantage.
            3) train with states, actions, rewards, advantages and old probabilities
            
        Returns the losses after training occurred.
        
        """
        discounted_rewards = self.discount(rewards,dones)
        old_logprobs = self.get_action_prob(states,actions)
        vlosses = []
        plosses = []
        for i in range(train_iters):
            values = self.get_value(states)
            advantages = discounted_rewards - values
#             for batch_idx in self.batcher(range(0, actions.shape[0]), minibatch):
#                 state_batch = states[batch_idx]
#                 action_batch = actions[batch_idx]
#                 old_logprob_batch = old_logprobs[batch_idx]
#                 advantage_batch = advantages[batch_idx]
#                 reward_batch = discounted_rewards[batch_idx]
#                 self.__train(state_batch,action_batch,reward_batch,advantage_batch,old_logprob_batch)
            self.__train(states,actions,rewards,advantages,old_logprobs)
            vlosses.append(self.get_value_loss(states,discounted_rewards))
            plosses.append(self.get_policy_loss(states,actions,advantages,old_logprobs))
        return np.mean(vlosses), np.mean(plosses)


