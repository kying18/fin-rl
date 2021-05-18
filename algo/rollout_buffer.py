import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import copy
import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from copy import deepcopy

class RolloutBuffer:
    def __init__(self, acmodel, env, discount=0.99, gae_lambda=0.95, device=None):
        # TODO changed for cartpole
        self.episode_length = env.episode_length
        self.env = env
#         self.episode_length = 200 - 1
        self.device = device
        self.acmodel = acmodel
        self.discount = discount
        self.gae_lambda = gae_lambda
        
        self.actions = None
        self.values = None
        self.rewards = None
        self.log_probs = None
        self.obss = None
        self.gaes = None
        self.returns = None
        
        self.num_rollouts = 3
        self.reset()
        
        
    def reset(self):
        self.actions = torch.zeros((self.num_rollouts*self.episode_length, len(self.env.tickers)+1), device=self.device)
        self.values = torch.zeros(self.num_rollouts*self.episode_length, device=self.device)
        self.rewards = torch.zeros(self.num_rollouts*self.episode_length, device=self.device)
        self.returns = torch.zeros(self.num_rollouts*self.episode_length, device=self.device)
        
        self.log_probs = torch.zeros((self.num_rollouts*self.episode_length, len(self.env.tickers)+1), device=self.device)
        self.obss = [None] * (self.episode_length*self.num_rollouts)
        
        self.gaes = torch.zeros(self.num_rollouts*self.episode_length, device=self.device)
        
    
    def process_obs(self, obs):
        # TODO: formatting stuff
        if isinstance(obs, list):
            obs = np.stack(obs)

        if len(obs.shape) == 1: # 1 dimensional
            obs = np.expand_dims(obs, axis=0)
            
        return torch.FloatTensor(obs)
    
    def collect_experience(self):
        
        total_return = 0
        self.returns = []
        
        for ep in range(self.num_rollouts):
            obs = self.env.reset()
            
            self.actions_ = torch.zeros((self.episode_length, len(self.env.tickers)+1), device=self.device)
            self.values_ = torch.zeros(self.episode_length, device=self.device)
            self.rewards_ = torch.zeros(self.episode_length, device=self.device)
            self.obss_ = [None] * (self.episode_length)
            self.log_probs_ = torch.zeros((self.episode_length, len(self.env.tickers)+1), device=self.device)
            
            
            T = 0
        
            while True:
                with torch.no_grad():
                    dist, value, _, _ = self.acmodel(self.process_obs(obs))

                action = dist.sample()

                self.obss_[T] = obs

                obs, reward, done, _ = self.env.step(action)
                
                total_return += reward
                
                self.actions_[T] = action[0]
                self.values_[T] = value
                self.rewards_[T] = float(reward)
                self.log_probs_[T] = dist.log_prob(action)[0]


                T += 1
                if done:
                    break
                    
            self.actions[ep*self.episode_length:(ep+1)*self.episode_length] = self.actions_[:T]
            self.values[ep*self.episode_length:(ep+1)*self.episode_length] = self.values_[:T]
            self.rewards[ep*self.episode_length:(ep+1)*self.episode_length] = self.rewards_[:T]
            
            
            discounted_reward = 0.0
            self.returns_ = []
            for r in reversed(self.rewards_):
                discounted_reward = r + self.discount*discounted_reward
                self.returns_.insert(0, discounted_reward)
            self.returns[ep*self.episode_length:(ep+1)*self.episode_length] = self.returns_[:T]
            

            self.log_probs[ep*self.episode_length:(ep+1)*self.episode_length] = self.log_probs_[:T]
            
            self.obss[ep*self.episode_length:(ep+1)*self.episode_length] = self.process_obs(self.obss_[:T])
            self.gaes_ = self.compute_advantage_gae(self.rewards_, self.values_, T)
            self.gaes[ep*self.episode_length:(ep+1)*self.episode_length] = self.gaes_[:T]
            
        self.obss = torch.FloatTensor(np.stack(self.obss))
        self.returns = torch.FloatTensor(self.returns)
        
        return total_return/self.num_rollouts, T
            
    def compute_advantage_gae(self, rewards, values, T):

        deltas = torch.cat((rewards[:-1] + self.discount*values[1:] - values[:-1], rewards[-1:] - values[-1:]))
        deltas_flip = torch.flip(deltas, [0])
        A = 0.0*deltas_flip[0]
        advantages = torch.zeros_like(values)
        for i in range(len(deltas_flip)):
            A = self.discount*self.gae_lambda*A + deltas_flip[i]
            advantages[i] = A
        advantages = torch.flip(advantages, [0])
        return advantages
            
        
    
    
        