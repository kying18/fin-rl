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

class PPO:
    def __init__(self,
                 acmodel,
                 clip_ratio=0.2,
                 entropy_coef=0.01,
                 lr=2e-4,
                 target_kl=0.01,
                 train_iters=10):
        
        self.acmodel = acmodel
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.target_kl=target_kl
        self.train_iters = train_iters
        
        self.optimizer = torch.optim.Adam(acmodel.parameters(), lr=lr)
        
    def update(self, rollouts):
        # rollouts should be RolloutBuffer object

        dist, _, _, _ = self.acmodel(rollouts.obss) # TODO may need to process these observations
        old_logp = dist.log_prob(rollouts.actions.view(-1,self.acmodel.action_dim)).detach()
        

#         policy_loss, _ = self._compute_policy_loss_ppo(rollouts.obss, old_logp, rollouts.actions, rollouts.gaes)
#         value_loss = self._compute_value_loss(rollouts.obss, rollouts.returns)
        
        batch_size = 64
        avg_policy_loss, avg_value_loss = 0, 0
        for i in range(self.train_iters):
            rand_idxs = np.random.choice(np.arange(len(rollouts.obss)), batch_size , replace = False)
            
            self.optimizer.zero_grad()
            pi_loss, approx_kl = self._compute_policy_loss_ppo(rollouts.obss[rand_idxs], old_logp[rand_idxs], 
                                                               rollouts.actions[rand_idxs], rollouts.gaes[rand_idxs])

            v_loss = self._compute_value_loss(rollouts.obss[rand_idxs], rollouts.returns[rand_idxs])
            #print ('vloss:', v_loss, 'pi_loss:', pi_loss)
            loss = .5*v_loss + pi_loss
            
            avg_policy_loss += pi_loss.item()
            avg_value_loss += v_loss.item()
   
            
            if approx_kl > 1.5 * self.target_kl:
                break
            
            loss.backward(retain_graph=True) # lol todo are we supposed to retain graph?
            #loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(acmodel.parameters(), .5)
            
            self.optimizer.step()
            
        return avg_policy_loss/self.train_iters, avg_value_loss/self.train_iters
        
    def _compute_policy_loss_ppo(self, obs, old_logp, actions, advantages):
        policy_loss, approx_kl = 0, 0
        

        dist, _, _, _ = self.acmodel(obs)
        
        new_logp = dist.log_prob(actions.view(-1,self.acmodel.action_dim))
        
        entropy = torch.mean(dist.entropy())
        entropy = entropy * self.entropy_coef

        r = torch.exp(new_logp - old_logp.detach())

        clamp_adv = torch.clamp(r, 1-self.clip_ratio, 1+self.clip_ratio)*advantages.view(-1,1)
        
        min_advs = torch.minimum(r*advantages.view(-1,1), clamp_adv)
        

        policy_loss = -torch.mean(min_advs) - entropy
        approx_kl = (old_logp - new_logp).mean()
        
        return policy_loss, approx_kl
    
    def _compute_value_loss(self, obs, returns):
        _, values, _, _ = self.acmodel(obs)
        
        value_loss = torch.mean((returns.view(-1,1) - values.view(-1,1))**2)

        return value_loss
