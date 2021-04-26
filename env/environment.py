import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import copy
import os

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

INITIAL_BALANCE = 1000.0
NUM_PAST_STATES = 2
EPISODE_LENGTH = 38 - NUM_PAST_STATES


class TradeEnv(gym.Env):
    def __init__(self, tickers, data_path):
        super(TradeEnv, self).__init__()

        self.tickers = tickers
        self.total = pd.read_csv(data_path)[tickers]
        
        self.means = np.mean(self.total[tickers].values, axis = 0)
        self.stds = np.std(self.total[tickers].values, axis = 0)

        self.episode_length = EPISODE_LENGTH #number of trading minutes in episode

        self.num_past_states = NUM_PAST_STATES #number of past days that are used in state

        self.action_space = spaces.Box(low=-10, high=10, shape=(len(self.tickers) + 1,))
                                            

        obs_length = len(self.tickers)*self.num_past_states #observation due to past stacked states
        obs_length += 1 #balance
        obs_length += len(self.tickers) #holdings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_length,))
        
        
  
    def step(self, action_):
        action = softmax(action_)
        
        
        self.balance += np.sum(self.holdings)

        self.holdings = self.balance*action[:-1]
        self.balance = self.balance*action[-1]
        
        self.last_net_worth = self.balance + np.sum(self.holdings)

        self.index += 1

        stock_obs = self.get_stock_obs(self.index)
        self.next_prices = stock_obs[-1]
        perc_change = np.divide(self.next_prices, self.curr_prices)
        self.holdings = np.multiply(self.holdings, perc_change)
        
        self.curr_prices = self.next_prices
        

        self.net_worth = self.balance + np.sum(self.holdings)


        rew = self.net_worth - self.last_net_worth # reward is the delta between last net worth and current net worth

        done = (self.net_worth <= 0) or (self.steps > self.episode_length)
        self.steps += 1

        
        obs = self.get_obs(stock_obs, self.balance, self.holdings)

        
    
        
        
        obs = self.get_obs(self.normalize_stock_obs(stock_obs), self.balance/1000.0, self.holdings/1000.0)

        return obs, 200.0*rew, done, {}
    
    def normalize_stock_obs(self, stock_obs):
        return np.divide(stock_obs - self.means, self.stds)
    
    def get_stock_obs(self, index):        
        return self.data[index - self.num_past_states:index][self.tickers].values #stack data

    def get_obs(self, stock_obs, balance, holdings):
        return np.concatenate([stock_obs.reshape(-1,), [balance], holdings])
        
    def reset(self):
        # df_idx = np.random.randint(len(self.total))
        # self.data = self.total[df_idx]
        self.data = self.total.sample(n=1)
        self.steps = 0
        self.index = NUM_PAST_STATES
        
        stock_obs = self.get_stock_obs(self.index)
        self.holdings = np.zeros(len(self.tickers)) #holdings of each stock in number of shares
        self.balance = INITIAL_BALANCE
        self.last_net_worth = INITIAL_BALANCE

        self.curr_prices = stock_obs[-1]

        obs = self.get_obs(self.normalize_stock_obs(stock_obs), self.balance/1000.0, self.holdings/1000.0)
        return obs  # reward, done, info can't be included


if __name__ == '__main__':
    trade_env = TradeEnv(tickers=[f'T{i}' for i in range(10)], data_path='../data/fake_data.csv')
    trade_env.reset()
    # action = np.array([0,0])
    # obs, rew, done = trade_env.step(action)
    # print (obs)
    # print()
