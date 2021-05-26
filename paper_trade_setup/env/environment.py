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

#OpenAI Gym style environment for RL
class TradeEnv(gym.Env):
    def __init__(self, tickers, data, features, episode_length, num_past_states):
        super(TradeEnv, self).__init__()

        self.tickers = tickers

        self.data = {}
        self.features_list = features
        self.features = {}
        self.means = {}
        self.stds = {}
        self.prices = {}
        for key, value in data.items():
            self.data[key] = value[features]
            self.means[key] = np.mean(value[features], axis = 0)
            self.stds[key] = np.std(value[features], axis = 0)
            
            #Normalize features to have zero mean and unit standard deviation
            self.features[key] = np.divide(data[key][features] - self.means[key],
                                          self.stds[key])
            
            self.prices[key] = data[key]['price'].values
        
        self.prices = pd.DataFrame.from_dict(self.prices)
        
        #self.prices is a dataframe with each ticker being a key 
        #and the corresponding series representing the stock prices
        
        #Will be used later for normalization
        self.price_means = np.mean(self.prices, axis = 0).values
        self.price_stds = np.std(self.prices, axis = 0).values
        
        self.episode_length = episode_length #number of trading minutes in episode

        self.num_past_states = num_past_states #number of past days that are used in state

        self.action_space = spaces.Box(low=-10, high=10, shape=(len(self.tickers) + 1,))
                                            

        obs_length = len(self.tickers)*(self.num_past_states) #observation due to past stacked states
        obs_length += 1 #balance
        obs_length += len(self.tickers) #holdings
        obs_length += len(self.tickers)*len(features) #number of technical analysis features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_length,))


    def step(self, action_):
        
        #Apply softmax to RL output so that actions sum to 1
        action = softmax(action_)

        #Liquidate past holdings
        self.balance += np.sum(self.holdings)
        
        
        #New Portfolio at end of day
        self.holdings = self.balance*action[:-1]
        self.balance = self.balance*action[-1]
        
        #Net worth at end of day
        self.last_net_worth = self.balance + np.sum(self.holdings)
        
        #Step into next day
        self.index += 1
        #Get stock prices at next day
        stock_obs = self.get_stock_obs(self.index)
        self.next_prices = stock_obs[-1]
        
        #Update value of current holdings
        perc_change = np.divide(self.next_prices, self.curr_prices)
        self.holdings = np.multiply(self.holdings, perc_change)

        self.curr_prices = self.next_prices
        
        self.net_worth = self.balance + np.sum(self.holdings)

        rew = self.net_worth - self.last_net_worth # reward is the delta between last net worth and current net worth

        self.steps += 1
        done = (self.net_worth <= 0) or (self.steps >= self.episode_length)

        obs = self.get_obs(stock_obs, self.balance, self.holdings, self.index)
        self.cum_rew += rew

        return obs, rew, done, {}
    
    
    def get_stock_obs(self, index):
        ret= self.prices[index - self.num_past_states:index][self.tickers].values #stack data
        return ret

    def get_obs(self, stock_obs, balance, holdings, index):
        #Normalize stock prices for inclusion in observations
        prices_norm = np.divide(stock_obs - self.price_means,
                               self.price_stds).reshape(-1,)
        
        feature_vals = np.array([])
        ix = index - 1
        #Add in features at current timestep, for each ticker
        for tic in self.tickers:
            feature_vals = np.append(feature_vals, (self.features[tic].iloc[ix][self.features_list].values))
        
        #Form observation and normalize balance and holdings
        return np.concatenate([prices_norm, [balance/1000.0], holdings/1000.0, feature_vals])

    def reset(self, index=None, balance=10000):
        self.cum_rew = 0.0
        self.steps = 0
        if index is None:
            self.index = np.random.randint(2*NUM_PAST_STATES, len(self.prices) - EPISODE_LENGTH - 10)
        else:
            self.index = index

        stock_obs = self.get_stock_obs(self.index)
        self.holdings = np.zeros(len(self.tickers)) #holdings of each stock in number of shares
        self.balance = balance
        self.last_net_worth = balance
        self.net_worth = balance
        
        self.curr_prices = stock_obs[-1]
        
        obs = self.get_obs(stock_obs, self.balance, self.holdings, self.index)
        return obs  # reward, done, info can't be included


if __name__ == '__main__':
    pass