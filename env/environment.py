import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import copy


INITIAL_BALANCE = 1000
EPISODE_LENGTH = 5
NUM_PAST_STATES = 2


class TradeEnv(gym.Env):
    def __init__(self, tickers, data_path = 'data.csv'):
        super(TradeEnv, self).__init__()

        self.tickers = tickers
        self.data = pd.read_csv(data_path) #cols = [datetime, stock1, stock2, stock3, ...]
        self.episode_length = EPISODE_LENGTH #number of trading days in episode
        self.data_length = len(self.data) #number of total days in historical data todo change
        self.num_past_states = NUM_PAST_STATES #number of past days that are used in state

        self.action_space = spaces.Box(low=-1., high=1., shape=(len(self.tickers),))
                                            

        obs_length = len(self.tickers)*self.num_past_states #observation due to past stacked states
        obs_length += 1 #balance
        obs_length += len(self.tickers) #holdings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_length,))
  
    def step(self, action):
        # + / - / 0
        # -1: sell everything of that stock (all shares), -0.5: sell half of your shares
        # 0: hold
        # 1: could be percentage of balance left?? 1 - spend remaining balance on this stock, .5 - spend half, etc.
        
        sell_mask = action <= 0.0
        sell_action = copy.copy(action)
        sell_action[~sell_mask] = 0.0

        buy_action = copy.copy(action)
        buy_action[sell_mask] = 0.0

        if np.sum(buy_action) >= 1.0:
            buy_action = buy_action/np.sum(buy_action)
        
        self.balance += np.dot(self.curr_prices, -np.multiply(sell_action, self.holdings))
        self.holdings += np.multiply(self.holdings, sell_action) # sell action is negative, so this is updating holdings after selling
        self.holdings += np.divide(self.balance*buy_action, self.curr_prices) # updating holdings to buy more
        
        self.net_worth = self.get_net_worth()
        
        rew = self.net_worth - self.last_net_worth # reward is the delta between last net worth and current net worth
        self.last_net_worth = self.net_worth # setting current net worth to equal last net worth now

        done = (self.net_worth <= 0) or (self.steps > self.episode_length)
        self.steps += 1
        self.index += 1

        stock_obs = self.get_stock_obs(index)
        
        obs = get_obs(stock_obs, self.balance, self.holdings)
        
        self.curr_prices = stock_obs[-1]

        
        return obs, rew, done, {}

    def get_net_worth(self):
        return np.multiply(self.holdings, self.curr_prices) + self.balance

    def get_stock_obs(self, index):
        return self.data[self.index - self.num_past_states:self.index][self.tickers].values.reshape(-1,) #stack data

    def get_obs(self, stock_obs, balance, holdings):
        return np.concatenate([[self.balance], self.holdings, self.stock_obs])
        
    def reset(self):
        self.index = 2
        # self.index = np.random.randint(self.num_past_states, self.data_length - self.episode_length) #starting index
        self.stock_obs = self.get_stock_obs(self.index)
        self.holdings = np.zeros(len(self.tickers)) #holdings of each stock in number of shares
        self.balance = INITIAL_BALANCE
        self.last_net_worth = INITIAL_BALANCE
        self.steps = 0

        self.curr_prices = self.stock_obs[-1]

        obs = self.get_obs(self.stock_obs, self.balance, self.holdings)
        return obs  # reward, done, info can't be included


if __name__ == '__main__':
    trade_env = TradeEnv(tickers=['AAPL', 'SPY'])
    print(trade_env.reset())
    action = np.array([0,0])
    obs, rew, done = trade_env.step(action)
    print (obs)
    # print()
