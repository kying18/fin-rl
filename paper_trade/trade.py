import numpy as np
import pandas as pd

import config
from data import get_data
from env.environment import TradeEnv
from trade_platform import TradePlatform

def get_tickers(ticker_file):
    return pd.read_csv(ticker_file)['Tickers']

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def get_action(tickers):
    model = PPO.load(config.PPO_MODEL_FILE)
    data = get_data()

    trade_env = TradeEnv(tickers=tickers, data=data)

    obs = trade_env.reset(index=-1, balance=config.INITIAL_BALANCE) # reset obs to yesterday
    action = model.predict(obs, deterministic = True)[0]

    return softmax(action) # returns the actions that we should be taking

def main():
    p = TradePlatform()
    tickers = get_tickers(config.TICKER_FILE)
    action = get_action(tickers)

    p.cancel_existing_orders() # should not really do anything, we shouldnt have any outstanding orders
    p.execute_action(action, tickers)

if __name__ == '__main__':
    main()
