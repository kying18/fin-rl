import numpy as np
import pandas as pd

import config
from data import get_data
from env.environment import TradeEnv

# api = tradeapi.REST(
#     base_url=config.BASE_URL,
#     key_id=config.KEY_ID,
#     secret_key=config.SECRET_KEY
# )

# session = requests.session()

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

# def trade(tickers, action):
#     api.submit_order(
#     symbol='SPY',
#     notional=450,  # notional value of 1.5 shares of SPY at $300
#     side='buy',
#     type='market',
#     time_in_force='day',
# )
