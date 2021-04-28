BASE_URL = "https://paper-api.alpaca.markets"
KEY_ID = "" # generate api keys from alpaca.markets and paste API Key ID here
SECRET_KEY = "" # and secret key here

TECHNICAL_INDICATORS_LIST = ["macd", "macds",
                             "boll_ub","boll_lb",
                             "rsi_5", "rsi_14", "rsi_30", 
                             "cci_30", "dx_30",
                             "open_5_sma", "open_14_sma", "open_30_sma"]
TA_LOOKBACK_NEEDED = 30 # lookback necessary to get first data point for any of the TA (ie. rsi_30 means 30 days)

TICKER_FILE = "dji.csv"
PPO_MODEL_FILE = "../rl_with_ta/best_model.zip"

INITIAL_BALANCE = 10000.0 # alpaca paper trading starts with 10000 i believe
NUM_PAST_STATES = 14
EPISODE_LENGTH = 30

SLIPPAGE = 0.05 # underestimate portfolio value by 5%.. may change