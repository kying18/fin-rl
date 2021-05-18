SMOOTH_REWARD_WINDOW = 30

# TECHNICAL_INDICATORS_LIST = ["macd", "macds",
#                              "boll_ub","boll_lb",
#                              "rsi_5", "rsi_14", "rsi_30", 
#                              "cci_30", "dx_30",
#                              "open_5_sma", "open_14_sma", "open_30_sma"]
TECHNICAL_INDICATORS_LIST = ["open_5_sma", "open_14_sma", "open_30_sma"]
                             
TA_LOOKBACK_NEEDED = 30 # lookback necessary to get first data point for any of the TA (ie. rsi_30 means 30 days)

TICKER_FILE = "dji.csv"
PPO_MODEL_FILE = "../../rl_with_ta/best_model.zip"

INITIAL_BALANCE = 10000.0
NUM_PAST_STATES = 14
EPISODE_LENGTH = 30