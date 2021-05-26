import datetime as dtt
import pandas as pd
import yfinance as yf
import numpy as np
import stockstats
import matplotlib.pyplot as plt
import copy
import pandas_market_calendars as mcal
import warnings
warnings.simplefilter(action='ignore') # COMMENT OUT IF YOU WANT WARNINGS

import config

###This class is from FinRL. Basically uses yahoo finance API to pull in stock data
class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            temp_df["tic"] = tic
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

def get_start_end_dates():
    today = dtt.date.today()
    end_date = (today - dtt.timedelta(days=1)).strftime("%Y-%m-%d") # end date is yesterday
    start_date = (today - dtt.timedelta(days=2*(config.TA_LOOKBACK_NEEDED+config.NUM_PAST_STATES))).strftime("%Y-%m-%d") # subtracting some extra from start date so we can match calendar

    nyse = mcal.get_calendar('NYSE')
    dates = nyse.schedule(start_date=start_date, end_date=end_date)
    dates = list(dates.index)

    end_date = dates[-1]
    start_date_lookback = config.TA_LOOKBACK_NEEDED + config.NUM_PAST_STATES # TODO might have an off by 1 error here, not sure
    start_date = dates[-start_date_lookback]

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    

def get_data(ticker_file, price_label='close'):
    start_date, end_date = get_start_end_dates()

    tickers = pd.read_csv(ticker_file)['Tickers']
    downloader = YahooDownloader(start_date=start_date,
                                 end_date=end_date,
                                 ticker_list=tickers)

    df = downloader.fetch_data()

    # check if data for all
    num_days = len(np.unique(df['date']))
    valid_tics = []
    for tic in tickers:
        if len(df.loc[df['tic'] == tic]) == num_days:
            valid_tics.append(tic)
        else:
            print("Invalid ticker:", tic)

    i_dfs = {}
    i_ss = {}

    df = df.drop(columns = ['day'])
    for tic in valid_tics:
        mask = df['tic'] == tic

        i_dfs[tic] = df[mask]
        i_ss[tic] = stockstats.StockDataFrame.retype(copy.deepcopy(df[mask]))
        
    for key, ss in i_ss.items():
        for ta in config.TECHNICAL_INDICATORS_LIST:
            i_dfs[key][ta] = ss.get(ta).values
        
    for key, val in i_dfs.items():
        val['price'] = val[price_label].values
        i_dfs[key] = val.drop(columns=['open', 'high', 'low', 'close'])
        i_dfs[key] = val[30:].reset_index(drop=True)

    return i_dfs

if __name__ == '__main__':
    get_data(ticker_file=config.TICKER_FILE, price_label='close')