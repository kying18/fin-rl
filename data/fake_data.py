import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

TICKER_PREFIX = 'T'

def get_tickers(num_tickers):
    ticker_list = [f"{TICKER_PREFIX}{i}" for i in range(num_tickers)]
    return ticker_list

def generate_data(ticker_list, start_date='2000-01-01', end_date='2010-01-01'):
    # Create a calendar
    nyse = mcal.get_calendar('NYSE')
    dates = nyse.schedule(start_date=start_date, end_date=end_date)
    dates['date'] = dates['market_open'].dt.date

    df = pd.DataFrame(index=range(len(dates['date'])))
    df = df.reset_index()
    df.index = dates['date']

    for ticker in ticker_list:
        a = np.random.uniform(1, 1000)
        b = np.random.uniform(0.05, 1)
        c = np.random.uniform(0.1, .4)
        d = np.random.uniform(0.1, (a-b)/20)
        e = np.random.uniform(-0.03, 0.08)
        f = np.random.uniform(10, 25)
        g = np.random.uniform(0.003, 0.008)
        shift1 = np.random.uniform(0, 2*math.pi)
        shift2 = np.random.uniform(0, 2*math.pi)
        
        df[ticker] = np.maximum(a + b*np.sin(c*df['index']+shift1) + np.random.normal(loc=0, scale=d, size=len(df['index'])) + e * df['index'] + f * np.sin(g*df['index']+shift2), 0.01)

    df = df[df.columns[1:]] # get rid of index col


    return df
    

def save_data(data, file_path):
    data.to_csv(file_path)

if __name__ == '__main__':
    ticker_list = get_tickers(50)
    # print(ticker_list)

    data = generate_data(ticker_list, start_date='2000-01-01', end_date='2005-01-01')
    # print(data)

    save_data(data, 'fake_data.csv')

    data.plot(use_index=True, y=data.columns[1:10])
    plt.show()