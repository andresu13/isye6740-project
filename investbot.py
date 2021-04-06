import os
import subprocess
import numpy as np
import pandas as pd
import datetime
import time
import pytz
from pytz import timezone
import alpaca_trade_api as tradeapi
#from config import ACCESS_KEYS
import bs4 as bs
import pickle
import requests
from scipy.optimize import minimize

# Gets the current 500 tickers from the S&P 500 and output a list of their respective symbols
def save_sp500_tickers():
   
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    tickers = map(lambda s: s.strip(), tickers)
    return list(tickers)


def prices_f(list_stocks, start, end, time_scale, ohlcv, tzone):
    # Description:  Obtain a pandas dataframe with historical prices for tradable stocks 
    # Inputs:
    #    list_stocks = [sec1, sec2, sec3]
    #    start/end respective dates 
    #    Data points quantity of data points to extract 
    #    time scale 'minute', 'hour', 'day'
    prices_df = pd.DataFrame()
    for sec in list_stocks: 
        # Extract data using data points 
        #sec_df = api.polygon.historic_agg(time_scale, sec, limit = #_data_points).df[['open']] 
        # Extract data using date for time period
        sec_df = api.polygon.historic_agg(time_scale, sec, _from = start, to = end).df[[ohlcv]]
        sec_df = sec_df.rename({ohlcv:sec}, axis = 'columns')
        prices_df = pd.concat([prices_df, sec_df], axis=1).ffill().bfill()
    prices_df2 = prices_df[(prices_df.index.time >= pd.Timestamp('09:30', tz=tzone).time()) & (prices_df.index.time <= pd.Timestamp('16:00', tz=tzone).time())]
    return prices_df2

def dp_to_srtend_time(dp,n, tzone): 
    # Description: Converts data points to timestring
    # Inputs: 
    #   dp = data points to capture 
    #   n = multiplier in case of mulitple data windows are needed for calculations
    #   tzone = time zone to be used. EST should be the standard 
    days = int((np.ceil(data_points/390)+2)*n) #Converts data points into days
    # The + 2 accounts for weekends & the *n n times the amount of data for calculation purposes
    days_str = str(days)+' days' #days into required string format 
    end_dt = pd.Timestamp.now(tz=tzone) # set end time being the current moment
    start_dt = end_dt - pd.Timedelta(days_str) # set start time 
    start = start_dt.strftime('%m-%d-%Y %H:%M:%S-400') # Convert to correct format
    end = end_dt.strftime('%m-%d-%Y %H:%M:%S-400') # Conver to correct format 
    return (start, end)

def order_size_buy(list_current_prices,Div_factor):
    # Description: Get order size for stocks to buy optimizing max usage of buying power under given contraints
    # Inputs
    # 1) Pandas series with current prices of stocks to buy
    # 2) The amount of stocks we want our portfolio to be divided by
    # Outputs
    # 1) Pandas series with order sizes for stocks to buy
    def objective(qty):
        obj = float(account.buying_power)*.9-np.dot(pd.Series(qty),list_current_prices).sum()
        #print(obj)
        return obj
    qty0 = pd.Series([10]*len(list_current_prices))
    Upper_b = np.floor((float(account.portfolio_value)/Div_factor)/list_current_prices)
    Lower_b = pd.Series([0]*len(Upper_b))
    bnds = tuple(zip(Lower_b,Upper_b))
    # Cons is set to add more constraints laters on
    def constraint_BP(qty):
        # Constraints Buying power - stocks to buy >0 
        return float(account.buying_power)*.9-np.dot(pd.Series(qty),list_current_prices).sum()
    con1 = {'type':'ineq','fun':constraint_BP}
    cons = [con1]
    sol = minimize(objective,qty0,method='SLSQP',bounds=bnds,constraints = cons)
    R = pd.Series(np.floor(sol.x),index=list_current_prices.index)
    return R[R!=0]

### Main/Driver Code ###

# Initialize variables
api = tradeapi.REST(key_id='PKHE6FHRIO5DS3TY47FS',secret_key='/r1wAfhxP2tg4cpoygUxlzoS0UvyJRmQ3nKtkI5J',
            base_url='https://paper-api.alpaca.markets')
NY = timezone('US/Eastern') # Timezone
time_scale = 'minute' # data point time scale 
data_points = 60 # number of data points to use
universe = save_sp500_tickers()
ohlcv = 'open' # choose between open, high, low, close, volume

(start, end) = dp_to_srtend_time(data_points, 1, NY)

prices_df = prices_f(universe, start, end, time_scale, ohlcv, NY) # Gets historical data
#prices_df.melt(id_)
print(prices_df)
prices_df.to_csv('prices_df.csv')