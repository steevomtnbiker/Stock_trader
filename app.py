#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:49:20 2022

@author: stephengordon
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from polygon.rest.client import RESTClient
import datetime 
from datetime import date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from stqdm import stqdm
import altair as alt

markets = ['crypto', 'stocks', 'fx']

class MyRESTClient(RESTClient):
    def __init__(self, auth_key: str='xv1hEAplCjSiahoU11LP5ivLG8voopm4', timeout:int=5):
        super().__init__(auth_key)
        retry_strategy = Retry(total=10,
                               backoff_factor=10,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount('https://', adapter)


    def get_tickers(self, market:str=None) -> pd.DataFrame:
            if not market in markets:
                raise Exception(f'Market must be one of {markets}.')
    
            resp = self.reference_tickers_v3(market=market)
            if hasattr(resp, 'results'):
                df = pd.DataFrame(resp.results)
    
                while hasattr(resp, 'next_url'):
                    resp = self.reference_tickers_v3(next_url=resp.next_url)
                    df = df.append(pd.DataFrame(resp.results))
    
                if market == 'crypto':
                    df = df[df['currency_symbol'] == 'USD']
                    df['name'] = df['base_currency_name']
                    df = df[['ticker', 'name', 'market', 'active']]
    
                df = df.drop_duplicates(subset='ticker')
                return df
            return None
        
    def get_bars(self, market:str=None, ticker:str=None, multiplier:int=1,
                 timespan:str='minute', from_:date=None, to:date=None) -> pd.DataFrame:

        if not market in markets:
            raise Exception(f'Market must be one of {markets}.')

        if ticker is None:
            raise Exception('Ticker must not be None.')

        from_ = from_ if from_ else date(2000,1,1)
        to = to if to else date.today()

        if market == 'crypto':
            resp = self.crypto_aggregates(ticker, multiplier, timespan,
                                          from_.strftime('%Y-%m-%d'), to.strftime('%Y-%m-%d'),
                                          limit=50000)
        elif market == 'stocks':
            
            resp = self.stocks_equities_aggregates(ticker, multiplier, timespan,
                                          from_.strftime('%Y-%m-%d'), to.strftime('%Y-%m-%d'),
                                          limit=50000)
            
            
        df = pd.DataFrame(resp.results)
        last_minute = 0
        while resp.results[-1]['t'] > last_minute:
            last_minute = resp.results[-1]['t'] 
            
            last_minute_date = datetime.datetime.fromtimestamp(last_minute/1000).strftime('%Y-%m-%d')
            resp = self.crypto_aggregates(ticker, multiplier, timespan,
                                      last_minute_date, to.strftime('%Y-%m-%d'),
                                      limit=50000)
            new_bars = pd.DataFrame(resp.results)
            df = df.append(new_bars[new_bars['t'] > last_minute])

        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o':'open',
                                'h':'high',
                                'l':'low',
                                'c':'close',
                                'v':'volume',
                                'vw':'vwap',
                                'n':'transactions'})
        df = df[['date','open','high','low','close','volume']]

        return df
    

def Run_backtestV0(market, ticker, test_start, test_end, lag_number, rsi_window, buy_threshold, market_hours):
    
    
    df = client.get_bars(market=market, ticker=ticker.upper(), from_=test_start-pd.Timedelta(days=rsi_window), to=test_end)
    
    # Denote 9:30 AM - 4 PM data (2:30 PM to 9 PM UTC)
    df['market_hours'] = np.where((df.date.dt.hour*60 + df.date.dt.minute >= 870) & (df.date.dt.hour < 21),1,0)

    # Create minute number of day
    temp = df[df.market_hours==1]
    temp['date_date'] = temp.date.dt.date
    temp = temp.sort_values(['date'])
    temp['min_number'] = temp.groupby(['date_date']).cumcount()+1
    temp = temp[['date','min_number']]
    df = df.merge(temp,on='date',how='left')
    df = df.sort_values(['date'])
    
    df['close_lag1'] = df.close.shift(1)
    df['rsi'] = ta.rsi(df['close'],length=rsi_window)
    df['rsi_lag'+str(lag_number)] = df.rsi.shift(lag_number)
    df['change'] = (df.close - df.close.shift(1))/df.close.shift(1) 
    df = df.dropna(subset=['rsi_lag'+str(lag_number),'close','change'])
        
    # Train only on trading hours
    if market_hours == '9:30 AM to 4 PM ET':
        df = df[(df.market_hours == 1) & (df.min_number > 1)]
        
    results = pd.DataFrame()
    test = df[df.date >= pd.Timestamp(test_start)]
    
    
    bought = 0
    pva = 100
    
    
    for i in stqdm(range(len(test))):
        # TO DO: DEBUG THIS, WHY IT'S NOT SHOWING ANY RESULTS
        if bought == 0:
                        
            if test['rsi_lag' + str(lag_number)].iloc[i] < 30:
                                
                bought = 1
                value0 = test.close.iloc[i]
                
                temp = pd.DataFrame(['bought',value0,test.date.iloc[i],pva]).transpose()
                temp.columns = ['action','price','date','v']
                                
                results = pd.concat([temp,results])
    
        else:
            
            growth = (test.close.iloc[i] - value0)/value0
            
            if growth >= buy_threshold:
                
                
                pva += growth*pva
                bought = 0
                
                temp = pd.DataFrame(['sold',test.close.iloc[i],test.date.iloc[i],pva]).transpose()
                temp.columns = ['action','price','date','v']
                
                results = pd.concat([temp,results])
                                
    # check at the end
    if bought == 1:
        
        growth = (test.close.iloc[len(test)-1] - value0)/value0
        pva += growth*pva
        
        temp = pd.DataFrame(['sold',test.close.iloc[len(test)-1],test.date.iloc[len(test)-1],pva]).transpose()
        temp.columns = ['action','price','date','v']
        results = pd.concat([temp,results])

        
    # Daily average 
    results['date_date'] = results.date.dt.date
    results2 = results.groupby('date_date',as_index=False)['v'].mean()
    results2['Strategy'] = 'Brent'
        
        
    return(pva,results,results2)
            
    


def Run_backtest(market, ticker, start_date, test_start, test_end, lag_number, rsi_window, buy_threshold, market_hours, train_freq):
    
    df = client.get_bars(market=market, ticker=ticker.upper(), from_=start_date, to=test_end)
    
    # Denote 9:30 AM - 4 PM data (2:30 PM to 9 PM UTC)
    df['market_hours'] = np.where((df.date.dt.hour*60 + df.date.dt.minute >= 870) & (df.date.dt.hour < 21),1,0)

    # Create minute number of day
    temp = df[df.market_hours==1]
    temp['date_date'] = temp.date.dt.date
    temp = temp.sort_values(['date'])
    temp['min_number'] = temp.groupby(['date_date']).cumcount()+1
    temp = temp[['date','min_number']]
    df = df.merge(temp,on='date',how='left')
    df = df.sort_values(['date'])
    
    df['close_lag1'] = df.close.shift(1)
    df['rsi'] = ta.rsi(df['close'],length=rsi_window)
    df['rsi_lag'+str(lag_number)] = df.rsi.shift(lag_number)
    df['change'] = (df.close - df.close.shift(1))/df.close.shift(1) 
    df = df.dropna(subset=['rsi_lag'+str(lag_number),'close','change'])
        
        
    results = pd.DataFrame()
    train, test = df[df.date < pd.Timestamp(test_start)], df[df.date >= pd.Timestamp(test_start)]
    
    # Loop through chunks    
    for i in stqdm(range(0, len(test), train_freq)):   
    
        if i > 0:
            extra = test.iloc[i-train_freq:i]
            train = pd.concat([train,extra])
        test2 = test.iloc[i:i+train_freq]
        
        # Train only on trading hours
        if market_hours == '9:30 AM to 4 PM ET':
            train = train[(train.market_hours == 1) & (train.min_number > 1)]
        
        if len(test2) > 0:
            
            # Train model

            model = LinearRegression()
            
            model.fit(train[['rsi_lag'+str(lag_number)]],train['change'])
               
            test2 = test2.assign(prediction = model.predict(test2[['rsi_lag'+str(lag_number)]]))
            results = pd.concat([results,test2])
            
    # Decide to buy
    results['growth_rate'] = np.where(results.prediction >= buy_threshold, results.change, 0)
    
    
    
    if market_hours == '9:30 AM to 4 PM ET':
        # Just trade during normal market hours and not for the first minute (9:30 AM)
        results['growth_rate'] = np.where((results.market_hours == 1) & (results.min_number > 1), results.growth_rate, 0)

    results['multiplier'] = results.growth_rate + 1
    
    # Random strategy
    results['growth_rate_random'] = np.random.randint(0, 2, size=len(results))
    results['growth_rate_random'] = np.where(results.growth_rate_random == 1,results.change,0)    
    if market_hours == '9:30 AM to 4 PM ET':
        # Just trade during normal market hours and not for the first minute (9:30 AM)
        results['growth_rate_random'] = np.where((results.market_hours == 1) & (results.min_number > 1), results.growth_rate_random, 0)
    results['multiplier_random'] = results.growth_rate_random + 1 

    
    # Calculate value of investment
    initial_investment = 100

    # The strategy
    pv = initial_investment
    results['value'] = pv
    for i in range(len(results)):
        pv = pv*results.multiplier.iloc[i]
        results['value'].iloc[i] = pv
    
    # With buy and hold
    start_price = results[results.date == results.date.min()]['close_lag1'].iloc[0]
    end_price = results[results.date == results.date.max()]['close'].iloc[0]
    pv_BH = initial_investment*(end_price/start_price)
    results['multiplier_BH'] = results['close']/start_price
    results['value_BH'] = initial_investment*results.multiplier_BH
    
    # Random strategy
    pv_random = initial_investment
    results['value_random'] = pv_random
    for i in range(len(results)):
        pv_random = pv_random*results.multiplier_random.iloc[i]
        results['value_random'].iloc[i] = pv_random

    # Daily average 
    results['date_date'] = results.date.dt.date
    results2 = results.groupby('date_date',as_index=False)['value','value_BH','value_random'].mean()
    results2 = pd.melt(results2,id_vars=['date_date'],var_name=['Strategy'],value_name='v')
    
    return(results, results2, pv, pv_BH, pv_random, start_price, end_price)

def Plot_daily_value(results2, lag_number, rsi_window, buy_threshold):
    
    # Chart appearance
    axis_font_size = 25
    title_font_size = 30
    width = 1100
    height = 900
    title = 'Model: ' + 'RSI window = ' + str(rsi_window) + ', Lag number = ' + str(lag_number) + ', Buy threshold = ' + str(buy_threshold) + ', Market hours = ' + str(market_hours) 
        
    chart = alt.Chart(results2).mark_line().encode(
            x='monthdate(date_date):T',
            y='v:Q',
            color='Strategy:N'
        ) 
        
    chart = alt.layer(chart , chart.mark_point(size=100, opacity=0, tooltip=alt.TooltipContent("data"))).properties(
        title=alt.TitleParams(
    title,
    baseline='bottom',
    orient='bottom',
    anchor='middle',
    fontWeight='normal',
    fontSize=10
        ),
        width=width,
        height=height)
        
        
    chart = alt.concat(chart,
            title=alt.TitleParams(
            'Backtest value over time for: '+ticker,
            color='lightgray',
            baseline='top',
            orient='top',
            anchor='start'
                    )).configure_axis(labelFontSize=axis_font_size,  
                                      titleFontSize=axis_font_size).configure_title(fontSize=title_font_size)
    
    return(chart)


client = MyRESTClient('xv1hEAplCjSiahoU11LP5ivLG8voopm4')





st.title("Stock trading")

# Market
market = st.sidebar.selectbox("Select market: ",['stocks','crypto'],index=0)

# Ticker
ticker = st.sidebar.text_input("Ticker: ", value='AAPL')

# Dates
start_date = st.sidebar.date_input('Start date of training data: ', date(2022,1,1))

# How often (minutes) to retrain model
train_freq = st.sidebar.number_input('How often to retrain model (minutes): ',value=1000)

st.sidebar.write('Model configuration')
market_hours = st.sidebar.radio('Market hours: ',('9:30 AM to 4 PM ET','24-7'))
rsi_window = st.sidebar.number_input('RSI window: ',14)
lag_number = st.sidebar.number_input('How far (minutes) back to lag RSI in model (must be at least 1): ',1)
buy_threshold = st.sidebar.number_input("Buy threshold (% change): ",value=.0000001,format='%f')
Brent_threshold = st.sidebar.number_input("Brent's sell threshold (% change): ",value=.0033,format='%f')

if st.sidebar.button('Predict!'):
    
    df = client.get_bars(market=market, ticker=ticker.upper(), from_=start_date)
        
    df['rsi'] = ta.rsi(df['close'],length=rsi_window)
    
    df['rsi_lag'+str(lag_number)] = df.rsi.shift(lag_number)
    df['change'] = (df.close - df.close.shift(1))/df.close.shift(1) 
    
    # Train model
    model = LinearRegression()
    
    df = df.dropna(subset=['rsi_lag'+str(lag_number),'close','change'])
    
    model.fit(df[['rsi_lag'+str(lag_number)]],df['change'])

    df_predict = pd.DataFrame(df.iloc[len(df)-1])
    df_predict = df_predict.transpose()
    prediction = model.predict(df_predict[['rsi_lag'+str(lag_number)]])[0]
    st.write('Predicted change in close price next minute from last: ',(prediction*100).round(3),'%')


# Test date start
test_start = st.sidebar.date_input('Start of backtest data: ', date(2022,2,1))

# Test date end
test_end = st.sidebar.date_input('End of backtest data: ', date(2022,6,1))


if st.sidebar.button('Backtest!'):
    
    results, results2, pv, pv_BH, pv_random, start_price, end_price = Run_backtest(market, ticker, start_date, test_start, test_end, lag_number, rsi_window, buy_threshold, market_hours, train_freq)
    pvV0, V0logs, results2V0 =  Run_backtestV0(market, ticker, test_start, test_end, lag_number, rsi_window, Brent_threshold, market_hours)
        
    # Expand number of days in Brent strategy
    temp = pd.DataFrame(results2[['date_date']]).drop_duplicates()
    results2V0 = results2V0.merge(temp,on=['date_date'],how='outer')
    results2V0 = results2V0.sort_values(['date_date'])
    results2V0['v'] = np.where(results2V0.v.isna(),results2V0.v.ffill(),results2V0.v)
    results2V0['Strategy'] = 'Brent'
        
    results2 = pd.concat([results2,results2V0])
    results2 = results2.sort_values(['Strategy','date_date'])          
    
    # Result
    st.write('Value of \$100 with Brent strategy from ',test_start, ' to present: ',round(pvV0,2))
    st.write('Value of \$100 with this strategy from ',test_start, ' to present: ',round(pv,2))
    st.write('Value of \$100 with buy and hold from ',test_start, ' to present: ',round(pv_BH,2), ', based on starting price of ',start_price,' and ending price of ',end_price)
    st.write('Value of \$100 with random strategy from ',test_start, ' to present: ',round(pv_random,2))
    

    # Chart of daily average
    chart = Plot_daily_value(results2, lag_number, rsi_window, buy_threshold)
    st.altair_chart(chart)
    
    st.download_button('Download logs for Brent strategy!', V0logs.to_csv().encode('utf-8'),file_name='Stock trading results.csv',mime='text/csv')
    
    st.write(results2)
    
# TO DO: Add in multiple backtests grouped together
# if st.sidebar.button('Backtest!'):
    
#     results, results2, pv, pv_BH, pv_random, start_price, end_price = Run_backtest(market, ticker, start_date, test_start, test_end, lag_number, rsi_window, buy_threshold, market_hours, train_freq)
           
#     # Result
#     st.write('Value of \$100 with this strategy from ',test_start, ' to present: ',round(pv,2))
#     st.write('Value of \$100 with buy and hold from ',test_start, ' to present: ',round(pv_BH,2), ', based on starting price of ',start_price,' and ending price of ',end_price)
#     st.write('Value of \$100 with random strategy from ',test_start, ' to present: ',round(pv_random,2))

    
#     # Chart of daily average
#     chart = Plot_daily_value(results2, lag_number, rsi_window, buy_threshold)
#     st.altair_chart(chart)

    
#     st.write(results2)
    
#     st.download_button('Download data!', results.to_csv().encode('utf-8'),file_name='Stock trading results.csv',mime='text/csv')
    
       
    
