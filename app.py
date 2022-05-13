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
    




client = MyRESTClient('xv1hEAplCjSiahoU11LP5ivLG8voopm4')





st.title("Stock trading")

# Market
market = st.sidebar.selectbox("Select market: ",['stocks','crypto'],index=0)

# Ticker
ticker = st.sidebar.text_input("Ticker: ")

# Dates
start_date = st.sidebar.date_input('Start date of training data: ', date(2022,1,1))



if st.sidebar.button('Predict!'):
    
    df = client.get_bars(market=market, ticker=ticker.upper(), from_=start_date)
        
    df['rsi'] = ta.rsi(df['close'])
    
    df['rsi_lag1'] = df.rsi.shift(1)
    df['change'] = (df.close - df.close.shift(1))/df.close.shift(1) 
    
    # Train model
    model = LinearRegression()
    
    df = df.dropna(subset=['rsi_lag1','close','change'])
    
    model.fit(df[['rsi_lag1']],df['change'])

    df_predict = pd.DataFrame(df.iloc[len(df)-1])
    df_predict = df_predict.transpose()
    prediction = model.predict(df_predict[['rsi']])[0]
    st.write('Predicted change in close price next minute from last: ',(prediction*100).round(3),'%')

test_start = st.sidebar.date_input('Start of backtest data: ', date(2022,2,1))



if st.sidebar.button('Backtest!'):
    
    df = client.get_bars(market=market, ticker=ticker.upper(), from_=start_date)
    
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
    df['rsi'] = ta.rsi(df['close'],length=14)
    df['rsi_lag1'] = df.rsi.shift(1)
    df['below30'] = np.where(df.rsi_lag1 < 30,1,0)
    df['change'] = (df.close - df.close.shift(1))/df.close.shift(1) 
    df = df.dropna(subset=['rsi_lag1','close','change'])

    results = pd.DataFrame()
    
    # Just trade during normal market hours and not for the first minute (9:30 AM)
    df = df[(df.market_hours == 1) & (df.min_number > 1)]
    
    for t in stqdm(range(len(df[(df.date>=pd.Timestamp(test_start))]))):
    
        train = df[df.date < pd.Timestamp(test_start) + pd.DateOffset(minutes=t)]
    
        test = df[df.date == pd.Timestamp(test_start) + pd.DateOffset(minutes=t)]
        
        if len(test) > 0:
            
            # Train model
            model = LinearRegression()
            
            model.fit(train[['rsi_lag1']],train['change'])
        
            df_predict = pd.DataFrame(test.iloc[0])
            df_predict = df_predict.transpose()
            test['prediction'] = model.predict(df_predict[['rsi_lag1']])[0]
            results = pd.concat([results,test])
            
    results['growth_rate'] = np.where(results.prediction >= .0000001, results.change,0)

    results['multiplier'] = results.growth_rate + 1
    
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
    
    # Result
    st.write('Value of \$100 invested with this strategy from ',test_start, ' to present: \$',round(pv,2))
    st.write('Value of \$100 invested with buy and hold from ',test_start, ' to present: \$',round(pv_BH,2))

    
    # Chart
    
    # Daily average 
    results['date_date'] = results.date.dt.date
    results2 = results.groupby('date_date',as_index=False)['value','value_BH'].mean()
    results2 = pd.melt(results2,id_vars=['date_date'],var_name=['Strategy'],value_name='v')
    

    # Chart appearance
    axis_font_size = 25
    title_font_size = 30
    width = 1100
    height = 900
    title = ''
        
    chart = alt.Chart(results2).mark_line().encode(
            x='date_date:T',
            y='v:Q',
            color='strategy:N'
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
            'Value over time',
            color='lightgray',
            baseline='top',
            orient='top',
            anchor='start'
                    )).configure_axis(labelFontSize=axis_font_size,  
                                      titleFontSize=axis_font_size).configure_title(fontSize=title_font_size)
    st.altair_chart(chart)

    
    st.write(results)
