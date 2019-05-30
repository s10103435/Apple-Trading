#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:22:57 2019

@author: rikayo
"""
import numpy as np
import pandas as pd

# Generate New Features

#Generate potential features for stock analysis, all the features will be input into a df called aapl_features
#ROC
aapl = pd.read_excel('AAPL.xlsx')
fang = pd.read_excel('FANG.xlsx',sheet_name=1)
fang = fang.set_index(pd.to_datetime(fang['DATE']))
ff = pd.read_csv('F-F_Research_Data_Factors_daily.csv')


def ROC(data,n):
    N = data['Close '].diff(n)
    D = data['Close '].shift(n)
    ROC = pd.Series(N/D,name='Rate of Change')
    data = data.join(ROC)
    return data 
n = 2
aapl = ROC(aapl,n)

#ForceIndex

def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close '].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data
n = 3
aapl = ForceIndex(aapl,n)

# #CCI
# def CCI(close, high, low, n, constant): 
#     TP = (high + low + close) / 3 
#     CCI = pd.Series((TP - TP.shift(1).rolling(20).mean()) / (constant * TP.rolling(20).std()), name = 'CCI_' + str(n)) 
#     return CCI

# aapl['CCI'] = CCI(aapl['Close '], aapl['High '], aapl['Low '], 20, 0.015)
# #aapl['CCI'] = aapl['CCI'].fillna(0)

#EVM
def EVM(data, ndays): 
    dm = ((data['High '] + data['Low '])/2) - ((data['High '].shift(1) + data['Low '].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High '] - data['Low ']))
    EVM = dm / br 
    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data

aapl = EVM(aapl, 2)
#aapl['EVM'] = aapl['EVM'].fillna(0)

#OBV
def on_balance_volume(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'Close '] - df.loc[i, 'Close '] > 0:
            OBV.append(df.loc[i + 1, 'Volume'])
        if df.loc[i + 1, 'Close '] - df.loc[i, 'Close '] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'Close '] - df.loc[i, 'Close '] < 0:
            OBV.append(-df.loc[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.shift(1).rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
    df = df.join(OBV_ma)
    return df

aapl = on_balance_volume(aapl,2)

#MOM
mom = pd.Series([])
for i in range(0,len(aapl['Close '])):
    price_today = aapl.iat[i,1]
    price_5Days = aapl.iat[i-5,1]
    if(i < 5):
        mom[i] = 0
    else:
        momentum =(price_today/price_5Days)*100
        mom[i] = momentum
aapl['Momentum'] = mom.to_frame()


#RSI
def rsi_fn(n):
    delta = aapl['Close '].diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean().abs()
    rs = roll_up / roll_down
    rsi= 100.0 - (100.0 / (1.0 + rs))
    return rsi
aapl['RSI'] = rsi_fn(2)

aapl = aapl.set_index(['Date'])
aapl_features = aapl.iloc[:,6:]
aapl_features = aapl_features.dropna()