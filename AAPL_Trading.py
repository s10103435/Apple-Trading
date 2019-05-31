#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:50:54 2019

@author: Shalabh Mittal
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Getting Result From other self-created script
import NewFeatures as nf
import lstm
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
        
# Import All the data 
aapl = pd.read_excel('AAPL.xlsx')
fang = pd.read_excel('FANG.xlsx',sheet_name=1)
fang = fang.set_index(pd.to_datetime(fang['DATE']))
#ff = pd.read_csv('F-F_Research_Data_Factors_daily.csv')

# change the format of original datas
aapl = pd.read_excel('AAPL.xlsx')
fang = pd.read_excel('FANG.xlsx',sheet_name=1)
ff = pd.read_csv('F-F_Research.csv',skiprows=4,nrows=24390)
ads = pd.read_excel('ADS_Index_Most_Current_Vintage.xlsx')
# change the index to date
fang = fang.set_index(pd.to_datetime(fang['DATE']))
ff = ff.set_index(pd.to_datetime(ff['0']))
# shift the apple stock price part so we can use the past data to predict the future data
aapl.index = aapl['Date']
aapl = aapl.drop(['Date'],axis=1)
#aapl.index = aapl.index.shift(-1,freq='D')
#combine them altogether
data = aapl.join([fang,ff,ads])
# Time shifting in here
data['Close_next_day'] = data['Close '].shift(-1)
data = data.dropna()


X = data
Y = data['Close_next_day']
# train and test split
rand = np.random.choice(len(data),600,replace=True)
X_train = data.iloc[rand]
y_train = Y.iloc[rand]
X_test = data.iloc[563:]
y_test = Y.iloc[563:]

# =============================================================================
# # choose data randomly
# other_rand = np.setdiff1d(np.arange(len(data)),rand)
# X_test = data.iloc[other_rand]
# y_test = Y.iloc[other_rand]
# =============================================================================

# in order to align with LSTM model, the test sets will be the data for last 60 days
_avg_R_square_train = list()
_beta_model_rat = list()
_beta_model_ff = list() 
_beta_model_poly = list()
_beta_model_pca = list()
_eig_Vecs_pca = list()

def fit(X_train,y_train,nums):
    '''The fit function will call the single_train function 'nums' times;
        each time will pass one bootstrap number into single_train so it can use the 'random'
        to derive bootstrap samples.'''
    for i in range(nums):
        random = np.random.choice(len(X_train),500)
        single_train(random)
    print('The average R2 score of all the model is:', sum(_avg_R_square_train)/len(_avg_R_square_train))
  
            
def single_train(random):   
    Rsq_dict = dict()
    ff = ff_ads_models(random)
    rat = ols_rat(random)
    poly = poly_models(random)
#   pca = pca_models(random)
    
    
    Rsq_dict[ff[0]]= 'ff' 
    Rsq_dict[rat[0]]= 'rat'
#    Rsq_dict[pca[0]]= 'pca'
    Rsq_dict[poly[0]]= 'poly'
    
    best_model = sorted(Rsq_dict.items(),reverse=True)[0][1]
    
    if best_model == 'rat':             
        _beta_model_rat.append(rat[1])
        _avg_R_square_train.append(rat[0])
    elif best_model == 'poly':    
         _beta_model_poly.append(poly[1])
         _avg_R_square_train.append(poly[0])
    elif best_model == 'ff':      
         _beta_model_ff.append(ff[1])
         _avg_R_square_train.append(ff[0])
    elif best_model == 'pca':      
          _beta_model_pca.append(pca[1])
          _avg_R_square_train.append(pca[0])
          _eig_Vecs_pca.append(pca[2])

def ols_rat(random): 
    cols = ['RF','ADS_Index', 'TSLA']
    X0 = np.array(X_train[cols])
    X = X0[random]
    
    Y0 = np.array(y_train)
    Y = Y0[random]      
    
    X = np.column_stack([np.ones(len(X)),X])      
    
    invXX = np.linalg.inv(X.transpose()@X)       
    'OLS estimates for coefficients: X x 1'
    beta_hat= invXX @ X.transpose() @ Y
    'Predictive value of Y using OLS'
    y_hat = X@beta_hat
    'Residuals from OLS'
    residuals = Y - y_hat
    'Variance of residuals'
    sigma2 = residuals.transpose()@residuals        
    'Calculate R-square'
    R_square = 1- (residuals.transpose()@residuals)/(len(X)*np.var(Y))
    return (R_square,beta_hat)

def ff_ads_models(random):
       # X_train = X_train.loc[:,['Mkt-RF','SMB','HML','ADS_Index']]
    cols = ['Mkt-RF','SMB','HML','ADS_Index'] 
    X0 = np.array(X_train[cols])
    X = X0[random]
    Y0 = np.array(y_train)
    Y = Y0[random]      
    X = np.column_stack([np.ones(len(X)),X])      
    invXX = np.linalg.inv(X.transpose()@X)       
    'OLS estimates for coefficients: X x 1'
    beta_hat= invXX @ X.transpose() @ Y
    'Predictive value of Y using OLS'
    y_hat = X@beta_hat
    'Residuals from OLS'
    residuals = Y - y_hat       
    'Calculate R-square'
    R_square = 1- (residuals.transpose()@residuals)/(len(X)*np.var(Y))
    return (R_square,beta_hat)


def poly_models(random):
    cols = ['RF','GOOGL','ADS_Index','AAPL', 'AMZN', 'FB', 'GOOGL', 'NFLX', 'NVDA', 'TSLA']
    X0 = np.array(X_train[cols])
    X = X0[random]
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X)
    X = X_train_poly[:,[0,1,36]]         
    Y0 = np.array(y_train)
    Y = Y0[random]
    invXX = np.linalg.inv(X.transpose()@X)       
    'OLS estimates for coefficients: X x 1'   
    beta_hat= invXX @ X.transpose() @ Y    
    'Predictive value of Y using OLS'
    y_hat = X@beta_hat
    'Residuals from OLS'
    residuals = Y - y_hat 
    'Calculate R-square'
    R_square = 1- (residuals.transpose()@residuals)/(len(X)*np.var(Y))
    return (R_square,beta_hat)   

## PCA preprocession 


def PCA_train(X):
    normalise = StandardScaler().fit_transform(X.values)
    mean_vec = np.mean(normalise, axis=0)
    cov_mat = (normalise - mean_vec).T.dot((normalise - mean_vec)) / (normalise.shape[0]-1)
    cov_mat = np.cov(normalise.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    cor_mat1 = np.corrcoef(normalise.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    cor_mat2 = np.corrcoef(X.values.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

    u,s,v = np.linalg.svd(normalise.T)

    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#    print('ok')

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()
       
    pca1 = np.dot(X.values,eig_vecs[0])
    pca2=np.dot(X.values,eig_vecs[1])
    pca3=np.dot(X.values,eig_vecs[2])
    pca4=np.dot(X.values,eig_vecs[3])
    pca5=np.dot(X.values,eig_vecs[4])
    pca6=np.dot(X.values,eig_vecs[5])
    pca7=np.dot(X.values,eig_vecs[6])
    pca8=np.dot(X.values,eig_vecs[7])
    pca9=np.dot(X.values,eig_vecs[8])
    
    pca_train = pd.DataFrame()
    pca_train['pca1']=pca1
    pca_train['pca2']=pca2
    pca_train['pca3']=pca3
    pca_train['pca4']=pca4
    pca_train['pca5']=pca5
    pca_train['pca6']=pca6
    pca_train['pca7']=pca7
    pca_train['pca8']=pca8
    pca_train['pca9']=pca9
    return (pca_train, eig_vecs)

def PCA_test(X,Eig_Vecs):
    pca1= np.dot(X.values,Eig_Vecs[0])
    pca2=np.dot(X.values,Eig_Vecs[1])
    pca3=np.dot(X.values,Eig_Vecs[2])
    pca4=np.dot(X.values,Eig_Vecs[3])
    pca5=np.dot(X.values,Eig_Vecs[4])
    pca6=np.dot(X.values,Eig_Vecs[5])
    pca7=np.dot(X.values,Eig_Vecs[6])
    pca8=np.dot(X.values,Eig_Vecs[7])
    pca9=np.dot(X.values,Eig_Vecs[8])
    
    pca_test = pd.DataFrame()
    pca_test['pca1']=pca1
    pca_test['pca2']=pca2
    pca_test['pca3']=pca3
    pca_test['pca4']=pca4
    pca_test['pca5']=pca5
    pca_test['pca6']=pca6
    pca_test['pca7']=pca7
    pca_test['pca8']=pca8
    pca_test['pca9']=pca9
    pca_test.head()
    return pca_test
##
def pca_models(random):
    cols = ['Open ', 'High ', 'Low ', 'Close ', 'Unnamed: 6',
            'Volume', 'AAPL', 'AMZN', 'FB', 'GOOGL', 'NFLX', 
            'NVDA', 'TSLA', 'Mkt-RF',
            'SMB', 'HML', 'RF', 'ADS_Index', 'Close_next_day']    
    X0 = X_train[cols]
    X1 = X0.iloc[random]    
    pca_train,Eig_Vecs = PCA_train(X1) 
    Y0 = np.array(y_train)
    yi = Y0[random]  
    
    xi = pca_train.iloc[:,0:2].values
 
    invXX = np.linalg.inv(xi.transpose()@xi)
    # OLS estimates
    beta_hat = invXX@xi.transpose()@yi
    #predicted value of Y using OLS
    y_hat = xi@beta_hat
    residuals = yi - y_hat
    #Calculate R square
    R_Square = 1-(residuals.transpose()@residuals)/(len(yi)*np.var(yi))
    return (R_Square,beta_hat,Eig_Vecs)

def predict(X_test):
   # X_test = np.hstack([np.ones(y_len).reshape(y_len,1),X_test]) 
    X_len = len(X_test)
    ys_predict = []  
    
    for i in _beta_model_ff:
        cols = ['Mkt-RF','SMB','HML','ADS_Index'] 
        X_test0 = np.array(X_test.loc[:,cols])
        X_test1 = np.column_stack([np.ones(X_len),X_test0]) 
        y_hat = X_test1@i
        ys_predict.append(y_hat)

    for j in _beta_model_rat:
        cols = ['RF','ADS_Index', 'TSLA']
        X_test0 = np.array(X_test.loc[:,cols])
        X_test1 = np.column_stack([np.ones(X_len),X_test0]) 
        y_hat = X_test1@j
        ys_predict.append(y_hat)
    
    for z in _beta_model_poly:
        cols = ['RF','GOOGL','ADS_Index','AAPL', 'AMZN', 'FB', 'GOOGL', 'NFLX', 'NVDA', 'TSLA']
        x_temp = X_test[cols]    
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(x_temp)
        X = X_train_poly[:,[0,1,36]] 
        y_hat = X@z
        ys_predict.append(y_hat)
        
    
    for m in range(len(_beta_model_pca)):        
        X_test0 = PCA_test(X_test,_eig_Vecs_pca[m])
        xi = X_test0.iloc[:,0:2].values       
        y_hat = xi@_beta_model[m]
        ys_predict.append(y_hat)

    ys_predict = np.array(ys_predict)
    avg_y_hat = np.mean(ys_predict,axis=0)
   
    return(avg_y_hat)
    

def accuracy_score(y_test,y_hat):
    residuals = y_test - y_hat
    'Variance of residuals'     
    'Calculate R-square'
    R_square = 1- ((residuals.transpose()@residuals)/(len(y_test)*np.var(y_test)))
    return(R_square)

fit(X_train,y_train,1000)
y_hat = predict(X_test)

print('length of test sets: ',len(y_test))
print('R2 Score for test result: ',accuracy_score(predict(X),Y))

## use the lstm method

#print(y_hat[:5])
#print(np.array(y_test[:5]))
plt.plot(predict(X),'r')
plt.plot(np.array(Y))
plt.title('Regression Random Forest Result')
plt.show()

#%%
#%%
## SECOND PART: Getting the LSTM Result


data_to_use = Y.values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))

Rs_lstm,y_pred, y_hat_lstm = lstm.lstm_tf(scaled_data)

y_hat_lstm = np.array(y_hat_lstm)
print('the R_sqaure of lstm is: ',Rs_lstm[0][0])


#%%
## Average the Result of OLS models and lstm
y_hat_avg = np.mean([y_hat.reshape(-1,1),y_hat_lstm.reshape(-1,1)],axis=0)
plt.plot(y_hat_avg,'r')
plt.plot(np.array(y_test))
plt.title('Avg prediction VS True Value of Close price')
plt.show()

#print(np.array(Y.iloc[8:]).shape,np.array(y_hat_avg))
#print('R2 Score for test result: ',accuracy_score(y_test,np.array(y_hat_avg).reshape(-1)))



#%%
# THIRD PART: 
# Trading Policy: Compare the Open price and prediction price
# add the predicted price on the data array for trading profit calculation
X_test['predict_close'] = y_hat_avg

# Because the prediction_price in the format is one day ahead than the other price like Open, so we shift them
Profit_calculation = X_test[['Open ','Close ']][1:]
Profit_calculation['Close_pred']=X_test[['predict_close']].shift(1)

#%%
money = []
for index,row in Profit_calculation.iterrows():
     if row['Close_pred']>row['Open ']:
         money.append(row['Close ']-row['Open '])
print("STRATEGY 1 ")
print('''The Trading Strategy we are using right now is: Compare the Open price and Prediction of Close price. 
      
      If Open < Close:
          
      We purchase it and sell it at the closing price; Otherwise we won't trade.''')
      
print(" ")
###
gross_profit = sum([i for i in money if i>0])
gross_loss = sum([i for i in money if i<0 ])
total_money = float(input('How much money you want to invest in? '))
print(" ")
if total_money<170:
    total_money = float(input('Come on, one Apple share is at least $170, How much you can invest in us? '))
print(" ")
# Calculate the money of 
earned = round(sum(money)*total_money/170,2)
return_rate = round(earned/total_money,2) * 100

print('''Assuming we have $''' ,total_money,''' to invest, based on the approximate price of  $170 per share,
Then We can help you earn a profit of $''',earned,

         '''.\n The return rate is: ''',return_rate,'%  for the past 3 months.'
         )
print(" ")

print("Gross Profit: ", round(gross_profit,2))
print(" ")
print("Gross Loss: ", round(gross_loss,2))
print(" ")
net_profit = earned
trading_days = len(Profit_calculation)
std_close = np.std(money)

sharpe_ratio = (np.mean(money)*365)/(std_close*trading_days)

print('Sharpe Ratio using Strategy 1: ', round(sharpe_ratio,2))
print(" ")
#%%

#Bollinger Band Strategy
print("STRATEGY 2 - BOLLINGER BANDS ")
def bollinger_strat(df,window,no_of_std):
    rolling_mean = df['Close '].rolling(window).mean()
    rolling_std = df['Close '].rolling(window).std()
    
    df['Bollinger Mean'] = rolling_mean
    df['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)
    
    df['Short'] = None
    df['Long'] = None
    df['Position'] = None
    
    for row in range(len(df)):
    
        if (df['Close '].iloc[row] > df['Bollinger High'].iloc[row]) and (df['Close '].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
            df['Position'].iloc[row] = -1
        
        if (df['Close '].iloc[row] < df['Bollinger Low'].iloc[row]) and (df['Close '].iloc[row-1] > df['Bollinger Low'].iloc[row-1]):
            df['Position'].iloc[row] = 1
            
    df['Position'].fillna(method='ffill',inplace=True)
    
    df['Market Return'] = np.log(df['Close '] / df['Close '].shift(1))
    df['Strategy Return'] = df['Market Return'] * df['Position']
  
    #df['Strategy Return'].cumsum().plot()
    #df[['Bollinger Mean','Bollinger High','Bollinger Low','Close_pred']].plot()
    plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
    plt.title('Bollinger Bands on Apple Stocks')
    plt.xlabel('Days')
    plt.ylabel('Stock Prices')
    plt.plot(df['Bollinger Mean'], label= 'Bollinger Mean')
    plt.plot(df['Bollinger High'], label= 'Bollinger High')
    plt.plot(df['Bollinger Low'], label= 'Bollinger Low')
    plt.plot(df['Close_pred'], label= 'Close_pred')
    plt.legend()
    plt.show()
    return df
    
invest = bollinger_strat(Profit_calculation,20,2)


# Calculating Short and Long Positions
investment = invest.iloc[:,[0,1,2,3,4,5]]
investment = investment.dropna()
investment['Signal'] = ''

diff_sell = np.subtract(investment['Close_pred'],investment['Bollinger High'])
diff_sell = np.sign(diff_sell)
ind_list_sell = []
for i in range(0,len(diff_sell)-1):
   if (diff_sell[i] != diff_sell[i+1]):
       ind_list_sell.append(i+1)
       investment.iloc[i+1,6] = 'Sell'

# =============================================================================
# 
# for i in ind_list_sell:
#     print(investment.iloc[i])
# =============================================================================

diff_buy = np.subtract(investment['Close_pred'],investment['Bollinger Low'])
diff_buy = np.sign(diff_buy)
ind_list_buy = []
for i in range(0,len(diff_buy)-1):
   if (diff_buy[i] != diff_buy[i+1]):
       ind_list_buy.append(i+1)
       investment.iloc[i+1,6] = 'Buy'


# =============================================================================
# for i in ind_list_buy:
#     print(investment.iloc[i])
# =============================================================================

#%%

print(" ")  
total_money2 = float(input('How much money you want to invest using Bollinger Bands Strategy? '))
if total_money2<170:
     total_money2 = float(input('Come on, one Apple share is at least $170, How much you can invest in us? '))

print(" ")  
print('Assuming we have initially bought shares at price $170 per share with half of our initial investment')
sold = []
bought = []
returns = []
bought_amt = 0.5*total_money2
no_of_shares = bought_amt/170
rem_amt = total_money2 - bought_amt
money2 = 0

for i in ind_list_sell:
    sell = investment.iloc[i]
    s = 0.30*no_of_shares*(sell['Close_pred'])
    no_of_shares = no_of_shares - 0.30*no_of_shares
    sold.append(s)
    returns.append(sell['Close_pred'] - 170)

for i in ind_list_buy:
    buy = investment.iloc[i]
    shares = 0.40*rem_amt/buy['Close_pred']
    rem_amt = rem_amt - shares*buy['Close_pred']
    tshare = no_of_shares + shares
    bought.append(shares*buy['Close_pred'])

sh = sum(sold)
lg = sum(bought)
price = sell['Close_pred']
share_rem = no_of_shares*sell['Close_pred']
money2 = rem_amt + sh + lg + share_rem
profit2 = round(money2 - total_money2,2)
return_rate2 = round(profit2*100/total_money2,2)   

#trading_days2 = len(investment)
std_close2 = np.std(returns)

sharpe_ratio2 = (np.mean(returns)*365)/(std_close2*trading_days)

print(" ") 
print('''Assuming we have $''' ,total_money2,''' to invest using Bollinger Bands, based on the approximate price of  $170 per share,

Then We can help you earn a profit of $''',round(profit2,2),
'''.\nThe return rate is: ''',return_rate2,'%  for the past 3 months.'
         )
print(" ")
print('Sharpe Ratio using Bollinger Bands: ', round(sharpe_ratio2,2))
#%%

