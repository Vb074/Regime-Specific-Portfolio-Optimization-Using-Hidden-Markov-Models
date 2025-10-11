#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vadim Bodnarenco

-Research Question - How do the risk–return characteristics and optimal portfolio weights differ 
between bull and bear markets as identified 
by a Hidden Markov Model of market returns?

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog, minimize
from hmmlearn import hmm

market = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /Total WRLD fund.csv', parse_dates = True, index_col = 'Date', thousands=',')
market = market.sort_index(ascending=True)
market.index = pd.to_datetime(market.index)

r = np.log(market['Close']).diff().dropna()
X = r.values.reshape(-1,1)

bull_mean = r[r > 0].mean()
bear_mean = r[r<0].mean()

bull_var = r[r>0].var()
bear_var = r[r<0].var()


model = hmm.GaussianHMM(n_components = 2, covariance_type = 'diag', n_iter = 1)

model.startprob_ = np.array([0.5,0.5])
model.transmat_ = np.array([[0.98, 0.02],[0.05, 0.95]])
model.means_ = np.array([[bull_mean], [bear_mean]])
model.covars_ = np.array([[bull_var],[bear_var]])

Z = model.predict(X)

means = model.means_.flatten()
bull_state = np.argmax(means)
bear_state = np.argmin(means)

regimes = np.where(Z == bull_state, 'Bull', 'Bear')
market_reg = market.loc[r.index].copy()
market_reg['Regime'] = regimes

bull_series = r.copy()
bear_series = r.copy()

bull_series[market_reg['Regime'] != 'Bull'] = np.nan
bear_series[market_reg['Regime'] != 'Bear'] = np.nan


fig, ax = plt.subplots(figsize = (20,12))

ax.plot(bull_series.index, bull_series, c = 'green', alpha = 0.8, label = 'Bull')
ax.plot(bear_series.index, bear_series, c = 'red', alpha = 0.8, label = 'Bear')
plt.legend()

regime = market_reg['Regime'].values
dates = market_reg.index

in_bear = False
for i in range(len(regime)):
    if not in_bear and regime[i] == 'Bear':
        start = dates[i]
        in_bear = True
    elif in_bear and regime[i] == 'Bull':
        end = dates[i-1]
        print(f'Bear market from: {start.date()} to {end.date()}')
        in_bear = False
if in_bear:
    print(f'Bear market from {start.date()} to {dates[-1].date()}')
    

plt.plot(market['Close'])






#------------------------------------ Portfolio Optimization Next ----------------------------------------



meta = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /Meta.csv', index_col = 'Date', parse_dates=True, thousands=',')
btc = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /BTC index.csv', index_col = 'Date', parse_dates=True, thousands=',')
asml = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /asml.csv', index_col = 'Date', parse_dates=True, thousands=',')
etf = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /Total WRLD fund.csv', index_col = 'Date', parse_dates=True, thousands=',')
toyota = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /Toyota Motor.csv', index_col = 'Date', parse_dates=True, thousands=',')
bond = pd.read_csv('/Users/vadimbodnarenko/Desktop/Market State Project /US Gov Bond, 1-3y.csv', index_col = 'Date', parse_dates=True, thousands=',')


meta = meta.sort_index(ascending=True)
btc = btc.sort_index(ascending=True)
asml = asml.sort_index(ascending=True)
etf = etf.sort_index(ascending=True)
toyota = toyota.sort_index(ascending=True)
bond = bond.sort_index(ascending=True)

meta.index = pd.to_datetime(meta.index)
btc.index = pd.to_datetime(btc.index)
asml.index = pd.to_datetime(asml.index)
etf.index = pd.to_datetime(etf.index)
toyota.index = pd.to_datetime(toyota.index)
bond.index = pd.to_datetime(bond.index)

total_df = pd.concat(
    [meta['Close'], btc['Close'], asml['Close'], etf['Close'], toyota['Close'], bond['Close']], axis = 1, join = 'inner')

col_names = ['Meta', 'btc', 'asml', 'etf', 'toyota', 'bond']
total_df.columns = col_names

returns = pd.DataFrame(index=total_df.index[1:])
    
returns = total_df[:].pct_change().dropna()  #Specify the date range in [] if needed, e.g. for when a bear market starts/ends
    
mean_return = returns[:].mean()    #Specify the date range in [] if needed, e.g. for when a bear market starts/ends
cov = returns[:].cov()    #Specify the date range in [] if needed, e.g. for when a bear market starts/ends
cov_np = cov.to_numpy()

#252 Annual trading days
mean_return = returns.mean() * 252
cov = returns.cov() * 252
cov_np = cov.to_numpy()
D = len(mean_return)

position = int(input('Type 1 to allow short selling, for long only, type 2: '))


#Monte Carlo simulation function for 100000 portfolios, allowing short positions 
def Monte_Carlo_Long(mean_return, cov_np, N=1000000):
    all_returns = np.zeros(N)
    all_risks = np.zeros(N)
    weights = np.zeros((N, D))
    
    for i in range(N):
        w = np.random.random(D)
        w = w / w.sum()
        np.random.shuffle(w)
        
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov_np).dot(w))
        
        all_returns[i] = ret
        all_risks[i] = risk
        weights[i, :] = w
    
    return all_risks, all_returns, weights

def Monte_Carlo_Short(mean_return, cov_np, N=1000000):
    all_returns = np.zeros(N)
    all_risks = np.zeros(N)
    weights = np.zeros((N, D))
    
    for i in range(N):
        rand_range = 1.0
        w = np.random.random(D)*rand_range - rand_range/2
        w[-1] = 1 - w[:-1].sum()
        np.random.shuffle(w)
        
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov_np).dot(w))
        
        all_returns[i] = ret
        all_risks[i] = risk
        weights[i, :] = w
    
    return all_risks, all_returns, weights 

if position == 1:
    risks, rets, weights = Monte_Carlo_Short(mean_return, cov_np, N=200000)
    bounds = [(-0.5, None)] * D
else:
    risks, rets, weights = Monte_Carlo_Long(mean_return, cov_np, N=200000)
    bounds = [(0, 1)] * D

#Plotting  

bull_bear_periods = [('2024-09-10', '2024-12-08', 'Bull'), ('2024-12-09', '2025-01-13', 'Bear')]

#Optimizing for max return (- sign as linprog minimizes by default)
from scipy.optimize import linprog

A_eq = np.ones((1,D))
b_eq = np.ones(1)


#if position ==1:
 #   bounds = [(-0.5, None)] * D
#elif position == 2:
#    bounds  = [(0, 1)] * D

res = linprog(-mean_return.values, A_eq=A_eq, b_eq=b_eq, bounds = bounds)

#Maximum return:
#print(res)
#print('Maximum Return in %: ', -res.fun * 100)
max_return = -res.fun

res = linprog(mean_return.values, A_eq=A_eq, b_eq=b_eq, bounds = bounds)
#Min return 
min_return = res.fun


#Plotting an efficient frontier

target_returns = np.linspace(min_return, max_return, 100)
from scipy.optimize import minimize

def get_portfolio_variance(weights):
    return weights.dot(cov.dot(weights))

def target_return_constraint(weights, target):
    return weights.dot(mean_return) - target

def portfolio_constraint(weights):
    return weights.sum() - 1

#Defining a constraint dictionary

constraints = [
    {
     'type': 'eq',
     'fun': target_return_constraint,
     'args': [target_returns[0]],
     },
    {'type': 'eq',
     'fun': portfolio_constraint
     }
    ]

res = minimize(
    fun = get_portfolio_variance,
    x0 = np.ones(D)/D,
    method = 'SLSQP',
    constraints = constraints,
    bounds = bounds,
    )

#Running through all the target returns, to generate an efficient frontier

optimized_risks = []
for target in target_returns:
    constraints[0]['args'] = [target] #Each iteration enforces a different required return
    #Pasting a minimizing fucntion from before
    res = minimize(
        fun = get_portfolio_variance,
        x0 = np.ones(D)/D,
        method = 'SLSQP',
        constraints = constraints,
        bounds = bounds,
    )
    if res.success:
        optimized_risks.append(np.sqrt(res.fun))


risk_free_rate = 0.01925

#for maximization 
def neg_sharpe_ratio(weights): 
    mean = weights.dot(mean_return)
    sd = np.sqrt(weights.dot(cov).dot(weights))
    return -(mean-risk_free_rate)/sd

#Minimize function (to optimize SR, since it is negative)
res = minimize(
    fun = neg_sharpe_ratio,
    x0 = np.ones(D)/D,
    method = 'SLSQP',
    constraints = {
        'type': 'eq',
        'fun': portfolio_constraint
        },
    bounds = bounds
    )

best_sr, best_w = -res.fun, res.x


opt_risk = np.sqrt(best_w.dot(cov).dot(best_w))
opt_ret = mean_return.dot(best_w)


fig, ax = plt.subplots(figsize=(12,5))
ax.scatter(risks, rets, alpha=0.2, label='Monte Carlo')
ax.plot(optimized_risks, target_returns[:len(optimized_risks)], c='black', lw=2.5, label='Efficient Frontier') #Min. risk for each return level - efficient frontier
ax.scatter([opt_risk], [opt_ret], c='r', s=100, marker='*', label='Max Sharpe') # Max shapre ratio portfolio
ax.plot([0, opt_risk], [risk_free_rate, opt_ret], c='r', lw=1) #Capital Market line
plt.legend()

for name, w in zip(col_names, best_w):
    print(f"{name:>8}: {w:.4f}")
print(f"\nSum of weights: {best_w.sum():.4f}")
print(f"Max Sharpe: {(-res.fun):.3f}")
    




