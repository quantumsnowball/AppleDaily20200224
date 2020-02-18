import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get(ticker, *, start=None, end=None):
    ts = pd.read_csv(f'./assets/{ticker}.csv', index_col=0, parse_dates=True)
    ts = ts.loc[start:end]
    return ts
def get_close(ticker, **kwargs):
    return get(ticker, **kwargs)['Close']

timeline = get('^GSPC')
def sharpe(ts, rf=0.025):
    lndiffs = np.log(ts).diff()
    mu = lndiffs.mean()*255
    sigma = lndiffs.std()*252**.5
    sharpe = (mu-rf)/sigma    
    return (mu, sigma, sharpe, )
def sharpe_of(ticker, start=None, end=None, minlen=0.95, **kwargs):
    ts = get_close(ticker, start=start, end=end)    
    if len(ts)<minlen*len(timeline.loc[start:end]):
        print(f'Not enough data for {ticker}, please use smaller range.')
        return (None, None, None, )
    return sharpe(ts, **kwargs)
def beta(ts, ts_index):
    df = pd.concat([ts, ts_index], axis=1)
    chgs = np.log(df).diff().dropna()
    covm = chgs.cov()
    beta = covm.iloc[0,1]/covm.iloc[1,1]
    return beta
def beta_of(ticker, start=None, end=None, minlen=0.95):
    ts = get_close(ticker, start=start, end=end)
    ts_index = get_close('^GSPC', start=start, end=end)
    if len(ts)<minlen*len(timeline.loc[start:end]):
        print(f'Not enough data for {ticker}, please use smaller range.')
        return None
    return beta(ts, ts_index)

# stock and etf list
etf_lst = pd.read_csv('us_top100_etfs.csv', encoding='utf-16', index_col=0)
stock_lst = pd.read_csv('us_top100_stocks.csv', encoding='utf-8', index_col=0)

# calculate metrics
sr_beta = pd.Series((beta_of(t, start='2015-01-01') for t in etf_lst.index), name='beta', index=etf_lst.index)
df_sharpe = pd.DataFrame((sharpe_of(t, start='2015-01-01') for t in etf_lst.index), columns=('mu','sigma','sharpe',)).set_index(etf_lst.index)
sr_ratio = pd.Series(df_sharpe['sharpe']/sr_beta, name='sharpe/beta')
df = pd.concat((sr_ratio, sr_beta, df_sharpe, etf_lst, ), axis=1)
df['Total Assets ($MM)'] = df['Total Assets ($MM)'].map(lambda x:float(x[1:].replace(',','')))
df = df.sort_values(by='sharpe', ascending=False)
print(df.head(20))

# plot mean-variance chart
fig, ax = plt.subplots(1,1, figsize=(15,10,))
ax.axhline(0, c='k'); ax.axvline(0, c='k')
ax.set_xlabel('standard deviation')
ax.set_ylabel('expected return')
ax.set_title('ETF(red) vs Stock(blue)')
for tickers,color,marker in ((etf_lst.index, 'blue', 'X'),
                            (stock_lst.index, 'red', 'o')):
    for ticker in tickers:
        ts = get_close(ticker, start='2015-01-01')
        chgs = np.log(ts).diff()
        mu = chgs.mean()*252
        sigma = chgs.std()*252**.5
        ax.scatter(sigma, mu, c=color, marker=marker)
plt.show(block=True)