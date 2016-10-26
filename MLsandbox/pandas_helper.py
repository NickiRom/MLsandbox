
import pandas as pd

def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)

def divide(g, num_col, denom_col):
    return df.loc[g.index, num_col].sum()/df.loc[g.index, denom_col].sum().astype(float) 

def wavg(g, weight_col):
    return np.average(g, weights = df.loc[g.index, weight_col])

def pct_won(g):
    return divide(g, 'num_won', 'num_bids')*100
    
def w_avg_bid(g):
    return wavg(g, 'num_bids')

def w_avg_paid(g):
    return wavg(g, 'num_won')

def ctr(g):
    return divide(g, 'clicks', 'imps')*100

def ecpa(g):
    # total revenue divided by # cons
    return divide(g, 'revenue', 'cons')

def revenue(g):
    return df.loc[g.index, 'eCPM'].unique()[0]*df.loc[g.index, 'num_won'].sum()/1000
    
def margin(g):
    # revenue - total spend divided by revenue
    return 100*(df.loc[g.index, 'revenue'].sum() - df.loc[g.index, 'total_spend'].sum())/df.loc[g.index, 'revenue'].sum().astype(float)

def convert_to_minutes(txt):
    if 'day' in txt:
        split = [item for item in txt.split(' ') if item != '']
        start = int(split[0])*24*60
        end = int(split[1])*24*60
        return str(start) + ' ' + str(end) + ' ' + split[2].replace('day', 'minute')
    return txt


