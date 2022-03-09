from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_close.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace = True)
     
    for i in range(1, hm_days+1):
        # move future data to beside the current one
        df[f'{ticker}_{i}d'] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0,inplace = True)
    
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.025
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    df[f'{ticker}_target'] = list(map(buy_sell_hold,
                                      df[f'{ticker}_1d'],
                                      df[f'{ticker}_2d'],
                                      df[f'{ticker}_3d'],
                                      df[f'{ticker}_4d'],
                                      df[f'{ticker}_5d'],
                                      df[f'{ticker}_6d'],
                                      df[f'{ticker}_7d']
                                      ))        
    vals = df[f'{ticker}_target'].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0,inplace=True)

    """make sure the infinite changes (from 0 to a number)
    since we replace the NAN to 0 above"""
    df = df.replace([np.inf,-np.inf], np.nan)
    df.dropna(inplace = True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0,inplace=True)

    X = df_vals.values
    y = df[f"{ticker}_target"].values
    
    return X, y, df

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
    
    # clf = KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', LinearSVC()),
                            ('knm', KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))
    
    return confidence

do_ml('AAPL')