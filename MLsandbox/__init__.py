
import os
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import numpy as np
from sklearn import grid_search
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt


from sklearn.externals import joblib

import glob

def fill_nan_deterministic(df, fill_column, batch_by, dictionary={}):
    # fill_column: column with nan to be filled
    # batch_by: column to batch by, which determines correct value for fill_column
    # when there are NaN values for which other records have the correct value, leave dictionary blank
    # optional parameter dictionary, which should look like {batch_by_value: {'val': fill_value, ...}}
    
    rows_changed = 0
    
    if dictionary:
        for batch, value in dictionary.iteritems():
            num_rows = df.loc[df[batch_by]==batch, fill_column].isnull().sum()
            
            df.loc[df[batch_by]==batch, fill_column] = value['val']
            rows_changed += num_rows
                
        if df.loc[:, fill_column].isnull().sum() > 0:
            print str(df.loc[:, fill_column].isnull().sum()) + " rows remain unfilled."
            missing_list = df.ix[df[fill_column].isnull()==True, :][batch_by].unique()
            print "Please add the following " + batch_by + " batches to input dictionary: "+ ', '.join([str(x) for x in missing_list])

    else:
        fill_dict = {batch: df[df[batch_by]==batch][fill_column].unique() for batch in df[batch_by].unique()}
        for batch, value in fill_dict.iteritems(): 
            value = [x for x in value if str(x) !='nan']
            num_rows = df.loc[df[batch_by]==batch, fill_column].isnull().sum()
            
            if len(value) == 1:
                df.loc[df[batch_by]==batch, fill_column] = value[0]
                rows_changed += num_rows - df.loc[df[batch_by]==batch, fill_column].isnull().sum()
                
            elif len(value) > 1: print batch_by + ': ' + str(batch) + ' has multiple values: ' + str(value)
            else: print fill_column + ' values for ' + batch_by + ': ' + str(batch) + ' cannot be found'
            
    print 'Filling '+ str(fill_column) + ' by ' + str(batch_by) + ': ' + str(rows_changed) + ' rows have been filled.\n'
    return df

def find_nan(df):
    # shows the number of Null values for each column
    ready_features = []
    
    for col in df.columns:
        empty_rows = len(df[df[col].isnull()])
        try: pct_empty = 100*empty_rows/len(df[col])
        except: pct_empty = 'NaN'
        if empty_rows > 0:
            print '\t' + str(col) + ' has ' + str(empty_rows) + ' empty rows ({0}%)'.format(pct_empty)
        else: ready_features.append(col)

    print '\n'
    print 'Complete features: ' + ', '.join(ready_features) + '\n'
    return 

def success_breakdown(df, by_column='', nan_breakdown=False):
    # shows the number of successes and failures by value for a particular column
    df_good = df[df['conversion']==1]
    df_test = df[df['conversion']==0]
    
    if by_column == '':
        print 'good: ' + str(len(df_good)) + ' test: ' + str(len(df_test))
    else: 
        for v in df[by_column].unique():
            print str(v) + '// good: ' + str(len(df_good[df_good[by_column]==v])) + ' test: ' + str(len(df_test[df_test[by_column]==v])) 
            if nan_breakdown:
                find_nan(df[df[by_column]==v])
    print '\n'
    return df_good, df_test



