
# coding: utf-8

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
from collections import defaultdict
from sklearn.externals import joblib
from IPython.display import display, Markdown, Latex 
import glob

class ModelDict(object):
    
    def __init__(self, 
                 prefix,
                 estimator = {}, 
                 params = {},
                 evaluation_metrics = ['f1', 'recall', 'precision', 'f1_weighted'], 
                 model_dict = {},
                 initialization = 'retrieve'):

        self.prefix = prefix
        self.estimator = estimator
        self.params = params
        self.evaluation_metrics = evaluation_metrics
        self.model_dict = model_dict
        self.init = initialization
        
        if self.init == 'retrieve':
            self.model_dict = {}
            self.retrieve_models()
        elif self.init == 'make':
            #self.make_models()
	    pass
        elif (self.init == None) & (len(self.model_dict) > 0):
            pass
        else:
            raise ValueError('initialization takes "retrieve", "make", or None.  None must be accompanied by a model_dict input')

    
    def __getmodels__(self):
        for name, model in self.model_dict.items():
            yield name, model['model']

    def __getmodel__(self, name):
        try:
            assert name in self.model_dict
            return self.model_dict[name]['model']
        except AssertionError:
            raise KeyError("model not in model_dict")
            
    def __setmodel__(self, model, name):
        try:
            assert isinstance(model,ensemble.forest.RandomForestClassifier)
            self.model_dict[name] = {'name': name,                                      'model': model,                                      'files': joblib.dump(model, os.path.dirname(os.path.realpath(__file__)) + '/models/%s.pkl' % name)}
        except TypeError, e:
            raise TypeError("pass in a model of type RandomForestClassifier")
    
    def __getfeatures__(self):
        return (feature for feature in x_test.columns)
    
    def display_models(self):
        for name, model in self.model_dict.items():
            print name + ':\n\t'             + '\n\t'.join([str(key)+': '+str(value) for key, value in model['model'].get_params().items()])             + '\n'
    
    def display_performance(self, x_test, y_test):  
        for name, model in self.model_dict.items():
            print (model['name']).center(60, '-')
            for metric in self.evaluation_metrics:
                scorer = check_scoring(model['model'], scoring=metric)
                print (metric + ': %0.3f' %scorer(model['model'], x_test, y_test))
            
    def make_models(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        
        for metric in self.evaluation_metrics:
            
            model_name = self.prefix + '_%s' %metric
            rf_grid = grid_search.GridSearchCV(self.estimator, param_grid = self.params, scoring=metric, n_jobs=2, cv = 3)                      .fit(self.x, self.y)
            self.model_dict[model_name] = {'name': model_name,                                      'model': rf_grid.best_estimator_,                                      'files': joblib.dump(rf_grid.best_estimator_, 'models/P2A_%s.pkl' % model_name)}
    
    def retrieve_models(self):
        for metric in self.evaluation_metrics:
            model_name = self.prefix + '_' + metric
            model = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/'+model_name + '.pkl')
            files = glob.glob(os.path.dirname(os.path.realpath(__file__)) + '/models/%s*' %model_name)
            self.model_dict[model_name] = {'name': model_name,                                      'model': model,                                      'files': files}
        return self.model_dict
        
from abc import ABCMeta, abstractmethod
from sklearn.metrics.scorer import check_scoring, _BaseScorer


class MlPlot(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, 
                 model_dict, 
                 x_test, 
                 y_test, 
                 evaluation_metrics = ['f1', 'recall', 'precision']):
        
        self.x_test = x_test
        self.y_test = y_test
        self.model_dict = model_dict
        self.model_names = model_dict.keys()
        self.evaluation_metrics = evaluation_metrics
        
        
    
    def display_features(self):
        print '\n'.join([feature for feature in self.__getfeatures__()])
    
    def display_models(self):
        for name, model in self.model_dict.items():
            print name + ':\n\t'             + '\n\t'.join([str(key)+': '+str(value) for key, value in model['model'].get_params().items()])             + '\n'
                    
    def display_performance(self):      
        for name, model in self.model_dict.items():
            print (model['name']).center(60, '-')
            for metric in self.evaluation_metrics:
                scorer = check_scoring(model['model'], scoring=metric)
                print (metric + ': %0.3f' %scorer(model['model'], self.x_test, self.y_test))
    
    def __getmodels__(self):
        for name, model in self.model_dict.items():
            yield name, model['model']

    def __getmodel__(self, name):
        try:
            assert name in self.model_dict
            return self.model_dict[name]['model'].best_estimator_
        except AssertionError:
            raise KeyError("model not in model_dict")
            
    def __setmodel__(self, model, name):
        try:
            assert isinstance(model,ensemble.forest.RandomForestClassifier)
            self.model_dict[name] = {'name': name,                                      'model': model,                                      'files': joblib.dump(model, 'models/P2A_%s.pkl' % name)}
        except TypeError, e:
            raise TypeError("pass in a model of type RandomForestClassifier")
    
    def __getfeatures__(self):
        return (feature for feature in x_test.columns)

    def get_subplot_layout(self):
        self.num_subplots = len(self.model_dict)
        self.plot_rows = np.floor(self.num_subplots**0.5).astype(int)+1
        self.plot_cols = np.ceil(1.*self.num_subplots/self.plot_rows).astype(int)+1
        #return ','.join[str(x) for x in [row, col, [i for i in range(0,9)]]
        return ([row-1, col-1, i] for i in range(0,9))
    
    def plot_feature_imps(self):
        self.get_subplot_layout()
        fig = plt.figure(figsize=(6.*self.plot_rows,6.*self.plot_cols))
        return fig
        #for i, plot in enumerate(plots):
        #  ax = fig.add_subplot(row, col, i)
        #  ax.bar(range(0,len(bar_data)), bar_data)
        #  ticks = plt.xticks([x+0.5 for x in range(0,len(bar_data))], labels, rotation=90)
        #  ax.set_title("i = %d" % i)
        #fig.suptitle("n = %d" % num_subplots)
        #fig.set_tight_layout(True)
        #return fig
    
    
class Feature_importances(MlPlot):
    '''    
    def __init__(self, data_x, data_y):
        self.labels = data_x
        self.data = data_y
        self.plot_type = "bar" 
    '''

    def plot(self):
        pass
    def display(self):
        pass

class prediction_rank(MlPlot):
    
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
    
    def plot(self):
        pass
    
    def display(self):
        pass


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
    df_good = df[df['target']==1]
    df_test = df[df['target']==0]
    
    if by_column == '':
        print 'good: ' + str(len(df_good)) + ' test: ' + str(len(df_test))
    else: 
        for v in df[by_column].unique():
            print str(v) + '// good: ' + str(len(df_good[df_good[by_column]==v])) + ' test: ' + str(len(df_test[df_test[by_column]==v])) 
            if nan_breakdown:
                find_nan(df[df[by_column]==v])
    print '\n'
    return df_good, df_test

def print_features_by_type(features_dict):

    for key, item in features_dict.iteritems():
        display(Markdown('**{0}:** {1}'.format(key, ', '.join(item))))
        
        # commented out the line below to use Markdown instead:
        # print str(key) + ':   ', item, '\n'

def infer_data_types(df):
    #############################
    # using the data types specified by pandas, infer the data type of each feature column
    # df: a pandas dataframe
    #############################
    
    d= df.dtypes.to_dict()
    features_dict = defaultdict(list)

    # currently, the dict is organized with the feature name as the key and data type as the value.
    # Switch keys/values and group by key (data type)
    for key, value in sorted(d.iteritems()):
        features_dict[str(value)].append(str(key))
   
    # change data type names that are non-intuitive to feature data type names 
    features_dict['categorical'] = features_dict.pop('object')
    features_dict['discrete'] = features_dict.pop('int64')
    features_dict['continuous'] = features_dict.pop('float64')

    print_features_by_type(features_dict)

    return features_dict

def specify_ordinal(features_dict, ordinal_features):
    ##############################
    #for the features_dict returned in print_data_types(df), you will sometimes want to separate discrete 
    # features from ordinal features.  since this cannot be inferred directly from the data, you have
    # to specify the features that are to be treated as ordinal.
    #
    # features_dict: dictionary of feature names by their data types. a 'discrete' key must be present.
    # list_of_features: the feature names in features_dict['discrete'] that should be moved to 'ordinal'
    ##############################

    features_dict['ordinal'] = [x for x in features_dict['discrete'] if x in ordinal_features]
    features_dict['discrete'] = [x for x in features_dict['discrete'] if x not in ordinal_features]

    if len([x for x in ordinal_features if (x not in features_dict['discrete']) & (x not in features_dict['ordinal'])]) > 0:
        print "the following features were not found: "
        print  [x for x in ordinal_features if (x not in features_dict['discrete']) & (x not in features_dict['ordinal'])]
    
    print_features_by_type(features_dict)

    return features_dict 


def specify_categorical(features_dict, categorical_features):
    ##############################
    #for the features_dict returned in print_data_types(df), you will sometimes want to separate discrete 
    # features from categorical features.  Since this cannot be inferred directly from the data, you have
    # to specify the features that are to be treated as categorical.
    #
    # features_dict: dictionary of feature names by their data types. A 'discrete' key must be present.
    # list_of_features: the feature names in features_dict['discrete'] that should be moved to 'categorical'
    ##############################

    features_dict['categorical'].extend( [x for x in features_dict['discrete'] if x in categorical_features])
    features_dict['discrete'] = [x for x in features_dict['discrete'] if x not in categorical_features]

    if len([x for x in categorical_features if (x not in features_dict['discrete']) & (x not in features_dict['categorical'])]) > 0:
        print "the following features were not found: "
        print  [x for x in categorical_features if (x not in features_dict['discrete']) & (x not in features_dict['categorical'])]
    
    print_features_by_type(features_dict)

    return features_dict 
