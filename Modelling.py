#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import pickle
import swifter
import statsmodels.api as sm
sns.set_style('darkgrid')


from IPython.display import Audio
from scipy.stats import mode
from sklearn import linear_model
from sklearn import feature_selection
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

sound_file = 'Horn.wav'


# # 1. Methods

# In[2]:


def store_to_pickle(filename, data):
    '''store data to .pickle file'''
    file = open(f'{filename}.pickle', 'wb')
    pickle.dump(data, file)
    file.close()


# ## 1.1. Transformations

# In[3]:


def depreciate_patent(patent, rate):
    '''depreciate a patent by the given rate'''
    depreciated = pd.Series(index = range(len(patent)))

    for i in range(len(patent)):   
        if i == 0:
            depreciated[i] = patent.iloc[i]    ##Start with the non-depreciated value of the first year
        else:
            depreciated[i] = patent.iloc[i] + depreciated[i-1] * (1 - rate)  ##Depreciate the cumulative value from the year before and add the current year's value. 

    return depreciated.astype(float)


# In[4]:


def lag_and_dep(data):
    '''apply time lag and depreciation to the given indices of a data set.
    Methods works with swifter.apply() for faster execution'''

    df = data['data']
    
    for i in data['index']:
        temp = depreciate_patent(df[i], data['dep'])
        df[i] = temp.shift(int(data['lag']))
        

    return {'df': df, 'lag': data['lag'], 'dep':data['dep']}


# In[5]:


def create_tuning_data(data, price_col, lag_index, max_lag, dep_rates):
    '''create tuning data as DataFrame. This consists of the lagged and depreciated X and y series for each combination of lag and depreciation rate.
    data: The data to be transformed.
    price_col: The column that contains the price information to be used as y variable.
    lag_index: indicates the columns that should be lagged and depreciated.
    max_lag: maximum time lag to be applied.
    dep_rates: Depreciation rates to be applied.
    '''
    
    df = pd.DataFrame()
    full_set = pd.DataFrame(columns=['X', 'y', 'lag', 'dep'])
    lagged = []
    
    for lag in range(max_lag):
        for rate in pd.Series(dep_rates):
            df = df.append({'data': data.copy(), 'lag': lag+1, 'dep': rate, 'index': lag_index}, ignore_index = True)
        
    for i in range(df.shape[0]):
        df_log = lag_and_dep(df.iloc[i, :])
        df_log['df'] = np.log(df_log['df'].iloc[:, 1:]).replace(np.NINF, np.nan)
        df_log['df'] = df_log['df'].loc[df_log['df'][price_col] > 0].reset_index().drop(['index'], axis = 1).replace(np.nan, 0)
        lagged.append(df_log)
    
    for i, item in enumerate(lagged):
        full_set = full_set.append({'X': item['df'].drop(columns = price_col), 'y': item['df'].loc[:, price_col],
                                    'lag': item['lag'], 'dep': item['dep']}, ignore_index = True)
        
    return full_set


def create_model_data(data, price_col, lag_index, lag, dep_rate):
    '''transform given data for later use in model building. Returns only one data set, compared to create_tuning_data().
    data: The data to be transformed.
    price_col: The column that contains the price information to be used as y variable.
    lag_index: indicates the columns that should be lagged and depreciated.
    lag: lag to be applied.
    dep_rate: Depreciation rate to be applied.'''
    
    lag_params = {'data': data.copy(), 'lag': lag, 'index': lag_index, 'dep': dep_rate}
    
    df = lag_and_dep(lag_params)['df']

    df_log = np.log(df.iloc[: , 1:]).replace(np.NINF, np.nan)
    df_log = df_log.loc[df_log[price_col] > 0].reset_index().drop(['index'], axis = 1).replace(np.nan, 0)

    X = df_log.drop(columns = price_col)
    y = df_log.loc[:, price_col]
    
    return X, y


# ## 1.2. Tuning and building the model + bootstrap

# In[6]:


## For faster execution of the respective algorithm. Uses swifter.apply().
def multiprocess_algorithm(alg, n, X, y, params):
    
    df = pd.DataFrame({'X': [X]*n, 'y': [y]*n, 'params': [params]*n})
    result = df.swifter.apply(alg, axis = 1)    
        
    return result

def multiprocess_algorithm_no_params(alg, n, X, y):

    df = pd.DataFrame({'X': [X]*n, 'y': [y]*n})
    result = df.swifter.apply(alg, axis = 1)
        
    return result


# In[7]:


def tune_regression(data):
    lag = data['lag']
    dep = data['dep']
    
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size = 0.2)
    
    ols = sm.OLS(y_train, sm.add_constant(X_train, has_constant = 'add')).fit()
    score = mean_absolute_error(y_test, ols.predict(sm.add_constant(X_test, has_constant = 'add')))

    return score, lag, dep
    

def bootstrap_linear_regression(data, suppress = True):
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2)
        
    ols = sm.OLS(y_train, sm.add_constant(X_train, has_constant = 'add')).fit()
    y_pred = ols.predict(sm.add_constant(X_test, has_constant = 'add'))
    mape = mean_absolute_percentage_error(y_test, y_pred)
        
    return pd.Series([mape, ols, X_train, X_test, y_train, y_test])


# In[8]:


def tune_forest(data):
    lag = data['lag']
    dep = data['dep']
    rfr = RandomForestRegressor()
        
    param_grid = {
            'n_estimators' : [100, 200, 300],
            'max_features' : ['auto', 'sqrt', 'log2'],
            'max_depth' : [2,3,4],
            'criterion' : ['mse', 'mae']
            }
    
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size = 0.2)
       
    GSCV = GridSearchCV(estimator=rfr, param_grid=param_grid, cv = np.min([5, len(X_train)]), n_jobs=-1, error_score = 'raise')
    GSCV.fit(X_train, y_train)

    return GSCV.cv_results_, GSCV.best_params_, lag, dep


def bootstrap_forest(data, suppress = True):
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2)
    rfr_params = data[2]
    
    ## Build    
    rfr = RandomForestRegressor(criterion = rfr_params['criterion'],
                        max_depth=rfr_params['max_depth'], 
                        max_features=rfr_params['max_features'],
                        n_estimators= rfr_params['n_estimators'])
        
    rfr.fit(X_train, y_train)
        
    y_pred = rfr.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
        
    if not suppress:
        print('MAPE: ', np.mean(mape))
    
    return pd.Series([mape, rfr, X_train, X_test, y_train, y_test])


# In[9]:


def tune_adaboost(data):
    lag = data['lag']
    dep = data['dep']
    
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size = 0.2)
    
    ada = AdaBoostRegressor()
        
    param_grid = {
                'n_estimators' : [100, 200, 300],
                'learning_rate' : [0.9, 0.95, 1],
                'loss' : ['linear', 'square', 'exponential'],
            }
        
    GSCV = GridSearchCV(estimator=ada, param_grid=param_grid, cv = np.min([5, len(X_train)]), n_jobs=-1, error_score = 'raise')
    GSCV.fit(X_train, y_train) 

    return GSCV.cv_results_, GSCV.best_params_, lag, dep


def bootstrap_adaboost(data, suppress = True):

    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2)
    ada_params = data[2]

    ## Build    
    ada = AdaBoostRegressor(
                        learning_rate = ada_params['learning_rate'],
                        loss = ada_params['loss'],
                        n_estimators= ada_params['n_estimators'])
   
    ada.fit(X_train, y_train)
    
    y_pred = ada.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    if not suppress:
        print('MAPE: ', np.mean(mape))

    return pd.Series([mape, ada, X_train, X_test, y_train, y_test])


# In[10]:


def tune_gradientboost(data):
    lag = data['lag']
    dep = data['dep']
    
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size = 0.2)
    
    boost = GradientBoostingRegressor()

    param_grid_boost = {
                'n_estimators' : [100, 200, 300],
                'loss' : ['ls', 'lad', 'huber'],
                'criterion' : ['mse', 'friedman_mse'],
                'max_depth' : [2,3,4],
                'max_features' : ['auto', 'sqrt', 'log2'],
                }

    GSCV_boost = GridSearchCV(estimator=boost, param_grid=param_grid_boost, cv = np.min([5, len(X_train)]), n_jobs=-1, error_score = 'raise')
    GSCV_boost.fit(X_train, y_train)

    return GSCV_boost.cv_results_, GSCV_boost.best_params_, lag, dep 


def bootstrap_gradientboost(data, suppress = True):
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2)
    boost_params = data[2]
    
    ## Build      
    reg = GradientBoostingRegressor(criterion = boost_params['criterion'],
                                n_estimators=boost_params['n_estimators'],
                                loss=boost_params['loss'],
                                max_depth=boost_params['max_depth'], 
                                max_features=boost_params['max_features'])
        
    reg.fit(X_train, np.ravel(y_train))
        
    y_pred = reg.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    if not suppress:
        print('MAPE: ', np.mean(mape))
        
    return pd.Series([mape, reg, X_train, X_test, y_train, y_test])


# In[11]:


def tune_models(data, price_col, lag_index, max_lag, dep_rates, suffix):
    '''
    Creates an input dataset for and hands it over to all remote tuning methods.
    Makes the results readable using ray.get() and returns readable data.
    Creates a pickle of each completed get() operation for safety.
    '''
    
    start = time.time()
    df = create_tuning_data(data, price_col, lag_index, max_lag, dep_rates)
    
    results_glm = df.swifter.apply(tune_regression, axis = 1)
    store_to_pickle(f'tuning_results/readable_glm_{suffix}', results_glm)
    print(f'GLM is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
    
    results_ada = df.swifter.apply(tune_adaboost, axis = 1)
    store_to_pickle(f'tuning_results/readable_ada_{suffix}', results_ada)
    print(f'ADA is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
            
    results_rfr = df.swifter.apply(tune_forest, axis = 1)
    store_to_pickle(f'tuning_results/readable_rfr_{suffix}', results_rfr)
    print(f'RFR is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
    
    results_grad = df.swifter.apply(tune_gradientboost, axis = 1)
    store_to_pickle(f'tuning_results/readable_grad_{suffix}', results_grad)
    print(f'GRAD is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
     
    return results_glm, results_rfr, results_ada, results_grad


# In[12]:


def start_algs(data, price_col, lag_index, lags, dep_rates, params):
    '''
    data: Der Datensatz, der verwendet werden soll
    price_col: Name der Spalte, die den Preis enthält (= y-Variable)
    lag_index: Name der Spalten, die an die lag-Funktion übergeben werden
    max_lag: maximaler Lag in Jahren
    dep_rate: Linspace mit möglichen depreciation rates
    params: Parameter array for random forest, adaboost and gradientboost at index 0,1,2.
    '''
    
    ## Create parameters for multiprocessing
    start = time.time()
    n = 1000
    
    ## Start multiprocessing for each algorithm with optimal dataset
    X, y = create_model_data(data, price_col, lag_index, lags[0], dep_rates[0])
    glm = multiprocess_algorithm_no_params(bootstrap_linear_regression, n, X, y)
    print(f'GLM is done! Time elapsed: {(time.time()-start)} sec')

    X, y = create_model_data(data, price_col, lag_index, lags[2], dep_rates[2])
    ada = multiprocess_algorithm(bootstrap_adaboost, n, X, y, params[2])
    print(f'ADA is readable! Time elapsed: {(time.time()-start)} sec')
    
    X, y = create_model_data(data, price_col, lag_index, lags[1], dep_rates[1])
    rfr = multiprocess_algorithm(bootstrap_forest, n, X, y, params[1])
    print(f'RFR is readable! Time elapsed: {(time.time()-start)} sec')
    
    X, y = create_model_data(data, price_col, lag_index, lags[3], dep_rates[3])
    grad = multiprocess_algorithm(bootstrap_gradientboost, n, X, y, params[3])
    print(f'GRAD is readable! Time elapsed: {(time.time()-start)} sec')
    
    return glm, rfr, ada, grad


# In[13]:


## Methods to get the best parameters from the tuning results. Separate methods needed for regression and machine learning algorithms.

def get_best_params_regression(readable):
    best_score = 1
    best_params_position = -1
    
    for i in range(len(readable)):
        if readable[i][0] < best_score:
            best_score = np.round(readable[i][0], 4)
            best_params_position = i
       
        
    lag = int(readable[best_params_position][1])
    dep = readable[best_params_position][2]
            
    return best_score, lag, dep

def get_best_params(readable):
    best_score = 0
    best_params_position = -1

    for i in range(len(readable)):
        best_score_position = np.where(readable[i][0]['rank_test_score'] == 1)
        mean_score = np.nanmean([
                        readable[i][0]['split0_test_score'][best_score_position],
                        readable[i][0]['split1_test_score'][best_score_position],
                        readable[i][0]['split2_test_score'][best_score_position],
                        readable[i][0]['split3_test_score'][best_score_position],
                        readable[i][0]['split4_test_score'][best_score_position]
                        ])
        
        if mean_score > best_score:
            best_score = np.round(mean_score, 4)
            best_params_position = i
        
    best_params = readable[best_params_position][1]
    lag = int(readable[best_params_position][2])
    dep = readable[best_params_position][3]
            
    meta = pd.DataFrame({'position': best_params_position, 'score': best_score}, index = [0])

    return meta, lag, dep, best_params


# In[14]:


## get subsets of each data set. These will be used in model construction

data_fc = pd.read_excel('Sources/dataframes.xlsx', 'fuel_cell')
data_fc.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_fc = data_fc.sort_values(by = 'Year')
fc_subset = data_fc.loc[(data_fc['Year'] > 1982) & (data_fc['Year'] <= 2020)].reset_index().drop('index', axis = 1)

data_ev = pd.read_excel('Sources/dataframes.xlsx', 'li_ev')
data_ev.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_ev = data_ev.sort_values(by = 'Year')
ev_subset = data_ev.loc[(data_ev['Year'] > 1982) & (data_ev['Year'] <= 2020)].reset_index().drop('index', axis = 1)

data_res = pd.read_excel('Sources/dataframes.xlsx', 'li_res')
data_res.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_res = data_res.sort_values(by = 'Year')
res_subset = data_res.loc[(data_res['Year'] > 1982) & (data_res['Year'] <= 2020)].reset_index().drop('index', axis = 1)


# # Full run-down
# ## Fuel cell dataset
# 

# In[15]:


## Optional, loads tuning results for later use
results_fc = []
results_fc.append(pickle.load(open('tuning_results/readable_glm_fc.pickle', 'rb')))
results_fc.append(pickle.load(open('tuning_results/readable_rfr_fc.pickle', 'rb')))
results_fc.append(pickle.load(open('tuning_results/readable_ada_fc.pickle', 'rb')))
results_fc.append(pickle.load(open('tuning_results/readable_grad_fc.pickle', 'rb')))


# In[ ]:


warnings.filterwarnings(action = 'ignore')
results_fc = tune_models(fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells'],
                                        5, np.linspace(0, 0.2, 21), 'fc')

param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results_fc)):
    if i > 0:
        temp = get_best_params(results_fc[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results_fc[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
fc_final = start_algs(fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells'], param_info['lag'], param_info['dep'], param_info['params'])
fc_final_pat_only = start_algs(fc_subset.drop(columns = ['Hydrogen and fuel cells']), 'USD2020/kW', ['Fuel_cell_h01m8'], param_info['lag'], param_info['dep'], param_info['params'])

store_to_pickle('results/fc_final', fc_final)
store_to_pickle('results/fc_final_pat_only', fc_final_pat_only)


# ## Li-ion: res

# In[16]:


results_res = []
results_res.append(pickle.load(open('tuning_results/readable_glm_res.pickle', 'rb')))
results_res.append(pickle.load(open('tuning_results/readable_rfr_res.pickle', 'rb')))
results_res.append(pickle.load(open('tuning_results/readable_ada_res.pickle', 'rb')))
results_res.append(pickle.load(open('tuning_results/readable_grad_res.pickle', 'rb')))


# In[ ]:


warnings.filterwarnings(action = 'ignore')
results_res = tune_models(res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage'],
                                        5, np.linspace(0, 0.2, 21), 'res')


param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results_res)):
    if i > 0:
        temp = get_best_params(results_res[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results_res[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
res_final = start_algs(res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage'], param_info['lag'], param_info['dep'], param_info['params'])
res_final_pat_only = start_algs(res_subset.drop(columns = 'Energy storage'), 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10'], param_info['lag'], param_info['dep'], param_info['params'])


store_to_pickle('results/res_final', res_final)
store_to_pickle('results/res_final_pat_only', res_final_pat_only)


# ## Li-ion: ev

# In[17]:


results_ev = []
results_ev.append(pickle.load(open('tuning_results/readable_glm_ev.pickle', 'rb')))
results_ev.append(pickle.load(open('tuning_results/readable_rfr_ev.pickle', 'rb')))
results_ev.append(pickle.load(open('tuning_results/readable_ada_ev.pickle', 'rb')))
results_ev.append(pickle.load(open('tuning_results/readable_grad_ev.pickle', 'rb')))


# In[ ]:


warnings.filterwarnings(action = 'ignore')
results_ev = tune_models(ev_subset, 'USD2020/kWh', ['Li_ev_Y02T90', 'EV aggregated'],
                                        5, np.linspace(0, 0.2, 21), 'ev')


param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results_ev)):
    if i > 0:
        temp = get_best_params(results_ev[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results_ev[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
ev_final = start_algs(ev_subset, 'USD2020/kWh', ['Li_ev_Y02T90', 'EV aggregated'], param_info['lag'], param_info['dep'], param_info['params'])
ev_final_pat_only = start_algs(ev_subset.drop(columns = 'EV aggregated'), 'USD2020/kWh', ['Li_ev_Y02T90'], param_info['lag'], param_info['dep'], param_info['params'])

store_to_pickle('results/ev_final', ev_final)
store_to_pickle('results/ev_final_pat_only', ev_final_pat_only)


# ## Learning rates for ensemble algs
# Use the best tuning params to apply lag and depreciation to the base dataset.\
# We already know the real price in the first year, as well as the installed volume in the first and last year of the dataset. By predicting what the first-year-price would be if the installed volume would be the same as in the last year, the learning rate can be calculated by rearranging the learning rate formula:
# $$ Y = aX^b$$
# $$log(Y) = log(a) + log(X) * b$$
# $$b = \frac{log(Y)-log(a)}{log(X)}$$
# 
# where $b$ or $\beta$ is the learning rate, $a$ the initial price, $X$ the number of doublings of cumulative volume and $Y$ the price in the final year. Note that the data used is already on a log-scale at the time of prediction.

# In[18]:


def beta(Y, a, X):
    '''variables already on log scale! Returns beta'''
    return list(map(lambda Y, a, X: (Y-a)/X, Y,a,X))


# In[19]:


def get_betas(prediction):
    lr = []
    for alg in prediction:
        lr.append(beta(alg['Y_hat'], alg['a'], alg['X']))
    return lr  


# In[20]:


def find_first_non_zero(df, col_index, start = 'top'):
        
    for i in range(df.shape[0]):
        if start == 'bottom':
            pos = df.shape[0] - (i+1)
        else:
            pos = i        
        
        if df.iloc[pos, col_index] != 0:
            return pos
    


# In[21]:


def get_predictions_for_lr(final, results, volume_index, subset, price_col, lag_col):
    '''Acquire predictions for each machine learning algorithm for later learning rate calculation.
    final: the final result of the models
    results: tuning results
    base_row, target_row, volume_index: row with first (last) non-zero occurance of the cum. volume found in col with volume_index
    subset, price_col, lag_col: the dataset to be used and the columns that should be lagged or contain the price (y-variable)
    '''
    modeldata = []

    ## Re-create the appropriate model for all machine learning algorithms, NOT linear regression. We simply use its coefficients for learning rates
    for alg in results[1:]:
        temp = get_best_params(alg)
        modeldata.append(create_model_data(subset, price_col, lag_col, temp[1], temp[2]))


    ## Use the base-year row with one updated value to predict the price change dependent on the respective variable.
    learn_pred = []
    for i, alg in enumerate(final[1:]):        
        pred = pd.DataFrame()
        df = modeldata[i][0].copy()
        base_row = find_first_non_zero(df, volume_index)
        target_row = find_first_non_zero(df, volume_index, 'bottom')
        
        base_volume = df.iloc[base_row, volume_index]
        df.iloc[base_row, volume_index] = df.iloc[target_row, volume_index]
            
        row = pd.DataFrame(df.iloc[base_row])
    
        for est in alg[1]:
            pred = pred.append({'Y_hat': est.predict(row.T), 'a': modeldata[i][1].iloc[base_row], 
                                'Q_base': base_volume, 'Q_final': df.iloc[target_row, volume_index]},
                              ignore_index = True)
            
        ## calculate X for later use in learning rate
        pred['X'] = np.log(np.e**pred['Q_final'] / (np.e**pred['Q_base']))
        learn_pred.append(pred)
        
    return learn_pred


# ### Get predictions for each dataset

# In[ ]:


print('Fuel cell')
fc_beta = dict()
fc_beta['MW_dom'] = get_betas(get_predictions_for_lr(fc_final, results_fc, 0, fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells']))
fc_beta['Fuel_cell_h01m8'] = get_betas(get_predictions_for_lr(fc_final, results_fc, 1, fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells']))
fc_beta['Hydrogen and fuel cells'] = get_betas(get_predictions_for_lr(fc_final, results_fc, 3, fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells']))

print('RES')
res_beta = dict()
res_beta['MWh_combined'] = get_betas(get_predictions_for_lr(res_final, results_res, 3, res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage']))
res_beta['Li_ion_h01m10'] = get_betas(get_predictions_for_lr(res_final, results_res, 0, res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage']))
res_beta['Energy storage'] = get_betas(get_predictions_for_lr(res_final, results_res, 2, res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage']))

print('EV')
ev_beta = dict()
ev_beta['MWh_ev'] = get_betas(get_predictions_for_lr(ev_final, results_ev, 0, ev_subset, 'USD2020/kWh',['Li_ev_Y02T90', 'EV aggregated']))
ev_beta['Li_ev_Y02T90'] = get_betas(get_predictions_for_lr(ev_final, results_ev, 1, ev_subset, 'USD2020/kWh',['Li_ev_Y02T90', 'EV aggregated']))
ev_beta['EV aggregated'] = get_betas(get_predictions_for_lr(ev_final, results_ev, 4, ev_subset, 'USD2020/kWh',['Li_ev_Y02T90', 'EV aggregated']))


# In[ ]:


fc_beta_pat_only = dict()
fc_beta_pat_only['MW_dom'] = get_betas(get_predictions_for_lr(fc_final_pat_only, results_fc, 0, fc_subset.drop(columns = 'Hydrogen and fuel cells'), 'USD2020/kW', ['Fuel_cell_h01m8']))
fc_beta_pat_only['Fuel_cell_h01m8'] = get_betas(get_predictions_for_lr(fc_final_pat_only, results_fc, 1, fc_subset.drop(columns = 'Hydrogen and fuel cells'), 'USD2020/kW', ['Fuel_cell_h01m8']))

res_beta_pat_only = dict()
res_beta_pat_only['MWh_combined'] = get_betas(get_predictions_for_lr(res_final_pat_only, results_res, 2, res_subset.drop(columns = 'Energy storage'), 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10']))
res_beta_pat_only['Li_ion_h01m10'] = get_betas(get_predictions_for_lr(res_final_pat_only, results_res, 0, res_subset.drop(columns = 'Energy storage'), 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10']))

ev_beta_pat_only = dict()
ev_beta_pat_only['MWh_ev'] = get_betas(get_predictions_for_lr(ev_final_pat_only, results_ev, 0, ev_subset.drop(columns= 'EV aggregated'), 'USD2020/kWh',['Li_ev_Y02T90']))
ev_beta_pat_only['Li_ev_Y02T90'] = get_betas(get_predictions_for_lr(ev_final_pat_only, results_ev, 1, ev_subset.drop(columns= 'EV aggregated'), 'USD2020/kWh',['Li_ev_Y02T90']))


# In[ ]:


lrs = [ev_beta, res_beta, fc_beta, ev_beta_pat_only, res_beta_pat_only, fc_beta_pat_only]
store_to_pickle('results/learning_betas', lrs)


# ### Tuning results

# In[ ]:


sets = ['FC', 'EV', 'RES']
algs = ['GLM', 'RFR', 'ADA', 'GRAD']

## Iterate through data sets and algorithms and print the best params determined through tuning
for i, arr in enumerate([results_fc, results_ev, results_res]):
    for j, alg in enumerate(arr):
        if j == 0:
            print('Set', sets[i], 'Alg', algs[j], ':', get_best_params_regression(alg)[1:])
        else:
            print('Set', sets[i], 'Alg', algs[j], ':', get_best_params(alg)[1:])
    print('-'*140)


# In[ ]:




