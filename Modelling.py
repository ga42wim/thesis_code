#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import pickle
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
    file = open(f'{filename}.pickle', 'wb')
    pickle.dump(data, file)
    file.close()


# ## 1.1. Transformations

# In[3]:


def depreciate_patent(patent, rate):
    colname = patent.columns[0] + '_depreciated'
    depreciated = pd.DataFrame(columns = [colname], index = range(patent.shape[0]))
    
    for i in range(patent.shape[0]):
        if i == 0:
            depreciated[colname][i] = patent.iloc[i]
        else:
            depreciated[colname][i] = patent.iloc[i] + depreciated[colname][i-1] * (1 - rate)

    return depreciated.astype(float)


# In[4]:


def lag_and_dep(data, index, lag, dep_rate):
    df = data.copy()
    for i in index:
        temp = depreciate_patent(pd.DataFrame(df[i]), dep_rate).shift(int(lag))
        df[i] = temp   

    return df


# In[5]:


def create_tuning_data(data, price_col, lag_index, max_lag, dep_rates):
    
    df = data.copy()
    full_set = pd.DataFrame(columns=['X', 'y', 'lag', 'dep'])
    
    for lag in range(max_lag):
        for rate in pd.Series(dep_rates):
            df = lag_and_dep(df, lag_index, lag+1, rate)

            df_log = np.log(df.iloc[: , 1:].fillna(0)).replace(np.NINF, np.nan)

            df_log = df_log[df_log[price_col] > 0].reset_index().drop(['index'], axis = 1).fillna(0)

            full_set = full_set.append({'X': df_log.drop(columns = price_col), 'y': df_log.loc[:, price_col],
                          'lag': lag+1, 'dep': rate}, ignore_index=True)
            
    return full_set

def create_model_data(data, price_col, lag_index, lag, dep_rate):
    
    df = data.copy()
    
    df = lag_and_dep(df, lag_index, lag, dep_rate)

    df_log = np.log(df.iloc[: , 1:].fillna(0)).replace(np.NINF, np.nan)
    df_log = df_log[df_log[price_col] > 0].reset_index().drop(['index'], axis = 1).fillna(0)

    X = df_log.drop(columns = price_col)
    y = df_log.loc[:, price_col]
    
    return X, y


# ## 1.2. Tuning and building the model + bootstrap

# In[6]:


def multiprocess_algorithm(alg, n, X, y, params):
    result = []
    for _ in range(n):
        result.append(alg.remote(X, y, params))
        
    return result

def multiprocess_algorithm_no_params(alg, n, X, y):
    result = []
    for _ in range(n):
        result.append(alg.remote(X, y))
        
    return result


# In[7]:


@ray.remote
def tune_regression(X, y, lag, dep):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    glm = linear_model.LinearRegression().fit(X_train, y_train)
    score = glm.score(X_test, y_test)
    
    return score, lag, dep
    

@ray.remote
def bootstrap_linear_regression(X, y, suppress = True):
    y_pred = []
    mape = []


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        
    glm = linear_model.LinearRegression()
    glm.fit(X_train, y_train)
    y_pred = glm.predict(X_test)
    mape.append(mean_absolute_percentage_error(y_test, y_pred))
        
        
    if not suppress:
        print('MAPE: ', np.mean(mape))
    
    return mape, glm
 


# In[8]:


@ray.remote
def tune_forest(X, y, lag, dep):
       
    rfr = RandomForestRegressor()
        
    param_grid = {
            'n_estimators' : [100, 200, 300],
            'max_features' : ['auto', 'sqrt', 'log2'],
            'max_depth' : [2,3,4],
            'criterion' : ['mse', 'mae']
            }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
       
    GSCV = GridSearchCV(estimator=rfr, param_grid=param_grid, cv = 5, n_jobs=-1)
    GSCV.fit(X_train, y_train)

    return GSCV.cv_results_, GSCV.best_params_, lag, dep


@ray.remote
def bootstrap_forest(X, y, rfr_params, suppress = True):
    mape = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    ## Build    
    rfr = RandomForestRegressor(criterion = rfr_params['criterion'],
                        max_depth=rfr_params['max_depth'], 
                        max_features=rfr_params['max_features'],
                        n_estimators= rfr_params['n_estimators'])
        
    rfr.fit(X_train, y_train)
        
    y_pred = rfr.predict(X_test)
    mape.append(mean_absolute_percentage_error(y_test, y_pred))
        
    if not suppress:
        print('MAPE: ', np.mean(mape))
    
    return mape, rfr


# In[9]:


@ray.remote
def tune_adaboost(X, y, lag, dep):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    ada = AdaBoostRegressor()
        
    param_grid = {
                'n_estimators' : [100, 200, 300],
                'learning_rate' : [0.9, 0.95, 1],
                'loss' : ['linear', 'square', 'exponential'],
            }
        
    GSCV = GridSearchCV(estimator=ada, param_grid=param_grid, cv = 5, n_jobs=-1)
    GSCV.fit(X_train, y_train) 

    return GSCV.cv_results_, GSCV.best_params_, lag, dep


@ray.remote
def bootstrap_adaboost(X, y, ada_params, suppress = True):
    mape = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    ## Build    
    ada = AdaBoostRegressor(
                        learning_rate = ada_params['learning_rate'],
                        loss = ada_params['loss'],
                        n_estimators= ada_params['n_estimators'])
   
    ada.fit(X_train, y_train)
    
    y_pred = ada.predict(X_test)
    mape.append(mean_absolute_percentage_error(y_test, y_pred))
    
    if not suppress:
        print('MAPE: ', np.mean(mape))

    return mape, ada



# In[10]:


@ray.remote
def tune_gradientboost(X, y, lag, dep):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    boost = GradientBoostingRegressor()

    param_grid_boost = {
                'n_estimators' : [100, 200, 300],
                'loss' : ['ls', 'lad', 'huber'],
                'criterion' : ['mse', 'friedman_mse'],
                'max_depth' : [2,3,4],
                'max_features' : ['auto', 'sqrt', 'log2'],
                }

    GSCV_boost = GridSearchCV(estimator=boost, param_grid=param_grid_boost, cv = 5, n_jobs=-1)
    GSCV_boost.fit(X_train, y_train)

    return GSCV_boost.cv_results_, GSCV_boost.best_params_, lag, dep 


@ray.remote
def bootstrap_gradientboost(X, y, boost_params, suppress = True):
    mape = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    ## Build      
    reg = GradientBoostingRegressor(criterion = boost_params['criterion'],
                                n_estimators=boost_params['n_estimators'],
                                loss=boost_params['loss'],
                                max_depth=boost_params['max_depth'], 
                                max_features=boost_params['max_features'])
        
    reg.fit(X_train, np.ravel(y_train))
        
    y_pred = reg.predict(X_test)
    mape.append(mean_absolute_percentage_error(y_test, y_pred))
    
    if not suppress:
        print('MAPE: ', np.mean(mape))
        
    return mape, reg


# In[14]:


def tune_models(data, price_col, lag_index, max_lag, dep_rates, suffix):
    '''
    Creates an input dataset for and hands it over to all remote tuning methods.
    Makes the results readable using ray.get() and returns readable data.
    Creates a pickle of each completed get() operation for safety.
    '''
    
    start = time.time()
    df = create_tuning_data(data, price_col, lag_index, max_lag, dep_rates)
    
    results_glm = []
    for i in range(df.shape[0]):
        results_glm.append(tune_regression.remote(df['X'][i], df['y'][i], df['lag'][i], df['dep'][i]))
        
    results_rfr = []
    for i in range(df.shape[0]):
        results_rfr.append(tune_forest.remote(df['X'][i], df['y'][i], df['lag'][i], df['dep'][i]))
    
    results_ada = []
    for i in range(df.shape[0]):
        results_ada.append(tune_adaboost.remote(df['X'][i], df['y'][i], df['lag'][i], df['dep'][i]))
        
    results_grad = []
    for i in range(df.shape[0]):
        results_grad.append(tune_gradientboost.remote(df['X'][i], df['y'][i], df['lag'][i], df['dep'][i]))
        
    print(f'Results are in! Time elapsed: {(time.time()-start)} sec')
    
    start = time.time()
    
    readable_glm = ray.get(results_glm)
    store_to_pickle(f'readable_glm_{suffix}', readable_glm)
    print(f'GLM is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
    
    readable_rfr = ray.get(results_rfr)
    store_to_pickle(f'readable_rfr_{suffix}', readable_rfr)
    print(f'RFR is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
    
    readable_ada = ray.get(results_ada)
    store_to_pickle(f'readable_ada_{suffix}', readable_ada)
    print(f'ADA is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
    
    readable_grad = ray.get(results_grad)) 
    store_to_pickle(f'readable_grad_{suffix}', readable_grad)
    print(f'GRAD {i} is readable and saved to pickle! Time elapsed: {(time.time()-start)/60} min')
    
    return readable_glm, readable_rfr, readable_ada, readable_grad


# In[12]:


def get_best_params_regression(readable):
    best_score = 0
    best_params_position = -1
    
    for i in range(len(readable)):
        if readable[i][0] > best_score:
            best_score = readable[i][0]
            best_params_position = i
        
    lag = int(readable[best_params_position][1])
    dep = readable[best_params_position][2]
            
    return best_score, lag, dep

def get_best_params(readable):
    best_score = 0
    best_params_position = -1

    for i in range(len(readable)):
        best_score_position = np.where(readable[i][0]['rank_test_score'] == 1)
        mean_score = np.nanmean([readable[i][0]['split0_test_score'][best_score_position][0],
                        readable[i][0]['split1_test_score'][best_score_position][0],
                        readable[i][0]['split2_test_score'][best_score_position][0],
                        readable[i][0]['split3_test_score'][best_score_position][0],
                        readable[i][0]['split4_test_score'][best_score_position][0]])
    
        if mean_score > best_score:
            best_score = mean_score
            best_params_position = i
        
        best_params = readable[best_params_position][1]
        lag = int(readable[best_params_position][2])
        dep = readable[best_params_position][3]
            
        meta = pd.DataFrame({'position': best_params_position, 'score': best_score}, index = [0])

    return meta, lag, dep, best_params


# In[13]:


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
    n_1 = 50
    n_2 = 25
    
    ## Start multiprocessing for each algorithm with optimal dataset
    
    X, y = create_model_data(data, price_col, lag_index, lags[0], dep_rates[0])
    glm_future = multiprocess_algorithm_no_params(bootstrap_linear_regression, n_1, X, y)
    glm = ray.get(glm_future)
    print(f'GLM is readable! Time elapsed: {(time.time()-start)/60} min')
    
    X, y = create_model_data(data, price_col, lag_index, lags[2], dep_rates[2])
    ada_future = multiprocess_algorithm(bootstrap_adaboost, n_1, X, y, params[2])
    ada = ray.get(ada_future)
    print(f'ADA is readable! Time elapsed: {(time.time()-start)/60} min')
    
    X, y = create_model_data(data, price_col, lag_index, lags[1], dep_rates[1])
    rfr_future = multiprocess_algorithm(bootstrap_forest, n_1, X, y, params[1])
    rfr = ray.get(rfr_future)
    print(f'RFR is readable! Time elapsed: {(time.time()-start)/60} min')
    
    
    X, y = create_model_data(data, price_col, lag_index, lags[3], dep_rates[3])
    grad_future = multiprocess_algorithm(bootstrap_gradientboost, n_2, X, y, params[3])
    grad = ray.get(grad_future)
    print(f'GRAD is readable! Time elapsed: {(time.time()-start)/60} min')
    
    return rfr, glm, ada, grad


# In[ ]:


data_fc = pd.read_excel('dataframes.xlsx', 'fuel_cell')
data_fc.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_fc = data_fc.sort_values(by = 'Year')
fc_subset = data_fc.loc[(data_fc['Year'] > 1982) & (data_fc['Year'] <= 2020)].reset_index().drop('index', axis = 1)

data_ev = pd.read_excel('dataframes.xlsx', 'li_ev')
data_ev.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_ev = data_ev.sort_values(by = 'Year')
ev_subset = data_ev.loc[(data_ev['Year'] > 1982) & (data_ev['Year'] <= 2020)].reset_index().drop('index', axis = 1)
ev_subset['USD2020/kWh'] = ev_subset.loc[:, ['EV_USD2020/kWh','EV_2_USD2020/kWh']].mean(axis = 1)
ev_subset = ev_subset.drop(columns = ['EV_USD2020/kWh','EV_2_USD2020/kWh'])

data_res = pd.read_excel('dataframes.xlsx', 'li_res')
data_res.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_res = data_res.sort_values(by = 'Year')
res_subset = data_res.loc[(data_res['Year'] > 1982) & (data_res['Year'] <= 2020)].reset_index().drop('index', axis = 1)

data_gen = pd.read_excel('dataframes.xlsx', 'li_gen')
data_gen.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_gen = data_gen.sort_values(by = 'Year')
gen_subset = data_gen[(data_gen['Year'] > 1982) & (data_gen['Year'] <= 2020)].reset_index().drop(columns = ['index'])


# # Full run-down
# ## Fuel cell dataset
# 

# In[ ]:


warnings.filterwarnings(action = 'ignore')
ray.init(ignore_reinit_error=True)
results = tune_models(fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells'],
                                        7, np.linspace(0, 0.2, 21), 'fc')

param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results)):
    if i > 0:
        temp = get_best_params(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
fc_final = start_algs(fc_subset, 'USD2020/kW', ['Fuel_cell_h01m8', 'Hydrogen and fuel cells'], 
                      param_info['lag'], param_info['dep'], param_info['params'])

ray.shutdown()


# ## Li-ion: res

# In[ ]:


warnings.filterwarnings(action = 'ignore')
ray.init(ignore_reinit_error=True)
results = tune_models(res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage'],
                                        7, np.linspace(0, 0.2, 21), 'res')

param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results)):
    if i > 0:
        temp = get_best_params(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
res_final = start_algs(res_subset, 'Consumer_Cells_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage'], 
                      param_info['lag'], param_info['dep'], param_info['params'])

ray.shutdown()


# ## Li-ion: gen

# In[ ]:


warnings.filterwarnings(action = 'ignore')
ray.init(ignore_reinit_error=True)
results = tune_models(gen_subset, 'Avg_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage'],
                                        7, np.linspace(0, 0.2, 21), 'gen')

param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results)):
    if i > 0:
        temp = get_best_params(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
gen_final = start_algs(gen_subset, 'Avg_USD2020/kWh', ['Li_ion_h01m10', 'Energy storage'], 
                      param_info['lag'], param_info['dep'], param_info['params'])

ray.shutdown()


# ## Li-ion: ev

# In[ ]:


warnings.filterwarnings(action = 'ignore')
ray.init(ignore_reinit_error=True)
results = tune_models(ev_subset, 'USD2020/kWh', ['Li_ev_Y02T90', 'EV aggregated'],
                                        7, np.linspace(0, 0.2, 21), 'ev')

param_info = pd.DataFrame(columns = ['meta_data', 'lag', 'dep', 'params'])

for i in range(len(results)):
    if i > 0:
        temp = get_best_params(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2],
                                        'params': temp[3]}, ignore_index = True)
    else:
        temp = get_best_params_regression(results[i])
        param_info = param_info.append({'meta_data': temp[0], 'lag': temp[1], 'dep': temp[2], 
                                        'params': None}, ignore_index = True)
    
ev_final = start_algs(ev_subset, 'USD2020/kWh', ['Li_ev_Y02T90', 'EV aggregated'], 
                      param_info['lag'], param_info['dep'], param_info['params'])

ray.shutdown()


# In[ ]:

