#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import shap
import pickle
import scipy
import sklearn
import seaborn as sns
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')


# In[3]:


## Import datasets
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


# ## Transforming methods and Import

# In[4]:


def get_learning_rates(coefs_var):
    '''calculates the learning rate from an array model coefficients / betas
    '''
    return pd.Series([1-2**coef for coef in coefs_var])


# In[5]:


def flatten(arr):
    '''Flatten an array
    '''
    return [item for sublist in arr for item in sublist]


# In[6]:


def to_list_of_df(arr):
    '''Transform the result arrays to dataframes to make them easier to work with.
    '''
    list = []
    for alg in arr:
        df = pd.DataFrame(alg)
        df.columns = ['mape', 'estimator', 'X_train', 'X_test', 'y_train', 'y_test']
        list.append(df)
    return list


# In[7]:


def shap_values(arrs):
    '''calculate shap-values for a given result-dataframe. Needs to be transformed to df using to_list_of_df() first.
    '''
    shaps = []
    for i, arr in enumerate(arrs):
        values = []
        
        ### Different explainers are needed to work with the different models.
        
        if i == 0:
            for j in range(arr.shape[0]):
                explainer = shap.Explainer(arr['estimator'][j].predict, sm.add_constant(arr['X_train'][j], has_constant = 'add'))
                values.append(explainer(sm.add_constant(arr['X_test'][j], has_constant = 'add')))
            shaps.append(values)
        elif i == 2:
            for j in range(arr.shape[0]):
                explainer = shap.Explainer(arr['estimator'][j].predict,  arr['X_train'][j])
                values.append(explainer(arr['X_test'][j]))
            shaps.append(values)
        else:
            for j in range(arr.shape[0]):
                explainer = shap.TreeExplainer(arr['estimator'][j])
                values.append(explainer(arr['X_test'][j]))
            shaps.append(values)
                         
    return shaps


# In[8]:


def aggregate_explainers(shap_arrs):
    '''By default, multiple explainers are returned by shap_values(). Aggregate the results into one explainer to enable plotting.
    '''
    result = []
    for arr in shap_arrs:
        ex = arr.copy()[0]
        for i in range(len(arr) - 1):
            ex.values = np.concatenate((ex.values, arr[i+1].values))
            ex.base_values = np.concatenate((ex.base_values, arr[i+1].base_values))
            ex.data = np.concatenate((ex.data, arr[i+1].data))
        result.append(ex)
    return result


# In[9]:


def aggregate_test_sets(result_alg):
    '''Supply only the part of the results array that includes data for the desired algorithm, e.g. fc_results[alg]
    '''
    l = [item for arr in result_alg['X_test'] for item in arr.values]
    df = pd.DataFrame(l, columns = result_alg['X_test'][0].columns)
    return df


# In[10]:


def calculate_baseline(row):
    '''calculate baseline predictions for a given data set. The baseline is calculated off of the test and training sets that
    were used to construct the predictive models.
    '''
    d = []
    const_pred = np.mean(row['y_train'])
    y_true = row['y_test']
    return mean_absolute_percentage_error(y_true, [const_pred] * len(y_true))


def baseline_cis(dataset):
    '''calculate the confidence interval for the baseline errors
    '''
    mape = dataset.apply(calculate_baseline, axis = 1)
    sample_mean = np.mean(mape)
    standard_error = scipy.stats.sem(mape)
    degrees_of_freedom = len(mape)-1

    ci_95 = scipy.stats.t.interval(0.95, degrees_of_freedom, sample_mean, standard_error)
    
    return {'mean': sample_mean, 'confidence_interval_95_lo': ci_95[0], 'confidence_interval_95_hi': ci_95[1], 'alg': 'CONST'}
    
    
def baseline_all(arrs):
    '''Methdod that calculates baselines for all algorithms on all data sets
    '''
    datasets = ['EV', 'FC', 'RES']
    l = []
    for i, arr in enumerate(arrs):
        for a in arr:
            temp = baseline_cis(a)
            temp['dataset'] = datasets[i]
            l.append(temp)

    return pd.DataFrame(l)
    


# #### Load Data

# In[11]:


## import modelling results
ev_final = pickle.load(open('results/ev_final.pickle', 'rb'))
res_final = pickle.load(open('results/res_final.pickle', 'rb'))
fc_final = pickle.load(open('results/fc_final.pickle', 'rb'))

ev_final_pat_only = pickle.load(open('results/ev_final_pat_only.pickle', 'rb'))
res_final_pat_only = pickle.load(open('results/res_final_pat_only.pickle', 'rb'))
fc_final_pat_only = pickle.load(open('results/fc_final_pat_only.pickle', 'rb'))


# In[12]:


### transfer the results to a dataframe with a format that makes evaluation easier
ev_results = to_list_of_df(ev_final)
res_results = to_list_of_df(res_final)
fc_results = to_list_of_df(fc_final)

ev_results_pat_only = to_list_of_df(ev_final_pat_only)
res_results_pat_only = to_list_of_df(res_final_pat_only)
fc_results_pat_only = to_list_of_df(fc_final_pat_only)


# In[13]:


### import calculated learning betas for the machine learning algorithms
store = pickle.load(open('results/learning_betas.pickle', 'rb'))
ev_beta = store[0]
res_beta = store[1]
fc_beta = store[2]

ev_beta_pat_only = store[3]
res_beta_pat_only = store[4]
fc_beta_pat_only = store[5]


# In[14]:


## load saved (aggregated) shap values as calculation takes a while
store = pickle.load(open('results/shaps.pickle', 'rb'))
ev_shap = store[0]
res_shap = store[1]
fc_shap = store[2]
store = pickle.load(open('results/shaps_agg.pickle', 'rb'))
ev_shap_agg = store[0]
res_shap_agg = store[1]
fc_shap_agg = store[2]


# #### Calculate SHAPs and save to pickle

# In[ ]:


## calculate shap values and aggregate
ev_shap = shap_values(ev_results)
res_shap = shap_values(res_results)
fc_shap = shap_values(fc_results)

ev_shap_agg = aggregate_explainers(ev_shap)
res_shap_agg = aggregate_explainers(res_shap)
fc_shap_agg = aggregate_explainers(fc_shap)


# In[ ]:


## save all of the above
store = [ev_shap, res_shap, fc_shap]
file = open('results/shaps.pickle', 'wb')
pickle.dump(store, file)
file.close()
store = [ev_shap_agg, res_shap_agg, fc_shap_agg]
file = open('results/shaps_agg.pickle', 'wb')
pickle.dump(store, file)
file.close()


# ### Visualizing methods

# In[15]:


def calc_ci(df, colname):
    '''calculate the confidence interval of a given collection of MAPEs
    '''
    sample = df[colname]
    confidence_level_95 = 0.95
    sample_mean = np.mean(sample)
    standard_error = scipy.stats.sem(sample)
    degrees_of_freedom = len(sample)-1

    ci_95 = scipy.stats.t.interval(confidence_level_95, degrees_of_freedom, sample_mean, standard_error)
    
    return {'sample_mean': sample_mean, 'ci_95': ci_95}


# In[16]:


def get_confidence_intervals(arrs):
    '''get confidence intervals for all data sets and algorithms
    '''
    d = []
    algs = ['GLM', 'RFR', 'ADA', 'GRAD']
    datasets = ['EV', 'FC', 'RES']
    
    for i, arr in enumerate(arrs):
        for j, a in enumerate(arr):
            ci_data = calc_ci(a, 'mape')
    
            d.append({'mean': ci_data['sample_mean'], 'confidence_interval_95_lo': ci_data['ci_95'][0], 
                      'confidence_interval_95_hi': ci_data['ci_95'][1], 'dataset': datasets[i], 'alg': algs[j]})
    
    return pd.DataFrame(d)


# In[17]:


def plot_cis(cis, baseline = None, interval = 95):
    '''
    plot confidence intervals (of the mape) without showing variable distribution
    cis: Dataframe of confidence intervals as returned by get_confidence_intervals
    interval: plot the 95% interval by default. Other intervals need to be calculated and includeed in the parameters first.
    
    baseline: The basline error for each data set.
    '''
    
    interval = f'confidence_interval_{interval}'
    datasets = cis['dataset'].unique()
    fig, ax = plt.subplots(len(datasets), 1, figsize = [7, 9*len(datasets)])
    
    for i, d in enumerate(datasets):
        print(d)
        data = cis.loc[cis['dataset'] == d, ['mean', f'{interval}_lo', f'{interval}_hi', 'alg']]
        
        ax[i].scatter((data.index%4, data.index%4), (data[f'{interval}_lo'], data[f'{interval}_hi']),
                      c = 'orange', alpha = 0.5)
        ax[i].plot((data.index%4, data.index%4), (data[f'{interval}_lo'], data[f'{interval}_hi']),
                   c = 'orange')
        ax[i].scatter(data.index%4, data['mean'], c = 'b', marker = 'x')
        
        ax[i].set_xticklabels(data['alg'] + [''])
        ax[i].set_xticks([0, 1, 2, 3, 3.8])
        ax[i].set_ylabel('MAPE')
        
        if baseline is not None:
            ax[i].add_artist(AnchoredText(
                f'Baseline: {np.round(np.mean(baseline.loc[baseline["dataset"] == d, "mean"]), 4)}', 
                loc = 'upper right',
                borderpad = 0.1))


# In[18]:


def boxplot_all(arr_result, titles):
    '''boxplot the MAPE for each data set and algorithm
    '''
    fig, ax = plt.subplots(len(titles), 1, figsize = [7, 9*len(titles)])
    ci = []
    for j in range(len(arr_result)):
        for i in range(len(arr_result[j])):
            ax[j].boxplot(arr_result[j][i].loc[arr_result[j][i]['mape'] <= 2, 'mape'], positions = [i*0.5], showfliers = True, vert = True)
        ax[j].set_xticklabels(['GLM', 'RFR', 'ADA', 'GRAD'])
        ax[j].set_ylabel('MAPE')
        print(titles[j])


# ## Distribution

# In[19]:


### Get CIs for all sets and algorithms including the baseline and visualize the results in a table

ci = get_confidence_intervals([ev_results, fc_results, res_results])
baseline = baseline_all([ev_results, fc_results, res_results])

ci_pat = get_confidence_intervals([ev_results_pat_only, fc_results_pat_only, res_results_pat_only])
baseline_pat = baseline_all([ev_results_pat_only, fc_results_pat_only, res_results_pat_only])

### Table for the full data set
ci.append(baseline).groupby(by = ['dataset', 'alg']).mean().round(4)


# In[20]:


### Table for the reduced data set
ci_pat.append(baseline_pat).groupby(by = ['dataset', 'alg']).mean().round(4)


# In[21]:


plot_cis(ci, baseline)


# In[22]:


boxplot_all([ev_results, fc_results, res_results], ['EV', 'FC', 'RES'])


# ## Feature Importance, shap-values, learning rates

# ### Dataset I: Fuel cells

# In[23]:


## Maximum and minimum training set fit
a = [model.rsquared for model in fc_results[0]['estimator']]
print('Minimum R2:', np.min(a).round(4), 'Maximum R2:', np.max(a).round(4))


# #### Learning rates

# In[25]:


## Create dict with coefficients and p-values of linear regression
ols_results_fc = {'coef': pd.DataFrame([fc_results[0]['estimator'][i].params.values for i in range(len(fc_results[0]['estimator']))], columns = fc_results[0]['estimator'][0].params.index),
                'pval': pd.DataFrame([fc_results[0]['estimator'][i].pvalues.values for i in range(len(fc_results[0]['estimator']))], columns = fc_results[0]['estimator'][0].params.index)}

algs = ['Lin. Regression', 'Random Forest', 'AdaBoost', 'Gradient Boost']
df = pd.DataFrame()

## Iteratively calculate confidence intervals of MAPEs for each column and algorithm
for i, col in enumerate(['MW_dom', 'Fuel_cell_h01m8', 'Hydrogen and fuel cells']):
    d = calc_ci(pd.DataFrame(get_learning_rates(ols_results_fc['coef'][col])), 0)
    df = df.append({'alg': 'Lin. Regression', 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
    for j in range(3):
        d = calc_ci(pd.DataFrame(flatten(get_learning_rates(fc_beta[col][j]))), 0)
        df = df.append({'alg': algs[j+1], 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)

## Show
fc_lr = df
df.groupby(['alg', 'col']).mean().round(4)*100


# In[26]:


## Repeat procedure with reduced data set
ols_results_fc_pat = {'coef': pd.DataFrame([fc_results_pat_only[0]['estimator'][i].params.values for i in range(len(fc_results_pat_only[0]['estimator']))], columns = fc_results_pat_only[0]['estimator'][0].params.index),
                'pval': pd.DataFrame([fc_results_pat_only[0]['estimator'][i].pvalues.values for i in range(len(fc_results_pat_only[0]['estimator']))], columns = fc_results_pat_only[0]['estimator'][0].params.index)}

algs = ['Lin. Regression', 'Random Forest', 'AdaBoost', 'Gradient Boost']
df = pd.DataFrame()

for i, col in enumerate(['MW_dom', 'Fuel_cell_h01m8']):
    d = calc_ci(pd.DataFrame(get_learning_rates(ols_results_fc_pat['coef'][col])), 0)
    df = df.append({'alg': 'Lin. Regression', 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
    for j in range(3):
        d = calc_ci(pd.DataFrame(flatten(get_learning_rates(fc_beta_pat_only[col][j]))), 0)
        df = df.append({'alg': algs[j+1], 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)

fc_pat_lr = df
df.groupby(['alg', 'col']).mean().round(4)*100


# #### Shap

# In[28]:


## For each algorithm: Plot SHAP values and mean absolute impact of features
for alg in fc_shap_agg:  
    print(pd.DataFrame({'Mean':np.mean(np.abs(alg.values), axis = 0)[-4:], 'col': fc_results[0]['X_train'][0].columns}))
    shap.plots.beeswarm(alg, plot_size = [18,6])


# ### Dataset II: RES

# In[41]:


## Maximum and minimum training set fit
a = [model.rsquared for model in res_results[0]['estimator']]
print('Minimum R2:', np.min(a).round(4), 'Maximum R2:', np.max(a).round(4))


# #### Learning rates

# In[132]:


## Create dict with coefficients and p-values of linear regression
ols_results_res = {'coef': pd.DataFrame([res_results[0]['estimator'][i].params.values for i in range(len(res_results[0]['estimator']))], columns = res_results[0]['estimator'][0].params.index),
                'pval': pd.DataFrame([res_results[0]['estimator'][i].pvalues.values for i in range(len(res_results[0]['estimator']))], columns = res_results[0]['estimator'][0].params.index)}


algs = ['Lin. Regression', 'Random Forest', 'AdaBoost', 'Gradient Boost']
df = pd.DataFrame()

## Iteratively calculate confidence intervals of MAPEs for each column and algorithm
for i, col in enumerate(['MWh_combined', 'Li_ion_h01m10', 'Energy storage']):
    d = calc_ci(pd.DataFrame(get_learning_rates(ols_results_res['coef'][col])), 0)
    df = df.append({'alg': 'Lin. Regression', 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
    for j in range(3):
        d = calc_ci(pd.DataFrame(flatten(get_learning_rates(res_beta[col][j]))), 0)
        df = df.append({'alg': algs[j+1], 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
        
res_lr = df
df.groupby(['alg', 'col']).mean().round(4)*100


# In[133]:


## Repeat procedure with reduced data set
ols_results_res_pat = {'coef': pd.DataFrame([res_results_pat_only[0]['estimator'][i].params.values for i in range(len(res_results_pat_only[0]['estimator']))], columns = res_results_pat_only[0]['estimator'][0].params.index),
                'pval': pd.DataFrame([res_results_pat_only[0]['estimator'][i].pvalues.values for i in range(len(res_results_pat_only[0]['estimator']))], columns = res_results_pat_only[0]['estimator'][0].params.index)}

algs = ['Lin. Regression', 'Random Forest', 'AdaBoost', 'Gradient Boost']
df = pd.DataFrame()

for i, col in enumerate(['MWh_combined', 'Li_ion_h01m10']):
    d = calc_ci(pd.DataFrame(get_learning_rates(ols_results_res_pat['coef'][col])), 0)
    df = df.append({'alg': 'Lin. Regression', 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
    for j in range(3):
        d = calc_ci(pd.DataFrame(flatten(get_learning_rates(res_beta_pat_only[col][j]))), 0)
        df = df.append({'alg': algs[j+1], 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)

res_pat_lr = df
df.groupby(['alg', 'col']).mean().round(4)*100


# #### Shap

# In[44]:


## For each algorithm: Plot SHAP values and mean absolute impact of features
for i, alg in enumerate(res_shap_agg):
    print(pd.DataFrame({'Mean':np.mean(np.abs(alg.values), axis = 0)[-4:], 'col': res_results[0]['X_train'][0].columns}))
    shap.plots.beeswarm(alg, plot_size = [18,6])
    


# ### Dataset III: EV

# In[45]:


## Maximum and minimum training set fit
a = [model.rsquared for model in ev_results[0]['estimator']]
print('Minimum R2:', np.min(a).round(4), 'Maximum R2:', np.max(a).round(4))


# #### Learning rates

# In[130]:


## Create dict with coefficients and p-values of linear regression
ols_results_ev = {'coef': pd.DataFrame([ev_results[0]['estimator'][i].params.values for i in range(len(ev_results[0]['estimator']))], columns = ev_results[0]['estimator'][0].params.index),
                'pval': pd.DataFrame([ev_results[0]['estimator'][i].pvalues.values for i in range(len(ev_results[0]['estimator']))], columns = ev_results[0]['estimator'][0].params.index)}

algs = ['Lin. Regression', 'Random Forest', 'AdaBoost', 'Gradient Boost']
df = pd.DataFrame()

## Iteratively calculate confidence intervals of MAPEs for each column and algorithm
for i, col in enumerate(['MWh_ev', 'Li_ev_Y02T90', 'EV aggregated']):
    d = calc_ci(pd.DataFrame(get_learning_rates(ols_results_ev['coef'][col])), 0)
    df = df.append({'alg': 'Lin. Regression', 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
    for j in range(3):
        d = calc_ci(pd.DataFrame(flatten(get_learning_rates(ev_beta[col][j]))), 0)
        df = df.append({'alg': algs[j+1], 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)

ev_lr = df       
df.groupby(['alg', 'col']).mean().round(4)*100


# In[131]:


## Repeat procedure with reduced data set
ols_results_ev_pat = {'coef': pd.DataFrame([ev_results_pat_only[0]['estimator'][i].params.values for i in range(len(ev_results_pat_only[0]['estimator']))], columns = ev_results_pat_only[0]['estimator'][0].params.index),
                'pval': pd.DataFrame([ev_results_pat_only[0]['estimator'][i].pvalues.values for i in range(len(ev_results_pat_only[0]['estimator']))], columns = ev_results_pat_only[0]['estimator'][0].params.index)}

algs = ['Lin. Regression', 'Random Forest', 'AdaBoost', 'Gradient Boost']
df = pd.DataFrame()

for i, col in enumerate(['MWh_ev', 'Li_ev_Y02T90']):
    d = calc_ci(pd.DataFrame(get_learning_rates(ols_results_ev_pat['coef'][col])), 0)
    df = df.append({'alg': 'Lin. Regression', 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
    for j in range(3):
        d = calc_ci(pd.DataFrame(flatten(get_learning_rates(ev_beta_pat_only[col][j]))), 0)
        df = df.append({'alg': algs[j+1], 'col': col, 'mean_learn_rate': d['sample_mean'], 'delta': np.abs(d['ci_95'][0]- d['sample_mean'])}, ignore_index = True)
        
ev_pat_lr = df
df.groupby(['alg', 'col']).mean().round(4)*100


# #### Shap

# In[48]:


## For each algorithm: Plot SHAP values and mean absolute impact of features
for alg in ev_shap_agg:
    print(pd.DataFrame({'Mean':np.mean(np.abs(alg.values), axis = 0)[-5:], 'col': ev_results[0]['X_train'][0].columns}))
    shap.plots.beeswarm(alg, plot_size = [14,6])

