#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')

from sklearn.model_selection import train_test_split
from matplotlib import cm


# # 0. Transformation methods
# Includes methods for inflation adjustment and currency conversion

# In[2]:


## import and clean
inflation_rates = pd.read_csv('Sources/currency/Inflation_rates.csv')

col = inflation_rates.columns.str.split(';', expand = True)
inflation_rates = inflation_rates[inflation_rates.columns[0]].str.split(';', expand = True)
inflation_rates.columns = col[0]
inflation_rates.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)
inflation_rates.rename(columns = {'2020,': '2020'}, inplace = True)
inflation_rates


# In[3]:


## into long format, subset Europe and USA
inflation = inflation_rates.melt(id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'Inflation Rate')
inflation['Inflation Rate'].replace('', np.nan, inplace = True)
inflation['Inflation Rate'].replace(',', np.nan, inplace = True)

inflation['Inflation Rate'] = inflation['Inflation Rate'].astype('float') / 100 + 1
inflation['Year'] = inflation['Year'].astype('int')

inflation = inflation.loc[(inflation['Country Code'] == 'USA') | (inflation['Country Code'] == 'EMU') | (inflation['Country Code'] == 'EUU')]

for code in inflation['Country Code'].unique():
    inflation.loc[inflation['Country Code'] == code, 'Inflation to 2020'] = np.cumprod(np.flip(inflation.loc[inflation['Country Code'] == code, 'Inflation Rate']))

inflation


# In[4]:


def adjust_for_inflation(series, region = 'USA', base = None):
    '''
    Adjusts a series for inflation, depending on a currency and base year
    region: may be USA, EUU or EMU
    base: some data might already be adjusted to inflation for a year earlier than 2020. In that case, this year needs to be given as base year
    '''
    if base != None:
        a = series * float(inflation.loc[(inflation['Country Code'] == region) & (inflation['Year'] == base), 'Inflation to 2020'])
    else:
        a = series.to_numpy() * (inflation.loc[(inflation['Country Code'] == region), 'Inflation to 2020'][-len(series):]).to_numpy()
    return a


# In[5]:


def euro_to_usd(series): 
    '''
    Needs to be in 2020Euro! Inflation might have to be applied first, method: adjust_for_inflation()
    '''
    exchange_rate = 1.1422
    return series * exchange_rate


# # 1. General Data
# 
# First, data sets including multiple types of technology, and technology-independent data will be examined. The technologies themselves are grouped together below.

# ## 1.1. Electricity prices
#  

# In[6]:


## import US table as it contains industry and residential data
us_prices = pd.read_csv('Sources/electricity_prices/Electricity_prices_US.csv')
us_prices['Value'].replace(to_replace = 'Not Available', value = np.nan, inplace = True)
us_prices['Value'].replace(to_replace = 'Not Applicable', value = np.nan, inplace = True)

us_prices['YYYYMM'] = us_prices['YYYYMM'].astype(str)
us_prices['Value'] = us_prices['Value'].astype(float)

us_prices['Year'] = us_prices['YYYYMM'].str.slice(stop = 4).astype(int)
us_prices['Month'] = us_prices['YYYYMM'].str.slice(start = 4)

us_prices.drop(['MSN', 'Column_Order', 'Unit', 'YYYYMM'], inplace = True, axis = 1)
us_prices


# In[7]:


## Yearly avg is given as month 13, hence all other values can be dropped
us_prices = us_prices[us_prices['Month'] == '13']

## Append average of residential and commercial prices
for year in us_prices['Year'].unique():
    us_prices = us_prices.append(
        {'Year': year, 'Month' : '13', 'Description' : 'Residential and Commercial: Mean',
        'Value' : np.mean(us_prices.loc[(us_prices['Year'] == year) & 
                        ((us_prices['Description'] == 'Average Retail Price of Electricity, Residential') |
                         (us_prices['Description'] == 'Average Retail Price of Electricity, Commercial')),
                        'Value'])
        }, ignore_index = True)
    
## adjust for inflation
for d in us_prices['Description'].unique():
    us_prices.loc[us_prices['Description'] == d, 'Infl. Value'] = adjust_for_inflation(us_prices.loc[us_prices['Description'] == d, 'Value'])

us_prices


# In[8]:


fig, ax = plt.subplots(figsize = [18, 8])

for area in us_prices['Description'].unique()[1:-1]:
    ax.plot(us_prices.loc[us_prices['Description'] == area, 'Year'],
           us_prices.loc[us_prices['Description'] == area, 'Infl. Value'], alpha = 0.5)

##plot selected series separately for better visibility    
ax.plot(us_prices.loc[us_prices['Description'] == 'Residential and Commercial: Mean', 'Year'],
           us_prices.loc[us_prices['Description'] == 'Residential and Commercial: Mean', 'Infl. Value'], marker = '*')

ax.plot(us_prices.loc[us_prices['Description'] == 'Average Retail Price of Electricity, Residential', 'Year'],
           us_prices.loc[us_prices['Description'] == 'Average Retail Price of Electricity, Residential', 'Infl. Value'], marker = '*')

ax.legend(us_prices['Description'].unique())
ax.set_ylabel('Price [US Cents/kWh]')
fig.autofmt_xdate()


# ## 1.2. Patents
# 
# Multiple sources used: \
# Source for cols 2-5: IEA/EPA report (2020)\
# Source for other cols: Own collection from EPA site

# In[9]:


## Import
patents = pd.read_excel('Sources/MA_Data.xlsx', 'Patents')

## Plot patents per year
fig, (ax1, ax2) = plt.subplots(2,1, figsize = [18, 14])
for id in patents.columns[1:]:
    ax1.plot(patents['Year'], patents[id], marker = 'o')
    
ax1.legend(patents.columns[1:], loc = 'upper left')
ax1.set_ylabel('Patent families')

## Plot cumulative patents by year
for id in patents.columns[1:]:
    ax2.plot(patents['Year'], np.cumsum(patents[id]), marker = 'o')

ax2.set_ylabel('Number of patent families')
ax2.set_title('Cumulative patents');
ax2.legend(patents.columns[1:], loc = 'upper left')


# ## 1.3. RD&D

# In[10]:


## Import
rdd = pd.DataFrame()

for filename in os.listdir('Sources/RDD'):
    if filename.endswith('.xlsx'):
        file = pd.read_excel(os.path.join('Sources/RDD/', filename))
        if rdd.shape[0] == 0:
            rdd = file
        else:
            rdd = rdd.append(file)

## set country as factors
rdd['COUNTRY'] = rdd['COUNTRY'].astype('category')

## replace missing values with NaN, transform all values to float
rdd.iloc[:, 2:] = rdd.iloc[:, 2:].replace('..', np.nan)
rdd.iloc[:, 2:] = rdd.iloc[:, 2:].astype('float')

## clean column names
col_index = list(np.where(rdd.columns.str.contains('in Million Euro') == False))
col = rdd.columns[col_index]
rdd_euro = rdd.drop(col[2:], axis = 1)

rdd_euro.columns = rdd_euro.columns.str.replace('^\d+ ', '')

## show
rdd_euro


# Choosing columns was done with help of documentation. Reasoning follows:
# * Vehicle batteries/storage, Advd power elecs, EV infrastructure. Will be summed up together as EV aggregated (index = 6,7,8)
# * Hydrogen and fuel cells for obvious reasons (index = 21)
# * Energy storage total as a general fallback category (index = 23)
# 

# In[11]:


## create DataFrame rdd_selection to contain only selected columns for later use
rdd_selection = rdd_euro.iloc[:, [0,1,21,23]]
rdd_selection['EV aggregated'] = rdd_euro.iloc[:, [6,7,8]].sum(axis = 1)

## melt to long format
rdd_selection = rdd_selection.melt(id_vars = ['COUNTRY', 'TIME'], value_name = 'Value', var_name = 'Description')

## replace NaNs, transform TIME to float
rdd_selection['TIME'] = rdd_selection['TIME'].astype(float)
rdd_selection['Value'].replace(to_replace = '..', value = np.nan, inplace = True)

## adjust for inflation
rdd_selection['Value'] = euro_to_usd(adjust_for_inflation(
    rdd_selection['Value'].astype('float'), 
    region = 'EUU',
    base = 2019))

## clean Description values (= names), replace NaN values with 0
rdd_selection['Description'] = rdd_selection['Description'].str.replace('Total .*', '')
rdd_selection['Value'] = rdd_selection['Value'].fillna(0)

rdd_selection


# In[12]:


rdd_selection.drop(np.where(rdd_selection['COUNTRY'] == 'European Union')[0]).groupby(
    ['TIME', 'Description', 'COUNTRY']).mean()


# In[13]:


## Taking all countries into the aggregation means dropping the EU, since EU member countries are also listed separately
## using mean in aggregation to smooth over missing data, compared to sum
rdd_aggregate = rdd_selection.drop(np.where(rdd_selection['COUNTRY'] == 'European Union')[0]).groupby(
    ['TIME', 'Description']).mean()

rdd_aggregate = rdd_aggregate.reset_index()


## Choosing data from 1983 onwards deemed sufficient to construct cumulative series  
rdd_aggregate = rdd_aggregate.loc[rdd_aggregate['TIME'] >= 1983]

## rename columns for clearer titles
rdd_aggregate = rdd_aggregate.rename(
                        columns = {'TIME': 'Year'}).pivot_table(
                        index = 'Year', columns = 'Description', values = 'Value').rename(
                        columns = {'GROUP 5: HYDROGEN AND FUEL CELLS' : 'Hydrogen and fuel cells'})

rdd_aggregate


# In[14]:


## Plot

fig, ax = plt.subplots(figsize = [18, 8])

for col in rdd_aggregate.columns:
    ax.plot(rdd_aggregate.index, rdd_aggregate[col], marker = 'o')
ax.legend(['Hydrogen and fuel cells', 'EV aggregated', 'Energy storage'])
ax.set_ylabel('R&D expenditure in million USD [2020]')
ax


# # 2. Technologies

# ### Pre: Importing GESDB Projects
# 
# This dataset includes close to 1700 observations of different energy storage projects worldwide.

# In[15]:


storage = pd.read_excel('Sources/GESDB_Projects.xlsx', 'All Projects')


# ## 2.1. PHS
# ### 2.1.1. Market

# In[16]:


phs = storage.loc[storage['Technology Broad Category'] == 'Pumped Hydro Storage',
        ['Technology Broad Category', 'Status', 'Rated Power (kW)', 'Country', 'Year commissioned', 
         'Year constructed','Year decommissioned', 'Capital Expenditure']]
phs


# In[17]:


## Importing PHS data from "the future cost of electrical storage" (Schmidt, 2017)
phs_fut_cost = pd.read_excel('Sources/MA_Data.xlsx', 'PHS_fut_cost')
phs_market = phs_fut_cost.iloc[:, [0, 1,3]]
phs_market


# Too many gaps in the dataset to be useful

# In[18]:


## Create second df only including datapoints with dates (GESDB data)
## Reduced set
phs_red = phs[phs['Year commissioned'].notna()]
phs_red = phs_red[phs_red['Year commissioned'] <= 2020] ##exclude outliers to be constructed in the future

## DataFrame showing only installed capacity per year
phs_installed = pd.DataFrame(columns = ['Year commissioned', 'installed_capacity'])
phs_installed['Year commissioned'] = np.sort(phs_red['Year commissioned'].unique())

for year in phs_installed['Year commissioned']:
    phs_installed.loc[phs_installed['Year commissioned'] == year, 'installed_capacity'] = np.sum(phs_red.loc[phs_red['Year commissioned'] == year, 'Rated Power (kW)'])


# In[19]:


## Plotting overview of installed capacity by year, cumulative capacity and number of usable observations
## Usable observations are those which have a date assigned to them

print(f'All Datapoints: {phs.shape[0]}' + '\n' + f'only with date: {phs_red.shape[0]}');

## Installed capactity plot
fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = [14,18])
ax1.bar(phs_red['Year commissioned'], phs_red['Rated Power (kW)']/1000,
      color = "blue" )
ax1.set_ylabel('Power in MW')
ax1.set_title('Installed capacity')

## Cumulative capacity plot
ax2.plot(phs_installed['Year commissioned'], np.cumsum(phs_installed['installed_capacity'])/1000,
      color = "blue", marker = 'o' )
ax2.plot(phs_market['Year'], phs_market['Cum GWh']*1000, marker = 'o')
ax2.plot(phs_market['Year'], phs_market['Cum GW']*1000, marker = 'o')

ax2.legend(['GESDB data', 'Cum_GWh_fut_cost', 'Cum_GW_fut_cost'])
ax2.set_ylabel('Power in MW')
ax2.set_title('Cumulative installed capacity')

## Histogram of observations
ax3.hist(phs_red.loc[:, 'Year commissioned'], bins = phs_red['Year commissioned'].shape[0], color = 'blue');
ax3.set_title('Observations in the reduced set');


# ### 2.1.2. Price

# In[20]:


## Plot number of observations
fig, ax = plt.subplots(figsize = [14, 6])
ax.hist(phs_red.loc[phs_red['Capital Expenditure'] > 0, 'Year commissioned'],  bins = phs_red['Year commissioned'].shape[0]);
ax.set_title(f' CapEx observations in the reduced set (n = {phs_red[phs_red["Capital Expenditure"] > 0].shape[0]})');


# In[21]:


## Plot price
phs_price = phs_fut_cost.iloc[:, [0,2,4]]

fig, ax = plt.subplots(figsize = [20, 6])

ax.plot(phs_price['Year'], phs_price['USD2015/kWh'], marker = 'o')
ax.plot(phs_price['Year'], phs_price['USD2015/kW'], marker = 'o')
ax.set_title('Price development')
ax.set_title('Price per unit');


# No visible price trend that could be analyzed. There are also only 28 price observations in non-consecutive years

# ## 2.2. Li ion batteries
# 

# ### 2.2.1. Lithium prices

# In[22]:


## Import data for lithium prices
li_price = pd.read_excel('Sources/MA_Data.xlsx', 'Li_prices')
for col in li_price.columns[1:]:
    li_price[col] = adjust_for_inflation(li_price[col])

## plot
fig, ax = plt.subplots(figsize=[15,5])
for col in li_price.columns[1:]:
    ax.plot(li_price['Year'], li_price[col]/1000, marker = 'o')
    
ax.legend(['carbonate', 'hydroxide', 'battery_grade'])
ax.set_title('Price of different kinds of lithium, adjusted for inflation')
ax.set_ylabel('USD2020/kg');


# In[23]:


## Zoom-In plot on battery_grade series
fig, ax = plt.subplots(figsize=[18,5])
ax.plot(li_price['Year'], li_price['price_per_ton_battery_grade'], marker = 'o')
ax.set_ylabel('Lithium price [USD/ton]')


# ### 2.2.2. Market data
# #### 2.2.2.1. GESDB data
# 

# In[24]:


## Extract lithium ion specific data from GESDB set
li_ion = storage.loc[storage['Technology Mid-Type'] == 'Lithium-ion Battery',
        ['Technology Broad Category', 'Status', 'Rated Power (kW)', 'Country', 'Year commissioned', 
         'Year constructed','Year decommissioned', 'Capital Expenditure']]


# In[25]:


## Create second df only including datapoints with dates
## Reduced set
li_ion_red = li_ion[li_ion['Year commissioned'].notna()]

## Installed capacity by year
li_ion_installed = pd.DataFrame(columns = ['Year commissioned', 'installed_capacity_projects'])
li_ion_installed['Year commissioned'] = np.sort(li_ion_red['Year commissioned'].unique())

for year in li_ion_installed['Year commissioned']:
    li_ion_installed.loc[li_ion_installed['Year commissioned'] == year, 'installed_capacity_projects'] = np.sum(li_ion_red.loc[li_ion_red['Year commissioned'] == year, 'Rated Power (kW)'])


# In[26]:


print(f'All Datapoints: {li_ion.shape[0]}' + '\n' + f'only with date: {li_ion_red.shape[0]}');

## Plot capacity by year
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = [14, 18])
ax1.bar(li_ion_red['Year commissioned'], li_ion_red['Rated Power (kW)']/1000,
      color = "red" )
ax1.set_ylabel('Power in MW')
ax1.set_title('Installed capacity per year')

## Plot cumulative capacity by year
ax2.plot(li_ion_installed['Year commissioned'], np.cumsum(li_ion_installed['installed_capacity_projects'])/1000, color = 'red', marker = 'o')
ax2.set_ylabel('Power in MW')
ax2.set_title('Cumulative installed capacity')

## Histogram of observations
ax3.hist(li_ion_red.loc[:, 'Year commissioned'], bins = li_ion_red['Year commissioned'].shape[0], color = 'red');
ax3.set_title('Observations in the reduced set');


# Not a lot of observations at the beginning and end of the period, rather short period overall

# #### 2.2.2.2 Other data sources

# In[27]:


## DataFrame containing all data on lithium ion
cap = pd.read_excel('Sources/MA_Data.xlsx', 'Li_ion_cap')

## pick only installed capacity / sales data from the full DataFrame
cap_market = cap.loc[:, ['Year', 'PV_storage_GER', 'sales_jp', 'Cum_GWh_elec', 'Cum_GW_elec', 'Cum_GWh_ev', 'Cum_GW_ev', 'Cum_GWh', 'Cum_GWh_elec_2']]
cap_market.columns = ['Year','storage_installs_ger', 'sales_no_jp', 'Cum_MWh_elec', 'Cum_MW_elec', 'Cum_MWh_ev', 'Cum_MW_ev', 'Cum_MWh', 'Cum_MWh_elec_2']
cap_market = cap_market.fillna(0)

## transfer GW(h) into MW(h)
cap_market.iloc[:, 3:9] = cap_market.iloc[:, 3:9]*1000


# In[28]:


## get numbers per year instead of cumulative
cap_market.iloc[:, [1,3,4,5,6,7,8]] = cap_market.iloc[:, [1,3,4,5,6,7,8]].diff()

for col in cap_market.columns:
    cap_market.loc[cap_market[col] <= 0, col] = np.nan


# In[29]:


fig, ax2 = plt.subplots(2, 1, figsize =[18, 15])

## Plotting cumulative capacities
for col in cap_market.columns[[1,2,3,4,7,8]]:
    ax2[0].plot(cap_market['Year'], np.cumsum(cap_market[col]), marker = 'o')
    
ax2[0].set_title('Cumulative development of metrics')
ax2[0].set_ylabel('Metric in 10m')

legend = list(cap_market.columns[[1,2,3,4,7,8]])
ax2[0].legend(legend)

## Zooming in
for col in cap_market.columns[[1,3,4,7,8]]:
    ax2[1].plot(cap_market['Year'], np.cumsum(cap_market[col]), marker = 'o') 

legend = list(cap_market.columns[[1,3,4,7,8]])
ax2[1].legend(legend)
ax2[1].set_title('Cumulative development of metrics (zoomed in)')
ax2[1].set_ylabel('Metric');


# Percentage-wise, total MWh, sales in JP and MW_elec show similar patterns. Other than that, correlation analysis would be needed. \
# JP sales are by far the largest number (cumulatively) and seem to increase linearly, while all other metrics show a very similar pattern with near exponential development in their later years.

# ### 2.2.3. Price data

# In[30]:


## import and merge to one single price dataframe

bnef_prices = pd.read_excel('Sources/MA_Data.xlsx', 'BNEF')

cap_prices = cap.loc[:, ['Year', 'USD2015/kWh_elec', 'USD2015/kW_elec', 'USD2015/kWh_ev', 'USD2015/kW_ev', 
                         'USD2015/kWh', 'USD2015/kWh_elec_2']]
prices = bnef_prices.merge(cap_prices, how = 'outer', on = 'Year')


# In[31]:


## Plot prices in whole period
fig, (ax1, ax2) = plt.subplots(2,1, figsize = [14, 14])

for col in prices.columns[1:]:
    ax1.plot(prices['Year'], prices[col], marker = 'o')
    
ax1.set_ylabel('Price in USD/kWh or USD/kW')
ax1.set_title('Whole Period (1993 - 2020)');
ax1.legend(prices.columns[1:], loc = 'upper right')

## Plot only between 2010 and 2020
for col in prices.columns[1:]:
    ax2.plot(prices.loc[prices['Year'] >= 2010, 'Year'], prices.loc[prices['Year'] >= 2010, col], marker = 'o')

ax2.set_ylabel('Price in USD/kWh')
ax2.set_title('Zoomed in (2010 - 2020)');
ax2.legend(prices.columns[1:], loc = 'upper right')


# ### 2.2.4. EVs

# In[32]:


## Select columns that concern EV
ev = cap.loc[:, ['Year', 'Market_vehicles_millions', 'Cum_GWh_ev', 'Cum_GW_ev', 'USD2015/kWh_ev']]
ev.iloc[:,[2,3]] = ev.iloc[:,[2,3]]*1000
ev.columns = ['Year', 'Market_vehicles_millions', 'Cum_MWh_ev', 'Cum_MW_ev', 'USD2015/kWh_ev']


# In[33]:


fig, (ax1, ax2) = plt.subplots(2,1,figsize = [14, 12])
fig, ax3 = plt.subplots(figsize = [14, 6])

ax1.plot(ev['Year'], ev['Market_vehicles_millions']/1000, marker = 'o')
ax1.plot(ev['Year'], np.cumsum(ev['Market_vehicles_millions']/1000), marker = 'o')
    
for col in ev.columns[2:4]:
    ax2.plot(ev['Year'], ev[col], marker = 'o')
    

ax3.plot(ev['Year'], ev['USD2015/kWh_ev'], marker = 'o')


ax1.set_ylabel('No. of vehicles sold (Tsd)')
ax1.set_title('Electric vehicles, per year and cumulative');
ax1.set_yticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]);

ax2.legend(ev.columns[2:4])
ax2.set_ylabel('Installed capacity in MW(h)')

ax3.set_ylabel('Price in 2015 USD');


# ### 2.2.5. Preparations for final assembly 

# In[34]:


## aggregate all data in one dataframe
li_ion_full = prices
li_ion_full = li_ion_full.merge(cap_market[['Year', 'Cum_MWh_elec', 'Cum_MW_elec', 'Cum_MWh_elec_2', 'Cum_MWh']], how = 'outer', on = 'Year')
li_ion_full = li_ion_full.merge(li_ion_installed, how = 'left', left_on = 'Year', right_on = 'Year commissioned')
li_ion_full = li_ion_full.merge(cap_market.drop('sales_no_jp', axis = 1), how = 'left', on = 'Year')
li_ion_full = li_ion_full.iloc[:, :-6]
li_ion_full = li_ion_full.merge(ev.loc[:, ['Year','Market_vehicles_millions','Cum_MWh_ev']], how = 'left', on = 'Year')
li_ion_full = li_ion_full.drop('Year commissioned', axis = 1)
li_ion_full = li_ion_full.merge(patents.reindex(columns = ['Year', 'Li_ev_Y02T90', 'Li_ion_h01m10']), on = 'Year', how = 'outer')
li_ion_full.columns


# In[35]:


## create dataframe for each application

li_ev = li_ion_full[['Year', 'EV_2020USD/kWh', 'USD2015/kWh_ev', 'weighted_avg_2020USD/kWh', 'Cum_MWh_ev', 'Market_vehicles_millions', 'Li_ev_Y02T90']].sort_values('Year').reset_index().drop('index', axis = 1)
li_ev = li_ev.rename(columns= {'Cum_MWh_ev' : 'MWh_ev'})


li_res = li_ion_full[['Year', 'Consumer_Cells_2019USD/kWh', 'USD2015/kWh_elec', 'USD2015/kW_elec', 
                      'USD2015/kWh', 'USD2015/kWh_elec_2','Cum_MWh_elec_x', 'Cum_MW_elec_x', 'Cum_MWh_x', 
                      'Cum_MWh_elec_2_x', 'Li_ion_h01m10']].sort_values('Year').reset_index().drop('index', axis = 1)
li_res = li_res.rename(columns= {'Cum_MWh_elec_x' : 'MWh_elec',
                       'Cum_MW_elec_x' : 'MW_elec',
                       'Cum_MWh_x' : 'MWh',
                       'Cum_MW_x' : 'MW',
                       'Cum_MWh_elec_2_x' : 'MWh_elec_2'})


# In[36]:


## Combine all prices (adjusted for inflation) to single dataframe
prices_li = pd.DataFrame()
prices_li['Year'] = li_ev['Year']
prices_li['Consumer_Cells_USD2020/kWh'] = adjust_for_inflation(li_res['Consumer_Cells_2019USD/kWh'], base = 2019)
prices_li['EV_USD2020/kWh'] = li_ev['EV_2020USD/kWh']
prices_li['EV_2_USD2020/kWh'] = adjust_for_inflation(li_ev['USD2015/kWh_ev'], base = 2015)


# ## 2.3. Fuel cells

# In[37]:


## import main source. Includes price and production / shipment quantities. See columns below
fc = pd.read_excel('Sources/MA_Data.xlsx', 'Fuel_cells')
fc.columns


# ### 2.3.1. Price

# In[38]:


## Collect fuel cell price columns
fc_price = fc.iloc[:,[0, 3, 6]]

## Plot
fig, ax = plt.subplots(figsize=[14,6])

for col in fc_price.columns[1:]:
    ax.plot(fc_price['Year'], fc_price[col], marker = 'o')

ax.legend(fc_price.columns[1:])
ax.set_title('Price development of fuel cells');
ax.set_ylabel('Price per metric in 2015USD');


# ### 2.3.2. Market
# 

# In[39]:


## Collect data concerning installed volumes
fc_market = fc.drop(['USD2015/kWh_dom', 'USD2015/kW_dom'] , axis = 1)


# In[40]:


fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=[15,15])
ax10 = ax1.twinx()
ax11 = ax2.twinx()
ax12 = ax3.twinx()

## Plot installed capacity in GWh
ax1.bar(fc_market['Year'], fc_market['GWh_dom'], alpha = .5)
ax1.bar(fc_market['Year'], fc_market['GW_dom'], alpha = .5, 
            bottom = fc_market['GWh_dom'])

for (i, col) in enumerate(fc_market.columns[[1, 3]]):   
    ax10.plot(fc_market['Year'], np.cumsum(fc_market[col]), marker = 'o')

    
## Plot shipments in thousands   
for (i, col) in enumerate(fc_market.columns[5:8]):
    ax2.bar(fc_market['Year'], fc_market[col], alpha = .5, 
            bottom = np.sum(fc_market[fc_market.columns[5:5+i]], axis = 1))
    ax11.plot(fc_market['Year'], np.cumsum(fc_market[col]), marker = 'o')
    
ax11.plot(fc_market['Year'], np.cumsum(np.sum(fc_market.iloc[:,5:8], axis = 1)), marker = 'v')


## Plot shipments in MW
for (i, col) in enumerate(fc_market.columns[8:]):
    ax3.bar(fc_market['Year'], fc_market[col], alpha = .5, 
            bottom = np.sum(fc_market[fc_market.columns[8:8+i]], axis = 1))
    ax12.plot(fc_market['Year'], np.cumsum(fc_market[col]), marker = 'o')

ax12.plot(fc_market['Year'], np.cumsum(np.sum(fc_market.iloc[:,8:], axis = 1)), marker = 'v')

ax1.legend(fc_market.columns[[2, 4]])


## legends and titles
legend = list(fc_market.columns[5:8])
legend.append('sum')
ax11.legend(legend)
legend = list(fc_market.columns[8:])
legend.append('sum')
ax12.legend(legend)

ax1.set_title('Market development of fuel cells - Domestic installations');
ax2.set_title('Market development of fuel cells - Shipments in thousand units');
ax3.set_title('Market development of fuel cells - Shipments in MW');


# ### 2.3.3. Preparation for final assembly

# In[41]:


## create dataframe with all points
fc_full = fc_market.drop(['Cum_GWh_dom', 'Cum_GW_dom', 'portable_shipments_tsd', 'stationary_shipments_tsd',
                         'transport_shipments_tsd'], axis = 1)
fc_full.iloc[:, 1:3] = fc_full.iloc[:, 1:3]*1000
fc_full.iloc[:, 1:] = np.cumsum(fc_full.iloc[:, 1:])
fc_full = fc_full.merge(fc_price, how = 'outer', on = 'Year')
fc_full = fc_full.merge(patents.reindex(columns = ['Year', 'Fuel_cell_h01m8']), on = 'Year', how = 'outer')
fc_full = fc_full.rename(columns = {'GWh_dom' : 'MWh_dom',
                         'GW_dom' : 'MW_dom'})
fc_full.columns


# In[42]:


fuel_cells = fc_full.copy()

# Add 2020 row to work with inflation methods
fuel_cells = fuel_cells.append(pd.DataFrame({'Year' : 2020}, index = [len(fuel_cells)]))
fuel_cells = fuel_cells.sort_values('Year')
fuel_cells = fuel_cells.reset_index().drop('index', axis = 1)

# adjust for inflation
fuel_cells['USD2020/kW'] = adjust_for_inflation(fuel_cells['USD2015/kW_dom'], base= 2015)

fuel_cells['shipments_MW'] = np.sum(
    fuel_cells[['stationary_shipments_MW', 'portable_shipments_MW', 'transport_shipments_MW']], axis = 1)

fuel_cells = fuel_cells.drop(['USD2015/kWh_dom','USD2015/kW_dom', 'portable_shipments_MW', 
                              'transport_shipments_MW', 'stationary_shipments_MW'], axis = 1)


# # 3. Final assembly of frames
# 
# Must add lithium prices, electricity prices from the US, the RDD expenditures and patent data. \
# The following are the final dataframes:
# * Li ion for EVs
# * Li ion residential / consumer applications
# * Fuel cells

# In[43]:


## Merge fuel cell datasets
data_fc = fuel_cells.merge(
    us_prices.loc[us_prices['Description'] == 'Average Retail Price of Electricity, Residential', 
                  ['Value', 'Year']], 
    how = 'outer', 
    on = 'Year').rename(columns = {'Value': 'US_elec_price_res'})

data_fc = data_fc.merge(
    rdd_aggregate['Hydrogen and fuel cells'],
    how = 'outer',
    on = 'Year')

data_fc.drop(['MWh_dom', 'shipments_MW'], axis = 1, inplace = True)


# In[44]:


## Merge EV datasets
data_ev = li_ev.merge(us_prices.loc[us_prices['Description'] == 'Residential and Commercial: Mean', ['Value', 'Year']], how = 'left',
                   on = 'Year').rename(columns = {'Value': 'US_elec_price_res_comm'})

data_ev = data_ev.merge(li_price.reindex(columns = ['price_per_ton_battery_grade', 'Year']), 
                        on = 'Year', 
                        how = 'left')

data_ev = data_ev.merge(prices_li.reindex(columns = ['EV_USD2020/kWh', 'EV_2_USD2020/kWh', 'Year']), 
                    how = 'left', 
                    on = 'Year')

data_ev = data_ev.merge(
    rdd_aggregate['EV aggregated'],
    how = 'left',
    on = 'Year')

data_ev['USD2020/kWh'] = np.mean(data_ev[['EV_USD2020/kWh', 'EV_2_USD2020/kWh']], axis = 1)

data_ev.drop(['EV_2020USD/kWh', 'USD2015/kWh_ev', 'weighted_avg_2020USD/kWh', 'Market_vehicles_millions',
             'EV_USD2020/kWh', 'EV_2_USD2020/kWh'], 
             axis = 1, inplace = True)


# In[45]:


## Merge RES datasets
data_res = li_res.merge(us_prices.loc[us_prices['Description'] == 'Average Retail Price of Electricity, Residential',
                                      ['Value', 'Year']], 
                        how = 'left',
                        on = 'Year').rename(columns = {'Value': 'US_elec_price_res'})

data_res = data_res.merge(
                    prices_li.reindex(columns = ['Consumer_Cells_USD2020/kWh', 'Year']), 
                        how = 'left', 
                        on = 'Year')

data_res = data_res.merge(
    rdd_aggregate.reindex(columns = ['Energy storage']),
    how = 'outer',
    on = 'Year')

data_res['MWh_combined'] = np.cumsum(np.mean(data_res[['MWh_elec', 'MWh']], axis = 1))

data_res.drop(['Consumer_Cells_2019USD/kWh', 'USD2015/kWh_elec', 'USD2015/kW_elec',
            'USD2015/kWh', 'USD2015/kWh_elec_2', 'MWh_elec_2', 'MWh_elec', 'MW_elec', 'MWh'], axis = 1, inplace = True)


# In[46]:


with pd.ExcelWriter('Sources/dataframes.xlsx') as writer:
    data_fc.to_excel(writer, sheet_name = 'fuel_cell')
    data_ev.to_excel(writer, sheet_name = 'li_ev')
    data_res.to_excel(writer, sheet_name = 'li_res')


# In[47]:


print(data_fc.columns)
print(data_ev.columns)
print(data_res.columns)


# ## 4. Quantity and price plots

# In[48]:


fig, ax = plt.subplots(figsize = [14, 6])

sns.scatterplot(x = 'MW_dom', y = 'USD2020/kW', data = np.log(data_fc))
sns.scatterplot(x = 'MWh_combined', y = 'Consumer_Cells_USD2020/kWh', data = np.log(data_res))
sns.scatterplot(x = 'MWh_ev', y = 'USD2020/kWh', data = np.log(data_ev))
plt.legend(['Fuel cells', 'Lithium ion', 'EV'])
plt.ylabel('Unit price [USD/kW(h)]');
plt.xlabel('Cumulative installed capacity [MW(h)]')

## Annotations
year = ['2005', '2015', '1995', '2015', '2010', '2016']
pos = [(-1.5, 11.4), (4.67, 9.3), (6.6, 9), (12.1, 5.9), (5.9, 7.2), (10.6, 5.8)]
for i in range(6):
    plt.text(pos[i][0], pos[i][1], year[i])


# In[49]:


data_ev.dropna()


# In[ ]:




