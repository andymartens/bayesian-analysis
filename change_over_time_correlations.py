# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:07:17 2019

@author: AMartens
"""

cd \\chgoldfs\dmhhs\CIDI Staff\Projects\Columbia Capstone\Columbia Capstone 2019\Final Deliverables\For CIDI Draft Final Deliverables 5-15\5. Time Trend Files_5.14.19

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.decomposition import FactorAnalysis
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform  # , model_to_graphviz

pd.set_option("display.max_columns", 10)
sns.set_style('white')


df_2015 = pd.read_excel('time_trends_2015_andy.xlsx')
df_2017 = pd.read_excel('time_trends_2017_andy.xlsx')
df_2019 = pd.read_excel('time_trends_2019_andy.xlsx')
df_indicators_direction = pd.read_excel('indicators_direction_andy.xlsx')

domain_to_direction_dict = dict(zip(df_indicators_direction['Indicator'], df_indicators_direction['Direction']))

domain_to_direction_dict['Housing Cost Burden - Renter (GRAPI)'] = domain_to_direction_dict['Housing Cost Burden - Owner (GRAPI)']
del domain_to_direction_dict['Housing Cost Burden - Owner (GRAPI)']

domain_to_indicator_dict = {'economics':['Household Poverty', 'Income', 'Unemployment Rate'],
                            'health':['Asthma - Current', 'Did Not Get Needed Medical Care', 
                                      'Health Insurance Coverage', 'Late or No Prenatal Care', 'Poor Health',
                                      'Poor Mental Health', 'Preterm Birth', 'Self-Reported Health'],
                            'education':["Bachelor's Degree and Above", 'Chronic Absenteeism', 
                                        'On Time High School Graduation Rate', 'Preschool Enrollment',
                                        'State Test ELA Proficiency', 'State Test Math Proficiency'],
                            'housing':['Noise Complaints', 'Overcrowding Housing',
                                       'Housing Cost Burden - Owner (SMOCAPI)',
                                       'Housing Cost Burden - Renter (GRAPI)'],
                            'safety':['Index Crime Rate ', 'Pedestrain Injuries', 
                                     'Perception Neighborhood Safe'],  # , 'Jail Incarceration'],
                            'infrastructure':['Commute Time', 'Internet Subscription Rate', 
                                              'Pothole Complaints'],
                            'connectedness':['Disconnected Youth', 'Election Voter Turnout Rate', 
                                             'Helpful Neighbor', 'Jail Incarceration']}

df_2015.head()
df_2017.head()
df_2019.head()

df_2015.shape
df_2017.shape
df_2019.shape

df_2015.columns == df_2017.columns 
df_2017.columns == df_2019.columns

df_2015['year'] = 2015
df_2017['year'] = 2017
df_2019['year'] = 2019

df_years = pd.concat([df_2015, df_2017, df_2019], ignore_index=True)
df_years.shape
df_years.head(10)

df_years = df_years.sort_values(by=['NTA', 'year'])
df_years = df_years.reset_index(drop=True)

# remove bad ntas (report removes them)
ntas_to_remove_list = ['BK99', 'BX98', 'BX99', 'MN99', 'QN98', 'QN99', 'SI99']
df_years = df_years[-df_years['NTA'].isin(ntas_to_remove_list)]
df_2015 = df_2015[-df_2015['NTA'].isin(ntas_to_remove_list)]
df_2017 = df_2017[-df_2017['NTA'].isin(ntas_to_remove_list)]
df_2019 = df_2019[-df_2019['NTA'].isin(ntas_to_remove_list)]

len(df_years['NTA'].unique())  #188 - good, matches report
len(df_2019['NTA'].unique())  #188 - good, matches report

def standardize_ivs_using_first_year_mean_and_std(df_first_year, df_second_year, df_third_year, df_all_years, iv_list):
    for var in iv_list:
        if df_first_year[var].mean() < 1000000000000:            
            var_mean = df_first_year[var].mean()
            var_std = df_first_year[var].std() 
        elif df_second_year[var].mean() < 1000000000000:
            var_mean = df_second_year[var].mean()
            var_std = df_second_year[var].std() 
        else:
            var_mean = df_third_year[var].mean()
            var_std = df_third_year[var].std() 
        df_all_years[var] = (df_all_years[var] - var_mean) / var_std
        print(var, np.round(df_all_years[var].mean(),2), np.round(df_all_years[var].std(),2))
    return df_all_years

def standardize_ivs(df_all_years, iv_list):
    for var in iv_list:
        var_mean = df_all_years[var].mean()
        var_std = df_all_years[var].std() 
        df_all_years['z_'+var] = (df_all_years[var] - var_mean) / var_std
        print(var, np.round(df_all_years['z_'+var].mean(),2), np.round(df_all_years['z_'+var].std(),2))
    return df_all_years

# mimic report -- min-max scaling of 2019
def min_max_2019_ivs(df_2019, iv_list):
    for var in iv_list:
        var_min = df_2019[var].min() 
        var_max = df_2019[var].max()
        df_2019[var] = (df_2019[var] - var_min) / (var_max - var_min) * 100
        print(var, np.round(df_2019[var].mean(),2), np.round(df_2019[var].std(),2))
    return df_2019

df_years = standardize_ivs(df_years, df_years.columns[2:-1])
df_years.tail(10)
df_years.iloc[:, 2:4]

indicators_list = [indicator for indicators_in_domain_list in list(domain_to_indicator_dict.values()) for indicator in indicators_in_domain_list]
z_indicators_list = ['z_'+indicator for indicator in indicators_list]

g = sns.PairGrid(df_years[indicators_list[25:]])
g = g.map_diag(plt.hist, edgecolor='w', alpha=.75, color='dodgerblue')
g = g.map_offdiag(plt.scatter, edgecolor='w', alpha=.25, s=40)

# 'Health Insurance Coverage' looks a bit wacky -- big outlier(s)?
# 'Poor Mental Health' -- couple outliers
# 'Noise Complaints' wacky -- big outliers
# 'Index Crime Rate ' -- a few big outliers
# 'Pedestrain Injuries' -- a few big outliers
# 'Internet Subscription Rate' -- one big outlier
# 'Pothole Complaints' -- check, might have a big outlier

df_years['Pothole Complaints'].hist(alpha=.75)
plt.xlabel('Pothole Complaints', fontsize=15)
plt.xticks(fontsize=13)

df_years['Noise Complaints'].hist(alpha=.75)
df_years['Pedestrain Injuries'].hist(alpha=.75)
df_years['Index Crime Rate '].hist(alpha=.75)
df_years['Internet Subscription Rate'].hist(alpha=.75)
df_years['Health Insurance Coverage'].hist(alpha=.75)

# check these. might have to do w change. check distributions
# and corr matrix within each year too.
# 2019
g = sns.PairGrid(df_2019[indicators_list[25:]])
g = g.map_diag(plt.hist, edgecolor='w', alpha=.75, color='dodgerblue')
g = g.map_offdiag(plt.scatter, edgecolor='w', alpha=.25, s=40)

# outliers
# 'Health Insurance Coverage' 
# 'Poor Mental Health'
# 'Noise Complaints'
# 'Index Crime Rate '
# 'Internet Subscription Rate'

# 2017
g = sns.PairGrid(df_2017[[indicators_list[24], indicators_list[25], 
                         indicators_list[26], indicators_list[27],
                         indicators_list[28]]])
g = g.map_diag(plt.hist, edgecolor='w', alpha=.75, color='dodgerblue')
g = g.map_offdiag(plt.scatter, edgecolor='w', alpha=.25, s=40)

# outliers
# 'Health Insurance Coverage' 
# 'Noise Complaints'
# 'Index Crime Rate '
# 'Pedestrain Injuries'

# what to do about outliers on given variables?
# above can see outliers in a given metric in a given year.
# if really extreme, (1) double check. i think the capstone team
# double checked and they're accurate. (2) could shrink them 
# towards mean wtih bayesian or other approach, (3) could remove
# them from analyses. may be good to shrink the outliers and
# do anys and then make sure the anys wouldn't change substantially
# if the outliers are removed.

# What to do about outliers when examining variables from year to year
# i.e., change in a variable in a nta from year to year that's extreme.
# first plot year to year scatter matrices to examine issues.

# shoud i z-score for year to year relationships? and if so, get mean
# and std from the first year? or z-score all-together. think what i
# shouldn't do is z score each year individually. then can't see change
# over time. but should be ok if z-score all together. do this for now
# but do sensitivity test by using first year's mean and std and see
# if any different. 


# make all so that higher is better
for var in z_indicators_list:
    if domain_to_direction_dict[var[2:]] == 'Bad':
        df_years[var] = df_years[var]*-1
    else:
        None

#sns.set(font_scale=1.25)

var = z_indicators_list[8]

# put something else on upper diagonal?

for var in z_indicators_list[:]:
    print(var)
    df_2015 = df_years[df_years['year']==2015][['NTA', var]]
    df_2017 = df_years[df_years['year']==2017][['NTA', var]]
    df_2019 = df_years[df_years['year']==2019][['NTA', var]]
    df_indicator = pd.merge(df_2015, df_2017, on='NTA', how='outer')
    df_indicator = pd.merge(df_indicator, df_2019, on='NTA', how='outer')
    df_indicator.columns = ['NTA', '2015', '2017', '2019']
    df_indicator = df_indicator.reset_index(drop=True)
    for year in ['2015', '2017', '2019']:
        if df_indicator[year].mean() > -1000000000000:
            None
        else:
            del df_indicator[year]
    if len(df_indicator.columns) == 4:
        # plot
        years_list = list(df_indicator.columns[1:])
        min_value = np.nanmin(df_indicator[years_list].values)
        max_value = np.nanmax(df_indicator[years_list].values)
        bin_list = np.linspace(min_value, max_value, 10)
        g = sns.PairGrid(df_indicator[years_list])
        g = g.map_diag(plt.hist, edgecolor='w', alpha=.5, 
                       color='dodgerblue', bins=bin_list)
        g = g.map_upper(plt.scatter, edgecolor='w', alpha=.5, s=40, color='dodgerblue')
        g = g.map_lower(plt.scatter, edgecolor='w', alpha=.5, s=40, color='dodgerblue')
        g.axes[0,1].plot([min_value, max_value], [min_value, max_value], '--', 
                 color='red', alpha=.5)
        g.axes[0,2].plot([min_value, max_value], [min_value, max_value], '--', 
                 color='red', alpha=.5)
        g.axes[1,2].plot([min_value, max_value], [min_value, max_value], '--', 
                 color='red', alpha=.5)
        g.axes[1,0].plot([min_value, max_value], [min_value, max_value], '--', 
                 color='red', alpha=.5)
        g.axes[2,0].plot([min_value, max_value], [min_value, max_value], '--', 
                 color='red', alpha=.5)
        g.axes[2,1].plot([min_value, max_value], [min_value, max_value], '--', 
                 color='red', alpha=.5)
        plt.suptitle(var[2:]+'\n\n', fontsize=15)
        plt.tight_layout()
        print()
        print()
        #g = g.map_offdiag(plt.scatter, edgecolor='w', alpha=.25, s=40)
        #plt.xlim(-2.5,2.5)
        #plt.ylim(-2.5,2.5)
        #plt.plot([min_value, max_value], [min_value, max_value], 
        #         '--', color='red', alpha=.5)
        #g = g.map_lower(plt.plot([min_value, max_value], [min_value, max_value]))
    elif len(df_indicator.columns) == 3:
        years_list = list(df_indicator.columns[1:])
        min_value = np.nanmin(df_indicator[years_list].values)
        max_value = np.nanmax(df_indicator[years_list].values)
        bin_list = np.linspace(min_value, max_value, 10)
        g = sns.PairGrid(df_indicator[years_list])
        g = g.map_diag(plt.hist, edgecolor='w', alpha=.5, 
                       color='dodgerblue', bins=bin_list)
        g = g.map_upper(plt.scatter, edgecolor='w', alpha=.5, s=40, color='dodgerblue')
        plt.plot([min_value, max_value], [min_value, max_value], 
                 '--', color='red', alpha=.5)
        g = g.map_lower(plt.scatter, edgecolor='w', alpha=.5, s=40, color='dodgerblue')
        plt.plot([min_value, max_value], [min_value, max_value], 
                 '--', color='red', alpha=.5)
        plt.suptitle(var[2:]+'\n\n', fontsize=15)
        plt.tight_layout()
        print()
        print()        
    else:
        None




# look up map upper or something to get dashed line in lower left

# examine specific indicator
# and clean
var = 'z_Noise Complaints'
var = 'z_Pothole Complaints'
var = 'z_Pedestrain Injuries'
var = 'z_Index Crime Rate '

len(df_years[df_years['z_Noise Complaints']<-1.5])  
df_years.loc[df_years['z_Noise Complaints']<-1.5, 'z_Noise Complaints'] = np.nan

len(df_years[df_years['z_Pothole Complaints']<-2.5])  
df_years.loc[df_years['z_Pothole Complaints']<-2.5, 'z_Pothole Complaints'] = np.nan

len(df_years[df_years['z_Pedestrain Injuries']<-1.5])  
df_years.loc[df_years['z_Pedestrain Injuries']<-1.5, 'z_Pedestrain Injuries'] = np.nan

len(df_years[df_years['z_Index Crime Rate ']<-4])  
df_years.loc[df_years['z_Index Crime Rate ']<-4, 'z_Index Crime Rate '] = np.nan

df_years.loc[df_years['year']==2017, 'z_Asthma - Current'] = np.nan
var = 'z_Asthma - Current'

df_years.loc[df_years['year']==2017, 'z_Self-Reported Health'] = np.nan
var = 'z_Self-Reported Health'

df_years.loc[df_years['year']==2017, 'z_Preschool Enrollment'] = np.nan
var = 'z_Preschool Enrollment'

df_years.loc[df_years['year']==2017, 'z_Perception Neighborhood Safe'] = np.nan
var = 'z_Perception Neighborhood Safe'

df_years.loc[df_years['year']==2017, 'z_Perception Neighborhood Safe'] = np.nan
var = 'z_Commute Time'

df_years[var]

df_2015 = df_years[df_years['year']==2015][['NTA', var]]
df_2017 = df_years[df_years['year']==2017][['NTA', var]]
df_2019 = df_years[df_years['year']==2019][['NTA', var]]
df_indicator = pd.merge(df_2015, df_2017, on='NTA', how='outer')
df_indicator = pd.merge(df_indicator, df_2019, on='NTA', how='outer')
df_indicator.columns = ['NTA', '2015', '2017', '2019']
df_indicator = df_indicator.reset_index(drop=True)
for year in ['2015', '2017', '2019']:
    if df_indicator[year].mean() > -1000000000000:
        None
    else:
        del df_indicator[year]

years_list = list(df_indicator.columns[1:])

min_value = np.nanmin(df_indicator[years_list].values)
max_value = np.nanmax(df_indicator[years_list].values)
bin_list = np.linspace(min_value, max_value, 10)

plt.hist(df_indicator['2015'], alpha=.5, bins=bin_list, 
         histtype='step', linewidth=5, label='2015', color='black')
plt.hist(df_indicator['2017'], alpha=.7, bins=bin_list-.05, 
         histtype='step', linewidth=5, label='2017', color='dodgerblue')
plt.hist(df_indicator['2019'], alpha=.7, bins=bin_list, 
         histtype='step', linewidth=5, label='2019', color='darkorange')
plt.axvline(df_indicator['2017'].mean(), linestyle='--', color='dodgerblue', alpha=.65)
plt.axvline(df_indicator['2019'].mean(), linestyle='--', color='darkorange', alpha=.75)
plt.legend()

year_1 = '2015'
year_2 = '2017'

year_1 = '2017'
year_2 = '2019'

plt.scatter(df_indicator[year_1], df_indicator[year_2], alpha=.3, color='dodgerblue')
plt.xlabel(year_1)
plt.ylabel(year_2)
plt.plot([np.min(df_indicator[[year_1, year_2]].values),
          np.max(df_indicator[[year_1, year_2]].values)], 
            [np.min(df_indicator[[year_1, year_2]].values),
             np.max(df_indicator[[year_1, year_2]].values)], 
             '--', alpha=.5, color='black')

plt.scatter(df_indicator['2017'], df_indicator['2019'], alpha=.25)

df_indicator[['2017', '2019']].corr()




# ----------------------------------------------------------------------------
# examine change over time.
z_indicators_list

df_corrs = df_years.groupby('NTA')[['z_Household Poverty', 'z_Income']].corr()
df_corrs = df_corrs.reset_index()
df_corrs = df_corrs[df_corrs['level_1']=='z_Income']
df_corrs = df_corrs[['NTA', 'z_Household Poverty']]
df_corrs.columns = ['NTA', 'poverty_income_corr']
np.round(df_corrs.mean().values[0], 3)
plt.hist(df_corrs['poverty_income_corr'], alpha=.5, color='dodgerblue')
plt.axvline(np.round(df_corrs.mean().values[0], 3), linestyle='--', 
            alpha=.75, linewidth=1)

# get list of all corrs
corr_dict = {'x':[], 'y':[], 'corr':[]}
for indicator_x in z_indicators_list[:]:
    for indicator_y in z_indicators_list[:]:
        if indicator_x != indicator_y and not (indicator_x in corr_dict['y'] and indicator_y in corr_dict['x']):
            print(indicator_x, indicator_y)
            df_corrs = df_years.groupby('NTA')[[indicator_x, indicator_y]].corr().reset_index()
            df_corrs = df_corrs[df_corrs['level_1']==indicator_x]
            df_corrs = df_corrs[['NTA', indicator_y]]
            df_corrs.columns = ['NTA', 'corr']
            corr_dict['x'].append(indicator_x)
            corr_dict['y'].append(indicator_y)
            corr_dict['corr'].append(np.round(df_corrs.mean().values[0],3))

        else:
            None

df_xy_over_time_corrs = pd.DataFrame(corr_dict)    
df_xy_over_time_corrs = df_xy_over_time_corrs.sort_values(by='corr')    
df_xy_over_time_corrs = df_xy_over_time_corrs.reset_index(drop=True)

plt.hist(df_xy_over_time_corrs['corr'])

# look at some of these negative corrs in bayesian

# raw corr
df_years[[x_indicator, y_indicator]].corr()
plt.scatter(df_years[x_indicator].values, df_years[y_indicator].values)
sns.lmplot(x=x_indicator, y=y_indicator, data=df_years)
# pos corr w raw data -- makes sense -- ntas in which ps have more perceived
# safety have better mental health. but change over time is opposite, sorta:
# ntas that have an improvement in perceived safety have a worsening in mental health
# why would that be? though might help to break down by nta.

# network based on corrs?
    
# do bayesian anys of above w partial pooling.
# can i do this if only two columns, i.e., two years of data for each variable?

df_years[['z_Household Poverty', 'z_Income']].corr()

df_years[df_years['year']==2015][['z_Household Poverty', 'z_Income']].corr()
df_years[df_years['year']==2017][['z_Household Poverty', 'z_Income']].corr()
df_years[df_years['year']==2019][['z_Household Poverty', 'z_Income']].corr()

df_years_change = df_years.copy(deep=True)
#df_years_change = df_years_change[df_years_change['year']!=2017]
df_years_change[['NTA', 'year', 'z_Household Poverty', 'z_Income']]
df_years_change['poverty_prior'] = df_years_change.groupby('NTA')['z_Household Poverty'].shift(1)
df_years_change['income_prior'] = df_years_change.groupby('NTA')['z_Income'].shift(1)
df_years_change[['year', 'z_Household Poverty', 'poverty_prior', 'z_Income', 'income_prior']]
df_years_change = df_years_change[df_years_change['year']!=2015]
df_years_change['poverty_change'] = df_years_change['z_Household Poverty'] - df_years_change['poverty_prior']
df_years_change['income_change'] = df_years_change['z_Income'] - df_years_change['income_prior']
df_years_change = df_years_change.reset_index(drop=True)
np.round(df_years_change[['poverty_change', 'income_change']].corr().values[0][1], 3)
plt.scatter(df_years_change['poverty_change'], df_years_change['income_change'])
df_years_change = df_years_change[df_years_change['income_change']>-.5]
df_years_change = df_years_change[df_years_change['income_change']<1]

print(np.round(df_years_change[['poverty_change', 'income_change']].corr().values[0][1], 3))
plt.scatter(df_years_change['poverty_change'], df_years_change['income_change'], alpha=.5)
sns.lmplot(x='poverty_change', y='income_change', data=df_years_change, 
           scatter_kws={'alpha':.5})


# just take ntas w three years of data for starters
df_years.groupby('NTA')['z_Household Poverty', 'z_Income'].apply(lambda x: len(x)).value_counts()
np.min(df_corrs)

# parital pooling: varying intercept and slope model

# first plot priors

y_poverty = df_years['z_Household Poverty'].values
x_income = df_years['z_Income'].values
number_of_ntas = len(df_years['NTA'].unique())
# put nta in numerical form
df_nta = pd.DataFrame(df_years['NTA'].unique()).reset_index().rename(columns={0:'nta'})
nta_to_number_dict = dict(zip(df_nta['nta'], df_nta['index']))
df_years['nta_numeric'] = df_years['NTA'].map(nta_to_number_dict)
nta = df_years['nta_numeric'].values

# start w simplest model
with Model() as pooled_model:
    # priors set by me
    a = Normal('a', 0, sd=1)      
    beta = Normal('beta', 0, sd=10)
    sigma = HalfCauchy('sigma', beta=5)
    # linear model
    mu_model = a + beta*x_income
    y = Normal('y', mu_model, sd=sigma, observed=y_poverty)

with pooled_model:
    pooled_model_trace = sample(500, tune=500, chains=4, cores=1)

pm.traceplot(pooled_model_trace[:])
pm.forestplot(pooled_model_trace)


with Model() as unpooled_model:
    a = Normal('a', 0, sd=.5, shape=number_of_ntas)
    beta = Normal('beta', 0, sd=2, shape=number_of_ntas)  # shape=counties here too because give prior for b for each county
    sigma = HalfCauchy('sigma', beta=2)
    mu_model = a[nta] + beta[nta]*x_income
    y = Normal('y', mu_model, sd=sigma, observed=y_poverty)

with unpooled_model:
    unpooled_model_trace = sample(500, tune=500, chains=2, cores=1)

pm.traceplot(pooled_model_trace[:])
pm.forestplot(pooled_model_trace)


# partial pooling w varying intercept and slope
with Model() as varying_intercept_slope:
    # Priors
    mu_a = Normal('mu_a', mu=0, sd=.25)
    sigma_a = HalfCauchy('sigma_a', beta=.25)
    mu_beta = Normal('mu_beta', mu=0, sd=.5)
    sigma_beta = HalfCauchy('sigma_beta', beta=.25)
    # Random intercepts - one adaptive prior for each nta
    a_nta = Normal('a_nta', mu=mu_a, sd=sigma_a, shape=number_of_ntas)
    # Random slopes - one adaptive prior for each nta
    beta_nta = Normal('beta_nta', mu=mu_beta, sd=sigma_beta, shape=number_of_ntas)
    # Model error
    sigma_y = HalfCauchy('sigma_y', beta=2.5)
    # linear model - specifying a diff "a" and diff "beta" for ea nta
    y_hat = a_nta[nta] + beta_nta[nta]*x_income
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=y_poverty)


simulate_priors_varying_intercept_slope = pm.sample_prior_predictive(100, model=varying_intercept_slope)
simulate_priors_varying_intercept_slope.keys()
#x = np.array([0,1])
x = x_income
x = np.linspace(-2.5,2.5,len(x_income))
for i in range(len(simulate_priors_varying_intercept_slope['mu_beta'])):
    plt.plot(x, simulate_priors_varying_intercept_slope['mu_a'][i] + 
             simulate_priors_varying_intercept_slope['mu_beta'][i]*x, 
             color='blue', alpha=.3)
plt.ylim(-2.5,2.5)
plt.xlim(-2.5,2.5)

with varying_intercept_slope:
    varying_intercept_slope_trace = sample(500, tune=500, chains=4, cores=1)

pm.traceplot(varying_intercept_slope_trace[:])
pm.forestplot(varying_intercept_slope_trace)
pm.forestplot(varying_intercept_slope_trace, varnames=['mu_beta'])

plt.figure(figsize=(5,20))
pm.forestplot(varying_intercept_slope_trace, varnames=['beta_nta'])

pm.traceplot(varying_intercept_slope_trace)

pm.forestplot(varying_intercept_slope_trace, varnames=['mu_beta'])

plt.figure(figsize=(5,20))
pm.forestplot(varying_intercept_slope_trace, varnames=['beta_nta'])

# plot regression lines

# sample without replacemnt. about the same as with replacment
sampled_indices_list = np.random.choice(np.arange(len(varying_intercept_slope_trace)), 250, replace=False)  # the rethinking textbook samples the posterior with replacment

varying_intercept_slope_trace[index].keys()

x = np.linspace(-2.5,2.5,len(x_income))
for index in sampled_indices_list:
    plt.plot(x, varying_intercept_slope_trace[index]['mu_a'] + 
             varying_intercept_slope_trace[index]['mu_beta']*x,
             alpha=.015, color='black')
plt.xlim(-2.6,2.6)
plt.ylim(-2.6,2.6)



# parital pooling: varying intercept and slope model
# taking into account the relationship between a and beta in the data 
with Model() as varying_intercept_slope_corr_structure:
    # compute the corr structure in form of chol
    sd_dist = pm.HalfCauchy.dist(beta=5) # 
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=2, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
    # random intercepts and betas, taking into account their corr    
    mu_beta = Normal('mu_beta', mu=0, sd=10)
    mu_a = Normal('mu_a', mu=0, sd=.5)    
    a_and_beta_nta = pm.MvNormal('a_and_beta_nta', mu=(mu_a, mu_beta),  
                                    chol=chol, shape=(number_of_ntas, 2)) 
    a_nta = a_and_beta_nta[:, 0]
    beta_nta = a_and_beta_nta[:, 1]
    # Model error
    sigma_y = HalfCauchy('sigma_y', beta=2.5)
    # linear model - specifying a diff "a" and diff "beta" for ea nta
    y_hat = a_nta[(nta)] + beta_nta[(nta)] * x_income
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=y_poverty)

with varying_intercept_slope_corr_structure:
    varying_intercept_slope_corr_structure = sample(500, tune=500, chains=1, cores=1)




# -----------------------------------------
# code so can add various x and y indicators
z_indicators_list

x_indicator = 'z_Income'   #  "z_Bachelor's Degree and Above"   # 'z_Income'
y_indicator = 'z_Overcrowding Housing'

# just take ntas w three years of data for starters?

df_data = df_years[['NTA', x_indicator, y_indicator, 'year']].dropna()

#df_data = df_data[df_data['year']!=2015]

y_data = df_data[x_indicator].values
x_data = df_data[y_indicator].values
number_of_ntas = len(df_data['NTA'].unique())
# put nta in numerical form
df_nta = pd.DataFrame(df_data['NTA'].unique()).reset_index().rename(columns={0:'nta'})
nta_to_number_dict = dict(zip(df_nta['nta'], df_nta['index']))
df_data['nta_numeric'] = df_data['NTA'].map(nta_to_number_dict)
nta = df_data['nta_numeric'].values

# partial pooling w varying intercept and slope
with Model() as varying_intercept_slope:
    mu_a = Normal('mu_a', mu=0, sd=.25)
    sigma_a = HalfCauchy('sigma_a', beta=.25)
    mu_beta = Normal('mu_beta', mu=0, sd=.5)
    sigma_beta = HalfCauchy('sigma_beta', beta=.25)
    a_nta = Normal('a_nta', mu=mu_a, sd=sigma_a, shape=number_of_ntas)
    beta_nta = Normal('beta_nta', mu=mu_beta, sd=sigma_beta, shape=number_of_ntas)
    sigma_y = HalfCauchy('sigma_y', beta=2.5)
    y_hat = a_nta[nta] + beta_nta[nta]*x_data
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=y_data)

simulate_priors_varying_intercept_slope = pm.sample_prior_predictive(100, model=varying_intercept_slope)
simulate_priors_varying_intercept_slope.keys()
x = x_data
x = np.linspace(-2.5,2.5,len(x_data))
for i in range(len(simulate_priors_varying_intercept_slope['mu_beta'])):
    plt.plot(x, simulate_priors_varying_intercept_slope['mu_a'][i] + 
             simulate_priors_varying_intercept_slope['mu_beta'][i]*x, 
             color='blue', alpha=.25)
plt.ylim(-2.5,2.5)
plt.xlim(-2.5,2.5)

with varying_intercept_slope:
    varying_intercept_slope_trace = sample(500, tune=500, chains=2, cores=1)

pm.traceplot(varying_intercept_slope_trace[:])
pm.forestplot(varying_intercept_slope_trace)
pm.forestplot(varying_intercept_slope_trace, varnames=['mu_beta', 'mu_a'])

plt.figure(figsize=(5,20))
pm.forestplot(varying_intercept_slope_trace, varnames=['beta_nta'])

# sample without replacemnt. about the same as with replacment
sampled_indices_list = np.random.choice(np.arange(len(varying_intercept_slope_trace)), 
                                        250, replace=True)  # the rethinking textbook samples the posterior with replacment

varying_intercept_slope_trace[index].keys()

x = np.linspace(-2.5,2.5,len(x_income))
for index in sampled_indices_list:
    plt.plot(x, varying_intercept_slope_trace[index]['mu_a'] + 
             varying_intercept_slope_trace[index]['mu_beta']*x,
             alpha=.015, color='black')
plt.xlim(-2.6,2.6)
plt.ylim(-2.6,2.6)

plt.hist(varying_intercept_slope_trace['mu_beta'])

beta_list = []
beta_list.append(varying_intercept_slope_trace['mu_beta'])


# lagged corr
df_data['x_lag'] = df_data.groupby('NTA')[x_indicator].shift(1)
df_data['y_lag'] = df_data.groupby('NTA')[y_indicator].shift(1)

df_data_lag = df_data.dropna()

#x_data = df_data_lag[x_indicator].values
y_data = df_data_lag[y_indicator].values
x_lag = df_data_lag['x_lag'].values
y_lag = df_data_lag['y_lag'].values
number_of_ntas = len(df_data_lag['NTA'].unique())
nta = df_data_lag['nta_numeric'].values

# simple lagged model
# this works for now.
with Model() as lagged_simple:
    a = Normal('a', 0, sd=.25)      
    beta_x_lag = Normal('beta_x_lag', 0, sd=.5)
    beta_y_lag = Normal('beta_y_lag', 0, sd=.5)
    sigma = HalfCauchy('sigma', beta=.1)
    mu_model = a + beta_x_lag*x_lag + beta_y_lag*y_lag
    y = Normal('y', mu_model, sd=sigma, observed=y_data)

#with Model() as lagged_simple:
#    a = Normal('a', 0, sd=.25)      
#    beta_x_lag = Normal('beta_x_lag', 0, sd=.5)
#    sigma = HalfCauchy('sigma', beta=.1)
#    mu_model = a + beta_x_lag*x_lag
#    y = Normal('y', mu_model, sd=sigma, observed=y_data)

simulate_priors = pm.sample_prior_predictive(100, model=lagged_simple)
simulate_priors.keys()
x = np.linspace(-2.5,2.5,len(x_data))
for i in range(len(simulate_priors['beta_x_lag'])):
    plt.plot(x, simulate_priors['a'][i] + 
             simulate_priors['beta_x_lag'][i]*x, 
             color='blue', alpha=.25)
plt.ylim(-2.5,2.5)
plt.xlim(-2.5,2.5)

with lagged_simple:
    lagged_simple_trace = sample(500, tune=500, chains=2, cores=1)

pm.traceplot(lagged_simple_trace[:])
pm.forestplot(lagged_simple_trace)


# partial pooling w varying intercept and slope
with Model() as lagged_multilevel:
    mu_a = Normal('mu_a', mu=0, sd=.25)
    sigma_a = HalfCauchy('sigma_a', beta=.1)
    mu_beta_x_lag = Normal('mu_beta_x_lag', mu=0, sd=.1)
    mu_beta_y_lag = Normal('mu_beta_y_lag', mu=0, sd=.1)
    sigma_beta = HalfCauchy('sigma_beta', beta=.1)
    a_nta = Normal('a_nta', mu=mu_a, sd=sigma_a, shape=number_of_ntas)
    beta_x_lag_nta = Normal('beta_x_lag_nta', mu=mu_beta_x_lag, sd=sigma_beta, shape=number_of_ntas)
    beta_y_lag_nta = Normal('beta_y_lag_nta', mu=mu_beta_y_lag, sd=sigma_beta, shape=number_of_ntas)
    sigma_y = HalfCauchy('sigma_y', beta=.25)
    y_hat = a_nta[nta] + beta_x_lag_nta[nta]*x_lag + beta_y_lag_nta[nta]*y_lag
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=y_data)

simulate_priors = pm.sample_prior_predictive(100, model=lagged_multilevel)
simulate_priors.keys()
x = np.linspace(-2.5,2.5,len(x_data))
for i in range(len(simulate_priors['beta_x_lag_nta'])):
    plt.plot(x, simulate_priors['mu_a'][i] + 
             simulate_priors['beta_x_lag_nta'][i]*x, 
             color='blue', alpha=.25)
plt.ylim(-2.5,2.5)
plt.xlim(-2.5,2.5)

with lagged_multilevel:
    lagged_multilevel_trace = sample(500, tune=500, chains=2, cores=1)

pm.traceplot(lagged_multilevel_trace[:])
pm.forestplot(lagged_multilevel_trace,  varnames=['mu_beta_x_lag', 'mu_beta_y_lag', 'mu_a'])



# corr between ntas
y_data = df_data[x_indicator].values
x_data = df_data[y_indicator].values
number_of_ntas = len(df_data['NTA'].unique())
# put nta in numerical form
df_nta = pd.DataFrame(df_data['NTA'].unique()).reset_index().rename(columns={0:'nta'})
nta_to_number_dict = dict(zip(df_nta['nta'], df_nta['index']))
df_data['nta_numeric'] = df_data['NTA'].map(nta_to_number_dict)
nta = df_data['nta_numeric'].values

with Model() as between_ntas:
    a = Normal('a', 0, sd=.25)      
    beta = Normal('beta', 0, sd=.5)
    sigma = HalfCauchy('sigma', beta=.1)
    mu_model = a + beta*x_data
    y = Normal('y', mu_model, sd=sigma, observed=y_data)

simulate_priors_between_ntas = pm.sample_prior_predictive(100, model=between_ntas)
simulate_priors_between_ntas.keys()
x = x_data
x = np.linspace(-2.5,2.5,len(x_data))
for i in range(len(simulate_priors_between_ntas['beta'])):
    plt.plot(x, simulate_priors_between_ntas['a'][i] + 
             simulate_priors_between_ntas['beta'][i]*x, 
             color='blue', alpha=.25)
plt.ylim(-2.5,2.5)
plt.xlim(-2.5,2.5)

with between_ntas:
    between_ntas_trace = sample(500, tune=500, chains=2, cores=1)

pm.traceplot(between_ntas_trace[:])
pm.forestplot(between_ntas_trace)


# pooled and lagged
#with Model() as between_ntas:
#    a = Normal('a', 0, sd=.25)      
#    beta = Normal('beta', 0, sd=.5)
#    sigma = HalfCauchy('sigma', beta=.1)
#    mu_model = a + beta*x_data
#    y = Normal('y', mu_model, sd=sigma, observed=y_data)




# ----------------
# ----------------
# replicate report
df_2019_min_max_scaled = min_max_2019_ivs(df_2019, df_years.columns[2:-1])
df_2019_min_max_scaled = df_2019_min_max_scaled.reset_index(drop=True)
#df_2019_min_max_scaled = df_2019_min_max_scaled.dropna()

for col in df_2019_min_max_scaled.columns:
    print(col, len(df_2019_min_max_scaled[df_2019_min_max_scaled[col].isnull()]))

df_2019_min_max_scaled[df_2019_min_max_scaled['Asthma - Current'].isnull()][['NTA', 'Asthma - Current']]
df_2019_min_max_scaled[df_2019_min_max_scaled['NTA'].isin(['BX22', 'BX29', 'QN05', 'QN33', 'QN43', 'QN44'])]
# are these really missing? look. yes. what do they do in report w missing?

for domain in domain_to_indicator_dict.keys():  
    df_2019_min_max_scaled[domain] = df_2019_min_max_scaled[domain_to_indicator_dict[domain]].mean(axis=1)

df_2019_domains = df_2019_min_max_scaled[list(domain_to_indicator_dict.keys())]
df_2019_domains.corr()
# --------------------

# variable will have all nulls if only from 2019. because can't look at change in these cases.
df_years['Election Voter Turnout Rate']

# reverse code
for var in df_years.columns[2:-1]:
    if domain_to_direction_dict[var] == 'Bad':
        df_years[var] = df_years[var]*-1
    else:
        None

for domain in domain_to_indicator_dict.keys():  
    df_years[domain] = df_years[domain_to_indicator_dict[domain]].mean(axis=1)

df_years['safety'].mean()
df_2019_domains = df_years[list(domain_to_indicator_dict.keys())+['year']]
df_2019_domains = df_2019_domains[df_2019_domains['year']==2019]
df_2019_domains = df_2019_domains[list(domain_to_indicator_dict.keys())]
df_2019_domains = df_2019_domains.reset_index(drop=True)
#del df_2019_domains['connectedness']
df_2019_domains = df_2019_domains.dropna()

sns.clustermap(df_2019_domains.transpose(), row_cluster=True, 
               col_cluster=False, xticklabels=False)

# would these clustere emerge if using individual measure too? 
# and in factor anys? can try and try cluster anys w 7 clusters

fa_stand = FactorAnalysis(n_components=2).fit(df_2019_domains)
df_fa_results = pd.DataFrame(fa_stand.components_, columns=df_2019_domains.columns)
df_fa_results[(df_fa_results < .2) & (df_fa_results > 0)] = np.nan
df_fa_results[(df_fa_results > -.2) & (df_fa_results < 0)] = np.nan
df_fa_transposed = df_fa_results.transpose()
df_fa_transposed.round(2)
#                   0     1
#economics       0.75   NaN
#health          0.57   NaN
#education       0.73   NaN
#housing         0.40   NaN
#safety          0.24 -0.29
#infrastructure  0.36  0.29
#connectedness   1.28 -0.50

df_2019_domains.corr()

# clustermap for actual standardized values of measures
# but that's clustering of actual values, not change in values.

# which one's change together?

#col = 'Household Poverty'
individual_indicators_list = df_years.columns[2:-8]
for col in individual_indicators_list:
    df_years[col+' prior'] = df_years.groupby('NTA')[col].shift(1)
    df_years[col+' change'] = df_years[col] - df_years[col+' prior']

change_columns_list = []
for col in df_years.columns:
    if 'change' in col:
        change_columns_list.append(col)
    else:
        None

len(change_columns_list)

df_corr = df_years[change_columns_list].corr().replace(np.nan, 0)
g = sns.clustermap(df_corr)

df_corr = df_corr.abs()
df_corr.shape
df_corr.head()

df_corr.columns

g.data2d

cols_w_all_nulls = ['Helpful Neighbor change',
'Election Voter Turnout Rate change',        
'Disconnected Youth change',                             
'Internet Subscription Rate change',                                   
'On Time High School Graduation Rate change',                           
'Chronic Absenteeism change',                                                     
'Late or No Prenatal Care change',                                                     
'Preterm Birth change']                                               

change_columns_list_filtered = [col for col in change_columns_list if col not in cols_w_all_nulls]
len(change_columns_list_filtered)

df_corr = df_years[change_columns_list_filtered].corr().replace(np.nan, 0)
#df_corr = df_corr.abs()
g = sns.clustermap(df_corr)

for col in df_corr.columns:
    df_corr_col = df_corr[col].sort_values(ascending=False)
    df_corr_col = df_corr_col[df_corr_col>.6]
    print()
    print(df_corr_col.round(2))

# how to deal with removing the obvious ones? or ok that they're there?
# could also just use the aggregated domains and see how change is corr
# vs raw corrs on 2019 indicators.

df_years[['Jail Incarceration', 'Perception Neighborhood Safe']].corr()  # this doesn't match the report
df_years[['Jail Incarceration change', 'Perception Neighborhood Safe change']].corr()

df_2019_min_max_scaled[['Jail Incarceration', 'Perception Neighborhood Safe']].corr()  # this doesn't match the report
# this matches report. but i reverse coded mine. 
# so less jail incarceration corr w more perceived neighbohood safety. 
# makes sense -- ntas w lower incarceration rates tend to be low crime
# and better off and so also have more perceived safety.

# but change scores show the oppostie. drops in jail incarceration in a
# nta are associated/corr w drops in perceived neighbohood safety.

df_2019_min_max_scaled[['Poor Mental Health', 'Self-Reported Health', 'Preterm Birth']].corr()  # this doesn't match the report
#                      Poor Mental Health  
#Poor Mental Health              1.000000             
#Self-Reported Health            0.400248              
#Preterm Birth                   0.406236              
# these don't replicate report at all. huh. though would this be diff if get
# entire corr matrix and so n is different?
d_corr_test = df_2019_min_max_scaled[individual_indicators_list].corr()
d_corr_test['Poor Mental Health']
#Preterm Birth                            0.406
#Self-Reported Health                     0.400
# the same as above when corr individually. 
# but diff than the rreport in which the corr is about half, .2 to .3

# what id i just do a raw corr on the actual data
df_raw_corr_2019 = pd.read_excel('time_trends_2019_andy.xlsx')
df_raw_corr_2019 = df_raw_corr_2019[-df_raw_corr_2019['NTA'].isin(ntas_to_remove_list)]
df_raw_corr_2019[['Poor Mental Health', 'Self-Reported Health', 'Preterm Birth']].corr()
df_raw_corr_2019[['Self-Reported Health', 'Preterm Birth']].corr()
d_raw_corr_test = df_raw_corr_2019[individual_indicators_list].corr()
d_raw_corr_test['Poor Mental Health']
d_raw_corr_test['Preterm Birth']

# nope -- same w just the raw corrs. why so diff?
# create a big matrix w just the diagonals to compare w theirs.

plt.scatter(df_raw_corr_2019['Preterm Birth'], df_raw_corr_2019['Self-Reported Health'])
sns.lmplot(x='Preterm Birth', y='Self-Reported Health', data=df_raw_corr_2019)

df_test = pd.read_excel('wellbeing_tests_andy.xlsx')
df_test.corr()

for col in df_test.columns:
    print(col, len(df_test[df_test[col].isnull()]))

sns.lmplot(x='Preterm_Birth', y='tt_self_reported_health', data=df_test)
sns.lmplot(x='Preterm_Birth', y='Self_Reported_Health', data=df_test)
# crazy 0 outlier

sns.lmplot(x='Preterm Birth', y='Self-Reported Health', data=df_raw_corr_2019)
sns.lmplot(x='tt_preterm_birth', y='tt_self_reported_health', data=df_test)
# the diff between these two is that that the df_test is counting 0s for 
# ntas that should be taken out because they're cemetaries, etc.

df_test = df_test[-df_test['NTA'].isin(ntas_to_remove_list)]
df_test.corr()



































