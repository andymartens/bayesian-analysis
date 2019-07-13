# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:51:58 2019

@author: AMartens

Bayesian Analysis - starter code
"""


#cd /Users/charlesmartens/Google Drive/Code and Tools/bayesian

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

import pymc3 as pm
from pymc3 import get_data
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform  # , model_to_graphviz
from pymc3 import __version__
print('Running on PyMC3 v{}'.format(__version__))

#from pymc3 import *
#import pymc3 as pm
#from scipy.stats import multivariate_normal
#from scipy.linalg import cholesky
#from scipy.stats import pearsonr
#from scipy import stats
#from theano import tensor as tt


# ---------------------------------------------------------------------------
# bayesian anys

# we want to map  x to y. so we have a model with paramters to do this. 
# it's a linear model: mu = intercept + x_coeff * x, and this produces 
# the outcome/y with error, normally distributed. then we see how well 
# each combo of uknown paramters fits or maps to or produces the y/dv. 
# this is the definition of the likelihood. so the likelihood is a 
# distribution of the number of ways that each combo of paramters would 
# produce the dv/y. and we specify here the actual data for y with observed=y.

# "Every probabilistic program consists of observed and unobserved Random Variables (RVs). 
# Observed RVs are defined via likelihood distributions, while unobserved RVs are defined 
# via prior distributions."

# Every unobserved RV has the following calling signature: name (str), parameter 
# keyword arguments. Thus, a normal prior can be defined in a model context like 
# this, setting the unobserved variable to some function that's the prior. 
# The first argument is always the name of the random variable, which should 
# almost always match the name of the Python variable being assigned to, 
# since it is sometimes used to retrieve the variable from the model for 
# summarizing output. 

#with pm.Model():
#    x = pm.Normal('x', mu=0, sd=1)

# Observed RVs are defined just like unobserved RVs but require data to be 
# passed into the observed keyword argument. i.e., we set the observed variable
# to some function we want to model it. observed supports lists, numpy.ndarray, 
# theano and pandas data structures.
#with pm.Model():
#    obs = pm.Normal('x', mu=0, sd=1, observed=np.random.randn(100))

# The main entry point to MCMC sampling algorithms is via the pm.sample() 
# function. By default, this function tries to auto-assign the right sampler(s) 
# and auto-initialize if you don’t pass anything.

# multi-level model
# we're modeling - eg linear model - for each cluster.
# but could do this but model each separately with no input from other clusters
# when computing estimate, i.e., w no pooling. these are "index" variables, 
# then, using the rethinking stats lingo. but better if pool -- actually, if
# partially pool. Complete pooling is when not actually modeling each cluster
# but instead ignoreing cluster alltogether. to partially pool, set the prior
# of the cluster paramters to the distribution of cluster paramters. then 
# have a fixed prior for this distribution of cluster paramters. examples:

# regular total-pooled (i.e., not taking into consideration clusters)
# y = a + mx
# prior for a = Normal(0, 10)
# prior for b = Normal(0, 10)

# unpooled taking into consideration clusters.
# here we're modeling each cluster, getting paramteres a and b for each cluster
# and they're being informated by a prior that I give it. but not being informed
# by the data from the other clusters.
# linear model: y = a[cluster] + b[cluster] * x
# prior for a[cluster] = Normal(0, 10)
# prior for b[cluster] = Normal(0, 10)

# partially pooled - i.e., taking into consideration clusters
# modeling each cluster's paramters a and b and they're being informed both
# by the data from the other clusters and by the prior that i give it. this
# is also referred to as using "adaptive priors" in rethinking stats lingo.
# "A partial pooling model represents a compromise between the pooled and 
# unpooled extremes, approximately a weighted average (based on sample size) 
# of the unpooled county estimates and the pooled estimates.
# y = a[cluster] + b[cluster] * x
# adaptive prior for a[cluster] = Normal(a_bar, a_sigma_bar)
# adaptive prior for b[cluster] = Normal(b_bar, b_sigma_bar)
# prior for a_bar
# prior for a_sigma_bar
# prior for b_bar
# prior for b_sigma_bar

# partially pooled taking into account the relationship between a and b in the data 
# by drawin on the variance-covariance matrix for the parameters a and b
# so here the adaptive prior uses this, draws from this? 




# -----------------------------------------------------------------------------
# Import radon data
srrs2 = pd.read_csv(get_data('srrs2.dat'))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state=='MN'].copy()
srrs_mn.shape
srrs_mn.head()

srrs_mn['fips'] = srrs_mn.stfips*1000 + srrs_mn.cntyfips
cty = pd.read_csv(get_data('cty.dat'))
cty_mn = cty[cty.st=='MN'].copy()
cty_mn['fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
u = np.log(srrs_mn.Uppm)
n = len(srrs_mn)

srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

# create local copies of variables.
# pretty sure the counties index/cluster variable needs to be in numeric form (not string0)
county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values

# Distribution of radon levels in MN (log scale):
srrs_mn.activity.apply(lambda x: np.log(x+0.1)).hist(bins=25);
srrs_mn.activity.hist(bins=25);



# complete pooling model
floor = srrs_mn.floor.values
log_radon = srrs_mn.log_radon.values

with Model() as pooled_model:
    # priors set by me
    a = Normal('a', 0, sd=10)      
    beta = Normal('beta', 0, sd=10)
    sigma = HalfCauchy('sigma', 5)
    # linear model
    mu_model = a + beta*floor
    y = Normal('y', mu_model, sd=sigma, observed=log_radon)

with pooled_model:
    pooled_trace = sample(500, tune=500, cores=1)

pm.traceplot(pooled_trace[:])
pm.forestplot(pooled_trace)
# floor value 0=basement, 1=first-floor. beta is negative means less radon on 
# first floor than basement

# examine the regression lines that the priors would predict if no data.
help(pm.sample_prior_predictive(100, model=pooled_model))
simulate_priors = pm.sample_prior_predictive(100, model=pooled_model)
simulate_priors.keys()
plt.hist(simulate_priors['beta'], alpha=.5)
plt.hist(simulate_priors['a'], alpha=.5)
plt.hist(simulate_priors['sigma'], alpha=.5)
plt.hist(simulate_priors['sigma_log__'], alpha=.5)

#i = 1
#x = np.linspace(0, 1, 100)
x = np.array([0,1])
for i in range(len(simulate_priors['beta'])):
    plt.plot(x, simulate_priors['a'][i] + simulate_priors['beta'][i]*x, 
             color='blue', alpha=.3)

# standardize the variables and plot on scale -2 and +2 SDs to make 
# sure the priors don't produce crazy results.
# standardize the dv:
log_radon_std = [(lr-np.mean(log_radon))/np.std(log_radon) for lr in log_radon]    
plt.hist(log_radon_std)
#x_std = [(f-np.mean(floor))/np.std(floor) for f in floor]   
#plt.hist(x_std)

# build model again but with standardized dv
with Model() as pooled_model_std:
    a = Normal('a', 0, sd=.5)  # can see sd=10 produces outrageous answers    
    beta = Normal('beta', 0, sd=1)  # sd=10
    sigma = HalfCauchy('sigma', 2.5)
    mu_model = a + beta*floor
    y = Normal('y', mu_model, sd=sigma, observed=log_radon_std)
    #pooled_trace_std = sample(500, tune=500, cores=1)  # don't need this to sample the prior

# textbook suggests that at least for continuous variables centered at 0
# because they're standardized, we can set the prior for the intercept 
# pretty tightly around 0. e.g., w sd=.2 or something like that.
simulate_priors_std = pm.sample_prior_predictive(100, model=pooled_model_std)
x = np.array([0,1])
for i in range(len(simulate_priors_std['beta'])):
    plt.plot(x, simulate_priors_std['a'][i] + simulate_priors_std['beta'][i]*x, 
             color='blue', alpha=.3)
plt.ylim(-2.5,2.5)

# how to do for multivariate regression?

# see if maping the slopes looks the same as if i did it manually
# with a simple model.

dir(pooled_model)
pooled_model.sample_prior()
pooled_trace.sample_prior()

with Model() as pooled_model:
    # priors set by me
    a = Normal('a', 0, sd=10)      
    beta = Normal('beta', 0, sd=10)
    sigma = HalfCauchy('sigma', 5)
    # linear model
    mu_model = a + beta*floor
    y = Normal('y', mu_model, sd=sigma, observed=log_radon)
    prior = pm.sample_prior_predictive(100)

help(pm.sample_prior_predictive)


# take cluster -- i.e., county -- into account. unpooling model
# this is also just including a categorial variables as an index variable
# and it'll produce an alpha for each level of the categorical variable (gender, county, whatever)
# and that alpha is the estimate for the mean of that level of the category.
with Model() as unpooled_model:
    # priors set by me
    # because modeling a for each counry, give prior for each county, i.e., shape=counties
    # because really these are different "a" paramters, one for each county.
    # shouldn't this look like: a[county] = Normal('a', 0, sd=10) would that 
    # work the same? it doesn't.
    a = Normal('a', 0, sd=10, shape=counties)  
    beta = Normal('beta', 0, sd=10)
    sigma = HalfCauchy('sigma', 5)
    # linear model
    mu_model = a[county] + beta*floor  # these are really many different "a"s --
    # one for each county. so don't read as there is only one a value -- because
    # [county] is saying we want "a" (i.e., intercept) for each county
    y = Normal('y', mu_model, sd=sigma, observed=log_radon)

with unpooled_model:
    unpooled_trace = sample(500, tune=500, cores=1)

pm.traceplot(unpooled_trace[:])
pm.forestplot(unpooled_trace)


# couldn't i also model beta[county] to get different unpooled betas for each county? 
# yes, and that's the code in R/Stan. But here specify shape = # of counties
# and that produces a prior for each county instead of beta[county]
with Model() as unpooled_model_2:
    # priors set by me
    a = Normal('a', 0, sd=10, shape=counties)
    beta = Normal('beta', 0, sd=10, shape=counties)  # shape=counties here too because give prior for b for each county
    sigma = HalfCauchy('sigma', 5)
    # linear model
    mu_model = a[county] + beta[county]*floor
    y = Normal('y', mu_model, sd=sigma, observed=log_radon)

with unpooled_model_2:
    unpooled_trace_2 = sample(500, tune=500, cores=1)

pm.traceplot(unpooled_trace_2[:])
pm.forestplot(unpooled_trace_2)

plt.figure(figsize=(6,14))
pm.forestplot(unpooled_trace_2, varnames=['beta'])

unpooled_beta_estimates = pd.Series(unpooled_trace_2['beta'].mean(axis=0), index=mn_counties)
unpooled_beta_se = pd.Series(unpooled_trace_2['beta'].std(axis=0), index=mn_counties)
beta_order = unpooled_beta_estimates.sort_values().index

plt.scatter(range(len(unpooled_beta_estimates)), unpooled_beta_estimates[beta_order])
for i, m, se in zip(range(len(unpooled_beta_estimates)), unpooled_beta_estimates[beta_order], unpooled_beta_se[beta_order]):
    plt.plot([i,i], [m-se, m+se], 'b-')
plt.ylabel('Radon estimate')
plt.xlabel('Ordered county')
plt.xlim(-1,86)
plt.ylim(-5,4)

unpooled_a_estimates = pd.Series(unpooled_trace_2['a'].mean(axis=0), index=mn_counties)
unpooled_a_se = pd.Series(unpooled_trace_2['a'].std(axis=0), index=mn_counties)
a_order = unpooled_a_estimates.sort_values().index

plt.scatter(range(len(unpooled_a_estimates)), unpooled_a_estimates[a_order])
for i, m, se in zip(range(len(unpooled_a_estimates)), unpooled_a_estimates[a_order], unpooled_a_se[a_order]):
    plt.plot([i,i], [m-se, m+se], 'b-')
plt.xlim(-1,86); 
plt.ylim(-.5,4)
plt.ylabel('Radon estimate');
plt.xlabel('Ordered county');

# q: what happens if i just use radon levels but not logged? understand why 
# loggig here? notes about making computationally easier -- scaling and 
# addition vs. multiplication?
    
# parital pooling: varying intercept model
with Model() as varying_intercept:
    # Priors
    mu_a = Normal('mu_a', mu=0., tau=0.0001)
    sigma_a = HalfCauchy('sigma_a', 5)
    # Random intercepts
    # i don't understand how it does that mu_a and sigma_a represents 
    # the distribution of cluster means. where is that specified?
    a = Normal('a', mu=mu_a, sd=sigma_a, shape=counties)
    # Common slope
    beta = Normal('beta', mu=0., sd=1e5)
    # Model error
    sd_y = HalfCauchy('sd_y', 5)
    # linear model 
    y_hat = a[county] + beta * floor_measure
    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sd=sd_y, observed=log_radon)

with varying_intercept:
    varying_intercept_trace = sample(500, tune=500)


pm.traceplot(varying_intercept_trace[:])
pm.forestplot(varying_intercept_trace, varnames=['a'])
pm.forestplot(varying_intercept_trace, varnames=['beta'])

# The estimate for the floor coefficient is approximately -0.66, which can be 
# interpreted as houses without basements having about half (exp(−0.66)=0.52) 
# the radon levels of those with basements, after accounting for county.
# q: look into this more. this exponentiating has to do with log-odds. i 
# didn't specify log odds anywhere in model but does it do this automatically?

xvals = np.arange(2)
bp = varying_intercept_trace['a'].mean(axis=0)
mp = varying_intercept_trace['beta'].mean()
for bi in bp:
    plt.plot(xvals, mp*xvals + bi, 'bo-', alpha=0.25)
plt.xlim(-0.1,1.1);
# same betas, different intercepts for each county
# and these intercepts should be more conserative than from a model w/out partial 
# pooling (i.e., adaptive priors), especially for countries with fewer cases in the data.
  

# parital pooling: varying intercept and slope model
with Model() as varying_intercept_slope:
    # Priors
    mu_a = Normal('mu_a', mu=0., sd=10)
    sigma_a = HalfCauchy('sigma_a', beta=5)
    mu_beta = Normal('mu_beta', mu=0., sd=10)
    sigma_beta = HalfCauchy('sigma_beta', beta=2.5)
    # Random intercepts - one adaptive prior for each county
    a_county = Normal('a_county', mu=mu_a, sd=sigma_a, shape=counties)
    # Random slopes - one adaptive prior for each county
    beta_county = Normal('beta_county', mu=mu_beta, sd=sigma_beta, shape=counties)
    # Model error
    sigma_y = Uniform('sigma_y', lower=0, upper=100)
    # linear model - specifying a diff "a" and diff "beta" for ea county
    y_hat = a_county[county] + beta_county[county] * floor_measure
    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)

with varying_intercept_slope:
    varying_intercept_slope_trace = sample(500, tune=500)

pm.traceplot(varying_intercept_slope_trace[:])
pm.forestplot(varying_intercept_slope_trace, varnames=['beta_county'])
pm.forestplot(varying_intercept_slope_trace, varnames=['mu_beta'])

xvals = np.arange(2)
b = varying_intercept_slope_trace['a_county'].mean(axis=0)
m = varying_intercept_slope_trace['beta_county'].mean(axis=0)
for bi, mi in zip(b, m):
    plt.plot(xvals, mi*xvals + bi, 'bo-', alpha=0.25)
plt.xlim(-0.1, 1.1);


# parital pooling: varying intercept and slope model
# taking into account the relationship between a and beta in the data 

with Model() as varying_intercept_slope_corr_structure:
    # compute the corr structure in form of chol
    sd_dist = pm.HalfCauchy.dist(beta=2) # This is the same as sigma_cafe ~ dcauchy(0,2)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=2, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
    #cov = tt.dot(chol, chol.T)
    
    # Extract the standard deviations and rho
    # not sure if need these three lines?
    #sigma_ab = pm.Deterministic('sigma_cafe', tt.sqrt(tt.diag(cov)))
    #corr = tt.diag(sigma_ab**-1).dot(cov.dot(tt.diag(sigma_ab**-1)))
    #r = pm.Deterministic('Rho', corr[np.triu_indices(2, k=1)])

    # random intercepts and betas, taking into account their corr
    # shape needs to be (counties, 2) because we're getting back both a and beta for each county
    # in R: c(a_cafe,b_cafe)[cafe] ~ multi_normal( c(a,b) , Rho , sigma_cafe ),
    # in pymc3 nb: mu_a_and_beta = pm.Normal('mu_a_and_beta', mu=0, sd=10, shape=2)  # prior for average intercept and slope
    
    mu_beta = Normal('mu_beta', mu=0., sd=10)
    mu_a = Normal('mu_a', mu=0., sd=10)    
    a_and_beta_county = pm.MvNormal('a_and_beta_county', mu=(mu_a, mu_beta),  # [mu_a, mu_beta] works too
                                    chol=chol, shape=(counties, 2)) 
    # the shape of a_and_beta_county.shape is in the form of array([10,  2])
    # as coded outside this with statement. so to get each distribution:
    a_county = a_and_beta_county[:, 0]
    beta_county = a_and_beta_county[:, 1]
    # but once wrap up a and beta, or any 2+ variables into a cov matrix
    # they're two connected paramters for the sampler to figure out together it seems

    # so instead of chol=chol, could put the covariance matrix? with cov=cov
    # but doc string says "Most of the time it is preferable to specify the 
    # cholesky factor of the covariance instead."

    # Model error
    sigma_y = Uniform('sigma_y', lower=0, upper=100)  # prior sd within county
    # linear model - specifying a diff "a" and diff "beta" for ea county
    # doesn't throw warning messages when put county in (), i.e., (county)
    y_hat = a_county[(county)] + beta_county[(county)] * floor_measure
    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)

with varying_intercept_slope_corr_structure:
    varying_intercept_slope_corr_structure_trace = sample(500, tune=500, cores=1)
    # need to set cores=1 for this. stuff online too about issues in ipy nb
    # with multiple cores.
    
pm.traceplot(varying_intercept_slope_corr_structure_trace[:])
pm.forestplot(varying_intercept_slope_corr_structure_trace, varnames=['mu_beta'])
pm.forestplot(varying_intercept_slope_corr_structure_trace, varnames=['mu_a'])
pm.forestplot(varying_intercept_slope_corr_structure_trace, varnames=['a_and_beta_county'])
varying_intercept_slope_corr_structure_trace.varnames

plt.figure(figsize=(15, 20))
pm.forestplot(varying_intercept_slope_corr_structure_trace, varnames=['a_and_beta_county'])

df_trace = pm.trace_to_dataframe(varying_intercept_slope_corr_structure_trace)
df_trace.columns
df_trace[['mu_beta', 'mu_a']].mean()
#mu_a       1.488
#mu_beta   -0.649

# this was able to sample from the posterior - though took a lot longer
# than other models. not sure if there's anything i could do to speed it - 
# could check pymc3 for tips on making more efficient (in the same way 
# that stan-R has things to do to speed it, like "refactoring" or something.)

mu = np.zeros(2)

dir(pm.MvNormal)
help(pm.MvNormal)

# check to see diff in slopes when i take into corr w intercept w cov 
# matrix vs when i do not.
with Model() as varying_intercept_slope:
    mu_a = Normal('mu_a', mu=0., sd=10)    
    mu_beta = Normal('mu_beta', mu=0., sd=10)
    sigma_a = HalfCauchy('sigma_a', beta=5)
    sigma_beta = HalfCauchy('sigma_beta', beta=5)
    a_county = pm.Normal('a_county', mu=mu_a, sd=sigma_a, shape=counties)
    beta_county = pm.Normal('beta_county', mu=mu_beta, sd=sigma_beta, shape=counties)
    sigma_y = Uniform('sigma_y', lower=0, upper=100)  
    y_hat = a_county[(county)] + beta_county[(county)] * floor_measure
    y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)
     
with varying_intercept_slope:
    varying_intercept_slope = sample(500, tune=500, cores=1)

df_trace = pm.trace_to_dataframe(varying_intercept_slope)
df_trace.columns
df_trace[['mu_a', 'mu_beta']].mean()
#mu_a       1.492
#mu_beta   -0.645
# bascially the same as above when taking corr matrix into account


with pm.Model() as m_13_7:
    etasq = pm.HalfCauchy('etasq', 1)
    rhosq = pm.HalfCauchy('rhosq', 1)
    Kij = etasq*(tt.exp(-rhosq*Dmatsq)+np.diag([.01]*Nsociety))
    
    g = pm.MvNormal('g', mu=np.zeros(Nsociety), cov=Kij, shape=Nsociety)
    
    a = pm.Normal('a', 0, 10)
    bp = pm.Normal('bp', 0, 1)
    lam = pm.math.exp(a + g[dk.society.values] + bp*dk.logpop)
    obs = pm.Poisson('total_tools', lam, observed=dk.total_tools)
    trace_13_7 = pm.sample(1000, tune=1000)



# -----------------------------------------------------------------------------
# generate data
size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=.5, size=size)
data = dict(x=x, y=y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Generated data and underlying model')
ax.plot(x, y, 'x', label='sampled data')
ax.plot(x, true_regression_line, label='true regression line', lw=2.)
plt.legend(loc=0);

with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    # have to do this first, before likelihood below. not sure i like this
    # because helps to have the model with the params that need priors first
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('intercept', 0, sd=20)
    x_coeff = pm.Normal('x_coeff', 0, sd=20)
    # Define likelihood
    # i like this -- we want to map  x to y. so we have a model with paramters
    # to do this. it's a linear model: mu = intercept + x_coeff * x
    # and this produces the outcome/y with error, normally distributed.
    # then we see how well each combo of uknown paramters fits or maps to 
    # or produces the y/dv. this is the definition of the likelihood. so 
    # the likelihood is a distribution of the number of ways that each combo
    # of paramters would produce the dv/y. and we specify here the actual
    # data for y with observed=y.
    likelihood = pm.Normal('y', mu = intercept + x_coeff * x,
                        sd=sigma, observed=y)
    # Inference
    trace = pm.sample(500, tune=500, cores=4, chains=1)  # cores=1. draw 500 posterior samples using 
    # defult sampler. think it will choose NUTS sampler here. Here we draw 1000 
    # samples from the posterior and allow the sampler to adjust its parameters 
    # in an additional 500 iterations. These 500 samples are discarded by default.
    # chains=1 says how many times the mcmc samples posterior. if chains=2 and
    # sample=1000 and tune=500 it'll sample 3000 times (1000*2 + 500*2)
    # For typical regression models, you can live by the motto four short chains 
    # to check, one long chain for inference.

    
# or almost all continuous models, ``NUTS`` should be preferred. There are 
# hard-to-sample models for which NUTS will be very slow causing many users to 
# use Metropolis instead. This practice, however, is rarely successful. NUTS is 
# fast on simple models but can be slow if the model is very complex or it is 
# badly initialized. In the case of a complex model that is hard for NUTS, 
# Metropolis, while faster, will have a very low effective sample size or not 
# converge properly at all. A better approach is to instead try to improve 
# initialization of NUTS, or reparameterize the model.

# plot the posterior distribution of our parameters and the individual samples we drew.
plt.figure(figsize=(7, 7))
pm.traceplot(trace[:])
# why two lines/colors for each distribution even though i specified 1 core/chain?

# plot regression lines
plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
pm.plot_posterior_predictive_glm(trace, samples=250, alpha=.25, 
                                 lm = lambda x, sample: sample['intercept'] + sample['x_coeff'] * x,
                                 label='posterior predictive regression lines')
# the above lm line is all to say what the linear model is, ans uses same variable names assigned in priors above
#plt.plot(x, true_regression_line, label='true regression line', lw=1., c='y')
plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y');

# get credible intercvals
pm.forestplot(trace);


with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('intercept', 0, sd=20)
    x_coeff = pm.Normal('x_coeff', 0, sd=20)
    # Define likelihood
    # changed mu to "Deterministic" - this puts mu in the trace so will have the mean sampled each time?
    likelihood = pm.Normal('y', mu = pm.Deterministic('mu', intercept + x_coeff * x),
                        sd=sigma, observed=y)
    # Sample prior for inference
    trace = pm.sample(500, tune=500, cores=4, chains=1)   


pm.traceplot(trace[:])

trace.varnames
trace['mu']
plt.hist(trace['x_coeff'])

len(trace)  # number of samples
trace[5]  # produces a row, one posterior sample

pm.plot_posterior_predictive_glm(trace, samples=250, alpha=.25, 
                                 lm = lambda x, sample: sample['intercept'] + sample['x_coeff'] * x,
                                 label='posterior predictive regression lines')


len(trace[5]['mu'])
plt.plot(trace[5]['mu'])

sampled_indices_list = np.random.choice(np.arange(len(trace)), 250, replace=True)  # the rethinking textbook samples the posterior with replacment
for index in sampled_indices_list:
    plt.plot(trace[index]['mu'], alpha=.025, color='black')

# sample without replacemnt. about the same as with replacment
sampled_indices_list = np.random.choice(np.arange(len(trace)), 250, replace=False)  # the rethinking textbook samples the posterior with replacment

for index in sampled_indices_list:
    plt.plot(x, trace[index]['mu'], alpha=.025, color='black')

for index in sampled_indices_list:
    plt.plot(x, trace[index]['mu'], alpha=.025, color='black')

for index in sampled_indices_list:
    plt.plot(x, trace[index]['intercept'] + trace[index]['x_coeff'] * x, color='black', alpha=.025)
    #plt.plot(x, trace['intercept'][index] + trace['x_coeff'][index] * x, color='black', alpha=.025)

# when i sample the lines with mu, is that a joint distribution, i.e.,
# taking into account the corr betwen intercept and x_coeff (by way of
# taking into account the variance-covariance matrix?). when i plot samples
# of the intercept and x_coff to make the lines, is this taking into account
# the variance-covaraiance matrix?

# get again but without mu
with pm.Model() as model: 
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('intercept', 0, sd=20)
    x_coeff = pm.Normal('x_coeff', 0, sd=20)
    likelihood = pm.Normal('y', mu = intercept + x_coeff * x,
                        sd=sigma, observed=y)
    trace = pm.sample(500, tune=500, cores=4, chains=1)   

pm.traceplot(trace)

pm.forestplot(trace);

# to get variance-covariance matrix
trace_df = pm.trace_to_dataframe(trace)
trace_df.head()
trace_df.cov()
dir(trace_df)
trace_df.mean()
trace_df.corr()
# super high corr between intercept and x_coeff. centering should help this?

x_c = x - np.mean(x)
np.mean(x_c)
# and really should just compute a z score
# worry about standardizing predictors/ivs but don't think need to standardize y/DV

with pm.Model() as model: 
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('intercept', 0, sd=20)
    x_coeff = pm.Normal('x_coeff', 0, sd=20)
    likelihood = pm.Normal('y', mu = intercept + x_coeff * x_c,
                        sd=sigma, observed=y)
    trace = pm.sample(500, tune=500, cores=4, chains=1)   

pm.traceplot(trace)

trace_df = pm.trace_to_dataframe(trace)
trace_df.head()
trace_df.cov()
trace_df.corr()  # nice!

pm.forestplot(trace)  # and the x_coeff is the same (only intercept changed)

# not sure yet about how to get credible intervals?
np.quantile(trace['x_coeff'], .01)
np.quantile(trace['x_coeff'], .99)
pm.hpd(trace['x_coeff'], alpha=.01)
# ok, these are about the same. but not quite.

# generate posterior predictive samples. 
# i.e., incorporate the sigma.
help(pm.sample_ppc)  # depricated for:
help(pm.sampling.sample_posterior_predictive)
y_predictions = pm.sampling.sample_posterior_predictive(trace=trace, samples=100, model=model)
y_predictions.keys()
len(y_predictions['y'])  # 100
len(y_predictions['y'][0])  # 200
# for each of 200 x values in original data, we get 100 predictions

# or generate posterior predictive samples for specified new x values
#x_seq = np.arange(-1, 2)
x_seq = np.linspace(-.5, .5, 50)
post_samples = []
for _ in range(100): # number of samples from the posterior
    i = np.random.randint(len(trace))
    mu_pred = trace['intercept'][i] + trace['x_coeff'][i] * x_seq
    sigma_pred = trace['sigma'][i]
    post_samples.append(np.random.normal(mu_pred, sigma_pred))

post_samples_hpd = pm.hpd(np.array(post_samples))

plt.scatter(x_c, y, alpha=.5)
#plt.plot(x_seq, mu_mean, 'C2')
#plt.fill_between(x_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25)
plt.fill_between(x_seq, post_samples_hpd[:,0], post_samples_hpd[:,1], color='C2', alpha=0.25)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xlim(np.min(x_c), np.max(x_c));
#mu_at_50 = trace['intercept'] + trace['x_coeff'] * 50
#plt.hist(mu_at_50)




# -----
# as an aside, here we are producing samples from a multivariate gaussian
# with means and var-cov matrix that's the same as our posterior samples
# so this should be pretty close to the actual posterior samples we already 
# have. but this below is from a theoretical multivariate dsitribution
# rather than the actual sampled posterior.
from scipy import stats
stats.multivariate_normal.rvs(mean=trace_df.mean(), cov=trace_df.cov(), size=10)
# -----




# -------------------------------------------------------------------------
# example from https://docs.pymc.io/notebooks/getting_started.html

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');

basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    # shape=2 means that two of these, and can see how distinguish below
    # w beta[0] and beta[1]. but each beta has the same prior
    #beta = pm.Normal('beta', mu=0, sd=10, shape=2)  
    beta_0 = pm.Normal('beta_0', mu=0, sd=10)  
    beta_1 = pm.Normal('beta_1', mu=0, sd=10)  
    sigma = pm.HalfNormal('sigma', sd=1)
    # Expected value of outcome
    #mu = alpha + beta[0]*X1 + beta[1]*X2
    mu = alpha + beta_0*X1 + beta_1*X2
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

with basic_model:
    # draw 250 posterior samples
    trace = pm.sample(250)

pm.traceplot(trace)

# the summary function provides a text-based output of common posterior statistics:
pm.summary(trace).round(2)

pm.plot_posterior_predictive_glm(trace, samples=100, alpha=.25, 
                                 lm = lambda x, sample: sample['alpha'] + sample['beta_1'] * X2)

# this doesn't make sense
plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
pm.plot_posterior_predictive_glm(trace, samples=100, alpha=.25, 
                                 lm = lambda x, sample: sample['alpha']+sample['beta_0']*X1,  # +sample['beta_1']*X2,
                                 label='posterior predictive regression lines')
# the above lm line is all to say what the linear model is, and uses same variable names assigned in priors above
plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y');

help(trace)
dir(trace)
trace.varnames
plt.hist(trace['beta_0'])

len(trace)  # number of samples
trace[5]  # produces a row, one posterior sample
# could randomly take 100 of these rows for one beta and plot 100 lines


# -------------------------------------------------------------------------












