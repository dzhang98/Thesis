#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.stats import jarque_bera
from scipy.stats import norm
from scipy.stats import f
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import seaborn as sns


# In[5]:


def get_basic_stats(arr):
    print("mean is: ", np.mean(arr))
    print("standard deviation is: ", np.std(arr))
    print("skew is: ", skew(arr))
    print("kurtosis is: ", kurtosis(arr))
    return np.mean(arr), np.std(arr), skew(arr), kurtosis(arr)


# In[7]:


def plot_hist_qq(arr):
    plot_range = np.linspace(min(arr), max(arr), num=5000 )
    pdf_series = norm.pdf(plot_range, loc=np.mean(arr), scale=np.std(arr))

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Histogram
    sns.distplot(arr, kde=False, norm_hist=True, ax=ax[0]) # STYLIZED FACT: high peak

    ax[0].set_title('Distribution of returns', fontsize=16)
    ax[0].plot(plot_range, pdf_series, 'b', lw=2, label=f'N({np.mean(arr):.2f}, {np.std(arr)**2:.4f})')
    ax[0].legend(loc='upper left')

    # QQ Plot
    qq = sm.qqplot(arr, line='s', ax=ax[1]) # STYLIZED FACT: fat tails
    ax[1].set_title('Q-Q plot', fontsize = 16)


# In[8]:


def compute_jarque_bera(arr):
    """
    H_0 : distribution is normal at 99% confidence level
    H_1 : distribution is not normal at 99% confidence level

    - checks whether a distribution has skewness and kurtosis values matching that of a normal distribution
    - result is a non-negative value - the farther from zero, the greater it deviates from normal distribution
    """
    value = jarque_bera(arr)[0]
    p_value = jarque_bera(arr)[1]
    print("The Jarque-Bera test statistic value is", value, "with probability of", p_value)


# In[9]:


def plot_volclusters(arr):
    plt.plot([i for i in range(len(arr))], arr)


# In[10]:


def plot_autocorrelation(arr):
    fig, ax = plt.subplots(3, figsize=(14, 20))

    # For returns
    smt.graphics.plot_acf(arr, lags=30, alpha=0.05, ax = ax[0])

    ax[0].set(title='Autocorrelation plots of log returns', ylabel='Log Returns', xlabel='Lag')


    # Using statsmodels library to obtain acf plot for squared returns
    smt.graphics.plot_acf(arr ** 2, lags=100, alpha=0.05, ax = ax[1])

    # Setting title, y-axis labels and x-axis labels of first subplot 
    ax[1].set(title='Autocorrelation plots of sqaured log returns', ylabel='Squared Log Returns', xlabel='Lag')

    # Obtaining acf plot of absolute returns using abs function of numpy library
    smt.graphics.plot_acf(np.abs(arr), lags=100, alpha=0.05, ax = ax[2])

    # Setting title, y-axis labels and x-axis labels of second subplot 
    ax[2].set(title='Autocorrelation plots of absolute log returns', ylabel='Absolute Log Returns', xlabel='Lag')


# In[11]:


def KS_test(original_arr, gen_arr):
    return ks_2samp(original_arr, gen_arr)


# In[ ]:




