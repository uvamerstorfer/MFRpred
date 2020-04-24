# !/usr/bin/python
# coding: utf8

"""
Prediction of flux rope magnetic fields

Analyses HELCATS ICMECAT for predicting labels of CME MFRs
Authors: U.V. Amerstorfer, Space Research Institute IWF Graz, Austria
Last update: Nov 2019

How to predict the rest of the MFR if first 10, 20, 30, 40, 50% are seen?
Everything should be automatically with a deep learning method or ML fit methods

"""
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from scipy import stats
import scipy.io
import sunpy.time
import numpy as np
import time
import pickle
import seaborn as sns
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from pandas.plotting import scatter_matrix


from sunpy.time import parse_time

import warnings
warnings.filterwarnings('ignore')


# sns.set_context("talk")
# sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=0.4, rc={"lines.linewidth": 2.5})

plt.close('all')

# READ INPUT OPTIONS FROM COMMAND LINE
argv = sys.argv[1:]

if len(argv) != 2:
    print('Invalid Numbers of Arguments. Script will be terminated.')
else:
    print('Read in test and train data from:', argv[0])
    print('Save final model to:', argv[1])


# ####################### functions ###############################################


# use different models
def get_models(models=dict()):
    # linear models
    models['lr'] = LinearRegression()
    models['lasso'] = Lasso()
    models['ridge'] = Ridge()
    models['en'] = ElasticNet()
    models['huber'] = HuberRegressor()
    models['lars'] = Lars()
    models['llars'] = LassoLars()
    models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    models['ranscac'] = RANSACRegressor()
    models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
    models['rfr'] = RandomForestRegressor()
    print('Defined %d models' % len(models))
    return models


# fit model, evaluate it and get scores
def sklearn_predict(model, X, y):
    # fit the model
    model.fit(X, y)
    # prediction
    y_predict = model.predict(X)
    score, mean_score, std_score = evaluate_forecast(model, X, y, y_predict)
    return score, mean_score, std_score, y_predict


# define scores
def evaluate_forecast(model, X, y, y_predict):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    score = np.sqrt(-scores)
    mean_score = score.mean()  # root mean squared error
    std_score = score.std()
    return score, mean_score, std_score


#####################################################################################
# ####################### main program ###############################################
#####################################################################################

    
# ------------------------ READ ICMECAT    

filename_icmecat = 'data/HELCATS_ICMECAT_v20_pandas.p'
[ic,header,parameters] = pickle.load(open(filename_icmecat, "rb" ))

print()
print()
print('load icmecat')

#ic is the pandas dataframe with the ICMECAT
#print(ic.keys())



# ------------------------ get all parameters from ICMECAT for easier handling
# id for each event
iid = ic.loc[:,'icmecat_id']

# observing spacecraft
isc = ic.loc[:,'sc_insitu'] 

icme_start_time = ic.loc[:,'icme_start_time']
icme_start_time_num = parse_time(icme_start_time).plot_date

mo_start_time = ic.loc[:,'mo_start_time']
mo_start_time_num = parse_time(mo_start_time).plot_date

mo_end_time = ic.loc[:,'mo_end_time']
mo_end_time_num = parse_time(mo_end_time).plot_date

sc_heliodistance = ic.loc[:,'mo_sc_heliodistance']
sc_long_heeq = ic.loc[:,'mo_sc_long_heeq']
sc_lat_heeq = ic.loc[:,'mo_sc_long_heeq']
mo_bmax = ic.loc[:,'mo_bmax']
mo_bmean = ic.loc[:,'mo_bmean']
mo_bstd = ic.loc[:,'mo_bstd']

mo_duration = ic.loc[:,'mo_duration']


# get indices of events by different spacecraft
istaind = np.where(isc == 'STEREO-A')[0]
istbind = np.where(isc == 'STEREO-B')[0]
iwinind = np.where(isc == 'Wind')[0]



# ############################# load spacecraft data ################################


print('load Wind data')
[win,winheader] = pickle.load(open("data/wind_2007_2019_heeq_ndarray.p", "rb"))

print('load STEREO-A data')
[sta,att, staheader] = pickle.load(open("data/stereoa_2007_2019_sceq_ndarray.p", "rb"))

print('load STEREO-B data')
[stb,att, stbheader] = pickle.load(open("data/stereob_2007_2014_sceq_ndarray.p", "rb"))


print()



#############################################################################
# =========================== READ TRAIN AND TEST DATA ===========================================
fname = 'mfr_predict/' + argv[0]
n_iwinind, n_istaind, n_istbind, win_train_ind, win_test_ind, sta_train_ind, sta_test_ind, stb_train_ind, stb_test_ind, train_ind, test_ind, X_train, X_test, y_train, y_test, feature_hours = pickle.load(open(fname, 'rb'))

# ############################# Models ####################################

models = get_models()

mean_score = np.zeros(len(models))
std_score = np.zeros(len(models))
final_model_name = ''
print('')
imod = 0
best_score = 10.

for name, model in models.items():
    # print(ind)
    # fit model, evaluate and get scores
    score, mean_score[imod], std_score[imod], y_predict = sklearn_predict(model, X_train, y_train)
    # summarize scores
    print(name, mean_score[imod], std_score[imod])  # , score)
    if imod > 0:
        # print(ind)
        if mean_score[imod] < best_score:
            best_score = mean_score[imod]
            final_model_name = name
            final_model = model
            print(final_model_name)
    # plot scores
    m_score = np.zeros(len(score))
    m_score[:] = mean_score[imod]
    plt.plot(score, marker='o', label=name)
   # plt.plot(m_score, linestyle='--', label=name)
    imod = imod + 1
# show plot
plt.legend()
print(' ')
print('final model:', final_model_name)

# =========================== SAVE FINAL MODEL ===================================================
# save the model to disk
filename = 'mfr_predict/' + argv[1]
pickle.dump(final_model, open(filename, 'wb'))


# plt.show()
