# !/usr/bin/python
# coding: utf8

"""
Prediction of flux rope magnetic fields

This file is part of the mfr_predict-package.
It loads the final machine learning model and applies it to the test data set
generated in mfr_predict_v1p1.py.

Authors: U.V. Amerstorfer, Space Research Institute IWF Graz, Austria
Last update: Nov 2019

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

from sunpy.time import parse_time

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

sns.set_context("poster")
sns.set_style("darkgrid")

plt.close('all')

# READ INPUT OPTIONS FROM COMMAND LINE
argv = sys.argv[1:]

if len(argv) != 5:
    print('Invalid Numbers of Arguments. Script will be terminated.')
else:
    print('Load train and test data from:', argv[0])  # file name of train and test data
    print('Load final model from:', argv[1])  # file name of final model
    print('Save WIND plots to:', argv[2])  # file name of plot to be saved
    print('Save STA plots to:', argv[3])  # file name of plot to be saved
    print('Save STB plots to:', argv[4])  # file name of plot to be saved

#####################################################################################
# ####################### main program ###############################################
#####################################################################################

# =========================== READ TRAIN AND TEST DATA ===========================================
fname = 'mfr_predict/' + argv[0]
n_iwinind, n_istaind, n_istbind, win_train_ind, win_test_ind, sta_train_ind, sta_test_ind, stb_train_ind, stb_test_ind, train_ind, test_ind, X_train, X_test, y_train, y_test, feature_hours = pickle.load(open(fname, 'rb'))

# =========================== LOAD FINAL MODEL ==================================================

# load the ML model
filename = 'mfr_predict/' + argv[1]
model = pickle.load(open(filename, 'rb'))

# ============================ FINAL MODEL =======================================================
# use model on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-scores)
print('')
print('ML model on test set (Btot mean)')
print('Cross validation scores:')
print('Root mean squared error: ', rmse_scores)
print('Mean:', rmse_scores.mean())
print('Std:', rmse_scores.std())
print('Root mean squared error:', rmse)
print('R-squared:', r2score)
print('')
print("Accuracy: %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))







#############################################################################



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

























# ############################### PLOTS #############################
# WIND
# get number of rows for figure
nRows_wind = np.size(win_test_ind)

plt.figure(figsize=(20, 8 * nRows_wind))

for iEv in range(0, nRows_wind):
    ind = n_iwinind[test_ind[iEv]]

    istart = np.where(win['time'] >= icme_start_time_num[ind])[0][0]
    iend = np.where(win['time'] >= mo_end_time_num[ind])[0][0]
    mostart = np.where(win['time'] >= mo_start_time_num[ind])[0][0]
    mostart_fh = mostart + feature_hours / 24.0
    moend = np.where(win['time'] >= mo_end_time_num[ind])[0][0]

    larr = len(win['time'][int(mostart_fh):int(moend)])
    predVal = np.zeros(larr)
    yObs = np.zeros(larr)
    predVal[:] = y_pred[iEv]
    yObs[:] = y_test[iEv]

    test_larr = len(win['time'][int(mostart):int(mostart_fh)])
    # test_larr = len(win['time'][int(istart):int(mostart)])
    X_test_plot = np.zeros(test_larr)
    X_test_plot[:] = X_test[iEv]

    plt.subplot(nRows_wind, 1, iEv + 1)
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # xg, yg, dxg, dyg = geom.getRect()
    # Newxg = xg + 1400
    # Newyg = yg + 400
    # mngr.window.setGeometry(Newxg, Newyg, dxg, dyg)

    plt.plot(win['time'][int(istart):int(iend)], win['bt'][int(istart):int(iend)])
    plt.plot(win['time'][int(mostart_fh):int(moend)], predVal, 'r-', label='mean B$_{tot}^{MFR}$ pred WIND')
    plt.plot(win['time'][int(mostart_fh):int(moend)], yObs, 'b-', label='mean B$_{tot}^{MFR}$ obs WIND')
    #  plt.plot(win['time'][int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR}$')
    # plt.plot(win['time'][int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR} (5h)$')
    plt.axvline(x=win['time'][int(mostart)], color='r', linestyle='--')
    plt.axvline(x=win['time'][int(moend)], color='r', linestyle='--')
    plt.fill_between(win['time'][int(mostart_fh):int(moend)], predVal - rmse_scores.mean(), predVal + rmse_scores.mean(), facecolor='slategrey', alpha=0.2, edgecolor='none')
    plt.xlabel('Time')
    plt.ylabel('B$_{tot}$ [nT]')
    plt.legend(numpoints=1, ncol=2, loc='best')

plt.savefig('plots/' + argv[2], bbox_inches='tight')

# STEREO-A
nRows_sta = np.size(sta_test_ind)

plt.figure(figsize=(20, 8 * nRows_sta))

for iEv in range(len(win_test_ind), len(win_test_ind) + len(sta_test_ind)):
    ind = n_istaind[test_ind[iEv]]

    istart = np.where(sta['time'] >= icme_start_time_num[ind])[0][0]
    iend = np.where(sta['time'] >= mo_end_time_num[ind])[0][0]
    mostart = np.where(sta['time'] >= mo_start_time_num[ind])[0][0]
    mostart_fh = mostart + feature_hours / 24.0
    # mostart_fh = np.where(win['time'] == mo_start_time_num[ind] + feature_hours / 24.0)[0]
    moend = np.where(sta['time'] >= mo_end_time_num[ind])[0][0]

    larr = len(sta['time'][int(mostart_fh):int(moend)])
    predVal = np.zeros(larr)
    yObs = np.zeros(larr)
    predVal[:] = y_pred[iEv]
    yObs[:] = y_test[iEv]

    test_larr = len(sta['time'][int(mostart):int(mostart_fh)])
    # test_larr = len(win['time'][int(istart):int(mostart)])
    X_test_plot = np.zeros(test_larr)
    X_test_plot[:] = X_test[iEv]

    plt.subplot(nRows_sta, 1, (iEv - len(win_test_ind) + 1))
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # xg, yg, dxg, dyg = geom.getRect()
    # Newxg = xg + 1400
    # Newyg = yg + 400
    # mngr.window.setGeometry(Newxg, Newyg, dxg, dyg)

    plt.plot(sta['time'][int(istart):int(iend)], sta['bt'][int(istart):int(iend)])
    plt.plot(sta['time'][int(mostart_fh):int(moend)], predVal, 'r-', label='mean B$_{tot}^{MFR}$ pred STA')
    plt.plot(sta['time'][int(mostart_fh):int(moend)], yObs, 'b-', label='mean B$_{tot}^{MFR}$ obs STA')
    #  plt.plot(win['time'][int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR}$')
    # plt.plot(win['time'][int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR} (5h)$')
    plt.axvline(x=sta['time'][int(mostart)], color='r', linestyle='--')
    plt.axvline(x=sta['time'][int(moend)], color='r', linestyle='--')
    plt.fill_between(sta['time'][int(mostart_fh):int(moend)], predVal - rmse_scores.mean(), predVal + rmse_scores.mean(), facecolor='slategrey', alpha=0.2, edgecolor='none')
    plt.xlabel('Time')
    plt.ylabel('B$_{tot}$ [nT]')
    plt.legend(numpoints=1, ncol=2, loc='best')

plt.savefig('plots/' + argv[3], bbox_inches='tight')

# STEREO-B
nRows_stb = np.size(stb_test_ind)

plt.figure(figsize=(20, 8 * nRows_stb))

for iEv in range(len(win_test_ind) + len(sta_test_ind), len(test_ind)):
    ind = n_istbind[test_ind[iEv]]

    istart = np.where(stb['time'] >= icme_start_time_num[ind])[0][0]
    iend = np.where(stb['time'] >= mo_end_time_num[ind])[0][0]
    mostart = np.where(stb['time'] >= mo_start_time_num[ind])[0][0]
    mostart_fh = mostart + feature_hours / 24.0
    # mostart_fh = np.where(win['time'] == mo_start_time_num[ind] + feature_hours / 24.0)[0]
    moend = np.where(stb['time'] >= mo_end_time_num[ind])[0][0]

    larr = len(stb['time'][int(mostart_fh):int(moend)])
    predVal = np.zeros(larr)
    yObs = np.zeros(larr)
    predVal[:] = y_pred[iEv]
    yObs[:] = y_test[iEv]

    test_larr = len(stb['time'][int(mostart):int(mostart_fh)])
    # test_larr = len(win['time'][int(istart):int(mostart)])
    X_test_plot = np.zeros(test_larr)
    X_test_plot[:] = X_test[iEv]

    plt.subplot(nRows_stb, 1, (iEv - len(win_test_ind) - len(sta_test_ind) + 1))
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # xg, yg, dxg, dyg = geom.getRect()
    # Newxg = xg + 1400
    # Newyg = yg + 400
    # mngr.window.setGeometry(Newxg, Newyg, dxg, dyg)

    plt.plot(stb['time'][int(istart):int(iend)], stb['bt'][int(istart):int(iend)])
    plt.plot(stb['time'][int(mostart_fh):int(moend)], predVal, 'r-', label='mean B$_{tot}^{MFR}$ pred STB')
    plt.plot(stb['time'][int(mostart_fh):int(moend)], yObs, 'b-', label='mean B$_{tot}^{MFR}$ obs STB')
    #  plt.plot(win['time'][int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR}$')
    # plt.plot(win['time'][int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR} (5h)$')
    plt.axvline(x=stb['time'][int(mostart)], color='r', linestyle='--')
    plt.axvline(x=stb['time'][int(moend)], color='r', linestyle='--')
    plt.fill_between(stb['time'][int(mostart_fh):int(moend)], predVal - rmse_scores.mean(), predVal + rmse_scores.mean(), facecolor='slategrey', alpha=0.2, edgecolor='none')
    plt.xlabel('Time')
    plt.ylabel('B$_{tot}$ [nT]')
    plt.legend(numpoints=1, ncol=2, loc='best')

plt.savefig('plots/' + argv[4], bbox_inches='tight')

# plt.show()
