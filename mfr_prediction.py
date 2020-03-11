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

################################ load WIND times #############################
win = pickle.load(open("../catpy/DATACAT/WIND_2007to2018_HEEQ.p", "rb"))
[win_time] = pickle.load(open("../catpy/DATACAT/insitu_times_mdates_win_2007_2018.p", "rb"))

################################ load STEREO-A times #############################
sta = pickle.load(open("../catpy/DATACAT/STA_2007to2015_SCEQ.p", "rb"))
[sta_time] = pickle.load(open("../catpy/DATACAT/insitu_times_mdates_sta_2007_2015.p", "rb"))

################################ load STEREO-B times #############################
stb = pickle.load(open("../catpy/DATACAT/STB_2007to2014_SCEQ.p", "rb"))
[stb_time] = pickle.load(open("../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "rb"))

################################ load ICME times #############################
[icme_start_time, icme_end_time, mo_start_time, mo_end_time] = pickle.load(open("../catpy/DATACAT/icme_times_string.p", "rb"))
[icme_start_time_num, icme_end_time_num, mo_start_time_num, mo_end_time_num, iwinind, istaind, istbind] = pickle.load(open("../catpy/DATACAT/icme_times.p", "rb"))

# ############################### PLOTS #############################
# WIND
# get number of rows for figure
nRows_wind = np.size(win_test_ind)

plt.figure(figsize=(20, 8 * nRows_wind))

for iEv in range(0, nRows_wind):
    ind = n_iwinind[test_ind[iEv]]

    istart = np.where(win_time >= icme_start_time_num[ind])[0][0]
    iend = np.where(win_time >= icme_end_time_num[ind])[0][0]
    mostart = np.where(win_time >= mo_start_time_num[ind])[0][0]
    mostart_fh = mostart + feature_hours / 24.0
    moend = np.where(win_time >= mo_end_time_num[ind])[0][0]

    larr = len(win_time[int(mostart_fh):int(moend)])
    predVal = np.zeros(larr)
    yObs = np.zeros(larr)
    predVal[:] = y_pred[iEv]
    yObs[:] = y_test[iEv]

    test_larr = len(win_time[int(mostart):int(mostart_fh)])
    # test_larr = len(win_time[int(istart):int(mostart)])
    X_test_plot = np.zeros(test_larr)
    X_test_plot[:] = X_test[iEv]

    plt.subplot(nRows_wind, 1, iEv + 1)
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # xg, yg, dxg, dyg = geom.getRect()
    # Newxg = xg + 1400
    # Newyg = yg + 400
    # mngr.window.setGeometry(Newxg, Newyg, dxg, dyg)

    plt.plot(win_time[int(istart):int(iend)], win.btot[int(istart):int(iend)])
    plt.plot(win_time[int(mostart_fh):int(moend)], predVal, 'r-', label='mean B$_{tot}^{MFR}$ pred WIND')
    plt.plot(win_time[int(mostart_fh):int(moend)], yObs, 'b-', label='mean B$_{tot}^{MFR}$ obs WIND')
    #  plt.plot(win_time[int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR}$')
    # plt.plot(win_time[int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR} (5h)$')
    plt.axvline(x=win_time[int(mostart)], color='r', linestyle='--')
    plt.axvline(x=win_time[int(moend)], color='r', linestyle='--')
    plt.fill_between(win_time[int(mostart_fh):int(moend)], predVal - rmse_scores.mean(), predVal + rmse_scores.mean(), facecolor='slategrey', alpha=0.2, edgecolor='none')
    plt.xlabel('Time')
    plt.ylabel('B$_{tot}$ [nT]')
    plt.legend(numpoints=1, ncol=2, loc='best')

plt.savefig('plots/' + argv[2], bbox_inches='tight')

# STEREO-A
nRows_sta = np.size(sta_test_ind)

plt.figure(figsize=(20, 8 * nRows_sta))

for iEv in range(len(win_test_ind), len(win_test_ind) + len(sta_test_ind)):
    ind = n_istaind[test_ind[iEv]]

    istart = np.where(sta_time >= icme_start_time_num[ind])[0][0]
    iend = np.where(sta_time >= icme_end_time_num[ind])[0][0]
    mostart = np.where(sta_time >= mo_start_time_num[ind])[0][0]
    mostart_fh = mostart + feature_hours / 24.0
    # mostart_fh = np.where(win_time == mo_start_time_num[ind] + feature_hours / 24.0)[0]
    moend = np.where(sta_time >= mo_end_time_num[ind])[0][0]

    larr = len(sta_time[int(mostart_fh):int(moend)])
    predVal = np.zeros(larr)
    yObs = np.zeros(larr)
    predVal[:] = y_pred[iEv]
    yObs[:] = y_test[iEv]

    test_larr = len(sta_time[int(mostart):int(mostart_fh)])
    # test_larr = len(win_time[int(istart):int(mostart)])
    X_test_plot = np.zeros(test_larr)
    X_test_plot[:] = X_test[iEv]

    plt.subplot(nRows_sta, 1, (iEv - len(win_test_ind) + 1))
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # xg, yg, dxg, dyg = geom.getRect()
    # Newxg = xg + 1400
    # Newyg = yg + 400
    # mngr.window.setGeometry(Newxg, Newyg, dxg, dyg)

    plt.plot(sta_time[int(istart):int(iend)], sta.btot[int(istart):int(iend)])
    plt.plot(sta_time[int(mostart_fh):int(moend)], predVal, 'r-', label='mean B$_{tot}^{MFR}$ pred STA')
    plt.plot(sta_time[int(mostart_fh):int(moend)], yObs, 'b-', label='mean B$_{tot}^{MFR}$ obs STA')
    #  plt.plot(win_time[int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR}$')
    # plt.plot(win_time[int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR} (5h)$')
    plt.axvline(x=sta_time[int(mostart)], color='r', linestyle='--')
    plt.axvline(x=sta_time[int(moend)], color='r', linestyle='--')
    plt.fill_between(sta_time[int(mostart_fh):int(moend)], predVal - rmse_scores.mean(), predVal + rmse_scores.mean(), facecolor='slategrey', alpha=0.2, edgecolor='none')
    plt.xlabel('Time')
    plt.ylabel('B$_{tot}$ [nT]')
    plt.legend(numpoints=1, ncol=2, loc='best')

plt.savefig('plots/' + argv[3], bbox_inches='tight')

# STEREO-B
nRows_stb = np.size(stb_test_ind)

plt.figure(figsize=(20, 8 * nRows_stb))

for iEv in range(len(win_test_ind) + len(sta_test_ind), len(test_ind)):
    ind = n_istbind[test_ind[iEv]]

    istart = np.where(stb_time >= icme_start_time_num[ind])[0][0]
    iend = np.where(stb_time >= icme_end_time_num[ind])[0][0]
    mostart = np.where(stb_time >= mo_start_time_num[ind])[0][0]
    mostart_fh = mostart + feature_hours / 24.0
    # mostart_fh = np.where(win_time == mo_start_time_num[ind] + feature_hours / 24.0)[0]
    moend = np.where(stb_time >= mo_end_time_num[ind])[0][0]

    larr = len(stb_time[int(mostart_fh):int(moend)])
    predVal = np.zeros(larr)
    yObs = np.zeros(larr)
    predVal[:] = y_pred[iEv]
    yObs[:] = y_test[iEv]

    test_larr = len(stb_time[int(mostart):int(mostart_fh)])
    # test_larr = len(win_time[int(istart):int(mostart)])
    X_test_plot = np.zeros(test_larr)
    X_test_plot[:] = X_test[iEv]

    plt.subplot(nRows_stb, 1, (iEv - len(win_test_ind) - len(sta_test_ind) + 1))
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # xg, yg, dxg, dyg = geom.getRect()
    # Newxg = xg + 1400
    # Newyg = yg + 400
    # mngr.window.setGeometry(Newxg, Newyg, dxg, dyg)

    plt.plot(stb_time[int(istart):int(iend)], stb.btot[int(istart):int(iend)])
    plt.plot(stb_time[int(mostart_fh):int(moend)], predVal, 'r-', label='mean B$_{tot}^{MFR}$ pred STB')
    plt.plot(stb_time[int(mostart_fh):int(moend)], yObs, 'b-', label='mean B$_{tot}^{MFR}$ obs STB')
    #  plt.plot(win_time[int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR}$')
    # plt.plot(win_time[int(mostart):int(mostart_fh)], X_test_plot, 'g--', label='feature mean B$_{tot}^{MFR} (5h)$')
    plt.axvline(x=stb_time[int(mostart)], color='r', linestyle='--')
    plt.axvline(x=stb_time[int(moend)], color='r', linestyle='--')
    plt.fill_between(stb_time[int(mostart_fh):int(moend)], predVal - rmse_scores.mean(), predVal + rmse_scores.mean(), facecolor='slategrey', alpha=0.2, edgecolor='none')
    plt.xlabel('Time')
    plt.ylabel('B$_{tot}$ [nT]')
    plt.legend(numpoints=1, ncol=2, loc='best')

plt.savefig('plots/' + argv[4], bbox_inches='tight')

# plt.show()
