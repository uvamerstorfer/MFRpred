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

from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif

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

import warnings
warnings.filterwarnings('ignore')





#get all variables from the input.py file:

from input import *

#make new directory if it not exists
mfrdir='mfr_predict'
if os.path.isdir(mfrdir) == False: os.mkdir(mfrdir)

plotdir='plots'
if os.path.isdir(plotdir) == False: os.mkdir(plotdir)




# sns.set_context("talk")
# sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=0.4, rc={"lines.linewidth": 2.5})

plt.close('all')

# READ INPUT OPTIONS FROM COMMAND LINE
argv = sys.argv[1:]
# if --features is set, then code will produce pickle-file with features and labels
# if --features is not set, then code will read from already existing pickle-file
# you only have to set features at the first run of the code, or if you changed something in the corresponding parts of the code

#  first three arguments need to be file names to save features into -
#  WIND features: argv[0]
#  ST-A features: argv[1]
#  ST-B features: argv[2]

#  then --featuers if features need to be determined again
#  and --mfr if there are shall be no sheath features determined

features = False
if "--features" in argv:
    features = True
    print("get features")

mfr = False
if "--mfr" in argv:
    mfr = True
    print("only mfr features")

# ####################### functions ###############################################


def get_feature(sc_time, start_time, end_time, sc_ind, sc_feature, feature_hours, *VarArgs):
    feature_mean = np.zeros(np.size(sc_ind))
    feature_max = np.zeros(np.size(sc_ind))
    feature_std = np.zeros(np.size(sc_ind))

    for Arg in VarArgs:
        if Arg == 'mean':
            for p in np.arange(0, np.size(sc_ind)):
                # extract values from MFR data
                feature_temp = sc_feature[np.where(np.logical_and(sc_time > start_time[sc_ind[p]], sc_time < end_time[sc_ind[p]] + feature_hours / 24.0))]
                # print(feature_temp)
                feature_mean[p] = np.nanmean(feature_temp)
                # print('mean')
        elif Arg == 'std':
            for p in np.arange(0, np.size(sc_ind)):
                # extract values from MFR data
                feature_temp = sc_feature[np.where(np.logical_and(sc_time > start_time[sc_ind[p]], sc_time < end_time[sc_ind[p]] + feature_hours / 24.0))]
                # print(feature_temp)
                feature_std[p] = np.nanstd(feature_temp)
        elif Arg == 'max':
            for p in np.arange(0, np.size(sc_ind)):
                # extract values from MFR data
                feature_temp = sc_feature[np.where(np.logical_and(sc_time > start_time[sc_ind[p]], sc_time < end_time[sc_ind[p]] + feature_hours / 24.0))]
                # print(feature_temp)
                feature_temp = feature_temp[np.isfinite(feature_temp)]
                try:
                    feature_max[p] = np.max(feature_temp)
                except ValueError:  # raised if `y` is empty.
                    pass

                # print('max')

    if np.any(feature_mean) and np.any(feature_max) and np.any(feature_std):
        # print('mean and std and max')
        return feature_mean, feature_max, feature_std
    elif np.any(feature_mean) and np.any(feature_max) and (not np.any(feature_std)):
        # print('mean and max')
        return feature_mean, feature_max
    elif np.any(feature_mean) and (not np.any(feature_max)) and (not np.any(feature_std)):
        # print('only mean')
        return feature_mean
    elif (not np.any(feature_mean)) and np.any(feature_max) and np.any(feature_std):
        # print('max and std')
        return feature_max, feature_std
    elif (not np.any(feature_mean)) and (not np.any(feature_max)) and np.any(feature_std):
        # print('only std')
        return feature_std
    elif (not np.any(feature_mean)) and np.any(feature_max) and (not np.any(feature_std)):
        # print('only max')
        return feature_max
    elif np.any(feature_mean) and (not np.any(feature_max)) and np.any(feature_std):
        # print('mean and std')
        return feature_mean, feature_std


def get_label(sc_time, start_time, end_time, sc_ind, sc_label, feature_hours, *VarArgs):
    label_mean = np.zeros(np.size(sc_ind))
    label_max = np.zeros(np.size(sc_ind))

    for Arg in VarArgs:
        if Arg == 'mean':
            for p in np.arange(0, np.size(sc_ind)):
                label_temp = sc_label[np.where(np.logical_and(sc_time > start_time[sc_ind[p]] + feature_hours / 24.0, sc_time < end_time[sc_ind[p]]))]
                label_mean[p] = np.nanmean(label_temp)
        elif Arg == 'max':
            for p in np.arange(0, np.size(sc_ind)):
                label_temp = sc_label[np.where(np.logical_and(sc_time > start_time[sc_ind[p]] + feature_hours / 24.0, sc_time < end_time[sc_ind[p]]))]
                label_max[p] = np.nanmax(label_temp)

    if np.any(label_mean) and (not np.any(label_max)):
        # print('only mean')
        return label_mean
    elif (not np.any(label_mean)) and np.any(label_max):
        # print('only mean')
        return label_max
#####################################################################################
# ####################### main program ###############################################
#####################################################################################

# ############################# get spacecraft data ################################
# we use the HELCAT files only to get the times and indices of the events
# then we use other data files to get the spacecraft data


# get ICME times
print('get ICME times')
[icme_start_time_num, icme_end_time_num, mo_start_time_num, mo_end_time_num, iwinind, istaind, istbind] = pickle.load(open("data/icme_times.p", "rb"))
print('get ICME times done')

# ############################# get Wind data ################################

print('read Wind data')
# get insitu date
win = pickle.load(open("data/WIND_2007to2018_HEEQ.p", "rb"))
[win_time] = pickle.load(open("data/insitu_times_mdates_win_2007_2018.p", "rb"))
print('read data done')

# ############################# get Stereo-A data ################################

print('read Stereo-A data')
# get insitu data
sta = pickle.load(open("data/STA_2007to2015_SCEQ.p", "rb"))
[sta_time] = pickle.load(open("data/insitu_times_mdates_sta_2007_2015.p", "rb"))
print('read data done')

# ############################# get Stereo-B data ################################

print('read Stereo-B data')
# get insitu data
stb = pickle.load(open("data/STB_2007to2014_SCEQ.p", "rb"))
[stb_time] = pickle.load(open("data/insitu_times_mdates_stb_2007_2014.p", "rb"))
print('read data done')

#############################################################################

# Version (1.1)  - prediction of scalar labels with a linear model, start with Btot

# ################################# spacecraft #####################################
# wind data: win_time win.bx win.by ... win.vtot win.vy etc.
# sheath time: icme_start_time_num[iwinind] mo_start_time[iwinind]
# mfr time: mo_start_time[iwinind]  mo_end_time[iwinind]

# Stereo-A data: sta_time sta.bx sta.by ... sta.vtot sta.vy etc.
# sheath time: icme_start_time_num[istaind] mo_start_time[istaind]
# mfr time: mo_start_time[istaind]  mo_end_time[istaind]

# Stereo-B data: stb_time stb.bx stb.by ... stb.vtot stb.vy etc.
# sheath time: icme_start_time_num[istbind] mo_start_time[istbind]
# mfr time: mo_start_time[istbind]  mo_end_time[istbind]

# use some hours of MFR for feature
# only sheath for features: feature_hours = 0


# only take events where there is a sheath, so where the start of the ICME is NOT equal to the start of the flux rope
n_iwinind = np.where(icme_start_time_num[iwinind] != mo_start_time_num[iwinind])[0]
n_istaind = np.where(icme_start_time_num[istaind] != mo_start_time_num[istaind])[0]
n_istbind = np.where(icme_start_time_num[istbind] != mo_start_time_num[istbind])[0]
if features:
    # List of features - go through each ICME and extract values characterising them
    # only features of the sheath
    # syntax: get_features(spacecraft time, start time of intervall for values, end time of intervall for values, event index of spacecraft, value to be extracted, "mean", "std", "max")

    ################################ WIND #############################
    feature_bzmean, feature_bzstd = get_feature(win_time, icme_start_time_num, mo_start_time_num, n_iwinind, win.bz, feature_hours, "mean", "std")
    feature_bymean, feature_bystd = get_feature(win_time, icme_start_time_num, mo_start_time_num, n_iwinind, win.by, feature_hours, "mean", "std")
    feature_bxmean, feature_bxstd = get_feature(win_time, icme_start_time_num, mo_start_time_num, n_iwinind, win.bx, feature_hours, "mean", "std")
    feature_btotmean, feature_btotstd = get_feature(win_time, icme_start_time_num, mo_start_time_num, n_iwinind, win.btot, feature_hours, "mean", "std")
    feature_btotmean, feature_btotmax, feature_btotstd = get_feature(win_time, icme_start_time_num, mo_start_time_num, n_iwinind, win.btot, feature_hours, "mean", "max", "std")
    feature_vtotmean, feature_vtotmax, feature_vtotstd = get_feature(win_time, icme_start_time_num, mo_start_time_num, n_iwinind, win.vtot, feature_hours, "mean", "std", "max")

    if mfr:
        feature_bzmean, feature_bzstd = get_feature(win_time, mo_start_time_num, mo_start_time_num, n_iwinind, win.bz, feature_hours, "mean", "std")
        feature_bymean, feature_bystd = get_feature(win_time, mo_start_time_num, mo_start_time_num, n_iwinind, win.by, feature_hours, "mean", "std")
        feature_bxmean, feature_bxstd = get_feature(win_time, mo_start_time_num, mo_start_time_num, n_iwinind, win.bx, feature_hours, "mean", "std")
        feature_btotmean, feature_btotstd = get_feature(win_time, mo_start_time_num, mo_start_time_num, n_iwinind, win.btot, feature_hours, "mean", "std")
        feature_btotmean, feature_btotmax, feature_btotstd = get_feature(win_time, mo_start_time_num, mo_start_time_num, n_iwinind, win.btot, feature_hours, "mean", "max", "std")
        feature_vtotmean, feature_vtotmax, feature_vtotstd = get_feature(win_time, mo_start_time_num, mo_start_time_num, n_iwinind, win.vtot, feature_hours, "mean", "std", "max")
    # ------------------
    # label
    label_btotmean = get_label(win_time, mo_start_time_num, mo_end_time_num, n_iwinind, win.btot, feature_hours, "mean")

    # ------------------

    dwin = {'$<B_{tot}>$': feature_btotmean, 'btot_std': feature_btotstd, '$max(B_{tot})$': feature_btotmax, '$<B_{x}>$': feature_bxmean, 'bx_std': feature_bxstd, '$<B_{y}>$': feature_bymean, 'by_std': feature_bystd, '$<B_{z}>$': feature_bzmean, 'bz_std': feature_bzstd, '$<v_{tot}>$': feature_vtotmean, '$max(v_{tot})$': feature_vtotmax, 'vtot_std': feature_vtotstd, '<B> label': label_btotmean}

    dfwin = pd.DataFrame(data=dwin)
    pickle.dump(dfwin, open("mfr_predict/" + argv[0], "wb"))

    ################################ STEREO-A #############################
    feature_bzmean, feature_bzstd = get_feature(sta_time, icme_start_time_num, mo_start_time_num, n_istaind, sta.bz, feature_hours, "mean", "std")
    feature_bymean, feature_bystd = get_feature(sta_time, icme_start_time_num, mo_start_time_num, n_istaind, sta.by, feature_hours, "mean", "std")
    feature_bxmean, feature_bxstd = get_feature(sta_time, icme_start_time_num, mo_start_time_num, n_istaind, sta.bx, feature_hours, "mean", "std")
    feature_btotmean, feature_btotstd = get_feature(sta_time, icme_start_time_num, mo_start_time_num, n_istaind, sta.btot, feature_hours, "mean", "std")
    feature_btotmean, feature_btotmax, feature_btotstd = get_feature(sta_time, icme_start_time_num, mo_start_time_num, n_istaind, sta.btot, feature_hours, "mean", "max", "std")
    feature_vtotmean, feature_vtotmax, feature_vtotstd = get_feature(sta_time, icme_start_time_num, mo_start_time_num, n_istaind, sta.vtot, feature_hours, "mean", "std", "max")

    if mfr:
        feature_bzmean, feature_bzstd = get_feature(sta_time, mo_start_time_num, mo_start_time_num, n_istaind, sta.bz, feature_hours, "mean", "std")
        feature_bymean, feature_bystd = get_feature(sta_time, mo_start_time_num, mo_start_time_num, n_istaind, sta.by, feature_hours, "mean", "std")
        feature_bxmean, feature_bxstd = get_feature(sta_time, mo_start_time_num, mo_start_time_num, n_istaind, sta.bx, feature_hours, "mean", "std")
        feature_btotmean, feature_btotstd = get_feature(sta_time, mo_start_time_num, mo_start_time_num, n_istaind, sta.btot, feature_hours, "mean", "std")
        feature_btotmean, feature_btotmax, feature_btotstd = get_feature(sta_time, mo_start_time_num, mo_start_time_num, n_istaind, sta.btot, feature_hours, "mean", "max", "std")
        feature_vtotmean, feature_vtotmax, feature_vtotstd = get_feature(sta_time, mo_start_time_num, mo_start_time_num, n_istaind, sta.vtot, feature_hours, "mean", "std", "max")
    # ------------------
    # label

    label_btotmean = get_label(sta_time, mo_start_time_num, mo_end_time_num, n_istaind, sta.btot, feature_hours, "mean")
    # ------------------

    dsta = {'$<B_{tot}>$': feature_btotmean, 'btot_std': feature_btotstd, '$max(B_{tot})$': feature_btotmax, '$<B_{x}>$': feature_bxmean, 'bx_std': feature_bxstd, '$<B_{y}>$': feature_bymean, 'by_std': feature_bystd, '$<B_{z}>$': feature_bzmean, 'bz_std': feature_bzstd, '$<v_{tot}>$': feature_vtotmean, '$max(v_{tot})$': feature_vtotmax, 'vtot_std': feature_vtotstd, '<B> label': label_btotmean}

    dfsta = pd.DataFrame(data=dsta)
    pickle.dump(dfsta, open("mfr_predict/" + argv[1], "wb"))

    print('')
    print('feature hours = ', feature_hours)
    print('saved features and label')

    ################################ STEREO-B #############################
    feature_bzmean, feature_bzstd = get_feature(stb_time, icme_start_time_num, mo_start_time_num, n_istbind, stb.bz, feature_hours, "mean", "std")
    feature_bymean, feature_bystd = get_feature(stb_time, icme_start_time_num, mo_start_time_num, n_istbind, stb.by, feature_hours, "mean", "std")
    feature_bxmean, feature_bxstd = get_feature(stb_time, icme_start_time_num, mo_start_time_num, n_istbind, stb.bx, feature_hours, "mean", "std")
    feature_btotmean, feature_btotstd = get_feature(stb_time, icme_start_time_num, mo_start_time_num, n_istbind, stb.btot, feature_hours, "mean", "std")
    feature_btotmean, feature_btotmax, feature_btotstd = get_feature(stb_time, icme_start_time_num, mo_start_time_num, n_istbind, stb.btot, feature_hours, "mean", "max", "std")
    feature_vtotmean, feature_vtotmax, feature_vtotstd = get_feature(stb_time, icme_start_time_num, mo_start_time_num, n_istbind, stb.vtot, feature_hours, "mean", "std", "max")

    if mfr:
        feature_bzmean, feature_bzstd = get_feature(stb_time, mo_start_time_num, mo_start_time_num, n_istbind, stb.bz, feature_hours, "mean", "std")
        feature_bymean, feature_bystd = get_feature(stb_time, mo_start_time_num, mo_start_time_num, n_istbind, stb.by, feature_hours, "mean", "std")
        feature_bxmean, feature_bxstd = get_feature(stb_time, mo_start_time_num, mo_start_time_num, n_istbind, stb.bx, feature_hours, "mean", "std")
        feature_btotmean, feature_btotstd = get_feature(stb_time, mo_start_time_num, mo_start_time_num, n_istbind, stb.btot, feature_hours, "mean", "std")
        feature_btotmean, feature_btotmax, feature_btotstd = get_feature(stb_time, mo_start_time_num, mo_start_time_num, n_istbind, stb.btot, feature_hours, "mean", "max", "std")
        feature_vtotmean, feature_vtotmax, feature_vtotstd = get_feature(stb_time, mo_start_time_num, mo_start_time_num, n_istbind, stb.vtot, feature_hours, "mean", "std", "max")
    # ------------------
    # label

    label_btotmean = get_label(stb_time, mo_start_time_num, mo_end_time_num, n_istbind, stb.btot, feature_hours, "mean")
    # ------------------

    dstb = {'$<B_{tot}>$': feature_btotmean, 'btot_std': feature_btotstd, '$max(B_{tot})$': feature_btotmax, '$<B_{x}>$': feature_bxmean, 'bx_std': feature_bxstd, '$<B_{y}>$': feature_bymean, 'by_std': feature_bystd, '$<B_{z}>$': feature_bzmean, 'bz_std': feature_bzstd, '$<v_{tot}>$': feature_vtotmean, '$max(v_{tot})$': feature_vtotmax, 'vtot_std': feature_vtotstd, '<B> label': label_btotmean}

    dfstb = pd.DataFrame(data=dstb)
    pickle.dump(dfstb, open("mfr_predict/" + argv[2], "wb"))

    print('')
    print('feature hours = ', feature_hours)
    print('saved features and label')


if not features:
    print('')
    print("Features read in from featues_save.p")
    dfwin = pickle.load(open("mfr_predict/" + argv[0], "rb"))
    dfsta = pickle.load(open("mfr_predict/" + argv[1], "rb"))
    dfstb = pickle.load(open("mfr_predict/" + argv[2], "rb"))

print('NaNs in each feature/label - WIND')
print(dfwin.isnull().sum(axis=0))
# clean the data frame
dfwin = dfwin.dropna()
print('cleaned data frame - WIND')
print(dfwin.isnull().sum(axis=0))

print('NaNs in each feature/label - STA')
print(dfsta.isnull().sum(axis=0))
# clean the data frame
dfsta = dfsta.dropna()
print('cleaned data frame - STA')
print(dfsta.isnull().sum(axis=0))

print('NaNs in each feature/label - STB')
print(dfstb.isnull().sum(axis=0))
# clean the data frame
dfstb = dfstb.dropna()
print('cleaned data frame - STB')
print(dfstb.isnull().sum(axis=0))

# merge the dataframes
frames = [dfwin, dfsta, dfstb]
df = pd.concat(frames)

# ############################################ FEATURE SELECTION #############################
# feature selection will be done only on training data set

# -------------- split into train and test data sets -------------
win_train, win_test = train_test_split(dfwin, test_size=0.3, random_state=42)
sta_train, sta_test = train_test_split(dfsta, test_size=0.3, random_state=42)
stb_train, stb_test = train_test_split(dfstb, test_size=0.3, random_state=42)

train_frames = [win_train, sta_train, stb_train]
test_frames = [win_test, sta_test, stb_test]
train = pd.concat(train_frames)
test = pd.concat(test_frames)

# train, test = train_test_split(df, test_size=0.3, random_state=42)

print('')
print(train.info())
# print(win_train.describe())
# print(win_test.describe())
print('')

# ------------------------------------------------

# saves the index of the events, as appearing in the training set, to a numpy array -> can be used for plotting or other stuff later on
win_train_ind = win_train.index.to_numpy()
win_test_ind = win_test.index.to_numpy()
sta_train_ind = sta_train.index.to_numpy()
sta_test_ind = sta_test.index.to_numpy()
stb_train_ind = stb_train.index.to_numpy()
stb_test_ind = stb_test.index.to_numpy()


train_ind = train.index.to_numpy()
test_ind = test.index.to_numpy()

# -------------- calculate correlations -----------------------------
# plot correlation maps

spp = sns.pairplot(train, size=2.5)
spp.savefig('plots/Pairplot_fh=5.png')

# Pearson correlation is used
corr_mat = train.corr()
print(corr_mat["<B> label"].sort_values(ascending=False))

plt.figure(figsize=(4, 4))
svm = sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1)
figure = svm.get_figure()
figure.savefig('plots/CorrelationMatrix_fh=5.png', dpi=400)

# find correlated features
# features should be uncorrelated to each other
# if correlated, only one of them should be kept

correlated_features = set()  # empty set to put correlated features into

# loop through correlation matrix and find correlated features
# define threshold for correlation - let's say 0.5
print('')
for i in range(len(corr_mat.columns)):
    for j in range(len(corr_mat.columns)):
        if (i != j) and abs(corr_mat.iloc[i, j]) > 0.75:
            # print(i, j)
            colname = corr_mat.columns[i]
            correlated_features.add(colname)

print('')
print('features that are strongly correlated to each other, cc > 0.75')
print(correlated_features)
# correlation with output variable
corr_target = abs(corr_mat['<B> label'])
# selecting only features with correlation coefficient > 0.5
relevant_features = corr_target[corr_target > 0.5]
print('')
print('Most relevant features from correlation (corr coeff > 0.5)')
print(relevant_features)

# print(df[["LSTAT","PTRATIO"]].corr())
# print(df[["RM","LSTAT"]].corr())

# plt.figure(figsize=(4, 4))
# svm = sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1)
# figure = svm.get_figure()
# figure.savefig('CorrelationMatrix.png', dpi=400)

# #######################################################################

X_train = np.array(train['$<B_{tot}>$']).reshape(-1, 1)
y_train = np.array(train['<B> label']).reshape(-1, 1)

X_test = np.array(test['$<B_{tot}>$']).reshape(-1, 1)
y_test = np.array(test['<B> label']).reshape(-1, 1)

# =========================== SAVE TRAIN AND TEST DATA ==========================================
fname = 'mfr_predict/train_test_data_fh=5.p'
pickle.dump([n_iwinind, n_istaind, n_istbind, win_train_ind, win_test_ind, sta_train_ind, sta_test_ind, stb_train_ind, stb_test_ind, train_ind, test_ind, X_train, X_test, y_train, y_test, feature_hours], open(fname, 'wb'))

# plt.show()
