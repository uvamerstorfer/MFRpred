# !/usr/bin/python
# coding: utf8

"""
Prediction of flux rope magnetic fields using the past analogue ensemble method

Author: U.V. Amerstorfer, Space Research Institute IWF Graz, Austria
Last update: Mar 2020

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

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


sns.set_context("talk")
# sns.set_style("darkgrid")
# sns.set_context("notebook", font_scale=0.4, rc={"lines.linewidth": 2.5})

plt.close('all')

# READ INPUT OPTIONS FROM COMMAND LINE
# argv = sys.argv[1:]

# ####################### functions ###############################################
# formatter function for ticks labels for plots


def hourMin(x, pos):
    temp = mdates.num2date(x)
    return str(temp.hour) + ':' + str(temp.minute)

#####################################################################################
# ####################### main program ###############################################
#####################################################################################

# ############################# get MFR duration ################################

mo_duration = pickle.load(open("../catpy/DATACAT/icme_mo_duration.p", "rb"))

# ############################# get spacecraft data ################################
# we use the HELCAT files only to get the times and indices of the events
# then we use other data files to get the spacecraft data


# get ICME times
print('get ICME times')
[icme_start_time_num, icme_end_time_num, mo_start_time_num, mo_end_time_num, iwinind, istaind, istbind] = pickle.load(open("../catpy/DATACAT/icme_times.p", "rb"))
[icme_start_time, icme_end_time, mo_start_time, mo_end_time] = pickle.load(open("../catpy/DATACAT/icme_times_string.p", "rb"))
print('get ICME times done')

# ############################# get Wind data ################################

print('read Wind data')
# get insitu date - resolution 1 min
win = pickle.load(open("../catpy/DATACAT/WIND_2007to2018_HEEQ.p", "rb"))
[win_time] = pickle.load(open("../catpy/DATACAT/insitu_times_mdates_win_2007_2018.p", "rb"))
print('read data done')

# ############################# get Stereo-A data ################################

print('read Stereo-A data')
# get insitu data
sta = pickle.load(open("../catpy/DATACAT/STA_2007to2015_SCEQ.p", "rb"))
[sta_time] = pickle.load(open("../catpy/DATACAT/insitu_times_mdates_sta_2007_2015.p", "rb"))
print('read data done')

# ############################# get Stereo-B data ################################

print('read Stereo-B data')
# get insitu data
stb = pickle.load(open("../catpy/DATACAT/STB_2007to2014_SCEQ.p", "rb"))
[stb_time] = pickle.load(open("../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "rb"))
print('read data done')

################################ times for plotting #############################

winTime = mdates.num2date(win_time)
############################ get data in training window around mfr arrival #########################################
# training window (tw) -> 24 hours prior to mfr arrival to 0.25 hours after mfr arrival
tw_hrs_start = 24  # hours
tw_hrs_end = 0.25  # hours

btot_tw = np.zeros(shape=(int((tw_hrs_start + tw_hrs_end) * 60), np.size(iwinind)))
bz_tw = np.zeros(shape=(int((tw_hrs_start + tw_hrs_end) * 60), np.size(iwinind)))
dens_tw = np.zeros(shape=(int((tw_hrs_start + tw_hrs_end) * 60), np.size(iwinind)))
vtot_tw = np.zeros(shape=(int((tw_hrs_start + tw_hrs_end) * 60), np.size(iwinind)))

for p in np.arange(0, np.size(iwinind)):
    btot_tw_temp = win.btot[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] - tw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + tw_hrs_end / 24.0)))]
    btot_tw[:, p] = btot_tw_temp

    bz_tw_temp = win.bz[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] - tw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + tw_hrs_end / 24.0)))]
    bz_tw[:, p] = bz_tw_temp

    dens_tw_temp = win.density[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] - tw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + tw_hrs_end / 24.0)))]
    dens_tw[:, p] = dens_tw_temp

    vtot_tw_temp = win.vtot[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] - tw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + tw_hrs_end / 24.0)))]
    vtot_tw[:, p] = vtot_tw_temp

############################ get data in forecast window after mfr arrival #########################################
# forecast window (fw) -> 0.25 hours after to 24 hours after mfr arrival
fw_hrs_start = 0.25  # hours
fw_hrs_end = 24  # hours

btot_fw = np.zeros(shape=(int((fw_hrs_end - fw_hrs_start) * 60), np.size(iwinind)))
bz_fw = np.zeros(shape=(int((fw_hrs_end - fw_hrs_start) * 60), np.size(iwinind)))
dens_fw = np.zeros(shape=(int((fw_hrs_end - fw_hrs_start) * 60), np.size(iwinind)))
vtot_fw = np.zeros(shape=(int((fw_hrs_end - fw_hrs_start) * 60), np.size(iwinind)))

for p in np.arange(0, np.size(iwinind)):
    btot_fw_temp = win.btot[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] + fw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + fw_hrs_end / 24.0)))]
    btot_fw[:, p] = btot_fw_temp

    bz_fw_temp = win.bz[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] + fw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + fw_hrs_end / 24.0)))]
    bz_fw[:, p] = bz_fw_temp

    dens_fw_temp = win.density[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] + fw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + fw_hrs_end / 24.0)))]
    dens_fw[:, p] = dens_fw_temp

    vtot_fw_temp = win.vtot[np.where(np.logical_and(win_time >= (mo_start_time_num[iwinind[p]] + fw_hrs_start / 24.0), win_time < (mo_start_time_num[iwinind[p]] + fw_hrs_end / 24.0)))]
    vtot_fw[:, p] = vtot_fw_temp

################################ PANDAS DATAFRAME #############################

# dwin_tw = {'$B_{tot}$': btot_tw, '$B_{z}$': bz_tw, '$v_{tot}$': vtot_tw, 'density': dens_tw}

# dfwin_tw = pd.DataFrame(data=dwin_tw, index=dwin_tw['density'][0, 1:], columns=dwin_tw['density'][1:, 0])

# dwin_fw = {'$B_{tot}$': btot_fw, '$B_{z}$': bz_fw, '$v_{tot}$': vtot_fw, 'density': dens_fw}
# ############################### PLOTS #############################
# WIND
# get number of rows for figure
nRows_wind = np.size(iwinind)

for iEv in range(0, nRows_wind):
    # for iEv in range(0, 1):

    windowStart = np.where(win_time >= (mo_start_time_num[iEv] - tw_hrs_start / 24.0))[0][0]
    windowEnd = np.where(win_time < (mo_start_time_num[iEv] + tw_hrs_end / 24.0))[0][-1]

    mostart = np.where(win_time >= mo_start_time_num[iEv])[0][0]
    moend = np.where(win_time >= mo_end_time_num[iEv])[0][0]

    # winTime[mostart] is start date and time of MFR
    # winTime[moend] is end date and time of MFR

    # for xlabel to have date
    start_year = str(winTime[windowStart].year)
    start_month = str(winTime[windowStart].month)
    start_day = str(winTime[windowStart].day)

    # for xticks labels to show hour and minute
    start_hr = str(winTime[windowStart].hour)
    start_min = str(winTime[windowStart].minute)
    start_time = start_hr + ':' + start_min
    end_hr = str(winTime[windowEnd].hour)
    end_min = str(winTime[windowEnd].minute)
    end_time = end_hr + ':' + end_min

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20, 24))

    ax1.plot(win_time[int(windowStart):int(windowEnd)], btot_tw[:-1, iEv], label='B$_{tot}$')
    ax1.plot(win_time[int(windowStart):int(windowEnd)], bz_tw[:-1, iEv], label='B$_{z}$')
    ax1.axvline(x=win_time[int(mostart)], color='r', linestyle='--')
    ax1.axvline(x=win_time[int(moend)], color='r', linestyle='--')
    # # ax1.set_xlabel('Start Time (' + start_month + '/' + start_day + '/' + start_year + ' ' + start_hr + ':' + start_min + ')')
    ax1.set_ylabel('B [nT]')
    ax1.set_xlim([win_time[int(windowStart)], win_time[int(windowEnd)]])

    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.1))  # want to have ticks every 1/10th day
    # define formatter function for tick-labels
    fmtr = plt.FuncFormatter(hourMin)
    _ = ax1.xaxis.set_major_formatter(fmtr)

    ax1.legend(numpoints=1, ncol=2, loc='best')

    ax2.plot(win_time[int(windowStart):int(windowEnd)], vtot_tw[:-1, iEv], label='v$_{tot}$')
    ax2.axvline(x=win_time[int(mostart)], color='r', linestyle='--')
    ax2.axvline(x=win_time[int(moend)], color='r', linestyle='--')
    # # ax2.set_xlabel('Start Time (' + start_month + '/' + start_day + '/' + start_year + ' ' + start_hr + ':' + start_min + ')')
    ax2.set_ylabel('v [km/s]')
    ax2.set_xlim([win_time[int(windowStart)], win_time[int(windowEnd)]])

    ax2.legend(numpoints=1, ncol=2, loc='best')

    ax3.plot(win_time[int(windowStart):int(windowEnd)], dens_tw[:-1, iEv], label='density')
    ax3.axvline(x=win_time[int(mostart)], color='r', linestyle='--')
    ax3.axvline(x=win_time[int(moend)], color='r', linestyle='--')
    ax3.set_xlabel('Start Time (' + start_month + '/' + start_day + '/' + start_year + ' ' + start_hr + ':' + start_min + ')')
    ax3.set_ylabel('density [cm$^{-3}$]')
    ax3.set_xlim([win_time[int(windowStart)], win_time[int(windowEnd)]])

    ax3.legend(numpoints=1, ncol=2, loc='best')

    plt.savefig('plots/AnEn_trainingWindow_Event' + str(iEv) + '.png', bbox_inches='tight')
