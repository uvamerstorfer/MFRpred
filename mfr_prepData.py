# !/usr/bin/python
# coding: utf8

"""
This file is part of the mfr_predict-package. It prepares the data used to train the machine
learning model.

Analyses HELCATS ICMECAT for predicting labels of CME MFRs
Authors: C. Moestl, U.V. Amerstorfer, Space Research Institute IWF Graz, Austria
Last update: March 2020

"""
import scipy.io
from sunpy.time import parse_time
import numpy as np
import matplotlib.dates as mdates
import time
import pickle
import os

import warnings
warnings.filterwarnings('ignore')

# ####################### functions ###############################################


def getcat(filename):
    print('reading CAT')
    cat = scipy.io.readsav(filename, verbose='true')
    print('done CAT')
    return cat


def decode_array(bytearrin):
    '''
    for decoding the strings from the IDL .sav file to a list of python strings, not bytes
    make list of python lists with arbitrary length
    '''
    bytearrout = ['' for x in range(len(bytearrin))]
    for i in range(0, len(bytearrin)):
        bytearrout[i] = bytearrin[i].decode()
    # has to be np array so to be used with numpy "where"
    bytearrout = np.array(bytearrout)
    return bytearrout



#####################################################################################
# ####################### main program ###############################################
#####################################################################################

print('MFR classify.')

# solar radius
Rs_in_AU = 7e5 / 149.5e6

if not os.path.isdir('mfr_predict'):
    os.mkdir('mfr_predict')

filename_icmecat = 'data/HELCATS_ICMECAT_v10_SCEQ.sav'
i = getcat(filename_icmecat)

# now this is a scipy structured array
# access each element of the array see http://docs.scipy.org/doc/numpy/user/basics.rec.html
# access variables
# i.icmecat['id']
# look at contained variables
# print(i.icmecat.dtype)

# get spacecraft and planet positions
pos = getcat('data/positions_2007_2023_HEEQ_6hours.sav')

pos_time= decode_array(pos.time[0:-1]) 
pos_time_num=parse_time(pos_time[0:-1]).plot_date          

# ----------------- get all parameters from ICMECAT for easier handling

# id for each event
iid = i.icmecat['id']
# need to decode all strings
iid = decode_array(iid)

# observing spacecraft
isc = i.icmecat['sc_insitu']  # string
isc = decode_array(isc)

# all times need to be converted from the IDL format to matplotlib format
icme_start_time = i.icmecat['ICME_START_TIME']
icme_start_time_num = parse_time(decode_array(icme_start_time)).plot_date

mo_start_time = i.icmecat['MO_START_TIME']
mo_start_time_num = parse_time(decode_array(mo_start_time)).plot_date

mo_end_time = i.icmecat['MO_END_TIME']
mo_end_time_num = parse_time(decode_array(mo_end_time)).plot_date

# this time exists only for Wind, so set all other times to nan
icme_end_time = i.icmecat['ICME_END_TIME']
iwinind = np.where(isc == 'Wind')[0]
icme_end_time_dum=decode_array(icme_end_time)
icme_end_time_num=np.zeros(len(isc))*np.nan
icme_end_time_num[0:len(iwinind)] = parse_time(icme_end_time_dum[iwinind]).plot_date

sc_heliodistance = i.icmecat['SC_HELIODISTANCE']
sc_long_heeq = i.icmecat['SC_LONG_HEEQ']
sc_lat_heeq = i.icmecat['SC_LAT_HEEQ']
mo_bmax = i.icmecat['MO_BMAX']
mo_bmean = i.icmecat['MO_BMEAN']
mo_bstd = i.icmecat['MO_BSTD']
mo_bzmean = i.icmecat['MO_BZMEAN']
mo_bzmin = i.icmecat['MO_BZMIN']
mo_duration = i.icmecat['MO_DURATION']
mo_mva_axis_long = i.icmecat['MO_MVA_AXIS_LONG']
mo_mva_axis_lat = i.icmecat['MO_MVA_AXIS_LAT']
mo_mva_ratio = i.icmecat['MO_MVA_RATIO']
sheath_speed = i.icmecat['SHEATH_SPEED']
sheath_speed_std = i.icmecat['SHEATH_SPEED_STD']
mo_speed = i.icmecat['MO_SPEED']
mo_speed_st = i.icmecat['MO_SPEED_STD']
sheath_density = i.icmecat['SHEATH_DENSITY']
sheath_density_std = i.icmecat['SHEATH_DENSITY_STD']
mo_density = i.icmecat['MO_DENSITY']
mo_density_std = i.icmecat['MO_DENSITY_STD']
sheath_temperature = i.icmecat['SHEATH_TEMPERATURE']
sheath_temperature_std = i.icmecat['SHEATH_TEMPERATURE_STD']
mo_temperature = i.icmecat['MO_TEMPERATURE']
mo_temperature_std = i.icmecat['MO_TEMPERATURE_STD']
#sheath_pdyn = i.icmecat['SHEATH_PDYN']
#sheath_pdyn_std = i.icmecat['SHEATH_PDYN_STD']
#mo_pdyn = i.icmecat['MO_PDYN']
#mo_pdyn_std = i.icmecat['MO_PDYN_STD']

# get indices of events by different spacecraft
ivexind = np.where(isc == 'VEX')[0]
istaind = np.where(isc == 'STEREO-A')[0]
istbind = np.where(isc == 'STEREO-B')[0]
iwinind = np.where(isc == 'Wind')[0]
imesind = np.where(isc == 'MESSENGER')[0]
iulyind = np.where(isc == 'ULYSSES')[0]
imavind = np.where(isc == 'MAVEN')[0]

# take MESSENGER only at Mercury, only events after orbit insertion
imercind = np.where(np.logical_and(isc == 'MESSENGER', icme_start_time_num > parse_time('2011-03-18').plot_date))

# limits of solar minimum, rising phase and solar maximum

minstart = parse_time('2007-01-01').plot_date
minend =   parse_time('2009-12-31').plot_date

risestart = parse_time('2010-01-01').plot_date
riseend =   parse_time('2011-06-30').plot_date

maxstart =  parse_time('2011-07-01').plot_date
maxend =    parse_time('2014-12-31').plot_date

# extract events by limits of solar min, rising, max, too few events for MAVEN and Ulysses

iallind_min = np.where(np.logical_and(icme_start_time_num > minstart, icme_start_time_num < minend))[0]
iallind_rise = np.where(np.logical_and(icme_start_time_num > risestart, icme_start_time_num < riseend))[0]
iallind_max = np.where(np.logical_and(icme_start_time_num > maxstart, icme_start_time_num < maxend))[0]

iwinind_min = iallind_min[np.where(isc[iallind_min] == 'Wind')[0]]
iwinind_rise = iallind_rise[np.where(isc[iallind_rise] == 'Wind')[0]]
iwinind_max = iallind_max[np.where(isc[iallind_max] == 'Wind')[0]]

ivexind_min = iallind_min[np.where(isc[iallind_min] == 'VEX')[0]]
ivexind_rise = iallind_rise[np.where(isc[iallind_rise] == 'VEX')[0]]
ivexind_max = iallind_max[np.where(isc[iallind_max] == 'VEX')[0]]

imesind_min = iallind_min[np.where(isc[iallind_min] == 'MESSENGER')[0]]
imesind_rise = iallind_rise[np.where(isc[iallind_rise] == 'MESSENGER')[0]]
imesind_max = iallind_max[np.where(isc[iallind_max] == 'MESSENGER')[0]]

istaind_min = iallind_min[np.where(isc[iallind_min] == 'STEREO-A')[0]]
istaind_rise = iallind_rise[np.where(isc[iallind_rise] == 'STEREO-A')[0]]
istaind_max = iallind_max[np.where(isc[iallind_max] == 'STEREO-A')[0]]

istbind_min = iallind_min[np.where(isc[iallind_min] == 'STEREO-B')[0]]
istbind_rise = iallind_rise[np.where(isc[iallind_rise] == 'STEREO-B')[0]]
istbind_max = iallind_max[np.where(isc[iallind_max] == 'STEREO-B')[0]]

# select the events at Mercury extra after orbit insertion, no events for solar minimum!
imercind_min = iallind_min[np.where(np.logical_and(isc[iallind_min] == 'MESSENGER', icme_start_time_num[iallind_min] >     parse_time('2011-03-18').plot_date))]
imercind_rise = iallind_rise[np.where(np.logical_and(isc[iallind_rise] == 'MESSENGER', icme_start_time_num[iallind_rise] > parse_time('2011-03-18').plot_date))]
imercind_max = iallind_max[np.where(np.logical_and(isc[iallind_max] == 'MESSENGER', icme_start_time_num[iallind_max] >     parse_time('2011-03-18').plot_date))]

################################ save MFR duration  #############################
print('save MFR duration')
pickle.dump([mo_duration], open("data/icme_mo_duration.p", "wb"))
print('save MFR duration done')
# ############################## save ICME times #############################

print('save ICME times')
# save ICME times and indices of spacecraft events
pickle.dump([icme_start_time_num, icme_end_time_num, mo_start_time_num, mo_end_time_num, iwinind, istaind, istbind], open("data/icme_times.p", "wb"))
pickle.dump([icme_start_time, icme_end_time, mo_start_time, mo_end_time], open("data/data_icme_times_string.p", "wb"))
print('save ICME times done')

# ############################# save spacecraft data ################################
# we use the HELCAT files only to get the times and indices of the events
# now we use other data files to get the spacecraft data

# ############################# save Wind data ################################

print('save Wind data')
# save insitu data
win = pickle.load(open("data/WIND_2007to2018_HEEQ.p", "rb"))
win_time = parse_time(win.time,format='utime').datetime
pickle.dump([win_time], open("data/insitu_times_mdates_win_2007_2018.p", "wb"))
print('save data done')

# ############################# save Stereo-A data ################################

print('save Stereo-A data')
# save insitu data
sta = pickle.load(open("data/STA_2007to2015_SCEQ.p", "rb"))
sta_time = parse_time(sta.time,format='utime').datetime
pickle.dump([sta_time], open("data/insitu_times_mdates_sta_2007_2015.p", "wb"))
print('save data done')

# ############################# save Stereo-B data ################################

print('save Stereo-B data')
# save insitu data
stb = pickle.load(open("data/STB_2007to2014_SCEQ.p", "rb"))
stb_time = parse_time(stb.time,format='utime').datetime
pickle.dump([stb_time], open("data/insitu_times_mdates_stb_2007_2014.p", "wb"))
print('save data done')
