# MFRpred

This python package is used for space weather research.  We try to predict the magnetic field of the magnetic flux rope (MFR) 
within an interplanetary coronal mass ejection (ICME) at Earth (L1) with (1) machine learning algorithms, 
and (2) an analogue ensemble method. 

by U.V. Amerstorfer and [C. Möstl](https://www.iwf.oeaw.ac.at/en/user-site/christian-moestl/), IWF Graz, Austria.

Current status (March 2020): **Work in progress!** 

If you want to use parts of this code for generating results for peer-reviewed scientific publications, 
please contact us per email (ute.amerstorfer@oeaw.ac.at, christian.moestl@oeaw.ac.at) or via https://twitter.com/chrisoutofspace .


## Installation 

Install python 3.7.6 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

Create a conda environment:

	  conda env create -f environment.yml

	  conda activate helio

	  pip install -r requirements.txt
	  
go to a directory of your choice

	  git clone https://github.com/uvamerstorfer/MFRpred


Before running the scripts, you need to download three data files (in total 1.5 GB) from this figshare repository, 

    https://doi.org/10.6084/m9.figshare.11841156.v1

and place them in the data/ folder.

    data/STA_2007to2015_SCEQ.p
    data/STB_2007to2014_SCEQ.p
    data/WIND_2007to2018_HEEQ.p 
    
    
     
  
## 1. Machine learning  
### Brief instruction for running the scripts  

Please run the scripts in the following order, make sure you have the conda helio environment activated:

    mfr_prepData.py
    mfr_featureSelection.py 
    mfr_findModel.py 
    mfr_prediction.py 


### mfr_prepData.py

Start with 

    python mfr_prepData.py
    
to prepare the data.

### mfr_featureSelection.py

Continue with

    python mfr_featureselection.py wind_features.p sta_features.p stb_features.p --features
 
The first three arguments need to be file names ending in .p (for python pickle) into which the features are saved (at first run of the script) 
or from which the features are read in (see below), e.g. for Wind its wind_features.p, for STEREO-A sta_features.p, and for STEREO-B stb_features.p, in exactly that order.


**--features**: set this if features need to be determined. If set, the code will produce a pickle-file with the features and the labels. 
If --features is not set, then they will be read them from an already existing pickle-file. 

**--mfr**: We try out different features from different regions of the MFR - 
only sheath features, sheath and MFR features, only MFR features. For the third case, you need to set --mfr. 
If you want to use MFR features, you also have to specify the variabel *feature_hours* in the file *input.py*. 
This parameter how much time of the MFR is taken for the feature; e.g. feature_hours=0 means only the sheath 
and no part of the MFR is taken, feature_hours=5 means the first five hours of the MFR are taken.

The features,train and test data sets are saved in pickle-files in a subdirectory 

    mfr_predict/
    
The corresponding plots are saved in a subdirectory 

    plots/


### mfr_findModel.py
To run mfr_findModel.py, two input parameters need to be specified:  
<ul>
<li> The first one is the pickle-file with the train and test data:  
print('Read in test and train data from:', argv[0])  
<li> The second on is the pickle-file, to which the selected final model will be saved to:  
print('Save final model to:', argv[1])  
</ul>

Both files need again the subdirectory mfr_predict.  

### mfr_prediction.py
To run mfr_prediction.py, five input parameters need to be specified:  
<ul>
<li> First, again the pickle-file with train and test data.  
<li> Second, the pickle-file with the final model.   
<li> Third, file name where plots from analysis of WIND data will be saved to (png).  
<li> Fourth, file name where plots from analysis of STA data will be saved to (png).  
<li> Fifth, file name where plots from analysis of STB data will be saved to (png).  
</ul>

Both subdirectories, mfr_prdict and plots, are needed here.  
