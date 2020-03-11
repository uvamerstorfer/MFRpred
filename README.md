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

    python mfr_prepData.py
    python mfr_featureSelection.py 
    python mfr_findModel.py 
    python mfr_prediction.py 


#### mfr_featureSelection.py
To run mfr_featureSelection.py, the following input parameters need to be specified:  
  
The first three arguments need to be file names to save features into (at first run of the script) or from which the features are read in (subsequent runs of the script - see below at --features):
<ul>
<li> WIND features: argv[0]  
<li> STA features: argv[1]  
<li> STB features: argv[2]  
</ul>
Then --features can be given as parameter, if features need to be determined. If --features is set, then the code will produce a pickle-file with the features and the label. If --features is not set, then the code will read from an already existing pickle-file (in other words, if you already calculated the features, you don't need to do this step again when running the script again). 

The last input parameter can be--mfr. We try out different features from different regions of the MFR - only sheath features, sheath and MFR features, only MFR features. For the third case, you need to give --mfr. 

If you want to use MFR features, you also have to specify a certain parameter INSIDE the script (that's not optimal so far and surely will be changed in the future) - namely feature_hours. This parameter so-to-speak specifies how much of the MFR is taken for the feature (e.g. feature_hours=0 means only the sheath and no part of the MFR is taken, feature_hours=1 means the first hour of the MFR is taken, etc.).

The features are saved in pickle-files in a subdirectory mfr_predict.  
The train and test data sets are also saved in this subdirectory.  
The corresponding plots are saved in a subdirectory plots.   
Both directories need to be created manually at the moment.  

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
