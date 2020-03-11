# MFRpred
Trying to predict the magnetic field of the magnetic flux rope (MFR) within an ICME at Earth (L1) with (1) machine learning algorithms, and (2) analogue ensemble method. 

If you clone the repository, it still won't work as it is, since there are a couple of pickle-files missing - unfortunately, they are too large to bit put on git (STA_2007to2015_SCEQ.p, STB_2007to2014_SCEQ.p, WIND_2007to2018_HEEQ.p). The missing pickle files are in a directory one level above the main scripts directory (../catpy/DATACAT/). The file and directory structure is important, since this structure is implemented in a static way, with paths directly coded in the Python scripts. 
  
# (1) Machine learning  
## Brief instruction for running the scripts  

Please run the scripts in the following order:  
<ol>
  <li> mfr_prepData.py <\li>
  <li> mfr_featureSelection.py <\li>
  <li> mfr_findModel.py <\li>
  <li> mfr_prediction.py <\li>
<\ol>

### ad 2.  
To run mfr_featureSelection.py, the following input parameters need to be specified:  
  
The first three arguments need to be file names to save features into (at first run of the script) or from which the features are read in (subsequent runs of the script - see below at --features)  
WIND features: argv[0]  
STA features: argv[1]  
STB features: argv[2]  

Then --features can be given as parameter, if features need to be determined. If --features is set, then the code will produce a pickle-file with the features and the label. If --features is not set, then the code will read from an already existing pickle-file (in other words, if you already calculated the features, you don't need to do this step again when running the script again). 

The last input parameter can be--mfr. We try out different features from different regions of the MFR - only sheath features, sheath and MFR features, only MFR features. For the third case, you need to give --mfr. 

If you want to use MFR features, you also have to specify a certain parameter INSIDE the script (that's not optimal so far and surely will be changed in the future) - namely feature_hours. This parameter so-to-speak specifies how much of the MFR is taken for the feature (e.g. feature_hours=0 means only the sheath and no part of the MFR is taken, feature_hours=1 means the first hour of the MFR is taken, etc.).

The features are saved in pickle-files in a subdirectory mfr_predict.  
The train and test data sets are also saved in this subdirectory.  
The corresponding plots are saved in a subdirectory plots.   
Both directories need to be created manually at the moment.  

### ad 3.  
To run mfr_findModel.py, two input parameters need to be specified:  

The first one is the pickle-file with the train and test data:  
print('Read in test and train data from:', argv[0])  
The second on is the pickle-file, to which the selected final model will be saved to:  
print('Save final model to:', argv[1])  

Both files need again the subdirectory mfr_predict.  

### ad 4.  
To run mfr_prediction.py, five input parameters need to be specified:  

First, again the pickle-file with train and test data.  
Second, the pickle-file with the final model.   
Third, file name where plots from analysis of WIND data will be saved to (png).  
Fourth, file name where plots from analysis of STA data will be saved to (png).  
Fifth, file name where plots from analysis of STB data will be saved to (png).  

Both subdirectories, mfr_prdict and plots, are needed here.  
