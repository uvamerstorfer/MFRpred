python mfr_prepData.py
python mfr_featureSelection.py wind_features.p sta_features.p stb_features.p --features
python mfr_findModel.py train_test_data_fh=5.p model1.p
python mfr_prediction.py train_test_data_fh=5.p model1.p wind_plot.png sta_plot.png stb_plot.png