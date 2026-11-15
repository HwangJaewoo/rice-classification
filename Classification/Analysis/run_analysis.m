% run_analysis.m : Analyses of the saved nested results (run after run_single, run_mixed and run_cross)
clc;

%% Confused cultivar pairs and confusion matrices
confusion_pairs(20);

%% Variance components, heritability and shift of the indices between the years
year_stability();

%% Predictions of every test plant in one CSV file
export_predictions();

%% Redundancy of the color and morphological features (correlation, effective dimension)
index_redundancy();
