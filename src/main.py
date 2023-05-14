import argparse, sys
import os
import yaml
import pandas as pd
import numpy as np
import time

from data_cleaning import drop_correlated_features
from data_cleaning import group_categorical_features
from data_cleaning import  prepare_data
from feature_selection import get_top_k_features_using_rfe_cv, get_top_k_features_using_rfe, plot_rfecv_scoring
from feature_selection import get_top_k_features_using_mi
from feature_engineering import encode_train_data, encode_test_data
from feature_engineering import normalize_train_data, normalize_test_data
from feature_engineering import get_quality_of_superstructure, get_risk_status_based_on_geo_level
from feature_engineering import dimensionality_reduction
import modelling

# Track time for execution
start_time = time.time()

# Parse config
print("Parse Config ...")
parser=argparse.ArgumentParser()
parser.add_argument("--config", help="Path to the config file")
args=parser.parse_args()

# Check if the given config file exists, if not use sample config
if not args.config:
    raise FileNotFoundError(f"{args.config} is missing.")

# Read config file
with open(args.config, "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.FullLoader)

# ToDo: Verify options of config (in the end, when all valid options are known)
# Check if data in config exists and read it
print("Read and Load Data ...")
train_values_path = cfg["paths"]["train_values"]
train_labels_path = cfg["paths"]["train_labels"]
test_values_path = cfg["paths"]["test_values"]
result_path = cfg["paths"]["result"]

if not os.path.normpath(train_values_path):
    raise FileNotFoundError(f"{train_values_path} is missing.")
if not os.path.normpath(train_labels_path):
    raise FileNotFoundError(f"{train_labels_path} is missing.")
if not os.path.normpath(test_values_path):
    raise FileNotFoundError(f"{test_values_path} is missing.")
    
# Load data
print("Loading Data ...")
train_values = pd.read_csv(train_values_path)
train_labels = pd.read_csv(train_labels_path)
test_values = pd.read_csv(test_values_path)
train_values.set_index("building_id", inplace=True)
test_values.set_index("building_id", inplace=True)

# Make Sample Size smaller for experimenting and testing; Keep commented!
#train_labels, train_values = stratify_dataframe_by_damage_grade(train_values, train_labels, 42)
#train_values = train_values.iloc[:15000]
#test_values = test_values.iloc[:7000]
#train_labels = train_labels.iloc[:15000]

# Data cleaning
# Prepare raw data
print("Cleaning Train Data ...")
binary_encoded_cols = [x for x in train_values.columns if x.startswith("has_")]
columns_to_ignore = cfg.get("data_cleaning", "NO DATA CLEANING DEFINED!").get("columns_to_ignore")
train_data_cleaned = prepare_data(df=train_values, config=cfg,
                                  ignore_cols=columns_to_ignore+binary_encoded_cols,
                                  outlier_method="replace")
print("Cleaning Test Data ...")
test_data_cleaned = prepare_data(df=test_values, config=cfg,
                                  ignore_cols=columns_to_ignore+binary_encoded_cols,
                                  outlier_method="replace")

# Correlated features
print("Drop correlated features...")
train_data_cleaned = drop_correlated_features(data=train_data_cleaned, config=cfg["data_cleaning"]["correlations"])
test_data_cleaned = drop_correlated_features(data=test_data_cleaned, config=cfg["data_cleaning"]["correlations"])

# Group categorical features with rarely occurring realizations
if not cfg["feature_engineering"]["group_categorical"]["skip"]:
    print("Grouping categorical features ...")
    train_data_cleaned = group_categorical_features(df=train_data_cleaned, default_val="others", verbose=False)
    test_data_cleaned = group_categorical_features(df=test_data_cleaned, default_val="others", verbose=False)

# Add new features for risk status
if not cfg["feature_engineering"]["risk_status"]["skip"]:
    print("Add risk status features...")
    test_data_cleaned = get_risk_status_based_on_geo_level(data=train_values, df_to_add_info=test_data_cleaned, labels=train_labels, geo_level=cfg["feature_engineering"]["risk_status"]["geo_level"])
    train_data_cleaned = get_risk_status_based_on_geo_level(data=train_values, df_to_add_info=train_data_cleaned, labels=train_labels, geo_level=cfg["feature_engineering"]["risk_status"]["geo_level"])

# Add superstructure quality
if not cfg["feature_engineering"]["superstructure_quality"]["skip"]:
    print("Add superstructure quality feature...")
    train_data_cleaned = get_quality_of_superstructure(raw_data=train_values, df_to_add_info=train_data_cleaned)
    test_data_cleaned = get_quality_of_superstructure(raw_data=test_values, df_to_add_info=test_data_cleaned)

# Apply One Hot Encoding on categorical features
if not cfg["feature_engineering"]["categorical_encoding"]["skip"]:
    if cfg["feature_engineering"]["categorical_encoding"]["method"] == "One-Hot":
        print("One Hot Encoding features ...")
        train_data_cleaned, ohe = encode_train_data(x_train=train_data_cleaned)
        test_data_cleaned = encode_test_data(x_test=test_data_cleaned, ohe=ohe)

# Apply StandardScaler (method="standard") or MinMax Scaler (method="minmax") on Features
if not cfg["feature_engineering"]["normalize"]["skip"]:
    print("Normalizing Data ...")
    train_data_cleaned, scaler = normalize_train_data(x_train=train_data_cleaned, method=cfg["feature_engineering"]["normalize"]["method"])
    test_data_cleaned = normalize_test_data(x_test=test_data_cleaned, scaler=scaler)

best_feats = []
# Feature Selection: Get top k features using RFE, RFECV, or use MI
if not cfg["feature_engineering"]["feature_selection"]["skip"]:
    if cfg["feature_engineering"]["feature_selection"]["method"] == "RFECV":
        print("Selecting best features using RFE CV ...")
        best_feats, rfecv = get_top_k_features_using_rfe_cv(x_train=train_data_cleaned,
                                                            y_train=train_labels,
                                                            min_features_to_select=5,
                                                            k_folds=5,
                                                            scoring="matthews_corrcoef",
                                                            step=cfg["feature_engineering"]["feature_selection"]["step"],
                                                            verbose=0)
        #plot_rfecv_scoring(rfecv)

    if cfg["feature_engineering"]["feature_selection"]["method"] == "RFE":
        print("Selecting best features using RFE...")
        best_feats, rfe = get_top_k_features_using_rfe(x_train=train_data_cleaned,
                                                         y_train=train_labels,
                                                         k=cfg["feature_engineering"]["feature_selection"]["k"], 
                                                         step=cfg["feature_engineering"]["feature_selection"]["step"], 
                                                         verbose=0)

    if cfg["feature_engineering"]["feature_selection"]["method"] == "MI":
        print("Selecting best features using MI...")
        best_feats, mi_scores = get_top_k_features_using_mi(x_train=train_data_cleaned,
                                                        y_train=train_labels,
                                                        k=cfg["feature_engineering"]["feature_selection"]["k"])

    #print(f"*** Number of best selected features: {rfecv.n_features_} of {rfecv.n_features_in_} in total ***")
    print(f"\nSelected feature set: {best_feats}\n")

    # Keep best columns
    train_data_cleaned = train_data_cleaned[train_data_cleaned.columns.intersection(best_feats)]
    test_data_cleaned = test_data_cleaned[test_data_cleaned.columns.intersection(best_feats)]

if not cfg["feature_engineering"]["dimensionality_reduction"]["skip"]:
    if len(best_feats) > cfg["feature_engineering"]["dimensionality_reduction"]["feature_threshold"]:
        print(f"Perform dimensionality reduction...")
        train_data_cleaned, test_data_cleaned = dimensionality_reduction(train_data=train_data_cleaned,
                                                                         test_data=test_data_cleaned, 
                                                                         method=cfg["feature_engineering"]["dimensionality_reduction"]["method"], 
                                                                         n_components=cfg["feature_engineering"]["dimensionality_reduction"]["n_components"])

# Model training: TBD
print("Modelling ...")
train_data_cleaned = train_data_cleaned.astype(np.float64)

# Return fitted model
model = modelling.hyperparameter_optimization(model="XGBoost",
                                              train_data=train_data_cleaned,
                                              train_labels=train_labels,
                                              scoring=cfg["modelling"]["scoring"])

# Make prediction: TBD
print("Make predictions ...")
modelling.make_prediction(model=model, test_data=test_data_cleaned, result_path=result_path)

# Track time for execution and print
end_time = time.time()
run_time = end_time - start_time
print(f"-- Pipeline took {run_time:.2f} seconds --")
