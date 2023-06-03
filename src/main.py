""" Main file for the project. This file is used to run the whole pipeline. """
import argparse
import time

import numpy as np
import pandas as pd

import modelling
from data_cleaning import drop_correlated_features
from data_cleaning import group_categorical_features
from data_cleaning import prepare_data
from feature_engineering import dimensionality_reduction
from feature_engineering import (encode_train_data, encode_test_data, normalize_train_data, normalize_test_data,
                                 get_quality_of_superstructure, get_risk_status_based_on_geo_level, get_geocoded_districts)
from feature_selection import (get_top_k_features_using_rfe_cv, get_top_k_features_using_rfe,
                               get_top_k_features_using_mi)
from utils import load_config, check_file_exists

# Start
print("Starting pipeline ...")

# Track time for execution
start_time = time.time()

# Parse config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to the config file")
args = parser.parse_args()

# Load config file
cfg = load_config(args.config)

# Check for verbosity
verbosity = cfg["modelling"].get("verbosity", None)
verbose = verbosity >= 2

# ToDo: Verify options of config (in the end, when all valid options are known)

# Check if data in config exists and read it
if verbosity >= 1:
    print("Searching for data paths in config ...")
train_values_path = cfg["paths"]["train_values"]
train_labels_path = cfg["paths"]["train_labels"]
test_values_path = cfg["paths"]["test_values"]
result_path = cfg["paths"]["result"]

check_file_exists(train_values_path)
check_file_exists(train_labels_path)
check_file_exists(test_values_path)
    
# Load data
if verbosity >= 1:
    print("Loading Data ...")
train_values = pd.read_csv(train_values_path).set_index("building_id")
train_labels = pd.read_csv(train_labels_path).set_index("building_id")
test_values = pd.read_csv(test_values_path).set_index("building_id")

""" Make Sample Size smaller for experimenting and testing; Keep commented! """
train_values = train_values.iloc[:10000]
train_labels = train_labels.iloc[:10000]


""" ########## Data Cleaning ########## """
# Prepare raw data
binary_encoded_cols = [x for x in train_values.columns if x.startswith("has_")]
columns_to_ignore = cfg.get("data_cleaning", "NO DATA CLEANING DEFINED!").get("columns_to_ignore", [])

if verbosity >= 1:
    print("Cleaning Train Data ...")
train_data_cleaned = prepare_data(df=train_values, config=cfg, ignore_cols=columns_to_ignore+binary_encoded_cols,
                                  outlier_method="replace", verbose=verbose)
if verbosity >= 1:
    print("Cleaning Test Data ...")
test_data_cleaned = prepare_data(df=test_values, config=cfg, ignore_cols=columns_to_ignore+binary_encoded_cols,
                                 outlier_method="replace", verbose=verbose)

# Correlated features
if verbosity >= 1:
    print("Drop correlated features...")
train_data_cleaned = drop_correlated_features(data=train_data_cleaned, config=cfg["data_cleaning"]["correlations"])
test_data_cleaned = drop_correlated_features(data=test_data_cleaned, config=cfg["data_cleaning"]["correlations"])

""" ########## Feature Engineering ########## """
# Group categorical features with rarely occurring realizations
if not cfg["feature_engineering"]["group_categorical"].get("skip", False):
    if verbosity >= 1:
        print("Grouping categorical features ...")
    train_data_cleaned = group_categorical_features(df=train_data_cleaned, default_val="others", verbose=verbose)
    test_data_cleaned = group_categorical_features(df=test_data_cleaned, default_val="others", verbose=verbose)

# Add new features for risk status
if not cfg["feature_engineering"]["risk_status"].get("skip", False):
    if verbosity >= 1:
        print("Add risk status features...")
    test_data_cleaned = get_risk_status_based_on_geo_level(data=train_values, df_to_add_info=test_data_cleaned,
                                                           labels=train_labels,
                                                           geo_level=cfg["feature_engineering"]["risk_status"]["geo_level"])
    train_data_cleaned = get_risk_status_based_on_geo_level(data=train_values, df_to_add_info=train_data_cleaned,
                                                            labels=train_labels,
                                                            geo_level=cfg["feature_engineering"]["risk_status"]["geo_level"])

# Add geocoded districts features
if not cfg["feature_engineering"]["geocode_districts"].get("skip", False):
    if verbosity >= 1:
        print("Add geocoded districts features (lat, long, district name, min distance to epicenter, "
              "max distance to epicenter) ...")
    train_data_cleaned = get_geocoded_districts(df=train_values,
                                                geo_path=cfg["feature_engineering"]["geocode_districts"]["path"],
                                                drop_key=cfg["feature_engineering"]["geocode_districts"]["drop_key"])
    test_data_cleaned = get_geocoded_districts(df=test_values,
                                               geo_path=cfg["feature_engineering"]["geocode_districts"]["path"],
                                               drop_key=cfg["feature_engineering"]["geocode_districts"]["drop_key"])

# Add superstructure quality
if not cfg["feature_engineering"]["superstructure_quality"].get("skip", False):
    if verbosity >= 1:
        print("Add superstructure quality feature...")
    train_data_cleaned = get_quality_of_superstructure(raw_data=train_values, df_to_add_info=train_data_cleaned)
    test_data_cleaned = get_quality_of_superstructure(raw_data=test_values, df_to_add_info=test_data_cleaned)

# Apply One Hot Encoding on categorical features
if not cfg["feature_engineering"]["categorical_encoding"].get("skip", False):
    if cfg["feature_engineering"]["categorical_encoding"]["method"] == "One-Hot":
        if verbosity >= 1:
            print("One Hot Encoding features ...")
        train_data_cleaned, ohe = encode_train_data(x_train=train_data_cleaned)
        test_data_cleaned = encode_test_data(x_test=test_data_cleaned, ohe=ohe)

# Apply StandardScaler (method="standard") or MinMax Scaler (method="minmax") on Features
if not cfg["feature_engineering"]["normalize"].get("skip", False):
    if verbosity >= 1:
        print("Normalizing Data ...")
    train_data_cleaned, scaler = normalize_train_data(x_train=train_data_cleaned,
                                                      method=cfg["feature_engineering"]["normalize"]["method"])
    test_data_cleaned = normalize_test_data(x_test=test_data_cleaned, scaler=scaler)

best_feats = []

# Feature Selection: Get top k features using RFE, RFECV, or use MI
feature_selection_config = cfg["feature_engineering"]["feature_selection"]
method = feature_selection_config.get("method")

if not feature_selection_config.get("skip", False) and method:
    if verbosity >= 1:
        print(f"Selecting best features using {method}...")

    if method == "RFECV":
        best_feats, rfecv = get_top_k_features_using_rfe_cv(x_train=train_data_cleaned,
                                                            y_train=train_labels,
                                                            min_features_to_select=5,
                                                            k_folds=5,
                                                            scoring="matthews_corrcoef",
                                                            step=feature_selection_config.get("step", 1),
                                                            verbose=0)

    elif method == "RFE":
        best_feats, rfe = get_top_k_features_using_rfe(x_train=train_data_cleaned,
                                                       y_train=train_labels,
                                                       k=feature_selection_config.get("k", 10),
                                                       step=feature_selection_config.get("step", 1),
                                                       verbose=0)

    elif method == "MI":
        best_feats, mi_scores = get_top_k_features_using_mi(x_train=train_data_cleaned,
                                                            y_train=train_labels,
                                                            k=feature_selection_config.get("k", 10))

    if verbosity >= 1:
        print(f"\nSelected feature set: {best_feats}\n")

    # Keep best columns
    train_data_cleaned = train_data_cleaned[best_feats]
    test_data_cleaned = test_data_cleaned[best_feats]

# Dimensionality Reduction (only if number of features is larger than threshold)
dimensionality_reduction_config = cfg["feature_engineering"]["dimensionality_reduction"]

if not dimensionality_reduction_config.get("skip", False) and len(best_feats) > dimensionality_reduction_config.get(
        "feature_threshold", 0):
    if verbosity >= 1:
        print("Performing dimensionality reduction...")
    train_data_cleaned, test_data_cleaned = dimensionality_reduction(train_data=train_data_cleaned,
                                                                     test_data=test_data_cleaned,
                                                                     method=dimensionality_reduction_config.get(
                                                                         "method"),
                                                                     n_components=dimensionality_reduction_config.get(
                                                                         "n_components"))

""" ########## Modelling ########## """
# Model training
if verbosity >= 1:
    print("Modelling ...")

# Convert to float64 for computation purposes
train_data_cleaned = train_data_cleaned.astype(np.float64)

# Return fitted model
model, cv_results = modelling.train_model(model="XGBoost",
                                          hyperparameter_grid=cfg["modelling"]["params_xgb"],
                                          train_data=train_data_cleaned,
                                          train_labels=train_labels,
                                          scoring=cfg["modelling"]["scoring"],
                                          verbose=verbose)

# Make prediction
if verbosity >= 1:
    print("Make predictions ...")
modelling.make_prediction(model=model, test_data=test_data_cleaned, result_path=result_path, verbose=verbose)

# Track time for execution and print
run_time = time.time() - start_time
print(f"Finished -- Pipeline took {run_time:.2f} seconds --")
