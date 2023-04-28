import argparse, sys
import os
import yaml
import pandas as pd

from data_cleaning import drop_correlated_features
import modelling


# Parse config
print("Parse config")
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
print("Read and load data")
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
train_values = pd.read_csv(train_values_path)
train_labels = pd.read_csv(train_labels_path)
test_values = pd.read_csv(test_values_path)
train_values.set_index("building_id")
test_values.set_index("building_id")


# Data cleaning
print("Data cleaning")
#train_data_cleaned = drop_correlated_features(data=train_values, config=cfg["data_cleaning"]["correlations"])
#test_data_cleaned = drop_correlated_features(data=test_values, config=cfg["data_cleaning"]["correlations"])
train_data_cleaned = train_values
test_data_cleaned = test_values

# Feature engineering: TBD


# Model training: TBD
print("Modelling")
model = modelling.hyperparameter_optimization(model="Dummy")
model.fit(train_data_cleaned, train_labels)

# Make prediction: TBD
print("Make predictions")
modelling.make_prediction(model=model, test_data=test_data_cleaned, result_path=result_path)
