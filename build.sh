#!/bin/bash

# Checkout the branch that you want to build
git checkout main

# Install the required packages and dependencies
pip install -r requirements.txt

# Run the script with Python 3.10
python3 src/main.py --config "configs/config.yml"
