"""
UFC Fight Outcome Prediction - Workflow
--------------------------------------
This script runs all steps: data exploration, preprocessing, model training, and analysis.
Each step is clearly documented and explained, so the workflow can be followed like a report.
"""

import os
from src.data_exploration import explore_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_and_analyze

# Step 1: Data Exploration & Cleaning
# -----------------------------------
# We start by loading the raw UFC fight data and performing an initial exploration.
# This includes checking for missing values, understanding the structure, and cleaning the data.
raw_data = os.path.join("data", "UFC Fight Statistics (July 2016 - Nov 2024).csv")
cleaned_data = os.path.join("data", "fight_totals_cleaned.csv")
explore_data(raw_data, cleaned_data)

# Step 2: Data Preprocessing
# --------------------------
# The cleaned data is further preprocessed: numeric features are scaled, categorical variables are encoded,
# and additional features such as weight class and gender are extracted.
preprocessed_data = os.path.join("data", "fight_totals_preprocessed.csv")
preprocess_data(cleaned_data, preprocessed_data)

# Step 3: Model Training & Analysis
# ---------------------------------
# For each weight class and gender, a separate Random Forest model is trained to predict the fight winner.
# After training, the most important features for each model are analyzed and visualized.
train_and_analyze(preprocessed_data)

# Note: The predict_fight function is not included in this script.
# It should be run separately after training the models.