
import numpy as np
import pandas as pd
import time
import logging

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report,  roc_auc_score, accuracy_score, recall_score

from resampling_and_classification import resampling_techniques, create_resamplers
from Preprocess_dataframe import preprocess_data, reshape_case, prefix_selection, encoding, add_label
from evaluation_metrics import calculate_averaged_results, write_data_to_excel, create_excel_report
from visualization import create_bar_charts, plot_distribution, create_bump_chart


data_path = 'data/6sample.csv' # possible options: sepsis_cases_2.csv, bpic2012_O_CANCELLED-COMPLETE.csv
df = pd.read_csv(data_path, sep=';')

# Encoding, available options: agg, static
dataset_name = "sepsis"  # options: sepsis, bpic
transformed_df = encoding(df, encoding_method="agg", dataset=dataset_name)

# add label to each case
transformed_df = add_label(df, transformed_df)

# Prepare columns for plotting distribution
activity_columns = [col for col in transformed_df.columns if "Activity" in col]

logging.info(f"Dataframe preprocessed. ")

# resample and train data
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
results = {}
accuracys = {}
AUCs = {}
time_report_all = {}
X = transformed_df.drop('label', axis=1)
y = transformed_df['label']

print(X,y)

total_majority = (y == 0).sum()
total_minority = (y == 1).sum()
logging.info(f"Total majority: {total_majority}, total minority: {total_minority}")

print(resampling_techniques)

for resampler_name, resampler in resampling_techniques.items():
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    print(X_resampled, y_resampled )