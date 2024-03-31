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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

data_path = 'data/sepsis_cases_1.csv' # options: sepsis_cases_1.csv, sepsis_cases_2.csv, bpic2012_O_DECLINED-COMPLETE.csv
df = pd.read_csv(data_path, sep=';')

filtered_df = df.groupby('Case ID').filter(lambda x: len(x) == 8)

# Prefix selection
n = 7
encoded_df = prefix_selection(df, n)

encoded_df = encoded_df # [["Activity", "timesincelastevent", "Case ID"]]

dataset_name = "Sepsis 1" # options: Sepsis 1, Sepsis 2, BPIC2012
transformed_df = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)

# [["Activity", "timesincelastevent", "Case ID"]]
transformed_df = add_label(df, transformed_df)

top_5_label_1 = transformed_df[transformed_df['label'] == 1].head(6)
top_10_label_0 = transformed_df[transformed_df['label'] == 0].head(12)
selected_rows = pd.concat([top_5_label_1, top_10_label_0])

print(selected_rows)
print("1")

X_train = selected_rows.drop('label', axis=1)
y_train = selected_rows['label']

output_path = 'D:/SS2023/MasterThesis/code/example_generate.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for resampler_name, resampler in resampling_techniques.items():
        sheet_name = resampler_name
        if resampler is not None:
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        print("-----------------", resampler_name, "-----------------")
        print(X_resampled)
        print(y_resampled)
        resample_concate = pd.concat([X_resampled, y_resampled], axis=1)
        resample_concate.to_excel(writer, sheet_name=sheet_name, index=False)