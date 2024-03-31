'''
This file generated examples of resampling-techniques-generated data
It uses an example dataset with 5 cases of label 1 and 10 cases of label 0.
The output is an excel file with the resampled data for each resampling technique.
Note: the generated data is not the original data, but prefix selected and aggregated data which is used in training.
'''

import pandas as pd
from Preprocess_dataframe import prefix_selection, encoding, add_label
from resampling_and_classification import resampling_techniques
from config import general_output_folder
import os


data_path = 'data/sepsis_cases_1.csv' # options: sepsis_cases_1.csv, sepsis_cases_2.csv, bpic2012_O_DECLINED-COMPLETE.csv
df = pd.read_csv(data_path, sep=';')
filtered_df = df.groupby('Case ID').filter(lambda x: len(x) == 8)

# Prefix selection
n = 7
encoded_df = prefix_selection(df, n)
encoded_df = encoded_df # it can changed to encoded_df[["Activity", "timesincelastevent", "Case ID"]] for easier understanding

# Encoding
dataset_name = "Sepsis 1" # options: Sepsis 1, Sepsis 2, BPIC2012
transformed_df = encoding(encoded_df, dataset=dataset_name)
transformed_df = add_label(df, transformed_df)

# Select 5 cases of label 1 and 10 cases of label 0
top_5_label_1 = transformed_df[transformed_df['label'] == 1].head(6)
top_10_label_0 = transformed_df[transformed_df['label'] == 0].head(12)
selected_rows = pd.concat([top_5_label_1, top_10_label_0])

X_train = selected_rows.drop('label', axis=1)
y_train = selected_rows['label']

# Output the example resampled data
output_path = general_output_folder + 'example_generated_data/example_generated.xlsx'
if not os.path.exists(general_output_folder + 'example_generated_data'):
    os.makedirs(general_output_folder + 'example_generated_data')

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