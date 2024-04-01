"""
This file calculates the mean and standard deviation of the metrics in the metrics report file.
"""

import pandas as pd
from config import timestr, general_output_folder

# Load the dataset
file_path = general_output_folder + 'metrics_report/report_Sepsis 1_20240331-2356.xlsx'
df = pd.read_excel(file_path, sheet_name='Original Data')
dataset_name = "Sepsis 1" # options: Sepsis 1, Sepsis 2, BPIC2012

# Ensure 'Method' retains its original order
df['Method'] = pd.Categorical(df['Method'], categories=pd.unique(df['Method']), ordered=True)

# Exclude 'Trial' and 'Support' from calculations
df_filtered = df.drop(columns=['Trial', 'Support', 'Accuracy'])

# Calculate mean and standard deviation
grouped_mean = df_filtered.groupby(['Method', 'Label']).mean()
grouped_std = df_filtered.groupby(['Method', 'Label']).std()

# Prepare DataFrame for combined mean and STD, next to each other for each metric
columns_ordered = []
for col in grouped_mean.columns:
    columns_ordered.append(f'{col} Mean')
    columns_ordered.append(f'{col} STD')

combined_stats_adjacent = pd.DataFrame(index=grouped_mean.index)
for col in grouped_mean.columns:
    combined_stats_adjacent[f'{col} Mean'] = grouped_mean[col]
    combined_stats_adjacent[f'{col} STD'] = grouped_std[col]

# Reorder the DataFrame according to the original order of appearance of methods
combined_stats_adjacent = combined_stats_adjacent.reindex(columns=columns_ordered)

# Save to a new Excel sheet
output_path_bpic2012_adjusted = general_output_folder + f'metrics_report/combined_stats_{dataset_name}.xlsx'
with pd.ExcelWriter(output_path_bpic2012_adjusted, engine='openpyxl') as writer:
    combined_stats_adjacent.to_excel(writer, sheet_name='Sheet1')
