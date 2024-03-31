import pandas as pd

# Load the dataset
file_path_bpic2012 = 'D:/SS2023/MasterThesis/code/metrics_report/report_Sepsis 2_20240316-1257.xlsx'
df_bpic2012 = pd.read_excel(file_path_bpic2012, sheet_name='Original Data')

# Ensure 'Method' retains its original order by converting it to a categorical type
# with categories set in the order they appear
df_bpic2012['Method'] = pd.Categorical(df_bpic2012['Method'], categories=pd.unique(df_bpic2012['Method']), ordered=True)

# Exclude 'Trial' and 'Support' from calculations
df_filtered = df_bpic2012.drop(columns=['Trial', 'Support'])

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

# Save to a new Excel sheet, maintaining the order of 'Method'
output_path_bpic2012_adjusted = 'D:/SS2023/MasterThesis/code/metrics_report/combined_stats_Sepsis2_adjusted.xlsx'
with pd.ExcelWriter(output_path_bpic2012_adjusted, engine='openpyxl') as writer:
    combined_stats_adjacent.to_excel(writer, sheet_name='Sheet1')

output_path_bpic2012_adjusted
