import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
from scipy.stats import chi2_contingency
from Preprocess_dataframe import preprocess_data, reshape_case, prefix_selection, encoding, add_label
from scipy.stats import f_oneway

data_path = 'data/sepsis_cases_1.csv'
df = pd.read_csv(data_path, sep=';')
n = 7
encoded_df = prefix_selection(df, n)
dataset_name = "sepsis 4"  # options: sepsis, bpic
df1 = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)

data_path = 'data/sepsis_cases_2.csv'
df = pd.read_csv(data_path, sep=';')
n = 7
encoded_df = prefix_selection(df, n)
dataset_name = "sepsis 4"  # options: sepsis, bpic
df2 = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)

data_path = 'data/sepsis_cases_4.csv'
df = pd.read_csv(data_path, sep=';')
n = 7
encoded_df = prefix_selection(df, n)
dataset_name = "sepsis 4"  # options: sepsis, bpic
df3 = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)


df1['Dataset'] = 'df1'
df2['Dataset'] = 'df2'
df3['Dataset'] = 'df3'
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Filter out non-Activity columns if necessary
activity_columns = [col for col in combined_df.columns if col.startswith('Activity')]
# Ensure to include the Dataset column for identification
activity_columns.append('Dataset')

# Step 2: Reshape to long format
melted_df = combined_df.melt(id_vars=['Dataset'], value_vars=activity_columns, var_name='Activity', value_name='Value')

# Optional Step: Clean up the Activity column to remove the "Activity_" prefix
melted_df['Activity'] = melted_df['Activity'].str.replace('Activity_', '')

# Group by Activity and Dataset, then describe each group
summary_stats = melted_df.groupby(['Activity', 'Dataset'])['Value'].describe()
print(summary_stats)

import seaborn as sns
import matplotlib.pyplot as plt

# Box plot for comparing distributions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Activity', y='Value', hue='Dataset', data=melted_df)
plt.title('Activity Distribution Across Datasets')
plt.xticks(rotation=45)  # Rotate activity names for better readability
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x='Activity', y='Value', hue='Dataset', data=melted_df, split=True)
plt.title('Activity Distribution Across Datasets')
plt.xticks(rotation=45)
plt.show()



# Assuming 'Activity1', 'Activity2', 'Activity3' are your activities of interest
# And you have two datasets 'df1' and 'df2'
activity_groups = melted_df[melted_df['Dataset'].isin(['df1', 'df2'])].groupby(['Activity', 'Dataset'])

anova_results = f_oneway(activity_groups.get_group(('Activity_Admission IC', 'df1'))['Value'],
                         activity_groups.get_group(('Activity_Admission IC', 'df2'))['Value'],
                         activity_groups.get_group(('Activity_Admission NC', 'df1'))['Value'],
                         activity_groups.get_group(('Activity_Admission NC', 'df2'))['Value'],
                         # Add more groups as needed
                        )
print(f"ANOVA F-statistic: {anova_results.statistic}, P-value: {anova_results.pvalue}")
