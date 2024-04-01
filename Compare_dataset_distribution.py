"""
This file compares the distribution of the two Sepsis dataset to ensure that the datasets are similarly distributed.

The following tests are done:
1. Mann-Whitney U test for all columns
2. Chi-square test for categorical columns
3. Mode comparison for all columns

"""


from Preprocess_dataframe import prefix_selection
from scipy.stats import mannwhitneyu
import pandas as pd
from scipy.stats import chi2_contingency

# Load dataframe
data_path1 = 'data/sepsis_cases_1.csv'
data_path2 = 'data/sepsis_cases_2.csv'
dataset1 = pd.read_csv(data_path1, sep=';')
dataset2 = pd.read_csv(data_path2, sep=';')

n_value = 7
filtered_dataset1 = prefix_selection(dataset1, n_value)
filtered_dataset2 = prefix_selection(dataset2, n_value)

cat_cols = ["Activity", 'org:group', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor', 'DiagnosticOther', 'DiagnosticSputum',
            'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
            'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate',
            'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore']

num_cols = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight",
            "timesincelastevent", "timesincecasestart", "event_nr", "open_cases", 'Age']

# ------1. Mann-Whitney U test for all columns------
mw_test_results = {}
for column in num_cols + cat_cols:  # Combine lists for iteration
    if column in filtered_dataset1.columns and column in filtered_dataset2.columns:
        # Comparison for numerical columns
        if column in num_cols:
            data1 = filtered_dataset1[column].dropna()
            data2 = filtered_dataset2[column].dropna()
            if len(data1.unique()) > 1 and len(data2.unique()) > 1:
                stat, p = mannwhitneyu(data1, data2)
                mw_test_results[column] = {'Mann-Whitney Statistic': stat, 'p-value': p}
            else:
                mw_test_results[column] = "Insufficient unique values for meaningful comparison"
        # Comparison for categorical columns
        elif column in cat_cols:
            counts_1 = filtered_dataset1[column].value_counts(normalize=True)
            counts_2 = filtered_dataset2[column].value_counts(normalize=True)
            common_categories = list(set(counts_1.index) & set(counts_2.index))
            if common_categories and not counts_1[common_categories].equals(counts_2[common_categories]):
                stat, p = mannwhitneyu(counts_1[common_categories], counts_2[common_categories])
                mw_test_results[column] = {'Mann-Whitney Statistic': stat, 'p-value': p}
            else:
                mw_test_results[column] = "Cannot perform Mann-Whitney U test due to identical values or empty data"

print("............Mann-Whitney U test results............")
for column, results in mw_test_results.items():
    print(f"Column: {column}, Results: {results}")

# ------2. Chi-square test for categorical columns------
filtered_dataset1['Dataset'] = 'Dataset1'
filtered_dataset2['Dataset'] = 'Dataset2'
combined_dataset = pd.concat([filtered_dataset1, filtered_dataset2])

chi_square_results = {}
for column in cat_cols:
    if column in combined_dataset.columns:
        contingency_table = pd.crosstab(combined_dataset[column], combined_dataset['Dataset'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi_square_results[column] = {'Chi-square Statistic': chi2, 'p-value': p, 'Degrees of Freedom': dof}
    else:
        chi_square_results[column] = "Column not in datasets"

print("............Chi-square test results............")
for column, results in chi_square_results.items():
    print(f"Column: {column}, Results: {results}")

# ------3. Mode comparison for all columns------
mode_comparison_results = {}

for column in num_cols + cat_cols:
    mode_dataset1 = filtered_dataset1[column].mode().iloc[0] if not filtered_dataset1[column].mode().empty else "No mode found"
    mode_dataset2 = filtered_dataset2[column].mode().iloc[0] if not filtered_dataset2[column].mode().empty else "No mode found"

    mode_comparison_results[column] = {'Dataset 1 Mode': mode_dataset1, 'Dataset 2 Mode': mode_dataset2,
                                       'Modes are equal': mode_dataset1 == mode_dataset2}

print("............Mode comparison results............")
for column, results in mode_comparison_results.items():
    print(f"Column: {column}, Results: {results}")