import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np
import scipy.stats as stats
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def perform_friedman_test(df, metrics=['Precision', 'Recall', 'F1-Score', 'AUC']):
    """
    Performs the Friedman test for specified metrics on a DataFrame filtered by Label=1.

    Parameters:
    - df: DataFrame containing the data.
    - metrics: List of metrics to perform the Friedman test on.

    Returns:
    - A dictionary containing Friedman test statistics and p-values for each metric.
    """
    results = {}

    for metric in metrics:
        # Creating a list for each method's metric values
        metric_values = [df[df['Method'] == method][metric].tolist() for method in
                         df['Method'].unique()]

        # Ensure that all methods have the same number of trials
        min_length = min(len(v) for v in metric_values)
        metric_values = [v[:min_length] for v in metric_values if len(v) >= min_length]

        # Check if there are at least two methods with values for comparison
        if len(metric_values) > 1:
            # Perform Friedman test
            statistic, p_value = friedmanchisquare(*metric_values)
            results[metric] = {'Friedman Test Statistic': statistic, 'p-value': p_value}
        else:
            results[metric] = {'Friedman Test Statistic': None, 'p-value': None,
                               'error': 'Not enough methods with data for comparison'}

    return results


def perform_posthoc_test(df, metrics):
    """
    Performs a posthoc test using the Conover method on specified metrics.

    Parameters:
    - df: DataFrame containing the data.
    - metrics: List of metrics to perform the posthoc test on.

    Returns:
    - A dictionary of DataFrames containing the p-values of the pairwise comparisons for each metric.
    """
    results = {}
    for metric in metrics:
        data_for_posthoc = pd.DataFrame({
            'Values': df[metric],
            'Groups': df['Method']
        })
        posthoc_results = sp.posthoc_conover(data_for_posthoc, val_col='Values', group_col='Groups')
        results[metric] = posthoc_results

    for key, value in results.items():
        print(key, ":", value)
    return results


def evaluate_methods(posthoc_results, df, alpha=0.05):
    """
    Evaluates methods that are statistically better than the "Original" method.
    "Recall" must be significantly better, while "Accuracy" and "AUC" should not be significantly worse.

    Parameters:
    - posthoc_results: The results from the perform_posthoc_test function.
    - df: Original DataFrame used for posthoc tests.
    - alpha: Significance level for determining statistical difference.

    Returns:
    - A list of methods that meet the criteria.
    """
    better_methods = []
    for method in df['Method'].unique():
        if method == 'Base':
            continue

        # Criteria for Precision to be significantly better
        precision_better = posthoc_results['Precision'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'Precision'].mean() > df[df['Method'] == 'Base']['Precision'].mean()
        # Criteria for Recall to be significantly better
        recall_better = posthoc_results['Recall'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'Recall'].mean() > df[df['Method'] == 'Base']['Recall'].mean()
        # Criteria for F1 to be significantly better
        f1_better = posthoc_results['F1-Score'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'F1-Score'].mean() > df[df['Method'] == 'Base']['F1-Score'].mean()
        # Criteria for AUC to be significantly better
        AUC_better = posthoc_results['AUC'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'AUC'].mean() > df[df['Method'] == 'Base']['AUC'].mean()
        TrainingTime_better = posthoc_results['Training Time'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'Training Time'].mean() < df[df['Method'] == 'Base']['Training Time'].mean()

        # # Criteria for Accuracy and AUC to not be significantly worse
        precision_not_worse = posthoc_results['Precision'].loc[method, 'Base'] >= alpha or df[df['Method'] == method][
            'Precision'].mean() >= df[df['Method'] == 'Base']['Precision'].mean()
        recall_not_worse = posthoc_results['Recall'].loc[method, 'Base'] >= alpha or df[df['Method'] == method][
            'Recall'].mean() >= df[df['Method'] == 'Base']['Recall'].mean()
        f1_not_worse = posthoc_results['F1-Score'].loc[method, 'Base'] >= alpha or df[df['Method'] == method][
            'F1-Score'].mean() >= df[df['Method'] == 'Base']['F1-Score'].mean()
        auc_not_worse = posthoc_results['AUC'].loc[method, 'Base'] >= alpha or df[df['Method'] == method][
            'AUC'].mean() >= df[df['Method'] == 'Base']['AUC'].mean()
        TrainingTime_not_worse = posthoc_results['Training Time'].loc[method, 'Base'] >= alpha or df[df['Method'] == method][
            'Training Time'].mean() >= df[df['Method'] == 'Base']['Training Time'].mean()

        # Criteria for Precision to be significantly worse
        precision_worse = posthoc_results['Precision'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'Precision'].mean() < df[df['Method'] == 'Base']['Precision'].mean()
        # Criteria for Recall to be significantly worse
        recall_worse = posthoc_results['Recall'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'Recall'].mean() < df[df['Method'] == 'Base']['Recall'].mean()
        # Criteria for F1 to be significantly worse
        f1_worse = posthoc_results['F1-Score'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'F1-Score'].mean() < df[df['Method'] == 'Base']['F1-Score'].mean()
        # Criteria for AUC to be significantly worse
        AUC_worse = posthoc_results['AUC'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'AUC'].mean() < df[df['Method'] == 'Base']['AUC'].mean()
        TrainingTime_worse = posthoc_results['Training Time'].loc[method, 'Base'] < alpha and df[df['Method'] == method][
            'Training Time'].mean() > df[df['Method'] == 'Base']['Training Time'].mean()

        print("_" * 50)
        print(method)
        print(precision_better, recall_better, f1_better, AUC_better, TrainingTime_better)
        # print(precision_not_worse, recall_not_worse, f1_not_worse, auc_not_worse)
        print(precision_worse, recall_worse, f1_worse, AUC_worse, TrainingTime_worse)

        if precision_better and recall_better and f1_better and auc_not_worse:
            better_methods.append(method)
    return better_methods


# report_Sepsis 1_20240316-1252
# report_Sepsis 2_20240316-1257
# report_BPIC2012_20240316-1300


file_path = 'D:/SS2023/MasterThesis/code/metrics_report/report_BPIC2012_20240316-1300' \
            '.xlsx'
sheet_name = 'Original Data'
df = pd.read_excel(file_path, sheet_name=sheet_name)
df_label_1 = df[df['Label'] == 1]
df_label_0 = df[df['Label'] == 0]
print(df_label_1)

results = perform_friedman_test(df_label_1)
for metric, result in results.items():
    if 'error' in result:
        print(f"Metric: {metric}, Error: {result['error']}")
    else:
        print(f"Metric: {metric}, Friedman Test Statistic: {result['Friedman Test Statistic']}, p-value: {result['p-value']}")

# Assuming the perform_posthoc_test function is already defined as given previously
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC', 'Training Time']
posthoc_results = perform_posthoc_test(df_label_1, metrics)

# Assuming the evaluate_methods function is already defined as given previously
better_methods = evaluate_methods(posthoc_results, df_label_1)
print("Methods statistically better than 'Original':")
print(better_methods)
print("BPIC")
