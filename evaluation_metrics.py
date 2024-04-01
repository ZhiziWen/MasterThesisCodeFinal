"""
This file calculates the evaluation metrics used in the experiments and save it in an excel file.
"""

import pandas as pd
from config import timestr, general_output_folder

def create_excel_report(results, accuracys, AUCs, time_report_all, filename='output_metrics.xlsx'):
    """
    Create an Excel report with two sheets based on the provided data.

    :param results: Dictionary containing detailed metrics for each method and trial.
    :param accuracys: Dictionary containing accuracy values for each method.
    :param AUCs: Dictionary containing AUC values for each method.
    :param time_report_all: Dictionary containing training time for each method.
    :param filename: Name of the Excel file to be created.
    """

    # Preparing data for Sheet 1
    data_sheet1 = []
    for method, trials in results.items():
        for idx, trial in enumerate(trials):
            for label, metrics in trial.items():
                if label in ['0', '1']:
                    accuracy = accuracys[method][idx]
                    auc = AUCs[method][idx]
                    time_report = time_report_all[method][idx]
                    data_sheet1.append([method, idx + 1, label] + list(metrics.values()) + [accuracy, auc, time_report])

    # Creating DataFrame for Sheet 1
    df_sheet1 = pd.DataFrame(data_sheet1, columns=['Method', 'Trial', 'Label', 'Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy', 'AUC', 'Training Time'])

    # Preparing data for Sheet 2
    data_sheet2 = []
    for method in results.keys():
        for label in ['0', '1']:
            avg_precision = sum(trial[label]['precision'] for trial in results[method]) / len(results[method])
            avg_recall = sum(trial[label]['recall'] for trial in results[method]) / len(results[method])
            avg_f1_score = sum(trial[label]['f1-score'] for trial in results[method]) / len(results[method])
            avg_support = sum(trial[label]['support'] for trial in results[method]) / len(results[method])
            avg_accuracy = sum(accuracys[method]) / len(accuracys[method])
            avg_auc = sum(AUCs[method]) / len(AUCs[method])
            avg_time = sum(time_report_all[method]) / len(time_report_all[method])

            data_sheet2.append([method, label, avg_precision, avg_recall, avg_f1_score, avg_support, avg_accuracy, avg_auc, avg_time])

    # Creating DataFrame for Sheet 2
    df_sheet2 = pd.DataFrame(data_sheet2, columns=['Method', 'Label', 'Precision', 'Recall', 'F1-Score', 'Support', 'Average Accuracy', 'Average AUC', 'Average Training Time'])

    # Writing to Excel
    out_path = general_output_folder + f'metrics_report/{filename}_{timestr}.xlsx'

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_sheet1.to_excel(writer, sheet_name='Original Data', index=False)
        df_sheet2.to_excel(writer, sheet_name='Aggregated Metrics', index=False)

    print(f"Excel file '{out_path}' has been created.")


