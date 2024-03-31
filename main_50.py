import numpy as np
import pandas as pd
import time
import logging

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report,  roc_auc_score, accuracy_score, recall_score, confusion_matrix

from resampling_and_classification import resampling_techniques, create_resamplers
from Preprocess_dataframe import preprocess_data, reshape_case, prefix_selection, encoding, add_label
from evaluation_metrics import calculate_averaged_results, write_data_to_excel, create_excel_report
from visualization import create_bar_charts, plot_distribution, create_bump_chart



if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Preprocessing starts.")

    # Load dataframe
    data_path = 'data/sepsis_cases_2.csv' # options: sepsis_cases_1.csv, sepsis_cases_2.csv, bpic2012_O_DECLINED-COMPLETE.csv
    df = pd.read_csv(data_path, sep=';')

    # Prefix selection
    n = 7
    encoded_df = prefix_selection(df, n)

    # Encoding, available options: agg, static
    dataset_name = "Sepsis 2" # options: Sepsis 1, Sepsis 2, BPIC2012
    transformed_df = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)

    # add label to each case
    transformed_df = add_label(df, transformed_df)

    # Prepare columns for plotting distribution
    activity_columns = ['Activity_Admission IC', 'Activity_Admission NC', 'Activity_CRP', 'Activity_ER Registration', 'Activity_ER Sepsis Triage',
                        'Activity_ER Triage', 'Activity_IV Antibiotics', 'Activity_IV Liquid', 'Activity_LacticAcid', 'Activity_Leucocytes',
                        'Activity_Release A', 'Activity_Return ER'] # 'Activity_Release B', 'Activity_Release C', 'Activity_Release D', 'Activity_other'

    plot_distribution(transformed_df, activity_columns, resampler_name="Prefix 7", dataset_name=dataset_name)

    logging.info(f"Dataframe preprocessed. ")

    # resample and train data
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    results = {}
    accuracys ={}
    AUCs = {}
    time_report_all = {}
    X = transformed_df.drop('label', axis=1)
    y = transformed_df['label']

    total_majority = (y == 0).sum()
    total_minority = (y == 1).sum()
    logging.info(f"Total majority: {total_majority}, total minority: {total_minority}")

    for resampler_name, resampler in resampling_techniques.items():
        logging.info(f"------ Using resampler: {resampler_name} ------")
        reports = []
        accuracy = []
        AUC = []
        time_report = []
        resampled_dfs = []
        recall_list = []

        for train_index, test_index in kf.split(X,y):

            # resample data
            start_time = time.time()

            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            if resampler is not None:
                X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            else:
                X_resampled, y_resampled = X_train, y_train
            # print(y_resampled.value_counts())

            # Store resampled DataFrame
            plot_df = pd.concat([X_resampled, y_resampled], axis=1)
            resampled_dfs.append(plot_df)

            logging.info(f"Resampling done with {resampler_name}")

            # train model
            model = XGBClassifier(random_state = 0)
            model.fit(X_resampled, y_resampled)

            end_time = time.time()
            execution_time = end_time - start_time
            time_report.append(execution_time)
            logging.info("Training done")

            # evaluate model
            y_pred = model.predict(X_test)
            print(confusion_matrix(y_test, y_pred, labels=[0,1]))
            roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            reports.append(classification_report(y_test, y_pred, output_dict=True))
            accuracy.append(accuracy_score(y_test, y_pred))
            AUC.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

        plot_distribution(resampled_dfs, activity_columns, resampler_name, dataset_name)
        results[resampler_name], accuracys[resampler_name], AUCs[resampler_name], time_report_all[resampler_name] = reports, accuracy, AUC, time_report

    create_excel_report(results, accuracys, AUCs, time_report_all, f'report_{dataset_name}')
    create_bar_charts(results, accuracys, AUCs, time_report_all, dataset_name)




