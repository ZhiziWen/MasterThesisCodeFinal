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
np.set_printoptions(precision=10)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Preprocessing starts.")

    # Load dataframe
    data_path = 'data/sepsis_cases_1.csv' # sepsis_cases_2.csv pic2012_O_CANCELLED-COMPLETE.csv
    df = pd.read_csv(data_path, sep=';')

    # Prefix selection
    n = 7
    encoded_df = prefix_selection(df, n)

    # Encoding, available options: agg, static
    dataset_name = "Sepsis 1"
    transformed_df = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)

    # add label to each case
    transformed_df = add_label(df, transformed_df)

    # Prepare columns for plotting distribution
    activity_columns = [col for col in transformed_df.columns if "Activity" in col]

    logging.info(f"Dataframe preprocessed. ")

    # resample and train data
    kf = StratifiedKFold(n_splits=2, random_state=0, shuffle=True)
    results = {}
    accuracys ={}
    AUCs = {}
    time_report_all = {}
    X = transformed_df.drop('label', axis=1)
    y = transformed_df['label']

    total_majority = (y == 0).sum()
    total_minority = (y == 1).sum()
    logging.info(f"Total majority: {total_majority}, total minority: {total_minority}")

    weight_range = range(20, 60, 10)
    recall_scores = {weight: {} for weight in weight_range}
    f1_scores = {weight: {} for weight in weight_range}

    for train_index, test_index in kf.split(X, y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        for weight in weight_range:
            resampling_techniques = create_resamplers(weight, (y_train == 0).sum(), (y_train == 1).sum())

            for resampler_name, resampler in resampling_techniques.items():
                logging.info(f"------ Using resampler: {resampler_name} for weight: {weight} ------")

                # Initialize lists to store metrics and reports for each resampling technique
                reports = []
                accuracy = []
                AUC = []
                time_report = []
                resampled_dfs = []
                recall_list = []
                f1_list = []

                # Resample data
                start_time = time.time()

                if resampler is not None:
                    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
                else:
                    X_resampled, y_resampled = X_train, y_train
                print(y_resampled.value_counts())

                # Store resampled DataFrame
                plot_df = pd.concat([X_resampled, y_resampled], axis=1)
                resampled_dfs.append(plot_df)

                logging.info(f"Resampling done with {resampler_name}")

                # Train model
                model = XGBClassifier(random_state=0)
                model.fit(X_resampled, y_resampled)

                end_time = time.time()
                execution_time = end_time - start_time
                time_report.append(execution_time)
                logging.info("Training done")

                # Evaluate model
                y_pred = model.predict(X_test)
                roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                reports.append(classification_report(y_test, y_pred, output_dict=True))
                accuracy.append(accuracy_score(y_test, y_pred))
                AUC.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

                recall = recall_score(y_test, y_pred, pos_label=1)
                if recall == 0.0:
                    recall = 0.0000000001
                recall_list.append(recall)
                f1 = reports[-1]['1']['f1-score']
                f1_list.append(f1)

                # Store metrics and reports for each resampling technique
                print(recall_list)
                recall_scores[weight][resampler_name] = np.mean(recall_list)
                print(recall_scores)
                f1_scores[weight][resampler_name] = np.mean(f1_list)

                plot_distribution(resampled_dfs, activity_columns, resampler_name, dataset_name)
                results[resampler_name], accuracys[resampler_name], AUCs[resampler_name], time_report_all[resampler_name] = reports, accuracy, AUC, time_report

    print(recall_list)
    print(recall_scores)
    create_bump_chart(recall_scores, dataset_name)
    print(f1_scores)
    create_bump_chart(f1_scores, dataset_name)
    create_excel_report(results, accuracys, AUCs, time_report_all, f'report_{dataset_name}')
    create_bar_charts(results, accuracys, AUCs, time_report_all, dataset_name)


