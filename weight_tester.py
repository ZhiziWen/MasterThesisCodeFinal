import numpy as np
import pandas as pd
import time
import logging

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report,  roc_auc_score, accuracy_score, recall_score, f1_score, precision_score

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
    data_path = 'data/bpic2012_O_DECLINED-COMPLETE.csv' # sepsis_cases_2.csv bpic2012_O_DECLINED-COMPLETE.csv
    df = pd.read_csv(data_path, sep=';')

    # Prefix selection
    n = 7
    encoded_df = prefix_selection(df, n)

    # Encoding, available options: agg, static
    dataset_name = "BPIC2012"
    transformed_df = encoding(encoded_df, encoding_method="agg", dataset=dataset_name)

    # add label to each case
    transformed_df = add_label(df, transformed_df)

    # Prepare columns for plotting distribution
    activity_columns = [col for col in transformed_df.columns if "Activity" in col]

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

    weight_range = range(30, 60, 10)

    f1_scores = {weight: {} for weight in weight_range}
    precision_scores = {weight: {} for weight in weight_range}
    recall_scores = {weight: {} for weight in weight_range}
    AUC_scores = {weight: {} for weight in weight_range}

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

                f1 = f1_score(y_test, y_pred, average='binary')
                f1_scores[weight].setdefault(resampler_name, []).append(f1)

                precision = precision_score(y_test, y_pred, average='binary')
                precision_scores[weight].setdefault(resampler_name, []).append(precision)

                recall = recall_score(y_test, y_pred, average='binary')
                recall_scores[weight].setdefault(resampler_name, []).append(recall)

                AUC = roc_auc_score(y_test, y_pred)
                AUC_scores[weight].setdefault(resampler_name, []).append(AUC)

    for weight in f1_scores:
        for resampler_name in f1_scores[weight]:
            f1_scores[weight][resampler_name] = np.mean(f1_scores[weight][resampler_name])
            precision_scores[weight][resampler_name] = np.mean(precision_scores[weight][resampler_name])
            recall_scores[weight][resampler_name] = np.mean(recall_scores[weight][resampler_name])
            AUC_scores[weight][resampler_name] = np.mean(AUC_scores[weight][resampler_name])

    create_bump_chart("F1 Scores", f1_scores, dataset_name)
    create_bump_chart("Precision Scores", precision_scores, dataset_name)
    create_bump_chart("Recall Scores", recall_scores, dataset_name)
    create_bump_chart("ROC-AUC Scores", AUC_scores, dataset_name)

