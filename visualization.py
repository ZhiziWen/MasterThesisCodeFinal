"""
This file contains the visualization functions used in the experiments.
It contains the following functions:
1. create_bar_charts: Create separate bar charts for each metric: Precision, Recall, F1-Score, Accuracy, AUC
    for both labels '0' (regular) and '1' (deviant), and a separate chart for mean training time.
    Different background colors are used for baseline, oversampling, undersampling and hybrid sampling methods.
2. plot_distribution: plot distribution of a column in a DataFrame.
3. create_bump_chart: Create a bump chart to visualize the rank of each resampling techniques.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from config import timestr, project_folder
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Create bar charts for each metric and each label
def create_bar_charts(results, accuracys, AUCs, time_report_all, dataset_name):
    """
    Create separate bar charts for each metric: Precision, Recall, F1-Score, Accuracy, AUC
    for both labels '0' (regular) and '1' (deviant), and a separate chart for mean training time.
    Different background colors are used for baseline, oversampling, undersampling and hybrid sampling methods.

    :param results: Dictionary containing detailed metrics for each method and trial.
    :param accuracys: Dictionary containing accuracy values for each method.
    :param AUCs: Dictionary containing AUC values for each method.
    :param time_report_all: Dictionary containing training time for each method.
    :param dataset_name: Name of the dataset used for analysis.
    """

    output_folder = project_folder + f"visualization_plot/{timestr}-{dataset_name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define background colors for each category
    bg_colors = {
        'original': "#C50F3C",
        'oversampling': "#004A9F",
        'undersampling': "#971B2F",
        'hybrid sampling': "#04316A"
    }

    # Define legend patches
    legend_patches = [
        mpatches.Patch(color=bg_colors['original'], label='Baseline'),
        mpatches.Patch(color=bg_colors['oversampling'], label='Oversampling'),
        mpatches.Patch(color=bg_colors['undersampling'], label='Undersampling'),
        mpatches.Patch(color=bg_colors['hybrid sampling'], label='Hybrid sampling'),
    ]

    # Preparing data for aggregation
    aggregated_data = []
    for i, method in enumerate(results.keys()):
        category_index = 0 if i == 0 else 1 if i <= 4 else 2 if i<= 14 else 3
        for label in ['0', '1']:
            avg_precision = sum(trial[label]['precision'] for trial in results[method]) / len(results[method])
            avg_recall = sum(trial[label]['recall'] for trial in results[method]) / len(results[method])
            avg_f1_score = sum(trial[label]['f1-score'] for trial in results[method]) / len(results[method])
            avg_accuracy = sum(accuracys[method]) / len(accuracys[method])
            avg_auc = sum(AUCs[method]) / len(AUCs[method])
            aggregated_data.append(
                [method, label, avg_precision, avg_recall, avg_f1_score, avg_accuracy, avg_auc, category_index])

    # Creating DataFrame
    df = pd.DataFrame(aggregated_data, columns=['Method', 'Label', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC',
                                                'Category Index'])
    df['Method'] = pd.Categorical(df['Method'], categories=results.keys(), ordered=True)

    # Separate charts for each metric
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC']

    for i, metric in enumerate(metrics):

        # Plotting charts for label '0' and '1'
        for label_df, label in [(df[df['Label'] == '0'], '0'), (df[df['Label'] == '1'], '1')]:
            label_text = "regular" if label == '0' else "deviant"
            plt.figure(figsize=(12, 7))
            ax = label_df.groupby('Method').mean()[metric].plot(kind='bar')

            # Update the title based on whether the metric is AUC
            if metric == 'AUC':
                plt.title(f"Mean ROC-AUC by Resampling Techniques - {dataset_name} Dataset", fontsize=12)
            else:
                plt.title(f'Mean {metric} for Label "{label_text}" by Resampling Techniques - {dataset_name} Dataset',
                          fontsize=12)

            plt.ylabel("Mean " + metric, fontsize=12)
            plt.xlabel("Resampling Techniques", fontsize=12)
            plt.xticks(rotation=45)

            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                            textcoords='offset points')

            # Setting background color based on category index
            for method in label_df['Method'].unique():
                category_index = label_df[label_df['Method'] == method]['Category Index'].iloc[0]
                ax.get_children()[list(results.keys()).index(method)].set_facecolor(
                    bg_colors[list(bg_colors.keys())[category_index]])

            # Adding custom legends outside the plot
            plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=12)
            plt.subplots_adjust(right=0.8)
            plt.savefig(os.path.join(output_folder, f"{metric}_Label_{label}_{dataset_name}.png"))
            plt.close()

    # Mean training time chart with color set by category_index, reusing the same legend
    method_category_indices = {method: (0 if i == 0 else 1 if i <= 4 else 2 if i<= 14 else 3) for i, method in enumerate(results.keys())}
    mean_training_time = {method: sum(times) / len(times) for method, times in time_report_all.items()}

    plt.figure(figsize=(12, 7))
    for method, time in mean_training_time.items():
        category_index = method_category_indices[method]
        bar_color = bg_colors[list(bg_colors.keys())[category_index]]
        plt.bar(method, time, color=bar_color)

    plt.title(f'Mean Resampling and Training Time - {dataset_name} Dataset', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.subplots_adjust(right=0.8)

    for i, (method, time) in enumerate(mean_training_time.items()):
        plt.annotate(f"{time:.2f}",
                     xy=(i, time),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12, color='black')

    plt.savefig(os.path.join(output_folder, "Mean_Training_Time.png"))
    plt.close()


# Function to plot distribution of a column in a DataFrame, currently used for plotting the distribution of Activity
def plot_distribution(dfs, columns_to_plot, resampler_name, dataset_name):
    # If dfs is a single DataFrame, wrap it in a list
    if not isinstance(dfs, list):
        dfs = [dfs]

    # Argument validation
    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise ValueError("dfs must be a list of pandas DataFrame objects or a single DataFrame.")
    if not isinstance(columns_to_plot, list) or not all(isinstance(col, str) for col in columns_to_plot):
        raise ValueError("columns_to_plot must be a list of strings.")
    if not isinstance(resampler_name, str) or not isinstance(dataset_name, str):
        raise ValueError("resampler_name and dataset_name must be strings.")

    output_folder = project_folder + f"visualization_plot/{timestr}-{dataset_name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    combined_df = pd.concat(dfs)

    # Ensure all columns_to_plot exist in combined_df, add them with 0s if they don't
    for col in columns_to_plot:
        if col not in combined_df.columns:
            combined_df[col] = 0

    # Calculate the proportion of 1s in activity columns for each label
    proportions = {col: combined_df.groupby('label')[col].mean() for col in columns_to_plot}

    # Preparing data for plotting
    plot_data = pd.DataFrame(proportions)

    # Plotting
    plt.figure(figsize=(13, 6))
    n_cols = len(columns_to_plot)
    index = np.arange(n_cols)
    bar_width = 0.35
    colors = ['#004A9F', '#C50F3C']

    # Remove "Activity_" prefix for x-axis labels
    x_axis_labels = [col.replace("Activity_", "") for col in columns_to_plot]

    for i, label in enumerate(plot_data.index):
        label_name = 'Regular' if label == 0 else 'Deviant'
        bars = plt.bar(index + i * bar_width, plot_data.loc[label], bar_width, label=label_name, color=colors[i])

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom', ha='center', color='black', fontsize = 10)


    plt.xlabel('Activity Names', fontsize = 14)
    plt.ylabel('Share of activities happening', fontsize = 14)
    plt.title(f'Share of activities by label - {dataset_name} - {resampler_name}',  fontsize = 14)
    plt.xticks(index + bar_width / 2, x_axis_labels, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    plt.tight_layout()

    # Save the plot and then close
    plt.savefig(os.path.join(output_folder, f"{resampler_name}_activity_distribution.png"))
    plt.close()


# Function to create a bump chart of the metrics scores
def create_bump_chart(metrics_name, metrics_scores, dataset_name):
    df = pd.DataFrame(metrics_scores).T  # Transpose: weights become rows, resamplers become columns
    ranks = df.rank(axis=1, method='min', ascending=False, na_option='bottom')
    ranks.fillna(len(df.columns) + 1, inplace=True)
    final_performance = ranks.iloc[-1].sort_values()
    colors = plt.cm.get_cmap('nipy_spectral', len(df.columns))
    plt.figure(figsize=(9, 6))
    for idx, (resampler, _) in enumerate(final_performance.iteritems()):
        plt.plot(ranks.index, ranks[resampler], marker='o', label=resampler, color=colors(idx / len(df.columns)),
                 alpha=0.4, linewidth=2.5)
    plt.title(f'Bump Chart of {metrics_name} Across Weights - {dataset_name}')
    plt.xlabel('Weight')
    plt.ylabel('Rank')
    plt.xticks(list(metrics_scores.keys()))
    plt.yticks(np.arange(1, len(df.columns) + 2, 1))
    plt.gca().invert_yaxis()
    plt.legend(title="Resampler", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_folder = project_folder + f"visualization_plot/{timestr}-{dataset_name}-bump_chart"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{dataset_name}_F1_Scores_Weight_Influence.png"))
    plt.show()
    plt.close()
