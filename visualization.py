import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import numpy as np
import os
from datetime import datetime
import time
from config import timestr
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# df = pd.read_csv('hospital_2_reshape.csv')
#
# def extract_means_by_label(dataframe, label_column='label'):
#     act_columns = [col for col in dataframe.columns if re.match(r'ACT_\d+_', col)]
#
#     means_label_0 = {}
#     means_label_1 = {}
#
#     for col in act_columns:
#         # Calculate the mean for label 0
#         mean_0 = dataframe[dataframe[label_column] == 0][col].mean()
#         means_label_0[col] = mean_0
#
#         # Calculate the mean for label 1
#         mean_1 = dataframe[dataframe[label_column] == 1][col].mean()
#         means_label_1[col] = mean_1
#
#     return means_label_0, means_label_1
#
# def visualize_means(dataframe, means_label_0, means_label_1, label_column='label'):
#     # Identifying unique n values
#     n_values = set(int(re.search(r'ACT_(\d+)_', col).group(1)) for col in dataframe.columns if re.match(r'ACT_\d+_', col))
#
#     for n in sorted(n_values):
#         # Filtering columns for ACT_{n}_{name}
#         act_n_columns = [col for col in dataframe.columns if re.match(fr'ACT_{n}_', col)]
#
#         # Extracting means for label 0 and 1 for each column
#         means_0 = [means_label_0[col] for col in act_n_columns]
#         means_1 = [means_label_1[col] for col in act_n_columns]
#
#         # Plotting
#         plt.figure(figsize=(15, 6))
#         x = range(len(act_n_columns))
#         plt.bar([i - 0.2 for i in x], means_0, width=0.4, label='Label 0', align='center')
#         plt.bar([i + 0.2 for i in x], means_1, width=0.4, label='Label 1', align='center')
#         plt.xlabel(f'Event_No.{n}')
#         plt.ylabel('Mean Values')
#         plt.title(f'Share of activity for event_No.{n} by Label')
#         plt.xticks(x, act_n_columns, rotation=90)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#
# # Using the function with the dataset
# means_0, means_1 = extract_means_by_label(df)
#
# # Displaying the results
# means_0, means_1
#
# visualize_means(df, means_0, means_1)

def create_bar_charts(results, accuracys, AUCs, time_report_all, dataset_name):
    """
    Create separate bar charts for each metric: Precision, Recall, F1-Score, Accuracy, AUC
    for both labels '0' (regular) and '1' (deviant), and a separate chart for mean training time.
    Different background colors are used for original, oversampling, and undersampling methods.
    Bars are ordered as per the original order in resampling_techniques.

    :param results: Dictionary containing detailed metrics for each method and trial.
    :param accuracys: Dictionary containing accuracy values for each method.
    :param AUCs: Dictionary containing AUC values for each method.
    :param time_report_all: Dictionary containing training time for each method.
    :param dataset_name: Name of the dataset used for analysis.
    """

    output_folder = f"D:/SS2023/MasterThesis/code/visulization_plot/{timestr}-{dataset_name}"
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
            plt.subplots_adjust(right=0.8)  # Adjust subplot to fit legend
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
                     xytext=(0, 5),  # 5 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12, color='black')

    plt.savefig(os.path.join(output_folder, "Mean_Training_Time.png"))
    plt.close()




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

    output_folder = f"D:/SS2023/MasterThesis/code/visulization_plot/{timestr}-{dataset_name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    combined_df = pd.concat(dfs)

    # Ensure all columns_to_plot exist in combined_df, add them with 0s if they don't
    for col in columns_to_plot:
        if col not in combined_df.columns:
            combined_df[col] = 0

    # Calculate the proportion of 1s (or 0s for missing columns) in activity columns for each label
    proportions = {col: combined_df.groupby('label')[col].mean() for col in columns_to_plot}

    # Preparing data for plotting
    plot_data = pd.DataFrame(proportions)

    # Set up the figure before plotting
    plt.figure(figsize=(13, 6))

    # Plotting
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
    # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    # output_folder = f"./visualization_plots/{timestr}-{dataset_name}"
    # os.makedirs(output_folder, exist_ok=True)
    # plt.savefig(os.path.join(output_folder, f"{dataset_name}_F1_Scores_Weight_Influence.png"))
    plt.show()
    plt.close()





# def create_bump_chart(recall_scores, dataset_name):
#     df = pd.DataFrame(recall_scores).T  # Transpose: weights become rows, resamplers become columns
#
#     # Rank resamplers, assuming higher recall is better.
#     ranks = df.rank(axis=1, method='min', ascending=False, na_option='bottom')
#
#     # Fill NaN values (which could be resulting from zeros) with a rank worse than the worst rank
#     ranks.fillna(len(df.columns) + 1, inplace=True)
#
#     # Sort resamplers based on their final performance at the last weight
#     final_performance = ranks.iloc[-1].sort_values()
#
#     # Generate a sufficient number of distinct colors
#     colors = plt.cm.get_cmap('nipy_spectral', len(df.columns))  # Using a colormap with enough entries
#
#     plt.figure(figsize=(9, 6))
#     # Iterate over resamplers according to their final performance ranking
#     for idx, (resampler, _) in enumerate(final_performance.iteritems()):
#         plt.plot(ranks.index, ranks[resampler], marker='o', label=resampler, color=colors(idx / len(df.columns)),
#                  alpha=0.4, linewidth=2.5)
#
#     plt.title(f'Bump Chart of Recall Scores for Label 1 Across Weights - {dataset_name}')
#     plt.xlabel('Weight')
#     plt.ylabel('Rank')
#     plt.xticks(list(recall_scores.keys()))
#
#     # Set y-ticks to show every rank, including the artificial rank for zero scores
#     plt.yticks(np.arange(1, len(df.columns) + 2, 1))  # Ensure all ranks are shown, including the "zero score" rank
#
#     # Ensure the best rank appears at the top of the y-axis
#     plt.gca().invert_yaxis()
#
#     # Place legend outside
#     plt.legend(title="Resampler", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#
#     # Ensure output folder exists
#     timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
#     output_folder = f"D:/SS2023/MasterThesis/code/visulization_plot/{timestr}-{dataset_name}"
#     os.makedirs(output_folder, exist_ok=True)  # Create the directory if it does not exist
#
#     plt.savefig(os.path.join(output_folder, f"{dataset_name}_Influence of weights.png"))
#     plt.show()
#     plt.close()

# def create_bump_chart(recall_scores, dataset_name):
#     df = pd.DataFrame(recall_scores).T  # Transpose: weights become rows, resamplers become columns
#     # Rank resamplers based on recall scores, assuming higher scores are better
#     ranks = df.rank(axis=1, method='min', ascending=False)
#
#     # Determine the y-position for each resampler's name based on their final rank
#     final_ranks = ranks.iloc[-1].sort_values(ascending=True)
#
#     colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns)))
#
#     plt.figure(figsize=(12, 8))
#     # Plot each resampler according to rank over weights
#     for idx, (resampler, rank) in enumerate(final_ranks.items()):
#         y_pos = ranks[resampler].iloc[-1]  # Final rank position
#         plt.plot(ranks.index, ranks[resampler], marker='o', label=None, color=colors[idx])
#         # Annotate the line with the resampler's name at the last weight point
#         plt.text(ranks.index[-1] + 0.5, y_pos, resampler, verticalalignment='center', color=colors[idx])
#
#     plt.title(f'Bump Chart of Recall Scores for Label 1 Across Weights - {dataset_name}')
#     plt.xlabel('Weight')
#     # Adjust y-axis label and ticks to reflect ranks
#     plt.ylabel('Rank')
#     plt.xticks(list(recall_scores.keys()))
#     plt.yticks(range(1, len(df.columns) + 1))
#     plt.gca().invert_yaxis()  # Invert y-axis to have best rank at top
#
#     plt.tight_layout()
#     plt.show()

#
# def create_bump_chart(recall_scores, dataset_name):
#     df = pd.DataFrame(recall_scores).T  # Transpose: weights become rows, resamplers become columns
#     # Assuming higher recall is better, rank with ascending=False. Adjust if necessary.
#     ranks = df.rank(axis=1, method='min', ascending=False).astype(int)  # Convert ranks to integers
#
#     # Sort resamplers based on their final performance at the last weight
#     final_performance = ranks.iloc[-1].sort_values()
#
#     colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns)))
#
#     fig, ax = plt.subplots(figsize=(12, 8))
#     # Iterate over resamplers according to their final performance ranking
#     for idx, (resampler, _) in enumerate(final_performance.iteritems()):
#         ax.plot(ranks.index, ranks[resampler], marker='o', color=colors[idx])
#
#     ax.set_title(f'Bump Chart of Recall Scores for Label 1 Across Weights - {dataset_name}')
#     ax.set_xlabel('Weight')
#     ax.set_ylabel('Rank')
#     ax.set_xticks(list(recall_scores.keys()))
#     # Ensure the best rank appears at the top of the y-axis
#     ax.invert_yaxis()
#     # Adjust the y-axis ticks to display only integer values
#     ax.set_yticks(range(1, len(df.columns) + 1))
#
#     # Adjust text annotations to be directly next to the rank text
#     for idx, (resampler, _) in enumerate(final_performance.iteritems()):
#         # Use ax.transData for y to keep alignment with data, but shift x outside the frame
#         trans = ax.get_yaxis_transform()  # This gets a transform that scales x like the figure, y like the data
#         ax.text(-0.03, ranks[resampler].iloc[0], resampler, transform=trans,
#                 va='center', ha='right', color=colors[idx], fontsize=9)
#
#     plt.show()