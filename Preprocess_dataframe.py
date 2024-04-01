"""
This file contains the data preprocessing functions used in the experiments.
"""

from AggregateTransformer import AggregateTransformer


def prefix_selection(df, n):
    """
    Selects the first n events for each case in the dataframe.
    :param df: The dataframe to be truncated.
    :param n: The number of events to keep for each case.
    :return: The truncated dataframe.
    """
    filtered_df = df.groupby("Case ID").filter(lambda x: len(x) >= n)
    return filtered_df.groupby("Case ID").apply(lambda x: x.head(n)).reset_index(drop=True)


def encoding(df, dataset="sepsis"):
    """
    Encodes the dataframe using the AggregateTransformer.
    :param df: The dataframe to be encoded.
    :param dataset: The dataset name.
    :return: The encoded dataframe.
    """

    # Aggregation encoding
    if "Sepsis" in dataset:
        dynamic_cat_cols = ["Activity", 'org:group']  # i.e. event attributes
        static_cat_cols = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                                    'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                                    'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                                    'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                                    'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                                    'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                                    'SIRSCritTemperature',
                                    'SIRSCriteria2OrMore']  # i.e. case attributes that are known from the start
        dynamic_num_cols = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight",
                                     "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
        static_num_cols = ['Age']


    else: # bpic2012
        dynamic_cat_cols = ["Activity", "Resource"]
        static_cat_cols = []
        dynamic_num_cols = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent",
                                     "timesincecasestart", "event_nr", "open_cases"]
        static_num_cols = ['AMOUNT_REQ']

    cat_cols = dynamic_cat_cols + static_cat_cols
    num_cols = dynamic_num_cols + static_num_cols

    transformer = AggregateTransformer(case_id_col='Case ID', cat_cols=cat_cols, num_cols=num_cols, boolean=True,
                                       fillna=True)

    transformer.fit(df)
    transformed_df = transformer.transform(df)

    return transformed_df


def add_label(original_df, transformed_df):
    """
    Adds the label to the transformed dataframe.
    :param original_df: The original dataframe.
    :param transformed_df: The transformed dataframe.
    :return: The transformed dataframe with the label.
    """

    unique_case_ids = transformed_df.index.unique()
    case_id_to_label = original_df.drop_duplicates(subset='Case ID').set_index('Case ID')['label']
    labels_for_trunc_df = unique_case_ids.map(case_id_to_label)
    transformed_df['label'] = labels_for_trunc_df
    transformed_df['label'] = transformed_df['label'].map({'regular': 0, 'deviant': 1})
    return transformed_df