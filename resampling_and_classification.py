"""
This file contains the resampling techniques used in the experiments.
resampling_techniques is a dictionary that contains the resampling techniques used with default weight (50:50 after resampling).
create_resamplers is a function that adjusts resampling techniques to achieve a specified percentage of the minority class after resampling
"""

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, AllKNN, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, TomekLinks, InstanceHardnessThreshold, NearMiss, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NeighbourhoodCleaningRule

resampling_techniques = {
        "Baseline": None,
        "RO": RandomOverSampler(random_state=0),
        "SM": SMOTE(random_state=0),
        "BS": BorderlineSMOTE(random_state=0),
        "AD": ADASYN(random_state=0),
        "RU": RandomUnderSampler(random_state=0),
        "NM": NearMiss(version=1),
        "CN": CondensedNearestNeighbour(random_state=0),
        "AL": AllKNN(),
        "EN": EditedNearestNeighbours(),
        "RE": RepeatedEditedNearestNeighbours(),
        "TM": TomekLinks(),
        "IH": InstanceHardnessThreshold(random_state=0),
        "OS": OneSidedSelection(random_state=0),
        "NC": NeighbourhoodCleaningRule(),
        "SE": SMOTEENN(random_state=0),
        "ST": SMOTETomek(random_state=0),
    }

def create_resamplers(weight, total_majority, total_minority):
    """
    Adjusts resampling techniques to achieve a specified percentage of the minority class after resampling,
    ensuring that the calculated sample targets are integers.

    :param weight: Target percentage of the minority class after resampling.
    :param total_majority: Total number of samples in the majority class before resampling.
    :param total_minority: Total number of samples in the minority class before resampling.
    """
    weight = weight / 100
    assert 0 < weight < 1, "Weight must be between 0 and 100 exclusive."

    target_minority = max(int(total_majority * weight / (1 - weight)), total_minority)
    target_majority_undersampling = min(int(total_minority / weight * (1 - weight)), total_majority)

    oversampling_strategy = {1: target_minority}
    undersampling_strategy = {0: target_majority_undersampling}

    if weight == 0.5:
        oversampling_strategy = 'auto'
        undersampling_strategy = 'auto'

    resampling_techniques = {
        "Baseline": None,
        "RO": RandomOverSampler(sampling_strategy=oversampling_strategy,random_state= 0),
        "SM": SMOTE(sampling_strategy=oversampling_strategy, random_state=0),
        "BS": BorderlineSMOTE(sampling_strategy=oversampling_strategy, random_state=0),
        "AD": ADASYN(sampling_strategy=oversampling_strategy, random_state=0),
        "RU": RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=0),
        "NM": NearMiss(version=1, sampling_strategy=undersampling_strategy),
        "CN": CondensedNearestNeighbour(random_state=0), # No sampling_strategy for cleaning methods
        "AL": AllKNN(), # No sampling_strategy for cleaning methods
        "EN": EditedNearestNeighbours(), # No sampling_strategy for cleaning methods
        "RE": RepeatedEditedNearestNeighbours(), # No sampling_strategy for cleaning methods
        "TM": TomekLinks(), # No sampling_strategy for cleaning methods
        "IH": InstanceHardnessThreshold(sampling_strategy=undersampling_strategy, random_state=0),
        "OS": OneSidedSelection(random_state=0), # No sampling_strategy for cleaning methods
        "NC": NeighbourhoodCleaningRule(), # No sampling_strategy for cleaning methods
        "SE": SMOTEENN(sampling_strategy=oversampling_strategy, random_state=0),
        "ST": SMOTETomek(sampling_strategy=oversampling_strategy, random_state=0),
    }

    return resampling_techniques