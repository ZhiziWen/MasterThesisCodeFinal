"""
This file contains the AggregateTransformer class which is used to aggregate the data based on the case id column.
The code used in this file is modified from the code in the following link:
https://github.com/irhete/predictive-monitoring-benchmark
"""


from time import time
import pandas as pd
from sklearn.base import TransformerMixin


class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.boolean = boolean
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            if len(self.num_cols) > 0:
                # Use a dictionary comprehension to apply multiple aggregation functions to each numeric column
                agg_functions = {col: ['mean', 'max', 'min', 'std', 'sum'] for col in self.num_cols}
                dt_numeric = X.groupby(self.case_id_col).agg(agg_functions)
                # Flatten the MultiIndex in columns
                dt_numeric.columns = ['{}_{}'.format(col[0], col[1]) for col in dt_numeric.columns.values]

        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        
        self.transform_time = time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns