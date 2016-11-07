import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaNVarianceThreshold(TransformerMixin, BaseEstimator):

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.variances_ = np.nanvar(X, 0)
        return self

    def transform(self, X, y=None):
        keep = self.variances_ > self.threshold

        if isinstance(X, pd.DataFrame):
            return X.loc[:, keep]
        else:
            return X[:, keep]
