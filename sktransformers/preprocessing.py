"""
Preprocessing Transformers
"""
# import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.base import TransformerMixin, BaseEstimator


class Imputer(BaseEstimator, TransformerMixin):

    def __init__(self, missing_values="NaN", strategy="mean"):
        self.missing_values = missing_values
        if strategy not in {'mean', 'median'}:
            raise TypeError("Bad strategy {}".format(strategy))
        self.strategy = strategy
        self.fill_value_ = None

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill_value_ = X.mean()
        elif self.strategy == 'median':
            self.fill_value_ = X.median()
        if isinstance(self.fill_value_, dd.Series):
            # TODO: Remove this block
            # Workaround for https://github.com/dask/dask/issues/1701
            self.fill_value_ = self.fill_value_.compute()

    def transform(self, X, y=None):
        if self.fill_value_ is None:
            raise TypeError("Must fit first")
        X = X.copy() if hasattr(X, 'copy') else X
        return X.fillna(self.fill_value_)


class CategoricalEncoder(TransformerMixin):
    def __init__(self, categories: dict=None, ordered: dict=None):
        self.categories = categories or {}
        self.ordered = ordered or {}
        self.cat_cols_ = None

    def fit(self, X, y=None):
        if not len(self.categories):
            categories = X.select_dtypes(include=[object]).columns
        else:
            categories = self.categories
        self.cat_cols_ = categories
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        categories = self.cat_cols_
        for k in categories:
            cat = (categories.get(k, None)
                   if hasattr(categories, 'get')
                   else None)
            ordered = self.ordered.get(k, False)
            X[k] = pd.Categorical(X[k],
                                  categories=cat,
                                  ordered=ordered)
        return X


class DummyEncoder(TransformerMixin):

    def __init__(self, columns: list=None, drop_first=False):
        self.columns = columns
        self.drop_first = drop_first

        self.columns_ = None
        self.cat_columns_ = None  # type: pd.Index
        self.non_cat_columns_ = None  # type: pd.Index
        self.categories_map_ = None
        self.ordered_map_ = None
        self.cat_blocks_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        if self.columns is None:
            self.cat_columns_ = X.select_dtypes(include=['category']).columns
        else:
            self.cat_columns_ = self.columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.categories_map_ = {col: X[col].cat.categories
                                for col in self.cat_columns_}
        self.ordered_map_ = {col: X[col].cat.ordered
                             for col in self.cat_columns_}

        left = len(self.non_cat_columns_)
        self.cat_blocks_ = {}
        for col in self.cat_columns_:
            right = left + len(X[col].cat.categories)
            self.cat_blocks_[col], left = slice(left, right), right
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return pd.get_dummies(X, drop_first=self.drop_first)
        elif isinstance(X, dd.DataFrame):
            return X.map_partitions(pd.get_dummies, drop_first=self.drop_first)
        else:
            raise TypeError

    def inverse_transform(self, X):
        non_cat = pd.DataFrame(X[:, :len(self.non_cat_columns_)],
                               columns=self.non_cat_columns_)
        cats = []
        for col in self.cat_columns_:
            slice_ = self.cat_blocks_[col]
            categories = self.categories_map_[col]
            ordered = self.ordered_map_[col]

            codes = X[:, slice_].argmax(1)
            series = pd.Series(pd.Categorical.from_codes(
                codes, categories, ordered=ordered
            ), name=col)
            cats.append(series)
        df = pd.concat([non_cat] + cats, axis=1)[self.columns_]
        return df
