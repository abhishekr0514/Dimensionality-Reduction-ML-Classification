import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector


# --------------------------------------------------
# Correlation Selector
# --------------------------------------------------

class CorrelationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        self.selected_features_ = [
            col for col in X.columns if col not in to_drop
        ]

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        return X[self.selected_features_].values


# --------------------------------------------------
# Base Wrapper Selector
# --------------------------------------------------

class BaseWrapperSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k_features=8, forward=True, floating=False):
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.selector_ = None

    def fit(self, X, y):
        base_model = RandomForestClassifier(n_estimators=100)

        self.selector_ = SequentialFeatureSelector(
            base_model,
            k_features=self.k_features,
            forward=self.forward,
            floating=self.floating,
            scoring="accuracy",
            cv=5,
            n_jobs=-1
        )

        self.selector_.fit(X, y)

        return self

    def transform(self, X):
        return self.selector_.transform(X)


# --------------------------------------------------
# Specific Wrapper Variants
# --------------------------------------------------

class SFSSelector(BaseWrapperSelector):
    def __init__(self, k_features=8):
        super().__init__(k_features=k_features, forward=True, floating=False)


class SBSSelector(BaseWrapperSelector):
    def __init__(self, k_features=8):
        super().__init__(k_features=k_features, forward=False, floating=False)


class SFFSSelector(BaseWrapperSelector):
    def __init__(self, k_features=8):
        super().__init__(k_features=k_features, forward=True, floating=True)


class SFBSSelector(BaseWrapperSelector):
    def __init__(self, k_features=8):
        super().__init__(k_features=k_features, forward=False, floating=True)