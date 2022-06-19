"""Feature Engineering module for Housing Prices prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ppscore


class BaseFeatureEngineer(ABC):
    """Template Feature Engineering class."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize FeatureEngineer."""
        # The report will be populated as part of the fitting process
        self.report: Dict[str, Any] = {}
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32], **kwargs
    ) -> BaseFeatureEngineer:
        """Fit feature engineer to data and observed values.

        Args:
            X (pd.DataFrame): Raw input features.
            y (npt.NDArray[np.float32]): Observed prices.

        Returns:
            BaseFeatureEngineer: Fit instance of self.
        """

    @abstractmethod
    def transform(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply fit transformation to input data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            npt.NDArray[np.float32]: Engineered features.
        """


class BayesianEncodingFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, **kwargs) -> None:
        self.bayesian_encoding_dict: Dict[str, Dict[str, float]] = {}
        super().__init__(**kwargs)

    def fit(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32], **kwargs
    ) -> BaseFeatureEngineer:
        # Find mean per label for all categorical features
        self._find_target_mean_per_label(X, y)

        return self

    def _find_target_mean_per_label(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32]
    ) -> None:
        # Find mean of target value per label for each categorical column
        categorical_columns = [col for col in X.columns if X[col].dtype == "category"]
        X_categorical = X.loc[:, categorical_columns]
        X_categorical["y"] = y

        self.bayesian_encoding_dict = {
            col: X_categorical.groupby(col).y.mean().to_dict()
            for col in X_categorical.columns
            if col != "y"
        }

    def transform(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Apply bayesian encoding
        X_prime = X.copy()

        for col, mapping in self.bayesian_encoding_dict.items():
            X_prime[col] = X_prime[col].map(mapping)

        # Get report
        report = self._get_report(X_prime, y)

        # Type cast features
        X_prime = X_prime.astype(np.float32)

        return X_prime, report

    def _get_report(
        self, X_prime: pd.DataFrame, y: Union[npt.NDArray[np.float32], None]
    ) -> Dict[str, Any]:
        # Get engineering report, with more metrics, if y is specified
        independent_report = self._get_independent_report(X_prime)

        if y is not None:
            dependent_report = self._get_dependent_report(X_prime, y)

        else:
            dependent_report = {}

        return {**independent_report, **dependent_report}

    def _get_independent_report(self, X: pd.DataFrame) -> Dict[str, Any]:
        # Get metrics that do not depend on the target variable
        report = {}

        # Multicorrelation
        report["multicorrelation"] = X.corr()

        # Variance Inflation Factor
        report["vif"] = {
            name: variance_inflation_factor(X.to_numpy(), index)
            for index, name in enumerate(X.columns)
        }

        return report

    def _get_dependent_report(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32]
    ) -> Dict[str, Any]:
        # Get metrics that do depend on the target variable
        report: Dict[str, Any] = {}

        # Correlation with target
        report["correlation"] = {
            col: np.corrcoef(X[col].to_numpy(), y)[0, 1] for col in X.columns
        }

        # Predictive power w.r.t. target
        df = X.copy()
        df["y"] = y
        report["predictive_power"] = [ppscore.score(df, col, "y") for col in X.columns]

        return report
