"""Preprocessing module for housing prices prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pandas as pd


class BasePreprocessor(ABC):
    """Template for preprocessing class."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize preprocessor."""
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs) -> BasePreprocessor:
        """Fit preprocessor to input data.

        Args:
            X (pd.DataFrame): Raw dataset.

        Returns:
            BasePreprocessor: Fit instance of self.
        """

    @abstractmethod
    def preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute preprocessing.

        Args:
            X (pd.DataFrame): Raw dataset.

        Returns:
            pd.DataFrame: Preprocessed data set.
        """


class Preprocessor(BasePreprocessor):
    """First iteration of preprocessing."""

    ID_COLUMN = "Id"

    def __init__(self, threshold_for_categorical: int = 30, **kwargs) -> None:
        """Initialize preprocessor.

        Args:
            threshold_for_categorical (int, optional): Number of maximum unique
                values above which an object variable is no longer considered
                categorical. Defaults to 30.
        """
        self.threshold_for_categorical = threshold_for_categorical
        self.cat_columns: List = []
        self.imputation_values: Dict[str, Union[str, float, int]] = {}
        super().__init__(**kwargs)

    def fit(self, X: pd.DataFrame, **kwargs) -> BasePreprocessor:  # noqa
        # Find categorical columns
        self.cat_columns = [
            col
            for col in X.columns
            if self._check_if_categorical(X[col])
            if col != self.ID_COLUMN
        ]

        # Find imputation values per column
        self.imputation_values = {
            col: X[col].mean()
            if col not in self.cat_columns
            else X[col].value_counts().index[0]
            for col in X.columns
            if col != self.ID_COLUMN
        }

        return self

    def _check_if_categorical(self, column: pd.Series) -> bool:
        # Check if a column is categorical
        return (column.dtype == "object") and (
            column.nunique() <= self.threshold_for_categorical
        )

    def preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # noqa
        # Set ID Column as table index
        X = X.set_index(self.ID_COLUMN)

        # Impute missing values
        X = X.fillna(self.imputation_values)

        # Set columns to categorical
        X.loc[:, self.cat_columns] = X.loc[:, self.cat_columns].astype("category")

        return X
