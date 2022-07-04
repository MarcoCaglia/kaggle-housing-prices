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

    def __init__(
        self,
        threshold_for_categorical: int = 30,
        threshold_for_boolean: float = 0.5,
        **kwargs,
    ) -> None:
        """Initialize preprocessor.

        Args:
            threshold_for_categorical (int, optional): Number of maximum unique
                values above which an object variable is no longer considered
                categorical. Defaults to 30.
            threshold_for_boolean (int, optional): Sets the share of values in
                a column above which the column will be converted to a boolean
                feature, where the boolean indicates whether the value was
                missing or not.
        """
        self.threshold_for_categorical = threshold_for_categorical
        self.cat_columns: List[str] = []
        self.imputation_values: Dict[str, Union[str, float, int]] = {}
        self.threshold_for_boolean = threshold_for_boolean
        self.boolean_columns: List[str] = []
        super().__init__(**kwargs)

    def fit(self, X: pd.DataFrame, **kwargs) -> BasePreprocessor:  # noqa
        # Find columns which are to be converted to boolean
        self.boolean_columns = [
            col
            for col in X.columns
            if self._check_boolean(X[col])
            if col != self.ID_COLUMN
        ]

        # Find categorical columns
        self.cat_columns = [
            col
            for col in X.columns
            if self._check_if_categorical(X[col])
            if (col != self.ID_COLUMN) and (col not in self.boolean_columns)
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

    def _check_boolean(self, column: pd.Series) -> bool:
        # Check if the number of missing values in the passed column
        # reaches the threshold
        return column.isna().mean() >= self.threshold_for_boolean

    def _check_if_categorical(self, column: pd.Series) -> bool:
        # Check if a column is categorical
        return (column.dtype == "object") and (
            column.nunique() <= self.threshold_for_categorical
        )

    def preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # noqa
        # Set ID Column as table index
        X = X.set_index(self.ID_COLUMN)

        # Convert boolean columns to boolean
        for col in self.boolean_columns:
            X[col] = X[col].isna()

        # Impute missing values
        X = X.fillna(self.imputation_values)

        # Set columns to categorical
        X.loc[:, self.cat_columns] = X.loc[:, self.cat_columns].astype("category")

        return X
