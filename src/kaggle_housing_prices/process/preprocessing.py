"""Preprocessing module for housing prices prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BasePreprocessor(ABC):
    """Template for preprocessing class."""

    def __init__(self, **kwargs) -> None:
        """Initialize preprocessor."""

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
