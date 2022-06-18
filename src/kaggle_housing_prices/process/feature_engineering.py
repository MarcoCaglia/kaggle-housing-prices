"""Feature Engineering module for Housing Prices prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy.typing as npt
import pandas as pd
import numpy as np


class BaseFeatureEngineer(ABC):
    """Template Feature Engineering class."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize FeatureEngineer."""

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
    def transform(self, X: pd.DataFrame, **kwargs) -> npt.NDArray[np.float32]:
        """Apply fit transformation to input data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            npt.NDArray[np.float32]: Engineered features.
        """

    @abstractmethod
    def get_scores(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32], **kwargs
    ) -> Dict[str, Any]:
        """Return label-dependent metrics from OOS observation.

        Args:
            X (pd.DataFrame): Input features.
            y (npt.NDArray[np.float32]): Observed housing prices.

        Returns:
            Dict[str, Any]: Dictionary of metric name and metric value(s).
        """

    @abstractmethod
    def get_report(self) -> Dict[str, Any]:
        """Get training report from last fit.

        Returns:
            Dict[str, Any]: Dictionary of metric name and metric value(s).
        """
