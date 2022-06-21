"""This module house the generalized price prediction model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from kaggle_housing_prices.process import feature_engineering, preprocessing, regression


class BasePriceModel(ABC):
    """Template class for price prediction model."""

    @abstractmethod
    def __init__(
        self,
        preprocessor: preprocessing.BasePreprocessor,
        feature_engineer: feature_engineering.BaseFeatureEngineer,
        regressor: regression.BaseRegressor,
        **kwargs,
    ) -> None:
        """Initialize Price Model.

        Args:
            preprocessor (preprocessing.BasePreprocessor): Initialized
                preprocessing class.
            feature_engineer (feature_engineering.BaseFeatureEngineer):
                Initialized Feature engineer.
            regressor (regression.BaseRegressor): Regressor instance.
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32], **kwargs
    ) -> BasePriceModel:
        """Fit model to observed data.

        Args:
            X (pd.DataFrame): Raw, unprocessed features.
            y (npt.NDArray[np.float32]): Observed prices.

        Returns:
            BasePriceModel: Fit PriceModel.
        """

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """Predict sales price.

        Args:
            X (pd.DataFrame): Raw, unprocessed features
            y (Union[npt.NDArray[np.float32], None], optional): Observed sales
                prices. If specified, the returned report will contain label
                dependent variables. Defaults to None.

        Returns:
            Tuple[npt.NDArray[np.float32], Dict[str, Any]]: Tuple of prediction
                and prediction report.
        """
