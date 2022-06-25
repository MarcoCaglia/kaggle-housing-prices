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
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.regressor = regressor
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # Declare empty results
        self.engineered_features: pd.DataFrame = pd.DataFrame
        self.predictions: npt.NDArray[np.float32] = np.array([])

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
    def predict(self, X: pd.DataFrame, **kwargs) -> npt.NDArray[np.float32]:
        """Predict sales price.

        Args:
            X (pd.DataFrame): Raw, unprocessed features

        Returns:
            npt.NDArray[np.float32]: Predicted prices.
        """

    @abstractmethod
    def get_report(
        self, y: Union[npt.NDArray[np.float32], None] = None
    ) -> Dict[str, Any]:
        """Return model report.

        Args:
            y (Union[npt.NDArray[np.float32], None], optional): Observed
                prices. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary of
        """


class PriceModel(BasePriceModel):
    """Simple implementation of price model."""

    def __init__(
        self,
        preprocessor: preprocessing.BasePreprocessor,
        feature_engineer: feature_engineering.BaseFeatureEngineer,
        regressor: regression.BaseRegressor,
        **kwargs,
    ) -> None:
        super().__init__(
            preprocessor=preprocessor,
            feature_engineer=feature_engineer,
            regressor=regressor,
            **kwargs,
        )

    def fit(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32], **kwargs
    ) -> BasePriceModel:
        # Fit transform with preprocessor
        X_prime = self.preprocessor.fit(X).preprocess(X)

        # Fit transform with feature engineer
        X_prime_fe = self.feature_engineer.fit(X_prime, y).transform(X_prime)

        # Fit the regressor
        _ = self.regressor.fit(X_prime_fe, y)

        return self

    def predict(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> npt.NDArray[np.float32]:
        # Fit transform with preprocessor
        X_prime = self.preprocessor.fit(X).preprocess(X)

        # Transform with feature engineer
        X_prime_fe = self.feature_engineer.transform(X_prime)
        self.engineered_features = X_prime_fe

        # Predict with regressor
        prediction = self.regressor.predict(X_prime_fe, y)
        self.predictions = prediction

        return self.predictions

    def get_report(
        self, y: Union[npt.NDArray[np.float32], None] = None
    ) -> Dict[str, Any]:
        # Get FE report
        fe_report = self.feature_engineer.get_report(self.engineered_features, y)

        # Get regression report
        regression_report = self.regressor.get_report(self.predictions, y)

        return {**fe_report, **regression_report}
