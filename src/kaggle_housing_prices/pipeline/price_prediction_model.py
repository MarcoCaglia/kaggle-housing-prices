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
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.regressor = regressor
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
        X_prime = self.preprocessor.fit(X).preprocess(X.copy())

        # Fit transform with feature engineer
        X_prime_fe, _ = self.feature_engineer.fit(X_prime, y).transform(
            X_prime.copy(), skip_report=kwargs.get("skip_report", False)
        )

        # Fit the regressor
        _ = self.regressor.fit(X_prime_fe, y)

        return self

    def predict(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        # Fit transform with preprocessor
        X_prime = self.preprocessor.fit(X).preprocess(X.copy())

        # Transform with feature engineer
        X_prime_fe, fe_report = self.feature_engineer.transform(
            X_prime.copy(), skip_report=kwargs.get("skip_report", False)
        )

        # Predict with regressor
        prediction, prediction_report = self.regressor.predict(X_prime_fe, y)

        # Compile report
        report = {**fe_report, **prediction_report}

        return prediction, report
