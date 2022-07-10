"""This module house the generalized price prediction model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from kaggle_housing_prices.process import feature_engineering, preprocessing, regression
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


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
        prediction = self.regressor.predict(X_prime_fe)
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


class PricePredictionPipeline(PriceModel):

    # Define scope of hyperparameters
    SEARCH_SPACE = {
        "scaler": Categorical(
            [StandardScaler(), MinMaxScaler(), RobustScaler(), "passthrough"]
        ),
        "outlier_marker__n_estimators": Integer(50, 500, "uniform"),
        "feature_selector__percentile": Integer(50, 100, "uniform"),
        "regressor__grad_boost__learning_rate": Real(0.01, 1, prior="log-uniform"),
        "regressor__grad_boost__n_estimators": Integer(50, 500, prior="log-uniform"),
        "regressor__grad_boost__subsample": Real(0.1, 1.0, prior="uniform"),
        "regressor__grad_boost__criterion": Categorical(["friedman_mse", "mse"]),
        "regressor__grad_boost__min_samples_split": Integer(
            2, 100, prior="log-uniform"
        ),
        "regressor__grad_boost__min_samples_leaf": Integer(1, 100, prior="log-uniform"),
        "regressor__grad_boost__max_depth": Integer(1, 10, prior="log-uniform"),
        # "regressor__grad_boost__min_impurity_decrease": Real(
        #     0.0, 10.0, prior="log-uniform"
        # ),
        # "regressor__svr__C": Real(1.0, 1e10, prior="uniform"),
        "regressor__ridge__alpha": Real(1e-15, 1.0, "log-uniform"),
    }

    def __init__(
        self,
        preprocessor: preprocessing.BasePreprocessor,
        feature_engineer: feature_engineering.BaseFeatureEngineer,
        regressor: regression.BaseRegressor,
        **kwargs,
    ) -> None:
        """Construct a SKlearn Pipeline from the passed components."""
        pipeline = Pipeline(
            [
                ("preprocessing", preprocessor),
                ("encoder", feature_engineer),
                ("scaler", "passthrough"),
                ("outlier_marker", feature_engineering.OutlierMarker()),
                (
                    "feature_selector",
                    SelectPercentile(mutual_info_regression, percentile=100),
                ),
                ("regressor", regressor),
            ]
        )
        self.bayes_pipe = BayesSearchCV(
            estimator=pipeline,
            search_spaces=self.SEARCH_SPACE,
            refit=True,
            return_train_score=True,
            scoring=self.rmsle_score,
        )

        for attr, value in kwargs.items():
            if attr in vars(self.bayes_pipe):
                setattr(self.bayes_pipe, attr, value)

    def fit(
        self, X: pd.DataFrame, y: npt.NDArray[np.float32], **kwargs
    ) -> BasePriceModel:
        """Fit pipeline with hyperparameter tuning."""
        self.bayes_pipe.fit(X, y)

        return self

    def predict(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> npt.NDArray[np.float32]:
        return self.bayes_pipe.predict(X)

    def get_report(
        self, X: pd.DataFrame, y: Union[npt.NDArray[np.float32], None] = None, **kwargs
    ) -> Dict[str, Any]:
        return cross_validate(
            self.bayes_pipe,
            X,
            y,
            scoring=self.rmsle_score,
            return_train_score=True,
            **kwargs,
        )

    @staticmethod
    def rmsle_score(model, X, y):
        y_hat = model.predict(X)
        y_hat[np.where(y_hat <= 0)] = 0
        return (mean_squared_log_error(y, y_hat) ** 0.5) * -1
