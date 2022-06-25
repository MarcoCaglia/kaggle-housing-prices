"""Test module for feature engineering."""

from typing import Any, Dict, List

import lorem
import numpy as np
import pandas as pd
import pytest
from typeguard import check_type
from kaggle_housing_prices.process.feature_engineering import (
    BaseFeatureEngineer,
    BayesianEncodingFeatureEngineer,
)
from kaggle_housing_prices.process.preprocessing import BasePreprocessor, Preprocessor
from kaggle_housing_prices.process.regression import BaseRegressor, SklearnRegressor
from sklearn.datasets import make_regression
from kaggle_housing_prices.pipeline.price_prediction_model import (
    BasePriceModel,
    PriceModel,
)

PREPROCESSORS_TO_TEST: List[BasePreprocessor] = [Preprocessor]
FEATURE_ENGINEERS_TO_TEST: List[BaseFeatureEngineer] = [BayesianEncodingFeatureEngineer]
REGRESSORS_TO_TEST: List[BaseRegressor] = [SklearnRegressor]
MODELS_TO_TEST: List[BasePriceModel] = [PriceModel]

RANDOM_SEED = 42
RNG = np.random.default_rng(seed=RANDOM_SEED)


class CommonTestFixtures:
    """Define data to test."""

    INT_FEATURE_NAME = "int_feature"
    FLOAT_FEATURE_NAME = "float_feature"
    CATEGORICAL_FEATURE_NAME = "string_feature"
    NUM_RANGE = (0, 1_000)
    CATEGORIES = lorem.data.WORDS[:5]
    NUM_SAMPLES = 1_000
    NUM_ENGINEERED_FEATURES = 10

    @pytest.fixture(scope="class")
    def mock_features(self):
        """Simulate sample houses."""
        mock_data = [
            [
                RNG.choice(self.CATEGORIES),
                np.int16(RNG.uniform(*self.NUM_RANGE)),
                np.float16(RNG.uniform(*self.NUM_RANGE)),
            ]
            for _ in range(self.NUM_SAMPLES)
        ]

        mock_df = pd.DataFrame(
            mock_data,
            columns=[
                self.CATEGORICAL_FEATURE_NAME,
                self.INT_FEATURE_NAME,
                self.FLOAT_FEATURE_NAME,
            ],
        )
        mock_df[self.CATEGORICAL_FEATURE_NAME] = mock_df[
            self.CATEGORICAL_FEATURE_NAME
        ].astype("category")

        return mock_df

    @pytest.fixture(scope="class")
    def mock_prices(self):
        """Simulate observed prices."""
        prices = RNG.normal(loc=10_000, scale=2_500, size=self.NUM_SAMPLES)

        # Censor negative values
        prices[np.where(prices < 0)] = 0

        return prices

    @pytest.fixture(scope="class")
    def mock_features_with_missing_values(self, mock_features):
        """Simulate raw input data with missing values."""
        mock_features_missing = mock_features.applymap(
            lambda value: np.NaN if RNG.uniform() < 0.1 else value
        )

        return mock_features_missing

    @pytest.fixture(scope="class")
    def mock_engineered_data(self, mock_prices):
        """Simulate engineered features."""
        simulated_features, simulated_values = make_regression(
            n_samples=self.NUM_SAMPLES,
            n_features=self.NUM_ENGINEERED_FEATURES,
            n_informative=self.NUM_ENGINEERED_FEATURES,
            bias=np.mean(mock_prices),
        )

        simulated_features_df = pd.DataFrame(simulated_features)

        return simulated_features_df, simulated_values


class TestPreprocessing(CommonTestFixtures):
    """Test basic functionality of preprocessor."""

    @pytest.fixture(scope="class")
    def fit_instance(self, request, mock_features_with_missing_values):
        """Generate a fit instance of preprocessor."""
        instance = request.param().fit(mock_features_with_missing_values)

        return instance

    @pytest.mark.parametrize("fit_instance", PREPROCESSORS_TO_TEST, indirect=True)
    def fit_returns_self_test(self, fit_instance):
        """Assert that fit return value is instance of self."""
        assert isinstance(fit_instance, BasePreprocessor)

    @pytest.mark.parametrize("fit_instance", PREPROCESSORS_TO_TEST, indirect=True)
    def preprocess_returns_df_without_missing_values_test(
        self, fit_instance, mock_features_with_missing_values
    ):
        """Assert, that the result of preprocess is a pandas df, with not remaining missing values."""
        actual = fit_instance.preprocess(mock_features_with_missing_values)

        assert isinstance(actual, pd.DataFrame)
        assert actual.isna().sum().sum() == 0


class TestFeatureEngineering(CommonTestFixtures):
    """Test basic functionality of feature engineering."""

    @pytest.fixture(scope="class")
    def fit_instance(self, request, mock_features, mock_prices):
        """Fit feature engineer instance."""
        instance = request.param().fit(mock_features, mock_prices)

        return instance

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def fit_returns_self_test(self, fit_instance):
        """Assert, that call to fit returns a fit instance of self."""
        assert isinstance(fit_instance, BaseFeatureEngineer)

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def transform_returns_pandas_df_test(self, fit_instance, mock_features):
        """Assert, that transform returns a numpy array."""
        actual = fit_instance.transform(mock_features)

        assert isinstance(actual, pd.DataFrame)
        assert actual.to_numpy().dtype == np.float32

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def get_report_retruns_dict_of_artifacts_test(
        self, fit_instance, mock_engineered_data
    ):
        """Assert, that get_report returns returns the expected type."""
        actual = fit_instance.get_report(
            mock_engineered_data[0], mock_engineered_data[1]
        )

        check_type("report", actual, Dict[str, Any])


class TestRegressor(CommonTestFixtures):
    """Test basic functionality of regressor."""

    @pytest.fixture(scope="class")
    def fit_test_instance(self, request, mock_engineered_data):
        """INstantiate regressor and fit."""
        fit_instance = request.param().fit(
            mock_engineered_data[0], mock_engineered_data[1]
        )

        return fit_instance

    @pytest.mark.parametrize("fit_test_instance", REGRESSORS_TO_TEST, indirect=True)
    def fit_returns_self_test(self, fit_test_instance):
        """Assert, that fit returns a fit instance of self."""
        assert isinstance(fit_test_instance, BaseRegressor)

    @pytest.mark.parametrize("fit_test_instance", REGRESSORS_TO_TEST, indirect=True)
    def predict_with_labels_returns_tuple_of_prediction_and_report_test(
        self, fit_test_instance, mock_engineered_data
    ):
        """Assert, that passing labels to predict will result in a prediction and report."""
        actual, _ = fit_test_instance.predict(
            mock_engineered_data[0], mock_engineered_data[1]
        )

        assert isinstance(actual, np.ndarray)

    @pytest.mark.parametrize("fit_test_instance", REGRESSORS_TO_TEST, indirect=True)
    def predict_without_labels_returns_tuple_of_prediction_and_report_test(
        self, fit_test_instance, mock_engineered_data
    ):
        """Assert, that not passing labels to predict will result in a prediction and report."""
        actual, _ = fit_test_instance.predict(mock_engineered_data[0])

        assert isinstance(actual, np.ndarray)


class TestPricePredictionModel(CommonTestFixtures):
    """Test basic functionality of price prediction models."""

    @pytest.fixture(scope="class")
    def fit_test_instance(
        self, request, mock_features_with_missing_values, mock_prices
    ):
        """Fit model to test to mock_data."""
        instance = request.param(
            Preprocessor(), BayesianEncodingFeatureEngineer(), SklearnRegressor()
        ).fit(mock_features_with_missing_values, mock_prices)

        return instance

    @pytest.mark.parametrize("fit_test_instance", MODELS_TO_TEST, indirect=True)
    def fit_returns_self_test(self, fit_test_instance):
        """Assert, that fit returned a fit method of self."""
        assert isinstance(fit_test_instance, BasePriceModel)

    @pytest.mark.parametrize("fit_test_instance", MODELS_TO_TEST, indirect=True)
    def predict_with_labels_returns_tuple_of_prediction_and_report_test(
        self, fit_test_instance, mock_features_with_missing_values, mock_prices
    ):
        """Assert, that a call to predict returns a tuple of a prediction and a
        prediction report, if labels are passed."""
        actual, report = fit_test_instance.predict(
            mock_features_with_missing_values, mock_prices
        )

        assert isinstance(actual, np.ndarray)
        assert actual.shape == (self.NUM_SAMPLES,)
        assert isinstance(report, dict)

    @pytest.mark.parametrize("fit_test_instance", MODELS_TO_TEST, indirect=True)
    def predict_without_labels_returns_tuple_of_prediction_and_report_test(
        self, fit_test_instance, mock_features_with_missing_values
    ):
        """Assert, that a call to predict returns a tuple of a prediction and a
        prediction report, if no labels are passed."""
        actual, report = fit_test_instance.predict(mock_features_with_missing_values)

        assert isinstance(actual, np.ndarray)
        assert actual.shape == (self.NUM_SAMPLES,)
        assert isinstance(report, dict)
