"""Test module for feature engineering."""

from typing import List
import numpy as np
import pandas as pd
import pytest
import lorem

from kaggle_housing_prices.process.feature_engineering import BaseFeatureEngineer
from kaggle_housing_prices.process.preprocessing import BasePreprocessor

PREPROCESSORS_TO_TEST: List[BasePreprocessor] = []
FEATURE_ENGINEERS_TO_TEST: List[BaseFeatureEngineer] = []

RANDOM_SEED = 42
RNG = np.random.default_rng(seed=RANDOM_SEED)


class CommonTestFixtures:
    """Define data to test."""

    INT_FEATURE_NAME = "int_feature"
    FLOAT_FEATURE_NAME = "float_feature"
    STRING_FEATURE_NAME = "string_feature"
    NUM_RANGE = (0, 1_000)
    STRINGS = lorem.data.WORDS
    NUM_SAMPLES = 1_000

    @pytest.fixture(scope="class")
    def mock_features(self):
        """Simulate sample houses."""
        mock_data = [
            [
                RNG.choice(self.STRINGS),
                np.int16(RNG.uniform(*self.NUM_RANGE)),
                np.float16(RNG.uniform(*self.NUM_RANGE)),
            ]
            for _ in range(self.NUM_SAMPLES)
        ]

        mock_df = pd.DataFrame(
            mock_data,
            columns=[
                self.STRING_FEATURE_NAME,
                self.INT_FEATURE_NAME,
                self.FLOAT_FEATURE_NAME,
            ],
        )

        return mock_df

    @pytest.fixture(scope="class")
    def mock_prices(self):
        """Simulate observed prices."""
        prices = RNG.normal(loc=10_000, scale=2_500, size=self.NUM_SAMPLES)

        return prices

    @pytest.fixture(scope="class")
    def mock_features_with_missing_values(self, mock_input):
        """Simulate raw input data with missing values."""
        mock_features_missing = mock_input.applymap(
            lambda value: np.NaN if RNG.unfiorm() < 0.1 else value
        )

        return mock_features_missing


class TestPreprocessing(CommonTestFixtures):
    """Test basic functionality of preprocessor."""

    @pytest.fixture(scope="class")
    def fit_test_instance(self, request, mock_features_with_missing_labels):
        """Generate a fit instance of preprocessor."""
        instance = request.param().fit(mock_features_with_missing_labels)

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
    def fit_instance(self, request, mock_features, mock_labels):
        """Fit feature engineer instance."""
        instance = request.param().fit(mock_features, mock_labels)

        return instance

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def fit_returns_self_test(self, fit_instance):
        """Assert, that call to fit returns a fit instance of self."""
        assert isinstance(fit_instance, BaseFeatureEngineer)

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def transform_returns_numpy_array_test(self, fit_instance, mock_input):
        """Assert, that transform returns a numpy array."""
        actual = fit_instance.transform(mock_input)

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float32

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def get_score_returns_dict_test(self, fit_instance, mock_input, mock_labels):
        """Assert, that get_score returns a dictionary."""
        actual = fit_instance.get_score(mock_input, mock_labels)

        assert isinstance(actual, dict)

    @pytest.mark.parametrize("fit_instance", FEATURE_ENGINEERS_TO_TEST, indirect=True)
    def get_report_returns_dict_test(self, fit_instance):
        """Assert, that get_score returns a dictionary."""
        actual = fit_instance.get_report()

        assert isinstance(actual, dict)
