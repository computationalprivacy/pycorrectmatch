"""Tests for the correctmatch module."""

import correctmatch
import numpy as np
import pandas as pd
import pytest


class TestJuliaIntegration:
    """Test that Julia and CorrectMatch.jl are properly installed."""

    def test_import_module(self) -> None:
        """Test that the correctmatch module can be imported."""
        assert hasattr(correctmatch, "uniqueness")
        assert hasattr(correctmatch, "correctness")
        assert hasattr(correctmatch, "fit_model")
        assert hasattr(correctmatch, "sample_model")
        assert hasattr(correctmatch, "individual_uniqueness")
        assert hasattr(correctmatch, "individual_correctness")

    def test_julia_correctmatch_imported(self) -> None:
        """Test that the Julia CorrectMatch package is accessible."""
        from juliacall import Main as jl

        assert jl.CorrectMatch is not None


class TestUniqueness:
    """Test the uniqueness function."""

    def test_uniqueness_all_unique(self) -> None:
        """Test uniqueness with a dataset where all rows are unique."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = correctmatch.uniqueness(arr)
        assert result == 1.0

    def test_uniqueness_with_duplicates(self) -> None:
        """Test uniqueness with a dataset containing duplicates."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = correctmatch.uniqueness(arr)
        assert 0.0 <= result <= 1.0
        assert result < 1.0

    def test_uniqueness_all_same(self) -> None:
        """Test uniqueness with a dataset where all rows are identical."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        result = correctmatch.uniqueness(arr)
        assert result == 0.0


class TestCorrectness:
    """Test the correctness function."""

    def test_correctness_all_unique(self) -> None:
        """Test correctness with a dataset where all rows are unique."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = correctmatch.correctness(arr)
        assert result == 1.0

    def test_correctness_with_duplicates(self) -> None:
        """Test correctness with a dataset containing duplicates."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = correctmatch.correctness(arr)
        assert result == (1/2 * 2 + 1) / 3

    def test_correctness_all_same(self) -> None:
        """Test correctness with a dataset where all rows are identical."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        result = correctmatch.correctness(arr)
        assert result == 1/3


class TestGaussianCopulaModel:
    """Test Gaussian copula model fitting and sampling."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample data for model testing."""
        rng = np.random.default_rng(42)
        return rng.integers(1, 10, size=(1000, 5))

    def test_fit_model(self, sample_data: np.ndarray) -> None:
        """Test that a model can be fitted to data."""
        model = correctmatch.fit_model(sample_data)
        assert model is not None

    def test_sample_model(self, sample_data: np.ndarray) -> None:
        """Test sampling from a fitted model."""
        model = correctmatch.fit_model(sample_data)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        assert samples.shape == (50, sample_data.shape[1])


class TestIndividualMetrics:
    """Test individual uniqueness and correctness estimation."""

    @pytest.fixture
    def model_and_individual(self) -> tuple:
        """Create a model and individual for testing."""
        rng = np.random.default_rng(42)
        data = rng.integers(1, 10, size=(1000, 5))
        model = correctmatch.fit_model(data)
        individual = data[0]
        return model, individual

    def test_individual_uniqueness(self, model_and_individual: tuple) -> None:
        """Test individual uniqueness estimation."""
        model, individual = model_and_individual
        result = correctmatch.individual_uniqueness(model, individual, 1000)
        assert 0.0 <= result <= 1.0

    def test_individual_correctness(self, model_and_individual: tuple) -> None:
        """Test individual correctness estimation."""
        model, individual = model_and_individual
        result = correctmatch.individual_correctness(model, individual, 1000)
        assert 0.0 <= result <= 1.0


class TestDataFrameIntegerColumns:
    """Test functions with DataFrames containing integer columns."""

    @pytest.fixture
    def int_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with integer columns."""
        rng = np.random.default_rng(42)
        data = rng.integers(1, 10, size=(100, 5))
        return pd.DataFrame(data, columns=["a", "b", "c", "d", "e"])

    def test_uniqueness_dataframe_int(self, int_dataframe: pd.DataFrame) -> None:
        """Test uniqueness with a DataFrame of integers."""
        result = correctmatch.uniqueness(int_dataframe)
        assert 0.0 <= result <= 1.0

    def test_correctness_dataframe_int(self, int_dataframe: pd.DataFrame) -> None:
        """Test correctness with a DataFrame of integers."""
        result = correctmatch.correctness(int_dataframe)
        assert 0.0 <= result <= 1.0

    def test_fit_model_dataframe_int(self, int_dataframe: pd.DataFrame) -> None:
        """Test fitting a model with a DataFrame of integers."""
        model = correctmatch.fit_model(int_dataframe)
        assert model is not None

    def test_sample_model_dataframe_int(self, int_dataframe: pd.DataFrame) -> None:
        """Test sampling from a model fitted with a DataFrame of integers."""
        model = correctmatch.fit_model(int_dataframe)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        assert len(samples) == 50

    def test_individual_uniqueness_dataframe_int(
        self, int_dataframe: pd.DataFrame
    ) -> None:
        """Test individual uniqueness with a DataFrame of integers."""
        model = correctmatch.fit_model(int_dataframe)
        individual = int_dataframe.iloc[0]
        result = correctmatch.individual_uniqueness(model, individual, 100)
        assert 0.0 <= result <= 1.0

    def test_individual_correctness_dataframe_int(
        self, int_dataframe: pd.DataFrame
    ) -> None:
        """Test individual correctness with a DataFrame of integers."""
        model = correctmatch.fit_model(int_dataframe)
        individual = int_dataframe.iloc[0]
        result = correctmatch.individual_correctness(model, individual, 100)
        assert 0.0 <= result <= 1.0


class TestDataFrameCategoryColumns:
    """Test functions with DataFrames containing categorical columns."""

    @pytest.fixture
    def category_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with categorical columns."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "color": rng.choice(["red", "green", "blue"], size=100),
                "size": rng.choice(["small", "medium", "large"], size=100),
                "shape": rng.choice(["circle", "square", "triangle"], size=100),
            }
        )
        for col in df.columns:
            df[col] = df[col].astype("category")
        return df

    def test_uniqueness_dataframe_category(
        self, category_dataframe: pd.DataFrame
    ) -> None:
        """Test uniqueness with a DataFrame of categorical columns."""
        result = correctmatch.uniqueness(category_dataframe)
        assert 0.0 <= result <= 1.0

    def test_correctness_dataframe_category(
        self, category_dataframe: pd.DataFrame
    ) -> None:
        """Test correctness with a DataFrame of categorical columns."""
        result = correctmatch.correctness(category_dataframe)
        assert 0.0 <= result <= 1.0

    def test_fit_model_dataframe_category(
        self, category_dataframe: pd.DataFrame
    ) -> None:
        """Test fitting a model with a DataFrame of categorical columns."""
        model = correctmatch.fit_model(category_dataframe)
        assert model is not None

    def test_sample_model_dataframe_category(
        self, category_dataframe: pd.DataFrame
    ) -> None:
        """Test sampling from a model fitted with categorical columns."""
        model = correctmatch.fit_model(category_dataframe)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        assert len(samples) == 50

    def test_individual_uniqueness_dataframe_category(
        self, category_dataframe: pd.DataFrame
    ) -> None:
        """Test individual uniqueness with a DataFrame of categorical columns."""
        model = correctmatch.fit_model(category_dataframe)
        individual = category_dataframe.iloc[0]
        result = correctmatch.individual_uniqueness(model, individual, 100)
        assert 0.0 <= result <= 1.0

    def test_individual_correctness_dataframe_category(
        self, category_dataframe: pd.DataFrame
    ) -> None:
        """Test individual correctness with a DataFrame of categorical columns."""
        model = correctmatch.fit_model(category_dataframe)
        individual = category_dataframe.iloc[0]
        result = correctmatch.individual_correctness(model, individual, 100)
        assert 0.0 <= result <= 1.0


class TestDataFrameStringColumns:
    """Test functions with DataFrames containing string columns."""

    @pytest.fixture
    def string_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with string columns."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "city": rng.choice(["Paris", "London", "Berlin", "Madrid"], size=100),
                "country": rng.choice(["France", "UK", "Germany", "Spain"], size=100),
                "language": rng.choice(
                    ["French", "English", "German", "Spanish"], size=100
                ),
            }
        )

    def test_uniqueness_dataframe_string(
        self, string_dataframe: pd.DataFrame
    ) -> None:
        """Test uniqueness with a DataFrame of string columns."""
        result = correctmatch.uniqueness(string_dataframe)
        assert 0.0 <= result <= 1.0

    def test_correctness_dataframe_string(
        self, string_dataframe: pd.DataFrame
    ) -> None:
        """Test correctness with a DataFrame of string columns."""
        result = correctmatch.correctness(string_dataframe)
        assert 0.0 <= result <= 1.0

    def test_fit_model_dataframe_string(self, string_dataframe: pd.DataFrame) -> None:
        """Test fitting a model with a DataFrame of string columns."""
        model = correctmatch.fit_model(string_dataframe)
        assert model is not None

    def test_sample_model_dataframe_string(
        self, string_dataframe: pd.DataFrame
    ) -> None:
        """Test sampling from a model fitted with string columns."""
        model = correctmatch.fit_model(string_dataframe)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        assert len(samples) == 50

    def test_individual_uniqueness_dataframe_string(
        self, string_dataframe: pd.DataFrame
    ) -> None:
        """Test individual uniqueness with a DataFrame of string columns."""
        model = correctmatch.fit_model(string_dataframe, exact_marginal=True)
        individual = string_dataframe.iloc[0]
        result = correctmatch.individual_uniqueness(model, individual, 100)
        assert 0.0 <= result <= 1.0

    def test_individual_correctness_dataframe_string(
        self, string_dataframe: pd.DataFrame
    ) -> None:
        """Test individual correctness with a DataFrame of string columns."""
        model = correctmatch.fit_model(string_dataframe, exact_marginal=True)
        individual = string_dataframe.iloc[0]
        result = correctmatch.individual_correctness(model, individual, 100)
        assert 0.0 <= result <= 1.0


class TestDataFrameMixedColumns:
    """Test functions with DataFrames containing mixed column types."""

    @pytest.fixture
    def mixed_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with mixed column types."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "age": rng.integers(18, 80, size=100),
                "gender": pd.Categorical(rng.choice(["M", "F"], size=100)),
                "city": rng.choice(["Paris", "London", "Berlin"], size=100),
            }
        )
        return df

    def test_uniqueness_dataframe_mixed(self, mixed_dataframe: pd.DataFrame) -> None:
        """Test uniqueness with a DataFrame of mixed column types."""
        result = correctmatch.uniqueness(mixed_dataframe)
        assert 0.0 <= result <= 1.0

    def test_correctness_dataframe_mixed(self, mixed_dataframe: pd.DataFrame) -> None:
        """Test correctness with a DataFrame of mixed column types."""
        result = correctmatch.correctness(mixed_dataframe)
        assert 0.0 <= result <= 1.0

    def test_fit_model_dataframe_mixed(self, mixed_dataframe: pd.DataFrame) -> None:
        """Test fitting a model with a DataFrame of mixed column types."""
        model = correctmatch.fit_model(mixed_dataframe)
        assert model is not None

    def test_sample_model_dataframe_mixed(self, mixed_dataframe: pd.DataFrame) -> None:
        """Test sampling from a model fitted with mixed column types."""
        model = correctmatch.fit_model(mixed_dataframe)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        assert len(samples) == 50
