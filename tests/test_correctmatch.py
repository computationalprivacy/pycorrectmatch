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
    """Test the uniqueness function with known expected values."""

    @pytest.mark.parametrize(
        ("arr", "expected"),
        [
            (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1.0),  # all unique
            (np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), 0.0),  # all same
        ],
    )
    def test_uniqueness_exact(self, arr: np.ndarray, expected: float) -> None:
        """Test uniqueness with exact expected values."""
        assert correctmatch.uniqueness(arr) == expected

    def test_uniqueness_with_duplicates(self) -> None:
        """Test uniqueness with partial duplicates returns value in (0, 1)."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = correctmatch.uniqueness(arr)
        assert 0.0 < result < 1.0


class TestCorrectness:
    """Test the correctness function with known expected values."""

    @pytest.mark.parametrize(
        ("arr", "expected"),
        [
            (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1.0),  # all unique
            (np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), (1 / 2 * 2 + 1) / 3),  # partial
            (np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), 1 / 3),  # all same
        ],
    )
    def test_correctness(self, arr: np.ndarray, expected: float) -> None:
        """Test correctness with exact expected values."""
        assert correctmatch.correctness(arr) == expected


class TestModelOperations:
    """Test model fitting, sampling, and individual metrics across all data types."""

    DATA_TYPES = ["numpy", "integer", "category", "string", "mixed"]

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    def test_fit_model(self, dataframe_factory: callable, data_type: str) -> None:
        """Test model fitting works for all data types."""
        data = dataframe_factory(data_type)
        model = correctmatch.fit_model(data)
        assert model is not None

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    def test_sample_model(self, dataframe_factory: callable, data_type: str) -> None:
        """Test sampling returns correct shape for all data types."""
        data = dataframe_factory(data_type)
        model = correctmatch.fit_model(data)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        assert len(samples) == 50

    @pytest.mark.parametrize(
        "data_type,exact_marginal,size",
        [
            ("numpy", False, 1000),
            ("integer", False, 1000),
            ("category", False, 100),
            ("string", True, 100),
            # Note: mixed type not tested for individual metrics (numerically unstable)
        ],
    )
    @pytest.mark.parametrize(
        "metric_fn",
        [
            correctmatch.individual_uniqueness,
            correctmatch.individual_correctness,
        ],
        ids=["uniqueness", "correctness"],
    )
    def test_individual_metrics(
        self,
        dataframe_factory: callable,
        data_type: str,
        exact_marginal: bool,
        size: int,
        metric_fn: callable,
    ) -> None:
        """Test individual metrics return valid [0,1] values for all data types.

        Note: numpy/integer/mixed require larger datasets (1000 rows) for numerical
        stability in Julia's Gaussian copula fitting without exact_marginal.
        """
        data = dataframe_factory(data_type, size=size)
        model = correctmatch.fit_model(data, exact_marginal=exact_marginal)
        individual = data[0] if isinstance(data, np.ndarray) else data.iloc[0]
        result = metric_fn(model, individual, 100)
        assert 0.0 <= result <= 1.0


class TestDataFrameMetrics:
    """Test uniqueness/correctness with different DataFrame types."""

    DF_TYPES = ["integer", "category", "string", "mixed"]

    @pytest.mark.parametrize("data_type", DF_TYPES)
    @pytest.mark.parametrize(
        "metric_fn",
        [
            correctmatch.uniqueness,
            correctmatch.correctness,
        ],
        ids=["uniqueness", "correctness"],
    )
    def test_dataframe_metrics(self, dataframe_factory, data_type: str, metric_fn) -> None:
        """Test metrics return valid [0,1] values for all DataFrame types."""
        df = dataframe_factory(data_type)
        result = metric_fn(df)
        assert 0.0 <= result <= 1.0


class TestSeriesToVector:
    """Test _series_to_vector helper function preserves correct types for Julia."""

    def test_numpy_array_passthrough(self) -> None:
        """Numpy arrays should be returned unchanged."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = correctmatch._series_to_vector(arr)
        assert result is arr  # Same object, not a copy

    def test_integer_series_returns_numpy_array(self) -> None:
        """Integer Series should return numpy array to preserve Int64 type in Julia."""
        series = pd.Series([1, 2, 3], dtype=np.int64)
        result = correctmatch._series_to_vector(series)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    def test_float_series_returns_numpy_array(self) -> None:
        """Float Series should return numpy array to preserve Float64 type in Julia."""
        series = pd.Series([1.0, 2.0, 3.0], dtype=np.float64)
        result = correctmatch._series_to_vector(series)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_categorical_series_returns_list(self) -> None:
        """Categorical Series should return list with string values."""
        series = pd.Series(pd.Categorical(["a", "b", "c"]))
        result = correctmatch._series_to_vector(series)
        assert isinstance(result, list)
        assert result == ["a", "b", "c"]

    def test_string_series_returns_list(self) -> None:
        """String (object) Series should return list."""
        series = pd.Series(["a", "b", "c"], dtype=object)
        result = correctmatch._series_to_vector(series)
        assert isinstance(result, list)
        assert result == ["a", "b", "c"]

    def test_mixed_dtype_series_returns_list(self) -> None:
        """Mixed dtype Series (object) should return list."""
        # This happens when extracting a row from a DataFrame with mixed column types
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        series = df.iloc[0]  # dtype=object due to mixed types
        result = correctmatch._series_to_vector(series)
        assert isinstance(result, list)

    def test_integer_series_values_preserved(self) -> None:
        """Integer values should be preserved exactly."""
        series = pd.Series([10, 20, 30], dtype=np.int64)
        result = correctmatch._series_to_vector(series)
        np.testing.assert_array_equal(result, [10, 20, 30])

    def test_julia_receives_correct_type_for_integers(self) -> None:
        """Verify Julia receives Int64 (not Any) for integer Series."""
        from juliacall import Main as jl

        series = pd.Series([1, 2, 3], dtype=np.int64)
        converted = correctmatch._series_to_vector(series)
        julia_eltype = str(jl.seval("eltype")(converted))
        assert "Int64" in julia_eltype, f"Expected Int64, got {julia_eltype}"

    def test_julia_receives_correct_type_for_floats(self) -> None:
        """Verify Julia receives Float64 (not Any) for float Series."""
        from juliacall import Main as jl

        series = pd.Series([1.0, 2.0, 3.0], dtype=np.float64)
        converted = correctmatch._series_to_vector(series)
        julia_eltype = str(jl.seval("eltype")(converted))
        assert "Float64" in julia_eltype, f"Expected Float64, got {julia_eltype}"
