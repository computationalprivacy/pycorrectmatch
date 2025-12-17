"""Tests for the correctmatch module."""

import correctmatch
import numpy as np
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
    """Test the correctness function.

    Note: The correctness function is defined in the Python wrapper but
    not currently exported by the Julia CorrectMatch package.
    These tests are skipped until the Julia package adds this functionality.
    """

    @pytest.mark.skip(reason="correctness not available in CorrectMatch.jl")
    def test_correctness_all_unique(self) -> None:
        """Test correctness with a dataset where all rows are unique."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = correctmatch.correctness(arr)
        assert result == 1.0

    @pytest.mark.skip(reason="correctness not available in CorrectMatch.jl")
    def test_correctness_with_duplicates(self) -> None:
        """Test correctness with a dataset containing duplicates."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = correctmatch.correctness(arr)
        assert 0.0 <= result <= 1.0

    @pytest.mark.skip(reason="correctness not available in CorrectMatch.jl")
    def test_correctness_all_same(self) -> None:
        """Test correctness with a dataset where all rows are identical."""
        arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        result = correctmatch.correctness(arr)
        assert result == 0.0


class TestGaussianCopulaModel:
    """Test Gaussian copula model fitting and sampling."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample data for model testing."""
        rng = np.random.default_rng(42)
        return rng.integers(1, 10, size=(100, 5))

    def test_fit_model(self, sample_data: np.ndarray) -> None:
        """Test that a model can be fitted to data."""
        model = correctmatch.fit_model(sample_data)
        assert model is not None

    def test_fit_model_exact_marginal(self, sample_data: np.ndarray) -> None:
        """Test fitting a model with exact marginal option."""
        model = correctmatch.fit_model(sample_data, exact_marginal=True)
        assert model is not None

    def test_sample_model(self, sample_data: np.ndarray) -> None:
        """Test sampling from a fitted model."""
        model = correctmatch.fit_model(sample_data)
        samples = correctmatch.sample_model(model, 50)
        assert samples is not None
        # samples shape is (n_samples, n_features)
        assert samples.shape[0] == 50
        assert samples.shape[1] == sample_data.shape[1]


class TestIndividualMetrics:
    """Test individual uniqueness and correctness estimation."""

    @pytest.fixture
    def model_and_individual(self) -> tuple:
        """Create a model and individual for testing."""
        rng = np.random.default_rng(42)
        data = rng.integers(1, 10, size=(100, 5))
        model = correctmatch.fit_model(data, exact_marginal=True)
        individual = data[0]
        return model, individual

    def test_individual_uniqueness(self, model_and_individual: tuple) -> None:
        """Test individual uniqueness estimation."""
        model, individual = model_and_individual
        result = correctmatch.individual_uniqueness(model, individual, 1000)
        assert 0.0 <= result <= 1.0

    @pytest.mark.skip(reason="individual_correctness not available in CorrectMatch.jl")
    def test_individual_correctness(self, model_and_individual: tuple) -> None:
        """Test individual correctness estimation."""
        model, individual = model_and_individual
        result = correctmatch.individual_correctness(model, individual, 1000)
        assert 0.0 <= result <= 1.0
