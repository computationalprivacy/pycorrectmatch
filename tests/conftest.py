"""Shared pytest fixtures for correctmatch tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def dataframe_factory(rng: np.random.Generator) -> callable:
    """Create test data by type."""

    def _create(data_type: str, size: int | None = None) -> pd.DataFrame | np.ndarray:
        # Default size: 1000 for numpy (numerical stability), 100 for DataFrames
        if size is None:
            size = 1000 if data_type == "numpy" else 100

        if data_type == "numpy":
            return rng.integers(1, 10, size=(size, 5))

        if data_type == "integer":
            data = rng.integers(1, 10, size=(size, 5))
            return pd.DataFrame(data, columns=["a", "b", "c", "d", "e"])

        if data_type == "category":
            df = pd.DataFrame(
                {
                    "color": rng.choice(["red", "green", "blue"], size=size),
                    "size": rng.choice(["small", "medium", "large"], size=size),
                    "shape": rng.choice(["circle", "square", "triangle"], size=size),
                }
            )
            for col in df.columns:
                df[col] = df[col].astype("category")
            return df

        if data_type == "string":
            return pd.DataFrame(
                {
                    "city": rng.choice(["Paris", "London", "Berlin", "Madrid"], size=size),
                    "country": rng.choice(["France", "UK", "Germany", "Spain"], size=size),
                    "language": rng.choice(["French", "English", "German", "Spanish"], size=size),
                }
            )

        if data_type == "mixed":
            return pd.DataFrame(
                {
                    "age": rng.integers(18, 80, size=size),
                    "gender": pd.Categorical(rng.choice(["M", "F"], size=size)),
                    "city": rng.choice(["Paris", "London", "Berlin"], size=size),
                }
            )

        raise ValueError(f"Unknown data type: {data_type}")

    return _create
