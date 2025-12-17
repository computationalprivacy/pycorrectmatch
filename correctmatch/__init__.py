# CorrectMatch
# Copyright © 2019 Université catholique de Louvain, UCLouvain
# Copyright © 2019 Imperial College London
# by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from juliacall import Main as jl  # noqa: N813

jl.seval("using CorrectMatch")
jl.seval("using PythonCall")
jl.seval("using DataFrames")
cm = jl.CorrectMatch


def _to_julia_data(arr: np.ndarray | pd.DataFrame):
    """Convert input data to the appropriate Julia format (numpy array or DataFrame)."""
    if isinstance(arr, np.ndarray):
        return arr

    if isinstance(arr, pd.DataFrame):
        return jl.DataFrame(jl.PythonCall.pytable(arr))

    raise TypeError("Input data must be a numpy array or a pandas DataFrame.")


def _to_integer_array(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert a pandas DataFrame to an integer numpy array for simple functions."""
    if isinstance(arr, pd.DataFrame):
        result = np.empty((len(arr), len(arr.columns)), dtype=np.int64)
        for i, col in enumerate(arr.columns):
            if arr[col].dtype == "category":
                result[:, i] = arr[col].cat.codes
            elif arr[col].dtype == object:
                result[:, i] = pd.Categorical(arr[col]).codes
            else:
                result[:, i] = arr[col].to_numpy()
        return result
    return arr


def uniqueness(arr: np.ndarray | pd.DataFrame) -> float:
    """Compute the fraction of unique individuals in a discrete multivariate dataset."""
    arr = _to_integer_array(arr)
    return cm.uniqueness(arr)


def correctness(arr: np.ndarray | pd.DataFrame) -> float:
    """Compute the fraction of correctly-identified individuals in a discrete multivariate dataset."""
    arr = _to_integer_array(arr)
    return cm.correctness(arr)


def fit_model(
    arr: np.ndarray | pd.DataFrame,
    *,
    exact_marginal: bool = True,
    adaptative_threshold: int = 100,
    mi_abs_tol: float = 1e-5,
):
    """Fit a Gaussian copula model to a discrete multivariate dataset and return a Julia object with the estimated model."""
    data = _to_julia_data(arr)
    return cm.fit_mle(
        cm.GaussianCopula,
        data,
        exact_marginal=exact_marginal,
        adaptative_threshold=adaptative_threshold,
        mi_abs_tol=mi_abs_tol,
    )


def sample_model(m, size: int) -> np.ndarray | pd.DataFrame:
    """Sample a synthetic dataset of `size` individuals from a given Gaussian copula model."""
    res = jl.rand(m, size)
    if cm.returns_dataframe(m):
        return jl.PythonCall.pytable(res)
    return np.array(res)


def _series_to_vector(indiv: np.ndarray | pd.Series) -> list | np.ndarray:
    """Convert a pandas Series to a Julia vector, preserving original values."""
    if isinstance(indiv, pd.Series):
        return list(indiv.values)
    return indiv


def individual_uniqueness(m, indiv: np.ndarray | pd.Series, n: int) -> float:
    """Estimate individual uniqueness for one given record."""
    indiv = _series_to_vector(indiv)
    return cm.individual_uniqueness(m, indiv, n)


def individual_correctness(m, indiv: np.ndarray | pd.Series, n: int) -> float:
    """Estimate individual correctness for one given record."""
    indiv = _series_to_vector(indiv)
    return cm.individual_correctness(m, indiv, n)
