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
from juliacall import Main as jl  # noqa: N813

jl.seval("using CorrectMatch")
cm = jl.CorrectMatch


def uniqueness(arr: np.ndarray) -> float:
    """Compute the fraction of unique individuals in a discrete multivariate dataset."""
    return cm.uniqueness(arr)


def correctness(arr: np.ndarray) -> float:
    """Compute the fraction of correctly-identified individuals in a discrete multivariate dataset."""
    return cm.correctness(arr)


def fit_model(
    arr: np.ndarray,
    *,
    exact_marginal: bool = False,
    adaptative_threshold: int = 100,
    mi_abs_tol: float = 1e-5,
):
    """Fit a Gaussian copula model to a discrete multivariate dataset and return a Julia object with the estimated model."""
    return cm.fit_mle(
        cm.GaussianCopula,
        arr,
        exact_marginal=exact_marginal,
        adaptative_threshold=adaptative_threshold,
        mi_abs_tol=mi_abs_tol,
    )


def sample_model(m, size: int) -> np.ndarray:
    """Sample a synthetic dataset of `size` individuals from a given Gaussian copula model."""
    return jl.rand(m, size)


def individual_uniqueness(m, indiv: np.ndarray, n: int) -> float:
    """Estimate individual uniqueness for one given record."""
    return cm.individual_uniqueness(m, indiv, n)


def individual_correctness(m, indiv: np.ndarray, n: int) -> float:
    """Estimate individual correctness for one given record."""
    return cm.individual_correctness(m, indiv, n)
