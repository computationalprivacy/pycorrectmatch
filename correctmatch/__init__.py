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

from julia import Main
jeval = Main.eval

def precompile():
    """
    Precompile and load the Julia package CorrectMatch.jl
    """
    jeval("using CorrectMatch; using CorrectMatch: Copula, Uniqueness, Individual")


def uniqueness(arr):
    """
    Compute the fraction of unique individuals in a discrete multivariate
    dataset.
    """
    return jeval("uniqueness")(arr)


def fit_model(arr, exact_marginal=False):
    """
    Fit a Gaussian copula model to a discrete multivariate dataset
    and return a Julia object with the estimated model.
    """
    return jeval("(x, y) -> fit_mle(GaussianCopula, x; exact_marginal=y)")(arr, exact_marginal)


def sample_model(m, size):
    """
    Sample a synthetic dataset of `size` individuals from a given Gaussian
    copula model.
    """
    return jeval("rand")(m, size)


def individual_uniqueness(m, indiv, n):
    """ Estimate individual uniqueness for one given record. """
    return jeval("individual_uniqueness")(m, indiv, n)
