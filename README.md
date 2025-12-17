# CorrectMatch

[![CI](https://github.com/computationalprivacy/pycorrectmatch/actions/workflows/test.yml/badge.svg)](https://github.com/computationalprivacy/pycorrectmatch/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/correctmatch.svg)](https://pypi.org/project/correctmatch/)

A thin Python wrapper around the Julia module CorrectMatch.jl, to estimate uniqueness from small population samples.

## Installation

Install CorrectMatch using your favorite package manager, e.g., pip:
```pip install correctmatch```
or uv:
```uv add correctmatch```


We use [JuliaCall](https://github.com/JuliaPy/PythonCall.jl) to seamlessly run Julia code from Python. The Julia binary and its dependencies, including [CorrectMatch.jl](https://github.com/computationalprivacy/CorrectMatch.jl), will be automatically installed on first use.

## Usage

This module estimates the uniqueness of a population, on which multiple discrete attributes can be collected. For instance, the following array is a sample of 1000 rows for five discrete attributes:
```python
>>> import numpy as np
>>> arr = np.random.randint(1, 5, size=(1000, 5))
>>> arr[:3, :]
array([[1, 1, 1, 3, 1],
       [3, 3, 2, 3, 3],
       [3, 3, 4, 3, 2]])
```

We can estimate the uniqueness of a population of 1000 individuals, or 10000 individuals, from this sample:

```python
>>> import correctmatch

>>> correctmatch.uniqueness(arr)  # empirical uniqueness for 1,000 records
0.371
>>> correctmatch.correctness(arr)  # empirical correctness for 1,000 records
0.637
```

by fitting a copula model to the observed records:

```python
>>> fitted_model = correctmatch.fit_model(arr)
>>> fitted_arr = correctmatch.sample_model(fitted_model, 1000)
>>> fitted_arr[:3, :]
array([[4, 2, 1, 4, 1],
       [4, 2, 3, 2, 3],
       [1, 3, 1, 3, 1]])
>>> correctmatch.uniqueness(fitted_arr)
0.373
>>> correctmatch.correctness(fitted_arr)
0.639
```

In the demo/ folder, we have compiled more examples with real-world datasets.

## License
GNU General Public License v3.0

See LICENSE to see the full text.

Patent-pending code. Additional support and details are available for commercial uses.
