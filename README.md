# CorrectMatch

A thin Python wrapper around the Julia module CorrectMatch.jl, to estimate uniqueness from small population samples.

## Installation

Install first [Julia](http://julialang.org) and [CorrectMatch.jl](https://github.com/computationalprivacy/CorrectMatch.jl), then this Python wrapper:
```pip install correctmatch```

We use [PyJulia](https://github.com/JuliaPy/pyjulia) to seemingly run Julia code from Python. Your Julia installation should be automatically detected, otherwise follow the instruction on the [PyJulia documentation](https://github.com/JuliaPy/pyjulia).

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
>>> correctmatch.precompile()  # precompile the Julia module

>>> correctmatch.uniqueness(arr)  # true uniqueness for 1,000 records
0.38
```

by fitting a copula model to the observed records:

```
>>> fitted_model = correctmatch.fit_model(arr)
>>> fitted_arr = correctmatch.sample_model(fitted_model, 1000)
>>> fitted_arr[:3, :]
array([[4, 2, 1, 4, 1],
       [4, 2, 3, 2, 3],
       [1, 3, 1, 3, 1]])
>>> correctmatch.uniqueness(fitted_arr)
0.393
```

In the demo/ folder, we have compiled more examples with real-world datasets.

## License
GNU General Public License v3.0

See LICENSE to see the full text.

Patent-pending code. Additional support and details are available for commercial uses.
