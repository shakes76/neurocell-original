# Neurocell Library
Cellular machines library for automated systems and self assembly with John von Neumann Networks (VNNs).

## Structure
In general, driver scripts are prefixed with tests_* and unit tests as unit_*.

The descriptions of the various scripts can be found on the [experiments page](experiments.md)

## Implementation
JAX library based implementation of the cellular machines as JAX Numpy arrays

Includes implementation of the Von Neumann Networks and Game of Life (GoL) in JAX.

Notes can be found in the [docs](doc/README.md) page.

## Requirements
See the requirements file provided.

The following setup was needed for Ubuntu/Pop OS! 24.04. Firstly, we needed to install JAX with this particular version found to be stable
```
pip install -U "jax[cuda13]"==0.7.2
```
Then we need to install Haiku and other dependencies with stable versions found where indicated
```
pip install optax einops flax==0.10.7 dm-haiku==0.0.14
pip install scikit-learn scikit-image seaborn
```

## License
This software and notes are made available under the GNU Lesser General Public License v. 3 (“LGPL”). This means you are able to use it as a library, while keeping your source code closed as long as LGPL requirements are met. Changes to the library or notes need to be released under the same license as per the LGPL.

Private licenses are available on request.
