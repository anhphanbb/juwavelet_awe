JUWAVELET
=========

This package  implements the continuous wavelet transform in 1-D, 2-D and 3-D employing the Morlet wavelet.

Please contact <j.ungermann@fz-juelich.de> if you require assistance.


Installation
------------

Clone the repository and install it with `pip install .`.
For fast execution, there are optional depencies, mainly mkl\_fft (or pyfftw)
for fast execution of FFT and numba. Use `pip install .\[compiled\]` to
install juwavelet using these additional dependencies.


Testing
-------

Tests can be executed using a local install with `pip install -e .` and `pytest`.


Examples
--------

The examples directory contains several examples for analysis and filtering.
