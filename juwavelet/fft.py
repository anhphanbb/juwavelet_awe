import numpy.fft


def use_numpy():
    global fft
    fft = numpy.fft


def use_mkl():
    import mkl_fft.interfaces.numpy_fft
    global fft
    fft = mkl_fft.interfaces.numpy_fft


def use_pyfftw():
    import pyfftw
    pyfftw.interfaces.cache.enable()
    import pyfftw.interfaces.numpy_fft
    global fft
    fft = pyfftw.interfaces.numpy_fft


def use_cupy():
    import cupy
    import cupy.fft
    global fft
    fft = cupy.fft


try:
    use_mkl()
except ImportError:
    try:
        use_pyfftw()
    except ImportError:
        use_numpy()
