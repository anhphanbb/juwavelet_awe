import matplotlib.pyplot as plt
import numpy as np

import juwavelet.morlet as wavelet


def example_wavelet1d():
    dx = 0.1
    xs = np.arange(-8000, 8000, dx)
    fourier_factors = dict(
        [(param, 2 * np.pi / param)
         for param in [2, 4, 6, 8, 10, 12]])
    fig, axs = plt.subplots(2, 1)
    for param in [2, 4, 6, 8, 10]:
        wave = wavelet.morlet1d_fourier(
            len(xs), dx=dx, scale=100 / fourier_factors[param], param=param)
        kwave = np.fft.fftfreq(len(xs), d=dx)
        lw = 1 if param != 6 else 4
        wave /= wave.max()
        axs[1].plot(np.fft.fftshift(kwave), np.fft.fftshift(wave),
                    label=str(param), lw=lw)

        wave = wavelet.morlet1d_real(
            len(xs), dx=dx, scale=100 / fourier_factors[param], param=param)
        wave /= wave.max()
        axs[0].plot(xs, wave, label=str(param), lw=lw)
    axs[0].set_title("Morlet 1-D wavelet (varying scales, wavelength=100 km)")
    axs[0].set_xlim(-400, 400)
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel("distance (km)")
    axs[1].set_xticks([-0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04])
    axs[1].set_xticklabels([-25, -50, -100, "inf", 100, 50, 25])
    axs[1].set_xlim(-0.01, 0.02)
    axs[1].set_xlabel("wavelength (km)")
    axs[1].legend(loc="upper right")
    plt.savefig("example_wavelet1d.png")
    plt.show()


if __name__ == "__main__":
    example_wavelet1d()
