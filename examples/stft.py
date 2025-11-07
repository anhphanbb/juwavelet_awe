import matplotlib.pyplot as plt
import numpy as np

import juwavelet.morlet as morlet


def example_wavelet_stft():
    dx = 0.01
    xs = np.arange(-80, 80, dx)
    fourier_factors = dict(
        [(param, 2 * np.pi / param)
         for param in [2, 4, 6, 8, 10, 12]])
    fig1, axs1 = plt.subplots(2, 1)
    fig2, axs2 = plt.subplots(2, 1)
    fig3, axs3 = plt.subplots(2, 1)
    fig4, axs4 = plt.subplots(2, 1)
    figs = [fig1, fig2, fig3, fig4]
    param = 6
    win = 10
    for i, period in enumerate([16, 8, 4, 2]):
        wave = morlet.morlet1d_real(
            len(xs), dx=dx, scale=period / fourier_factors[param], param=param)
        wave /= wave.max()

        wave_hat = morlet.morlet1d_fourier(
            len(xs), dx=dx, scale=period / fourier_factors[param], param=param)
        wave_hat = np.fft.fftshift(wave_hat)
        wave_hat /= wave_hat.max()

        ys = morlet.morlet1d_real(
            len(xs), dx=dx, scale=win, param=win * 2 * np.pi / period)
        ys /= ys.max()

        ys_hat = morlet.morlet1d_fourier(
            len(xs), dx=dx, scale=win, param=win * 2 * np.pi / period)
        ys_hat = np.fft.fftshift(ys_hat)
        ys_hat /= ys_hat.max()

        y2s = (np.exp(1j * (2 * np.pi) * xs / period))
        y2s[xs < -10] = 0
        y2s[xs > 10] = 0
        y2s /= ys.max()

        y2s_hat = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y2s)))
        y2s_hat /= y2s_hat.max()

        axs1[0].plot(xs, 2 * i + wave, label=str(period))
        axs1[1].plot(xs, 2 * i + ys.real, label=str(period))
        axs2[0].plot(xs, 2 * i + wave_hat.real, label=str(period))
        axs2[1].plot(xs, 2 * i + ys_hat.real, label=str(period))
        axs3[0].plot(xs, 2 * i + ys.real, label=str(period))
        axs3[1].plot(xs, 2 * i + y2s.real, label=str(period))
        axs4[0].plot(xs, 2 * i + ys_hat.real, label=str(period))
        axs4[1].plot(xs, 2 * i + y2s_hat.real, label=str(period))
    for axs in [axs1, axs2]:
        axs[0].set_title("Morlet 1-D wavelet (varying scales)")
        axs[1].set_title("Short-time Fourier transform with Gaussian window")
        axs[0].legend(loc="upper right")
        axs[1].legend(loc="upper right")
    for axs in [axs3, axs4]:
        axs[0].set_title("Short-time Fourier transform with Gaussian window")
        axs[1].set_title("Short-time Fourier transform with rectangular window")
    for axs in [axs1, axs3]:
        axs[0].set_xlim(-40, 40)
        axs[1].set_xlim(-40, 40)
    for axs in [axs2, axs4]:
        axs[0].set_xlim(-0.5, 1.5)
        axs[1].set_xlim(-0.5, 1.5)
    for fig, name in zip(figs, ("a", "b", "c", "d")):
        fig.tight_layout()
        fig.savefig("example_1dcomp_wav_stft_" + name + ".png")
        plt.close(fig)


if __name__ == "__main__":
    example_wavelet_stft()
