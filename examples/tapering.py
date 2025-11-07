import matplotlib.pyplot as plt
import numpy as np

import juwavelet.utils as utils


def example_tapering():
    data = np.ones(100)
    plt.plot(utils.smooth_edges(data, 20), label="blackman")
    plt.plot(utils.smooth_edges(data, 20, window="kaiser"), label="kaiser")
    plt.plot(utils.smooth_edges(data, 20, window="cos"), label="cos")
    plt.plot(utils.smooth_edges(data, 20, window="linear"), label="linear")
    plt.legend()
    plt.savefig("example_tapering1d.png")
    plt.figure()
    data = np.ones((100, 100))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(utils.smooth_edges(data, 20), label="blackman")
    plt.subplot(2, 2, 2)
    plt.pcolormesh(utils.smooth_edges(data, 20, window="kaiser"), label="kaiser")
    plt.subplot(2, 2, 3)
    plt.pcolormesh(utils.smooth_edges(data, 20, window="cos"), label="cos")
    plt.subplot(2, 2, 4)
    plt.pcolormesh(utils.smooth_edges(data, 20, window="linear"), label="linear")
    plt.colorbar()
    plt.savefig("example_tapering2d.png")
    # plt.show()


if __name__ == "__main__":
    example_tapering()
