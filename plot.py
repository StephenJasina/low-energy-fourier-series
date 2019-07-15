import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

class FourierSeries:
    """Class representing a band-limited double Fourier series.

    Attributes:
        coefficients: Dictionary of coefficients. The value
            corresponding to the key (k1, k2) is the coefficient on
            the term exp(2 * pi * i * (k1 * x2 + k2 * x2)).
    """

    def __init__(self, coefficients):
        self._coefficients = copy.deepcopy(coefficients)

    def get_coefficients(self):
        return copy.deepcopy(self._coefficients)

    def plot(self, x1, x2, figure = None):
        """
        Plots the real part of the series evaluated on the rectangle
        spanned by ``x1`` and ``x2``. Both ranges are assumed to be
        equally spaced and increasing.

        Args:
            x1 (np.array): Range for the first coordinate.
            x2 (np.array): Range for the second coordinate.
        """

        x1, x2 = np.meshgrid(x1, x2, indexing = 'ij')

        x3 = np.zeros(x1.shape)
        for k, ak in self._coefficients.items():
            x3 += (ak * np.exp(2 * np.pi * 1j * (k[0] * x1 + k[1] * x2))).real

        plt.contourf(x1, x2, x3, 20, cmap='RdGy')
        plt.colorbar()

    def plot_hessian_determinant(self, x1, x2, figure = None):
        """
        Plots the real part of the Hessian determinant of the series
        evaluated on the rectangle spanned by ``x1`` and ``x2``. Both
        ranges are assumed to be equally spaced and increasing.

        Args:
            x1 (np.array): Range for the first coordinate.
            x2 (np.array): Range for the second coordinate.
        """

        x1, x2 = np.meshgrid(x1, x2, indexing = 'ij')

        h00 = np.zeros(x1.shape)
        h01 = np.zeros(x1.shape)
        h11 = np.zeros(x1.shape)

        for k, ak in self._coefficients.items():
            h00 += (-(2 * np.pi * k[0])**2 * ak
                    * np.exp(2 * np.pi * 1j * (k[0] * x1 + k[1] * x2))).real
            h01 += (-(2 * np.pi)**2 * k[0] * k[1] * ak
                    * np.exp(2 * np.pi * 1j * (k[0] * x1 + k[1] * x2))).real
            h11 += (-(2 * np.pi * k[1])**2 * ak
                    * np.exp(2 * np.pi * 1j * (k[0] * x1 + k[1] * x2))).real

        x3 = h00 * h11 - h01 * h01

        plt.contourf(x1, x2, x3, 20, cmap='RdGy')
        plt.colorbar()


if len(sys.argv) < 2:
    print("Must supply file")
    sys.exit()

x1 = np.arange(0, 2, .01)
x2 = np.arange(0, 2, .01)

with open(sys.argv[1], 'r') as file:
    coefficients = eval(file.read())

    for k, ak in sorted(coefficients.items(), key = lambda i : -abs(i[1])):
        if abs(ak) > 1e-3:
            print('{0:<24}{1}'.format((k[0]**2 + k[1]**2)**0.5, abs(ak)))
        # Uncomment the following two lines to ignore coefficients that are too
        # small
        """else:
            del(coefficients[k])"""

    series = FourierSeries(coefficients)

    plt.figure(1)
    series.plot(x1, x2)
    plt.figure(2)
    series.plot_hessian_determinant(x1, x2)
    plt.show()
