# Portions of this module are adapted from XRD-AutoAnalyzer
# (https://github.com/njszym/XRD-AutoAnalyzer),
# Copyright (c) 2024 Nathan Szymanski, released under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import numpy as np
import random


class ShiftGen(object):
    """
    This class shifts the peaks by directly changing the diffraction angle.
    """

    def __init__(self, struc, max_shift=0.5, min_angle=10.0, max_angle=80.0):
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_shift = max_shift
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    def calc_std_dev(self, two_theta, tau):
        K = 0.9
        wavelength = self.calculator.wavelength * 0.1
        theta = np.radians(two_theta/2.)
        beta = (K * wavelength) / (np.cos(theta) * tau)
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2

    @property
    def shifted_spectrum(self):
        shift_range = np.linspace(-self.max_shift, self.max_shift, 1000)
        shift = random.choice(shift_range)

        angles = self.angles
        angles = np.array(angles) + shift

        intensities = self.intensities

        steps = np.linspace(self.min_angle, self.max_angle, 4501)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            idx = np.argmin(np.abs(ang-steps))
            signals[i, idx] = intensities[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        # domain_size differs from the other modules here.
        domain_size = 20.0
        step_size = (self.max_angle - self.min_angle) / 4501
        for i in range(signals.shape[0]):
            row = signals[i, :]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size,
                                                  mode='constant')
        # Sum the rows into a single signal.
        signal = np.sum(signals, axis=0)
        # Normalize the signal.
        norm_signal = 100 * signal / max(signal)

        noise = np.random.normal(0, 0.25, 4501)
        noisy_signal = norm_signal + noise

        form_signal = [[val] for val in noisy_signal]
        return form_signal


def main(struc, num_broadened, max_shift, min_angle=10.0, max_angle=80.0):
    shift_generator = ShiftGen(struc, max_shift, min_angle, max_angle)
    shifted_patterns = [shift_generator.shifted_spectrum for i in range(num_broadened)]
    return shifted_patterns





