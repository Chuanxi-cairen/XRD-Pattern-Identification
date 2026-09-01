# Portions of this module are adapted from XRD-AutoAnalyzer
# (https://github.com/njszym/XRD-AutoAnalyzer),
# Copyright (c) 2024 Nathan Szymanski, released under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import random
import numpy as np


class TextureGen(object):
    """
    This class simulates peak intensity changes by scaling the intensities
    according to texture along a randomly chosen crystal direction.
    """

    def __init__(self, struc, max_texture=0.6, min_angle=10.0, max_angle=80.0):
        """
        :param max_texture: maximum intensity change due to texture.
            For example, max_texture=0.6 means peak intensities change by up
            to +/-60% from the original values.
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_texture = max_texture
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def pattern(self):
        struc = self.struc
        return self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensity(self):
        return self.pattern.y

    @property
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

    def map_interval(self, v):
        """
        Map a value v from [0, 1] to [1 - max_texture, 1].
        """
        bound = 1.0 - self.max_texture
        return bound + (((1.0 - bound) / (1.0 - 0.0)) * (v - 0.0))

    @property
    def texture_intensities(self):
        hkls, intensities = self.hkl_list, self.intensity
        scaled_intensities = []

        # Hexagonal systems have four Miller indices.
        if self.struc.lattice.is_hexagonal():
            check = 0.0
            while check == 0.0:
                preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]),
                                       random.choice([0, 1])]
                check = np.dot(np.array(preferred_direction),
                               np.array(preferred_direction))  # Check that the vector is not the zero vector.

        # Other crystal systems have three Miller indices.
        else:
            check = 0.0
            while check == 0.0:
                preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
                check = np.dot(np.array(preferred_direction),
                               np.array(preferred_direction))  # Check that the vector is not the zero vector.

        # Compute the crystallographic texture factor for each peak and scale its intensity.
        for (hkl, peak) in zip(hkls, intensities):
            norm_1 = np.sqrt(np.dot(np.array(hkl), np.array(hkl)))  # Norm of the hkl vector.
            norm_2 = np.sqrt(np.dot(np.array(preferred_direction), np.array(preferred_direction)))
            total_norm = norm_1 * norm_2

            # Compute the texture factor.
            texture_factor = abs(np.dot(np.array(hkl), np.array(preferred_direction)) / total_norm)
            texture_factor = self.map_interval(texture_factor)  # Map the texture factor.

            # Multiply the original peak intensity by the texture factor.
            scaled_intensities.append(peak * texture_factor)

        return scaled_intensities

    def calc_std_dev(self, two_theta, tau):
        """
        Compute the standard deviation from the angle and the domain size.
        :param two_theta: diffraction angle.
        :param tau: domain size.
        :return: standard deviation for the Gaussian kernel.
        """
        # Compute the FWHM via the Scherrer equation.
        K = 0.9  # Shape factor.
        wavelength = self.calculator.wavelength * 0.1
        theta = np.radians(two_theta / 2.)  # Convert the angle to radians for the Bragg equation.
        beta = (K * wavelength) / (np.cos(theta) * tau)

        # Convert the FWHM to a Gaussian standard deviation.
        sigma = np.sqrt(1 / (2 * np.log(2)) * 0.5 * np.degrees(beta))
        return sigma ** 2

    @property
    def textured_spectrum(self):

        angles = self.angles
        intensities = self.texture_intensities

        steps = np.linspace(self.min_angle, self.max_angle, 4501)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            # Map the angle to the closest grid point.
            idx = np.argmin(np.abs(ang - steps))
            signals[i, idx] = intensities[i]

        # Convolute every row with a unique kernel.
        # Iterate over rows; not vectorizable, changing kernel for every row.
        domain_size = 25.0
        step_size = (self.max_angle - self.min_angle) / 4501
        for i in range(signals.shape[0]):
            row = signals[i, :]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # The Gaussian kernel expects a unit step, so rescale std_dev.
            signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size,
                                              mode='constant')

        # Combine the signals.
        signal = np.sum(signals, axis=0)

        # Normalize the signal.
        norm_signal = 100 * signal / max(signal)

        noise = np.random.normal(0, 0.25, 4501)
        noisy_signal = norm_signal + noise

        # Format for the CNN.
        form_signal = [[val] for val in noisy_signal]  # val is each element of noisy_signal.

        return form_signal


def main(struc, num_textured, max_texture=0.6, min_angle=10.0, max_angle=80.0):
    texture_generator = TextureGen(struc, max_texture, min_angle, max_angle)
    textured_patterns = [texture_generator.textured_spectrum for i in range(num_textured)]
    return textured_patterns
