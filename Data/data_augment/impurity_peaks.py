# Portions of this module are adapted from XRD-AutoAnalyzer
# (https://github.com/njszym/XRD-AutoAnalyzer),
# Copyright (c) 2024 Nathan Szymanski, released under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from pymatgen.core import Structure
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d  # Gaussian smoothing of 1-D arrays.
import numpy as np
import random
import os

class ImpurGen(object):
    """
    This class adds impurity (background) peaks to the pattern.
    """
    def __init__(self, struc, impur_amt, ref_dir='./Unique_Perovskite', min_angle=10.0, max_angle=80.0):
        """
        :param struc: Structure used to simulate the XRD pattern.
        :param impur_amt: maximum impurity magnitude relative to the main peak.
        :param ref_dir: directory of reference structures.
        :param min_angle: minimum diffraction angle of the XRD pattern.
        :param max_angle: maximum diffraction angle of the XRD pattern.
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.impur_amt = impur_amt
        self.ref_dir = ref_dir
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

        # Generate a clean single-phase pattern for each reference phase.
        self.saved_patterns = self.clean_spaces

    @property
    def clean_spaces(self):

        # Iterate over all reference structures.
        ref_patterns = []
        for struc in self.ref_strucs:

            pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
            angles = pattern.x
            intensities = pattern.y

            steps = np.linspace(self.min_angle, self.max_angle, 4501)
            signals = np.zeros([len(angles), steps.shape[0]])  # One row per angle, one column per grid point.

            for i, ang in enumerate(angles):  # i is the index, ang the angle value.
                # Map the angle to the closest grid point.
                idx = np.argmin(np.abs(
                    ang - steps))  # Index of the grid point closest to ang.
                signals[i, idx] = intensities[i]
                # This standardizes the pattern to a fixed length (4501) by mapping each peak onto the grid.

            # Convolute every row with a unique kernel.
            # Iterate over rows; not vectorizable, changing kernel for every row.
            domain_size = 25.0
            step_size = (self.max_angle - self.min_angle) / 4501  # Grid step size.
            for i in range(signals.shape[0]):  # Iterate over the rows of signals.
                row = signals[i, :]  # Select all elements of row i.
                ang = steps[np.argmax(row)]  # Diffraction angle at the maximum of the row.
                std_dev = self.calc_std_dev(ang, domain_size)  # Std of the Gaussian kernel from angle and domain size.
                # The Gaussian kernel expects a unit step, so rescale std_dev.
                signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size, mode='constant')
                # gaussian_filter1d smooths a 1-D signal by convolving it with a Gaussian kernel.

            # Sum the signals.
            signal = np.sum(signals, axis=0)  # Sum over rows (axis=0).

            # Normalize the signal.
            norm_signal = 100 * signal / max(signal)

            ref_patterns.append(norm_signal)  # Append the normalized pattern.

        return ref_patterns

    @property
    def ref_strucs(self):
        current_lat = self.struc.lattice.abc
        all_strucs = []
        for fname in os.listdir(self.ref_dir):
            fpath = os.path.join(self.ref_dir, fname)  # Join the directory and the filename.
            struc = Structure.from_file(fpath, occupancy_tolerance=1.1)

            # Exclude duplicate structures.
            if False in np.isclose(struc.lattice.abc, current_lat, atol=0.01):  # atol is the absolute tolerance.
                all_strucs.append(struc)
        return all_strucs

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    @property
    def impurity_spectrum(self):
        signal = random.choice(self.saved_patterns)
        return signal

    def calc_std_dev(self, two_theta, tau):  # Compute the Gaussian std via the Scherrer equation.
        """
        calculate standard deviation based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm
        Returns:
            standard deviation for gaussian kernel
        """
        ## Calculate FWHM based on the Scherrer equation
        K = 0.9 ## shape factor
        wavelength = self.calculator.wavelength * 0.1 ## angstrom to nm
        theta = np.radians(two_theta/2.) ## Bragg angle in radians
        beta = (K * wavelength) / (np.cos(theta) * tau) # FWHM, in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2

    @property
    def spectrum(self):

        angles = self.angles
        intensities = self.intensities

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
        signal = 100 * signal / max(signal)

        # Add the impurity peak.
        impurity_signal = self.impurity_spectrum
        impurity_magnitude = random.choice(np.linspace(0, self.impur_amt, 100))  # Impurity peak magnitude.
        impurity_signal = impurity_magnitude * impurity_signal / max(impurity_signal)  # Scale the impurity signal.
        signal += impurity_signal

        # Renormalize the signal.
        norm_signal = 100 * signal / max(signal)

        noise = np.random.normal(0, 0.25, 4501)
        noisy_signal = norm_signal + noise

        # Format for the CNN.
        form_signal = [[val] for val in noisy_signal]

        return form_signal


# impur_amt is the ratio of the impurity peak to the maximum peak height.
def main(struc, num_impure, impur_amt=70.0, min_angle=10.0, max_angle=80.0, ref_dir='./Unique_Perovskite'):

    impurity_generator = ImpurGen(struc, impur_amt, ref_dir, min_angle, max_angle)

    impure_patterns = [impurity_generator.spectrum for i in range(num_impure)]

    return impure_patterns
