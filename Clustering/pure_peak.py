from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import random
import numpy as np

class BroadGen(object):
    """
    This class simulates peak broadening caused by domain size.
    """

    def __init__(self, struc, domain_size=25, min_angle=10.0, max_angle=80.0):
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.domain_size = domain_size
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    @property
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

    def calc_std_dev(self, two_theta, tau):
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
        beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2

    def broadened_spectrum(self, dimension=4501):

        angles = self.angles
        intensities = self.intensities
        steps = np.linspace(self.min_angle, self.max_angle, dimension)
        signals = np.zeros([len(angles), steps.shape[0]])

        # Not used.
        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang - steps))
            signals[i, idx] = intensities[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = self.domain_size
        step_size = (self.max_angle - self.min_angle) / dimension
        for i in range(signals.shape[0]):
            row = signals[i, :]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size,
                                                  mode='constant')

        # Combine signals
        signal = np.sum(signals, axis=0)

        # Normalize signal
        norm_signal = 100 * signal / max(signal)

        # Formatted for CNN
        form_signal = [[val] for val in norm_signal]

        return form_signal


def main(struc, num_broadened, domain_size, min_angle=10.0, max_angle=80.0, dimension=4501):

    broad_generator = BroadGen(struc, domain_size, min_angle, max_angle)

    broadened_patterns = [broad_generator.broadened_spectrum(dimension=dimension) for i in range(num_broadened)]

    return broadened_patterns
