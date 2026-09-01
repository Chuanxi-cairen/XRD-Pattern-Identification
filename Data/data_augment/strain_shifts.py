# Portions of this module are adapted from XRD-AutoAnalyzer
# (https://github.com/njszym/XRD-AutoAnalyzer),
# Copyright (c) 2024 Nathan Szymanski, released under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Lattice
from pyxtal import pyxtal
import numpy as np
import pymatgen as mg
import random


class StainGen(object):  # In Python 3, the explicit (object) base class is optional.
    '''
    This class applies random strain to a crystal structure while preserving its symmetry.
    '''

    def __init__(self, struc, max_strain=0.04, min_angle=10.0, max_angle=80.0):
        '''
        :param struc: pymatgen Structure object.
        :param max_strain: maximum allowed magnitude of the strain tensor components.
        '''
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_strain = max_strain
        self.strain_range = np.linspace(0.0, max_strain, 100)  # Randomly sample strain values within the allowed range.
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def sg(self):
        return self.struc.get_space_group_info()[1]  # Extract the space group number.

    @property
    def conv_struc(self):
        """
        Convert the structure to the conventional standard cell defined by
        Setyawan & Curtarolo (2010); compare it against get_refined_structure()
        to assess their effect on model performance.
        :return: the conventional standard structure.
        """
        sga = mg.symmetry.analyzer.SpacegroupAnalyzer(self.struc)
        return sga.get_conventional_standard_structure()

    @property
    def lattice(self):
        return self.struc.lattice

    @property
    def matrix(self):
        return self.struc.lattice.matrix  # 3x3 lattice matrix describing the lattice parameters.

    @property
    def diag_range(self):
        """
        Generate the diagonal strain values.
        """
        max_strain = self.max_strain
        return np.linspace(1-max_strain, 1+max_strain, 1000)

    @property
    def off_diag_range(self):
        max_strain = self.max_strain
        return np.linspace(0-max_strain, 0+max_strain, 1000)

    @property
    def sg_class(self):
        sg = self.sg  # Space group number.
        if sg in list(range(195, 231)):
            return 'cubic'
        elif sg in list(range(16, 76)):
            return 'orthorhombic'
        elif sg in list(range(3, 16)):
            return 'monoclinic'
        elif sg in list(range(1, 3)):
            return 'triclinic'
        elif sg in list(range(76, 195)):
            if sg in list(range(75, 83)) + list(range(143, 149)) + list(range(168, 175)):
                return 'low-sym hexagonal/tetragonal'
            else:
                return 'high-sym hexagonal/tetragonal'

    @property
    def strain_tensor(self):
        diag_range = self.diag_range
        off_diag_range = self.off_diag_range
        s11, s22, s33 = [random.choice(diag_range) for v in range(3)]
        s12, s13, s21, s23, s31, s32 = [random.choice(off_diag_range) for v in range(6)]
        sg_class = self.sg_class

        if sg_class in ['cubic', 'orthorhombic', 'monoclinic', 'high-sym hexagonal/tetragonal']:
            v1 = [s11, 0, 0]
        elif sg_class == 'low-sym hexagonal/teragonal':
            v1 = [s11, s12, 0]
        elif sg_class == 'triclinic':
            v1 = [s11, s12, s13]

        if sg_class in ['cubic', 'high-sym hexagonal/tetragonal']:
            v2 = [0, s11, 0]
        elif sg_class == 'orthorhombic':
            v2 = [0, s22, 0]
        elif sg_class == 'monoclinic':
            v2 = [0, s22, s23]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v2 = [-s12, s22, 0]
        elif sg_class == 'triclinic':
            v2 = [s21, s22, s23]

        if sg_class == 'cubic':
            v3 = [0, 0, s11]
        elif sg_class == 'high-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'orthorhombic':
            v3 = [0, 0, s33]
        elif sg_class == 'monoclinic':
            v3 = [0, s23, s33]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'triclinic':
            v3 = [s31, s32, s33]

        return np.array([v1, v2, v3])


    @property
    def strained_matrix(self):
        return np.matmul(self.matrix, self.strain_tensor)  # Multiply the lattice matrix by the strain tensor.

    @property
    def strained_lattice(self):
        return Lattice(self.strained_matrix)

    @property
    def strained_struc(self):
        ref_struc = self.struc.copy()
        # Pyxtal cannot handle partial occupancy, but converting the pymatgen
        # object into a pyxtal object makes it easier to apply perturbations.
        if ref_struc.is_ordered:  # Check whether the structure is ordered.
            xtal_struc = pyxtal()
            xtal_struc.from_seed(ref_struc)  # Initialize the pyxtal object from the seed; a disordered crystal would give a wrong object.
            current_strain = random.choice(self.strain_range)  # Randomly pick a strain value.
            # d_lat perturbs the lattice parameters, d_coor perturbs the atomic coordinates.
            xtal_struc.apply_perturbation(d_lat=current_strain, d_coor=0.0)
            pmg_struc = xtal_struc.to_pymatgen()
            return pmg_struc
        else:
            ref_struc.lattice = self.strained_lattice  # If disordered, update the lattice parameters directly.
            return ref_struc


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
        theta = np.radians(two_theta/2.)  # Convert the angle to radians for the Bragg equation.
        beta = (K * wavelength) / (np.cos(theta) * tau)

        # Convert the FWHM to a Gaussian standard deviation.
        sigma = np.sqrt(1/(2*np.log(2))*0.5*np.degrees(beta))
        return sigma**2

    @property
    def strained_spectrum(self):
        struc = self.strained_struc
        pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
        angles, intensities = pattern.x, pattern.y
        steps = np.linspace(self.min_angle, self.max_angle, 4501)
        signals = np.zeros([len(angles), steps.shape[0]])
        for i, ang in enumerate(angles):
            # Map each diffraction angle to the nearest grid point; this makes it
            # easy to build the full pattern by superimposing Gaussian peak shapes.
            idx = np.argmin(np.abs(ang-steps))
            signals[i, idx] = intensities[i]  # Place the intensity at the nearest grid point.

        # Convolve each row with a unique Gaussian kernel to form the peak shape.
        # Iterate row by row.
        domain_size = 25.0
        step_size = (self.max_angle - self.min_angle)/4501
        for i in range(signals.shape[0]):
            row = signals[i, :]
            ang = steps[np.argmax(row)]  # Index of the maximum in the row.
            std_dev = self.calc_std_dev(ang, domain_size)
            # Adjust the standard deviation of the Gaussian kernel.
            # mode='constant' pads the boundary with a constant value (default 0).
            signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size, mode='constant')
        # Sum the rows into a single signal.
        signal = np.sum(signals, axis=0)
        # Normalize the signal.
        norm_signal = 100 * signal / max(signal)
        # np.random.normal(loc, scale, size): loc=mean, scale=std, size=output shape.
        noise = np.random.normal(0, 0.25, 4501)  # Add random noise to simulate background.
        noisy_signal = norm_signal + noise
        # Format for the CNN.
        form_signal = [[val] for val in noisy_signal]  # Wrap each value in a single-element list.
        return form_signal


def main(struc, num_strains, max_strain, min_angle=10.0, max_angle=80.0):
    strain_generator = StainGen(struc, max_strain, min_angle, max_angle)
    strained_patterns = [strain_generator.strained_spectrum for i in range(num_strains)]
    return strained_patterns
