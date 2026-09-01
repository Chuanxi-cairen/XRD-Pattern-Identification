import data_augment
import numpy as np

if __name__ == '__main__':  # This block runs only when the script is executed directly.

    max_texture = 0.5  # default: texture associated with up to +/- 50% changes in peak intensities
    min_domain_size, max_domain_size = 5.0, 30.0  # default: domain sizes ranging from 5 to 30 nm
    max_strain = 0.03  # default: up to +/- 3% strain
    max_shift = 0.5  # default: up to +/- 0.5 degrees shift in two-theta
    impur_amt = 70.0  # Max amount of impurity phases to include (%)
    num_spectra = 50  # Number of spectra to simulate per phase
    separate = True  # If False: apply all artifacts simultaneously
    min_angle, max_angle = 10.0, 80.0
    batch_size = 100
    skip_filter = False
    include_elems = True
    enforce_order = False

    oxi_filter = False

    # Simulate and save augmented XRD spectra
    xrd_obj = data_augment.SpectraGenerator('./icsd_perov_all', 'aug_icsd_all', num_spectra, max_texture, min_domain_size,
                                                   max_domain_size, max_strain, max_shift, impur_amt, min_angle,
                                                   max_angle, batch_size, separate)
    xrd_specs = xrd_obj.generate_and_save()