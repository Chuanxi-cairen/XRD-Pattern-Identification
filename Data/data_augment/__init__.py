# Portions of this module are adapted from XRD-AutoAnalyzer
# (https://github.com/njszym/XRD-AutoAnalyzer),
# Copyright (c) 2024 Nathan Szymanski, released under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from data_augment import strain_shifts, uniform_shifts, intensity_changes, peak_broadening, impurity_peaks, mixed
from pymatgen.core import Structure
import multiprocessing
from functools import partial
import numpy as np
import os


def _process_single_file(args, generator):
    """
    Process a single file: read the structure and augment it.
    Made a static function to avoid serializing the whole self during multiprocessing.
    """
    fpath, out_dir = args
    try:
        # Check whether this file has already been processed.
        filename = os.path.basename(fpath)
        basename = os.path.splitext(filename)[0]
        aug_file_path = os.path.join(out_dir, f"{basename}_aug.npy")

        if os.path.exists(aug_file_path):
            # If it exists, skip it and return the size of the existing data.
            existing_data = np.load(aug_file_path, mmap_mode='r')
            return len(existing_data)

        # Read the structure file.
        struc = Structure.from_file(fpath, occupancy_tolerance=1.1)

        if generator.separate:
            patterns = []
            patterns.extend(
                strain_shifts.main(struc, generator.num_spectra, generator.max_strain, 
                                 generator.min_angle, generator.max_angle))
            patterns.extend(
                uniform_shifts.main(struc, generator.num_spectra, generator.max_shift, 
                                   generator.min_angle, generator.max_angle))
            patterns.extend(
                peak_broadening.main(struc, generator.num_spectra, generator.min_domain_size, 
                                    generator.max_domain_size, generator.min_angle, 
                                    generator.max_angle))
            patterns.extend(
                intensity_changes.main(struc, generator.num_spectra, generator.max_texture, 
                                      generator.min_angle, generator.max_angle))
            patterns.extend(
                impurity_peaks.main(struc, generator.num_spectra, generator.impur_amt, 
                                   generator.min_angle, generator.max_angle))
        else:
            patterns = mixed.main(struc, 5 * generator.num_spectra, generator.max_shift, 
                                 generator.max_strain, generator.min_domain_size, 
                                 generator.max_domain_size, generator.max_texture,
                                 generator.impur_amt, generator.min_angle, generator.max_angle)

        # Save the results.
        np.save(aug_file_path, patterns)
        return len(patterns)


    except Exception as e:
        print(f"[Error] Failed to process {fpath}: {e}")
        return 0


class SpectraGenerator(object):
    """
    This class generates augmented patterns.
    """

    def __init__(self, reference_dir, out_dir, num_spectra=50, max_texture=0.6, min_domain_size=1.0, max_domain_size=100.0,
                 max_strain=0.04, max_shift=0.2, impur_amt=70.0, min_angle=10.0, max_angle=80.0, batch_size=50, separate=True):
                 
        self.ref_dir = reference_dir
        self.out_dir = out_dir
        self.num_spectra = num_spectra
        self.max_texture = max_texture
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_strain = max_strain
        self.max_shift = max_shift
        self.impur_amt = impur_amt
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.separate = separate
        self.batch_size = batch_size
        
        # Create the output directory.
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_and_save(self):
        """
        Process with multiprocessing while avoiding complex object passing.
        """
        # Collect all files to process.
        filenames = sorted([f for f in os.listdir(self.ref_dir) if f.endswith('.cif')])
        filepaths = [os.path.join(self.ref_dir, fname) for fname in filenames]
        
        print(f"Found {len(filepaths)} CIF files to process")
        
        # Prepare the argument list.
        args_list = [(fpath, self.out_dir) for fpath in filepaths]
        
        total_spectra = 0
        total_batches = (len(filepaths) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(filepaths))
            batch_args = args_list[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} "
                  f"({len(batch_args)} files)")
            
            # Create the process pool.
            with multiprocessing.Pool(processes=24) as pool:
                # Use partial to fix the generator argument.
                worker_func = partial(_process_single_file, generator=self)
                results = []
                
                # Use imap to avoid memory issues.
                for result in pool.imap(worker_func, batch_args):
                    results.append(result)
            
            batch_total = sum(results)
            total_spectra += batch_total
            print(f"Batch {batch_idx + 1} completed: {batch_total} spectra generated")
            
            # Free memory.
            del results
            
        print(f"\nCompleted! Generated {total_spectra} augmented spectra in total.")
        return total_spectra
