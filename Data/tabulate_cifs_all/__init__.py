# Portions of this module are adapted from XRD-AutoAnalyzer
# (https://github.com/njszym/XRD-AutoAnalyzer),
# Copyright (c) 2024 Nathan Szymanski, released under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from pymatgen.core import Structure, Composition, PeriodicSite
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.signal import find_peaks, filtfilt, resample
from itertools import combinations_with_replacement
from scipy.ndimage import gaussian_filter1d
from pymatgen.io.cif import CifParser
from collections import defaultdict
from functools import lru_cache
from itertools import product
from tqdm import tqdm
from functools import reduce
from shutil import copytree
import numpy as np
import math
import time
import os
import re


# Possible oxidation states of each element, used to verify charge neutrality.
common_oxi = {
    'H': [1, -1],  # Hydrogen
    'He': [0],  # Helium
    'Li': [1],  # Lithium
    'Be': [2],  # Beryllium
    'B': [3],  # Boron
    'C': [-4, 4],  # Carbon
    'N': [-3],  # Nitrogen
    'O': [-2],  # Oxygen
    'F': [-1],  # Fluorine
    'Ne': [0],  # Neon
    'Na': [1],  # Sodium
    'Mg': [2],  # Magnesium
    'Al': [3],  # Aluminum
    'Si': [-4, 4],  # Silicon
    'P': [-3, 3, 5],  # Phosphorus
    'S': [-2, 4, 6],  # Sulfur
    'Cl': [-1],  # Chlorine
    'Ar': [0],  # Argon
    'K': [1],  # Potassium
    'Ca': [2],  # Calcium
    'Sc': [3],  # Scandium
    'Ti': [2, 3, 4],  # Titanium
    'V': [2, 3, 4, 5],  # Vanadium
    'Cr': [2, 3, 6],  # Chromium
    'Mn': [2, 3, 4, 7],  # Manganese
    'Fe': [2, 3],  # Iron
    'Co': [2, 3],  # Cobalt
    'Ni': [2],  # Nickel
    'Cu': [1, 2],  # Copper
    'Zn': [2],  # Zinc
    'Ga': [3],  # Gallium
    'Ge': [2, 4],  # Germanium
    'As': [-3, 3, 5],  # Arsenic
    'Se': [-2, 4, 6],  # Selenium
    'Br': [-1],  # Bromine
    'Kr': [0],  # Krypton
    'Rb': [1],  # Rubidium
    'Sr': [2],  # Strontium
    'Y': [3],  # Yttrium
    'Zr': [4],  # Zirconium
    'Nb': [3, 5],  # Niobium
    'Mo': [2, 3, 4, 5, 6],  # Molybdenum
    'Tc': [7],  # Technetium
    'Ru': [2, 3, 4, 6, 8],  # Ruthenium
    'Rh': [2, 3, 4],  # Rhodium
    'Pd': [2, 4],  # Palladium
    'Ag': [1],  # Silver
    'Cd': [2],  # Cadmium
    'In': [3],  # Indium
    'Sn': [2, 4],  # Tin
    'Sb': [-3, 3, 5],  # Antimony
    'Te': [-2, 4, 6],  # Tellurium
    'I': [-1],  # Iodine
    'Xe': [0],  # Xenon
    'Cs': [1],  # Cesium
    'Ba': [2],  # Barium
    'La': [3],  # Lanthanum
    'Ce': [3, 4],  # Cerium
    'Pr': [3],  # Praseodymium
    'Nd': [3],  # Neodymium
    'Pm': [3],  # Promethium
    'Sm': [2, 3],  # Samarium
    'Eu': [2, 3],  # Europium
    'Gd': [3],  # Gadolinium
    'Tb': [3, 4],  # Terbium
    'Dy': [3],  # Dysprosium
    'Ho': [3],  # Holmium
    'Er': [3],  # Erbium
    'Tm': [2, 3],  # Thulium
    'Yb': [2, 3],  # Ytterbium
    'Lu': [3],  # Lutetium
    'Hf': [4],  # Hafnium
    'Ta': [5],  # Tantalum
    'W': [2, 3, 4, 5, 6],  # Tungsten
    'Re': [2, 3, 4, 6, 7],  # Rhenium
    'Os': [2, 3, 4, 6, 8],  # Osmium
    'Ir': [2, 3, 4, 6],  # Iridium
    'Pt': [2, 4],  # Platinum
    'Au': [1, 3],  # Gold
    'Hg': [1, 2],  # Mercury
    'Tl': [1, 3],  # Thallium
    'Pb': [2, 4],  # Lead
    'Bi': [3, 5],  # Bismuth
    'Th': [4],  # Thorium
    'Pa': [5],  # Protactinium
    'U': [3, 4, 5, 6],  # Uranium
    'Np': [3, 4, 5, 6, 7],  # Neptunium
    'Pu': [3, 4, 5, 6, 7, 8],  # Plutonium
    'Am': [2, 3, 4, 5, 6],  # Americium
    'Cm': [3],  # Curium
    'Bk': [3, 4],  # Berkelium
    'Cf': [2, 3, 4],  # Californium
    'Es': [3],  # Einsteinium
    'Fm': [3],  # Fermium
    'Md': [2, 3],  # Mendelevium
    'No': [2, 3],  # Nobelium
    'Lr': [3],  # Lawrencium
    'Rf': [4],  # Rutherfordium
    'Db': [5],  # Dubnium
    'Sg': [6],  # Seaborgium
    'Bh': [7],  # Bohrium
    'Hs': [8],  # Hassium
}


def round_dict_values(data):
    """
    Used to round off coefficients of highly complex formulae.
    """

    for key, value in data.items():
        if value > 1e5:
            data[key] = round(value, -3)  # Round to three places before the decimal point (e.g. 123456 -> 123000).
        elif value > 1e4:
            data[key] = round(value, -2)
        elif value > 1e3:
            data[key] = round(value, -1)

    # Reduce the coefficients by their gcd.
    gcd = reduce(math.gcd, list(data.values()))  # reduce() applies a function cumulatively over a sequence.
    for key in data:
        data[key] = int(data[key] / gcd)

    return data


def parse_formula(formula):  # Parse the chemical formula.

    # Convert to alphabetical order (no parentheses).
    c = Composition(formula)  # Composition already removes parentheses.
    formula = c.alphabetical_formula.replace(' ', '')  # Remove the spaces in the formula.

    # element_pattern and compound_pattern are regexes:
    # [A-Z] matches an uppercase letter (the first letter of an element symbol),
    # [a-z]* matches zero or more lowercase letters, and \d* matches digits.
    # ([A-Z][a-z]*\d*) matches the parenthesized part, and (\d*) matches the
    # multiplier after an element or a parenthesis; \( and \) match parentheses.
    element_pattern = r'([A-Z][a-z]*)(\d*)'  # r marks a raw string.
    compound_pattern = r'\(([A-Z][a-z]*\d*)\)(\d*)'  # Matches strings like (Compound)Multiplier.

    # Expand parenthesized compounds.
    while '(' in formula:
        match = re.search(compound_pattern, formula)
        compound, multiplier = match.groups()  # e.g. for (OH)3, compound='OH', multiplier='3'.
        expanded = ''.join(f"{element}{int(count) * int(multiplier)}" for element, count in
                           re.findall(element_pattern, compound))  # ''.join() concatenates all the generated strings.
        formula = formula.replace(match.group(), expanded)

    # Parse elements and their counts
    parsed = re.findall(element_pattern, formula)
    counts = {element: int(count) if count else 1 for element, count in parsed}  # The ':' separates keys and values.

    multi_oxi = False
    for elem in counts.keys():
        if len(common_oxi[elem]) > 1:
            multi_oxi = True

    """
    If coefficients of the chemical formula are unreasonably large, reduce them by rounding.
    However, only do this in the case of multiple oxidation states per element.
    Without rounding, these can lead to combinatorial explosion.
    """
    if multi_oxi:
        counts = round_dict_values(counts)

    return counts


def balance_oxidation_states(formula, oxidation_states, max_time=10):
    """
    Note: this is *not* an exhaustive oxidation state solver.
    Rather, it will find if there exists at least one solution
    that satisfies charge balance given the possible oxidation
    states. This method is fast and suitable for the current
    application; however, caution should be used if one
    implements it outside of XRD-AutoAnalyzer.
    """
    element_counts = parse_formula(formula)
    elements = list(element_counts.keys())

    balanced_combinations = []

    for el in elements:
        if len(oxidation_states[el]) > 1:
            multi_valent_element = el
            multi_valent_count = element_counts[el]
            possible_states = oxidation_states[el]
            start_time = time.time()  # Start the timer.
            # Generate all combinations of oxidation states with repetition; the second argument must be an integer.
            for combination in combinations_with_replacement(possible_states, multi_valent_count):
                current_time = time.time()  # Record the current time.
                sum_states = sum(
                    [oxidation_states[el][0] * element_counts[el] for el in elements if el != multi_valent_element])
                sum_states += sum(combination)  # Add the charge of the non-multivalent elements to the combination.
                if current_time - start_time > max_time:
                    break
                if sum_states == 0:
                    unique_combination = tuple(set(combination))
                    if len(unique_combination) == 1:
                        unique_combination = unique_combination[0]
                    balanced_combinations.append(
                        {**{el: oxidation_states[el][0] for el in elements if el != multi_valent_element},
                         multi_valent_element: unique_combination})  # ** unpacks the dictionary.
                    break

    if not balanced_combinations:  # no multivalent elements or no solution found yet
        # * passes each list element as a separate argument to product(), which generates all combinations.
        all_state_combinations = product(*[oxidation_states[el] for el in elements])

        for state_combination in all_state_combinations:
            if sum(element_counts[el] * state for el, state in zip(elements, state_combination)) == 0:
                balanced_combination = dict(zip(elements, state_combination))
                if balanced_combination not in balanced_combinations:
                    balanced_combinations.append(balanced_combination)

    return balanced_combinations


class StructureFilter(object):
    """
    Class used to parse a list of CIFs and choose unique,
    stoichiometric reference phases that were measured
    under (or nearest to) ambient conditions.
    """

    def __init__(self, cif_directory, enforce_order):
        """
        Args:
            cif_directory: path to directory containing
                the CIF files to be considered as
                possible reference phases
        """

        self.cif_dir = cif_directory
        self.enforce_order = enforce_order
        

    @property
    def stoichiometric_info(self):
        """
        Filter strucures to include only those which do not have
        fraction occupancies and are ordered. For those phases, tabulate
        the measurement conditions of the associated CIFs.

        Returns:
            stoich_strucs: a list of ordered pymatgen Structure objects
            temps: temperatures that each were measured at
            dates: dates the measurements were reported
        """

        strucs = []
        grouped = defaultdict(list)
        for cmpd in os.listdir(self.cif_dir):
            # Allowing some tolerance in site occupancies
            cif_file_path = os.path.join(self.cif_dir, cmpd)
            try:
                parser = CifParser(cif_file_path, occupancy_tolerance=1.5)  # Join the directory and filename into a full path.
            except ValueError as e:
                print(f"{cmpd} parse error -- {e}")
            try:
                struc = parser.parse_structures(primitive=True)[0]
            except ValueError as e:
                print(f"{cmpd} has a structure problem: {e}")
                
            if self.enforce_order:  # Check whether ordered structures are required.
                # If so, disordered structures are removed.
                if struc.is_ordered:
                    strucs.append(struc)
            else:
                strucs.append(struc)
                
            formula = struc.composition.reduced_formula
            try:
                sg = struc.get_space_group_info()[1]  # Space group number.
            except:
                sg = None
                print(f"Can't calculate{self.cif_dir}'s space_group ")
                
            # Group by (formula, space group).
            key = (formula, sg)
            grouped[key].append(cmpd)

        return strucs, grouped


def oxi_filter(cif_dir):
    """
    Removes any reference compounds that have
    unusual oxidation states.

    Args:
        cif_dir: directory containing CIFs
    """

    for filename in os.listdir(cif_dir):

        oxi_okay = False

        struc = Structure.from_file('%s/%s' % (cif_dir, filename))
        formula = struc.composition.get_integer_formula_and_factor()[0]

        oxi_guesses = balance_oxidation_states(formula, common_oxi)

        if len(oxi_guesses) > 0:

            check_list = []
            for oxi_dict in oxi_guesses:
                plausible = True
                for elem in oxi_dict.keys():
                    if type(oxi_dict[elem]) is tuple:
                        for oxi_state in oxi_dict[elem]:
                            if int(oxi_state) not in common_oxi[elem]:
                                plausible = False
                    else:
                        if int(oxi_dict[elem]) not in common_oxi[elem]:
                            plausible = False
                check_list.append(plausible)

            if True in check_list:
                oxi_okay = True

        if not oxi_okay:
            os.remove('%s/%s' % (cif_dir, filename))


def write_cifs(unique_strucs, dir, include_elems):
    """
    Write structures to CIF files

    Args:
        strucs: list of pymatgen Structure objects
        dir: path to directory where CIF files will be written
    """

    if not os.path.isdir(dir):
        os.mkdir(dir)

    for struc in tqdm(unique_strucs, desc="Writing CIF files"):
        num_elems = len(struc.composition.elements)
        if num_elems == 1:
            if not include_elems:
                continue  # Skip the rest of this iteration.
        f = struc.composition.reduced_formula
        try:
            sg = struc.get_space_group_info()[1]
            filepath = '%s/%s_%s.cif' % (dir, f, sg)  # Rename the CIF to "formula_spacegroup.cif".
            struc.to(filename=filepath, fmt='cif')
        except:
            try:
                print('%s Space group cannot be determined, lowering tolerance' % str(f))
                sg = struc.get_space_group_info(symprec=0.1, angle_tolerance=5.0)[1]  # Increase the tolerance.
                filepath = '%s/%s_%s.cif' % (dir, f, sg)
                struc.to(filename=filepath, fmt='cif')  # fmt is the output format (cif).
            except:
                print('%s Space group cannot be determined even after lowering tolerance, Setting to None' % str(f))

    # assert raises an error if the condition is false.
    assert len(os.listdir(dir)) > 0, 'Something went wrong. No reference phases were found.'


def main(cif_directory, ref_directory, filter_oxi=False, include_elems=True, enforce_order=False):
    if filter_oxi:
        copytree(cif_directory, 'Filtered_CIFs')  # Copy cif_directory into a new directory Filtered_CIFs (created if missing).
        oxi_filter('Filtered_CIFs')
        cif_directory = 'Filtered_CIFs'

    # Get unique structures
    struc_filter = StructureFilter(cif_directory, enforce_order)
    final_refs, same_name = struc_filter.stoichiometric_info
    
    # Collect files with the same name into a TXT file.
    output_file = "same_name_cif.txt"
    with open(output_file, 'w') as f:
        for key in same_name:
            if len(same_name[key]) > 1:
                f.write(f"compound_name&sg: {key}\n")
                f.write("cif_names:\n")
                for member in same_name[key]:
                    f.write(f"    -{member}\n")
                f.write("\n")

    # Write unique structures (as CIFs) to reference directory
    write_cifs(final_refs, ref_directory, include_elems)
