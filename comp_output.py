#! /usr/bin/env python3
from collections.abc import MutableSequence

import numpy as np

from AaronTools import addlogger, getlogger
from AaronTools.atoms import Atom
from AaronTools.const import PHYSICAL, UNIT
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import float_vec, uptri2sym


def obj_to_dict(obj, skip_attrs=None):
    # log = getlogger(level="debug")
    if skip_attrs is None:
        skip_attrs = []
    if isinstance(obj, Geometry):
        return obj.comment, [str(a) for a in obj]
    rv = {}
    if hasattr(obj, "__dict__"):
        for attr in obj.__dict__:
            if attr in skip_attrs:
                continue
            val = getattr(obj, attr)
            if isinstance(val, MutableSequence):
                val = [obj_to_dict(v) for v in val]
            else:
                val = obj_to_dict(val)
            rv[str(attr)] = val
        return rv
    return obj


@addlogger
class CompOutput:
    """
    Attributes:
        geometry    the last Geometry
        opts        list of Geometry for each optimization steps
        frequency   Frequency object
        archive     a string containing the archive entry
        energy, enthalpy, free_energy, grimme_g,
        mass, temperature, rotational_temperature,
        multiplicity, charge, rotational_symmetry_number
        error, error_msg, finished,
        gradient, E_ZPVE, ZPVE
    """

    QUASI_HARMONIC = "QHARM"
    QUASI_RRHO = "QRRHO"
    RRHO = "RRHO"
    LOG = None
    LOGLEVEL = "debug"

    def __init__(self, fname="", get_all=True, freq_name=None, conf_name=None):
        self.geometry = None
        self.opts = None
        self.opt_steps = None
        self.frequency = None
        self.archive = None
        self.other = None
        self.conformers = None

        self.gradient, self.E_ZPVE, self.ZPVE = ({}, None, None)
        self.energy, self.enthalpy = (None, None)
        self.free_energy, self.grimme_g = (None, None)

        self.mass, self.temperature = (None, None)
        self.multiplicity, self.charge = (None, None)

        self.rotational_temperature = None
        self.rotational_symmetry_number = None

        self.error, self.error_msg, self.finished = (None, None, None)

        # these will be pulled out of FileReader.other dict
        keys = [
            "opt_steps",
            "energy",
            "error",
            "error_msg",
            "gradient",
            "finished",
            "frequency",
            "mass",
            "temperature",
            "rotational_temperature",
            "free_energy",
            "multiplicity",
            "charge",
            "E_ZPVE",
            "ZPVE",
            "rotational_symmetry_number",
            "enthalpy",
            "archive",
        ]

        if isinstance(fname, (str, tuple)) and len(fname) > 0:
            from_file = FileReader(
                fname,
                get_all,
                just_geom=False,
                freq_name=freq_name,
                conf_name=conf_name,
            )
        elif isinstance(fname, FileReader):
            from_file = fname
        else:
            return

        if from_file.atoms:
            self.geometry = Geometry(
                from_file.atoms, comment=from_file.comment
            )
        if from_file.all_geom:
            self.opts = []
            for g in from_file.all_geom:
                self.opts += [Geometry(g[0])]
        if "conformers" in from_file.other:
            self.conformers = []
            for comment, atoms in from_file.other["conformers"]:
                self.conformers.append(Geometry(atoms, comment=comment))
            del from_file.other["conformers"]

        for k in keys:
            if k in from_file.other:
                setattr(self, k, from_file.other[k])
        
        self.other = {k:v for k, v in from_file.other.items() if k not in keys}

        if self.rotational_temperature is None and self.geometry:
            self.compute_rot_temps()

        if self.frequency:
            self.grimme_g = self.calc_Grimme_G()
            # recalculate ZPVE b/c our constants and the ones in various programs
            # might be slightly different
            self.ZPVE = self.calc_zpe()
            self.E_ZPVE = self.energy + self.ZPVE

    def to_dict(self, skip_attrs=None):
        return obj_to_dict(self, skip_attrs=skip_attrs)

    def get_progress(self):
        rv = ""
        grad = self.gradient
        if not grad:
            rv += "Progress not found"
            return rv

        for name in grad:
            rv += "{:>9}:{}/{:<3} ".format(
                name,
                grad[name]["value"],
                "YES" if grad[name]["converged"] else "NO",
            )

        return rv.rstrip()

    def calc_zpe(self):
        """returns ZPVE correction"""
        hc = PHYSICAL.PLANCK * PHYSICAL.SPEED_OF_LIGHT / UNIT.HART_TO_JOULE
        vib = sum(self.frequency.real_frequencies)
        zpve = 0.5 * hc * vib
        return zpve

    def therm_corr(self, temperature=None, v0=100, method="RRHO"):
        """
        returns thermal correction to energy, enthalpy correction to energy, and entropy
        for the specified cutoff frequency and temperature
        in that order (Hartrees for corrections, Eh/K for entropy)

        temperature: float, temperature in K- None will use self.temperature
        v0: float, cutoff/damping parameter for quasi G corrections
        method: str - type of quasi treatment:
                RRHO  - no quasi treatment
                QRRHO - Grimme's quasi-RRHO
                see Grimme, S. (2012), Supramolecular Binding Thermodynamics by
                Dispersion‐Corrected Density Functional Theory. Chem. Eur. J.,
                18: 9955-9964. (DOI: 10.1002/chem.201200497) for details
                QHARM - Truhlar's quasi-harmonic
                see J. Phys. Chem. B 2011, 115, 49, 14556–14562
                (DOI: 10.1021/jp205508z) for details
        """
        if self.frequency is None:
            msg = "Vibrational frequencies not found, "
            msg += "cannot calculate vibrational entropy."
            raise AttributeError(msg)

        rot = [temp for temp in self.rotational_temperature if temp != 0]
        T = temperature if temperature is not None else self.temperature
        if T == 0:
            return 0, 0, 0

        mass = self.mass
        sigmar = self.rotational_symmetry_number
        if sigmar is None and len(self.geometry.atoms) == 1:
            sigmar = 3
        mult = self.multiplicity
        freqs = np.array(self.frequency.real_frequencies)

        vib_unit_convert = (
            PHYSICAL.SPEED_OF_LIGHT * PHYSICAL.PLANCK / PHYSICAL.KB
        )
        vibtemps = np.array(
            [f_i * vib_unit_convert for f_i in freqs if f_i > 0]
        )
        if method == "QHARM":
            harm_vibtemps = np.array(
                [
                    f_i * vib_unit_convert
                    if f_i > v0
                    else v0 * vib_unit_convert
                    for f_i in freqs
                    if f_i > 0
                ]
            )
        else:
            harm_vibtemps = vibtemps

        Bav = PHYSICAL.PLANCK ** 2 / (24 * np.pi ** 2 * PHYSICAL.KB)
        Bav *= sum([1 / r for r in rot])

        # Translational
        qt = 2 * np.pi * mass * PHYSICAL.KB * T / (PHYSICAL.PLANCK ** 2)
        qt = qt ** (3 / 2)
        qt *= PHYSICAL.KB * T / PHYSICAL.STANDARD_PRESSURE
        St = PHYSICAL.GAS_CONSTANT * (np.log(qt) + (5 / 2))
        Et = 3 * PHYSICAL.GAS_CONSTANT * T / 2

        # Electronic
        Se = PHYSICAL.GAS_CONSTANT * (np.log(mult))

        # Rotational
        if all(r == np.inf for r in rot):
            # atomic
            qr = 1
            Sr = 0
        elif len(rot) == 3:
            # non linear molecules
            qr = np.sqrt(np.pi) / sigmar
            qr *= T ** (3 / 2) / np.sqrt(rot[0] * rot[1] * rot[2])
            Sr = PHYSICAL.GAS_CONSTANT * (np.log(qr) + 3 / 2)
        elif len(rot) == 2:
            # linear molecules
            qr = (1 / sigmar) * (T / np.sqrt(rot[0] * rot[1]))
            Sr = PHYSICAL.GAS_CONSTANT * (np.log(qr) + 1)
        else:
            # atoms
            qr = 1
            Sr = 0

        if all(r == np.inf for r in rot):
            Er = 0
        else:
            Er = len(rot) * PHYSICAL.GAS_CONSTANT * T / 2

        # Vibrational
        if method == self.QUASI_HARMONIC:
            Sv = np.sum(
                harm_vibtemps / (T * (np.exp(harm_vibtemps / T) - 1))
                - np.log(1 - np.exp(-harm_vibtemps / T))
            )
        elif method == self.RRHO:
            Sv = np.sum(
                vibtemps / (T * (np.exp(vibtemps / T) - 1))
                - np.log(1 - np.exp(-vibtemps / T))
            )
        elif method == self.QUASI_RRHO:
            mu = PHYSICAL.PLANCK
            mu /= 8 * np.pi ** 2 * freqs * PHYSICAL.SPEED_OF_LIGHT
            mu = mu * Bav / (mu + Bav)
            Sr_eff = 1 / 2 + np.log(
                np.sqrt(
                    8
                    * np.pi ** 3
                    * mu
                    * PHYSICAL.KB
                    * T
                    / PHYSICAL.PLANCK ** 2
                )
            )

            weights = weight = 1 / (1 + (v0 / freqs) ** 4)

            Sv = np.sum(
                weights
                * (
                    harm_vibtemps / (T * (np.exp(harm_vibtemps / T) - 1))
                    - np.log(1 - np.exp(-harm_vibtemps / T))
                )
                + (1 - weights) * Sr_eff
            )

        Ev = np.sum(vibtemps * (1.0 / 2 + 1 / (np.exp(vibtemps / T) - 1)))

        Ev *= PHYSICAL.GAS_CONSTANT
        Sv *= PHYSICAL.GAS_CONSTANT

        Ecorr = (Et + Er + Ev) / (UNIT.HART_TO_KCAL * 1000)
        Hcorr = Ecorr + (
            PHYSICAL.GAS_CONSTANT * T / (UNIT.HART_TO_KCAL * 1000)
        )
        Stot = (St + Sr + Sv + Se) / (UNIT.HART_TO_KCAL * 1000)

        return Ecorr, Hcorr, Stot

    def calc_G_corr(self, temperature=None, v0=0, method="RRHO"):
        """
        returns quasi rrho free energy correction (Eh)
        temperature: float, temperature; default is self.temperature
        v0: float, parameter for quasi-rrho or quasi-harmonic entropy
        method: str (RRHO, QRRHO, QHARM) method for treating entropy
                see CompOutput.therm_corr for references
        """
        Ecorr, Hcorr, Stot = self.therm_corr(temperature, v0, method)
        T = temperature if temperature is not None else self.temperature
        Gcorr_qRRHO = Hcorr - T * Stot

        return Gcorr_qRRHO

    def calc_Grimme_G(self, temperature=None, v0=100):
        """
        returns quasi rrho free energy (Eh)
        see Grimme, S. (2012), Supramolecular Binding Thermodynamics by
        Dispersion‐Corrected Density Functional Theory. Chem. Eur. J.,
        18: 9955-9964. (DOI: 10.1002/chem.201200497) for details
        """
        Gcorr_qRRHO = self.calc_G_corr(
            temperature=temperature, v0=v0, method=self.QUASI_RRHO
        )
        return Gcorr_qRRHO + self.energy

    def bond_change(self, atom1, atom2, threshold=0.25):
        """"""
        ref = self.opts[0]
        d_ref = ref.atoms[atom1].dist(ref.atoms[atom2])

        n = len(self.opts) - 1
        for i, step in enumerate(self.opts[::-1]):
            d = step.atoms[atom1].dist(step.atoms[atom2])
            if abs(d_ref - d) < threshold:
                n = len(self.opts) - 1 - i
                break
        return n

    def parse_archive(self):
        """
        Reads info from archive string

        Returns: a dictionary with the parsed information
        """

        def grab_coords(line):
            rv = {}
            for i, word in enumerate(line.split("\\")):
                word = word.split(",")
                if i == 0:
                    rv["charge"] = int(word[0])
                    rv["multiplicity"] = int(word[1])
                    rv["atoms"] = []
                    continue
                rv["atoms"] += [
                    Atom(element=word[0], coords=word[1:4], name=str(i))
                ]
            return rv

        rv = {}
        lines = iter(self.archive.split("\\\\"))
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("@"):
                line = line[1:]
                for word in line.split("\\"):
                    if "summary" not in rv:
                        rv["summary"] = [word]
                    elif word not in rv["summary"]:
                        rv["summary"] += [word]
                continue

            if line.startswith("#"):
                if "route" not in rv:
                    rv["route"] = line
                elif isinstance(rv["route"], list):
                    # for compound jobs, like opt freq
                    rv["route"] += [line]
                else:
                    # for compound jobs, like opt freq
                    rv["route"] = [rv["route"]] + [line]

                line = next(lines).strip()
                if "comment" not in line:
                    rv["comment"] = line

                line = next(lines).strip()
                for key, val in grab_coords(line).items():
                    rv[key] = val
                continue

            words = iter(line.split("\\"))
            for word in words:
                if not word:
                    # get rid of pesky empty elements
                    continue
                if "=" in word:
                    key, val = word.split("=")
                    rv[key.lower()] = float_vec(val)
                else:
                    if "hessian" not in rv:
                        rv["hessian"] = uptri2sym(
                            float_vec(word),
                            3 * len(rv["atoms"]),
                            col_based=True,
                        )
                    else:
                        rv["gradient"] = float_vec(word)
        return rv

    def follow(self, reverse=False, step=0.1):
        """
        Follow imaginary mode
        """
        # get geometry and frequency objects
        geom = self.geometry.copy()
        freq = self.frequency

        # make sure geom is a TS and has computed frequencies available
        if freq is None:
            raise AttributeError("Frequencies for this geometry not found.")
        if not freq.is_TS:
            raise RuntimeError("Geometry not a transition state")

        # get displacement vectors for imaginary frequency
        img_mode = freq.imaginary_frequencies[0]
        vector = freq.by_frequency[img_mode]["vector"]

        # apply transformation to geometry and return it
        for i, a in enumerate(geom.atoms):
            if reverse:
                a.coords -= vector[i] * step
            else:
                a.coords += vector[i] * step
        return geom

    def compute_rot_temps(self):
        """sets self's 'rotational_temperature' attribute by using self.geometry
        not recommended b/c atoms should be specific isotopes, but this uses
        average atomic weights
        exists because older versions of ORCA don't print rotational temperatures"""
        COM = self.geometry.COM(mass_weight=True)
        self.geometry.coord_shift(-COM)
        inertia_mat = np.zeros((3, 3))
        for atom in self.geometry.atoms:
            for i in range(0, 3):
                for j in range(0, 3):
                    if i == j:
                        inertia_mat[i][j] += sum(
                            [
                                atom.mass() * atom.coords[k] ** 2
                                for k in range(0, 3)
                                if k != i
                            ]
                        )
                    else:
                        inertia_mat[i][j] -= (
                            atom.mass() * atom.coords[i] * atom.coords[j]
                        )

        principal_inertia, vecs = np.linalg.eigh(inertia_mat)

        principal_inertia *= UNIT.AMU_TO_KG * 1e-20

        # rotational constants in Hz
        rot_consts = [
            PHYSICAL.PLANCK / (8 * np.pi ** 2 * moment)
            for moment in principal_inertia
            if moment > 0
        ]

        self.rotational_temperature = [
            PHYSICAL.PLANCK * const / PHYSICAL.KB for const in rot_consts
        ]
        # shift geometry back
        self.geometry.coord_shift(COM)
