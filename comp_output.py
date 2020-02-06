#! /usr/bin/env python3
import numpy as np

from AaronTools.atoms import Atom
from AaronTools.const import PHYSICAL, UNIT
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import float_vec, uptri2sym


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

    def __init__(self, fname="", get_all=True):
        self.geometry = None
        self.opts = None
        self.frequency = None
        self.archive = None

        self.gradient, self.E_ZPVE, self.ZPVE = (None, None, None)
        self.energy, self.enthalpy = (None, None)
        self.free_energy, self.grimme_g = (None, None)

        self.mass, self.temperature = (None, None)
        self.multiplicity, self.charge = (None, None)

        self.rotational_temperature = None
        self.rotational_symmetry_number = None

        self.error, self.error_msg, self.finished = (None, None, None)

        keys = [
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

        if isinstance(fname, str) and ".log" in fname:
            from_file = FileReader(fname, get_all, just_geom=False)
        elif isinstance(fname, tuple) and "log" == fname[1]:
            from_file = FileReader(fname, get_all, just_geom=False)
        elif isinstance(fname, FileReader):
            from_file = fname
        else:
            return

        self.geometry = Geometry(from_file)
        if "all_geom" in from_file.other:
            self.opts = []
            for g in from_file.other["all_geom"]:
                self.opts += [Geometry(g)]

        for k in keys:
            if k in from_file.other:
                self.__setattr__(k, from_file.other[k])

        if self.frequency:
            self.grimme_g = self.calc_Grimme_G()

    def get_progress(self):
        rv = ""
        grad = self.gradient
        if grad is None:
            rv += "Progress not found"
            return rv

        for name in grad:
            rv += "{:>9}:{}/{:<3} ".format(
                name,
                grad[name]["value"],
                "YES" if grad[name]["converged"] else "NO",
            )

        return rv.rstrip()

    def therm_corr(self, temperature=None, v0=100):
        """returns thermal correction to energy, enthalpy correction to energy, and
        entropy for the specified cutoff frequency and temperature in that order (Hartrees)"""
        if self.frequency is None:
            msg = "Vibrational frequencies not found, "
            msg += "cannot calculate Grimme free energy."
            raise AttributeError(msg)

        rot = [temp for temp in self.rotational_temperature if temp != 0]
        T = temperature if temperature is not None else self.temperature
        mass = self.mass
        sigmar = self.rotational_symmetry_number
        mult = self.multiplicity
        freqs = self.frequency.real_frequencies

        vibtemps = PHYSICAL.SPEED_OF_LIGHT * PHYSICAL.PLANK / PHYSICAL.KB
        vibtemps = [f_i * vibtemps for f_i in freqs]

        Bav = PHYSICAL.PLANK ** 2 / (24 * np.pi ** 2 * PHYSICAL.KB)
        Bav *= sum([1 / r for r in rot])

        # Translational
        qt = 2 * np.pi * mass * PHYSICAL.KB * T / (PHYSICAL.PLANK ** 2)
        qt = qt ** (3 / 2)
        qt *= PHYSICAL.KB * T / PHYSICAL.STANDARD_PRESSURE
        St = PHYSICAL.GAS_CONSTANT * (np.log(qt) + (5 / 2))
        Et = 3 * PHYSICAL.GAS_CONSTANT * T / 2

        # Electronic
        Se = PHYSICAL.GAS_CONSTANT * (np.log(mult))

        # Rotational
        if len(rot) == 3:
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

        Er = len(rot) * PHYSICAL.GAS_CONSTANT * T / 2

        # Vibrational
        Ev = 0
        Sv_qRRHO = 0
        for i, vib in enumerate(vibtemps):
            Sv_T = vib / (T * (np.exp(vib / T) - 1))
            Sv_T -= np.log(1 - np.exp(-vib / T))
            Ev += vib * (1 / 2 + 1 / (np.exp(vib / T) - 1))

            mu = PHYSICAL.PLANK
            mu /= 8 * np.pi ** 2 * freqs[i] * PHYSICAL.SPEED_OF_LIGHT
            mu = mu * Bav / (mu + Bav)
            Sr_eff = 1 / 2 + np.log(
                np.sqrt(
                    8 * np.pi ** 3 * mu * PHYSICAL.KB * T / PHYSICAL.PLANK ** 2
                )
            )
            weight = 1 / (1 + (v0 / freqs[i]) ** 4)

            Sv_qRRHO += weight * Sv_T + (1 - weight) * Sr_eff
        Ev *= PHYSICAL.GAS_CONSTANT
        Sv_qRRHO *= PHYSICAL.GAS_CONSTANT

        Ecorr = (Et + Er + Ev) / (UNIT.HART_TO_KCAL * 1000)
        Hcorr = Ecorr + (
            PHYSICAL.GAS_CONSTANT * T / (UNIT.HART_TO_KCAL * 1000)
        )
        Stot_qRRHO = (St + Sr + Sv_qRRHO + Se) / (UNIT.HART_TO_KCAL * 1000)

        return Ecorr, Hcorr, Stot_qRRHO

    def calc_Grimme_G_corr(self, temperature=None, v0=100):
        """returns quasi rrho free energy correction (Eh)"""
        Ecorr, Hcorr, Stot = self.therm_corr(temperature, v0)
        T = temperature if temperature is not None else self.temperature
        Gcorr_qRRHO = Hcorr - T * Stot

        return Gcorr_qRRHO

    def calc_Grimme_G(self, temperature=None, v0=100):
        """returns quasi rrho free energy (Eh)"""
        Gcorr_qRRHO = self.calc_Grimme_G_corr(temperature=temperature, v0=v0)
        return Gcorr_qRRHO + self.energy

    def bond_change(self, atom1, atom2, threshold=0.25):
        """

        """
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
