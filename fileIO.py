"""For parsing input/output files"""
import os
import re
from copy import deepcopy
from io import IOBase, StringIO

import numpy as np

from AaronTools import addlogger
from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS, PHYSICAL, UNIT
from AaronTools.theory import *

read_types = [
    "xyz",
    "log",
    "com",
    "gjf",
    "sd",
    "sdf",
    "mol",
    "mol2",
    "out",
    "dat",
    "fchk",
    "crest",
    "xtb",
    "sqmout",
]
write_types = ["xyz", "com", "inp", "in", "sqmin"]
file_type_err = "File type not yet implemented: {}"
float_num = re.compile("[-+]?\d+\.?\d*")
NORM_FINISH = "Normal termination"
ORCA_NORM_FINISH = "****ORCA TERMINATED NORMALLY****"
PSI4_NORM_FINISH = "*** Psi4 exiting successfully. Buy a developer a beer!"
ERROR = {
    "Convergence failure -- run terminated.": "SCF_CONV",
    "Inaccurate quadrature in CalDSu": "CONV_CDS",
    "Error termination request processed by link 9999": "CONV_LINK",
    "FormBX had a problem": "FBX",
    "NtrErr Called from FileIO": "CHK",
    "Wrong number of Negative eigenvalues": "EIGEN",
    "Erroneous write": "QUOTA",
    "Atoms too close": "CLASH",
    "The combination of multiplicity": "CHARGEMULT",
    "Bend failed for angle": "REDUND",
    "Linear angle in Bend": "REDUND",
    "Error in internal coordinate system": "COORD",
    "galloc: could not allocate memory": "GALLOC",
    "Error imposing constraints": "CONSTR",
    "End of file reading basis center.": "BASIS_READ",
    "Atomic number out of range for .* basis set.": "BASIS",
    "Unrecognized atomic symbol": "ATOM",
    "malloc failed.": "MEM",
    "A syntax error was detected in the input line": "SYNTAX",
    "Unknown message": "UNKNOWN",
}

ERROR_ORCA = {
    "SCF NOT CONVERGED AFTER": "SCF_CONV",
    # ORCA doesn't actually exit if the SCF doesn't converge...
    # "CONV_CDS": "",
    "The optimization did not converge but reached the maximum number": "OPT_CONV",
    # ORCA still prints the normal finish line if opt doesn't converge...
    # "FBX": "",
    # "CHK": "",
    # "EIGEN": "", <- ORCA doesn't seem to have this
    # "QUOTA": "",
    "Zero distance between atoms": "CLASH",  # <- only get an error if atoms are literally on top of each other
    "Error : multiplicity": "CHARGEMULT",
    # "REDUND": "",
    # "REDUND": "",
    # "GALLOC": "",
    # "CONSTR": "",
    "The basis set was either not assigned or not available for this element": "BASIS",
    "Element name/number, dummy atom or point charge expected": "ATOM",
    "Error  (ORCA_SCF): Not enough memory available!": "MEM",
    "WARNING: Analytical MP2 frequency calculations": "NUMFREQ",
    "WARNING: Analytical Hessians are not yet implemented for meta-GGA functionals": "NUMFREQ",
    "ORCA finished with error return": "UNKNOWN",
    "UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE": "TYPO",
}

# some exceptions are listed in https://psicode.org/psi4manual/master/_modules/psi4/driver/p4util/exceptions.html
ERROR_PSI4 = {
    "PsiException: Could not converge SCF iterations": "SCF_CONV",
    "psi4.driver.p4util.exceptions.SCFConvergenceError: Could not converge SCF iterations": "SCF_CONV",
    "OptimizationConvergenceError": "OPT_CONV",
    "TDSCFConvergenceError": "TDCF_CONV",
    "The INTCO_EXCEPTion handler": "INT_COORD",
    # ^ this is basically psi4's FBX
    # "CONV_CDS": "",
    # "CONV_LINK": "",
    # "FBX": "",
    # "CHK": "",
    # "EIGEN": "", <- psi4 doesn't seem to have this
    # "QUOTA": "",
    # "ValidationError:": "INPUT", <- generic input error, CHARGEMULT and CLASH would also get caught by this
    "qcelemental.exceptions.ValidationError: Following atoms are too close:": "CLASH",
    "qcelemental.exceptions.ValidationError: Inconsistent or unspecified chg/mult": "CHARGEMULT",
    "MissingMethodError": "INVALID_METHOD",
    # "REDUND": "",
    # "REDUND": "",
    # "GALLOC": "",
    # "CONSTR": "",
    "psi4.driver.qcdb.exceptions.BasisSetNotFound: BasisSet::construct: Unable to find a basis set for": "BASIS",
    "qcelemental.exceptions.NotAnElementError": "ATOM",
    "psi4.driver.p4util.exceptions.ValidationError: set_memory()": "MEM",
    # ERROR_PSI4[""] = "UNKNOWN",
}


def is_alpha(test):
    rv = re.search("^[a-zA-Z]+$", test)
    return bool(rv)


def is_int(test):
    rv = re.search("^[+-]?\d+$", test)
    return bool(rv)


def is_num(test):
    rv = re.search("^[+-]?\d+\.?\d*", test)
    return bool(rv)


def step2str(step):
    if int(step) == step:
        return str(int(step))
    else:
        return str(step).replace(".", "-")


def str2step(step_str):
    if "-" in step_str:
        return float(step_str.replace("-", "."))
    else:
        return float(step_str)


class FileWriter:
    @classmethod
    def write_file(
        cls, geom, style=None, append=False, outfile=None, *args, **kwargs
    ):
        """
        Writes file from geometry in the specified style

        :geom: the Geometry to use
        :style: the file type style to generate
            Currently supported options: xyz (default), com, inp, in
            if outfile has one of these extensions, default is that style
        :append: for *.xyz, append geometry to the same file
        :outfile: output destination - default is
                  [geometry name] + [extension] or [geometry name] + [step] + [extension]
                  if outfile is False, no output file will be written, but the contents will be returned
        :theory: for com, inp, and in files, an object with a get_header and get_footer method
        """
        if isinstance(outfile, str) and style is None:
            name, ext = os.path.splitext(outfile)
            style = ext.strip(".")

        elif style is None:
            style = "xyz"

        if style.lower() not in write_types:
            if style.lower() == "gaussian":
                style = "com"
            elif style.lower() == "orca":
                style = "inp"
            elif style.lower() == "psi4":
                style = "in"
            elif style.lower() == "sqm":
                style = "sqmin"
            else:
                raise NotImplementedError(file_type_err.format(style))

        if (
            outfile is None
            and os.path.dirname(geom.name)
            and not os.access(os.path.dirname(geom.name), os.W_OK)
        ):
            os.makedirs(os.path.dirname(geom.name))
        if style.lower() == "xyz":
            out = cls.write_xyz(geom, append, outfile)
        elif style.lower() == "com":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_com(geom, theory, outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'com/gjf' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "inp":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_inp(geom, theory, outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'inp' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "in":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_in(geom, theory, outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'in' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "sqmin":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_sqm(geom, theory, outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'sqmin' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )

        return out

    @classmethod
    def write_xyz(cls, geom, append, outfile=None):
        mode = "a" if append else "w"
        fmt = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f}\n"
        s = "%i\n" % len(geom.atoms)
        s += "%s\n" % geom.comment
        for atom in geom.atoms:
            s += fmt.format(atom.element, *atom.coords)

        if outfile is None:
            # if no output file is specified, use the name of the geometry
            with open(geom.name + ".xyz", mode) as f:
                f.write(s)
        elif outfile is False:
            # if no output file is desired, just return the file contents
            return s.strip()
        else:
            # write output to the requested destination
            with open(outfile, mode) as f:
                f.write(s)

        return

    @classmethod
    def write_com(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        # get file content string
        header, header_warnings = theory.make_header(
            geom, return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, return_warnings=True, **kwargs
        )
        footer, footer_warnings = theory.make_footer(
            geom, return_warnings=True, **kwargs
        )

        s = header + mol + footer
        warnings = header_warnings + mol_warnings + footer_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                fname = "{}.{}.com".format(geom.name, step2str(kwargs["step"]))
            else:
                fname = "{}.com".format(geom.name)
            with open(fname, "w") as f:
                f.write(s)
        elif outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings
        return

    @classmethod
    def write_inp(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        fmt = "{:<3s} {: 9.5f} {: 9.5f} {: 9.5f}\n"
        s, warnings = theory.make_header(
            geom, style="orca", return_warnings=True, **kwargs
        )
        for atom in geom.atoms:
            s += fmt.format(atom.element, *atom.coords)

        s += "*\n"

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                fname = "{}.{}.inp".format(geom.name, step2str(kwargs["step"]))
            else:
                fname = "{}.inp".format(geom.name)
            with open(fname, "w") as f:
                f.write(s)
        elif outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            with open(outfile, "w") as f:
                f.write(s)
        if return_warnings:
            return warnings

    @classmethod
    def write_in(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        can accept "monomers" as a kwarg
        this should be a list of lists of atoms corresponding to the
        separate monomers in a sapt calculation
        this will only be used if theory.method.sapt is True
        if a sapt method is used but no monoers are given,
        geom's components attribute will be used intead
        """
        if "monomers" in kwargs:
            monomers = kwargs["monomers"]
            del kwargs["monomers"]
        else:
            monomers = None

        header, header_warnings = theory.make_header(
            geom, style="psi4", return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, style="psi4", return_warnings=True, **kwargs
        )
        footer, footer_warnings = theory.make_footer(
            geom, style="psi4", return_warnings=True, **kwargs
        )

        s = header + mol + footer
        warnings = header_warnings + mol_warnings + footer_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                fname = "{}.{}.in".format(geom.name, step2str(kwargs["step"]))
            else:
                fname = "{}.in".format(geom.name)
            with open(fname, "w") as f:
                f.write(s)
        elif outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            with open(outfile, "w") as f:
                f.write(s)
        if return_warnings:
            return warnings

    @classmethod
    def write_sqm(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        header, header_warnings = theory.make_header(
            geom, style="sqm", return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, style="sqm", return_warnings=True, **kwargs
        )

        s = header + mol
        warnings = header_warnings + mol_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                fname = "{}.{}.sqmin".format(geom.name, step2str(kwargs["step"]))
            else:
                fname = "{}.sqmin".format(geom.name)
            with open(fname, "w") as f:
                f.write(s)

        elif outfile is False:
            if return_warnings:
                return s, warnings
            return s

        else:
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings



@addlogger
class FileReader:
    """
    Attributes:
        name ''
        file_type ''
        comment ''
        atoms [Atom]
        other {}
    """

    LOG = None
    LOGLEVEL = "DEBUG"

    def __init__(
        self,
        fname,
        get_all=False,
        just_geom=True,
        freq_name=None,
        conf_name=None,
    ):
        """
        :fname: either a string specifying the file name of the file to read
            or a tuple of (str(name), str(file_type), str(content))
        :get_all: if true, optimization steps are  also saved in
            self.all_geom; otherwise only saves last geometry
        :just_geom: if true, does not store other information, such as
            frequencies, only what is needed to construct a Geometry() obj
        :freq_name: Name of the file containing the frequency output. Only use
            if this information is in a different file than `fname` (eg: xtb runs
            using the --hess runtype option)
        """
        # Initialization
        self.name = ""
        self.file_type = ""
        self.comment = ""
        self.atoms = []
        self.other = {}
        self.content = None
        self.all_geom = None

        # get file name and extention
        if isinstance(fname, str):
            self.name, self.file_type = os.path.splitext(fname)
            self.file_type = self.file_type.lower()[1:]
        elif isinstance(fname, (tuple, list)):
            self.name = fname[0]
            self.file_type = fname[1]
            self.content = fname[2]
        if self.file_type not in read_types:
            raise NotImplementedError(file_type_err.format(self.file_type))

        # Fill in attributes with geometry information
        if self.content is None:
            self.read_file(
                get_all, just_geom, freq_name=freq_name, conf_name=conf_name
            )
        elif isinstance(self.content, str):
            f = StringIO(self.content)
        elif isinstance(self.content, IOBase):
            f = self.content

        if self.content is not None:
            if self.file_type == "log":
                self.read_log(f, get_all, just_geom)
            elif any(self.file_type == ext for ext in ["sd", "sdf", "mol"]):
                self.read_sd(f)
            elif self.file_type == "xyz":
                self.read_xyz(f, get_all)
            elif self.file_type == "mol2":
                self.read_mol2(f, get_all)
            elif any(self.file_type == ext for ext in ["com", "gjf"]):
                self.read_com(f)
            elif self.file_type == "out":
                self.read_orca_out(f, get_all, just_geom)
            elif self.file_type == "dat":
                self.read_psi4_out(f, get_all, just_geom)
            elif self.file_type == "fchk":
                self.read_fchk(f, just_geom)
            elif self.file_type == "crest":
                self.read_crest(f, conf_name=conf_name)
            elif self.file_type == "xtb":
                self.read_xtb(f, freq_name=freq_name)
            elif self.file_type == "sqmout":
                self.read_sqm(f)

    def read_file(
        self, get_all=False, just_geom=True, freq_name=None, conf_name=None
    ):
        """
        Reads geometry information from fname.
        Parameters:
            get_all     If false (default), only keep the last geom
                        If true, self is last geom, but return list
                            of all others encountered
        """
        if os.path.isfile(self.name):
            f = open(self.name)
        else:
            fname = ".".join([self.name, self.file_type])
            fname = os.path.expanduser(fname)
            if os.path.isfile(fname):
                f = open(fname)
            else:
                raise FileNotFoundError(
                    "Error while looking for %s: could not find %s or %s in %s"
                    % (self.name, fname, self.name, os.getcwd())
                )

        if self.file_type == "xyz":
            self.read_xyz(f, get_all)
        elif self.file_type == "log":
            self.read_log(f, get_all, just_geom)
        elif any(self.file_type == ext for ext in ["com", "gjf"]):
            self.read_com(f)
        elif any(self.file_type == ext for ext in ["sd", "sdf", "mol"]):
            self.read_sd(f)
        elif self.file_type == "mol2":
            self.read_mol2(f)
        elif self.file_type == "out":
            self.read_orca_out(f, get_all, just_geom)
        elif self.file_type == "dat":
            self.read_psi4_out(f, get_all, just_geom)
        elif self.file_type == "fchk":
            self.read_fchk(f, just_geom)
        elif self.file_type == "crest":
            self.read_crest(f, conf_name=conf_name)
        elif self.file_type == "xtb":
            self.read_xtb(f, freq_name=freq_name)
        elif self.file_type == "sqmout":
            self.read_sqm(f)

        f.close()
        return

    def skip_lines(self, f, n):
        for i in range(n):
            f.readline()
        return

    def read_xyz(self, f, get_all=False):
        self.all_geom = []
        # number of atoms
        f.readline()
        # comment
        self.comment = f.readline().strip()
        # atom info
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                int(line)
                if get_all:
                    self.all_geom += [
                        (deepcopy(self.comment), deepcopy(self.atoms))
                    ]
                self.comment = f.readline().strip()
                self.atoms = []
            except ValueError:
                line = line.split()
                self.atoms += [Atom(element=line[0], coords=line[1:4])]
                for i, a in enumerate(self.atoms):
                    a.name = str(i + 1)
        if get_all:
            self.all_geom += [(deepcopy(self.comment), deepcopy(self.atoms))]

    def read_sd(self, f, get_all=False):
        self.all_geom = []
        lines = f.readlines()
        progress = 0
        for i, line in enumerate(lines):
            progress += 1
            if "$$$$" in line:
                progress = 0
                if get_all:
                    self.all_geom.append(
                        [deepcopy(self.comment), deepcopy(self.atoms)]
                    )

                continue

            if progress == 3:
                self.comment = line.strip()

            if progress == 4:
                counts = line.split()
                natoms = int(counts[0])
                nbonds = int(counts[1])

            if progress == 5:
                self.atoms = []
                for line in lines[i : i + natoms]:
                    atom_info = line.split()
                    self.atoms += [
                        Atom(element=atom_info[3], coords=atom_info[0:3])
                    ]

                for line in lines[i + natoms : i + natoms + nbonds]:
                    a1, a2 = [int(x) - 1 for x in line.split()[0:2]]
                    self.atoms[a1].connected.add(self.atoms[a2])
                    self.atoms[a2].connected.add(self.atoms[a1])

                for j, a in enumerate(self.atoms):
                    a.name = str(j + 1)

    def read_mol2(self, f, get_all=False):
        """
        read TRIPOS mol2
        """
        atoms = []

        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("@<TRIPOS>MOLECULE"):
                self.comment = lines[i + 1]
                info = lines[i + 2].split()
                n_atoms = int(info[0])
                n_bonds = int(info[1])
                i += 3

            elif lines[i].startswith("@<TRIPOS>ATOM"):
                for j in range(0, n_atoms):
                    i += 1
                    info = lines[i].split()
                    # name = info[1]
                    coords = np.array([float(x) for x in info[2:5]])
                    element = re.match("([A-Za-z]+)", info[5]).group(1)
                    atoms.append(
                        Atom(element=element, coords=coords, name=str(j + 1))
                    )

                self.atoms = atoms

            elif lines[i].startswith("@<TRIPOS>BOND"):
                for j in range(0, n_bonds):
                    i += 1
                    info = lines[i].split()
                    a1, a2 = [int(ndx) - 1 for ndx in info[1:3]]
                    self.atoms[a1].connected.add(self.atoms[a2])
                    self.atoms[a2].connected.add(self.atoms[a1])

            i += 1

    def read_psi4_out(self, f, get_all=False, just_geom=True):
        def get_atoms(f, n):
            rv = []
            self.skip_lines(f, 1)
            n += 2
            line = f.readline()
            i = 0
            mass = 0
            while line.strip():
                i += 1
                line = line.strip()
                atom_info = line.split()
                element = atom_info[0]
                # might be a ghost atom - like for sapt
                if "Gh" in element:
                    element = element.strip("Gh(").strip(")")
                coords = np.array([float(x) for x in atom_info[1:-1]])
                rv += [Atom(element=element, coords=coords, name=str(i))]
                mass += float(atom_info[-1])

                line = f.readline()
                n += 1

            return rv, mass, n

        line = f.readline()
        n = 1
        read_geom = False
        while line != "":
            if "* O   R   C   A *" in line:
                self.file_type = "out"
                return self.read_orca_out(
                    f, get_all=get_all, just_geom=just_geom
                )
            if line.startswith("    Geometry (in Angstrom), charge"):
                if not just_geom:
                    self.other["charge"] = int(line.split()[5].strip(","))
                    self.other["multiplicity"] = int(
                        line.split()[8].strip(":")
                    )

            elif line.strip() == "SCF":
                read_geom = True

            elif line.strip().startswith("Center") and read_geom:
                read_geom = False
                if get_all and len(self.atoms) > 0:
                    if self.all_geom is None:
                        self.all_geom = []

                    self.all_geom += [
                        (deepcopy(self.atoms), deepcopy(self.other))
                    ]

                self.atoms, mass, n = get_atoms(f, n)
                if not just_geom:
                    self.other["mass"] = mass
                    self.other["mass"] *= UNIT.AMU_TO_KG

            if just_geom:
                line = f.readline()
                n += 1
                continue
            else:
                if line.strip().startswith("Total Energy ="):
                    self.other["energy"] = float(line.split()[-1])

                elif line.strip().startswith("Total E0"):
                    self.other["energy"] = float(line.split()[-2])

                elif line.strip().startswith("Correction ZPE"):
                    self.other["ZPVE"] = float(line.split()[-4])

                elif line.strip().startswith("Total ZPE"):
                    self.other["E_ZPVE"] = float(line.split()[-2])

                elif line.strip().startswith("Total H, Enthalpy"):
                    self.other["enthalpy"] = float(line.split()[-2])

                elif line.strip().startswith("Total G, Free"):
                    self.other["free_energy"] = float(line.split()[-2])
                    self.other["temperature"] = float(line.split()[-4])

                elif "symmetry no. =" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1].strip(")")
                    )

                elif (
                    line.strip().startswith("Rotational constants:")
                    and line.strip().endswith("[cm^-1]")
                    and "rotational_temperature" not in self.other
                ):
                    self.other["rotational_temperature"] = [
                        float(x) if is_num(x) else 0
                        for x in line.split()[-8:-1:3]
                    ]
                    self.other["rotational_temperature"] = [
                        x
                        * PHYSICAL.SPEED_OF_LIGHT
                        * PHYSICAL.PLANCK
                        / PHYSICAL.KB
                        for x in self.other["rotational_temperature"]
                    ]

                elif line.startswith("  Vibration "):
                    freq_str = ""
                    while not line.strip().startswith("=="):
                        freq_str += line
                        line = f.readline()
                        n += 1

                    self.other["frequency"] = Frequency(
                        freq_str, hpmodes=False, style="dat"
                    )

                elif PSI4_NORM_FINISH in line:
                    self.other["finished"] = True

                elif line.startswith("    Convergence Criteria"):
                    # for tolerances:
                    # psi4 puts '*' next to converged values and 'o' in place of things that aren't monitored
                    grad = {}

                    dE_tol = line[24:38]
                    if "o" in dE_tol:
                        dE_tol = None
                    else:
                        dE_tol = dE_tol.split()[0]

                    max_f_tol = line[38:52]
                    if "o" in max_f_tol:
                        max_f_tol = None
                    else:
                        max_f_tol = max_f_tol.split()[0]

                    rms_f_tol = line[52:66]
                    if "o" in rms_f_tol:
                        rms_f_tol = None
                    else:
                        rms_f_tol = rms_f_tol.split()[0]

                    max_d_tol = line[66:80]
                    if "o" in max_d_tol:
                        max_d_tol = None
                    else:
                        max_d_tol = max_d_tol.split()[0]

                    rms_d_tol = line[80:94]
                    if "o" in rms_d_tol:
                        rms_d_tol = None
                    else:
                        rms_d_tol = rms_d_tol.split()[0]

                    line = f.readline()
                    line = f.readline()
                    n += 2

                    # for convergence:
                    # psi4 puts '*' next to converged values and 'o' next to things that aren't monitored
                    if dE_tol is not None:
                        dE_conv = line[24:38]
                        dE = float(dE_conv.split()[0])
                        grad["Delta E"] = {}
                        grad["Delta E"]["value"] = dE
                        grad["Delta E"]["converged"] = "*" in dE_conv

                    if max_f_tol is not None:
                        max_f_conv = line[38:52]
                        max_f = float(max_f_conv.split()[0])
                        grad["Max Force"] = {}
                        grad["Max Force"]["value"] = max_f
                        grad["Max Force"]["converged"] = "*" in max_f_conv

                    if rms_f_tol is not None:
                        rms_f_conv = line[52:66]
                        rms_f = float(rms_f_conv.split()[0])
                        grad["RMS Force"] = {}
                        grad["RMS Force"]["value"] = rms_f
                        grad["RMS Force"]["converged"] = "*" in rms_f_conv

                    if max_d_tol is not None:
                        max_d_conv = line[66:80]
                        max_d = float(max_d_conv.split()[0])
                        grad["Max Disp"] = {}
                        grad["Max Disp"]["value"] = max_d
                        grad["Max Disp"]["converged"] = "*" in max_d_conv

                    if rms_d_tol is not None:
                        rms_d_conv = line[80:94]
                        rms_d = float(rms_d_conv.split()[0])
                        grad["RMS Disp"] = {}
                        grad["RMS Disp"]["value"] = rms_d
                        grad["RMS Disp"]["converged"] = "*" in max_d_conv

                    self.other["gradient"] = grad

                elif "Total Gradient" in line:
                    gradient = np.zeros((len(self.atoms), 3))
                    self.skip_lines(f, 2)
                    n += 2
                    for i in range(0, len(self.atoms)):
                        n += 1
                        line = f.readline()
                        info = line.split()
                        gradient[i] = np.array([float(x) for x in info[1:]])

                    self.other["forces"] = -gradient

                elif "SAPT Results" in line:
                    self.skip_lines(f, 1)
                    n += 1
                    while "Total sSAPT" not in line:
                        n += 1
                        line = f.readline()
                        if "---" in line:
                            break
                        if len(line.strip()) > 0:
                            if "Special recipe" in line:
                                continue
                            item = line[:26].strip()
                            val = 1e-3 * float(line[34:47])
                            self.other[item] = val

                elif "SCF energy" in line:
                    self.other["SCF energy"] = float(line.split()[-1])

                elif "correlation energy" in line and "=" in line:
                    item = line.split("=")[0].strip()
                    self.other[item] = float(line.split()[-1])

                elif "Full point group" in line:
                    self.other["full_point_group"] = line.split()[-1]

                elif "Molecular point group" in line:
                    self.other["molecular_point_group"] = line.split()[-1]

                elif (
                    "total energy" in line
                    and "=" in line
                    or re.search("\(.\) energy", line)
                ):
                    item = line.split("=")[0].strip().strip("*").strip()
                    self.other[item] = float(line.split()[-1])
                    # hopefully the highest level energy gets printed last
                    self.other["energy"] = self.other[item]

                elif "Total Energy" in line and "=" in line:
                    item = line.split("=")[0].strip().strip("*").strip()
                    self.other[item] = float(line.split()[-2])
                    # hopefully the highest level energy gets printed last
                    self.other["energy"] = self.other[item]

                elif "Correlation Energy" in line and "=" in line:
                    item = line.split("=")[0].strip().strip("*").strip()
                    if "DFT Exchange-Correlation" in item:
                        self.other[item] = float(line.split()[-1])
                    else:
                        self.other[item] = float(line.split()[-2])

                if "error" not in self.other:
                    for err in ERROR_PSI4:
                        if err in line:
                            self.other["error"] = ERROR_PSI4[err]
                            self.other["error_msg"] = line.strip()

                line = f.readline()
                n += 1

    def read_orca_out(self, f, get_all=False, just_geom=True):
        """read orca output file"""

        nrg_regex = re.compile("(?:[A-Za-z]+\s+)?E\((.*)\)\s*\.\.\.\s*(.*)$")

        def add_grad(grad, name, line):
            grad[name] = {}
            grad[name]["value"] = line.split()[-3]
            grad[name]["converged"] = line.split()[-1] == "YES"

        def get_atoms(f, n):
            """parse atom info"""
            rv = []
            self.skip_lines(f, 1)
            n += 2
            line = f.readline()
            i = 0
            while line.strip():
                i += 1
                line = line.strip()
                atom_info = line.split()
                element = atom_info[0]
                coords = np.array([float(x) for x in atom_info[1:]])
                rv += [Atom(element=element, coords=coords, name=str(i))]

                line = f.readline()
                n += 1

            return rv, n

        line = f.readline()
        n = 1
        while line != "":
            if (
                "Psi4: An Open-Source Ab Initio Electronic Structure Package"
                in line
            ):
                self.file_type = "dat"
                return self.read_psi4_out(
                    f, get_all=get_all, just_geom=just_geom
                )
            if line.startswith("CARTESIAN COORDINATES (ANGSTROEM)"):
                if get_all and len(self.atoms) > 0:
                    if self.all_geom is None:
                        self.all_geom = []
                    self.all_geom += [
                        (deepcopy(self.atoms), deepcopy(self.other))
                    ]

                self.atoms, n = get_atoms(f, n)

            if just_geom:
                line = f.readline()
                n += 1
                continue
            else:
                nrg = nrg_regex.match(line)
                if nrg is not None:
                    nrg_type = nrg.group(1)
                    # for some reason, ORCA prints MP2 correlation energy
                    # as E(MP2) for CC jobs
                    if nrg_type == "MP2":
                        nrg_type = "MP2 CORR"
                    self.other["E(%s)" % nrg_type] = float(nrg.group(2))

                if line.startswith("FINAL SINGLE POINT ENERGY"):
                    # if the wavefunction doesn't converge, ORCA prints a message next
                    # to the energy so we can't use line.split()[-1]
                    self.other["energy"] = float(line.split()[4])

                if line.startswith("TOTAL SCF ENERGY"):
                    self.skip_lines(f, 2)
                    line = f.readline()
                    n += 3
                    self.other["SCF energy"] = float(line.split()[3])

                elif "TOTAL ENERGY:" in line:
                    item = line.split()[-5] + " energy"
                    self.other[item] = float(line.split()[-2])

                elif "CORRELATION ENERGY" in line and "Eh" in line:
                    item = line.split()[-6] + " correlation energy"
                    self.other[item] = float(line.split()[-2])

                elif line.startswith("CARTESIAN GRADIENT"):
                    gradient = np.zeros((len(self.atoms), 3))
                    self.skip_lines(f, 2)
                    n += 2
                    for i in range(0, len(self.atoms)):
                        n += 1
                        line = f.readline()
                        # orca prints a warning before gradient if some
                        # coordinates are constrained
                        if line.startswith("WARNING:"):
                            continue
                        info = line.split()
                        gradient[i] = np.array([float(x) for x in info[3:]])

                    self.other["forces"] = -gradient

                elif line.startswith("VIBRATIONAL FREQUENCIES"):
                    stage = "frequencies"
                    freq_str = "VIBRATIONAL FREQUENCIES\n"
                    self.skip_lines(f, 4)
                    n += 5
                    line = f.readline()
                    while not (stage == "IR" and line == "\n") and line:
                        if "--" not in line and line != "\n":
                            freq_str += line

                        if "NORMAL MODES" in line:
                            stage = "modes"
                            self.skip_lines(f, 6)
                            n += 6

                        if "IR SPECTRUM" in line:
                            stage = "IR"
                            self.skip_lines(f, 2)
                            n += 2

                        n += 1
                        line = f.readline()

                    self.other["frequency"] = Frequency(
                        freq_str, hpmodes=False, style="out"
                    )

                elif line.startswith("Temperature"):
                    self.other["temperature"] = float(line.split()[2])

                elif line.startswith("Total Mass"):
                    # this may only get printed for freq jobs
                    self.other["mass"] = float(line.split()[3])
                    self.other["mass"] *= UNIT.AMU_TO_KG

                elif line.startswith(" Total Charge"):
                    self.other["charge"] = int(line.split()[-1])

                elif line.startswith(" Multiplicity"):
                    self.other["multiplicity"] = int(line.split()[-1])

                elif "rotational symmetry number" in line:
                    # TODO: make this cleaner
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-2]
                    )

                elif "Symmetry Number:" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1]
                    )

                elif line.startswith("Zero point energy"):
                    self.other["ZPVE"] = float(line.split()[4])

                elif line.startswith("Total Enthalpy"):
                    self.other["enthalpy"] = float(line.split()[3])

                elif line.startswith("Final Gibbs"):
                    # NOTE - Orca seems to only print Grimme's Quasi-RRHO free energy
                    # RRHO can be computed in AaronTool's CompOutput by setting the w0 to 0
                    self.other["free_energy"] = float(line.split()[5])

                elif line.startswith("Rotational constants in cm-1:"):
                    # orca doesn't seem to print rotational constants in older versions
                    self.other["rotational_temperature"] = [
                        float(x) for x in line.split()[-3:]
                    ]
                    self.other["rotational_temperature"] = [
                        x
                        * PHYSICAL.SPEED_OF_LIGHT
                        * PHYSICAL.PLANCK
                        / PHYSICAL.KB
                        for x in self.other["rotational_temperature"]
                    ]

                elif "Point Group:" in line:
                    self.other["full_point_group"] = line.split()[2][:-1]

                elif "Symmetry Number" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1]
                    )

                elif "sn is the rotational symmetry number" in line:
                    # older versions of orca print this differently
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-2]
                    )

                elif "Geometry convergence" in line:
                    grad = {}
                    self.skip_lines(f, 2)
                    n += 3
                    line = f.readline()
                    while line and re.search("\w", line):
                        if re.search("Energy\schange", line):
                            add_grad(grad, "Delta E", line)
                        elif re.search("RMS\sgradient", line):
                            add_grad(grad, "RMS Force", line)
                        elif re.search("MAX\sgradient", line):
                            add_grad(grad, "Max Force", line)
                        elif re.search("RMS\sstep", line):
                            add_grad(grad, "RMS Disp", line)
                        elif re.search("MAX\sstep", line):
                            add_grad(grad, "Max Disp", line)

                        line = f.readline()
                        n += 1

                    self.other["gradient"] = grad

                elif ORCA_NORM_FINISH in line:
                    self.other["finished"] = True

                # TODO E_ZPVE
                if "error" not in self.other:
                    for err in ERROR_ORCA:
                        if err in line:
                            self.other["error"] = ERROR_ORCA[err]
                            self.other["error_msg"] = line.strip()
                            break

                line = f.readline()
                n += 1

        if not just_geom:
            if "finished" not in self.other:
                self.other["finished"] = False

    def read_log(self, f, get_all=False, just_geom=True):
        def get_atoms(f, n):
            rv = []
            self.skip_lines(f, 4)
            line = f.readline()
            n += 5
            while "--" not in line:
                line = line.strip()
                line = line.split()
                for l in line:
                    try:
                        float(l)
                    except ValueError:
                        msg = "Error detected with log file on line {}"
                        raise IOError(msg.format(n))
                elem = ELEMENTS[int(line[1])]
                flag = not bool(line[2])
                rv += [Atom(element=elem, flag=flag, coords=line[3:])]
                line = f.readline()
                n += 1
            return rv, n

        def get_params(f, n):
            rv = []
            self.skip_lines(f, 2)
            n += 3
            line = f.readline()
            if "Definition" in line:
                definition = True
            else:
                definition = False
            self.skip_lines(f, 1)
            n += 2
            line = f.readline()
            while "--" not in line:
                line = line.split()
                param = line[1]
                if definition:
                    val = float(line[3])
                else:
                    val = float(line[2])
                rv.append((param, val))
                line = f.readline()
                n += 1
            return rv, n

        def get_modredundant(f, n):
            """read constraints for modredundant section"""
            rv = {}
            line = f.readline()
            n += 1
            while line.strip():
                atom_match = re.search("X\s+(\d+)\s+F", line)
                bond_match = re.search("B\s+(\d+)\s+(\d+)\s+F", line)
                angle_match = re.search("A\s+(\d+)\s+(\d+)\s+(\d+)\s+F", line)
                torsion_match = re.search(
                    "D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+F", line
                )
                if atom_match:
                    if "atoms" not in rv:
                        rv["atoms"] = ""
                    else:
                        rv["atoms"] += ","
                    rv["atoms"] += atom_match.group(1)
                elif bond_match:
                    if "bonds" not in rv:
                        rv["bonds"] = []
                    rv["bonds"].append(
                        ",".join([bond_match.group(1), bond_match.group(2)])
                    )
                elif angle_match:
                    if "angles" not in rv:
                        rv["angles"] = []
                    rv["angles"].append(
                        ",".join(
                            [
                                angle_match.group(1),
                                angle_match.group(2),
                                angle_match.group(3),
                            ]
                        )
                    )
                elif torsion_match:
                    if "torsions" not in rv:
                        rv["torsions"] = []
                    rv["torsions"].append(
                        ",".join(
                            [
                                torsion_match.group(1),
                                torsion_match.group(2),
                                torsion_match.group(3),
                                torsion_match.group(4),
                            ]
                        )
                    )

                line = f.readline()
                n += 1

            return rv, n

        self.all_geom = []
        line = f.readline()
        self.other["archive"] = ""
        constraints = {}
        self.other["opt_steps"] = 0
        found_archive = False
        n = 1
        route = None
        while line != "":
            # route
            # we need to grab the route b/c sometimes 'hpmodes' can get split onto multiple lines:
            # B3LYP/genecp EmpiricalDispersion=GD3 int=(grid=superfinegrid) freq=(h
            # pmodes,noraman,temperature=313.15)
            if line.strip().startswith("#") and route is None:
                route = ""
                while "------" not in line:
                    route += line.strip()
                    n += 1
                    line = f.readline()
            # archive entry
            if line.strip().startswith("1\\1\\"):
                found_archive = True
                line = "@" + line.strip()[4:]
            if found_archive and line.strip().endswith("@"):
                self.other["archive"] = self.other["archive"][:-2] + "\\\\"
                found_archive = False
            elif found_archive:
                self.other["archive"] += line.strip()

            # geometry
            if re.search("(Standard|Input) orientation:", line):
                if get_all and len(self.atoms) > 0:
                    self.all_geom += [
                        (deepcopy(self.atoms), deepcopy(self.other))
                    ]
                self.atoms, n = get_atoms(f, n)
                self.other["opt_steps"] += 1

            if re.search(
                "The following ModRedundant input section has been read:", line
            ):
                constraints, n = get_modredundant(f, n)

            if just_geom:
                line = f.readline()
                n += 1
                continue
                # z-matrix parameters
            if re.search("Optimized Parameters", line):
                self.other["params"], n = get_params(f, n)
            if "Symbolic Z-matrix:" in line:
                line = f.readline()
                n += 1
                match = re.search(
                    "Charge\s*=\s*(-?\d+)\s*Multiplicity\s*=\s*(\d+)", line
                )
                if match is not None:
                    self.other["charge"] = int(match.group(1))
                    self.other["multiplicity"] = int(match.group(2))

            # status
            if NORM_FINISH in line:
                self.other["finished"] = True
            if "SCF Done" in line:
                tmp = [word.strip() for word in line.split()]
                idx = tmp.index("=")
                self.other["energy"] = float(tmp[idx + 1])
            if "Molecular mass:" in line:
                self.other["mass"] = float(float_num.search(line).group(0))
                self.other["mass"] *= UNIT.AMU_TO_KG

            # Frequencies
            if route is not None and "hpmodes" in route.lower():
                self.other["hpmodes"] = True
            if "Harmonic frequencies" in line:
                freq_str = line
                line = f.readline()
                while line != "\n":
                    n += 1
                    freq_str += line
                    line = f.readline()
                if "hpmodes" not in self.other:
                    self.other["hpmodes"] = False
                self.other["frequency"] = Frequency(
                    freq_str, self.other["hpmodes"]
                )

            if "Anharmonic Infrared Spectroscopy" in line:
                self.skip_lines(f, 5)
                n += 5
                anharm_str = ""
                combinations_read = False
                combinations = False
                line = f.readline()
                while not combinations_read:
                    n += 1
                    anharm_str += line
                    if "Combination Bands" in line:
                        combinations = True
                    line = f.readline()
                    if combinations and line == "\n":
                        combinations_read = True
                
                self.other["frequency"].parse_gaussian_anharm(
                    anharm_str.splitlines()
                )

            # Thermo
            if re.search("Temperature\s*\d+\.\d+", line):
                self.other["temperature"] = float(
                    float_num.search(line).group(0)
                )
            if "Rotational constants (GHZ):" in line:
                rot = float_num.findall(line)
                rot = [
                    float(r) * PHYSICAL.PLANCK * (10 ** 9) / PHYSICAL.KB
                    for r in rot
                ]
                self.other["rotational_temperature"] = rot
            if "Sum of electronic and zero-point Energies=" in line:
                self.other["E_ZPVE"] = float(float_num.search(line).group(0))
            if "Sum of electronic and thermal Enthalpies=" in line:
                self.other["enthalpy"] = float(float_num.search(line).group(0))
            if "Sum of electronic and thermal Free Energies=" in line:
                self.other["free_energy"] = float(
                    float_num.search(line).group(0)
                )
            if "Zero-point correction=" in line:
                self.other["ZPVE"] = float(float_num.search(line).group(0))
            if "Rotational symmetry number" in line:
                self.other["rotational_symmetry_number"] = int(
                    re.search("\d+", line).group(0)
                )

            # Gradient
            if re.search("Threshold\s+Converged", line) is not None:
                line = f.readline()
                n += 1
                grad = {}

                def add_grad(line, name, grad):
                    line = line.split()
                    grad[name] = {
                        "value": line[2],
                        "threshold": line[3],
                        "converged": True if line[4] == "YES" else False,
                    }
                    return grad

                while line != "":
                    if "Predicted change in Energy" in line:
                        break
                    if re.search("Maximum\s+Force", line) is not None:
                        grad = add_grad(line, "Max Force", grad)
                    if re.search("RMS\s+Force", line) is not None:
                        grad = add_grad(line, "RMS Force", grad)
                    if re.search("Maximum\s+Displacement", line) is not None:
                        grad = add_grad(line, "Max Disp", grad)
                    if re.search("RMS\s+Displacement", line) is not None:
                        grad = add_grad(line, "RMS Disp", grad)
                    line = f.readline()
                    n += 1
                self.other["gradient"] = grad

            # symmetry
            if "Full point group" in line:
                self.other["full_point_group"] = line.split()[-3]

            if "Largest Abelian subgroup" in line:
                self.other["abelian_subgroup"] = line.split()[-3]

            if "Largest concise Abelian subgroup" in line:
                self.other["concise_abelian_subgroup"] = line.split()[-3]

            # forces
            if "Forces (Hartrees/Bohr)" in line:
                gradient = np.zeros((len(self.atoms), 3))
                self.skip_lines(f, 2)
                n += 2
                for i in range(0, len(self.atoms)):
                    n += 1
                    line = f.readline()
                    info = line.split()
                    gradient[i] = np.array([float(x) for x in info[2:]])

                self.other["forces"] = gradient

            # capture errors
            # only keep first error, want to fix one at a time
            if "error" not in self.other:
                for err in ERROR:
                    if re.search(err, line):
                        self.other["error"] = ERROR[err]
                        self.other["error_msg"] = line.strip()
                        break

            line = f.readline()
            n += 1

        if not just_geom:
            if route is not None:
                other_kwargs = {GAUSSIAN_ROUTE: {}}
                route_spec = re.compile("(\w+)=?\((.*)\)")
                method_and_basis = re.search(
                    "#(?:[NnPpTt]\s+?)(\S+)|#\s*?(\S+)", route
                )
                if method_and_basis is not None:
                    if method_and_basis.group(2):
                        method_info = method_and_basis.group(2).split("/")
                    else:
                        method_info = method_and_basis.group(1).split("/")

                    method = method_info[0]
                    if len(method_info) > 1:
                        basis = method_info[1]
                    else:
                        basis = None

                    route_options = route.split()
                    job_type = []
                    grid = None
                    solvent = None
                    for option in route_options:
                        if option.startswith("#"):
                            continue
                        elif option.startswith(method):
                            continue

                        option_lower = option.lower()
                        if option_lower.startswith("opt"):
                            ts = False
                            match = route_spec.search(option)
                            if match:
                                options = match.group(2).split(",")
                            elif option_lower.startswith("opt="):
                                options = ["".join(option.split("=")[1:])]
                            else:
                                if not constraints:
                                    # if we didn't read constraints, try using flagged atoms instead
                                    from AaronTools.finders import FlaggedAtoms

                                    constraints = {"atoms": FlaggedAtoms}
                                job_type.append(
                                    OptimizationJob(constraints=constraints)
                                )
                                continue

                            other_kwargs[GAUSSIAN_ROUTE]["opt"] = []

                            for opt in options:
                                if opt.lower() == "ts":
                                    ts = True
                                else:
                                    other_kwargs[GAUSSIAN_ROUTE]["opt"].append(
                                        opt
                                    )

                            job_type.append(
                                OptimizationJob(
                                    transition_state=ts,
                                    constraints=constraints,
                                )
                            )

                        elif option_lower.startswith("freq"):
                            temp = 298.15
                            match = route_spec.search(option)
                            if match:
                                options = match.group(2).split(",")
                            elif option_lower.startswith("freq="):
                                options = "".join(option.split("=")[1:])
                            else:
                                job_type.append(FrequencyJob())
                                continue

                            other_kwargs[GAUSSIAN_ROUTE]["freq"] = []

                            for opt in options:
                                if opt.lower().startswith("temp"):
                                    temp = float(opt.split("=")[1])
                                else:
                                    other_kwargs[GAUSSIAN_ROUTE][
                                        "freq"
                                    ].append(opt)

                            job_type.append(FrequencyJob(temperature=temp))

                        elif option_lower == "sp":
                            job_type.append(SinglePointJob())

                        elif option_lower.startswith("int"):
                            match = route_spec.search(option)
                            if match:
                                options = match.group(2).split(",")
                            elif option_lower.startswith("freq="):
                                options = "".join(option.split("=")[1:])
                            else:
                                job_type.append(FrequencyJob())
                                continue

                            for opt in options:
                                if opt.lower().startswith("grid"):
                                    grid_name = opt.split("=")[1]
                                    grid = IntegrationGrid(grid_name)
                                else:
                                    if "Integral" not in other_kwargs[GAUSSIAN_ROUTE]:
                                        other_kwargs[GAUSSIAN_ROUTE]["Integral"] = []
                                    other_kwargs[GAUSSIAN_ROUTE][
                                        "Integral"
                                    ].append(opt)

                        else:
                            # TODO: parse solvent
                            match = route_spec.search(option)
                            if match:
                                keyword = match.group(1)
                                options = match.group(2).split(",")
                                other_kwargs[GAUSSIAN_ROUTE][keyword] = options
                            elif "=" in option:
                                keyword = option.split("=")[0]
                                options = "".join(option.split("=")[1:])
                                other_kwargs[GAUSSIAN_ROUTE][keyword] = [
                                    options
                                ]
                            else:
                                other_kwargs[GAUSSIAN_ROUTE][option] = []
                                continue

                    self.other["other_kwargs"] = other_kwargs
                    try:
                        theory = Theory(
                            charge=self.other["charge"],
                            multiplicity=self.other["multiplicity"],
                            job_type=job_type,
                            basis=basis,
                            method=method,
                            grid=grid,
                            solvent=solvent,
                        )
                        self.other["theory"] = theory
                    except KeyError:
                        # if there is a serious error, too little info may be available
                        # to properly create the theory object
                        pass

        for i, a in enumerate(self.atoms):
            a.name = str(i + 1)

        if "finished" not in self.other:
            self.other["finished"] = False
        if "error" not in self.other:
            self.other["error"] = None
        return

    def read_com(self, f):
        found_atoms = False
        found_constraint = False
        atoms = []
        other = {}
        for line in f:
            # header
            if line.startswith("%"):
                continue
            if line.startswith("#"):
                method = re.search("^#([NnPpTt]\s+?)(\S+)|^#\s*?(\S+)", line)
                # route can be #n functional/basis ...
                # or #functional/basis ...
                # or # functional/basis ...
                if method.group(3):
                    other["method"] = method.group(3)
                else:
                    other["method"] = method.group(2)
                if "temperature=" in line:
                    other["temperature"] = re.search(
                        "temperature=(\d+\.?\d*)", line
                    ).group(1)
                if "solvent=" in line:
                    other["solvent"] = re.search(
                        "solvent=(\S+)\)", line
                    ).group(1)
                if "scrf=" in line:
                    # solvent model should be non-greedy b/c solvent name can have commas
                    other["solvent_model"] = re.search(
                        "scrf=\((\S+?),", line
                    ).group(1)
                if "EmpiricalDispersion=" in line:
                    other["emp_dispersion"] = re.search(
                        "EmpiricalDispersion=(\S+)", line
                    ).group(1)
                if "int=(grid" in line or "integral=(grid" in line.lower():
                    other["grid"] = re.search(
                        "(?:int||Integral)=\(grid[(=](\S+?)\)", line
                    ).group(1)
                # comments can be multiple lines long
                # but there should be a blank line between the route and the comment
                # and another between the comment and the charge+mult
                blank_lines = 0
                while blank_lines < 2:
                    line = f.readline().strip()
                    if len(line) == 0:
                        blank_lines += 1
                    else:
                        if "comment" not in other:
                            other["comment"] = ""
                        other["comment"] += "%s\n" % line
                other["comment"] = (
                    other["comment"].strip() if "comment" in other else ""
                )
                line = f.readline()
                if len(line.split()) > 1:
                    line = line.split()
                else:
                    line = line.split(",")
                other["charge"] = int(line[0])
                other["multiplicity"] = int(line[1])
                found_atoms = True
                continue
            # constraints
            if found_atoms and line.startswith("B") and line.endswith("F"):
                found_constraint = True
                if "constraint" not in other:
                    other["constraint"] = []
                other["constraint"] += [float_num.findall(line)]
                continue
            # footer
            if found_constraint:
                if "footer" not in other:
                    other["footer"] = ""
                other["footer"] += line
                continue
            # atom coords
            nums = float_num.findall(line)
            line = line.split()
            if len(line) == 5 and is_alpha(line[0]) and len(nums) == 4:
                if not is_int(line[1]):
                    continue
                a = Atom(element=line[0], coords=nums[1:], flag=nums[0])
                atoms += [a]
            elif len(line) == 4 and is_alpha(line[0]) and len(nums) == 3:
                a = Atom(element=line[0], coords=nums)
                atoms += [a]
            else:
                continue
        for i, a in enumerate(atoms):
            a.name = str(i + 1)
        self.atoms = atoms
        self.other = other
        return

    def read_fchk(self, f, just_geom=True):
        def parse_to_list(i, lines, length, data_type):
            """takes a block in an fchk file and turns it into an array
            block headers all end with N=   <int>
            the length of the array will be <int>
            the data type is specified by data_type"""
            i += 1
            line = lines[i]
            items_per_line = len(line.split())
            total_items = items_per_line
            num_lines = 1
            while total_items < length:
                total_items += items_per_line
                num_lines += 1

            block = ""
            for line in lines[i : i + num_lines]:
                block += " "
                block += line

            return (
                np.array([data_type(x) for x in block.split()]),
                i + num_lines,
            )

        self.atoms = []
        atom_numbers = []
        atom_coords = []

        other = {}

        int_info = re.compile("([\S\s]+?)\s*I\s*([N=]*)\s*(-?\d+)")
        real_info = re.compile(
            "([\S\s]+?)\s*R\s*([N=])*\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)"
        )

        theory = Theory()

        lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if i == 1:
                job_info = line.split()
                if job_info[0] == "SP":
                    theory.job_type = [SinglePointJob()]
                elif job_info[0] == "FOPT":
                    theory.job_type[OptimizationJob()]
                elif job_info[0] == "FTS":
                    theory.job_type = [OptimizationJob(transition_state=True)]
                elif job_info[0] == "FORCE":
                    theory.job_type = [ForceJob()]
                elif job_info[0] == "FREQ":
                    theory.job_type = [FrequencyJob()]

                theory.method = job_info[1]
                if len(job_info) > 2:
                    theory.basis = job_info[2]

                i += 1
                continue

            int_match = int_info.match(line)
            real_match = real_info.match(line)
            if int_match is not None:
                data = int_match.group(1)
                value = int_match.group(3)
                if data == "Charge" and not just_geom:
                    theory.charge = int(value)
                elif data == "Multiplicity" and not just_geom:
                    theory.multiplicity = int(value)
                elif data == "Atomic numbers":
                    atom_numbers, i = parse_to_list(i, lines, int(value), int)
                elif not just_geom:
                    if int_match.group(2):
                        other[data], i = parse_to_list(
                            i, lines, int(value), int
                        )
                        continue
                    else:
                        other[data] = int(value)

            elif real_match is not None:
                data = real_match.group(1)
                value = real_match.group(3)
                if data == "Current cartesian coordinates":
                    atom_coords, i = parse_to_list(i, lines, int(value), float)
                elif data == "Total Energy":
                    other["energy"] = float(value)
                elif not just_geom:
                    if real_match.group(2):
                        other[data], i = parse_to_list(
                            i, lines, int(value), float
                        )
                        continue
                    else:
                        other[data] = float(value)

            i += 1

        self.other = other
        self.other["theory"] = theory

        coords = np.reshape(atom_coords, (len(atom_numbers), 3))
        for n, (atnum, coord) in enumerate(zip(atom_numbers, coords)):
            atom = Atom(
                element=ELEMENTS[atnum],
                coords=UNIT.A0_TO_BOHR * coord,
                name=str(n + 1),
            )
            self.atoms.append(atom)

    def read_crest(self, f, conf_name=None):
        """
        conf_name = False to skip conformer loading (doesn't get written until crest job is done)
        """
        if conf_name is None:
            conf_name = os.path.join(
                os.path.dirname(self.name), "crest_conformers.xyz"
            )
        line = True
        self.other["finished"] = False
        self.other["error"] = None
        while line:
            line = f.readline()
            if "terminated normally" in line:
                self.other["finished"] = True
            elif "population of lowest" in line:
                self.other["best_pop"] = float(float_num.findall(line)[0])
            elif "ensemble free energy" in line:
                self.other["free_energy"] = float(float_num.findall(line)[0])
            elif "ensemble entropy" in line:
                self.other["entropy"] = (
                    float(float_num.findall(line)[1]) / 1000
                )
            elif "ensemble average energy" in line:
                self.other["energy"] = float(float_num.findall(line)[0])
            elif "E lowest" in line:
                self.other["best_energy"] = float(float_num.findall(line)[0])
            elif "T /K" in line:
                self.other["temperature"] = float(float_num.findall(line)[0])
            elif (
                line.strip()
                .lower()
                .startswith(("forrtl", "warning", "*warning"))
            ):
                self.other["error"] = "UNKNOWN"
                if "error_msg" not in self.other:
                    self.other["error_msg"] = ""
                self.other["error_msg"] += line
            elif "-chrg" in line:
                self.other["charge"] = int(float_num.findall(line)[0])
            elif "-uhf" in line:
                self.other["multiplicity"] = (
                    int(float_num.findall(line)[0]) + 1
                )

        if self.other["finished"] and conf_name:
            self.other["conformers"] = FileReader(
                conf_name,
                get_all=True,
            ).all_geom
            self.comment, self.atoms = self.other["conformers"][0]
            self.other["conformers"] = self.other["conformers"][1:]

    def read_xtb(self, f, freq_name=None):
        line = True
        self.other["finished"] = False
        self.other["error"] = None
        self.atoms = []
        self.comment = ""
        while line:
            line = f.readline()
            if "Optimized Geometry" in line:
                line = f.readline()
                n_atoms = int(line.strip())
                line = f.readline()
                self.comment = " ".join(line.strip().split()[2:])
                for i in range(n_atoms):
                    line = f.readline()
                    elem, x, y, z = line.split()
                    self.atoms.append(Atom(element=elem, coords=[x, y, z]))
            if "normal termination" in line:
                self.other["finished"] = True
            if "abnormal termination" in line:
                self.other["error"] = "UNKNOWN"
            if line.strip().startswith("#ERROR"):
                if "error_msg" not in self.other:
                    self.other["error_msg"] = ""
                self.other["error_msg"] += line
            if "charge" in line and ":" in line:
                self.other["charge"] = int(float_num.findall(line)[0])
            if "spin" in line and ":" in line:
                self.other["multiplicity"] = (
                    2 * float(float_num.findall(line)[0]) + 1
                )
            if "total energy" in line:
                self.other["energy"] = float(float_num.findall(line)[0])
            if "zero point energy" in line:
                self.other["ZPVE"] = float(float_num.findall(line)[0])
            if "total free energy" in line:
                self.other["free_energy"] = float(float_num.findall(line)[0])
            if "electronic temp." in line:
                self.other["temperature"] = float(float_num.findall(line)[0])
        if freq_name is not None:
            with open(freq_name) as f_freq:
                self.other["frequency"] = Frequency(f_freq.read())

    def read_sqm(self, f):
        lines = f.readlines()
        
        self.other["finished"] = False
        
        self.atoms = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Atomic Charges for Step" in line:
                elements = []
                for info in lines[i + 2:]:
                    if not info.strip() or not info.split()[0].isdigit():
                        break
                    ele = info.split()[1]
                    elements.append(ele)
                i += len(elements) + 2
                
            if "Final Structure" in line:
                k = 0
                for info in lines[i + 4:]:
                    data = info.split()
                    coords = np.array([x for x in data[4:7]])
                    self.atoms.append(
                        Atom(
                            name=str(k + 1),
                            coords=coords,
                            element=elements[k],
                        )
                    )
                    k += 1
                    if k == len(elements):
                        break
                i += k + 4
            
            if "Calculation Completed" in line:
                self.other["finished"] = True
    
            if "Total SCF energy" in line:
                self.other["energy"] = float(line.split()[4]) / UNIT.HART_TO_KCAL

            i += 1
        
        if not self.atoms:
            # there's no atoms if there's an error
            # error is probably on the last line
            self.other["error"] = "UNKNOWN"
            self.other["error_msg"] = line


@addlogger
class Frequency:
    """
    ATTRIBUTES
    :data: Data - contains frequencies, intensities, and normal mode vectors
    :imaginary_frequencies: list(float)
    :real_frequencies: list(float)
    :lowest_frequency: float
    :by_frequency: dict keyed by frequency containing intensities and vectors
        {freq: {intensity: float, vector: np.array}}
    :is_TS: bool - true if len(imaginary_frequencies) == 1, else False
    """

    LOG = None

    class Data:
        """
        ATTRIBUTES
        :frequency: float
        :intensity: float
        :vector: (2D array) normal mode vectors
        :forcek: float
        :symmetry: str
        """

        def __init__(
            self,
            frequency,
            intensity=None,
            vector=None,
            forcek=None,
            symmetry=None,
        ):
            if vector is None:
                vector = []

            self.frequency = frequency
            self.intensity = intensity
            self.symmetry = symmetry
            self.vector = np.array(vector)
            self.forcek = forcek


    class AnharmonicData:
        """
        ATTRIBUTES
        :frequency: float
        :harmonic: Data() or None
        :intensity: float
        :overtones: list(AnharmonicData)
        :combinations: dict(int: AnharmonicData)
        """

        def __init__(
            self,
            frequency,
            intensity,
            harmonic,
        ):
            self.frequency = frequency
            self.harmonic = harmonic
            if harmonic:
                self.delta_anh = frequency - harmonic.frequency
            self.intensity = intensity
            self.overtones = []
            self.combinations = dict()
        
        def __lt__(self, other):
            return self.frequency < other.frequency

        @property
        def harmonic_frequency(self):
            return self.harmonic.frequency

        @property
        def harmonic_intensity(self):
            return self.harmonic.intensity


    def __init__(self, data, hpmodes=None, style="log", harmonic=True):
        """
        :data: should either be a str containing the lines of the output file
            with frequency information, or a list of Data objects
        :hpmodes: required when data is a string
        :form:    required when data is a string; denotes file format (log, out, ...)
        :harmonic: bool, data is for anharmonic frequencies
        """
        self.data = []
        self.anharm_data = None
        self.imaginary_frequencies = None
        self.real_frequencies = None
        self.lowest_frequency = None
        self.by_frequency = {}
        self.is_TS = None

        if data and isinstance(data[0], Frequency.Data):
            self.data = data
            self.sort_frequencies()
            return
        elif data:
            if hpmodes is None:
                raise TypeError(
                    "hpmode argument required when data is a string"
                )
        else:
            return

        lines = data.splitlines()
        num_head = 0
        for line in lines:
            if "Harmonic frequencies" in line:
                num_head += 1
        if hpmodes and num_head != 2:
            self.LOG.warning("Log file damaged, cannot get frequencies")
            return
        
        if harmonic:
            if style == "log":
                self.parse_gaussian_lines(lines, hpmodes)
            elif style == "out":
                self.parse_orca_lines(lines, hpmodes)
            elif style == "dat":
                self.parse_psi4_lines(lines, hpmodes)
            else:
                raise RuntimeError(
                    "no harmonic frequency parser for %s files" % style
                )
        else:
            if style == "log":
                self.parse_gaussian_anharm(lines)
            else:
                raise RuntimeError(
                    "no anharmonic frequency parser for %s files" % style
                )

        self.sort_frequencies()
        return

    def parse_psi4_lines(self, lines, hpmodes):
        """parse lines of psi4 output related to frequencies
        hpmodes is not used"""
        # normal mode info appears in blocks, with up to 3 modes per block
        # at the top is the index of the normal mode
        # next is the frequency in wavenumbers (cm^-1)
        # after a line of '-----' are the normal displacements
        read_displacement = False
        modes = []
        for n, line in enumerate(lines):
            if len(line.strip()) == 0:
                read_displacement = False
                for i, data in enumerate(self.data[-nmodes:]):
                    data.vector = np.array(modes[i])

            elif read_displacement:
                info = [float(x) for x in line.split()[2:]]
                for i, mode in enumerate(modes):
                    mode.append(info[3 * i : 3 * (i + 1)])

            elif line.strip().startswith("Vibration"):
                nmodes = len(line.split()) - 1

            elif line.strip().startswith("Freq"):
                freqs = [float(x) for x in line.split()[2:]]
                for freq in freqs:
                    self.data.append(Frequency.Data(float(freq)))

            elif line.strip().startswith("Force const"):
                force_consts = [float(x) for x in line.split()[3:]]
                for i, data in enumerate(self.data[-nmodes:]):
                    data.forcek = force_consts[i]

            elif line.strip().startswith("Irrep"):
                # sometimes psi4 doesn't identify the irrep of a mode, so we can't
                # use line.split()
                symm = [
                    x.strip() if x.strip() else None
                    for x in [line[31:40], line[51:60], line[71:80]]
                ]
                for i, data in enumerate(self.data[-nmodes:]):
                    data.symmetry = symm[i]

            elif line.strip().startswith("----"):
                read_displacement = True
                modes = [[] for i in range(0, nmodes)]

    def parse_orca_lines(self, lines, hpmodes):
        """parse lines of orca output related to frequency
        hpmodes is not currently used"""
        # vibrational frequencies appear as a list, one per line
        # block column 0 is the index of the mode
        # block column 1 is the frequency in 1/cm
        # skip line one b/c its just "VIBRATIONAL FREQUENCIES" with the way we got the lines
        for n, line in enumerate(lines[1:]):
            if line == "NORMAL MODES":
                break

            freq = line.split()[1]
            self.data += [Frequency.Data(float(freq))]

        # all 3N modes are printed with six modes in each block
        # each column corresponds to one mode
        # the rows of the columns are x_1, y_1, z_1, x_2, y_2, z_2, ...
        displacements = np.zeros((len(self.data), len(self.data)))
        carryover = 0
        start = 0
        stop = 6
        for i, line in enumerate(lines[n + 2 :]):
            if "IR SPECTRUM" in line:
                break

            if i % (len(self.data) + 1) == 0:
                carryover = i // (len(self.data) + 1)
                start = 6 * carryover
                stop = start + 6
                continue

            ndx = (i % (len(self.data) + 1)) - 1
            mode_info = line.split()[1:]

            displacements[ndx][start:stop] = [float(x) for x in mode_info]

        # reshape columns into Nx3 arrays
        for k, data in enumerate(self.data):
            data.vector = np.reshape(
                displacements[:, k], (len(self.data) // 3, 3)
            )

        # purge rotational and translational modes
        n_data = len(self.data)
        k = 0
        while k < n_data:
            if self.data[k].frequency == 0:
                del self.data[k]
                n_data -= 1
            else:
                k += 1

        for k, line in enumerate(lines):
            if line.strip() == "IR SPECTRUM":
                intensity_start = k + 2

        # IR intensities are only printed for vibrational
        # the first column is the index of the mode
        # the second column is the frequency
        # the third is the intensity, which we read next
        for t, line in enumerate(lines[intensity_start:-1]):
            ir_info = line.split()
            inten = float(ir_info[2])
            self.data[t].intensity = inten

    def parse_gaussian_lines(self, lines, hpmodes):
        num_head = 0
        idx = -1
        modes = []
        for k, line in enumerate(lines):
            if "Harmonic frequencies" in line:
                num_head += 1
                if hpmodes and num_head == 2:
                    # if hpmodes, want just the first set of freqs
                    break
                continue
            if "Frequencies" in line and (
                (hpmodes and "---" in line) or ("--" in line and not hpmodes)
            ):
                for i, symm in zip(
                    float_num.findall(line), lines[k - 1].split()
                ):
                    self.data += [Frequency.Data(float(i), symmetry=symm)]
                    modes += [[]]
                    idx += 1
                continue

            if ("Force constants" in line and "---" in line and hpmodes) or (
                "Frc consts" in line and "--" in line and not hpmodes
            ):
                force_constants = float_num.findall(line)
                for i in range(-len(force_constants), 0, 1):
                    self.data[i].forcek = float(force_constants[i])
                continue

            if "IR Inten" in line and (
                (hpmodes and "---" in line) or (not hpmodes and "--" in line)
            ):
                intensities = float_num.findall(line)
                for i in range(-len(force_constants), 0, 1):
                    self.data[i].intensity = float(intensities[i])
                continue

            if hpmodes:
                match = re.search(
                    "^\s+\d+\s+\d+\s+\d+(\s+[+-]?\d+\.\d+)+$", line
                )
                if match is None:
                    continue
                values = float_num.findall(line)
                coord = int(values[0]) - 1
                atom = int(values[1]) - 1
                moves = values[3:]
                for i, m in enumerate(moves):
                    tmp = len(moves) - i
                    mode = modes[-tmp]
                    try:
                        vector = mode[atom]
                    except IndexError:
                        vector = [0, 0, 0]
                        modes[-tmp] += [[]]
                    vector[coord] = m
                    modes[-tmp][atom] = vector
            else:
                match = re.search("^\s+\d+\s+\d+(\s+[+-]?\d+\.\d+)+$", line)
                if match is None:
                    continue
                values = float_num.findall(line)
                atom = int(values[0]) - 1
                moves = np.array(values[2:], dtype=np.float)
                n_moves = len(moves) // 3
                for i in range(-n_moves, 0):
                    modes[i].append(
                        moves[3 * n_moves + 3 * i : 4 * n_moves + 3 * i]
                    )

        for mode, data in zip(modes, self.data):
            data.vector = np.array(mode, dtype=np.float64)
        return

    def parse_gaussian_anharm(self, lines):
        reading_combinations = False
        reading_overtones = False
        reading_fundamentals = False
        
        combinations = []
        overtones = []
        fundamentals = []
        
        mode_re = re.compile("(\d+)\((\d+)\)")
        
        for line in lines:
            if "---" in line or "Mode" in line or not line.strip():
                continue
            if "Fundamental Bands" in line:
                reading_fundamentals = True
                continue
            if "Overtones" in line:
                reading_overtones = True
                continue
            if "Combination Bands" in line:
                reading_combinations = True
                continue

            if reading_combinations:
                info = line.split()
                mode1 = mode_re.search(info[0])
                mode2 = mode_re.search(info[1])
                ndx_1 = int(mode1.group(1))
                exp_1 = int(mode1.group(2))
                ndx_2 = int(mode2.group(1))
                exp_2 = int(mode2.group(2))
                harm_freq = float(info[2])
                anharm_freq = float(info[3])
                anharm_inten = float(info[4])
                harm_inten = 0
                combinations.append(
                    (ndx_1, ndx_2, exp_1, exp_2, anharm_freq,
                    anharm_inten, harm_freq, harm_inten)
                )
            elif reading_overtones:
                info = line.split()
                mode = mode_re.search(info[0])
                ndx = int(mode.group(1))
                exp = int(mode.group(2))
                harm_freq = float(info[1])
                anharm_freq = float(info[2])
                anharm_inten = float(info[3])
                harm_inten = 0
                overtones.append(
                    (ndx, exp, anharm_freq, anharm_inten, harm_freq, harm_inten)
                )
            elif reading_fundamentals:
                info = line.split()
                harm_freq = float(info[1])
                anharm_freq = float(info[2])
                anharm_inten = float(info[4])
                harm_inten = float(info[3])
                fundamentals.append(
                    (anharm_freq, anharm_inten, harm_freq, harm_inten)
                )
        
        self.anharm_data = []
        for i, mode in enumerate(sorted(fundamentals, key=lambda pair: pair[2])):
            self.anharm_data.append(
                self.AnharmonicData(mode[0], mode[1], harmonic=self.data[i])
            )
        for overtone in overtones:
            ndx = len(fundamentals) - overtone[0]
            data = self.anharm_data[ndx]
            harm_data = self.Data(overtone[4], intensity=overtone[5])
            data.overtones.append(
                self.AnharmonicData(overtone[2], overtone[3], harmonic=harm_data)
            )
        for combo in combinations:
            ndx1 = len(fundamentals) - combo[0]
            ndx2 = len(fundamentals) - combo[1]
            data = self.anharm_data[ndx1]
            harm_data = self.Data(combo[6], intensity=combo[7])
            data.combinations[ndx2] = [
                self.AnharmonicData(combo[4], combo[5], harmonic=harm_data)
            ]

    def sort_frequencies(self):
        self.imaginary_frequencies = []
        self.real_frequencies = []
        for i, data in enumerate(self.data):
            freq = data.frequency
            if freq < 0:
                self.imaginary_frequencies += [freq]
            elif freq > 0:
                self.real_frequencies += [freq]
            self.by_frequency[freq] = {
                "intensity": data.intensity,
                "vector": data.vector,
            }
        if len(self.data) > 0:
            self.lowest_frequency = self.data[0].frequency
        else:
            self.lowest_frequency = None
        self.is_TS = True if len(self.imaginary_frequencies) == 1 else False

    def get_ir_data(
            self,
            point_spacing=None,
            fwhm=15.0,
            plot_type="transmittance",
            peak_type="pseudo-voigt",
            voigt_mixing=0.5,
            linear_scale=0.0,
            quadratic_scale=0.0,
            anharmonic=False,
    ):
        """
        returns arrays of x_values, y_values for an IR plot
        point_spacing - spacing between points, default is higher resolution around
                        each peak (i.e. not uniform)
                        this is pointless if peak_type == delta
        fwhm - full width at half max in 1/cm
        plot_type - transmittance or absorbance
        peak_type - pseudo-voigt, gaussian, lorentzian, or delta
        voigt_mixing - fraction of pseudo-voigt that is gaussian
        linear_scale - subtract linear_scale * frequency off each mode
        quadratic_scale - subtract quadratic_scale * frequency^2 off each mode
        """
        # scale frequencies
        if anharmonic and self.anharm_data:
            frequencies = []
            intensities = []
            for data in self.anharm_data:
                frequencies.append(data.frequency)
                intensities.append(data.intensity)
                for overtone in data.overtones:
                    frequencies.append(overtone.frequency)
                    intensities.append(overtone.intensity)
                for key in data.combinations:
                    for combo in data.combinations[key]:
                        frequencies.append(combo.frequency)
                        intensities.append(combo.intensity)
            frequencies = np.array(frequencies)
            intensities = np.array(intensities)
        else:
            if anharmonic:
                self.LOG.warning(
                    "plot of anharmonic frequencies requested but no anharmonic data"
                    "is present"
                )
            frequencies = np.array(
                [freq.frequency for freq in self.data if freq.frequency > 0]
            )
            intensities = [
                freq.intensity for freq in self.data if freq.frequency > 0
            ]

        frequencies -= linear_scale * frequencies + quadratic_scale * frequencies ** 2

        if point_spacing:
            x_values = []
            x = -point_spacing
            stop = max(frequencies)
            if peak_type.lower() != "delta":
                stop += 5 * fwhm
            while x < stop:
                x += point_spacing
                x_values.append(x)
            
            x_values = np.array(x_values)
        
        e_factor = -4 * np.log(2) / fwhm ** 2
        
        if peak_type.lower() != "delta":
            # get a list of functions
            # we'll evaluate these at each x point later
            functions = []
            if not point_spacing:
                x_values = np.linspace(0, max(frequencies) - 10 * fwhm, num=100).tolist()
            
            for freq, intensity in zip(frequencies, intensities):
                if intensity is not None:
                    if not point_spacing:
                        x_values.extend(
                            np.linspace(
                                max(freq - (3.5 * fwhm), 0), 
                                freq + (3.5 * fwhm), 
                                num=65,
                            ).tolist()
                        )
                        x_values.append(freq)
                    
                    if peak_type.lower() == "gaussian":
                        functions.append(
                            lambda x, x0=freq, inten=intensity: inten * np.exp(e_factor * (x - x0) ** 2)
                        )
        
                    elif peak_type.lower() == "lorentzian":
                        functions.append(
                            lambda x, x0=freq, inten=intensity: inten * 0.5 * (0.5 * fwhm / ((x - x0) ** 2 + (0.5 * fwhm) ** 2))
                        )
                    
                    elif peak_type.lower() == "pseudo-voigt":
                        functions.append(
                            lambda x, x0=freq, inten=intensity:
                                inten * (
                                    (1 - voigt_mixing) * 0.5 * (0.5 * fwhm / ((x - x0)**2 + (0.5 * fwhm)**2)) + 
                                    voigt_mixing * np.exp(e_factor * (x - x0)**2)
                                )
                        )
            
            if not point_spacing:
                x_values = np.array(list(set(x_values)))
                x_values.sort()
        
            y_values = np.sum([f(x_values) for f in functions], axis=0)
        
        else:
            x_values = []
            y_values = []

            for freq, intensity in zip(frequencies, intensities):
                if intensity is not None:
                    y_values.append(intensity)
                    x_values.append(freq)

            y_values = np.array(y_values)

        if len(y_values) == 0:
            self.LOG.warning("nothing to plot")
            return None

        y_values /= np.amax(y_values)


        if plot_type.lower() == "transmittance":
            y_values = np.array([10 ** (2 - y) for y in y_values])

        return x_values, y_values
  
    def plot_ir(
            self,
            figure,
            centers=None,
            widths=None,
            exp_data=None,
            plot_type="transmittance",
            peak_type="pseudo-voigt",
            reverse_x=True,
            **kwargs,
    ):
        """
        plot IR data on figure
        figure - matplotlib figure
        centers - array-like of float, plot is split into sections centered
                  on the frequency specified by centers
                  default is to not split into sections
        widths - array-like of float, defines the width of each section
        exp_data - other data to plot
                   should be a list of (x_data, y_data, color)
        reverse_x - if True, 0 cm^-1 will be on the right
        plot_type - see Frequency.get_ir_data
        peak_type - any value allowed by Frequency.get_ir_data
        kwargs - keywords for Frequency.get_ir_data
        """

        data = self.get_ir_data(
            plot_type=plot_type,
            peak_type=peak_type,
            **kwargs
        )
        if data is None:
            return
        
        x_values, y_values = data

        if not centers:
            # if no centers were specified, pretend they were so we
            # can do everything the same way
            axes = [figure.subplots(nrows=1, ncols=1)]
            widths = [max(x_values)]
            centers = [max(x_values) / 2]
        else:
            n_sections = len(centers)
            figure.subplots_adjust(wspace=0.05)
            # sort the sections so we don't jump around
            widths = [x for _, x in sorted(
                zip(centers, widths),
                key=lambda p: p[0],
                reverse=reverse_x,
            )]
            centers = sorted(centers, reverse=reverse_x)
            
            axes = figure.subplots(
                nrows=1,
                ncols=n_sections,
                sharey=True,
                gridspec_kw={'width_ratios': widths},
            )
            if not hasattr(axes, "__iter__"):
                # only one section was specified (e.g. zooming in on a peak)
                # make sure axes is iterable
                axes = [axes]
        
        for i, ax in enumerate(axes):
            if i == 0:
                if plot_type.lower() == "transmittance":
                    ax.set_ylabel("Transmittance (%)")
                else:
                    ax.set_ylabel("Absorbance (arb.)")
                
                # need to split plot into sections
                # put a / on the border at the top and bottom borders
                # of the plot
                if len(axes) > 1:
                    ax.spines["right"].set_visible(False)
                    ax.tick_params(labelright=False, right=False)
                    ax.plot(
                        [1, 1],
                        [0, 1],
                        marker=((-1, -1), (1, 1)),
                        markersize=5,
                        linestyle='none',
                        color='k',
                        mec='k',
                        mew=1,
                        clip_on=False,
                        transform=ax.transAxes,
                    )

            elif i == len(axes) - 1 and len(axes) > 1:
                # last section needs a set of / too, but on the left side
                ax.spines["left"].set_visible(False)
                ax.tick_params(labelleft=False, left=False)
                ax.plot(
                    [0, 0],
                    [0, 1],
                    marker=((-1, -1), (1, 1)),
                    markersize=5,
                    linestyle='none',
                    color='k',
                    mec='k',
                    mew=1,
                    clip_on=False,
                    transform=ax.transAxes,
                )

            elif len(axes) > 1:
                # middle sections need two sets of /
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.tick_params(labelleft=False, labelright=False, left=False, right=False)
                ax.plot(
                    [0, 0],
                    [0, 1],
                    marker=((-1, -1), (1, 1)),
                    markersize=5,
                    linestyle='none',
                    label="Silence Between Two Subplots",
                    color='k',
                    mec='k',
                    mew=1,
                    clip_on=False,
                    transform=ax.transAxes,
                )
                ax.plot(
                    [1, 1],
                    [0, 1],
                    marker=((-1, -1), (1, 1)),
                    markersize=5,
                    label="Silence Between Two Subplots",
                    linestyle='none',
                    color='k',
                    mec='k',
                    mew=1,
                    clip_on=False,
                    transform=ax.transAxes,
                )

            if peak_type.lower() != "delta":
                ax.plot(
                    x_values,
                    y_values,
                    color='k',
                    linewidth=0.5,
                    label="computed",
                )

            else:
                if plot_type.lower() == "transmittance":
                    ax.vlines(
                        x_values,
                        y_values,
                        [100 for y in y_values],
                        linewidth=0.5,
                        colors=['k' for x in x_values],
                        label="computed"
                    )
                    ax.hlines(
                        100,
                        0,
                        max(4000, *x_values),
                        linewidth=0.5,
                        colors=['k' for y in y_values],
                        label="computed",
                    )
                
                else:
                    ax.vlines(
                        x_values,
                        [0 for y in y_values],
                        y_values,
                        linewidth=0.5,
                        colors=['k' for x in x_values],
                        label="computed"
                    )
                    ax.hlines(
                        0,
                        0,
                        max(4000, *x_values),
                        linewidth=0.5,
                        colors=['k' for y in y_values],
                        label="computed"
                    )

            if exp_data:
                for x, y, color in exp_data:
                    ax.plot(x, y, color=color, zorder=-1, linewidth=0.5, label="observed")

            center = centers[i]
            width = widths[i]
            high = center + width / 2
            low = center - width / 2
            if reverse_x:
                ax.set_xlim(high, low)
            else:
                ax.set_xlim(low, high)
        
        # b/c we're doing things in sections, we can't add an x-axis label
        # well we could, but which section would be put it one?
        # it wouldn't be centered
        # so instead the x-axis label is this
        figure.text(0.5, 0.0, r"wavenumber (cm$^{-1}$)" , ha="center", va="bottom")
