"""For parsing input/output files"""
import os
import re
from copy import deepcopy
from io import IOBase, StringIO
from warnings import warn

import numpy as np

from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS, PHYSICAL, UNIT

read_types = ["xyz", "log", "com", "sd", "out", "dat"]
write_types = ["xyz", "com"]
file_type_err = "File type not yet implemented: {}"
float_num = re.compile("[-+]?\d+\.?\d*")
NORM_FINISH = "Normal termination"
ORCA_NORM_FINISH = "****ORCA TERMINATED NORMALLY****"
PSI4_NORM_FINISH = "*** Psi4 exiting successfully. Buy a developer a beer!"
ERRORS = {
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
    "galloc: could not allocate memory": "GALLOC",
    "Error imposing constraints": "CONSTR",
    "End of file reading basis center.": "BASIS",
    "Unrecognized atomic symbol": "ATOM",
    "malloc failed.": "MEM",
    "Unknown message": "UNKNOWN",
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
        cls, geom, style="xyz", append=False, outfile=None, *args, **kwargs
    ):
        """
        Writes file from geometry in the specified style

        :geom: the Geometry to use
        :style: the file type style to generate
            Currently supported options: xyz, com
        :append: for *.xyz, append geometry to the same file
        :options: for *.com files, the computational options
        """
        if style.lower() not in write_types:
            raise NotImplementedError(file_type_err.format(style))

        if outfile is None and \
            os.path.dirname(geom.name) and not os.access(
            os.path.dirname(geom.name), os.W_OK
        ):
            os.makedirs(os.path.dirname(geom.name))
        if style.lower() == "xyz":
            out = cls.write_xyz(geom, append, outfile)
        elif style.lower() == "com":
            if "theory" in kwargs and "step" in kwargs:
                step = kwargs["step"]
                theory = kwargs["theory"]
                del kwargs["step"]
                del kwargs["theory"]
                out = cls.write_com(geom, step, theory, outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing com files, **kwargs must include: theory=Aaron.Theory(), step=int/float()"
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
    def write_com(cls, geom, step, theory, outfile=None, **kwargs):
        # atom specs need flag column before coords if any atoms frozen
        has_frozen = False
        fmt = "{:<3s}" + " {:> 12.6f}" * 3 + "\n"
        for atom in geom:
            if atom.flag:
                fmt = "{:<3s}  {:> 2d}" + " {:> 12.6f}" * 3 + "\n"
                has_frozen = True
                break

        # get file content string
        s = theory.make_header(geom, step, **kwargs)
        for atom in geom.atoms:
            if has_frozen:
                s += fmt.format(atom.element, -atom.flag, *atom.coords)
            else:
                s += fmt.format(atom.element, *atom.coords)
        s += theory.make_footer(geom, step)

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            fname = "{}.{}.com".format(geom.name, step2str(step))
            with open(fname, "w") as f:
                f.write(s)
        elif outfile is False:
            return s
        else:
            with open(outfile, "w") as f:
                f.write(s)

        return


class FileReader:
    """
    Attributes:
        name ''
        file_type ''
        comment ''
        atoms [Atom]
        other {}
    """

    def __init__(self, fname, get_all=False, just_geom=True):
        """
        :fname: either a string specifying the file name of the file to read
            or a tuple of (str(name), str(file_type), str(content))
        :get_all: if true, optimization steps are  also saved in
            self.other['all_geom']; otherwise only saves last geometry
        :just_geom: if true, does not store other information, such as
            frequencies, only what is needed to construct a Geometry() obj
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
            fname = fname.rsplit(".", 1)
            self.name = fname[0]
            self.file_type = fname[1].lower()
        elif isinstance(fname, (tuple, list)):
            self.name = fname[0]
            self.file_type = fname[1]
            self.content = fname[2]
        if self.file_type not in read_types:
            raise NotImplementedError(file_type_err.format(self.file_type))

        # Fill in attributes with geometry information
        if self.content is None:
            self.read_file(get_all, just_geom)
        elif isinstance(self.content, str):
            f = StringIO(self.content)
        elif isinstance(self.content, IOBase):
            f = self.content

        if self.content is not None:
            if self.file_type == "log":
                self.read_log(f, get_all, just_geom)
            elif self.file_type == "sd":
                self.read_sd(f)
            elif self.file_type == "xyz":
                self.read_xyz(f, get_all)
            elif self.file_type == "com":
                self.read_com(f)
            elif self.file_type == "out":
                self.read_orca_out(f, get_all, just_geom)
            elif self.file_type == "dat":
                self.read_psi4_out(f, get_all, just_geom)

    def read_file(self, get_all=False, just_geom=True):
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
        elif self.file_type == "com":
            self.read_com(f)
        elif self.file_type == "sd":
            self.read_sd(f)
        elif self.file_type == "out":
            self.read_orca_out(f, get_all, just_geom)
        elif self.file_type == "dat":
            self.read_psi4_out(f, get_all, just_geom)

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
        self.comment = lines[0].strip()
        counts = lines[3].split()
        natoms = int(counts[0])
        nbonds = int(counts[1])
        self.atoms = []
        for line in lines[4 : 4 + natoms]:
            atom_info = line.split()
            self.atoms += [Atom(element=atom_info[3], coords=atom_info[0:3])]

        for i, a in enumerate(self.atoms):
            a.name = str(i + 1)

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
                coords = np.array([float(x) for x in atom_info[1:-1]])
                rv += [Atom(element=element, coords=coords, name=str(i))]
                mass += float(atom_info[-1])

                line = f.readline()
                n += 1

            return rv, mass, n

        line = f.readline()
        n = 1
        while line != "":
            if line.startswith('    Geometry (in Angstrom), charge'):
                if not just_geom:
                    self.other['charge'] = int(line.split()[5].strip(','))
                    self.other['multiplicity'] = int(line.split()[8].strip(':'))

            elif line.strip().startswith('Center'):
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
                if line.strip().startswith('Current energy'):
                    self.other["energy"] = float(line.split()[-1])

                elif line.strip().startswith('Total E0'):
                    self.other["energy"] = float(line.split()[-2])

                elif line.strip().startswith('Correction ZPE'):
                    self.other["ZPVE"] = float(line.split()[-4])

                elif line.strip().startswith('Total ZPE'):
                    self.other['E_ZPVE'] = float(line.split()[-2])

                elif line.strip().startswith('Total H, Enthalpy'):
                    self.other['enthalpy'] = float(line.split()[-2])

                elif line.strip().startswith('Total G, Free'):
                    self.other['free_energy'] = float(line.split()[-2])
                    self.other['temperature'] = float(line.split()[-4])

                elif 'symmetry no. =' in line:
                    self.other['rotational_symmetry_number'] = int(
                        line.split()[-1].strip(')')
                    )

                elif line.strip().startswith('Rotational constants:') and \
                         line.strip().endswith('[cm^-1]') and \
                         'rotational_temperature' not in self.other:
                    self.other["rotational_temperature"] = [
                        float(x) for x in line.split()[-8:-1:3]
                    ]
                    self.other["rotational_temperature"] = [
                        x
                        * PHYSICAL.SPEED_OF_LIGHT
                        * PHYSICAL.PLANK
                        / PHYSICAL.KB
                        for x in self.other["rotational_temperature"]
                    ]

                elif line.startswith('  Vibration '):
                    freq_str = ""
                    while not line.strip().startswith('=='):
                        freq_str += line
                        line = f.readline()
                        n += 1

                    self.other["frequency"] = Frequency(
                        freq_str, hpmodes=False, form="dat"
                    )

                elif PSI4_NORM_FINISH in line:
                    self.other["finished"] = True
                
                line = f.readline()
                n += 1

    def read_orca_out(self, f, get_all=False, just_geom=True):
        """read orca output file"""

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
                if line.startswith("FINAL SINGLE POINT ENERGY"):
                    self.other["energy"] = float(line.split()[-1])

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
                        freq_str, hpmodes=False, form="out"
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

                elif "rotational symmetry number" in line.strip():
                    # TODO: make this cleaner
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-2]
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
                        * PHYSICAL.PLANK
                        / PHYSICAL.KB
                        for x in self.other["rotational_temperature"]
                    ]

                elif "Symmetry Number" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1]
                    )

                elif "sn is the rotational symmetry number" in line:
                    # older versions of orca print this differently
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-2]
                    )

                elif ORCA_NORM_FINISH in line:
                    self.other["finished"] = True

                # TODO E_ZPVE
                # TODO gradient
                # TODO error

                line = f.readline()
                n += 1

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

        self.all_geom = []
        line = f.readline()
        self.other["archive"] = ""
        found_archive = False
        n = 1
        while line != "":
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
            if just_geom:
                line = f.readline()
                n += 1
                continue
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
                self.other["energy"] = float(line.split()[4])
            if "Molecular mass:" in line:
                self.other["mass"] = float(float_num.search(line).group(0))
                self.other["mass"] *= UNIT.AMU_TO_KG

            # Frequencies
            if "hpmodes" in line:
                self.other["hpmodes"] = True
            if "Harmonic frequencies" in line:
                freq_str = line
                while line != "\n":
                    line = f.readline()
                    n += 1
                    freq_str += line
                if "hpmodes" not in self.other:
                    self.other["hpmodes"] = False
                self.other["frequency"] = Frequency(
                    freq_str, self.other["hpmodes"]
                )

            # Thermo
            if "Temperature" in line:
                self.other["temperature"] = float(
                    float_num.search(line).group(0)
                )
            if "Rotational constants (GHZ):" in line:
                rot = float_num.findall(line)
                rot = [
                    float(r) * PHYSICAL.PLANK * (10 ** 9) / PHYSICAL.KB
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

            # capture errors
            # only keep first error, want to fix one at a time
            if "error" not in self.other:
                for err in ERRORS:
                    if err in line:
                        self.other["error"] = ERRORS[err]
                        self.other["error_msg"] = line.strip()
                        break

            line = f.readline()
            n += 1

        for i, a in enumerate(self.atoms):
            a.name = str(i + 1)

        if "finished" not in self.other:
            self.other["finished"] = False
        if "error" not in self.other:
            self.other["error"] = None
        if not self.other["finished"] and not self.other["error"]:
            self.other["error"] = ERRORS["Unknown message"]
            self.other["error_msg"] = "Unknown message"
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
                    #solvent model should be non-greedy b/c solvent name can have commas
                    other["solvent_model"] = re.search(
                        "scrf=\((\S+?),", line
                    ).group(1)
                if "EmpiricalDispersion=" in line:
                    other["emp_dispersion"] = re.search(
                        "EmpiricalDispersion=(\S+)", line
                    ).group(1)
                if "int=(grid" in line or "integral=(grid" in line.lower():
                    other["grid"] = re.search("(?:int||Integral)=\(grid[(=](\S+?)\)", line).group(1)
                #comments can be multiple lines long
                #but there should be a blank line between the route and the comment
                #and another between the comment and the charge+mult
                blank_lines = 0
                while blank_lines < 2:
                    line = f.readline().strip()
                    if len(line) == 0:
                        blank_lines += 1
                    else:
                        if 'comment' not in other:
                            other['comment'] = ""

                        other['comment'] += "%s\n" % line
                
                other['comment'] = other['comment'].strip()
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

    class Data:
        """
        ATTRIBUTES
        :frequency: float
        :intensity: float
        :vector: (2D array) normal mode vectors
        """

        def __init__(self, frequency, intensity=None, vector=[], forcek=[]):
            self.frequency = frequency
            self.intensity = intensity
            self.vector = np.array(vector)
            self.forcek = np.array(forcek)

    def __init__(self, data, hpmodes=None, form="log"):
        """
        :data: should either be a str containing the lines of the output file
            with frequency information, or a list of Data objects
        :hpmodes: required when data is a string
        :form:    required when data is a string; denotes file format (log, out, ...)
        """
        self.data = []
        self.imaginary_frequencies = None
        self.real_frequencies = None
        self.lowest_frequency = None
        self.by_frequency = {}
        self.is_TS = None

        if isinstance(data[0], Frequency.Data):
            self.data = data
            self.sort_frequencies()
            return
        else:
            if hpmodes is None:
                raise TypeError(
                    "hpmode argument required when data is a string"
                )

        lines = data.split("\n")
        num_head = 0
        for line in lines:
            if "Harmonic frequencies" in line:
                num_head += 1
        if hpmodes and num_head != 2:
            warn("Log file damaged, cannot get frequencies")
            return
        if form == "log":
            self.parse_lines(lines, hpmodes)
        elif form == "out":
            self.parse_orca_lines(lines, hpmodes)
        elif form == "dat":
            self.parse_psi4_lines(lines, hpmodes)
        else:
            raise RuntimeError("no frequency parser for %s files" % form)

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
                    mode.append(info[3*i:3*(i+1)])

            elif line.strip().startswith('Vibration'):
                nmodes = len(line.split()) - 1

            elif line.strip().startswith('Freq'):
                freqs = [float(x) for x in line.split()[2:]]
                for freq in freqs:
                    self.data.append(Frequency.Data(float(freq)))

            elif line.strip().startswith('Force const'):
                force_consts = [float(x) for x in line.split()[3:]]
                for i, data in enumerate(self.data[-nmodes:]):
                    data.forcek = force_consts[i]

            elif line.strip().startswith('----'):
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

        # IR intensities are only printed for vibrational
        # the first column is the index of the mode
        # the second column is the frequency
        # the third is the intensity, which we read next
        for t, line in enumerate(lines[n + 2 + i + carryover + 1 : -1]):
            ir_info = line.split()
            inten = float(ir_info[2])
            self.data[t].intensity = inten

    def parse_lines(self, lines, hpmodes):
        num_head = 0
        idx = -1
        modes = []
        for line in lines:
            if "Harmonic frequencies" in line:
                num_head += 1
                if hpmodes and num_head == 2:
                    # if hpmodes, want just the first set of freqs
                    break
                continue
            if "Frequencies" in line and (
                (hpmodes and "---" in line) or ("--" in line and not hpmodes)
            ):
                for i in float_num.findall(line):
                    self.data += [Frequency.Data(float(i))]
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
                for i in float_num.findall(line):
                    self.data[idx].intensity = float(i)
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
            data.vector = np.array(mode, dtype=np.float)
        return

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
        self.lowest_frequency = self.data[0].frequency
        self.is_TS = True if len(self.imaginary_frequencies) == 1 else False
