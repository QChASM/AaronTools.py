"""For parsing input/output files"""
import re
import os
from copy import deepcopy
from io import StringIO, IOBase
from warnings import warn

import numpy as np

from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS, PHYSICAL, UNIT

read_types = ["xyz", "log", "com", "sd"]
write_types = ["xyz", "com"]
file_type_err = "File type not yet implemented: {}"
float_num = re.compile("[-+]?\d+\.?\d*")
NORM_FINISH = "Normal termination"
ERRORS = {
    "Convergence failure -- run terminated.": "CONV",
    "Inaccurate quadrature in CalDSu": "CONV_CDS",
    "Error termination request processed by link 9999": "CONV_LINK",
    "FormBX had a problem": "FBX",
    "NtrErr Called from FileIO": "CHK",
    "Wrong number of Negative eigenvalues": "EIGEN",
    "Erroneous write": "QUOTA",
    "Atoms too close": "CLASH",
    "The combination of multiplicity": "CHARGEMULT",
    "Bend failed for angle": "REDUND",
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
    def write_file(cls, geom, style="xyz", append=False, outfile=None, *args, **kwargs):
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
            #if no output file is specified, use the name of the geometry
            with open(geom.name + ".xyz", mode) as f:
                f.write(s)
        elif outfile is False:
            #if no output file is desired, just return the file contents
            return s.strip()
        else:
            #write output to the requested destination
            with open(outfile, mode) as f:
                f.write(s)

        return

    @classmethod
    def write_com(cls, geom, step, theory, outfile=None, **kwargs):
        has_frozen = False
        fmt = "{:<3s}" + " {:> 12.6f}" * 3 + "\n"
        for atom in geom.atoms:
            if atom.flag:
                fmt = "{:<3s}  {:> 2d}" + " {:> 12.6f}" * 3 + "\n"
                has_frozen = True
                break

            s = theory.make_header(geom, step, **kwargs)
            for atom in geom.atoms:
                if has_frozen:
                    s += fmt.format(a.element, -a.flag, *a.coords)
                else:
                    s += fmt.format(a.element, *a.coords)

            s += theory.make_footer(geom, step)

        if outfile is None:
            #if outfile is not specified, name file in Aaron format
            fname = "{}.{}.com".format(geom.name, step2str(step))
            with open(fname, "w") as f:
                f.write(s)
        elif outfile is False:
            return s
        else:
            with open(outfile, 'w') as f:
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
            fname = '.'.join([self.name, self.file_type])
            if os.path.isfile(fname):
                f = open(fname)
            else:
                raise FileNotFoundError("Error while looking for %s: could not find %s or %s in %s" % \
                        (self.name, fname, self.name, os.getcwd()))

        if self.file_type == "xyz":
            self.read_xyz(f, get_all)
        elif self.file_type == "log":
            self.read_log(f, get_all, just_geom)
        elif self.file_type == "com":
            self.read_com(f)
        elif self.file_type == "sd":
            self.read_sd(f)

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
        for line in lines[4:4+natoms]:
            atom_info = line.split()
            self.atoms += [Atom(element=atom_info[3], coords = atom_info[0:3])]

        for i, a in enumerate(self.atoms):
            a.name = str(i + 1)

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
                    self.all_geom += [deepcopy(self.atoms)]
                self.atoms, n = get_atoms(f, n)
            if just_geom:
                line = f.readline()
                n += 1
                continue
            if "Symbolic Z-matrix:" in line:
                line = f.readline()
                n += 1
                match = re.search(
                    "Charge\s*=\s*(\d+)\s*Multiplicity\s*=\s*(\d+)", line
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
            for err in ERRORS:
                if err in line:
                    self.other["error"] = ERRORS[err]
                    self.other["error_msg"] = line.strip()
                    break
            else:
                self.other["error"] = None

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
                #route can be #n functional/basis ...
                #or #functional/basis ...
                #or # functional/basis ...
                if method.group(3):
                    other['method'] = method.group(3)
                else:
                    other['method'] = method.group(2)
                if "temperature=" in line:
                    other["temperature"] = re.search(
                        "temperature=(\d+\.?\d*)", line
                    ).group(1)
                if "solvent=" in line:
                    other["solvent"] = re.search(
                        "solvent=(\S+)\)", line
                    ).group(1)
                if "scrf=" in line:
                    other["solvent_model"] = re.search(
                        "scrf=\((\S+),", line
                    ).group(1)
                if "EmpiricalDispersion=" in line:
                    other["emp_dispersion"] = re.search(
                        "EmpiricalDispersion=(\S+)", line
                    ).group(1)
                if "int=(grid(" in line:
                    other["grid"] = re.search("int=\(grid(\S+)", line).group(1)
                for _ in range(4):
                    line = f.readline()
                line = line.split()
                other["charge"] = line[0]
                other["mult"] = line[1]
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

    def __init__(self, data, hpmodes=None):
        """
        :data: should either be a str containing the lines of the output file
            with frequency information, or a list of Data objects
        :hpmodes: required when data is a string
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
        self.parse_lines(lines, hpmodes)
        self.sort_frequencies()
        return

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
            if "Frequencies" in line and ((hpmodes and "---" in line) or \
                ("--" in line and not hpmodes)):
                for i in float_num.findall(line):
                    self.data += [Frequency.Data(float(i))]
                    modes += [[]]
                    idx += 1
                continue

            if ("Force constants" in line and "---" in line and hpmodes) or \
                ("Frc consts" in line and "--" in line and not hpmodes):
                force_constants = float_num.findall(line)
                for i in range(-len(force_constants), 0, 1):
                    self.data[i].forcek = float(force_constants[i])
                continue

            if "IR Inten" in line and ((hpmodes and "---" in line) or \
                (not hpmodes and "--" in line)):
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
                    modes[i].append(moves[3*n_moves+3*i:4*n_moves+3*i])

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
