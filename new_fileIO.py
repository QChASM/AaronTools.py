import re

import cclib

from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS, PHYSICAL, UNIT
from AaronTools.utils.utils import is_alpha, is_int

float_num = re.compile("[-+]?\d+\.?\d*")
READTYPES = ["XYZ", "Gaussian"]
WRITETYPES = ["XYZ", "Gaussian"]
NORM_FINISH = "Normal termination"
ERRORS = {
    "NtrErr Called from FileIO": "CHK",  # delete
    "Wrong number of Negative eigenvalues": "EIGEN",  # opt=noeigen
    "Convergence failure -- run terminated.": "CONV",  # scf=xqc
    # check quota and alert user; REMOVE error from end of file!
    "Erroneous write": "QUOTA",
    "Atoms too close": "CLASH",  # flag as CLASH
    # die and alert user to check catalyst structure or fix input file
    "The combination of multiplicity": "CHARGEMULT",
    "Bend failed for angle": "REDUND",  # Using opt=cartesian
    "Unknown message": "UNKNOWN",
}


class FileReader:
    """
    :name: file name without extension
    :file_ext: file extension
    :file_type:
    :comment:
    :atoms: list of Atom()
    :all_geom: list of [Atom()] if optimization steps requested
    :other: dict() containing additional info
    """

    def __init__(self, filename, get_all=False, just_geom=True):
        """
        :filename: a file name or a tuple(file_name, file_extension, IOstream)
        :get_all: if true, optimization steps are  also saved in
            self.other['all_geom']; otherwise only saves last geometry
        :just_geom: if true, does not store other information, such as
            frequencies, only what is needed to construct a Geometry() obj
        """
        if isinstance(filename, str):
            self.name, self.file_ext = filename.rsplit(".", 1)
        else:
            self.name, self.file_ext = filename[:2]
            filename = filename[2]
        self.file_type = ""
        self.comment = ""
        self.atoms = []
        self.all_geom = None
        self.other = {}

        try:
            parser = cclib.io.ccopen(filename)
            data = parser.parse()
            self.file_type = str(parser).split()[0].split(".")[-1]
            self.other = data.__dict__
        except AttributeError:
            if self.file_ext == "com":
                self.file_type = "Gaussian"
                self.read_com(filename)
                return

        for key in self.other:
            print(key)
            print(self.other[key])
        for i, (n, c) in enumerate(zip(data.atomnos, data.atomcoords[-1])):
            self.atoms += [Atom(element=ELEMENTS[n], coords=c, name=i + 1)]
        if len(data.atomcoords) == 1:
            # if > 1, there are more geometries to handle
            del self.other["atomnos"]
            del self.other["atomcoords"]
        elif get_all:
            # only handle them if get_all is true
            self.all_geom = []
            for i, coords in enumerate(data.atomcoords[:-1]):
                atoms = []
                for j, (n, c) in enumerate(zip(data.atomnos, coords)):
                    self.atoms += [
                        Atom(element=ELEMENTS[n], coords=c, name=j + 1)
                    ]
                self.all_geom += [atoms]
        # cclib doesn't store XYZ file comments
        if self.file_type == "XYZ":
            self.read_xyz(filename)
        # Grab things cclib doesn't from log files
        if self.file_type == "Gaussian" and self.file_ext == "log":
            self.read_log(filename)
        # fix naming conventions
        self.fix_names()
        return

    def read_log(self, filename):
        # Grab things cclib doesn't from log files
        if not self.other["metadata"]["success"]:
            if isinstance(filename, str):
                f = open(filename)
            else:
                f = filename
            for line in f:
                if "Molecular mass" in line:
                    self.other["mass"] = float(float_num.search(line).group(0))
                    self.other["mass"] *= UNIT.AMU_TO_KG
                if "Rotational constants (GHZ):" in line:
                    rot = float_num.findall(line)
                    rot = [
                        float(r) * PHYSICAL.PLANCK * (10 ** 9) / PHYSICAL.KB
                        for r in rot
                    ]
                    self.other["rotational_temperature"] = rot

    def read_xyz(self, filename):
        if isinstance(filename, str):
            f = open(filename)
        else:
            f = filename
        f.readline()
        self.comment = f.readline().strip()
        f.close()

    def read_com(self, filename):
        if isinstance(filename, str):
            f = open(filename)
        else:
            f = filename

        atoms = []
        other = {}
        found_atoms = False
        found_constraint = False
        for line in f:
            # header
            if line.startswith("%"):
                # checkfile spec
                other["checkfile"] = line.strip().split("=")[1]
                continue
            if line.startswith("#"):
                match = re.search("^#(\S+)", line).group(1)
                other["method"] = match.split("/")[0]
                other["basis"] = match.split("/")[1]
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
                        "EmpiricalDispersion=(\s+)", line
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

    def fix_names(self):
        if "metadata" in self.other:
            if "success" in self.other["metadata"]:
                self.other["finished"] = self.other["metadata"]["success"]
