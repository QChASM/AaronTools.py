"""For parsing input/output files"""
import re
import json
from io import StringIO
from warnings import warn
from copy import deepcopy

from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS, UNIT, PHYSICAL


read_types = ['xyz', 'log', 'com']
write_types = ['xyz', 'log']
file_type_err = "File type not yet implemented: {}"
float_num = re.compile("[-+]?\d+\.?\d*")
NORM_FINISH = "Normal termination"
ERRORS = {
    "NtrErr Called from FileIO": 'CHK',  # delete
    "Wrong number of Negative eigenvalues": 'EIGEN',  # opt=noeigen
    "Convergence failure -- run terminated.": 'CONV',  # scf=xqc
    # check quota and alert user; REMOVE error from end of file!
    "Erroneous write": 'QUOTA',
    "Atoms too close": 'CLASH',  # flag as CLASH
    # die and alert user to check catalyst structure or fix input file
    "The combination of multiplicity": 'CHARGEMULT',
    "Bend failed for angle": 'REDUND',  # Using opt=cartesian
    "Unknown message": 'UNKNOWN',
}


def is_alpha(test):
    rv = re.search('^[a-zA-Z]+$', test)
    return bool(rv)


def is_int(test):
    rv = re.search('^[+-]?\d+$', test)
    return bool(rv)


def is_num(test):
    rv = re.search('^[+-]?\d+\.?\d*', test)
    return bool(rv)


def write_file(geom, style='xyz'):
    if style.lower() not in write_types:
        raise NotImplementedError(file_type_err.format(style))

    def write_xyz(geom):
        with open(geom.name + ".xyz", 'w') as f:
            f.write(str(len(geom.atoms)) + "\n")
            f.write(geom.comment + "\n")
            for a in geom.atoms:
                s = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f}\n"
                f.write(s.format(a.element, *a.coords))
        return

    if style.lower() == 'xyz':
        write_xyz(geom)


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
        self.comment = ''
        self.atoms = []
        self.other = {}
        self.content = None

        # get file name and extention
        if isinstance(fname, str):
            fname = fname.rsplit('.', 1)
            self.name = fname[0]
            self.file_type = fname[1].lower()
        elif isinstance(fname, (tuple, list)):
            self.name = fname[0]
            self.file_type = fname[1]
            self.content = fname[2]
        if self.file_type not in read_types:
            raise NotImplementedError(file_type_err.format(self.file_type))

        # Fill in attributes with geometry information
        all_geom = None
        if self.content is None:
            self.read_file(get_all, just_geom)
        elif self.file_type == 'log':
            f = StringIO(self.content)
            all_geom = self.read_log(f, get_all, just_geom)
        if all_geom:
            self.other['all_geom'] = all_geom

    def read_file(self, get_all=False, just_geom=True):
        """
        Reads geometry information from fname.
        Parameters:
            get_all     If false (default), only keep the last geom
                        If true, self is last geom, but return list
                            of all others encountered
        """
        with open(self.name + "." + self.file_type) as f:
            if self.file_type == 'xyz':
                all_geom = self.read_xyz(f)
            elif self.file_type == 'log':
                all_geom = self.read_log(f, get_all, just_geom)
            elif self.file_type == 'com':
                all_geom = self.read_com(f)

            if all_geom is not None:
                self.other['all_geom'] = all_geom
        return

    def skip_lines(self, f, n):
        for i in range(n):
            f.readline()
        return

    def read_xyz(self, f, get_all=False):
        all_geom = []
        # number of atoms
        f.readline()
        # comment
        self.comment = f.readline().strip()
        # atom info
        for line in f:
            line = line.strip()
            if line == '\n':
                if re.search('^\d+$', f.readline().strip()) is not None:
                    if get_all:
                        all_geom += [deepcopy(self.atoms)]
                    self.comment = f.readline().strip()
                    self.atoms = []
                else:
                    break
            line = line.split()
            self.atoms += [Atom(element=line[0], coords=line[1:])]
            for i, a in enumerate(self.atoms):
                a.name = str(i + 1)
        if get_all:
            return all_geom
        return

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

        all_geom = []
        line = f.readline()
        n = 1
        while line != '':
            if re.search("(Standard|Input) orientation:", line):
                if get_all and len(self.atoms) > 0:
                    all_geom += [deepcopy(self.atoms)]
                self.atoms, n = get_atoms(f, n)
            if just_geom:
                line = f.readline()
                n += 1
                continue
            if "Symbolic Z-matrix:" in line:
                line = f.readline()
                n += 1
                match = re.search(
                    "Charge\s*=\s*(\d+)\s*Multiplicity\s*=\s*(\d+)",
                    line
                )
                if match is not None:
                    self.other['charge'] = int(match.group(1))
                    self.other['multiplicity'] = int(match.group(2))
            if NORM_FINISH in line:
                self.other['finished'] = True
            if "SCF Done" in line:
                self.other['energy'] = float(line.split()[4])
            if "Molecular mass:" in line:
                self.other['mass'] = float(float_num.search(line).group(0))
                self.other['mass'] *= UNIT.AMU_TO_KG

            # Frequencies
            if "hpmodes" in line:
                self.other['hpmodes'] = True
            if "Harmonic frequencies" in line:
                freq_str = line
                while line != "\n":
                    line = f.readline()
                    n += 1
                    freq_str += line
                if 'hpmodes' not in self.other:
                    self.other['hpmodes'] = False
                self.other['frequency'] = Frequency(
                    freq_str, self.other['hpmodes'])

            # Thermo
            if "Temperature" in line:
                self.other['temperature'] = float(
                    float_num.search(line).group(0))
            if "Rotational constants (GHZ):" in line:
                rot = float_num.findall(line)
                rot = [float(r) * PHYSICAL.PLANK * (10**9) / PHYSICAL.KB
                       for r in rot]
                self.other['rotational_temperature'] = rot
            if "Sum of electronic and zero-point Energies=" in line:
                self.other['E_ZPVE'] = float(float_num.search(line).group(0))
            if "Sum of electronic and thermal Enthalpies=" in line:
                self.other['enthalpy'] = float(float_num.search(line).group(0))
            if "Sum of electronic and thermal Free Energies=" in line:
                self.other['free_energy'] = float(
                    float_num.search(line).group(0))
            if "Zero-point correction=" in line:
                self.other['ZPVE'] = float(float_num.search(line).group(0))
            if "Rotational symmetry number" in line:
                self.other['rotational_symmetry_number'] \
                    = int(re.search('\d+', line).group(0))

            # Gradient
            if re.search("Threshold\s+Converged", line) is not None:
                line = f.readline()
                n += 1
                grad = {}

                def add_grad(line, name, grad):
                    line = line.split()
                    grad[name] = {
                        'value': line[2],
                        'threshold': line[3],
                        'converged': True if line[4] == 'YES' else False
                    }
                    return grad

                while line != '':
                    if "Predicted change in Energy" in line:
                        break
                    if re.search('Maximum\s+Force', line) is not None:
                        grad = add_grad(line, 'Max Force', grad)
                    if re.search('RMS\s+Force', line) is not None:
                        grad = add_grad(line, 'RMS Force', grad)
                    if re.search('Maximum\s+Displacement', line) is not None:
                        grad = add_grad(line, 'Max Disp', grad)
                    if re.search('RMS\s+Displacement', line) is not None:
                        grad = add_grad(line, 'RMS Disp', grad)
                    line = f.readline()
                    n += 1
                self.other['gradient'] = grad

            # capture errors
            for err in ERRORS:
                if err in line:
                    self.other['error'] = ERRORS[err]
                    self.other['error_msg'] = line.strip()
                    break
            else:
                self.other['error'] = None

            line = f.readline()
            n += 1

        for i, a in enumerate(self.atoms):
            a.name = str(i + 1)

        if 'finished' not in self.other:
            self.other['finished'] = False
        if 'error' not in self.other:
            self.other['error'] = None
        if not self.other['finished'] and not self.other['error']:
            self.other['error'] = ERRORS['Unknown message']
            self.other['error_msg'] = 'Unknown message'
        if get_all:
            return all_geom
        return

    def read_com(self, f):
        found_atoms = False
        found_constraint = False
        atoms = []
        other = {}
        for line in f:
            # header
            if line.startswith('%'):
                continue
            if line.startswith('#'):
                other['method'] = re.search('^#(\S+)', line).group(1)
                if 'temperature=' in line:
                    other['temperature'] = re.search(
                        'temperature=(\d+\.?\d*)', line
                    ).group(1)
                if 'solvent=' in line:
                    other['solvent'] = re.search(
                        'solvent=(\S+)\)', line
                    ).group(1)
                if 'scrf=' in line:
                    other['solvent_model'] = re.search(
                        'scrf=\((\S+),', line
                    ).group(1)
                if 'EmpiricalDispersion=' in line:
                    other['emp_dispersion'] = re.search(
                        'EmpiricalDispersion=(\s+)', line
                    ).group(1)
                if 'int=(grid(' in line:
                    other['grid'] = re.search(
                        'int=\(grid(\S+)', line
                    ).group(1)
                for _ in range(4):
                    line = f.readline()
                line = line.split()
                other['charge'] = line[0]
                other['mult'] = line[1]
                found_atoms = True
                continue
            # constraints
            if found_atoms and line.startswith('B') and line.endswith('F'):
                found_constraint = True
                if 'constraint' not in other:
                    other['constraint'] = []
                other['constraint'] += [float_num.findall(line)]
                continue
            # footer
            if found_constraint:
                if 'footer' not in other:
                    other['footer'] = ''
                other['footer'] += line
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
    class Data:
        def __init__(self, frequency):
            self.frequency = frequency
            self.intensity = None
            self.vector = {}

        def to_json(self):
            tmp = {}
            tmp['frequency'] = self.frequency
            tmp['intensity'] = self.intensity
            tmp['vector'] = self.vector
            return json.dumps(tmp)

    def __init__(self, lines, hpmodes):
        self.data = []
        self.frequencies = []
        self.intensities = []
        self.vectors = []

        self.imaginary_frequencies = None
        self.real_frequencies = None
        self.lowest_frequency = None
        self.by_frequency = {}
        self.is_TS = None

        lines = lines.split('\n')
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

    def to_json(self):
        tmp = {}
        tmp['data'] = [d.to_json() for d in self.data]
        tmp['is_TS'] = self.is_TS

    def parse_lines(self, lines, hpmodes):
        num_head = 0
        idx = -1
        for line in lines:
            if "Harmonic frequencies" in line:
                num_head += 1
                if hpmodes and num_head == 2:
                    # if hpmodes, want just the first set of freqs
                    break
                continue
            if "Frequencies" in line and "---" in line:
                for i in float_num.findall(line):
                    self.data += [Frequency.Data(float(i))]
                    idx += 1
                continue
            if "IR Intensit" in line and "---" in line:
                for i in float_num.findall(line):
                    self.data[idx].intensity = float(i)
                continue

            if hpmodes:
                match = re.search(
                    '^\s+\d+\s+\d+\s+\d+\s+([+-]?\d+\.\d+)+$', line)
                if match is None:
                    continue
                values = float_num.findall(line)
                coord = int(values[0]) - 1
                atom = int(values[1]) - 1
                moves = [float(i) for i in values[3:]]
                for d, m in zip(self.data[-len(moves):], moves):
                    try:
                        d.vector[atom][coord] = m
                    except IndexError:
                        d.vector[atom] = [0, 0, 0]
                        d.vector[atom][coord] = m

            else:
                match = re.search('^\s+\d+\s+\d+\s+([+-]?\d+\.\d+)+$', line)
                if match is None:
                    continue
                atom = int(values[0]) - 1
                moves = [float(i) for i in values[2:]]
                n_moves = len(moves) // 3
                for d, m in zip(self.data[-n_moves:], range(n_moves)):
                    d.vector[atom] = moves[m:m + 3]
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
                'intensity': data.intensity,
                'vector': data.vector
            }
        self.lowest_frequency = self.data[0].frequency
        self.is_TS = True if len(self.imaginary_frequencies) == 1 else False
