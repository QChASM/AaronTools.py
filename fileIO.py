"""For parsing input/output files"""
import concurrent.futures
import os
import re
import sys
from copy import deepcopy
from io import IOBase, StringIO

import numpy as np
from scipy.special import factorial2

from AaronTools import addlogger
from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS, PHYSICAL, UNIT
from AaronTools.spectra import Frequency, ValenceExcitations
from AaronTools.theory import *
from AaronTools.utils.utils import is_alpha, is_int, is_num, float_num

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
    "47",
    "31",
]
write_types = ["xyz", "com", "inp", "in", "sqmin", "cube"]
file_type_err = "File type not yet implemented: {}"
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

def expected_inp_ext(exec_type):
    """
    extension expected for an input file for exec_type
    Gaussian - .com (.gjf on windows)
    ORCA - .inp
    Psi4 - .in
    SQM - .mdin
    """
    if exec_type.lower() == "gaussian":
        if sys.platform.startswith("win"):
            return ".gjf"
        return ".com"
    if exec_type.lower() == "orca":
        return ".inp"
    if exec_type.lower() == "psi4":
        return ".in"
    if exec_type.lower() == "sqm":
        return ".mdin"

def expected_out_ext(exec_type):
    """
    extension expected for an input file for exec_type
    Gaussian - .log
    ORCA - .out
    Psi4 - .out
    SQM - .mdout
    """
    if exec_type.lower() == "gaussian":
        return ".log"
    if exec_type.lower() == "orca":
        return ".out"
    if exec_type.lower() == "psi4":
        return ".out"
    if exec_type.lower() == "sqm":
        return ".mdout"


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
        elif style.lower() == "cube":
            out = cls.write_cube(geom, outfile=outfile, **kwargs)

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
        """
        write Gaussian input file for given Theory() and Geometry()
        geom - Geometry()
        theory - Theory()
        outfile - None, False, or str
                  None - geom.name + ".com" is used as output destination
                  False - return contents of the input file as a str
                  str - output destination
        return_warnings - True to return a list of warnings (e.g. basis
                          set might be misspelled
        kwargs - passed to Theory methods (make_header, make_molecule, etc.)
        """
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
                outfile = "{}.{}.com".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.com".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{ name }", name)
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings
        return

    @classmethod
    def write_inp(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write ORCA input file for the given Theory() and Geometry()
        geom - Geometry()
        theory - Theory()
        outfile - None, False, or str
                  None - geom.name + ".com" is used as output destination
                  False - return contents of the input file as a str
                  str - output destination
        return_warnings - True to return a list of warnings (e.g. basis
                          set might be misspelled
        kwargs - passed to Theory methods (make_header, make_molecule, etc.)
        """
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
                outfile = "{}.{}.com".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.com".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{ name }", name)
            with open(outfile, "w") as f:
                f.write(s)

    @classmethod
    def write_in(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write Psi4 input file for the given Theory() and Geometry()
        geom - Geometry()
        theory - Theory()
        outfile - None, False, or str
                  None - geom.name + ".com" is used as output destination
                  False - return contents of the input file as a str
                  str - output destination
        return_warnings - True to return a list of warnings (e.g. basis
                          set might be misspelled
        kwargs - passed to Theory methods (make_header, make_molecule, etc.)
        """
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
                outfile = "{}.{}.in".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.in".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{ name }", name)
            with open(outfile, "w") as f:
                f.write(s)

    @classmethod
    def write_sqm(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write SQM input file for the given Theory() and Geometry()
        geom - Geometry()
        theory - Theory()
        outfile - None, False, or str
                  None - geom.name + ".com" is used as output destination
                  False - return contents of the input file as a str
                  str - output destination
        return_warnings - True to return a list of warnings (e.g. basis
                          set might be misspelled
        kwargs - passed to Theory methods (make_header, make_molecule, etc.)
        """
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
                outfile = "{}.{}.com".format(
                    geom.name, step2str(kwargs["step"])
                )
            else:
                outfile = "{}.com".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{ name }", name)
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings

    @classmethod
    def write_cube(
        cls,
        geom,
        orbitals=None,
        outfile=None,
        kind="homo",
        padding=4.0,
        spacing=0.2,
        alpha=True,
        xyz=False,
        n_jobs=1,
        delta=0.1,
        **kwargs,
    ):
        """
        write a cube file for a molecular orbital
        geom - geometry
        orbitals - Orbitals()
        outfile - output destination
        mo - index of molecular orbital or "homo" for ground state
             highest occupied molecular orbital or "lumo" for first
             ground state unoccupied MO
             can also be an array of MO coefficients
        ao - index of atomic orbital to print
        padding - padding around geom's coordinates
        spacing - targeted spacing between points
        n_jobs - number of parallel threads to use
                 this is on top of NumPy's multithreading, so
                 if NumPy uses 8 threads and n_jobs=2, you can
                 expect to see 16 threads in use
        delta - see Orbitals.fukui_donor_value or fukui_acceptor_value
        """
        if orbitals is None:
            raise RuntimeError(
                "no Orbitals() instance given to FileWriter.write_cube"
            )

        def get_standard_axis():
            """returns info to set up a grid along the x, y, and z axes"""
            geom_coords = geom.coords

            # get range of geom's coordinates
            x_min = np.min(geom_coords[:, 0])
            x_max = np.max(geom_coords[:, 0])
            y_min = np.min(geom_coords[:, 1])
            y_max = np.max(geom_coords[:, 1])
            z_min = np.min(geom_coords[:, 2])
            z_max = np.max(geom_coords[:, 2])

            # add padding, figure out vectors
            r1 = 2 * padding + x_max - x_min
            n_pts1 = int(r1 // spacing) + 1
            d1 = r1 / (n_pts1 - 1)
            v1 = np.array((d1, 0., 0.))
            r2 = 2 * padding + y_max - y_min
            n_pts2 = int(r2 // spacing) + 1
            d2 = r2 / (n_pts2 - 1)
            v2 = np.array((0., d2, 0.))
            r3 = 2 * padding + z_max - z_min
            n_pts3 = int(r3 // spacing) + 1
            d3 = r3 / (n_pts3 - 1)
            v3 = np.array((0., 0., d3))
            com = np.array([x_min, y_min, z_min]) - padding
            return n_pts1, n_pts2, n_pts3, v1, v2, v3, com

        if xyz:
            n_pts1, n_pts2, n_pts3, v1, v2, v3, com = get_standard_axis()
        else:
            test_coords = geom.coords - geom.COM()
            covar = np.dot(test_coords.T, test_coords)
            try:
                # use SVD on the coordinate covariance matrix
                # this decreases the volume of the box we're making
                # that means less work for higher resolution
                # for many structures, this only decreases the volume
                # by like 5%
                u, s, vh = np.linalg.svd(covar)
                v1 = u[:, 0]
                v2 = u[:, 1]
                v3 = u[:, 2]
                # change basis of coordinates to the singular vectors
                # this is how we determine the range + padding
                new_coords = np.dot(test_coords, u)
                xr_max = np.max(new_coords[:, 0])
                xr_min = np.min(new_coords[:, 0])
                yr_max = np.max(new_coords[:, 1])
                yr_min = np.min(new_coords[:, 1])
                zr_max = np.max(new_coords[:, 2])
                zr_min = np.min(new_coords[:, 2])
                com = np.array([xr_min, yr_min, zr_min]) - padding
                # move the COM back to the xyz space of the original molecule
                com = np.dot(u, com)
                com += geom.COM()
                r1 = 2 * padding + np.linalg.norm(xr_max - xr_min)
                r2 = 2 * padding + np.linalg.norm(yr_max - yr_min)
                r3 = 2 * padding + np.linalg.norm(zr_max - zr_min)
                n_pts1 = int(r1 // spacing) + 1
                n_pts2 = int(r2 // spacing) + 1
                n_pts3 = int(r3 // spacing) + 1
                v1 *= r1 / (n_pts1 - 1)
                v2 *= r2 / (n_pts2 - 1)
                v3 *= r3 / (n_pts3 - 1)
            except np.linalg.LinAlgError:
                n_pts1, n_pts2, n_pts3, v1, v2, v3, com = get_standard_axis()

        mo = None
        if kind.lower() == "homo":
            mo = max(orbitals.n_alpha, orbitals.n_beta) - 1
        elif kind.lower() == "lumo":
            mo = max(orbitals.n_alpha, orbitals.n_beta)
        elif kind.lower().startswith("mo"):
            mo = int(kind.split()[-1])
        elif kind.lower().startswith("ao"):
            mo = np.zeros(orbitals.n_mos)
            mo = mo[int(kind.split()[-1])] = 1
        
        s = ""
        s += " %s\n" % geom.comment
        s += " %s\n" % kind

        # the '-' in front of the number of atoms indicates that this is
        # MO info so there's an extra data entry between the molecule
        # and the function values
        bohr_com = com / UNIT.A0_TO_BOHR
        if mo:
            s += " -"
        else:
            s += "  "
        s += "%i %13.5f %13.5f %13.5f 1\n" % (
            len(geom.atoms), *bohr_com,
        )

        # the basis vectors of cube files are ordered based on the
        # spacing between points along that axis
        # or maybe it's the number of points?
        # we use the first one
        arr = []
        v_list = []
        n_list = []
        for n, v in sorted(
            zip([n_pts1, n_pts2, n_pts3], [v1, v2, v3]),
            key=lambda p: np.linalg.norm(p[1]),
        ):
            bohr_v = v / UNIT.A0_TO_BOHR
            s += " %5i %13.5f %13.5f %13.5f\n" % (
                n, *bohr_v
            )
            arr.append(np.linspace(0, n - 1, num=n, dtype=int))
            v_list.append(v)
            n_list.append(n)
        # contruct an array of points for the grid
        ndx = (
            np.vstack(np.mgrid[0 : n_list[0], 0 : n_list[1], 0 : n_list[2]])
            .reshape(3, np.prod(n_list))
            .T
        )
        coords = np.matmul(ndx, v_list)
        del ndx
        coords += com

        # write the structure in bohr
        for atom in geom.atoms:
            s += " %5i %13.5f %13.5f %13.5f %13.5f\n" % (
                ELEMENTS.index(atom.element),
                ELEMENTS.index(atom.element),
                atom.coords[0] / UNIT.A0_TO_BOHR,
                atom.coords[1] / UNIT.A0_TO_BOHR,
                atom.coords[2] / UNIT.A0_TO_BOHR,
            )

        # extra section - only for MO data
        if mo is not None:
            s += " %5i %5i\n" % (1, int(kind.split()[-1]))

        # get values for this MO
        if kind.lower() == "density":
            val = orbitals.density_value(coords, n_jobs=n_jobs)
            # val = orbitals.low_mem_density_value(coords, n_jobs=n_jobs)
        elif kind.lower() == "fukui donor":
            val = orbitals.fukui_donor_value(
                coords, n_jobs=n_jobs, delta=delta
            )
        elif kind.lower() == "fukui acceptor":
            val = orbitals.fukui_acceptor_value(
                coords, n_jobs=n_jobs, delta=delta
            )
        else:
            val = orbitals.mo_value(mo, coords, n_jobs=n_jobs)

        # write to a file
        for n1 in range(0, n_list[0]):
            for n2 in range(0, n_list[1]):
                val_ndx = n1 * n_list[2] * n_list[1] + n2 * n_list[2]
                val_subset = val[val_ndx : val_ndx + n_list[2]]
                for i, v in enumerate(val_subset):
                    if abs(v) < 1e-30:
                        v = 0
                    s += "%13.5e" % v
                    if (i + 1) % 6 == 0:
                        s += "\n"
                if (i + 1) % 6 != 0:
                    s += "\n"

        if outfile is None:
            # if no output file is specified, use the name of the geometry
            with open(geom.name + ".cube", "w") as f:
                f.write(s)
        elif outfile is False:
            # if no output file is desired, just return the file contents
            return s
        else:
            # write output to the requested destination
            with open(outfile, "w") as f:
                f.write(s)
        return


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
        nbo_name=None,
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
        :nbo_name: Name of the file containing the NBO orbital coefficients
            in the AO basis. Only used when reading *.47 files.
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
                get_all, just_geom,
                freq_name=freq_name, conf_name=conf_name, nbo_name=nbo_name
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
            elif self.file_type == "47":
                self.read_nbo_47(f, nbo_name=nbo_name)
            elif self.file_type == "31":
                self.read_nbo_31(f, nbo_name=nbo_name)

    def read_file(
        self, get_all=False, just_geom=True,
        freq_name=None, conf_name=None, nbo_name=None,
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
        elif self.file_type == "47":
            self.read_nbo_47(f, nbo_name=nbo_name)
        elif self.file_type == "31":
            self.read_nbo_31(f, nbo_name=nbo_name)

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
        # if get_all:
        #     self.all_geom += [(deepcopy(self.comment), deepcopy(self.atoms))]

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
                
                elif re.match("E\(\S+\)\s+...\s+-?\d+\.\d+$", line):
                    nrg = re.match("(E\(\S+\))\s+...\s+(-?\d+\.\d+)$", line)
                    self.other["energy"] = float(nrg.group(2))
                    self.other[nrg.group(1)] = float(nrg.group(2))

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
                    while not (stage == "THERMO" and line == "\n") and line:
                        if "--" not in line and line != "\n":
                            freq_str += line

                        if "NORMAL MODES" in line:
                            stage = "modes"
                            self.skip_lines(f, 6)
                            n += 6

                        if "RAMAN SPECTRUM" in line:
                            stage = "RAMAN"
                            self.skip_lines(f, 2)
                            n += 2

                        if "IR SPECTRUM" in line:
                            stage = "IR"
                            self.skip_lines(f, 2)
                            n += 2

                        if "THERMOCHEMISTRY" in line:
                            stage = "THERMO"

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

                elif "MAYER POPULATION ANALYSIS" in line:
                    self.skip_lines(f, 2)
                    n += 2
                    line = f.readline()
                    data = dict()
                    headers = []
                    while line.strip():
                        info = line.split()
                        header = info[0]
                        name = " ".join(info[2:])
                        headers.append(header)
                        data[header] = (name, [])
                        line = f.readline()
                    self.skip_lines(f, 1)
                    n += 1
                    for i in range(0, len(self.atoms)):
                        line = f.readline()
                        info = line.split()[2:]
                        for header, val in zip(headers, info):
                            data[header][1].append(float(val))

                    for header in headers:
                        self.other[data[header][0]] = np.array(data[header][1])

                elif line.startswith("LOEWDIN ATOMIC CHARGES"):
                    self.skip_lines(f, 1)
                    n += 1
                    charges = np.zeros(len(self.atoms))
                    for i in range(0, len(self.atoms)):
                        line = f.readline()
                        n += 1
                        charges[i] = float(line.split()[-1])
                    self.other["Löwdin Charges"] = charges

                elif line.startswith("BASIS SET IN INPUT FORMAT"):
                    # read basis set primitive info
                    self.skip_lines(f, 3)
                    n += 3
                    line = f.readline()
                    n += 1
                    self.other["basis_set_by_ele"] = dict()
                    while "--" not in line:
                        new_gto = re.search("NewGTO\s+(\S+)", line)
                        if new_gto:
                            ele = new_gto.group(1)
                            line = f.readline()
                            n += 1
                            primitives = []
                            while "end" not in line:
                                shell_type, n_prim = line.split()
                                n_prim = int(n_prim)
                                exponents = []
                                con_coeffs = []
                                for i in range(0, n_prim):
                                    line = f.readline()
                                    n += 1
                                    info = line.split()
                                    exponent = float(info[1])
                                    con_coeff = [float(x) for x in info[2:]]
                                    exponents.append(exponent)
                                    con_coeffs.extend(con_coeff)
                                primitives.append(
                                    (
                                        shell_type,
                                        n_prim,
                                        exponents,
                                        con_coeffs,
                                    )
                                )
                                line = f.readline()
                                n += 1
                            self.other["basis_set_by_ele"][ele] = primitives
                        line = f.readline()
                        n += 1

                elif line.startswith("MOLECULAR ORBITALS"):
                    # read molecular orbitals
                    self.skip_lines(f, 1)
                    n += 1
                    line = f.readline()
                    self.other["alpha_coefficients"] = []
                    self.other["beta_coefficients"] = []
                    self.other["alpha_nrgs"] = []
                    self.other["beta_nrgs"] = []
                    self.other["alpha_occupancies"] = []
                    self.other["beta_occupancies"] = []
                    at_info = re.compile(
                        "\s*(\d+)\S+\s+\d+(?:s|p[xyz]|d(?:z2|xz|yz|x2y2|xy)|[fghi][\+\-]?\d+)"
                    )
                    if self.other["multiplicity"] != 1:
                        args = [
                            ("alpha_coefficients", "beta_coefficients"),
                            ("alpha_nrgs", "beta_nrgs"),
                            ("alpha_occupancies", "beta_occupancies"),
                        ]
                    else:
                        args = [
                            ("alpha_coefficients",),
                            ("alpha_nrgs",),
                            ("alpha_occupancies",),
                        ]

                    for coeff_name, nrg_name, occ_name in zip(*args):
                        self.other["shell_to_atom"] = []
                        mo_coefficients = []
                        orbit_nrgs = []
                        occupancy = []
                        while line.strip() != "":
                            at_match = at_info.match(line)
                            if at_match:
                                ndx = int(at_match.group(1))
                                self.other["shell_to_atom"].append(ndx)
                                coeffs = []
                                # there might not always be a space between the coefficients
                                # so we can't just split(), but they are formatted(-ish)
                                for coeff in re.findall("-?\d+\.\d\d\d\d\d\d", line[16:]):
                                    coeffs.append(float(coeff))
                                for coeff, mo in zip(coeffs, mo_coefficients):
                                    mo.append(coeff)
                            elif "--" not in line:
                                orbit_nrgs = occupancy
                                occupancy = [float(x) for x in line.split()]
                            elif "--" in line:
                                self.other[nrg_name].extend(orbit_nrgs)
                                self.other[occ_name].extend(occupancy)
                                if mo_coefficients:
                                    self.other[coeff_name].extend(
                                        mo_coefficients
                                    )
                                    if not all(
                                        len(coeff) == len(mo_coefficients[0])
                                        for coeff in mo_coefficients
                                    ):
                                        self.LOG.warning(
                                            "orbital coefficients may not "
                                            "have been parsed correctly"
                                        )
                                mo_coefficients = [[] for x in orbit_nrgs]
                                orbit_nrgs = []
                            line = f.readline()
                            n += 1
                        self.other[coeff_name].extend(mo_coefficients)
                        line = f.readline()

                elif line.startswith("N(Alpha)  "):
                    self.other["n_alpha"] = int(
                        np.rint(float(line.split()[2]))
                    )

                elif line.startswith("N(Beta)  "):
                    self.other["n_beta"] = int(np.rint(float(line.split()[2])))

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

            if (
                "alpha_coefficients" in self.other
                and "basis_set_by_ele" in self.other
            ):
                self.other["orbitals"] = Orbitals(self)

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
            # read energies from different methods
            if "SCF Done" in line:
                tmp = [word.strip() for word in line.split()]
                idx = tmp.index("=")
                self.other["energy"] = float(tmp[idx + 1])
                self.other["scf_energy"] = float(tmp[idx + 1])

            else:
                nrg_match = re.search("\s+(E\(\S+\))\s*=\s*(\S+)", line)
                # ^ matches many methods
                # will also match the SCF line (hence the else here)
                # the match in the SCF line could be confusing b/c
                # the SCF line could be
                # SCF Done:  E(RB2PLYPD3) =  -76.2887108570     A.U. after   10 cycles
                # and later on, there will be a line...
                #  E2(B2PLYPD3) =    -0.6465105880D-01 E(B2PLYPD3) =    -0.76353361915801D+02
                # this will give:
                # * E(RB2PLYPD3) = -76.2887108570 
                # * E(B2PLYPD3) = -76.353361915801
                # very similar names for very different energies...
                if nrg_match:
                    self.other["energy"] = float(nrg_match.group(2).replace("D", "E"))
                    self.other[nrg_match.group(1)] = self.other["energy"]
            
            # CC energy
            if line.startswith(" CCSD(T)= "):
                self.other["energy"] = float(line.split()[-1].replace("D", "E"))
                self.other["E(CCSD(T))"] = self.other["energy"]
            
            # MP energies
            mp_match = re.search("([RU]MP\d+(?:\(\S+\))?)\s*=\s*(\S+)", line)
            if mp_match:
                self.other["energy"] = float(mp_match.group(2).replace("D", "E"))
                self.other["E(%s)" % mp_match.group(1)] = self.other["energy"]

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
                    freq_str, hpmodes=self.other["hpmodes"]
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

                self.other["frequency"].parse_gaussian_lines(
                    anharm_str.splitlines(), harmonic=False,
                )

            # X matrix for anharmonic
            if "Total Anharmonic X Matrix" in line:
                self.skip_lines(f, 1)
                n += 1
                n_freq = len(self.other["frequency"].data)
                n_sections = int(np.ceil(n_freq / 5))
                x_matrix = np.zeros((n_freq, n_freq))
                for section in range(0, n_sections):
                    header = f.readline()
                    n += 1
                    for j in range(5 * section, n_freq):
                        line = f.readline()
                        n += 1
                        ll = 5 * section
                        ul = 5 * section + min(j - ll + 1, 5)
                        x_matrix[j, ll:ul] = [
                            float(x.replace("D", "e"))
                            for x in line.split()[1:]
                        ]
                x_matrix += np.tril(x_matrix, k=-1).T
                self.other["X_matrix"] = x_matrix

            if "Total X0" in line:
                self.other["X0"] = float(line.split()[5])

            # TD-DFT output
            if line.strip().startswith("Ground to excited state"):
                uv_vis = ""
                highest_state = 0
                done = False
                read_states = False
                while not done:
                    n += 1
                    uv_vis += line
                    if not read_states and line.strip() and line.split()[0].isdigit():
                        state = int(line.split()[0])
                        if state > highest_state:
                            highest_state = state
                    if line.strip().startswith("Ground to excited state transition velocity"):
                        read_states = True
                    if re.search("Excited State\s*%i:" % highest_state, line):
                        done = True
                    if line.strip().startswith("Total Energy, E"):
                        nrg = re.search(
                            r"Total Energy, E\((\S+)\)\s*=\s*(-?\d+\.\d+)", line
                        )
                        self.other["E(%s)" % nrg.group(1)] = float(nrg.group(2))
                        self.other["energy"] = float(nrg.group(2))

                    line = f.readline()
                self.other["uv_vis"] = ValenceExcitations(
                    uv_vis, style="gaussian"
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

            # rotational constants from anharmonic frequency jobs
            if "Rotational Constants (in MHz)" in line:
                self.skip_lines(f, 2)
                n += 2
                equilibrium_rotational_temperature = np.zeros(3)
                ground_rotational_temperature = np.zeros(3)
                centr_rotational_temperature = np.zeros(3)
                for i in range(0, 3):
                    line = f.readline()
                    n += 1
                    info = line.split()
                    Be = float(info[1])
                    B00 = float(info[3])
                    B0 = float(info[5])
                    equilibrium_rotational_temperature[i] = Be
                    ground_rotational_temperature[i] = B00
                    centr_rotational_temperature[i] = B0
                equilibrium_rotational_temperature *= (
                    PHYSICAL.PLANCK * 1e6 / PHYSICAL.KB
                )
                ground_rotational_temperature *= (
                    PHYSICAL.PLANCK * 1e6 / PHYSICAL.KB
                )
                centr_rotational_temperature *= (
                    PHYSICAL.PLANCK * 1e6 / PHYSICAL.KB
                )
                self.other[
                    "equilibrium_rotational_temperature"
                ] = equilibrium_rotational_temperature
                self.other[
                    "ground_rotational_temperature"
                ] = ground_rotational_temperature
                self.other[
                    "centr_rotational_temperature"
                ] = centr_rotational_temperature

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

            # electronic properties
            if "Electrostatic Properties (Atomic Units)" in line:
                self.skip_lines(f, 5)
                n += 5
                self.other["electric_potential"] = []
                self.other["electric_field"] = []
                line = f.readline()
                while "--" not in line:
                    info = line.split()
                    self.other["electric_potential"].append(float(info[2]))
                    self.other["electric_field"].append([float(x) for x in info[3:]])
                    line = f.readline()
                    n += 1
                self.other["electric_potential"] = np.array(self.other["electric_potential"])
                self.other["electric_field"] = np.array(self.other["electric_field"])

            # optical features
            if "[Alpha]" in line:
                alpha_match = re.search("\[Alpha\].*\(\s*(.*\s?.*)\)\s*=\s*(-?\d+\.\d+)", line)
                self.other["optical_rotation_(%s)" % alpha_match.group(1)] = \
                float(alpha_match.group(2))

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

            # atomic charges
            charge_match = re.search("(\S+) charges:\s*$", line)
            if charge_match:
                self.skip_lines(f, 1)
                n += 1
                charges = []
                for i in range(0, len(self.atoms)):
                    line = f.readline()
                    n += 1
                    charges.append(float(line.split()[2]))
                self.other[charge_match.group(1) + " Charges"] = charges

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
                                    if not any(atom.flag for atom in self.atoms):
                                        constraints = None
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
                                    if (
                                        "Integral"
                                        not in other_kwargs[GAUSSIAN_ROUTE]
                                    ):
                                        other_kwargs[GAUSSIAN_ROUTE][
                                            "Integral"
                                        ] = []
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
        char_info = re.compile(
            "([\S\s]+?)\s*C\s*([N=])*\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)"
        )

        theory = Theory()

        lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if i == 0:
                other["comment"] = line.strip()
            elif i == 1:
                i += 1
                line = lines[i]
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
            char_match = char_info.match(line)
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

            # elif char_match is not None:
            #     data = char_match.group(1)
            #     value = char_match.group(3)
            #     if not just_geom:
            #         other[data] = lines[i + 1]
            #         i += 1

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

        try:
            self.other["orbitals"] = Orbitals(self)
        except NotImplementedError:
            pass

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
                self.other["free_energy"] = (
                    float(float_num.findall(line)[0]) / UNIT.HART_TO_KCAL
                )
            elif "ensemble entropy" in line:
                self.other["entropy"] = (
                    float(float_num.findall(line)[1]) / UNIT.HART_TO_KCAL
                )
            elif "ensemble average energy" in line:
                self.other["avg_energy"] = (
                    float(float_num.findall(line)[0]) / UNIT.HART_TO_KCAL
                )
            elif "E lowest" in line:
                self.other["energy"] = float(float_num.findall(line)[0])
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
                self.other["energy"] = (
                    float(float_num.findall(line)[0]) * UNIT.HART_TO_KCAL
                )
            if "zero point energy" in line:
                self.other["ZPVE"] = (
                    float(float_num.findall(line)[0]) * UNIT.HART_TO_KCAL
                )
            if "total free energy" in line:
                self.other["free_energy"] = (
                    float(float_num.findall(line)[0]) * UNIT.HART_TO_KCAL
                )
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
                for info in lines[i + 2 :]:
                    if not info.strip() or not info.split()[0].isdigit():
                        break
                    ele = info.split()[1]
                    elements.append(ele)
                i += len(elements) + 2

            if "Final Structure" in line:
                k = 0
                for info in lines[i + 4 :]:
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
                self.other["energy"] = (
                    float(line.split()[4]) / UNIT.HART_TO_KCAL
                )

            i += 1

        if not self.atoms:
            # there's no atoms if there's an error
            # error is probably on the last line
            self.other["error"] = "UNKNOWN"
            self.other["error_msg"] = line

    def read_nbo_47(self, f, nbo_name=None):
        lines = f.readlines()
        bohr = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(" $"):
                section = line.split()[0]
                if section.startswith("$COORD"):
                    i += 1
                    self.atoms = []
                    line = lines[i]
                    while not line.startswith(" $END"):
                        if re.search("\d+\s+\d+(?:\s+-?\d+\.\d+\s){3}", line):
                            info = line.split()
                            ndx = int(info[0])
                            coords = [float(x) for x in info[2:5]]
                            self.atoms.append(
                                Atom(
                                    element=ELEMENTS[ndx],
                                    name=str(len(self.atoms) + 1),
                                    coords=np.array(coords),
                                )
                            )
                        i += 1
                        line = lines[i]
                elif section.startswith("$BASIS"):
                    reading_centers = False
                    reading_labels = False
                    i += 1
                    line = lines[i]
                    while not line.startswith(" $END"):
                        if "CENTER" in line.upper():
                            self.other["shell_to_atom"] = [
                                int(x) for x in line.split()[2:]
                            ]
                            reading_centers = True
                            reading_labels = False
                        elif "LABEL" in line.upper():
                            self.other["momentum_label"] = [
                                int(x) for x in line.split()[2:]
                            ]
                            reading_labels = True
                            reading_centers = False
                        elif reading_centers:
                            self.other["shell_to_atom"].extend(
                                [int(x) for x in line.split()]
                            )
                        elif reading_labels:
                            self.other["momentum_label"].extend(
                                [int(x) for x in line.split()]
                            )
                        i += 1
                        line = lines[i]
                elif section.startswith("$CONTRACT"):
                    int_sections = {
                        "NCOMP": "funcs_per_shell",
                        "NPRIM": "n_prim_per_shell",
                        "NPTR": "start_ndx",
                    }
                    float_sections = {
                        "EXP": "exponents",
                        "CS": "s_coeff",
                        "CP": "p_coeff",
                        "CD": "d_coeff",
                        "CF": "f_coeff",
                    }
                    i += 1
                    line = lines[i]
                    while not line.startswith(" $END"):
                        if any(line.strip().startswith(section) for section in int_sections):
                            section = line.split()[0]
                            self.other[int_sections[section]] = [
                                int(x) for x in line.split()[2:]
                            ]
                            i += 1
                            line = lines[i]
                            while "=" not in line and "$" not in line:
                                self.other[int_sections[section]].extend([
                                    int(x) for x in line.split()
                                ])
                                i += 1
                                line = lines[i]
                        elif any(line.strip().startswith(section) for section in float_sections):
                            section = line.split()[0]
                            self.other[float_sections[section]] = [
                                float(x) for x in line.split()[2:]
                            ]
                            i += 1
                            line = lines[i]
                            while "=" not in line and "$" not in line:
                                self.other[float_sections[section]].extend([
                                    float(x) for x in line.split()
                                ])
                                i += 1
                                line = lines[i]
                        else:
                            i += 1
                            line = lines[i]
                            
                elif section.startswith("$GENNBO"):
                    if "BOHR" in section.upper():
                        bohr = True
                    nbas = re.search("NBAS=(\d+)", line)
                    n_funcs = int(nbas.group(1))
                    if "CUBICF" in section.upper():
                        self.LOG.warning("cubic F shell will not be handled correctly")
            i += 1
        
        if nbo_name is not None:
            self._read_nbo_coeffs(nbo_name)
    
    def _read_nbo_coeffs(self, nbo_name):
        """
        read coefficients in AO basis for NBO's/NLHO's/NAO's/etc.
        called by methods that read NBO input (.47) or output files (.31)
        """
        with open(nbo_name, "r") as f2:
            lines = f2.readlines()
        kind = re.search("P?(\S+)s", lines[1]).group(1)
        desc_file = os.path.splitext(nbo_name)[0] + ".46"
        if os.path.exists(desc_file):
            with open(desc_file, "r") as f3:
                desc_lines = f3.readlines()
                for k, line in enumerate(desc_lines):
                    if kind in line:
                        self.other["orbit_kinds"] = []
                        n_orbits = int(line.split()[1])
                        k += 1
                        while len(self.other["orbit_kinds"]) < n_orbits:
                            self.other["orbit_kinds"].extend([
                                desc_lines[k][i: i + 10]
                                for i in range(1, len(desc_lines[k]) - 1, 10)
                            ])
                            k += 1
        else:
            self.LOG.warning(
                "no .46 file found - orbital descriptions will be unavialable"
            )
                
        j = 3
        self.other["alpha_coefficients"] = []
        while len(self.other["alpha_coefficients"]) < sum(self.other["funcs_per_shell"]):
            mo_coeff = []
            while len(mo_coeff) < sum(self.other["funcs_per_shell"]):
                mo_coeff.extend([float(x) for x in lines[j].split()])
                j += 1
            self.other["alpha_coefficients"].append(mo_coeff)
        self.other["orbitals"] = Orbitals(self)

    def read_nbo_31(self, f, nbo_name=None):
        lines = f.readlines()
        comment = lines[0].strip()
        info = lines[3].split()
        n_atoms = int(info[0])
        self.atoms = []
        for i in range(5, 5 + n_atoms):
            atom_info = lines[i].split()
            ele = ELEMENTS[int(atom_info[0])]
            coords = np.array([float(x) for x in atom_info[1:4]])
            self.atoms.append(
                Atom(
                    element=ele,
                    coords=coords,
                    name=str(i-4),
                )
            )
        
        i = n_atoms + 6
        line = lines[i]
        self.other["shell_to_atom"] = []
        self.other["momentum_label"] = []
        self.other["funcs_per_shell"] = []
        self.other["start_ndx"] = []
        self.other["n_prim_per_shell"] = []
        while "---" not in line:
            info = line.split()
            ndx = int(info[0])
            funcs = int(info[1])
            start_ndx = int(info[2])
            n_prim = int(info[3])
            self.other["shell_to_atom"].extend([ndx for j in range(0, funcs)])
            self.other["funcs_per_shell"].append(funcs)
            self.other["start_ndx"].append(start_ndx)
            self.other["n_prim_per_shell"].append(n_prim)
            i += 1
            line = lines[i]
            momentum_labels = [int(x) for x in line.split()]
            self.other["momentum_label"].extend(momentum_labels)
            i += 1
            line = lines[i]
        
        i += 1
        self.other["exponents"] = []
        line = lines[i]
        while line.strip() != "":
            exponents = [float(x) for x in line.split()]
            self.other["exponents"].extend(exponents)
            i += 1
            line = lines[i]

        i += 1
        self.other["s_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["s_coeff"].extend(coeff)
            i += 1
            line = lines[i]
        
        i += 1
        self.other["p_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["p_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["d_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["d_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["f_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["f_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["g_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["g_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        if nbo_name is not None:
            self._read_nbo_coeffs(nbo_name)


@addlogger
class Orbitals:
    """
    stores functions for the shells in a basis set
    for evaluation at arbitrary points
    attributes:
    basis_functions - list(len=n_shell) of lists(len=n_prim_per_shell)
                      of functions
                      function takes the arguments:
                      r2 - float array like, squared distance from the
                           shell's center to each point being evaluated
                      x - float or array like, distance from the shell's
                          center to the point(s) being evaluated along
                          the x axis
                      y and z - same as x for the corresponding axis
                      mo_coeffs - list(len=funcs_per_shell), MO coefficients
                                  for the functions in this shell (e.g. 3
                                  coefficients for the p shell); order
                                  might depend on input file format
                                  for example, FCHK files will be px, py, pz
                                  ORCA files will be pz, px, py
    funcs_per_shell - list(len=n_shell), number of basis functions for
                      each shell
    alpha_coefficients - array(shape=(n_mos, n_mos)), coefficients of
                         molecular orbitals for alpha electrons
    beta_coefficients - same as alpha_coefficients for beta electrons
    shell_coords - array(shape=(n_shells, 3)), coordinates of each shell
                   in Angstroms
    shell_types - list(str, len=n_shell), type of each shell (e.g. s,
                  p, sp, 5d, 6d...)
    n_shell - number of shells
    n_prim_per_shell - list(len=n_shell), number of primitives per shell
    n_mos - number of molecular orbitals
    exponents - array, exponents for primitives in Eh
                each shell
    alpha_nrgs - array(len=n_mos), energy of alpha MO's
    beta_nrgs - array(len=n_mos), energy of beta MO's
    contraction_coeff - array, contraction coefficients for each primitive
                        in each shell
    n_alpha - int, number of alpha electrons
    n_beta - int, number of beta electrons
    """

    LOG = None

    def __init__(self, filereader):
        if filereader.file_type == "fchk":
            self._load_fchk_data(filereader)
        elif filereader.file_type == "out":
            self._load_orca_out_data(filereader)
        elif filereader.file_type == "47" or filereader.file_type == "31":
            self._load_nbo_data(filereader)
        else:
            raise NotImplementedError(
                "cannot load orbital info from %s files" % filereader.file_type
            )

    def _load_fchk_data(self, filereader):
        from scipy.special import factorial2

        if "Coordinates of each shell" in filereader.other:
            self.shell_coords = np.reshape(
                filereader.other["Coordinates of each shell"],
                (len(filereader.other["Shell types"]), 3),
            )
        else:
            center_coords = []
            for ndx in filereader.other["Shell to atom map"]:
                center_coords.append(filereader.atoms[ndx - 1].coords)
            self.center_coords = np.array(center_coords)
        self.shell_coords *= UNIT.A0_TO_BOHR
        self.contraction_coeff = filereader.other["Contraction coefficients"]
        self.exponents = filereader.other["Primitive exponents"]
        self.n_prim_per_shell = filereader.other["Number of primitives per shell"]
        self.alpha_nrgs = filereader.other["Alpha Orbital Energies"]
        self.beta_nrgs = None
        if "Beta Orbital Energies" in filereader.other:
            self.beta_nrgs = filereader.other["Beta Orbital Energies"]

        self.funcs_per_shell = []

        def gau_norm(a, l):
            """
            normalization for gaussian primitives that depends on
            the exponential (a) and the total angular momentum (l)
            """
            t1 = np.sqrt((2 * a) ** (l + 3 / 2)) / (np.pi ** (3.0 / 4))
            t2 = np.sqrt(2 ** l / factorial2(2 * l - 1))
            return t1 * t2

        # get functions for norm of s, p, 5d, and 7f
        s_norm = lambda a, l=0: gau_norm(a, l)
        p_norm = lambda a, l=1: gau_norm(a, l)
        d_norm = lambda a, l=2: gau_norm(a, l)
        f_norm = lambda a, l=3: gau_norm(a, l)
        g_norm = lambda a, l=4: gau_norm(a, l)
        h_norm = lambda a, l=5: gau_norm(a, l)
        i_norm = lambda a, l=6: gau_norm(a, l)

        self.basis_functions = list()

        self.n_mos = 0
        self.shell_types = []
        shell_i = 0
        for n_prim, shell in zip(
            self.n_prim_per_shell,
            filereader.other["Shell types"],
        ):
            exponents = self.exponents[shell_i : shell_i + n_prim]
            con_coeff = self.contraction_coeff[shell_i : shell_i + n_prim]

            if shell == 0:
                # s functions
                self.shell_types.append("s")
                self.n_mos += 1
                self.funcs_per_shell.append(1)
                norms = s_norm(exponents)
                if n_prim > 1:
                    def s_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        return mo_coeffs[0] * np.dot(con_coeff * norms, e_r2)
                else:
                    def s_shell(
                        r2, x, y, z, mo_coeff,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        return mo_coeff * con_coeff * norms * e_r2
                self.basis_functions.append(s_shell)

            elif shell == 1:
                # p functions
                self.shell_types.append("p")
                self.n_mos += 3
                self.funcs_per_shell.append(3)
                norms = p_norm(exponents)
                if n_prim > 1:
                    def p_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            res += mo_coeffs[0] * x
                        if mo_coeffs[1] != 0:
                            res += mo_coeffs[1] * y
                        if mo_coeffs[2] != 0:
                            res += mo_coeffs[2] * z
                        return res * s_val
                else:
                    def p_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        s_val = con_coeff * norms * e_r2
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            res += mo_coeffs[0] * x
                        if mo_coeffs[1] != 0:
                            res += mo_coeffs[1] * y
                        if mo_coeffs[2] != 0:
                            res += mo_coeffs[2] * z
                        return res * s_val
                self.basis_functions.append(p_shell)

            elif shell == -1:
                # s=p functions
                self.shell_types.append("sp")
                self.n_mos += 4
                self.funcs_per_shell.append(4)
                norm_s = s_norm(exponents)
                norm_p = p_norm(exponents)
                sp_coeff = filereader.other["P(S=P) Contraction coefficients"][shell_i: shell_i + n_prim]
                if n_prim > 1:
                    def sp_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents,
                        s_coeff=con_coeff,
                        p_coeff=sp_coeff,
                        s_norms=norm_s,
                        p_norms=norm_p,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        sp_val_s = np.dot(s_coeff * s_norms, e_r2)
                        sp_val_p = np.dot(p_coeff * p_norms, e_r2)
                        s_res = np.zeros(len(r2))
                        p_res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            s_res += mo_coeffs[0]
                        if mo_coeffs[1] != 0:
                            p_res += mo_coeffs[1] * x
                        if mo_coeffs[2] != 0:
                            p_res += mo_coeffs[2] * y
                        if mo_coeffs[3] != 0:
                            p_res += mo_coeffs[3] * z
                        return s_res * sp_val_s + p_res * sp_val_p
                else:
                    def sp_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents,
                        s_coeff=con_coeff,
                        p_coeff=sp_coeff,
                        s_norms=norm_s,
                        p_norms=norm_p,
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        sp_val_s = s_coeff * s_norms * e_r2
                        sp_val_p = p_coeff * p_norms * e_r2
                        s_res = np.zeros(len(r2))
                        p_res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            s_res += mo_coeffs[0]
                        if mo_coeffs[1] != 0:
                            p_res += mo_coeffs[1] * x
                        if mo_coeffs[2] != 0:
                            p_res += mo_coeffs[2] * y
                        if mo_coeffs[3] != 0:
                            p_res += mo_coeffs[3] * z
                        return s_res * sp_val_s + p_res * sp_val_p
                self.basis_functions.append(sp_shell)

            elif shell == 2:
                # cartesian d functions
                self.shell_types.append("6d")
                self.n_mos += 6
                self.funcs_per_shell.append(6)
                norms = d_norm(exponents)
                if n_prim > 1:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            xx = x ** 2
                            res += mo_coeffs[0] * xx
                        if mo_coeffs[1] != 0:
                            yy = y ** 2
                            res += mo_coeffs[1] * yy
                        if mo_coeffs[2] != 0:
                            zz = z ** 2
                            res += mo_coeffs[2] * zz
                        if mo_coeffs[3] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[3] * xy
                        if mo_coeffs[4] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[4] * xz
                        if mo_coeffs[5] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[5] * yz
                        return res * s_val
                else:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        s_val = con_coeff * norms * e_r2
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            xx = x ** 2
                            res += mo_coeffs[0] * xx
                        if mo_coeffs[1] != 0:
                            yy = y ** 2
                            res += mo_coeffs[1] * yy
                        if mo_coeffs[2] != 0:
                            zz = z ** 2
                            res += mo_coeffs[2] * zz
                        if mo_coeffs[3] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[3] * xy
                        if mo_coeffs[4] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[4] * xz
                        if mo_coeffs[5] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[5] * yz
                        return res * s_val
                self.basis_functions.append(d_shell)

            elif shell == -2:
                # pure d functions
                self.shell_types.append("5d")
                self.n_mos += 5
                self.funcs_per_shell.append(5)
                norms = d_norm(exponents)
                if n_prim > 1:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z2r2 = 0.5 * (3 * z ** 2 - r2)
                            res += mo_coeffs[0] * z2r2
                        if mo_coeffs[1] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[1] * xz
                        if mo_coeffs[2] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[2] * yz
                        if mo_coeffs[3] != 0:
                            x2y2 = np.sqrt(3) * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2y2
                        if mo_coeffs[4] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[4] * xy
                        return res * s_val
                else:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        s_val = con_coeff * norms * e_r2
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z2r2 = 0.5 * (3 * z ** 2 - r2)
                            res += mo_coeffs[0] * z2r2
                        if mo_coeffs[1] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[1] * xz
                        if mo_coeffs[2] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[2] * yz
                        if mo_coeffs[3] != 0:
                            x2y2 = np.sqrt(3) * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2y2
                        if mo_coeffs[4] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[4] * xy
                        return res * s_val
                self.basis_functions.append(d_shell)

            elif shell == 3:
                # 10f functions
                self.shell_types.append("10f")
                self.n_mos += 10
                self.funcs_per_shell.append(10)
                norms = f_norm(exponents)

                def f_shell(
                    r2,
                    x,
                    y,
                    z,
                    mo_coeffs,
                    alpha=exponents,
                    con_coeff=con_coeff,
                    norms=norms,
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    # ** 3 takes ~6x longer than x * x * x or x ** 2 * x
                    if mo_coeffs[0] != 0:
                        xxx = x * x * x
                        res += mo_coeffs[0] * xxx
                    if mo_coeffs[1] != 0:
                        yyy = y * y * y
                        res += mo_coeffs[1] * yyy
                    if mo_coeffs[2] != 0:
                        zzz = z * z * z
                        res += mo_coeffs[2] * zzz
                    if mo_coeffs[3] != 0:
                        xyy = np.sqrt(5) * x * y ** 2
                        res += mo_coeffs[3] * xyy
                    if mo_coeffs[4] != 0:
                        xxy = np.sqrt(5) * x ** 2 * y
                        res += mo_coeffs[4] * xxy
                    if mo_coeffs[5] != 0:
                        xxz = np.sqrt(5) * x ** 2 * z
                        res += mo_coeffs[5] * xxz
                    if mo_coeffs[6] != 0:
                        xzz = np.sqrt(5) * x * z ** 2
                        res += mo_coeffs[6] * xzz
                    if mo_coeffs[7] != 0:
                        yzz = np.sqrt(5) * y * z ** 2
                        res += mo_coeffs[7] * yzz
                    if mo_coeffs[8] != 0:
                        yyz = np.sqrt(5) * y ** 2 * z
                        res += mo_coeffs[8] * yyz
                    if mo_coeffs[9] != 0:
                        xyz = np.sqrt(15) * x * y * z
                        res += mo_coeffs[9] * xyz
                    return res * s_val

                self.basis_functions.append(f_shell)

            elif shell == -3:
                # pure f functions
                self.shell_types.append("7f")
                self.n_mos += 7
                self.funcs_per_shell.append(7)
                norms = f_norm(exponents)
                def f_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeff=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    if mo_coeffs[0] != 0:
                        z3zr2 = z * (5 * z ** 2 - 3 * r2) / 2
                        res += mo_coeffs[0] * z3zr2
                    if mo_coeffs[1] != 0:
                        xz2xr2 = np.sqrt(3) * x * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        res += mo_coeffs[1] * xz2xr2
                    if mo_coeffs[2] != 0:
                        yz2yr2 = np.sqrt(3) * y * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        res += mo_coeffs[2] * yz2yr2
                    if mo_coeffs[3] != 0:
                        x2zr2z = np.sqrt(15) * z * (x ** 2 - y ** 2) / 2
                        res += mo_coeffs[3] * x2zr2z
                    if mo_coeffs[4] != 0:
                        xyz = np.sqrt(15) * x * y * z
                        res += mo_coeffs[4] * xyz
                    if mo_coeffs[5] != 0:
                        x3r2x = np.sqrt(5) * x * (x ** 2 - 3 * y ** 2) / (2 * np.sqrt(2))
                        res += mo_coeffs[5] * x3r2x
                    if mo_coeffs[6] != 0:
                        x2yy3 = np.sqrt(5) * y * (3 * x ** 2 - y ** 2) / (2 * np.sqrt(2))
                        res += mo_coeffs[6] * x2yy3
                    return res * s_val
                self.basis_functions.append(f_shell)
                
            # elif shell == 4:
            elif False:
                # TODO: validate these - order might be wrong
                self.shell_types.append("15g")
                self.n_mos += 15
                self.funcs_per_shell.append(15)
                norms = g_norm(exponents)
                def g_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    if mo_coeffs[0] != 0:
                        x4 = (x ** 2) ** 2
                        res += mo_coeffs[0] * x4
                    if mo_coeffs[1] != 0:
                        x3y = (x ** 3) * x * y
                        res += mo_coeffs[1] * x3y
                    if mo_coeffs[2] != 0:
                        x3z = (x ** 3) * x * z
                        res += mo_coeffs[2] * x3z
                    if mo_coeffs[3] != 0:
                        x2y2 = (x ** 2) * (y ** 2)
                        res += mo_coeffs[3] * x2y2
                    if mo_coeffs[4] != 0:
                        x2yz = (x ** 2) * y * z
                        res += mo_coeffs[4] * x2yz
                    if mo_coeffs[5] != 0:
                        x2z2 = (x ** 2) * (z ** 2)
                        res += mo_coeffs[5] * x2z2
                    if mo_coeffs[6] != 0:
                        xy3 = x * y * y ** 2
                        res += mo_coeffs[6] * xy3
                    if mo_coeffs[7] != 0:
                        xy2z = x * z * y ** 2
                        res += mo_coeffs[7] * xy2z
                    if mo_coeffs[8] != 0:
                        xyz2 = x * y * z ** 2
                        res += mo_coeffs[8] * xyz2
                    if mo_coeffs[9] != 0:
                        xz3 = x * z * z ** 2
                        res += mo_coeffs[9] * xz3
                    if mo_coeffs[10] != 0:
                        y4 = (y ** 2) ** 2
                        res += mo_coeffs[10] * y4
                    if mo_coeffs[11] != 0:
                        y3z = (y ** 2) * y * z
                        res += mo_coeffs[11] * y3z
                    if mo_coeffs[12] != 0:
                        y2z2 = (y * z) ** 2
                        res += mo_coeffs[12] * y2z2
                    if mo_coeffs[13] != 0:
                        yz3 = y * z * z ** 2
                        res += mo_coeffs[13] * yz3
                    if mo_coeffs[14] != 0:
                        z4 = (z ** 2) ** 2
                        res += mo_coeffs[14] * z4

                    return res * s_val
                self.basis_functions.append(g_shell)

            elif shell == -4:
                self.shell_types.append("9g")
                self.n_mos += 9
                self.funcs_per_shell.append(9)
                norms = g_norm(exponents)
                def g_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    if mo_coeffs[0] != 0:
                        z4 = (35 * (z ** 4) - 30 * (r2 * z ** 2) + 3 * r2 ** 2) / 8
                        res += mo_coeffs[0] * z4
                    if mo_coeffs[1] != 0:
                        z3x = np.sqrt(10) * (x * z * (7 * z ** 2 - 3 * r2)) / 4
                        res += mo_coeffs[1] * z3x
                    if mo_coeffs[2] != 0:
                        z3y = np.sqrt(10) * (y * z * (7 * z ** 2 - 3 * r2)) / 4
                        res += mo_coeffs[2] * z3y
                    if mo_coeffs[3] != 0:
                        z2x2y2 = np.sqrt(5) * (x ** 2 - y ** 2) * (7 * z ** 2 - r2) / 4
                        res += mo_coeffs[3] * z2x2y2
                    if mo_coeffs[4] != 0:
                        z2xy = np.sqrt(5) * x * y * (7 * z ** 2 - r2) / 2
                        res += mo_coeffs[4] * z2xy
                    if mo_coeffs[5] != 0:
                        zx3 = np.sqrt(70) * x * z * (x ** 2 - 3 * y ** 2) / 4
                        res += mo_coeffs[5] * zx3
                    if mo_coeffs[6] != 0:
                        zy3 = np.sqrt(70) * z * y * (3 * x ** 2 - y ** 2) / 4
                        res += mo_coeffs[6] * zy3
                    if mo_coeffs[7] != 0:
                        x2 = x ** 2
                        y2 = y ** 2
                        x4y4 = np.sqrt(35) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)) / 8
                        res += mo_coeffs[7] * x4y4
                    if mo_coeffs[8] != 0:
                        xyx2y2 = np.sqrt(35) * x * y * (x ** 2 - y ** 2) / 2
                        res += mo_coeffs[8] * xyx2y2

                    return res * s_val
                self.basis_functions.append(g_shell)

            elif shell == -5:
                self.shell_types.append("11h")
                self.n_mos += 11
                self.funcs_per_shell.append(11)
                norms = h_norm(exponents)
                def h_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    z2 = z ** 2
                    if mo_coeffs[0] != 0:
                        z5z3r2zr4 = z * (63 * z2 ** 2 - 70 * z2 * r2 + 15 * r2 ** 2) / 8
                        res += mo_coeffs[0] * z5z3r2zr4
                    if mo_coeffs[1] != 0:
                        xz4xz2r2xr4 = np.sqrt(15) * x * (21 * z2 ** 2 - 14 * z2 * r2 + r2 ** 2) / 8
                        res += mo_coeffs[1] * xz4xz2r2xr4
                    if mo_coeffs[2] != 0:
                        yz4yz2r2yr4 = np.sqrt(15) * y * (21 * z2 ** 2 - 14 * z2 * r2 + r2 ** 2) / 8
                        res += mo_coeffs[2] * yz4yz2r2yr4
                    if mo_coeffs[3] != 0:
                        x2y3z3zr2 = np.sqrt(105) * (x ** 2 - y ** 2) * (3 * z2 - r2) * z / 4
                        res += mo_coeffs[3] * x2y3z3zr2
                    if mo_coeffs[4] != 0:
                        xyz3zr2 = np.sqrt(105) * x * y * z * (3 * z2 - r2) / 2
                        res += mo_coeffs[4] * xyz3zr2
                    if mo_coeffs[5] != 0:
                        xx2y2z2r2 = 35 * x * (x ** 2 - 3 * y ** 2) * (9 * z2 - r2) / (8 * np.sqrt(70))
                        res += mo_coeffs[5] * xx2y2z2r2
                    if mo_coeffs[6] != 0:
                        yx2y2z2r2 = 35 * y * (3 * x ** 2 - y ** 2) * (9 * z2 - r2) / (8 * np.sqrt(70))
                        res += mo_coeffs[6] * yx2y2z2r2
                    if mo_coeffs[7] != 0:
                        zx4x2y2y4 = 105 * z * ((x ** 2) ** 2 - 6 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(35))
                        res += mo_coeffs[7] * zx4x2y2y4
                    if mo_coeffs[8] != 0:
                        zx3yxy3 = 105 * x * y * z * (4 * x ** 2 - 4 * y ** 2) / (8 * np.sqrt(35))
                        res += mo_coeffs[8] * zx3yxy3
                    if mo_coeffs[9] != 0:
                        xx4y2x2y4 = 21 * x * ((x ** 2) ** 2 - 10 * (x * y) ** 2 + 5 * (y ** 2) ** 2) / (8 * np.sqrt(14))
                        res += mo_coeffs[9] * xx4y2x2y4
                    if mo_coeffs[10] != 0:
                        yx4y2x2y4 = 21 * y * (5 * (x ** 2) ** 2 - 10 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(14))
                        res += mo_coeffs[10] * yx4y2x2y4

                    return res * s_val
                self.basis_functions.append(h_shell)

            elif shell == -6:
                self.shell_types.append("13i")
                self.n_mos += 13
                self.funcs_per_shell.append(13)
                norms = i_norm(exponents)
                def i_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    z2 = z ** 2
                    if mo_coeffs[0] != 0:
                        z6z4r2z2r4r6 = (231 * z2 * z2 ** 2 - 315 * z2 ** 2 * r2 + 105 * z2 * r2 ** 2 - 5 * r2 * r2 ** 2) / 16
                        res += mo_coeffs[0] * z6z4r2z2r4r6
                    if mo_coeffs[1] != 0:
                        xz5z3r2zr4 = np.sqrt(21) * x * z * (33 * z2 ** 2 - 30 * z2 * r2 + 5 * r2 ** 2) / 8
                        res += mo_coeffs[1] * xz5z3r2zr4
                    if mo_coeffs[2] != 0:
                        yz5z3r2zr4 = np.sqrt(21) * y * z * (33 * z2 ** 2 - 30 * z2 * r2 + 5 * r2 ** 2) / 8
                        res += mo_coeffs[2] * yz5z3r2zr4
                    if mo_coeffs[3] != 0:
                        x2y2z4z2r2r3 = 105 * (x ** 2 - y ** 2) * (33 * z2 ** 2 - 18 * z2 * r2 + r2 ** 2) / (16 * np.sqrt(210))
                        res += mo_coeffs[3] * x2y2z4z2r2r3
                    if mo_coeffs[4] != 0:
                        xyz4z2r2r4 = 105 * x * y * (33 * z2 ** 2 - 18 * z2 * r2 + r2 ** 2) / (8 * np.sqrt(210))
                        res += mo_coeffs[4] * xyz4z2r2r4
                    if mo_coeffs[5] != 0:
                        xx2y2z3zr2 = 105 * x * z * (x ** 2 - 3 * y ** 2) * (11 * z2 - 3 * r2) / (8 * np.sqrt(210))
                        res += mo_coeffs[5] * xx2y2z3zr2
                    if mo_coeffs[6] != 0:
                        yx2y2z3zr2 = 105 * y * z * (3 * x ** 2 - y ** 2) * (11 * z2 - 3 * r2) / (8 * np.sqrt(210))
                        res += mo_coeffs[6] * yx2y2z3zr2
                    if mo_coeffs[7] != 0:
                        x4x2y2y4z2r2 = np.sqrt(63) * ((x ** 2) ** 2 - 6 * (x * y) ** 2 + (y ** 2) ** 2) * (11 * z2 - r2) / 16
                        res += mo_coeffs[7] * x4x2y2y4z2r2
                    if mo_coeffs[8] != 0:
                        xyx2y2z2r2 = np.sqrt(63) * x * y * (x ** 2 - y ** 2) * (11 * z2 - r2) / 4
                        res += mo_coeffs[8] * xyx2y2z2r2
                    if mo_coeffs[9] != 0:
                        xzx4x2y2y4 = 231 * x * z * ((x ** 2) ** 2 - 10 * (x * y) ** 2 + 5 * (y ** 2) ** 2) / (8 * np.sqrt(154))
                        res += mo_coeffs[9] * xzx4x2y2y4
                    if mo_coeffs[10] != 0:
                        yzx4x2y2y4 = 231 * y * z * (5 * (x ** 2) ** 2 - 10 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(154))
                        res += mo_coeffs[10] * yzx4x2y2y4
                    if mo_coeffs[11] != 0:
                        x6x4y2x2y4y6 = 231 * ((x * x ** 2) ** 2 - 15 * (x ** 2 * y) ** 2 + 15 * (x * y ** 2) ** 2 - (y * y ** 2) ** 2) / (16 * np.sqrt(462))
                        res += mo_coeffs[11] * x6x4y2x2y4y6
                    if mo_coeffs[12] != 0:
                        yx5x3y3xy5 = 231 * x * y * (6 * (x ** 2) ** 2 - 20 * (x * y) ** 2 + 6 * (y ** 2) ** 2) / (16 * np.sqrt(462))
                        res += mo_coeffs[12] * yx5x3y3xy5

                    return res * s_val
                self.basis_functions.append(i_shell)

            else:
                self.LOG.warning("cannot parse shell with type %i" % shell)

            shell_i += n_prim

        self.alpha_coefficients = np.reshape(
            filereader.other["Alpha MO coefficients"],
            (self.n_mos, self.n_mos),
        )
        if "Beta MO coefficients" in filereader.other:
            self.beta_coefficients = np.reshape(
                filereader.other["Beta MO coefficients"],
                (self.n_mos, self.n_mos),
            )
        else:
            self.beta_coefficients = None
        self.n_alpha = filereader.other["Number of alpha electrons"]
        if "Number of beta electrons" in filereader.other:
            self.n_beta = filereader.other["Number of beta electrons"]

    def _load_nbo_data(self, filereader):
        self.basis_functions = []
        self.exponents = np.array(filereader.other["exponents"])
        self.alpha_coefficients = np.array(filereader.other["alpha_coefficients"])
        self.beta_coefficients = None
        self.shell_coords = []
        self.funcs_per_shell = []
        self.shell_types = []
        self.n_shell = len(filereader.other["n_prim_per_shell"])
        self.alpha_nrgs = [0 for x in self.alpha_coefficients]
        self.n_mos = len(self.alpha_coefficients)
        self.n_alpha = 0
        self.n_beta = 0
        self.beta_nrgs = None

        label_i = 0
        # NBO includes normalization constant with the contraction coefficient
        # so we don't have a gau_norm function like gaussian or orca
        for n_prim, n_funcs, shell_i in zip(
            filereader.other["n_prim_per_shell"],
            filereader.other["funcs_per_shell"],
            filereader.other["start_ndx"],
            
        ):
            shell_i -= 1
            exponents = self.exponents[shell_i: shell_i + n_prim]
            shell_funcs = []
            con_coeffs = []
            shell_type = []
            self.funcs_per_shell.append(n_funcs)
            self.shell_coords.append(
                filereader.atoms[filereader.other["shell_to_atom"][label_i] - 1].coords
            )
            for i in range(0, n_funcs):
                shell = filereader.other["momentum_label"][label_i]
                label_i += 1
                # XXX: each function is treated as a different
                # shell because NBO allows them to be in any order
                # I think that technically means the functions in
                # the d shell for example don't need to be next
                # to each other
                if shell < 100:
                    shell_type.append("s")
                    # s - shell can be 1 or 51
                    con_coeff = filereader.other["s_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    def s_shell(
                        r2, x, y, z, s_val
                    ):
                        return s_val
                    shell_funcs.append(s_shell)
                elif shell < 200:
                    # p - shell can be 101, 102, 103, 151, 152, 153
                    con_coeff = filereader.other["p_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 101 or shell == 151:
                        shell_type.append("px")
                        def px_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x
                        shell_funcs.append(px_shell)
                    elif shell == 102 or shell == 152:
                        shell_type.append("py")
                        def py_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y
                        shell_funcs.append(py_shell)
                    elif shell == 103 or shell == 153:
                        shell_type.append("pz")
                        def pz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z
                        shell_funcs.append(pz_shell)
                elif shell < 300:
                    con_coeff = filereader.other["d_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 201:
                        shell_type.append("dxx")
                        def dxx_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x
                        shell_funcs.append(dxx_shell)
                    elif shell == 202:
                        shell_type.append("dxy")
                        def dxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y
                        shell_funcs.append(dxy_shell)
                    elif shell == 203:
                        shell_type.append("dxz")
                        def dxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * z
                        shell_funcs.append(dxz_shell)
                    elif shell == 204:
                        shell_type.append("dyy")
                        def dyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y
                        shell_funcs.append(dyy_shell)
                    elif shell == 205:
                        shell_type.append("dyz")
                        def dyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * z
                        shell_funcs.append(dyz_shell)
                    elif shell == 206:
                        shell_type.append("dzz")
                        def dzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * z
                        shell_funcs.append(dzz_shell)
                    elif shell == 251:
                        shell_type.append("5dxy")
                        def dxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * x * y
                        shell_funcs.append(dxy_shell)
                    elif shell == 252:
                        shell_type.append("5dxz")
                        def dxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * x * z
                        shell_funcs.append(dxz_shell)
                    elif shell == 253:
                        shell_type.append("5dyz")
                        def dyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * y * z
                        shell_funcs.append(dyz_shell)
                    elif shell == 254:
                        shell_type.append("5dx2-y2")
                        def dx2y2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * (x ** 2 - y ** 2) / 2
                        shell_funcs.append(dx2y2_shell)
                    elif shell == 255:
                        shell_type.append("5dz2")
                        def dz2_shell(
                            r2, x, y, z, s_val
                        ):
                            return (3 * z ** 2 - r2) * s_val / 2
                        shell_funcs.append(dz2_shell)
                elif shell < 400:
                    con_coeff = filereader.other["f_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 301:
                        shell_type.append("fxxx")
                        def fxxx_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x
                        shell_funcs.append(fxxx_shell)
                    if shell == 302:
                        shell_type.append("fxxy")
                        def fxxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * y
                        shell_funcs.append(fxxy_shell)
                    if shell == 303:
                        shell_type.append("fxxz")
                        def fxxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * z
                        shell_funcs.append(fxxz_shell)
                    if shell == 304:
                        shell_type.append("fxyy")
                        def fxyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * y
                        shell_funcs.append(fxyy_shell)
                    if shell == 305:
                        shell_type.append("fxyz")
                        def fxyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * z
                        shell_funcs.append(fxyz_shell)
                    if shell == 306:
                        shell_type.append("fxzz")
                        def fxzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * z * z
                        shell_funcs.append(fxzz_shell)
                    if shell == 307:
                        shell_type.append("fyyy")
                        def fyyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * y
                        shell_funcs.append(fyyy_shell)
                    if shell == 308:
                        shell_type.append("fyyz")
                        def fyyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * z
                        shell_funcs.append(fyyz_shell)
                    if shell == 309:
                        shell_type.append("fyzz")
                        def fyzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * z * z
                        shell_funcs.append(fyzz_shell)
                    if shell == 310:
                        shell_type.append("fzzz")
                        def fzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * z * z
                        shell_funcs.append(fzzz_shell)
                    if shell == 351:
                        shell_type.append("7fz3-zr2")
                        def fz3zr2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * (5 * z ** 2 - 3 * r2) / 2
                        shell_funcs.append(fz3zr2_shell)
                    if shell == 352:
                        shell_type.append("7fxz2-xr2")
                        def fxz2xr2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * x * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        shell_funcs.append(fxz2xr2_shell)
                    if shell == 353:
                        shell_type.append("7fyz2-yr2")
                        def fyz2yr2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        shell_funcs.append(fyz2yr2_shell)
                    if shell == 354:
                        shell_type.append("7fzx2-zy2")
                        def fzx2zy2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(15) * s_val * z * (x ** 2 - y ** 2) / 2
                        shell_funcs.append(fzx2zy2_shell)
                    if shell == 355:
                        shell_type.append("7fxyz")
                        def fxyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(15) * s_val * x * y * z
                        shell_funcs.append(fxyz_shell)
                    if shell == 356:
                        shell_type.append("7fx3-xy2")
                        def fx3xy2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(5) * s_val * x * (x ** 2 - 3 * y ** 2) / (2 * np.sqrt(2))
                        shell_funcs.append(fx3xy2_shell)
                    if shell == 357:
                        shell_type.append("7fyx2-y3")
                        def fyx2y3_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(5) * s_val * y * (3 * x ** 2 - y ** 2) / (2 * np.sqrt(2))
                        shell_funcs.append(fyx2y3_shell)
                elif shell < 500:
                    # I can't tell what NBO does with g orbitals
                    # I don't have any reference to compare to
                    self.LOG.warning(
                        "g shell results have not been verified for NBO\n"
                        "any LCAO's may be invalid"
                    )
                    con_coeff = filereader.other["g_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 401:
                        shell_type.append("gxxxx")
                        def gxxxx_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x * x
                        shell_funcs.append(gxxxx_shell)
                    if shell == 402:
                        shell_type.append("gxxxy")
                        def gxxxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x * y
                        shell_funcs.append(gxxxy_shell)
                    if shell == 403:
                        shell_type.append("gxxxz")
                        def gxxxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x * z
                        shell_funcs.append(gxxxz_shell)
                    if shell == 404:
                        shell_type.append("gxxyy")
                        def gxxyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * y * y
                        shell_funcs.append(gxxyy_shell)
                    if shell == 405:
                        shell_type.append("gxxyz")
                        def gxxyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * y * z
                        shell_funcs.append(gxxyz_shell)
                    if shell == 406:
                        shell_type.append("gxxzz")
                        def gxxzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * z * z
                        shell_funcs.append(gxxzz_shell)
                    if shell == 407:
                        shell_type.append("gxyyy")
                        def gxyyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * y * y
                        shell_funcs.append(gxyyy_shell)
                    if shell == 408:
                        shell_type.append("gxyyz")
                        def gxyyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * y * z
                        shell_funcs.append(gxyyz_shell)
                    if shell == 409:
                        shell_type.append("gxyzz")
                        def gxyzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * z * z
                        shell_funcs.append(gxyzz_shell)
                    if shell == 410:
                        shell_type.append("gxzzz")
                        def gxzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * z * z * z
                        shell_funcs.append(gxzzz_shell)
                    if shell == 411:
                        shell_type.append("gyyyy")
                        def gyyyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * y * y
                        shell_funcs.append(gyyyy_shell)
                    if shell == 412:
                        shell_type.append("gyyyz")
                        def gyyyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * y * z
                        shell_funcs.append(gyyyz_shell)
                    if shell == 413:
                        shell_type.append("gyyzz")
                        def gyyzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * z * z
                        shell_funcs.append(gyyzz_shell)
                    if shell == 414:
                        shell_type.append("gyzzz")
                        def gyzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * z * z * z
                        shell_funcs.append(gyzzz_shell)
                    if shell == 415:
                        shell_type.append("gzzzz")
                        def gzzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * z * z * z
                        shell_funcs.append(gzzzz_shell)
                    if shell == 451:
                        shell_type.append("9gz4")
                        def gz4_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * (35 * (z ** 2) ** 2 - 30 * z ** 2 * r2 + 3 * r2 ** 2) / 8
                        shell_funcs.append(gz4_shell)
                    if shell == 452:
                        shell_type.append("9gz3x")
                        def gz3x_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(10) * (x * z * (7 * z ** 2 - 3 * r2)) / 4
                        shell_funcs.append(gz3x_shell)
                    if shell == 453:
                        shell_type.append("9gz3y")
                        def gz3y_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(10) * (y * z * (7 * z ** 2 - 3 * r2)) / 4
                        shell_funcs.append(gz3y_shell)
                    if shell == 454:
                        shell_type.append("9gz2x2-z2y2")
                        def gz2x2y2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(5) * (x ** 2 - y ** 2) * (7 * z ** 2 - r2) / 4
                        shell_funcs.append(gz2x2y2_shell)
                    if shell == 455:
                        shell_type.append("9gz2xy")
                        def gz2xy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(5) * x * y * (7 * z ** 2 - r2) / 2
                        shell_funcs.append(gz2xy_shell)
                    if shell == 456:
                        shell_type.append("9gzx3")
                        def gzx3_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(70) * x * z * (x ** 2 - 3 * y ** 2) / 4
                        shell_funcs.append(gzx3_shell)
                    if shell == 457:
                        shell_type.append("9gzy3")
                        def gzy3_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(70) * z * y * (3 * x ** 2 - y ** 2) / 4
                        shell_funcs.append(gzy3_shell)
                    if shell == 458:
                        shell_type.append("9gx4y4")
                        def gx4y4_shell(
                            r2, x, y, z, s_val
                        ):
                            x2 = x ** 2
                            y2 = y ** 2
                            return s_val * np.sqrt(35) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)) / 8
                        shell_funcs.append(gx4y4_shell)
                    if shell == 459:
                        shell_type.append("9gxyx2y2")
                        def gxyx2y2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(35) * x * y * (x ** 2 - y ** 2) / 2
                        shell_funcs.append(gxyx2y2_shell)
                else:
                    self.LOG.warning("cannot handle shells with momentum label %i" % shell)
        
            def eval_shells(
                r2, x, y, z, mo_coeffs,
                alpha=exponents,
                con_coeffs=con_coeffs,
                shell_funcs=shell_funcs
            ):
                e_r2 =  np.exp(np.outer(-alpha, r2))
                res = np.zeros(len(r2))
                last_con_coeff = None
                for mo_coeff, con_coeff, func in zip(mo_coeffs, con_coeffs, shell_funcs):
                    if mo_coeff == 0:
                        continue
                    if last_con_coeff is None or any(
                        x - y != 0 for x, y in zip(last_con_coeff, con_coeff)
                    ):
                        s_val = np.dot(con_coeff, e_r2)
                    last_con_coeff = con_coeff
                    res += mo_coeff * func(r2, x, y, z, s_val)
                return res
            self.basis_functions.append(eval_shells)
            self.shell_types.append(", ".join(shell_type))
        
        self.shell_coords = np.array(self.shell_coords)

    def _load_orca_out_data(self, filereader):
        self.shell_coords = []
        self.basis_functions = []
        self.alpha_nrgs = np.array(filereader.other["alpha_nrgs"])
        self.alpha_coefficients = np.array(filereader.other["alpha_coefficients"])
        if not filereader.other["beta_nrgs"]:
            self.beta_nrgs = None
            self.beta_coefficients = None
        else:
            self.beta_nrgs = np.array(filereader.other["beta_nrgs"])
            self.beta_coefficients = np.array(filereader.other["beta_coefficients"])
        self.shell_types = []
        self.funcs_per_shell = []
        self.n_aos = 0
        self.n_mos = 0

        def gau_norm(a, l):
            """
            normalization for gaussian primitives that depends on
            the exponential (a) and the total angular momentum (l)
            """
            t1 = np.sqrt((2 * a) ** (l + 3 / 2)) / (np.pi ** (3.0 / 4))
            t2 = np.sqrt(2 ** l / factorial2(2 * l - 1))
            return t1 * t2

        # get functions for norm of s, p, 5d, and 7f
        s_norm = lambda a, l=0: gau_norm(a, l)
        p_norm = lambda a, l=1: gau_norm(a, l)
        d_norm = lambda a, l=2: gau_norm(a, l)
        f_norm = lambda a, l=3: gau_norm(a, l)

        # ORCA order differs from FCHK in a few places:
        # pz, px, py instead of ox, py, pz
        # f(3xy^2 - x^3) instead of f(x^3 - 3xy^2)
        # f(y^3 - 3x^2y) instead of f(3x^2y - y^3)
        # ORCA doesn't seem to print the coordinates of each
        # shell, but they should be the same as the atom coordinates
        for atom in filereader.atoms:
            ele = atom.element
            for shell_type, n_prim, exponents, con_coeff in filereader.other[
                "basis_set_by_ele"
            ][ele]:
                self.shell_coords.append(atom.coords)
                exponents = np.array(exponents)
                con_coeff = np.array(con_coeff)
                if shell_type.lower() == "s":
                    self.shell_types.append("s")
                    self.funcs_per_shell.append(1)
                    self.n_aos += 1
                    norms = s_norm(exponents)
                    if n_prim > 1:
                        def s_shell(
                            r2, x, y, z, mo_coeff,
                            alpha=exponents,
                            con_coeff=con_coeff,
                            norms=norms
                        ):
                            e_r2 = np.exp(np.outer(-alpha, r2))
                            return mo_coeff[0] * np.dot(con_coeff * norms, e_r2)
                    else:
                        def s_shell(
                            r2, x, y, z, mo_coeff,
                            alpha=exponents,
                            con_coeff=con_coeff,
                            norms=norms
                        ):
                            e_r2 = np.exp(-alpha * r2)
                            return mo_coeff * con_coeff * norms * e_r2
                    self.basis_functions.append(s_shell)
                elif shell_type.lower() == "p":
                    self.shell_types.append("p")
                    self.funcs_per_shell.append(3)
                    self.n_aos += 3
                    norms = p_norm(exponents)

                    def p_shell(
                        r2,
                        x,
                        y,
                        z,
                        mo_coeffs,
                        alpha=exponents,
                        con_coeff=con_coeff,
                        norms=norms,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)

                        if isinstance(r2, float):
                            res = 0
                        else:
                            res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            res += mo_coeffs[0] * z
                        if mo_coeffs[1] != 0:
                            res += mo_coeffs[1] * x
                        if mo_coeffs[2] != 0:
                            res += mo_coeffs[2] * y

                        return res * s_val

                    self.basis_functions.append(p_shell)
                elif shell_type.lower() == "d":
                    self.shell_types.append("5d")
                    self.funcs_per_shell.append(5)
                    self.n_aos += 5
                    norms = d_norm(exponents)

                    def d_shell(
                        r2,
                        x,
                        y,
                        z,
                        mo_coeffs,
                        alpha=exponents,
                        con_coeff=con_coeff,
                        norms=norms,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        if isinstance(r2, float):
                            res = 0
                        else:
                            res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z2r2 = 0.5 * (3 * z * z - r2)
                            res += mo_coeffs[0] * z2r2
                        if mo_coeffs[1] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[1] * xz
                        if mo_coeffs[2] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[2] * yz
                        if mo_coeffs[3] != 0:
                            x2y2 = np.sqrt(3) * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2y2
                        if mo_coeffs[4] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[4] * xy
                        return res * s_val

                    self.basis_functions.append(d_shell)
                elif shell_type.lower() == "f":
                    self.shell_types.append("7f")
                    self.funcs_per_shell.append(7)
                    self.n_aos += 7
                    norms = f_norm(exponents)

                    def f_shell(
                        r2,
                        x,
                        y,
                        z,
                        mo_coeffs,
                        alpha=exponents,
                        con_coeff=con_coeff,
                        norms=norms,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        if isinstance(r2, float):
                            res = 0
                        else:
                            res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z3zr2 = z * (5 * z ** 2 - 3 * r2) / 2
                            res += mo_coeffs[0] * z3zr2
                        if mo_coeffs[1] != 0:
                            xz2xr2 = (
                                np.sqrt(3)
                                * x
                                * (5 * z ** 2 - r2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[1] * xz2xr2
                        if mo_coeffs[2] != 0:
                            yz2yr2 = (
                                np.sqrt(3)
                                * y
                                * (5 * z ** 2 - r2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[2] * yz2yr2
                        if mo_coeffs[3] != 0:
                            x2zr2z = np.sqrt(15) * z * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2zr2z
                        if mo_coeffs[4] != 0:
                            xyz = np.sqrt(15) * x * y * z
                            res += mo_coeffs[4] * xyz
                        if mo_coeffs[5] != 0:
                            x3r2x = (
                                np.sqrt(5)
                                * x
                                * (3 * y ** 2 - x ** 2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[5] * x3r2x
                        if mo_coeffs[6] != 0:
                            x2yy3 = (
                                np.sqrt(5)
                                * y
                                * (y ** 2 - 3 * x ** 2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[6] * x2yy3
                        return res * s_val

                    self.basis_functions.append(f_shell)
                else:
                    self.LOG.warning(
                        "cannot handle shell of type %s" % shell_type
                    )

        self.n_mos = len(self.alpha_coefficients)

        if "n_alpha" not in filereader.other:
            tot_electrons = sum(
                ELEMENTS.index(atom.element) for atom in filereader.atoms
            )
            self.n_beta = tot_electrons // 2
            self.n_alpha = tot_electrons - self.n_beta
        else:
            self.n_alpha = filereader.other["n_alpha"]
            self.n_beta = filereader.other["n_beta"]

    def _get_value(self, coords, arr):
        """returns value for the MO coefficients in arr"""
        ao = 0
        prev_center = None
        if coords.ndim == 1:
            val = 0
        else:
            val = np.zeros(len(coords))
        for coord, shell, n_func, shell_type in zip(
            self.shell_coords,
            self.basis_functions,
            self.funcs_per_shell,
            self.shell_types,
        ):
            # don't calculate distances until we find an AO
            # in this shell that has a non-zero MO coefficient
            if not np.count_nonzero(arr[ao : ao + n_func]):
                ao += n_func
                continue
            # print(shell_type, arr[ao : ao + n_func])
            # don't recalculate distances unless this shell's coordinates
            # differ from the previous
            if (
                prev_center is None
                or np.linalg.norm(coord - prev_center) > 1e-13
            ):
                prev_center = coord
                d_coord = (coords - coord) / UNIT.A0_TO_BOHR
                if coords.ndim == 1:
                    r2 = np.dot(d_coord, d_coord)
                else:
                    r2 = np.sum(d_coord * d_coord, axis=1)
            if coords.ndim == 1:
                res = shell(
                    r2,
                    d_coord[0],
                    d_coord[1],
                    d_coord[2],
                    arr[ao : ao + n_func],
                )
            else:
                res = shell(
                    r2,
                    d_coord[:, 0],
                    d_coord[:, 1],
                    d_coord[:, 2],
                    arr[ao : ao + n_func],
                )
            val += res
            ao += n_func
        return val

    def mo_value(self, mo, coords, alpha=True, n_jobs=1):
        """
        get the MO evaluated at the specified coords
        m - index of molecular orbital or an array of MO coefficients
        coords - numpy array of points (N,3) or (3,)
        alpha - use alpha coefficients (default)
        n_jobs - number of parallel threads to use
                 this is on top of NumPy's multithreading, so
                 if NumPy uses 8 threads and n_jobs=2, you can
                 expect to see 16 threads in use
        """
        # val is the running sum of MO values
        if alpha:
            coeff = self.alpha_coefficients
        else:
            coeff = self.beta_coefficients

        if isinstance(mo, int):
            coeff = coeff[mo]
        else:
            coeff = mo

        # calculate AO values for each shell at each point
        # multiply by the MO coefficient and add to val
        if n_jobs > 1:
            # get all shells grouped by coordinates
            # this reduces the number of times we will need to
            # calculate the distance from all the coords to
            # a shell's center
            prev_coords = []
            arrays = []
            ndx = 0
            add_to = 0
            for i, coord in enumerate(self.shell_coords):
                for j, prev_coord in enumerate(prev_coords):
                    if np.linalg.norm(coord - prev_coord) < 1e-13:
                        add_to = j
                        break
                else:
                    prev_coords.append(coord)
                    add_to = len(arrays)
                    arrays.append(np.zeros(self.n_mos))
                arrays[add_to][ndx : ndx + self.funcs_per_shell[i]] = coeff[
                    ndx : ndx + self.funcs_per_shell[i]
                ]
                ndx += self.funcs_per_shell[i]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                out = [executor.submit(self._get_value, coords, arr) for arr in arrays]
            return sum([shells.result() for shells in out])
        val = self._get_value(coords, coeff)
        return val

    def density_value(
        self,
        coords,
        n_jobs=1,
        alpha_occ=None,
        beta_occ=None,
        low_mem=False,
    ):
        """
        returns the eletron density
        coords - coordinates to calculate e density at
        n_jobs - number of concurrent threads to use in calculation
        alpha_occ - array of alpha occupancies
                    if not specified, defaults to lowest self.n_alpha
                    orbitals
        beta_occ - same at alpha_occ, but for beta electrons
        """
        if low_mem:
            return self._low_mem_density_value(
                coords,
                n_jobs=n_jobs,
                alpha_occ=alpha_occ,
                beta_occ=beta_occ,
            )
        
        # set default occupancy
        if alpha_occ is None:
            if not self.n_alpha:
                self.LOG.warning("number of alpha electrons was not read")
            alpha_occ = np.zeros(self.n_mos, dtype=int)
            alpha_occ[0:self.n_alpha] = 1
        if beta_occ is None:
            beta_occ = np.zeros(self.n_mos, dtype=int)
            beta_occ[0:self.n_beta] = 1
        
        # val is output data
        # func_vals is the value of each basis function
        # at all coordinates
        if coords.ndim == 1:
            val = 0
            func_vals = np.zeros(
                sum(len(f) for f in self.basis_functions), dtype="float32"
            )
        else:
            val = np.zeros(len(coords))
            func_vals = np.zeros(
                (self.n_mos, *coords.shape,)
            )

        # get values of basis functions at all points
        arrays = np.eye(self.n_mos)
        if n_jobs > 1:
            # get all shells grouped by coordinates
            # this reduces the number of times we will need to
            # calculate the distance from all the coords to
            # a shell's center
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                out = [executor.submit(self._get_value, coords, arr) for arr in arrays]
            data = np.array([shells.result() for shells in out])

        else:
            data = np.array([
                self._get_value(coords, arr) for arr in arrays
            ])

        # multiply values by orbital coefficients and square
        for i, occ in enumerate(alpha_occ):
            if occ == 0:
                continue
            val += occ * np.dot(data.T, self.alpha_coefficients[i]) ** 2
        
        if self.beta_coefficients is not None:
            for i, occ in enumerate(beta_occ):
                if occ == 0:
                    continue
                val += occ * np.dot(data.T, self.beta_coefficients[i]) ** 2
        else:
            val *= 2

        return val

    def _low_mem_density_value(
        self,
        coords,
        n_jobs=1,
        alpha_occ=None,
        beta_occ=None
    ):
        """
        returns the eletron density
        same at self.density_value, but uses less memory at
        the cost of performance
        """
        # set initial occupancies
        if alpha_occ is None:
            if not self.n_alpha:
                self.LOG.warning("number of alpha electrons was not read")
            alpha_occ = np.zeros(self.n_mos, dtype=int)
            alpha_occ[0:self.n_alpha] = 1
        if beta_occ is None:
            beta_occ = np.zeros(self.n_mos, dtype=int)
            beta_occ[0:self.n_beta] = 1

        # val is output array
        if coords.ndim == 1:
            val = 0
        else:
            val = np.zeros(len(coords), dtype="float32")

        # calculate each occupied orbital
        # square it can add to output
        if n_jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                out = [
                    executor.submit(self.mo_value, i, coords, n_jobs=1)
                    for i, occ in enumerate(alpha_occ) if occ != 0
                ]
            val += sum([occ * orbit.result() ** 2 for orbit, occ in zip(out, alpha_occ)])
            
            if self.beta_coefficients is not None:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_jobs
                ) as executor:
                    out = [
                        executor.submit(self.mo_value, i, coords, alpha=False, n_jobs=1)
                        for i, occ in enumerate(beta_occ) if occ != 0
                    ]
                val += sum([occ * orbit.result() ** 2 for orbit, occ in zip(out, beta_occ)])
            else:
                val *= 2
        
        else:
            for i in range(0, self.n_alpha):
                val += self.mo_value(i, coords) ** 2
            
            if self.beta_coefficients is not None:
                for i in range(0, self.n_beta):
                    val += self.mo_value(i, coords, alpha=False) ** 2
            else:
                val *= 2

        return val

    def fukui_donor_value(self, coords, delta=0.1, **kwargs):
        """
        orbital-weighted fukui donor function
        electron density change for removing an electron
        orbital weighting from DOI 10.1002/jcc.24699 accounts
        for nearly degenerate orbitals
        coords - coordinate to evaluate function at
        delta - parameter for weighting
        kwargs - passed to density_value
        """
        CITATION = "doi:10.1002/jcc.24699"
        self.LOG.citation(CITATION)

        if self.beta_coefficients is None:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            minus_e = np.zeros(self.n_mos)
            for i in range(0, self.n_alpha):
                minus_e[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )

            minus_e /= sum(minus_e)
            minus_density = self.density_value(
                coords, alpha_occ=minus_e, beta_occ=None, **kwargs
            )
        
        else:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            if self.n_beta > self.n_alpha:
                homo_nrg = self.beta_nrgs[self.n_beta - 1]
            
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            if self.n_beta > self.n_alpha:
                lumo_nrg = self.beta_nrgs[self.n_beta]

            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            alpha_occ = beta_occ = np.zeros(self.n_mos)
            for i in range(0, self.n_alpha):
                alpha_occ[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )
            
            for i in range(0, self.n_beta):
                beta_occ[i] = np.exp(
                    -((chem_pot - self.beta_nrgs[i]) / delta) ** 2
                )

            alpha_occ /= sum(alpha_occ)
            beta_occ /= sum(beta_occ)
            minus_density = self.density_value(
                coords, alpha_occ=alpha_occ, beta_occ=beta_occ, **kwargs
            )

        return minus_density
    
    def fukui_acceptor_value(self, coords, delta=0.1, **kwargs):
        """
        orbital-weighted fukui acceptor function
        electron density change for removing an electron
        orbital weighting from DOI 10.1021/acs.jpca.9b07516 accounts
        for nearly degenerate orbitals
        coords - coordinate to evaluate function at
        delta - parameter for weighting
        kwargs - passed to density_value
        """
        CITATION = "doi:10.1021/acs.jpca.9b07516"
        self.LOG.citation(CITATION)
        
        alpha_occ = np.zeros(self.n_mos)
        alpha_occ[self.n_alpha - 1] = 1
        beta_occ = None
        if self.beta_coefficients is not None:
            beta_occ = np.zeros(self.n_mos)
            beta_occ[self.n_beta - 1] = 1

        if self.beta_coefficients is None:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            plus_e = np.zeros(self.n_mos)
            for i in range(self.n_alpha, self.n_mos):
                plus_e[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )

            plus_e /= sum(plus_e)
            plus_density = self.density_value(
                coords, alpha_occ=plus_e, beta_occ=beta_occ, **kwargs
            )
        
        else:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            if self.n_beta > self.n_alpha:
                homo_nrg = self.beta_nrgs[self.n_beta - 1]
            
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            if self.n_beta > self.n_alpha:
                lumo_nrg = self.beta_nrgs[self.n_beta]

            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            alpha_occ = np.zeros(self.n_mos)
            beta_occ = np.zeros(self.n_mos)
            for i in range(self.n_alpha, self.n_mos):
                alpha_occ[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )
            
            for i in range(self.n_beta, self.n_mos):
                beta_occ[i] = np.exp(
                    -((chem_pot - self.beta_nrgs[i]) / delta) ** 2
                )

            alpha_occ /= sum(alpha_occ)
            beta_occ /= sum(beta_occ)
            plus_density = self.density_value(
                coords, alpha_occ=alpha_occ, beta_occ=beta_occ, **kwargs
            )

        return plus_density
