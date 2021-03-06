"""for constructing headers and footers for input files"""

import re

from AaronTools.theory import (
    GAUSSIAN_COMMENT,
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_COORDINATES,
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
    GAUSSIAN_POST,
    GAUSSIAN_PRE_ROUTE,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_COMMENT,
    ORCA_COORDINATES,
    ORCA_ROUTE,
    PSI4_AFTER_JOB,
    PSI4_BEFORE_GEOM,
    PSI4_BEFORE_JOB,
    PSI4_COMMENT,
    PSI4_MOLECULE,
    PSI4_COORDINATES,
    PSI4_JOB,
    PSI4_OPTKING,
    PSI4_SETTINGS,
)
from AaronTools.utils.utils import combine_dicts
from AaronTools.const import UNIT, ELEMENTS

from .basis import ECP, BasisSet
from .emp_dispersion import EmpiricalDispersion
from .grid import IntegrationGrid
from .job_types import JobType
from .method import KNOWN_SEMI_EMPIRICAL, Method, SAPTMethod


class Theory:
    """
    A Theory object can be used to create an input file for different QM software.
    The creation of a Theory object does not depend on the specific QM software
    that is determined when the file is written

    attribute names are the same as initialization keywords
    valid initialization keywords are:
    geometry                -   AaronTools Geometry
    charge                  -   total charge
    multiplicity            -   electronic multiplicity
    job_type                -   JobType or list(JobType)

    method                  -   Method object (or str - Method instance will be created)
    basis                   -   BasisSet object (or str - will be set to BasisSet(Basis(keyword)))
    ecp                     -   str parsable by BasisSet.parse_basis_str
    empirical_dispersion    -   EmpiricalDispersion object (or str)
    grid                    -   IntegrationGrid object (or str)
    solvent                 -   ImplicitSolvent object

    memory                  -   int - allocated memory (GB)
    processors              -   int - allocated cores
    """

    ACCEPTED_INIT_KW = [
        "geometry",
        "memory",
        "processors",
        "job_type",
        "solvent",
        "grid",
        "empirical_dispersion",
        "basis",
        "method",
    ]

    def __init__(
            self,
            charge=0,
            multiplicity=1,
            method=None,
            basis=None,
            ecp=None,
            empirical_dispersion=None,
            grid=None,
            **kwargs,
    ):
        if not isinstance(charge, list):
            self.charge = int(charge)
        else:
            self.charge = charge

        if not isinstance(multiplicity, list):
            self.multiplicity = int(multiplicity)
        else:
            self.multiplicity = multiplicity

        self.method = None
        self.basis = None
        self.empirical_dispersion = None
        self.grid = None
        self.solvent = None
        self.job_type = None
        self.processors = None
        self.memory = None
        self.geometry = None
        self.kwargs = {}

        for key in self.ACCEPTED_INIT_KW:
            if key in kwargs:
                self.__setattr__(key, kwargs[key])
                del kwargs[key]
            else:
                self.__setattr__(key, None)

        self.kwargs = kwargs

        if isinstance(self.processors, str):
            processors = re.search(r"(\d+)", self.processors)
            if processors:
                self.processors = processors.group(1)

        if isinstance(self.memory, str):
            memory = re.search(r"(\d+)", self.memory)
            if memory:
                self.memory = memory.group(1)

        # if method, basis, etc aren't the expected classes, make them so
        if method is not None:
            if not isinstance(method, Method):
                self.method = Method(method, method.upper() in KNOWN_SEMI_EMPIRICAL)
            else:
                self.method = method

        if grid is not None:
            if not isinstance(grid, IntegrationGrid):
                self.grid = IntegrationGrid(grid)
            else:
                self.grid = grid

        if basis is not None:
            if not isinstance(basis, BasisSet):
                self.basis = BasisSet(basis)
            else:
                self.basis = basis

        if ecp is not None:
            if self.basis is None:
                self.basis = BasisSet(ecp=ecp)
            else:
                self.basis.ecp = BasisSet.parse_basis_str(ecp, cls=ECP)

        if empirical_dispersion is not None:
            if not isinstance(empirical_dispersion, EmpiricalDispersion):
                self.empirical_dispersion = EmpiricalDispersion(
                    empirical_dispersion
                )
            else:
                self.empirical_dispersion = empirical_dispersion

        if self.job_type is not None:
            if isinstance(self.job_type, JobType):
                self.job_type = [self.job_type]

            for i, job1 in enumerate(self.job_type):
                for job2 in self.job_type[i + 1 :]:
                    if type(job1) is type(job2):
                        raise TypeError(
                            "cannot run multiple jobs of the same type: %s, %s"
                            % (str(job1), str(job2))
                        )

    def __setattr__(self, attr, val):
        if isinstance(val, str):
            if attr == "method":
                super().__setattr__(attr, Method(val))
            elif attr == "basis":
                super().__setattr__(attr, BasisSet(val))
            elif attr == "empirical_dispersion":
                super.__setattr__(attr, EmpiricalDispersion(val))
            elif attr == "grid":
                super().__setattr__(attr, IntegrationGrid(val))
            else:
                super().__setattr__(attr, val)
        else:
            super().__setattr__(attr, val)

    def make_header(
            self,
            geom=None,
            style="gaussian",
            conditional_kwargs=None,
            **kwargs,
    ):
        """
        geom: Geometry
        style: str, gaussian, orca, or psi4
        conditional_kwargs: dict - keys are ORCA_*, PSI4_*, or GAUSSIAN_*
            items in conditional_kwargs will only be added
            to the input if they would otherwise be preset
            e.g. if self.job_type is FrequencyJob and a Gaussian
            input file is being written,
            conditional_kwargs = {GAUSSIAN_ROUTE:{'opt':['noeigentest']}}
            will not add opt=noeigentest to the route
            but if it's an OptimizationJob, it will add opt=noeigentest
        kwargs: keywords are ORCA_*, PSI4_*, or GAUSSIAN_*
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        if geom is not None:
            self.geometry = geom
        if self.basis is not None:
            self.basis.refresh_elements(self.geometry)

        kwargs = combine_dicts(self.kwargs, kwargs)

        other_kw_dict = {}
        for keyword in kwargs:
            if (
                    keyword.startswith("PSI4_")
                    or keyword.startswith("ORCA_")
                    or keyword.startswith("GAUSSIAN_")
            ):
                new_kw = eval(keyword)
                other_kw_dict[new_kw] = kwargs[keyword]

            elif hasattr(self, keyword):
                if keyword == "method":
                    self.method = Method(kwargs[keyword])

                elif keyword == "basis":
                    self.basis = BasisSet(kwargs[keyword])

                elif keyword == "grid":
                    self.grid = IntegrationGrid(kwargs[keyword])

                elif keyword == "empirical_dispersion":
                    self.grid = EmpiricalDispersion(kwargs[keyword])

                elif keyword == "job_type":
                    if isinstance(kwargs[keyword], JobType):
                        self.job_type = [kwargs[keyword]]
                    else:
                        self.job_type = kwargs[keyword]

                elif keyword in self.ACCEPTED_INIT_KW:
                    setattr(self, keyword, kwargs[keyword])

            else:
                other_kw_dict[keyword] = kwargs[keyword]

        if style == "gaussian":
            return self.get_gaussian_header(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        elif style == "orca":
            return self.get_orca_header(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        elif style == "psi4":
            return self.get_psi4_header(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        raise NotImplementedError("no get_header method for style: %s" % style)

    def make_molecule(
            self,
            geom=None,
            style="gaussian",
            conditional_kwargs=None,
            **kwargs,
    ):
        """
        geom: Geometry()
        style: gaussian, psi4
        conditional_kwargs: dict() of keyword: value pairs
        kwargs: keywords are GAUSSIAN_*, ORCA_*, or PSI4_*
        """
        if conditional_kwargs is None:
            conditional_kwargs = {}

        if self.basis is not None:
            self.basis.refresh_elements(geom)

        kwargs = combine_dicts(self.kwargs, kwargs)

        other_kw_dict = {}
        for keyword in kwargs:
            if (
                    keyword.startswith("PSI4_")
                    or keyword.startswith("ORCA_")
                    or keyword.startswith("GAUSSIAN_")
            ):
                new_kw = eval(keyword)
                other_kw_dict[new_kw] = kwargs[keyword]

            elif hasattr(self, keyword):
                if keyword == "method":
                    self.method = Method(kwargs[keyword])

                elif keyword == "basis":
                    self.basis = BasisSet(kwargs[keyword])

                elif keyword == "grid":
                    self.grid = IntegrationGrid(kwargs[keyword])

                elif keyword == "empirical_dispersion":
                    self.grid = EmpiricalDispersion(kwargs[keyword])

                elif keyword == "job_type":
                    if isinstance(kwargs[keyword], JobType):
                        self.job_type = [kwargs[keyword]]
                    else:
                        self.job_type = kwargs[keyword]

                elif keyword in self.ACCEPTED_INIT_KW:
                    setattr(self, keyword, kwargs[keyword])

            else:
                other_kw_dict[keyword] = kwargs[keyword]

        if style == "gaussian":
            return self.get_gaussian_molecule(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        elif style == "psi4":
            return self.get_psi4_molecule(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        NotImplementedError("no get_molecule method for style: %s" % style)

    def make_footer(
            self,
            geom=None,
            style="gaussian",
            conditional_kwargs=None,
            **kwargs,
    ):
        """geom: Geometry
        style: str, gaussian or psi4
        conditional_kwargs: dict, see make_header
        kwargs: keywords are GAUSSIAN_*, ORCA_*, or PSI4_*
        """
        if conditional_kwargs is None:
            conditional_kwargs = {}

        if self.basis is not None:
            self.basis.refresh_elements(geom)

        kwargs = combine_dicts(self.kwargs, kwargs)

        other_kw_dict = {}
        for keyword in kwargs:
            if (
                    keyword.startswith("PSI4_")
                    or keyword.startswith("ORCA_")
                    or keyword.startswith("GAUSSIAN_")
            ):
                new_kw = eval(keyword)
                other_kw_dict[new_kw] = kwargs[keyword]

            elif hasattr(self, keyword):
                if keyword == "method":
                    self.method = Method(kwargs[keyword])

                elif keyword == "basis":
                    self.basis = BasisSet(kwargs[keyword])

                elif keyword == "grid":
                    self.grid = IntegrationGrid(kwargs[keyword])

                elif keyword == "empirical_dispersion":
                    self.grid = EmpiricalDispersion(kwargs[keyword])

                elif keyword == "job_type":
                    if isinstance(kwargs[keyword], JobType):
                        self.job_type = [kwargs[keyword]]
                    else:
                        self.job_type = kwargs[keyword]

                elif keyword in self.ACCEPTED_INIT_KW:
                    setattr(self, keyword, kwargs[keyword])

            else:
                other_kw_dict[keyword] = kwargs[keyword]

        if style == "gaussian":
            return self.get_gaussian_footer(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        elif style == "psi4":
            return self.get_psi4_footer(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        NotImplementedError("no get_footer method for style: %s" % style)

    def get_gaussian_header(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """write Gaussian09/16 input file header (up to charge mult)
        other_kw_dict is a dictionary with file positions (using GAUSSIAN_*)
        corresponding to options/keywords
        returns warnings if a certain feature is not available in Gaussian"""

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_gaussian()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        if (
                GAUSSIAN_COMMENT not in other_kw_dict
                or not other_kw_dict[GAUSSIAN_COMMENT]
        ):
            if self.geometry.comment:
                other_kw_dict[GAUSSIAN_COMMENT] = [self.geometry.comment]
            else:
                other_kw_dict[GAUSSIAN_COMMENT] = [self.geometry.name]

        # add EmpiricalDispersion info
        if self.empirical_dispersion is not None:
            disp, warning = self.empirical_dispersion.get_gaussian()
            other_kw_dict = combine_dicts(other_kw_dict, disp)
            if warning is not None:
                warnings.append(warning)

        # add Integral(grid=X)
        if self.grid is not None:
            grid, warning = self.grid.get_gaussian()
            other_kw_dict = combine_dicts(other_kw_dict, grid)
            if warning is not None:
                warnings.append(warning)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_gaussian()
            if warning is not None:
                warnings.extend(warning)
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        out_str = ""

        # processors, memory, and other link 0 stuff
        if self.processors is not None:
            out_str += "%%NProcShared=%i\n" % self.processors

        if self.memory is not None:
            out_str += "%%Mem=%iGB\n" % self.memory

        if GAUSSIAN_PRE_ROUTE in other_kw_dict:
            for key in other_kw_dict[GAUSSIAN_PRE_ROUTE]:
                out_str += "%%%s" % key
                if other_kw_dict[GAUSSIAN_PRE_ROUTE][key]:
                    out_str += "=%s" % ",".join(
                        other_kw_dict[GAUSSIAN_PRE_ROUTE][key]
                    )

                if not out_str.endswith("\n"):
                    out_str += "\n"

        # start route line with method
        out_str += "#n "
        if self.method is not None:
            func, warning = self.method.get_gaussian()
            if warning is not None:
                warnings.append(warning)
            out_str += "%s" % func
            if not self.method.is_semiempirical and self.basis is not None:
                basis_info = self.basis.get_gaussian_basis_info()
                # check basis elements to make sure no element is
                # in two basis sets or left out of any
                if self.geometry is not None:
                    basis_warning = self.basis.check_for_elements(
                        self.geometry
                    )
                    if basis_warning is not None:
                        warnings.append(basis_warning)

                if GAUSSIAN_ROUTE in basis_info:
                    out_str += "%s" % basis_info[GAUSSIAN_ROUTE]

            out_str += " "

        # add other route options
        # only one option can be specfied
        # e.g. for {'Integral':['grid=X', 'grid=Y']}, only grid=X will be used
        if GAUSSIAN_ROUTE in other_kw_dict.keys():
            for option in other_kw_dict[GAUSSIAN_ROUTE].keys():
                known_opts = []
                out_str += option
                if option.lower() == "opt":
                    # need to specified CalcFC for gaussian ts optimization
                    if any(
                            x.lower() == "ts"
                            for x in other_kw_dict[GAUSSIAN_ROUTE][option]
                    ) and not any(
                        x.lower() == y
                        for y in [
                                "calcfc",
                                "readfc",
                                "rcfc",
                                "readcartesianfc",
                                "calcall",
                                "calchffc",
                        ]
                        for x in other_kw_dict[GAUSSIAN_ROUTE][option]
                    ):
                        other_kw_dict[GAUSSIAN_ROUTE][option].append("CalcFC")

                if (
                        other_kw_dict[GAUSSIAN_ROUTE][option]
                        or (
                            other_kw_dict[GAUSSIAN_ROUTE][option] and
                            len(other_kw_dict[GAUSSIAN_ROUTE][option]) == 1 and (
                                "=" in other_kw_dict[GAUSSIAN_ROUTE][option][0]
                                or "(" in other_kw_dict[GAUSSIAN_ROUTE][option][0]
                            )
                        )
                ):
                    out_str += "=("
                    for x in other_kw_dict[GAUSSIAN_ROUTE][option]:
                        opt = x.split("=")[0]
                        if opt not in known_opts:
                            if known_opts:
                                out_str += ","
                            known_opts.append(opt)
                            out_str += x
                    out_str += ")"

                elif (
                        other_kw_dict[GAUSSIAN_ROUTE][option] and
                        len(other_kw_dict[GAUSSIAN_ROUTE][option]) == 1
                ):
                    out_str += "=%s" % other_kw_dict[GAUSSIAN_ROUTE][option][0]

                out_str += " "

        out_str += "\n\n"

        # add comment, removing any trailing newlines
        if GAUSSIAN_COMMENT in other_kw_dict:
            if other_kw_dict[GAUSSIAN_COMMENT]:
                out_str += "\n".join(
                    [x.rstrip() for x in other_kw_dict[GAUSSIAN_COMMENT]]
                )
            else:
                out_str += "comment"

            if not out_str.endswith("\n"):
                out_str += "\n"

        out_str += "\n"

        # charge mult
        out_str += "%i %i\n" % (int(self.charge), int(self.multiplicity))

        if return_warnings:
            return out_str, warnings

        return out_str

    def get_gaussian_molecule(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """
        get molecule specification for gaussian input files
        """
        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_gaussian()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        if GAUSSIAN_COORDINATES not in other_kw_dict:
            other_kw_dict[GAUSSIAN_COORDINATES] = {}

        if "coords" not in other_kw_dict[GAUSSIAN_COORDINATES]:
            other_kw_dict[GAUSSIAN_COORDINATES]["coords"] = self.geometry.coords

        s = ""

        for atom, coord in zip(
                self.geometry.atoms,
                other_kw_dict[GAUSSIAN_COORDINATES]["coords"]
        ):
            s += "%-2s" % atom.element
            for val in coord:
                s += "  "
                if isinstance(val, float):
                    s += " %9.5f" % val
                elif isinstance(val, str):
                    s += " %5s" % val
                elif isinstance(val, int):
                    s += " %3i" % val
                else:
                    warnings.append("unknown coordinate type: %s" % type(val))
            s += "\n"

        if (
                "variables" in other_kw_dict[GAUSSIAN_COORDINATES] and
                other_kw_dict[GAUSSIAN_COORDINATES]["variables"]
        ):
            s += "Variable:\n"
            for var in other_kw_dict[GAUSSIAN_COORDINATES]["variables"]:
                s += "%4s = %9.5f\n" % tuple(var)

        if (
                "constants" in other_kw_dict[GAUSSIAN_COORDINATES] and
                other_kw_dict[GAUSSIAN_COORDINATES]["constants"]
        ):
            s += "Constant:\n"
            for var in other_kw_dict[GAUSSIAN_COORDINATES]["constants"]:
                s += "%4s = %9.5f\n" % tuple(var)

        if return_warnings:
            return s, warnings

        return s

    def get_gaussian_footer(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """write footer of gaussian input file"""

        if conditional_kwargs is None:
            conditional_kwargs = {}

        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_gaussian()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        out_str = ""
        warnings = []

        # if method is not semi emperical, basis set might be gen or genecp
        # get basis info (will be written after constraints)
        if (
                self.method is not None
                and not self.method.is_semiempirical
                and self.basis is not None
        ):
            basis_info = self.basis.get_gaussian_basis_info()

        elif (
                self.method is not None
                and not self.method.is_semiempirical
                and self.basis is None
        ):
            basis_info = {}
            warnings.append("no basis specfied")

        out_str += "\n"

        # bond, angle, and torsion constraints
        if GAUSSIAN_CONSTRAINTS in other_kw_dict:
            for constraint in other_kw_dict[GAUSSIAN_CONSTRAINTS]:
                out_str += constraint
                out_str += "\n"

            out_str += "\n"

        # write gen info
        if self.method is not None and not self.method.is_semiempirical:
            if GAUSSIAN_GEN_BASIS in basis_info:
                out_str += basis_info[GAUSSIAN_GEN_BASIS]

                out_str += "\n"

            if GAUSSIAN_GEN_ECP in basis_info:
                out_str += basis_info[GAUSSIAN_GEN_ECP]

        # post info e.g. for NBOREAD
        if GAUSSIAN_POST in other_kw_dict:
            for item in other_kw_dict[GAUSSIAN_POST]:
                out_str += item
                out_str += " "

            out_str += "\n"

        out_str = out_str.rstrip()

        # new lines
        out_str += "\n\n\n"

        if return_warnings:
            return out_str, warnings

        return out_str

    def get_orca_header(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """
        get ORCA input file header
        other_kw_dict is a dictionary with file positions (using ORCA_*)
        corresponding to options/keywords
        returns file content and warnings e.g. if a certain feature is not available in ORCA
        returns str of header content
        if return_warnings, returns str, list(warning)
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_orca()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        warnings = []

        # if method isn't semi-empirical, get basis info to write later
        if not self.method.is_semiempirical and self.basis is not None:
            basis_info = self.basis.get_orca_basis_info()
            if self.geometry is not None:
                warning = self.basis.check_for_elements(self.geometry)
                if warning is not None:
                    warnings.append(warning)

        else:
            basis_info = {}

        other_kw_dict = combine_dicts(basis_info, other_kw_dict)

        # get grid info
        if self.grid is not None:
            grid_info, warning = self.grid.get_orca()
            if warning is not None:
                warnings.append(warning)

            if any(
                    "finalgrid" in x.lower() for x in other_kw_dict[ORCA_ROUTE]
            ):
                grid_info[ORCA_ROUTE].pop(1)

            other_kw_dict = combine_dicts(grid_info, other_kw_dict)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_orca()
            warnings.extend(warning)
            other_kw_dict = combine_dicts(solvent_info, other_kw_dict)

        # dispersion
        if self.empirical_dispersion is not None:
            dispersion, warning = self.empirical_dispersion.get_orca()
            if warning is not None:
                warnings.append(warning)

            other_kw_dict = combine_dicts(dispersion, other_kw_dict)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        # start building input file header
        out_str = ""

        # comment
        if ORCA_COMMENT not in other_kw_dict:
            if self.geometry.comment:
                other_kw_dict[ORCA_COMMENT] = [self.geometry.comment]
            else:
                other_kw_dict[ORCA_COMMENT] = [self.geometry.name]
        for comment in other_kw_dict[ORCA_COMMENT]:
            for line in comment.split("\n"):
                out_str += "#%s\n" % line

        out_str += "!"
        # method
        if self.method is not None:
            func, warning = self.method.get_orca()
            if warning is not None:
                warnings.append(warning)
            out_str += " %s" % func

        # add other route options
        if ORCA_ROUTE in other_kw_dict:
            if not out_str.endswith(" "):
                out_str += " "

            out_str += " ".join(other_kw_dict[ORCA_ROUTE])

        out_str += "\n"

        # procs
        if self.processors is not None:
            out_str += "%%pal\n    nprocs %i\nend\n" % self.processors

            # orca memory is per core, so only specify it if processors are specified
            if self.memory is not None:
                out_str += "%%MaxCore %i\n" % (
                    int(1000 * self.memory / self.processors)
                )

        # add other blocks
        if ORCA_BLOCKS in other_kw_dict:
            for keyword in other_kw_dict[ORCA_BLOCKS]:
                if any(other_kw_dict[ORCA_BLOCKS][keyword]):
                    out_str += "%%%s\n" % keyword
                    for opt in other_kw_dict[ORCA_BLOCKS][keyword]:
                        out_str += "    %s\n" % opt
                    out_str += "end\n"

            out_str += "\n"

        # start of coordinate section - end of header
        out_str += "*xyz %i %i\n" % (self.charge, self.multiplicity)

        if return_warnings:
            return out_str, warnings

        return out_str

    def get_psi4_header(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """
        write Psi4 input file
        other_kw_dict is a dictionary with file positions (using PSI4_*)
        corresponding to options/keywords
        returns file content and warnings e.g. if a certain feature is not available in Psi4
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_psi4()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)

        warnings = []

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_psi4()
            warnings.extend(warning)
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        # get basis info if method is not semi empirical
        if not self.method.is_semiempirical and self.basis is not None:
            basis_info = self.basis.get_psi4_basis_info(isinstance(self.method, SAPTMethod))
            if self.geometry is not None:
                warning = self.basis.check_for_elements(self.geometry)
                if warning is not None:
                    warnings.append(warning)

            # aux basis sets might have a '%s' b/c the keyword to apply them depends on
            # the method - replace %s with the appropriate thing for the method
            for key in basis_info:
                for i in range(0, len(basis_info[key])):
                    if "%s" in basis_info[key][i]:
                        if "cc" in self.method.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace(
                                "%s", "cc"
                            )

                        elif "dct" in self.method.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace(
                                "%s", "dct"
                            )

                        elif "mp2" in self.method.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace(
                                "%s", "mp2"
                            )

                        elif isinstance(self.method, SAPTMethod):
                            basis_info[key][i] = basis_info[key][i].replace(
                                "%s", "sapt"
                            )

                        elif "scf" in self.method.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace(
                                "%s", "scf"
                            )

                        elif "ci" in self.method.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace(
                                "%s", "mcscf"
                            )

        else:
            basis_info = {}

        combined_dict = combine_dicts(other_kw_dict, basis_info)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        # start building input file header
        out_str = ""

        # comment
        if PSI4_COMMENT not in combined_dict:
            if self.geometry.comment:
                combined_dict[PSI4_COMMENT] = [self.geometry.comment]
            else:
                combined_dict[PSI4_COMMENT] = [self.geometry.name]
        for comment in combined_dict[PSI4_COMMENT]:
            for line in comment.split("\n"):
                out_str += "#%s\n" % line

        # procs
        if self.processors is not None:
            out_str += "set_num_threads(%i)\n" % self.processors

        # mem
        if self.memory is not None:
            out_str += "memory %i GB\n" % self.memory

        # before geometry options e.g. basis {} or building a dft method
        if PSI4_BEFORE_GEOM in combined_dict:
            if combined_dict[PSI4_BEFORE_GEOM]:
                for opt in combined_dict[PSI4_BEFORE_GEOM]:
                    out_str += opt
                    out_str += "\n"
                out_str += "\n"

        if return_warnings:
            return out_str, warnings

        return out_str

    def get_psi4_molecule(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """
        get molecule specification for psi4 input files
        """
        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        use_bohr = False
        pubchem = False
        use_molecule_array = False
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_psi4()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )
        if PSI4_COORDINATES not in other_kw_dict:
            other_kw_dict[PSI4_COORDINATES] = {}

        if "coords" not in other_kw_dict[PSI4_COORDINATES]:
            other_kw_dict[PSI4_COORDINATES]["coords"] = self.geometry.coords

        s = ""
        if (
                isinstance(self.method, SAPTMethod) and
                sum(self.multiplicity[1:]) - len(self.multiplicity[1:]) + 1 > self.multiplicity[0]
        ):
            use_molecule_array = True
            s += "mol = psi4.core.Molecule.from_arrays(\n"
            s += "    molecular_multiplicity=%i,\n" % self.multiplicity[0]
            s += "    molecular_charge=%i,\n" % self.charge[0]
            if PSI4_MOLECULE in other_kw_dict:
                for keyword in other_kw_dict[PSI4_MOLECULE]:
                    if other_kw_dict[keyword]:
                        s += "    %s=%s,\n" % (keyword, repr(other_kw_dict[keyword][0]))

        else:
            s += "molecule {\n"
            if isinstance(self.method, SAPTMethod):
                s += "    %2i %i\n" % (self.charge[0], self.multiplicity[0])
            else:
                s += "    %2i %i\n" % (self.charge, self.multiplicity)

            if PSI4_MOLECULE in other_kw_dict:
                for keyword in other_kw_dict[PSI4_MOLECULE]:
                    if other_kw_dict[PSI4_MOLECULE][keyword]:
                        opt = other_kw_dict[PSI4_MOLECULE][keyword][0]
                        if (
                                "pubchem" in keyword.lower() and
                                not keyword.strip().endswith(":")
                        ):
                            keyword = keyword.strip() + ":"
                            pubchem = True
                        s += "     %s %s\n" % (keyword.strip(), opt)
                        if keyword == "units":
                            if opt.lower() in ["bohr", "au", "a.u."]:
                                use_bohr = True

                    else:
                        s += "     %s\n" % keyword

        if use_molecule_array:
            # psi4 input is VERY different for sapt jobs with the low-spin
            # combination of fragments
            monomers = [comp.atoms for comp in self.geometry.components]
            atoms_in_monomer = []
            seps = []
            for i, m1 in enumerate(self.geometry.components[:-1]):
                seps.append(0)
                for m2 in monomers[:i + 1]:
                    seps[-1] += len(m2)

            s += "    fragment_separators=%s,\n" % repr(seps)
            s += "    elez=%s,\n" % repr(
                [
                    ELEMENTS.index(atom.element)
                    for monomer in monomers
                    for atom in monomer
                ]
            )
            s += "    fragment_multiplicities=%s,\n" % repr(
                self.multiplicity[1:]
            )
            s += "    fragment_charges=%s,\n" % repr(self.charge[1:])
            s += "    geom=["
            i = 0
            for monomer in monomers:
                s += "\n"
                for atom in monomer:
                    if atom not in atoms_in_monomer:
                        atoms_in_monomer.append(atom)
                    else:
                        warnings.append("atom in two monomers: %s" % atom.name)
                    ndx = self.geometry.atoms.index(atom)
                    coord = other_kw_dict[PSI4_COORDINATES]["coords"][ndx]
                    for val in coord:
                        s += "       "
                        if isinstance(val, float):
                            if use_bohr:
                                s += "%9.5f," % (val / UNIT.A0_TO_BOHR)
                            else:
                                s += "%9.5f," % val
                        else:
                            warnings.append("unknown coordinate type: %s" % type(val))
                    s += "\n"

            s += "    ],\n"
            s += ")\n\n"
            s += "activate(mol)\n"

            if len(atoms_in_monomer) != len(self.geometry.atoms):
                from AaronTools.finders import NotAny
                warnings.append(
                    "there are atoms not in any monomers: %s" % (
                        ", ".join([atom.name for atom in self.geometry.find(NotAny(atoms_in_monomer))])
                    )
                )

        elif isinstance(self.method, SAPTMethod):
            monomers = [comp.atoms for comp in self.geometry.components]
            atoms_in_monomer = []
            for monomer, mult, charge in zip(
                monomers, self.multiplicity[1:], self.charge[1:]
            ):
                s += "    --\n"
                s += "    %2i %i\n" % (charge, mult)
                for atom in monomer:
                    s += "    %2s" % atom.element
                    if atom not in atoms_in_monomer:
                        atoms_in_monomer.append(atom)
                    else:
                        warnings.append("atom in two monomers: %s" % atom.name)
                    ndx = self.geometry.atoms.index(atom)
                    coord = other_kw_dict[PSI4_COORDINATES]["coords"][ndx]
                    for val in coord:
                        s += "     "
                        if isinstance(val, float):
                            if use_bohr:
                                s += " %9.5f" % (val / UNIT.A0_TO_BOHR)
                            else:
                                s += " %9.5f" % val
                        elif isinstance(val, str):
                            s += "    %9s" % val
                        else:
                            warnings.append("unknown coordinate type: %s" % type(val))
                    s += "\n"

            if "variables" in other_kw_dict[PSI4_COORDINATES]:
                for (name, val, angstrom) in other_kw_dict[PSI4_COORDINATES]["variables"]:
                    if use_bohr and angstrom:
                        val /= UNIT.A0_TO_BOHR
                    s += "     %3s = %9.5f\n" % (name, val)

            s += "}\n"

            if len(atoms_in_monomer) != len(self.geometry.atoms):
                from AaronTools.finders import NotAny
                warnings.append(
                    "there are atoms not in any monomers: %s" % (
                        ", ".join([atom.name for atom in self.geometry.find(NotAny(atoms_in_monomer))])
                    )
                )

        elif not pubchem:
            for atom, coord in zip(
                    self.geometry.atoms,
                    other_kw_dict[PSI4_COORDINATES]["coords"]
            ):
                s += "    %2s" % atom.element
                for val in coord:
                    s += "     "
                    if isinstance(val, float):
                        s += " %9.5f" % val
                    elif isinstance(val, str):
                        s += " %9s" % val
                    else:
                        warnings.append("unknown coordinate type: %s" % type(val))
                s += "\n"

            if "variables" in other_kw_dict[PSI4_COORDINATES]:
                for (name, val, angstrom) in other_kw_dict[PSI4_COORDINATES]["variables"]:
                    if use_bohr and angstrom:
                        val /= UNIT.A0_TO_BOHR
                    s += "     %3s = %9.5f\n" % (name, val)

            s += "}\n"
        else:
            s += "}\n"

        if return_warnings:
            return s, warnings

        return s

    def get_psi4_footer(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """
        get psi4 footer
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict = job.get_psi4()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_psi4()
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        # grid
        if self.grid is not None:
            grid_info, warning = self.grid.get_psi4()
            if warning is not None:
                warnings.append(warning)
            other_kw_dict = combine_dicts(other_kw_dict, grid_info)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        out_str = "\n"

        # settings
        # a setting will only get added if its list has at least
        # one item, but only the first item will be used
        if (
                PSI4_SETTINGS in other_kw_dict and
                any(
                    other_kw_dict[PSI4_SETTINGS][setting]
                    for setting in other_kw_dict[PSI4_SETTINGS]
                )
        ):
            out_str += "set {\n"
            for setting in other_kw_dict[PSI4_SETTINGS]:
                if other_kw_dict[PSI4_SETTINGS][setting]:
                    if isinstance(other_kw_dict[PSI4_SETTINGS][setting], str):
                        val = other_kw_dict[PSI4_SETTINGS][setting]
                    else:
                        val = other_kw_dict[PSI4_SETTINGS][setting][0]

                    out_str += "    %-20s    %s\n" % (setting, val)

            out_str += "}\n\n"

        if (
                PSI4_OPTKING in other_kw_dict and
                any(
                    other_kw_dict[PSI4_OPTKING][setting]
                    for setting in other_kw_dict[PSI4_OPTKING]
                )
        ):
            out_str += "set optking {\n"
            for setting in other_kw_dict[PSI4_OPTKING]:
                if other_kw_dict[PSI4_OPTKING][setting]:
                    out_str += "    %-20s    %s\n" % (
                        setting,
                        other_kw_dict[PSI4_OPTKING][setting][0],
                    )

            out_str += "}\n\n"

        # method is method name + dispersion if there is dispersion
        method = self.method.get_psi4()[0]
        if self.empirical_dispersion is not None:
            method += self.empirical_dispersion.get_psi4()[0]

        # after job stuff - replace METHOD with method
        if PSI4_BEFORE_JOB in other_kw_dict:
            for opt in other_kw_dict[PSI4_BEFORE_JOB]:
                if "$METHOD" in opt:
                    opt = opt.replace("$METHOD", "'%s'" % method)

                out_str += opt
                out_str += "\n"

        # for each job, start with nrg = f('method'
        # unless return_wfn=True, then do nrg, wfn = f('method'
        if PSI4_JOB in other_kw_dict:
            for func in other_kw_dict[PSI4_JOB].keys():
                if any(
                        [
                            "return_wfn" in kwarg
                            and ("True" in kwarg or "on" in kwarg)
                            for kwarg in other_kw_dict[PSI4_JOB][func]
                        ]
                ):
                    out_str += "nrg, wfn = %s('%s'" % (func, method)
                else:
                    out_str += "nrg = %s('%s'" % (func, method)

                known_kw = []
                for keyword in other_kw_dict[PSI4_JOB][func]:
                    key = keyword.split("=")[0].strip()
                    if key not in known_kw:
                        known_kw.append(key)
                        out_str += ", "
                        out_str += keyword.replace("$METHOD", "'%s'" % method)

                out_str += ")\n"

        # after job stuff - replace METHOD with method
        if PSI4_AFTER_JOB in other_kw_dict:
            for opt in other_kw_dict[PSI4_AFTER_JOB]:
                if "$METHOD" in opt:
                    opt = opt.replace("$METHOD", "'%s'" % method)

                out_str += opt
                out_str += "\n"

        if return_warnings:
            return out_str, warnings

        return out_str
