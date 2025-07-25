"""for constructing headers and footers for input files"""
import re

from AaronTools import addlogger
from AaronTools.const import ELEMENTS, UNIT
from AaronTools.theory import (
    GAUSSIAN_COMMENT,
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_COORDINATES,
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
    GAUSSIAN_MM,
    GAUSSIAN_ONIOM,
    GAUSSIAN_MM_PARAMS,
    GAUSSIAN_POST,
    GAUSSIAN_PRE_ROUTE,
    GAUSSIAN_ROUTE,
    GAUSSIAN_HIGH_ROUTE,
    GAUSSIAN_MEDIUM_ROUTE,
    GAUSSIAN_LOW_ROUTE,
    GAUSSIAN_CONNECTIVITY,
    ORCA_BLOCKS,
    ORCA_COMMENT,
    ORCA_COORDINATES,
    ORCA_ROUTE,
    PSI4_AFTER_JOB,
    PSI4_BEFORE_GEOM,
    PSI4_BEFORE_JOB,
    PSI4_COMMENT,
    PSI4_COORDINATES,
    PSI4_JOB,
    PSI4_MOLECULE,
    PSI4_OPTKING,
    PSI4_SETTINGS,
    PSI4_SOLVENT,
    SQM_COMMENT,
    SQM_QMMM,
    QCHEM_MOLECULE,
    QCHEM_REM,
    QCHEM_COMMENT,
    QCHEM_SETTINGS,
    XTB_CONTROL_BLOCKS,
    XTB_COMMAND_LINE,
    CREST_COMMAND_LINE,
    FrequencyJob,
    job_from_string,
)
from AaronTools.utils.utils import combine_dicts, subtract_dicts
from AaronTools.theory.basis import ECP, Basis, BasisSet
from AaronTools.theory.emp_dispersion import EmpiricalDispersion
from AaronTools.theory.grid import IntegrationGrid
from AaronTools.theory.job_types import (
    JobType, SinglePointJob, OptimizationJob, NMRJob,
)
from AaronTools.theory.method import KNOWN_SEMI_EMPIRICAL, KNOWN_MM, Method, SAPTMethod


@addlogger
class Theory:
    """
    A Theory object can be used to create an input file for different QM software.
    The creation of a Theory object does not depend on the specific QM software
    that is determined when the file is written

    attribute names are the same as initialization keywords (with the exception of ecp, which
    is added to the basis attribute)
    valid initialization keywords are:
    
    * geometry                -   AaronTools Geometry
    * charge                  -   total charge
    * multiplicity            -   electronic multiplicity
    * job_type                -   JobType or list(JobType)

    * method                  -   Method object (or str - Method instance will be created) or list(Method) for ONIOM
    * basis                   -   BasisSet object (or str - will be set to BasisSet(Basis(keyword)))
    * ecp                     -   str parsable by BasisSet.parse_basis_str
    * empirical_dispersion    -   EmpiricalDispersion object (or str)
    * grid                    -   IntegrationGrid object (or str)
    * solvent                 -   ImplicitSolvent object

    * memory                  -   int - allocated memory (GB)
    * processors              -   int - allocated cores
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
        "high_method",
        "medium_method",
        "low_method",
        "high_basis",
        "medium_basis",
        "low_basis",
        "high_ecp",
        "medium_ecp",
        "low_ecp"
    ]

    # if there's a setting that should be an array and Psi4 errors out
    # if it isn't an array, put it in this list (lower case)
    # generally happens if the setting value isn't a string
    # don't add settings that need > 1 value in the array
    FORCED_PSI4_ARRAY = [
        "cubeprop_orbitals",
        "cubeprop_tasks",
        "docc",
        "frac_occ",
    ]

    # commonly used settings that do not take array values
    FORCED_PSI4_SINGLE = [
        "reference",
        "scf_type",
        "freeze_core",
        "diag_method",
        "ex_level",
        "fci",
        "maxiter",
        "t",
        "p",
        "opt_type",
        "dft_radial_points",
        "dft_spherical_points",
    ]
    
    FORCED_PSI4_SOLVENT_SINGLE = [
        "units",
        "codata",
        "type",
        "npzfile",
        "area",
        "scaling",
        "raddiset",
        "minradius",
        "mode",
        "nonequilibrium",
        "solvent",
        "solvertype",
        "matrixsymm",
        "correction",
        "diagonalintegrator",
        "diagonalscaling",
        "proberadius",
    ]
    
    # some blocks need to go after the molecule
    # because they refer to atom indices, so orca needs
    # to read the structure first
    ORCA_BLOCKS_AFTER_MOL = ["eprnmr"]
    
    LOG = None

    def __init__(
        self,
        charge=0,
        multiplicity=1,
        method=None,
        basis=None,
        ecp=None,
        empirical_dispersion=None,
        grid=None,
        high_method=None,
        medium_method=None,
        low_method=None,
        high_ecp=None,
        medium_ecp=None,
        low_ecp=None,
        high_basis=None,
        medium_basis=None,
        low_basis=None,
        **kwargs,
    ):
        if isinstance(charge, list):
            self.charge = charge
        elif isinstance(charge, int):
            self.charge = charge
        elif isinstance(charge, str):
            if len(charge.split()) == 1:
                self.charge = int(charge)
            elif len(charge.split()) > 1:
                charge = charge.split()
                for x in charge:
                    x = int(x)
                self.charge = charge

        if isinstance(multiplicity, list):
            self.multiplicity = multiplicity
        elif isinstance(multiplicity, int):
            self.multiplicity = multiplicity
        elif isinstance(multiplicity, str):
            if len(multiplicity.split()) == 1:
                self.multiplicity = int(multiplicity)
            elif len(multiplicity.split()) > 1:
                multiplicity = multiplicity.split()
                for x in multiplicity:
                    x = int(x)
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
        self.high_method = None
        self.medium_method = None
        self.low_method = None
        self.high_basis = None
        self.medium_basis = None
        self.low_basis = None
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
                self.method = Method(
                    method, method.upper() in KNOWN_SEMI_EMPIRICAL
                )
            else:
                self.method = method

        if high_method is not None:
            if not isinstance(high_method, Method):
                self.high_method = Method(high_method)
            elif isinstance(high_method, Method):
                self.high_method = high_method

        if medium_method is not None:
            if not isinstance(medium_method, Method):
                self.medium_method = Method(medium_method)
            elif isinstance(medium_method, Method):
                self.medium_method = medium_method

        if low_method is not None:
            if not isinstance(low_method, Method):
                self.low_method = Method(low_method)
            elif isinstance(low_method, Method):
                self.low_method = low_method

        if grid is not None:
            if not isinstance(grid, IntegrationGrid):
                self.grid = IntegrationGrid(grid)
            else:
                self.grid = grid

        if basis is not None:
            self.basis = BasisSet(basis=basis)

        if high_basis is not None:
            if isinstance(high_basis, Basis):
                high_basis.oniom_layer = "H"
                self.high_basis = BasisSet(basis=high_basis)
            elif isinstance(high_basis, BasisSet):
                self.high_basis = high_basis
            elif isinstance(high_basis, str):
                self.high_basis = BasisSet(basis=high_basis) 
                if self.high_basis.basis:
                    for basis in self.high_basis.basis:
                        #new_high_basis = Basis(basis.name, user_defined = basis.user_defined, oniom_layer = "H") #, atoms = basis.elements)
                        basis.oniom_layer = "H"
                if self.high_basis.ecp:
                    for basis in self.high_basis.ecp:
                        basis.oniom_layer = "H"
            else:
                self.high_basis = high_basis
            if high_ecp is not None:
                self.high_basis.ecp = BasisSet(ecp=high_ecp).ecp

        if medium_basis is not None:
            if isinstance(medium_basis, Basis):
                medium_basis.oniom_layer = "M"
                self.medium_basis = BasisSet(basis=medium_basis)
            elif isinstance(medium_basis, BasisSet):
                for basis in medium_basis.basis:
                    basis.oniom_layer = "M"
                self.medium_basis = medium_basis
            elif isinstance(medium_basis, str):
                self.medium_basis = BasisSet(basis=medium_basis) 
                if self.medium_basis.basis:
                    for basis in self.medium_basis.basis:
                        #new_medium_basis = Basis(basis.name, elements = basis.elements, user_defined = basis.user_defined, oniom_layer = "H", atoms = basis.atoms)
                        basis.oniom_layer = "M"
            else:
                self.medium_basis = medium_basis
            if medium_ecp is not None:
                self.medium_basis.ecp = BasisSet(ecp=medium_ecp).ecp

        if low_basis is not None:
            if isinstance(low_basis, Basis):
                low_basis.oniom_layer = "L"
                self.low_basis = BasisSet(basis=low_basis)
            elif isinstance(low_basis, BasisSet):
                for basis in low_basis.basis:
                    basis.oniom_layer = "L"
            elif isinstance (low_basis, str):
                self.low_basis = BasisSet(basis=low_basis) 
                if self.low_basis.basis:
                    for basis in self.low_basis.basis:
                        #new_low_basis = Basis(basis.name, elements = basis.elements, user_defined = basis.user_defined, oniom_layer = "L", atoms = basis.atoms)
                        basis.oniom_layer = "L"
            else:
                self.low_basis = low_basis
            if low_ecp is not None:
                self.low_basis.ecp = BasisSet(ecp=low_ecp).ecp

        if self.basis is None and any((self.high_basis is not None, self.medium_basis is not None, self.low_basis is not None)):
            if self.high_basis is not None and self.medium_basis is not None and self.low_basis is not None:
                self.basis = BasisSet(basis = (self.high_basis, self.medium_basis, self.low_basis))
            elif self.high_basis is not None and self.medium_basis is None and self.low_basis is not None:
                self.basis = BasisSet(basis = (self.high_basis, self.low_basis))
            elif self.high_basis is not None and self.medium_basis is None and self.low_basis is None:
                if self.low_method.is_mm == True:
                    self.basis = BasisSet(basis = self.high_basis)
                elif not self.low_method.is_semiempirical: 
                    raise ValueError("low_method requires low_basis if not an MM method")
        #print("basis is " + str(self.basis))

        if ecp is not None:
            if self.basis is None:
                self.basis = BasisSet(ecp=ecp)
            else:
                self.basis.ecp = BasisSet(ecp=ecp).ecp

        if empirical_dispersion is not None:
            if not isinstance(empirical_dispersion, EmpiricalDispersion):
                self.empirical_dispersion = EmpiricalDispersion(
                    empirical_dispersion
                )
            else:
                self.empirical_dispersion = empirical_dispersion

        if self.job_type is not None:
            if isinstance(self.job_type, str):
                self.job_type = job_from_string(self.job_type)

            if isinstance(self.job_type, JobType):
                self.job_type = [self.job_type]

            # for i, job1 in enumerate(self.job_type):
            #     for job2 in self.job_type[i + 1 :]:
            #         if type(job1) is type(job2):
            #             self.LOG.warning(
            #                 "multiple jobs of the same type: %s, %s"
            #                 % (str(job1), str(job2))
            #             )

    def __setattr__(self, attr, val):
        if isinstance(val, str):
            if attr == "method":
                super().__setattr__(attr, Method(val))
            elif attr == "high_method":
                super().__setattr__(attr, Method(val))
            elif attr == "medium_method":
                super().__setattr__(attr, Method(val))
            elif attr == "low_method":
                super().__setattr__(attr, Method(val))
            elif attr == "basis":
                super().__setattr__(attr, BasisSet(val))
            elif attr == "high_basis":
                super().__setattr__(attr, BasisSet(val))
            elif attr == "medium_basis":
                super().__setattr__(attr, BasisSet(val))
            elif attr == "low_basis":
                super().__setattr__(attr, BasisSet(val))
            elif attr == "empirical_dispersion":
                super().__setattr__(attr, EmpiricalDispersion(val))
            elif attr == "grid":
                super().__setattr__(attr, IntegrationGrid(val))
            elif attr == "job_type" and isinstance(val, str):
                job_type = []
                for job in val.split("+"):
                    job_type.append(job_from_string(job))
                super().__setattr__(attr, job_type)
            else:
                super().__setattr__(attr, val)
        elif attr == "job_type" and isinstance(val, JobType):
            super().__setattr__(attr, [val])
        else:
            super().__setattr__(attr, val)

    def __eq__(self, other):
        if self.method != other.method:
            # print("method")
            return False
        if self.charge != other.charge:
            # print("charge")
            return False
        if self.multiplicity != other.multiplicity:
            # print("multiplicity")
            return False
        if self.basis != other.basis:
            # print("basis")
            return False
        if self.empirical_dispersion != other.empirical_dispersion:
            # print("disp")
            return False
        if self.grid != other.grid:
            # print("grid")
            return False
        if self.solvent != other.solvent:
            # print("solvent")
            return False
        if self.processors != other.processors:
            # print("procs", self.processors, other.processors)
            return False
        if self.memory != other.memory:
            # print("mem")
            return False
        if self.job_type and other.job_type:
            if len(self.job_type) != len(other.job_type):
                return False
            for job1, job2 in zip(self.job_type, other.job_type):
                if job1 != job2:
                    return False

        return True

    def copy(self):
        new_dict = dict()
        new_kwargs = dict()
        for key, value in self.__dict__.items():
            try:
                if key == "kwargs":
                    new_kwargs = value.copy()
                else:
                    new_dict[key] = value.copy()
            except AttributeError:
                new_dict[key] = value
                # ignore chimerax objects so seqcrow doesn't print a
                # warning when a geometry is copied
                if "chimerax" in value.__class__.__module__:
                    continue
                if value.__class__.__module__ != "builtins":
                    self.LOG.warning(
                        "No copy method for {}: in-place changes may occur".format(
                            type(value)
                        )
                    )

        return self.__class__(**new_dict, **new_kwargs)

    def add_kwargs(self, **kwargs):
        """
        add kwargs to the theory
        """
        new_kwargs = dict()
        for keyword in kwargs.keys():
            try:
                new_kw = eval(keyword)
                new_kwargs[new_kw] = kwargs[keyword]
            except NameError:
                new_kwargs[keyword] = kwargs[keyword]

        self.kwargs = combine_dicts(
            new_kwargs, self.kwargs
        )

    def remove_kwargs(self, **kwargs):
        """
        remove kwargs from the theory
        """
        new_kwargs = dict()
        for keyword in kwargs.keys():
            try:
                new_kw = eval(keyword)
                new_kwargs[new_kw] = kwargs[keyword]
            except NameError:
                new_kwargs[keyword] = kwargs[keyword]

        self.kwargs = subtract_dicts(
            self.kwargs, new_kwargs
        )

    def make_header(
        self,
        geom=None,
        style="gaussian",
        conditional_kwargs=None,
        sanity_check_method=False,
        **kwargs,
    ):
        """
        :param Geometry geom: structure
        :param str style: file format (e.g. gaussian, orca, psi4, oniom, or sqm)
        :param dict conditional_kwargs: keys are ORCA_*, PSI4_*, or GAUSSIAN_*
            
            items in conditional_kwargs will only be added
            to the input if they would otherwise be preset.
            For example, if self.job_type is FrequencyJob and a Gaussian
            input file is being written,
            conditional_kwargs = {GAUSSIAN_ROUTE:{'opt':['noeigentest']}}
            will not add opt=noeigentest to the route
            but if it's an OptimizationJob, it will add opt=noeigentest
        :param bool sanity_check_method: check if method is available in recent version
                             of the target software package (Psi4 checks when its
                             footer is created)
        :param dict kwargs: see AaronTools.theory parameters for more details 
        """
        if geom is None:
            geom = self.geometry

        if conditional_kwargs is None:
            conditional_kwargs = {}

        if geom is not None:
            self.geometry = geom
        if self.basis is not None:
            self.basis.refresh_elements(self.geometry)
            #self.basis.refresh_atoms(self.geometry)

        kwargs = combine_dicts(self.kwargs, kwargs)

        other_kw_dict = {}
        for keyword in kwargs:
            if (
                keyword.startswith("PSI4_")
                or keyword.startswith("ORCA_")
                or keyword.startswith("GAUSSIAN_")
                or keyword.startswith("QCHEM_")
            ):
                new_kw = eval(keyword)
                other_kw_dict[new_kw] = kwargs[keyword]

            elif hasattr(self, keyword):
                if keyword == "method":
                    self.method = Method(kwargs[keyword])

                elif keyword == "high_method":
                    self.high_method = Method(kwargs[keyword])

                elif keyword == "medium_method":
                    self.medium_method = Method(kwargs[keyword])

                elif keyword == "low_method":
                    self.low_method = Method(kwargs[keyword])

                elif keyword == "basis":
                    self.basis = BasisSet(kwargs[keyword])

                elif keyword == "high_basis":
                    self.high_basis = BasisSet(kwargs[keyword])

                elif keyword == "medium_basis":
                    self.medium_basis = BasisSet(kwargs[keyword])

                elif keyword == "low_basis":
                    self.low_basis = BasisSet(kwargs[keyword])

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

        elif style == "sqm":
            return self.get_sqm_header(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        elif style == "qchem":
            return self.get_qchem_header(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        raise NotImplementedError("no get_header method for style: %s" % style)

    def make_molecule(
        self,
        geom=None,
        style="gaussian",
        conditional_kwargs=None,
        return_warnings=True,
        **kwargs,
    ):
        """
        :param Geometry geom: structure
        :param str style: gaussian, psi4, or sqm
        :param dict conditional_kwargs: theory parameters, which will
            only be added if the corresponding section is used elsewhere
        :param dict kwargs: see AaronTools.theory parameters for more details 
        """
        if geom is None:
            geom = self.geometry
        else:
            self.geometry = geom

        if conditional_kwargs is None:
            conditional_kwargs = {}

        if self.basis is not None:
            self.basis.refresh_elements(geom)
            #self.basis.refresh_atoms(geom)

        kwargs = combine_dicts(self.kwargs, kwargs)

        other_kw_dict = {}
        for keyword in kwargs:
            if (
                keyword.startswith("PSI4_")
                or keyword.startswith("ORCA_")
                or keyword.startswith("GAUSSIAN_")
                or keyword.startswith("QCHEM_")
            ):
                new_kw = eval(keyword)
                other_kw_dict[new_kw] = kwargs[keyword]

            elif hasattr(self, keyword):
                if keyword == "method":
                    self.method = Method(kwargs[keyword])

                elif keyword == "high_method":
                    self.high_method = Method(kwargs[keyword])

                elif keyword == "medium_method":
                    self.medium_method = Method(kwargs[keyword])

                elif keyword == "low_method":
                    self.low_method = Method(kwargs[keyword])

                elif keyword == "basis":
                    self.basis = BasisSet(kwargs[keyword])

                elif keyword == "high_basis":
                    self.high_basis = BasisSet(kwargs[keyword])

                elif keyword == "medium_basis":
                    self.medium_basis = BasisSet(kwargs[keyword])

                elif keyword == "low_basis":
                    self.low_basis = BasisSet(kwargs[keyword])

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
            out = self.get_gaussian_molecule(
                conditional_kwargs=conditional_kwargs,
                return_warnings=return_warnings,
                **other_kw_dict
            )

        elif style == "psi4":
            out = self.get_psi4_molecule(
                conditional_kwargs=conditional_kwargs,
                return_warnings=return_warnings,
                **other_kw_dict
            )

        elif style == "sqm":
            out = self.get_sqm_molecule(
                conditional_kwargs=conditional_kwargs,
                return_warnings=return_warnings,
                **other_kw_dict
            )

        elif style == "qchem":
            out = self.get_qchem_molecule(
                conditional_kwargs=conditional_kwargs,
                return_warnings=return_warnings,
                **other_kw_dict
            )
        
        else:
            NotImplementedError("no get_molecule method for style: %s" % style)
        
        if return_warnings:
            s, warnings = out
            q = self.charge
            mult = self.multiplicity
            if not isinstance(q, int):
                return s, warnings
            Z = 0
            for atom in geom.atoms:
                if not atom.is_dummy and not atom.is_ghost:
                    try:
                        Z += ELEMENTS.index(atom.element)
                    except IndexError:
                        warnings.append("unknown element number")
                        return s, warnings
            
            if abs(Z - q) % 2 == mult % 2:
                warnings.append("incompatible charge and multiplicity")
            
            return s, warnings
        return out

    def make_footer(
        self,
        geom=None,
        style="gaussian",
        conditional_kwargs=None,
        sanity_check_method=False,
        **kwargs,
    ):
        """
        :param Geometry geom: structure
        :param str style: program name
        :param dict conditional_kwargs: see Theory.make_header
        :param bool sanity_check_method: check if method is available in recent version
            of the target software package (Psi4 only)
        :param dict kwargs: see AaronTools.theory parameters for more details 
        """
        if geom is None:
            geom = self.geometry

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

                elif keyword == "high_method":
                    self.high_method = Method(kwargs[keyword])

                elif keyword == "medium_method":
                    self.medium_method = Method(kwargs[keyword])

                elif keyword == "low_method":
                    self.low_method = Method(kwargs[keyword])

                elif keyword == "basis":
                    self.basis = BasisSet(kwargs[keyword])

                elif keyword == "high_basis":
                    self.high_basis = BasisSet(kwargs[keyword])

                elif keyword == "medium_basis":
                    self.medium_basis = BasisSet(kwargs[keyword])

                elif keyword == "low_basis":
                    self.low_basis = BasisSet(kwargs[keyword])

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

        elif style == "orca":
            return self.get_orca_footer(
                conditional_kwargs=conditional_kwargs, **other_kw_dict
            )

        NotImplementedError("no get_footer method for style: %s" % style)

    def get_gaussian_header(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        **other_kw_dict,
    ):
        """
        write Gaussian09/16 input file header (up to charge mult)
        
        other_kw_dict is a dictionary with file positions (using GAUSSIAN_*)
        corresponding to options/keywords
        
        returns warnings if a certain feature is not available in Gaussian
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        nmr = False
        opt = False
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_gaussian()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)
                if isinstance(job, OptimizationJob):
                    opt = True
                if isinstance(job, NMRJob):
                    nmr = True
        if opt and nmr:
            warnings.append("Opt+NMR is not allowed in Gaussian 16 and lower")

        if (
            GAUSSIAN_COMMENT not in other_kw_dict
            or not other_kw_dict[GAUSSIAN_COMMENT]
        ):
            if self.geometry.comment:
                other_kw_dict[GAUSSIAN_COMMENT] = [self.geometry.comment]
            elif self.geometry.name:
                other_kw_dict[GAUSSIAN_COMMENT] = [self.geometry.name]
            else:
                other_kw_dict[GAUSSIAN_COMMENT] = ["comment"]

        # add EmpiricalDispersion info
        if self.empirical_dispersion is not None:
            disp, warning = self.empirical_dispersion.get_gaussian()
            if sum([int(getattr(self, "%s_method" % layer) is not None) for layer in ["high", "medium", "low"]]) > 1:
                try:
                    disp[GAUSSIAN_HIGH_ROUTE] = disp[GAUSSIAN_ROUTE]
                    del disp[GAUSSIAN_ROUTE]
                except KeyError:
                    pass
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

        if self.memory:
            out_str += "%%Mem=%iGB\n" % self.memory

        if GAUSSIAN_PRE_ROUTE in other_kw_dict:
            for key in other_kw_dict[GAUSSIAN_PRE_ROUTE]:
                out_str += "%%%s" % key
                if other_kw_dict[GAUSSIAN_PRE_ROUTE][key]:
                    if not any(key.lower() == x for x in ["kjob", "subst"]):
                        out_str += "=%s" % ",".join(
                            other_kw_dict[GAUSSIAN_PRE_ROUTE][key]
                        )
                    else:
                        out_str += " %s" % ",".join(
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
            warning = self.method.sanity_check_method(func, "gaussian")
            if warning:
                warnings.append(warning)
            out_str += "%s" % func
            if (
                not self.method.is_semiempirical 
                and not self.method.is_mm
                and self.basis is not None
            ):
                basis_info, basis_warnings = self.basis.get_gaussian_basis_info()
                warnings.extend(basis_warnings)
                # check basis elements to make sure no element is
                # in two basis sets or left out of any
                if self.geometry is not None:
                    basis_warning = self.basis.check_for_elements(
                        self.geometry
                    )
                    if basis_warning is not None:
                        warnings.append(basis_warning)

                if GAUSSIAN_ROUTE in basis_info:
                    for key in basis_info[GAUSSIAN_ROUTE]:
                        if "/" in key:
                            out_str += "%s" % key
                        del basis_info[GAUSSIAN_ROUTE][key]
                        break
                    other_kw_dict = combine_dicts(
                        other_kw_dict,
                        basis_info,
                    )

            out_str += " "

        layered_charge_mult = False

        if self.method is None and any((self.high_method is not None, self.medium_method is not None, self.low_method is not None)):
            layered_charge_mult = True
            methods = {
                layer: getattr(self, "%s_method" % layer) for layer in ["high", "medium", "low"]
                if getattr(self, "%s_method" % layer) is not None
            }
            n_layers = len(methods)
            if n_layers < 2:
                raise RuntimeError("at least two layers must be defined for ONIOM")
            basis_sets = {layer: getattr(self, "%s_basis" % layer) for layer in ["high", "medium", "low"] if getattr(self, "%s_basis" % layer)}
            if self.basis:
                for basis in self.basis.basis:
                    if basis.oniom_layer == "H":
                        basis_sets.setdefault("high", [])
                        basis_sets["high"].append(basis)
                    elif basis.oniom_layer == "M":
                        basis_sets.setdefault("medium", [])
                        basis_sets["medium"].append(basis)
                    elif basis.oniom_layer == "L":
                        basis_sets.setdefault("low", [])
                        basis_sets["low"].append(basis)
            out_str += "oniom("
            for i, layer in enumerate(["high", "medium", "low"], start=1):
                try:
                    method = methods[layer]
                except KeyError:
                    try:
                        if not basis_sets[layer].basis:
                            continue
                        raise RuntimeError("no method for %s layer" % layer)
                    except KeyError:
                        continue

                func, warning = method.get_gaussian()
                if warning is not None:
                    warnings.append(warning)
                warning = method.sanity_check_method(func, "gaussian")
                if warning:
                    warnings.append(warning)
                out_str += "%s" % func
                if not (method.is_semiempirical or method.is_mm):
                    try:
                        basis = basis_sets[layer]
                        basis.refresh_elements(self.geometry)
                    except KeyError:
                        raise AttributeError("need to include a basis set for %s" % method.name)
                    basis_info, basis_warnings = basis.get_gaussian_basis_info()
                    warnings.extend(basis_warnings)
                    if GAUSSIAN_ROUTE in basis_info:
                        for key in basis_info[GAUSSIAN_ROUTE]:
                            if "/" in key:
                                out_str += "%s" % key
                            del basis_info[GAUSSIAN_ROUTE][key]
                            break

                    if isinstance(self.basis, Basis):
                        if basis.oniom_layer.lower() == layer[0]:
                            basis = BasisSet(basis=basis)
                            basis_info, basis_warnings = basis.get_gaussian_basis_info()
                            warnings.extend(basis_warnings)
                            # check basis elements to make sure no element is
                            # in two basis sets or left out of any
                            if self.geometry is not None:
                                basis_warning = basis.check_for_elements(
                                    self.geometry
                                )
                                if basis_warning is not None:
                                    warnings.append(basis_warning)

                            if GAUSSIAN_ROUTE in basis_info:
                                out_str += "%s" % basis_info[GAUSSIAN_ROUTE]

                elif method.is_mm:
                    if GAUSSIAN_MM in other_kw_dict.keys():
                        for option in other_kw_dict[GAUSSIAN_MM].keys():
                            known_opts = []
                            if option.upper() == method.name.upper():
                                out_str += "=("
                                for x in other_kw_dict[GAUSSIAN_MM][option]:
                                    if x not in known_opts:
                                        if known_opts:
                                            out_str += ","
                                        known_opts.append(x)
                                        out_str += x
                                out_str += ")"
                
                try:
                    if layer == "high":
                        layer_route = other_kw_dict[GAUSSIAN_HIGH_ROUTE]
                    elif layer == "medium":
                        layer_route = other_kw_dict[GAUSSIAN_MEDIUM_ROUTE]
                    elif layer == "low":
                        layer_route = other_kw_dict[GAUSSIAN_LOW_ROUTE]
                    
                    for key, items in layer_route.items():
                        out_str += " "
                        out_str += key
                        if items:
                            out_str += "=(%s)" % ",".join(items)

                except KeyError:
                    pass

                out_str += ":"

            out_str = out_str.rstrip(":")
            out_str += ")"

            if GAUSSIAN_ONIOM in other_kw_dict.keys():
                known_opts = []
                out_str += "="
                for x in other_kw_dict[GAUSSIAN_ONIOM]:
                    if x not in known_opts:
                        if known_opts:
                            out_str += ","
                        known_opts.append(x)
                        out_str += x
            out_str += " "

        # add other route options
        # only one option can be specfied
        # e.g. for {'Integral':['grid=X', 'grid=Y']}, only grid=X will be used
        other_kw_dict.setdefault(GAUSSIAN_ROUTE, dict())
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

            if other_kw_dict[GAUSSIAN_ROUTE][option] or (
                other_kw_dict[GAUSSIAN_ROUTE][option]
                and len(other_kw_dict[GAUSSIAN_ROUTE][option]) == 1
                and (
                    "=" in other_kw_dict[GAUSSIAN_ROUTE][option][0]
                    or "(" in other_kw_dict[GAUSSIAN_ROUTE][option][0]
                )
            ):
                if option.lower() == "iop":
                    out_str += "("
                else:
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
                other_kw_dict[GAUSSIAN_ROUTE][option]
                and len(other_kw_dict[GAUSSIAN_ROUTE][option]) == 1
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
                for x in other_kw_dict[GAUSSIAN_COMMENT]:
                    for c in "@#!-_\\":
                        if c in x:
                            warnings.append("avoid using %s in the comment/title section of Gaussian input" % c)
                        
            else:
                out_str += "comment"

            if not out_str.endswith("\n"):
                out_str += "\n"

        out_str += "\n"

        # charge mult
        if layered_charge_mult == True:
            if isinstance(self.charge, list):
                charge_list = self.charge
            elif isinstance(self.charge, int):
                charge_list = list(str(self.charge))
            if isinstance(self.multiplicity, list):
                mult_list = self.multiplicity
            elif isinstance(self.multiplicity, int):
                mult_list = list(str(self.multiplicity))
            if len(charge_list) > 1 and len(charge_list) != len(mult_list):
                raise ValueError("Charge and multiplicity must match in length. see gaussian.com/oniom/")
            elif len(charge_list) > 1 and len(charge_list) == len(mult_list):
                for i in range(len(charge_list)):
                    out_str += "%i %i" % (int(charge_list[i]), int(mult_list[i]))
                    if i < (len(charge_list) - 1):
                        out_str += " "
                    else:
                        out_str += "\n"
            elif len(charge_list) == 1 and len(charge_list) == len(mult_list):
                out_str += "%i %i %i %i %i %i \n" % (int(self.charge), int(self.multiplicity), int(self.charge), int(self.multiplicity), int(self.charge), int(self.multiplicity))

        elif layered_charge_mult == False:
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

                job_dict, job_warnings = job.get_gaussian()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        if GAUSSIAN_COORDINATES not in other_kw_dict:
            other_kw_dict[GAUSSIAN_COORDINATES] = {}

        s = ""

        # atom specs need flag column before coords if any atoms frozen
        oniom = False
        mm = False
        
        if self.method is None and self.high_method is not None:
            oniom = True
        
        if any(method is not None and method.is_mm for method in [
            self.method, self.high_method, self.medium_method, self.low_method
        ]):
            mm = True

        for i, atom in enumerate(self.geometry.atoms):
            ele = ""
            ele += "%s" % atom.element
            
            if mm:
                try:
                    if atom.atomtype:
                        ele += "-%s" % atom.atomtype
                except AttributeError:
                    pass

                try:
                    ele += "-%f" % atom.charge
                except (AttributeError, TypeError):
                    pass
            
                s += "%-20s" % ele
            else:
                s += "%-2s" % ele
            
            try:
                coord = other_kw_dict[GAUSSIAN_COORDINATES]["coords"][i]
                for val in coord:
                    s += "  "
                    try:
                        s += " %i" % val
                    except TypeError:
                        try:
                            s += " %9.5f" % val
                        except TypeError:
                            s += " %5s" % str(val)

            except KeyError:
                coord = tuple(atom.coords)
                s += "   %9.5f   %9.5f   %9.5f" % coord

            if oniom:
                s += " %s" % atom.layer
                if not atom.link_info:
                    pass
                else:
                    s += " %s" % atom.link_info["element"]
                    if has_type:
                        try:
                            s += "-%s" % atom.link_info["atomtype"]
                        except KeyError:
                            if atom.atomtype.lower() == "ca":
                                link_type = "ha"
                            else:
                                connected_elements = []
                                for connected in atom.connected:
                                    connected_elements.append(connected.element)
                                if "C" in connected_elements:
                                    link_type = "hc"
                                elif "C" not in connected_elements and "N" in connected_elements:
                                    link_type = "hn"
                                elif "C" not in connected_elements and "O" in connected_elements:
                                    link_type = "ho"
                                elif "C" not in connected_elements and "S" in connected_elements:
                                    link_type = "hs"
                                elif "C" not in connected_elements and "P" in connected_elements:
                                    link_type = "hp"
                            s += "-%s" % link_type
                    if has_charge:
                        try:
                            s += "-%s" % atom.link_info["charge"]
                        except KeyError:
                            s += "-%f" % atom.charge
                    s += " %s" % atom.link_info["connected"]
                    #scale_fax = re.search("scale factors ([()0-9,]+)", str(atom.tags))
                    #if scale_fax:
                    #    for scale_fac in scal_fax.group(1).split(","):
                    #        s += " %i" % int(scale_fac)
            s += "\n"

        s += "\n"

        if (
            "variables" in other_kw_dict[GAUSSIAN_COORDINATES]
            and other_kw_dict[GAUSSIAN_COORDINATES]["variables"]
        ):
            s += "Variable:\n"
            for var in other_kw_dict[GAUSSIAN_COORDINATES]["variables"]:
                s += "%4s = %9.5f\n" % tuple(var)

        if (
            "constants" in other_kw_dict[GAUSSIAN_COORDINATES]
            and other_kw_dict[GAUSSIAN_COORDINATES]["constants"]
        ):
            s += "Constant:\n"
            for var in other_kw_dict[GAUSSIAN_COORDINATES]["constants"]:
                s += "%4s = %9.5f\n" % tuple(var)

        if (
            GAUSSIAN_CONNECTIVITY in other_kw_dict
            and other_kw_dict[GAUSSIAN_CONNECTIVITY]
        ):
            s += "\n".join(other_kw_dict[GAUSSIAN_CONNECTIVITY])
            s += "\n\n"

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

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_gaussian()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        out_str = ""

        # if method is not semi emperical, basis set might be gen or genecp
        # get basis info (will be written after constraints)
        if (
            self.method is not None
            and not self.method.is_semiempirical
            and self.basis is not None
        ):
            basis_info, basis_warnings = self.basis.get_gaussian_basis_info()

        elif (
            self.method is not None
            and not self.method.is_semiempirical
            and self.basis is None
        ):
            basis_info = {}
            warnings.append("no basis specfied")

        elif any((self.high_method is not None, self.medium_method is not None, self.low_method is not None)):
            high_basis_info = {}
            medium_basis_info = {}
            low_basis_info = {}
            high_warnings = []
            medium_warnings = []
            low_warnings = []
            if self.high_method is not None and not self.high_method.is_semiempirical and self.high_basis is not None:
                high_basis_info, high_warnings = self.high_basis.get_gaussian_basis_info()
            if self.medium_method is not None and not self.medium_method.is_semiempirical and self.medium_basis is not None:
                medium_basis_info, medium_warnings = self.medium_basis.get_gaussian_basis_info()
                if GAUSSIAN_GEN_BASIS in medium_basis_info or GAUSSIAN_GEN_ECP in medium_basis_info:
                    medium_basis_info = {}
                    medium_warnings.append("ONIOM gen basis sets only supported for the high layer")
            if self.low_method is not None and not self.low_method.is_semiempirical and self.low_basis is not None:
                low_basis_info, low_warnings = self.low_basis.get_gaussian_basis_info()
                if GAUSSIAN_GEN_BASIS in low_basis_info or GAUSSIAN_GEN_ECP in low_basis_info:
                    low_basis_info = {}
                    low_warnings.append("ONIOM gen basis sets only supported for the high layer")

            #basis_info = {} #list((high_basis_info, medium_basis_info, low_basis_info))
            basis_info = combine_dicts(high_basis_info, medium_basis_info, low_basis_info)
            warnings = [*high_warnings, *medium_warnings, *low_warnings]

            # basis_info, warnings = self.basis.get_gaussian_basis_info()

        elif any((self.high_method is not None, self.medium_method is not None, self.low_method is not None)) and self.high_basis is None and self.medium_basis is None and self.low_basis is None:
            basis_info = {}
            warnings.append("no basis specified")

        elif (
            self.method is not None and self.method.is_semiempirical
        ):
            basis_info = {}

        # bond, angle, and torsion constraints
        if GAUSSIAN_CONSTRAINTS in other_kw_dict:
            out_str = out_str.rstrip()
            out_str += "\n\n"
            for constraint in other_kw_dict[GAUSSIAN_CONSTRAINTS]:
                out_str += constraint
                out_str += "\n"

            out_str += "\n"

        #mm param file
        if GAUSSIAN_MM_PARAMS in other_kw_dict:
            out_str = out_str.rstrip()
            out_str += "\n\n"

            for param_path in other_kw_dict[GAUSSIAN_MM_PARAMS]:
                #param_path_list = param_path.split("/")
                #param_file = param_path_list[len(param_path_list)-1]
                out_str += "@%s" % param_path
                if GAUSSIAN_GEN_BASIS in basis_info:
                    warnings.append("Parameter file specification is according to Gaussian 16 syntax and will not work for Gaussian 09 jobs")
                    if GAUSSIAN_CONSTRAINTS in other_kw_dict:
                        warnings.append("If using Gaussian 09, constraints are incompatible with MM parameter files and job will not run.")
                out_str += "\n"

        # write gen info
        if any(getattr(self, "%smethod" % layer) is not None for layer in ["", "high_", "medium_", "low_"]):
            try:
                # check whether there is gen basis info
                # IndexError or KeyError if there isn't
                basis_info[GAUSSIAN_GEN_BASIS][0]
                out_str = out_str.rstrip()
                out_str += "\n\n"
                out_str += basis_info[GAUSSIAN_GEN_BASIS]
                out_str += "\n"
            except (KeyError, IndexError):
                pass

            try:
                out_str = out_str.rstrip()
                out_str += "\n\n"
                basis_info[GAUSSIAN_GEN_ECP][0]
                out_str += basis_info[GAUSSIAN_GEN_ECP]
                out_str += "\n"
            except (KeyError, IndexError):
                pass

        # post info e.g. for NBOREAD
        if GAUSSIAN_POST in other_kw_dict:
            out_str = out_str.rstrip()
            out_str += "\n\n"
            for item in other_kw_dict[GAUSSIAN_POST]:
                out_str += item
                out_str += "\n"

            out_str += "\n"

        out_str = out_str.strip()

        # new lines
        out_str += "\n\n\n\n\n"

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
        
        if ``return_warnings==True``, returns file content and warnings
            e.g. if a certain feature is not available in ORCA
        
        else, returns str of header content
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_orca()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)
                warnings.extend(job_warnings)

        # if method isn't semi-empirical, get basis info to write later
        if not self.method.is_semiempirical and self.basis is not None:
            basis_info, basis_warnings = self.basis.get_orca_basis_info()
            warnings.extend(basis_warnings)
            if self.geometry is not None:
                warning = self.basis.check_for_elements(self.geometry)
                if warning is not None:
                    warnings.append(warning)

        else:
            basis_info = {}

        other_kw_dict = combine_dicts(other_kw_dict, basis_info)

        # get grid info
        if self.grid is not None:
            grid_info, warning = self.grid.get_orca()
            if warning is not None:
                warnings.append(warning)

            if any(
                "finalgrid" in x.lower() for x in other_kw_dict[ORCA_ROUTE]
            ):
                grid_info[ORCA_ROUTE].pop(1)

            other_kw_dict = combine_dicts(other_kw_dict, grid_info)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_orca()
            warnings.extend(warning)
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        # dispersion
        if self.empirical_dispersion is not None:
            dispersion, warning = self.empirical_dispersion.get_orca()
            if warning is not None:
                warnings.append(warning)

            other_kw_dict = combine_dicts(other_kw_dict, dispersion)

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
            func, method_warn = self.method.get_orca()
            warnings.extend(method_warn)
            other_kw_dict = combine_dicts(
                func, other_kw_dict
            )

        # add other route options
        if ORCA_ROUTE in other_kw_dict:
            used_keywords = []
            for kw in other_kw_dict[ORCA_ROUTE]:
                if any(kw.lower() == used_kw for used_kw in used_keywords):
                    continue
                used_keywords.append(kw.lower())
                out_str += " %s" % kw

        out_str += "\n"

        # procs
        if self.processors is not None:
            out_str += "%%pal\n    nprocs %i\nend\n" % self.processors

            # orca memory is per core, so only specify it if processors are specified
            if self.memory:
                out_str += "%%MaxCore %i\n" % (
                    int(1000 * self.memory / self.processors)
                )

        # add other blocks
        if ORCA_BLOCKS in other_kw_dict:
            for keyword in other_kw_dict[ORCA_BLOCKS]:
                if any(keyword.lower() == name for name in self.ORCA_BLOCKS_AFTER_MOL):
                    continue
                if any(other_kw_dict[ORCA_BLOCKS][keyword]):
                    used_settings = []
                    if keyword == "base":
                        out_str += "%%%s " % keyword
                        if isinstance(
                            other_kw_dict[ORCA_BLOCKS][keyword], str
                        ):
                            out_str += (
                                '"%s"\n' % other_kw_dict[ORCA_BLOCKS][keyword]
                            )
                        else:
                            out_str += (
                                '"%s"\n'
                                % other_kw_dict[ORCA_BLOCKS][keyword][0]
                            )
                    else:
                        out_str += "%%%s\n" % keyword
                        for opt in other_kw_dict[ORCA_BLOCKS][keyword]:
                            if any(
                                keyword.lower() == block_name for block_name in [
                                    "freq", "geom",
                                ]
                            ) and any(
                                opt.split()[0].lower() == prev_opt for prev_opt in used_settings
                            ):
                                continue
                            if opt.split()[0].lower() != "scan":
                                used_settings.append(opt.split()[0].lower())
                            out_str += "    %s\n" % opt
                        out_str += "end\n"

            out_str += "\n"

        # start of coordinate section - end of header
        out_str += "*xyz %i %i\n" % (self.charge, self.multiplicity)

        if return_warnings:
            return out_str, warnings
        return out_str

    def get_orca_footer(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        **other_kw_dict,
    ):
        """
        get ORCA input file header
        
        other_kw_dict is a dictionary with file positions (using ORCA_*)
        corresponding to options/keywords
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_orca()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)

        # if method isn't semi-empirical, get basis info to write later
        if not self.method.is_semiempirical and self.basis is not None:
            basis_info, basis_warnings = self.basis.get_orca_basis_info()
            warnings.extend(basis_warnings)
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
        out_str = "\n"

        # add other blocks
        if ORCA_BLOCKS in other_kw_dict:
            for keyword in other_kw_dict[ORCA_BLOCKS]:
                if not any(keyword.lower() == name for name in self.ORCA_BLOCKS_AFTER_MOL):
                    continue
                if any(other_kw_dict[ORCA_BLOCKS][keyword]):
                    if keyword == "base":
                        out_str += "%%%s " % keyword
                        if isinstance(
                            other_kw_dict[ORCA_BLOCKS][keyword], str
                        ):
                            out_str += (
                                '"%s"\n' % other_kw_dict[ORCA_BLOCKS][keyword]
                            )
                        else:
                            out_str += (
                                '"%s"\n'
                                % other_kw_dict[ORCA_BLOCKS][keyword][0]
                            )
                    else:
                        out_str += "%%%s\n" % keyword
                        for opt in other_kw_dict[ORCA_BLOCKS][keyword]:
                            out_str += "    %s\n" % opt
                        out_str += "end\n"

            out_str += "\n"

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
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_psi4()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)
                warnings.extend(job_warnings)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_psi4()
            warnings.extend(warning)
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        # get basis info if method is not semi empirical
        if not self.method.is_semiempirical and self.basis is not None:
            basis_info, basis_warnings = self.basis.get_psi4_basis_info(
                isinstance(self.method, SAPTMethod)
            )
            warnings.extend(basis_warnings)
            if self.geometry is not None:
                warning = self.basis.check_for_elements(self.geometry)
                if warning is not None:
                    warnings.append(warning)

            # aux basis sets might have a '%s' b/c the keyword to apply them depends on
            # the method - replace %s with the appropriate thing for the method
            for key in basis_info:
                if not isinstance(basis_info[key], list):
                    continue
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
        if self.memory:
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

                job_dict, job_warnings = job.get_psi4()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )
        if PSI4_COORDINATES not in other_kw_dict:
            other_kw_dict[PSI4_COORDINATES] = {}

        if "coords" not in other_kw_dict[PSI4_COORDINATES]:
            other_kw_dict[PSI4_COORDINATES]["coords"] = self.geometry.coords

        s = ""

        if isinstance(self.method, SAPTMethod) and not hasattr(
            self.charge, "__iter__"
        ):
            warnings.append(
                "for a SAPTMethod, charge and multiplicity should both be lists\n"
                "with the first item being the overall charge/multiplicity and\n"
                "subsequent items being the charge/multiplicity of the\n"
                "corresponding monomer"
            )
            return s, warnings

        if (
            isinstance(self.method, SAPTMethod)
            and sum(self.multiplicity[1:]) - len(self.multiplicity[1:]) + 1
            > self.multiplicity[0]
        ):
            use_molecule_array = True
            s += "mol = psi4.core.Molecule.from_arrays(\n"
            s += "    molecular_multiplicity=%i,\n" % self.multiplicity[0]
            s += "    molecular_charge=%i,\n" % self.charge[0]
            if PSI4_MOLECULE in other_kw_dict:
                for keyword in other_kw_dict[PSI4_MOLECULE]:
                    if other_kw_dict[keyword]:
                        s += "    %s=%s,\n" % (
                            keyword,
                            repr(other_kw_dict[keyword][0]),
                        )

        else:
            s += "molecule {\n"
            if isinstance(self.method, SAPTMethod):
                if not hasattr(self.charge, "__iter__"):
                    warnings.append(
                        "for a SAPTMethod, charge and multiplicity should both be lists\n"
                        "with the first item being the overall charge/multiplicity and\n"
                        "subsequent items being the charge/multiplicity of the\n"
                        "corresponding monomer"
                    )
                s += "    %2i %i\n" % (self.charge[0], self.multiplicity[0])
                if len(self.charge) > 1 and self.charge[0] != sum(
                    self.charge[1:]
                ):
                    warnings.append(
                        "total charge is not equal to sum of monomer charges"
                    )
            else:
                s += "    %2i %i\n" % (self.charge, self.multiplicity)

            if PSI4_MOLECULE in other_kw_dict:
                for keyword in other_kw_dict[PSI4_MOLECULE]:
                    if other_kw_dict[PSI4_MOLECULE][keyword]:
                        opt = other_kw_dict[PSI4_MOLECULE][keyword][0]
                        if (
                            "pubchem" in keyword.lower()
                            and not keyword.strip().endswith(":")
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
                for m2 in monomers[: i + 1]:
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
                                s += "%9.5f," % (val * UNIT.ANG_TO_BOHR)
                            else:
                                s += "%9.5f," % val
                        else:
                            warnings.append(
                                "unknown coordinate type: %s" % type(val)
                            )
                    s += "\n"

            s += "    ],\n"
            s += ")\n\n"
            s += "activate(mol)\n"

            if len(atoms_in_monomer) != len(self.geometry.atoms):
                from AaronTools.finders import NotAny

                warnings.append(
                    "there are atoms not in any monomers: %s"
                    % (
                        ", ".join(
                            [
                                atom.name
                                for atom in self.geometry.find(
                                    NotAny(atoms_in_monomer)
                                )
                            ]
                        )
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
                                s += " %9.5f" % (val * UNIT.ANG_TO_BOHR)
                            else:
                                s += " %9.5f" % val
                        elif isinstance(val, str):
                            s += "    %9s" % val
                        else:
                            warnings.append(
                                "unknown coordinate type: %s" % type(val)
                            )
                    s += "\n"

            if "variables" in other_kw_dict[PSI4_COORDINATES]:
                for (name, val, angstrom) in other_kw_dict[PSI4_COORDINATES][
                    "variables"
                ]:
                    if use_bohr and angstrom:
                        val *= UNIT.ANG_TO_BOHR
                    s += "     %3s = %9.5f\n" % (name, val)

            s += "}\n"

            if len(atoms_in_monomer) != len(self.geometry.atoms):
                from AaronTools.finders import NotAny

                warnings.append(
                    "there are atoms not in any monomers: %s"
                    % (
                        ", ".join(
                            [
                                atom.name
                                for atom in self.geometry.find(
                                    NotAny(atoms_in_monomer)
                                )
                            ]
                        )
                    )
                )

        elif not pubchem:
            for atom, coord in zip(
                self.geometry.atoms, other_kw_dict[PSI4_COORDINATES]["coords"]
            ):
                s += "    %2s" % atom.element
                for val in coord:
                    s += "     "
                    if isinstance(val, float):
                        s += " %9.5f" % val
                    elif isinstance(val, str):
                        s += " %9s" % val
                    else:
                        warnings.append(
                            "unknown coordinate type: %s" % type(val)
                        )
                s += "\n"

            if "variables" in other_kw_dict[PSI4_COORDINATES]:
                for (name, val, angstrom) in other_kw_dict[PSI4_COORDINATES][
                    "variables"
                ]:
                    if use_bohr and angstrom:
                        val *= UNIT.ANG_TO_BOHR
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

                job_dict, job_warnings = job.get_psi4()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_psi4()
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)
        
        # basis set
        if not self.method.is_semiempirical and self.basis is not None:
            basis_info, basis_warnings = self.basis.get_psi4_basis_info(
                isinstance(self.method, SAPTMethod)
            )
            other_kw_dict = combine_dicts(other_kw_dict, basis_info)
        
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
        if PSI4_SETTINGS in other_kw_dict and any(
            other_kw_dict[PSI4_SETTINGS][setting]
            for setting in other_kw_dict[PSI4_SETTINGS]
        ):
            out_str += "set {\n"
            for setting in other_kw_dict[PSI4_SETTINGS]:
                if other_kw_dict[PSI4_SETTINGS][setting]:
                    if isinstance(other_kw_dict[PSI4_SETTINGS][setting], str):
                        val = other_kw_dict[PSI4_SETTINGS][setting]
                    else:
                        if len(
                            other_kw_dict[PSI4_SETTINGS][setting]
                        ) == 1 and (
                            not any(
                                array_setting == setting.strip().lower()
                                for array_setting in self.FORCED_PSI4_ARRAY
                            )
                            or any(
                                single_setting == setting.strip().lower()
                                for single_setting in self.FORCED_PSI4_SINGLE
                            )
                        ):
                            val = other_kw_dict[PSI4_SETTINGS][setting][0]
                        else:
                            # array of values
                            val = "["
                            for v in other_kw_dict[PSI4_SETTINGS][setting]:
                                try:
                                    float(v)
                                    val += str(v)
                                except ValueError:
                                    val += "\"%s\"" % v
                                val += ","
                            val = val.rstrip(",")
                            val += "]"

                    out_str += "    %-20s    %s\n" % (setting, val)

            out_str += "}\n\n"


        if PSI4_SOLVENT in other_kw_dict:
            out_str += "pcm = {\n"
            for setting in other_kw_dict[PSI4_SOLVENT]:
                if other_kw_dict[PSI4_SOLVENT][setting]:
                    if isinstance(other_kw_dict[PSI4_SOLVENT][setting], str):
                        val = other_kw_dict[PSI4_SOLVENT][setting]
                        out_str += "    %s = %s\n" % (setting, val)
                    else:
                        if any(
                            single_setting == setting.strip().lower()
                            for single_setting in self.FORCED_PSI4_SOLVENT_SINGLE
                        ):
                            val = other_kw_dict[PSI4_SOLVENT][setting][0]
                            out_str += "    %s = %s\n" % (setting, val)
                        else:
                            # array of values
                            if not out_str.endswith("\n\n") and not out_str.endswith("{\n"):
                                out_str += "\n"
                            out_str += "    %s {\n" % setting
                            for val in other_kw_dict[PSI4_SOLVENT][setting]:
                                out_str += "        %s\n" % val
                            out_str += "    }\n\n"

            out_str += "}\n\n"


        if PSI4_OPTKING in other_kw_dict and any(
            other_kw_dict[PSI4_OPTKING][setting]
            for setting in other_kw_dict[PSI4_OPTKING]
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
            disp = self.empirical_dispersion.get_psi4()[0]
            if "%s" in method:
                method = method % disp
            else:
                method += disp
        elif "%s" in method:
            method = method.replace("%s", "")

        warning = self.method.sanity_check_method(method, "psi4")
        if warning:
            warnings.append(warning)

        # after job stuff - replace METHOD with method
        if PSI4_BEFORE_JOB in other_kw_dict:
            for opt in other_kw_dict[PSI4_BEFORE_JOB]:
                if "$METHOD" in opt:
                    opt = opt.replace("$METHOD", "'%s'" % method)

                # values in other_kw_dict[PSI4_BEFORE_JOB] come in as list of lists,
                # so just grabbing first one
                out_str += opt[0]
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
                    if func == "gradient":
                        out_str += "grad, wfn = %s('%s'" % (func, method)
                    else:
                        out_str += "nrg, wfn = %s('%s'" % (func, method)
                else:
                    if func == "gradient":
                        out_str += "grad = %s('%s'" % (func, method)
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

    def get_xtb_cmdline(self, config):
        """
        Uses the config and job type to set command line options for xtb and crest jobs

        Returns a dictionary of option=val pairs; val is None when option doesn't take
        an argument. This dict should be parsed by the caller into the command line
        string.
        """
        if len(self.job_type) > 1:
            raise NotImplementedError(
                "Multiple job types not supported for crest/xtb"
            )
        cmdline = {}
        job_type = self.job_type[0]
        style = config["Job"]["exec_type"]
        if style not in ["xtb", "crest"]:
            raise NotImplementedError(
                "Wrong executable type: %s (can only get command line options "
                "for `xtb` or `crest`)" % style
            )

        # pull in stuff set by resolve_error
        if config._args:
            for arg in config._args:
                cmdline[arg] = None
        if config._kwargs:
            for key, val in config._kwargs.items():
                cmdline[key] = val
        # run types
        if "gfn" in config["Theory"]:
            cmdline["--gfn"] = config["Job"]["gfn"]
        if (
            style == "xtb"
            and hasattr(job_type, "transition_state")
            and job_type.transition_state
        ):
            cmdline["--optts"] = None
        elif style == "xtb":
            cmdline["--opt"] = None

        # charge/mult/temp
        if self.charge != 0:
            cmdline["--chrg"] = self.charge
        if self.multiplicity != 1:
            cmdline["--uhf"] = self.multiplicity - 1
        if style == "crest":
            cmdline["--temp"] = config["Theory"].get(
                "temperature", fallback="298"
            )
        else:
            cmdline["--etemp"] = config["Theory"].get(
                "temperature", fallback="298"
            )

        # screening parameters
        if (
            style == "crest"
            and "energy_cutoff" in config["Job"]
            and config["Job"].getfloat("energy_cutoff") > 6
        ):
            cmdline["--ewin"] = config["Job"]["energy_cutoff"]
        if (
            style == "crest"
            and "rmsd_cutoff" in config["Job"]
            and config["Job"].getfloat("rmsd_cutoff") < 0.125
        ):
            cmdline["--rthr"] = config["Job"]["rmsd_cutoff"]

        # solvent stuff
        if (
            "solvent_model" in config["Theory"]
            and config["Theory"]["solvent_model"] == "alpb"
        ):
            solvent = config["Theory"]["solvent"].split()
            if len(solvent) > 1:
                solvent, ref = solvent
            elif len(solvent) == 1:
                solvent, ref = solvent[0], None
            else:
                raise ValueError
            if solvent.lower() not in [
                "acetone",
                "acetonitrile",
                "aniline",
                "benzaldehyde",
                "benzene",
                "ch2cl2",
                "chcl3",
                "cs2",
                "dioxane",
                "dmf",
                "dmso",
                "ether",
                "ethylacetate",
                "furane",
                "hexandecane",
                "hexane",
                "methanol",
                "nitromethane",
                "octanol",
                "woctanol",
                "phenol",
                "toluene",
                "thf",
                "water",
            ]:
                raise ValueError("%s is not a supported solvent" % solvent)
            if ref is not None and ref.lower() not in ["reference", "bar1m"]:
                raise ValueError(
                    "%s Gsolv reference state not supported" % ref
                )
            if style.lower() == "crest" or ref is None:
                cmdline["--alpb"] = "{}".format(solvent)
            else:
                cmdline["--alpb"] = "{} {}".format(solvent, ref)
        elif (
            "solvent_model" in config["Theory"]
            and config["Theory"]["solvent_model"] == "gbsa"
        ):
            solvent = config["Theory"]["solvent"].split()
            if len(solvent) > 1:
                solvent, ref = solvent
            else:
                solvent, ref = solvent[0], None
            if solvent.lower() not in [
                "acetone",
                "acetonitrile",
                "benzene",
                "ch2cl2",
                "chcl3",
                "cs2",
                "dmf",
                "dmso",
                "ether",
                "h2o",
                "methanol",
                "n-hexane",
                "thf",
                "toluene",
            ]:
                gfn = config["Theory"].get("gfn", fallback="2")
                if gfn != "1" and solvent.lower() in ["benzene"]:
                    raise ValueError("%s is not a supported solvent" % solvent)
                elif gfn != "2" and solvent.lower() in ["DMF", "n-hexane"]:
                    raise ValueError("%s is not a supported solvent" % solvent)
                else:
                    raise ValueError("%s is not a supported solvent" % solvent)
            if ref is not None and ref.lower() not in ["reference", "bar1m"]:
                raise ValueError(
                    "%s Gsolv reference state not supported" % ref
                )
            if style.lower() == "crest" or ref is None:
                cmdline["--gbsa"] = "{}".format(solvent)
            else:
                cmdline["--gbsa"] = "{} {}".format(solvent, ref)

        other = config["Job"].get("cmdline", fallback="").split()
        i = 0
        while i < len(other):
            if other[i].startswith("-"):
                key = other[i]
                cmdline[key] = None
            else:
                cmdline[key] = other[i]
            i += 1
        return cmdline

    def get_xcontrol(self, config, ref=None):
        if len(self.job_type) > 1:
            raise NotImplementedError(
                "Multiple job types not supported for crest/xtb"
            )
        job_type = self.job_type[0]
        return job_type.get_xcontrol(config, ref=ref)

    def get_sqm_header(
            self,
            return_warnings=False,
            conditional_kwargs=None,
            **other_kw_dict,
    ):
        """retruns header, warnings_list for sqm job"""
        
        warnings = []
   

        if conditional_kwargs is None:
            conditional_kwargs = {}

        # job stuff
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_sqm()
                warnings.extend(job_warnings)
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        s = ""

        # charge and mult
        other_kw_dict = combine_dicts(
            {
                SQM_QMMM: {
                    "qmcharge": [str(self.charge)],
                    "spin": [str(self.multiplicity)],
                }
            },
            other_kw_dict,
        )

        # comment
        if SQM_COMMENT not in other_kw_dict:
            if self.geometry.comment:
                other_kw_dict[SQM_COMMENT] = [self.geometry.comment]
            else:
                other_kw_dict[SQM_COMMENT] = [self.geometry.name]

        for comment in other_kw_dict[SQM_COMMENT]:
            for line in comment.split("\n"):
                s += "%s" % line
            s += "\n"
        
        # method
        if self.method:
            method = self.method.get_sqm()
            warning = self.method.sanity_check_method(method, "sqm")
            if warning:
                warnings.append(warning)
            other_kw_dict = combine_dicts(
                other_kw_dict,
                {SQM_QMMM: {"qm_theory": [method]}}
            )
        
        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        # options
        s += " &qmmm\n"
        for key in other_kw_dict[SQM_QMMM]:
            if not other_kw_dict[SQM_QMMM][key]:
                continue
            s += "   %s=" % key
            option = other_kw_dict[SQM_QMMM][key][0]
            if re.match("-?\d+", option):
                s += "%s,\n" % option
            elif any(option.lower() == b for b in [".true.", ".false."]):
                s += "%s,\n" % option
            else:
                s += "'%s',\n" % option
        s += " /\n"
        
        return s, warnings
    
    def get_sqm_molecule(
        self,
        **kwargs,
    ):
        """returns molecule specification for sqm input"""
        
        warnings = []
        s = ""

        for atom in self.geometry.atoms:
            s += " %2i  %2s  %9.5f  %9.5f  %9.5f\n" % (
                ELEMENTS.index(atom.element),
                atom.element,
                atom.coords[0],
                atom.coords[1],
                atom.coords[2],
            )
        
        return s.rstrip(), warnings

    def get_qchem_header(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        **other_kw_dict,
    ):
        """
        write QChem input file header (up to charge mult)
        
        other_kw_dict is a dictionary with file positions (using QCHEM_*)
        corresponding to options/keywords
        
        returns warnings if a certain feature is not available in QChem
        """

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        job_type_count = 0
        if self.job_type is not None:
            for i, job in enumerate(self.job_type[::-1]):
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_qchem()
                if isinstance(job, FrequencyJob) and job.temperature != 298.15:
                    warnings.append(
                        "thermochemistry data in the output file might be for 298.15 K\n"
                        "in spite of the user setting %.2f K\n" % (job.temperature) + \
                        "free energy corrections can be calculated at different\n"
                        "temperatures using AaronTools grabThermo.py script or\n"
                        "SEQCROW's Procress QM Thermochemistry tool"
                    )
                if QCHEM_REM in job_dict and any(key.lower() == "job_type" for key in job_dict[QCHEM_REM]):
                    job_type_count += 1
                if job_type_count > 1:
                    raise NotImplementedError("cannot put multiple JOB_TYPE entries in one Q-Chem header")
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)

        if (
            QCHEM_COMMENT not in other_kw_dict
            or not other_kw_dict[QCHEM_COMMENT]
        ):
            if self.geometry.comment:
                other_kw_dict[QCHEM_COMMENT] = [self.geometry.comment]
            else:
                other_kw_dict[QCHEM_COMMENT] = [self.geometry.name]

        if QCHEM_SETTINGS in other_kw_dict:
            other_kw_dict = combine_dicts(
                other_kw_dict,
                {QCHEM_SETTINGS: {QCHEM_COMMENT: other_kw_dict[QCHEM_COMMENT]}},
            )
        else:
            other_kw_dict[QCHEM_SETTINGS] = {QCHEM_COMMENT: other_kw_dict[QCHEM_COMMENT]}

        # add memory info
        if self.memory:
            other_kw_dict = combine_dicts(
                other_kw_dict,
                {QCHEM_REM: {"MEM_TOTAL": str(1000 * self.memory)}}
            )

        # add EmpiricalDispersion info
        if self.empirical_dispersion is not None:
            disp, warning = self.empirical_dispersion.get_qchem()
            other_kw_dict = combine_dicts(other_kw_dict, disp)
            if warning is not None:
                warnings.append(warning)

        # add Integral(grid=X)
        if self.grid is not None:
            grid, warning = self.grid.get_qchem()
            other_kw_dict = combine_dicts(other_kw_dict, grid)
            if warning is not None:
                warnings.append(warning)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_qchem()
            if warning is not None:
                warnings.extend(warning)
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        if self.method is not None:
            func, warning = self.method.get_qchem()
            if warning is not None:
                warnings.append(warning)

            warning = self.method.sanity_check_method(func, "qchem")
            if warning:
                warnings.append(warning)
            
            # Q-Chem seems to still require a basis set for HF-3c
            # if not self.method.is_semiempirical and self.basis is not None:
            if self.basis is not None:
                (
                    basis_info,
                    basis_warnings,
                ) = self.basis.get_qchem_basis_info(self.geometry)
                warnings.extend(basis_warnings)
                # check basis elements to make sure no element is
                # in two basis sets or left out of any
                other_kw_dict = combine_dicts(
                    other_kw_dict,
                    basis_info,
                )
                if self.geometry is not None:
                    basis_warning = self.basis.check_for_elements(
                        self.geometry,
                    )
                    if basis_warning is not None:
                        warnings.append(basis_warning)

            other_kw_dict = combine_dicts(
                other_kw_dict, {QCHEM_REM: {"METHOD": func}},
            )

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        out_str = ""

        if QCHEM_REM in other_kw_dict and QCHEM_SETTINGS in other_kw_dict:
            other_kw_dict[QCHEM_SETTINGS] = combine_dicts(
                {"rem": other_kw_dict[QCHEM_REM]}, other_kw_dict[QCHEM_SETTINGS],
            )
        
        elif QCHEM_REM in other_kw_dict:
            other_kw_dict[QCHEM_SETTINGS] = {"rem": other_kw_dict[QCHEM_REM]}
        
        else:
            warnings.append("no REM section")

        if self.memory:
            other_kw_dict = combine_dicts(
                other_kw_dict,
                {"rem": {"MEM_TOTAL": "%i" % (1000 * self.memory)}}
            )

        if QCHEM_SETTINGS in other_kw_dict:
            for section in other_kw_dict[QCHEM_SETTINGS]:
                settings = other_kw_dict[QCHEM_SETTINGS][section]
                out_str += "$%s\n" % section
                for setting in settings:
                    if not setting:
                        continue
                    if isinstance(settings, dict):
                        opt = settings[setting]
                        if not opt:
                            continue
                        if isinstance(opt, str):
                            val = opt
                            out_str += "    %-20s =   %s\n" % (setting, val)
                        elif isinstance(opt, dict):
                            for s, v in opt.items():
                                out_str += "    %-20s =   %s\n" % (s, v)
                        else:
                            if len(opt) == 1:
                                val = opt[0]
                                out_str += "    %-20s =   %s\n" % (setting, val)
                            elif not opt:
                                out_str += "    %-20s\n" % setting
                            else:
                                if section.lower() == "rem" and setting.lower() == "job_type":
                                    raise NotImplementedError(
                                        "cannot put multiple JOB_TYPE entries in one Q-Chem header"
                                    )
                                out_str += "    %-20s =   %s\n" % (setting, ", ".join(opt))

    
                    elif hasattr(setting, "__iter__") and not isinstance(setting, str):
                        for val in setting:
                            out_str += "    %s\n" % val
                    
                    else:
                        out_str += "    %s\n" % setting
    
                out_str += "$end\n\n"
        
        return out_str, warnings
    
    def get_qchem_molecule(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        **other_kw_dict,
    ):

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_qchem()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)

        if (
            QCHEM_COMMENT not in other_kw_dict
            or not other_kw_dict[QCHEM_COMMENT]
        ):
            if self.geometry.comment:
                other_kw_dict[QCHEM_COMMENT] = [self.geometry.comment]
            else:
                other_kw_dict[QCHEM_COMMENT] = [self.geometry.name]

        # add EmpiricalDispersion info
        if self.empirical_dispersion is not None:
            disp, warning = self.empirical_dispersion.get_qchem()
            other_kw_dict = combine_dicts(other_kw_dict, disp)
            if warning is not None:
                warnings.append(warning)

        # add Integral(grid=X)
        if self.grid is not None:
            grid, warning = self.grid.get_qchem()
            other_kw_dict = combine_dicts(other_kw_dict, grid)
            if warning is not None:
                warnings.append(warning)

        # add implicit solvent
        if self.solvent is not None:
            solvent_info, warning = self.solvent.get_qchem()
            if warning is not None:
                warnings.extend(warning)
            other_kw_dict = combine_dicts(other_kw_dict, solvent_info)

        other_kw_dict = combine_dicts(
            other_kw_dict, conditional_kwargs, dict2_conditional=True
        )

        out_str = "$molecule\n    %i %i\n" % (
            self.charge, self.multiplicity
        )

        if QCHEM_MOLECULE in other_kw_dict:
            for line in other_kw_dict[QCHEM_MOLECULE]:
                out_str += "    %-20s\n" % line
        elif not self.geometry:
            warnings.append("no molecule")
        
        for atom in self.geometry.atoms:
            out_str += "    %-2s" % atom.element
            out_str += "   %9.5f    %9.5f    %9.5f\n" % tuple(atom.coords)

        out_str += "$end\n"
        
        return out_str, warnings

    def get_xtb_control(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        crest=False,
        **other_kw_dict,
    ):
        
        other_kw_dict = combine_dicts(other_kw_dict, self.kwargs)
        
        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                if crest:
                    job_dict, job_warnings = job.get_crest()
                else:
                    job_dict, job_warnings = job.get_xtb()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)
        
        if self.method is not None:
            func, warning = self.method.get_xtb()
            if warning is not None:
                warnings.append(warning)
            other_kw_dict = combine_dicts(func, other_kw_dict)

            warning = self.method.sanity_check_method(self.method.name, "xtb")
            if warning:
                warnings.append(warning)

        write_geom = False

        xc_str = "$chrg %i\n" % self.charge
        if self.multiplicity > 0:
            xc_str += "$spin %i\n" % (self.multiplicity - 1)
        else:
            xc_str += "$spin %i\n" % (self.multiplicity + 1)

        other_kw_dict = combine_dicts(other_kw_dict, conditional_kwargs, dict2_conditional=True)

        if XTB_CONTROL_BLOCKS in other_kw_dict:
            for block, settings in other_kw_dict[XTB_CONTROL_BLOCKS].items():
                if not settings:
                    continue
                xc_str += "$%s\n" % block
                for setting in settings:
                    xc_str += "  %s\n" % setting
                    if block.lower() == "metadyn" and setting.lower().startswith("coord"):
                        write_geom = "=".join(setting.split("=")[1:])
                    elif block.lower() == "constrain" and setting.lower().startswith("reference"):
                        write_geom = "=".join(setting.split("=")[1:])
                xc_str += "\n"
        
        return xc_str, warnings, write_geom
    
    def get_xtb_cmd(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        split_words=False,
        **other_kw_dict,
    ):
        other_kw_dict = combine_dicts(other_kw_dict, self.kwargs)

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_xtb()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)
        
        if self.method is not None:
            func, warning = self.method.get_xtb()
            if warning is not None:
                warnings.append(warning)
            other_kw_dict = combine_dicts(func, other_kw_dict)
        
        if self.solvent is not None:
            solvent, warning = self.solvent.get_xtb()
            if warning:
                warnings.extend(warning)
            other_kw_dict = combine_dicts(solvent, other_kw_dict)

        if self.processors:
            other_kw_dict = combine_dicts(
                {XTB_COMMAND_LINE: {"parallel": [str(self.processors)]}},
                other_kw_dict,
            )

        other_kw_dict = combine_dicts(other_kw_dict, conditional_kwargs, dict2_conditional=True)

        out = ["xtb", "--input", "{{ name }}.xc", "{{ name }}.xyz"]
        if XTB_COMMAND_LINE in other_kw_dict:
            for flag, option in other_kw_dict[XTB_COMMAND_LINE].items():
                out.append("--%s" % flag)
                if option:
                    out.append(",".join(option))
        
        if not split_words:
            out = " ".join(out)
        
        return out, warnings
    
    def get_crest_cmd(
        self,
        return_warnings=False,
        conditional_kwargs=None,
        split_words=False,
        **other_kw_dict,
    ):
        other_kw_dict = combine_dicts(other_kw_dict, self.kwargs)

        if conditional_kwargs is None:
            conditional_kwargs = {}

        warnings = []
        if self.job_type is not None:
            for job in self.job_type[::-1]:
                if hasattr(job, "geometry"):
                    job.geometry = self.geometry

                job_dict, job_warnings = job.get_crest()
                other_kw_dict = combine_dicts(job_dict, other_kw_dict)
                warnings.extend(job_warnings)
        
        if self.method is not None:
            func, warning = self.method.get_crest()
            if warning is not None:
                warnings.append(warning)
            other_kw_dict = combine_dicts(func, other_kw_dict)
        
        if self.solvent is not None:
            solvent, warning = self.solvent.get_xtb()
            if warning:
                warnings.extend(warning)
            other_kw_dict = combine_dicts(solvent, other_kw_dict)

        if self.processors:
            other_kw_dict = combine_dicts(
                {CREST_COMMAND_LINE: {"T": [str(self.processors)]}},
                other_kw_dict,
            )

        other_kw_dict = combine_dicts(other_kw_dict, conditional_kwargs, dict2_conditional=True)

        out = ["crest", "{{ name }}.xyz", "--cinp", "{{ name }}.xc"]
        if CREST_COMMAND_LINE in other_kw_dict:
            for flag, option in other_kw_dict[CREST_COMMAND_LINE].items():
                out.append("--%s" % flag)
                if option:
                    out.append(",".join(option))
        
        if not split_words:
            out = " ".join(out)
        
        return out, warnings
        
