"""placeholder stuff until Aaron.Theory is moved"""

from AaronTools.utils.utils import combine_dicts
from AaronTools.theory import ORCA_ROUTE, ORCA_BLOCKS, ORCA_COORDINATES, ORCA_COMMENT, \
                              \
                              PSI4_SETTINGS, PSI4_BEFORE_GEOM, PSI4_AFTER_JOB, PSI4_BEFORE_JOB, \
                              PSI4_COMMENT, PSI4_COORDINATES, PSI4_JOB, PSI4_OPTKING, \
                              \
                              GAUSSIAN_PRE_ROUTE, GAUSSIAN_ROUTE, GAUSSIAN_COORDINATES, \
                              GAUSSIAN_CONSTRAINTS, GAUSSIAN_GEN_BASIS, GAUSSIAN_GEN_ECP, \
                              GAUSSIAN_POST, GAUSSIAN_COMMENT

from .emp_dispersion import EmpiricalDispersion
from .grid import IntegrationGrid
from .basis import BasisSet
from .functional import Functional, KNOWN_SEMI_EMPIRICAL

def probable_job_order(word):
    """sorts job keywords so that the job is other settings -> opt -> freq -> sp"""
    if word.lower() in ['energy', 'sp']:
        return 9003
    elif word.lower().startswith('opt'):
        return 9001
    elif word.lower().startswith('freq') or word.lower().startswith('numfreq'):
        return 9002
    else:
        return 0

class Theory:
    """a Theory object can be used to create an input file for different QM software
    The creation of a Theory object does not depend on the specific QM software - that is determined when the file is written
    attribute names are the same as initialization keywords
    valid initialization key words are:
    geometry                -   AaronTools Geometry 
    charge                  -   total charge
    multiplicity            -   electronic multiplicity
    job_types               -   list(JobType)

    functional              -   Functional object (or str - Functional instance will be created)
    basis                   -   BasisSet object (or str - will be set to BasisSet(Basis(kw)))
    empirical_dispersion    -   EmpiricalDispersion object (or str)
    grid                    -   IntegrationGrid object (or str)
    
    memory                  -   allocated memory (GB)
    processors              -   allocated cores

    methods that construct headers and footers can specify some keyword arguments
    keywords are ORCA_*, PSI4_*, or GAUSSIAN_* (imported from AaronTools.theory)
    ORCA_ROUTE: list(str)
    ORCA_BLOCKS: dict(list(str)) - keys are block names minus %
    ORCA_COORDINATES: ignored
    ORCA_COMMENT: list(str)

    PSI4_SETTINGS: dict(setting_name: [value])
    PSI4_BEFORE_GEOM: list(str)
    PSI4_AFTER_JOB: list(str) - $FUNCTIONAL will be replaced with functional name
    PSI4_COMMENT: list(str)
    PSI4_COORDINATES: dict(str:list(str)) e.g. {'symmetry': ['c1']}
    PSI4_JOB: dict(optimize/frequencies/etc: list(str - $FUNCTIONAL replaced w/ functional))
    PSI4_OPTKING: dict(setting_name: [value])

    GAUSSIAN_PRE_ROUTE: dict(list(str)) - keys are link0 minus %
    GAUSSIAN_ROUTE: dict(list(str)) - e.g. {'opt': ['NoEigenTest', 'Tight']}
    GAUSSIAN_COORDINATES: ignored
    GAUSSIAN_CONSTRAINTS: list(str)
    GAUSSIAN_GEN_BASIS: list(str) - only filled by BasisSet automatically when writing footer
    GAUSSIAN_GEN_ECP: list(str) - only filled by BasisSet automatically when writing footer
    GAUSSIAN_POST: list(str)
    GAUSSIAN_COMMENT: list(str)
"""

    ACCEPTED_INIT_KW = ['geometry', \
                        'memory', \
                        'processors', \
                        'job_types']

    def __init__(self, charge=0, multiplicity=1, functional=None, basis=None, empirical_dispersion=None, grid=None, **kw):
        self.charge = charge
        self.multiplicity = multiplicity

        for key in self.ACCEPTED_INIT_KW:
            if key in kw:
                self.__setattr__(key, kw[key])
            else:
                self.__setattr__(key, None)

            #if functional, basis, etc aren't the expected classes, make them so
            if functional is not None:
                if not isinstance(functional, Functional):
                    functional = Functional(functional, functional.upper() in KNOWN_SEMI_EMPIRICAL)
            
            self.functional = functional

            if grid is not None:
                if not isinstance(grid, IntegrationGrid):
                    grid = IntegrationGrid(grid)
            
            self.grid = grid

            if basis is not None:
                if not isinstance(basis, BasisSet):
                    basis = BasisSet(basis)
            
            self.basis = basis

            if empirical_dispersion is not None:
                if not isinstance(empirical_dispersion, EmpiricalDispersion):
                    empirical_dispersion = EmpiricalDispersion(empirical_dispersion)
            
            self.empirical_dispersion = empirical_dispersion

    def make_header(self, geom, style='gaussian', **kwargs):
        """geom: Geometry
        step: float
        form: str, gaussian, orca, or psi4
        kwargs: keys are ORCA_*, PSI4_*, or GAUSSIAN_*"""

        self.geometry = geom
        if self.basis is not None:
            self.basis.refresh_elements(self.geometry)

        other_kw_dict = {}
        for kw in kwargs:
            if kw.startswith('PSI4_') or kw.startswith('ORCA_') or kw.startswith('GAUSSIAN_'):
                new_kw = eval(kw)
                other_kw_dict[new_kw] = kwargs[kw]
            else:
                other_kw_dict[kw] = kwargs[kw]

        if style == "gaussian":
            return self.get_gaussian_header(**other_kw_dict)
        
        elif style == "orca":
            return self.get_orca_header(**other_kw_dict)
        
        elif style == "psi4":
            return self.get_psi4_header(**other_kw_dict)
    
    def make_footer(self, geom, style='gaussian', **kwargs):
        """geom: Geometry
        step: float (ignored)
        form: str, gaussian or psi4
        kwargs: keys are GAUSSIAN_*, ORCA_*, or PSI4_*
        """
        if self.basis is not None:
            self.basis.refresh_elements(geom)
        
        other_kw_dict = {}
        for kw in kwargs:
            if kw.startswith('PSI4_') or kw.startswith('ORCA_') or kw.startswith('GAUSSIAN_'):
                new_kw = eval(kw)
                other_kw_dict[new_kw] = kwargs[kw]
            else:
                other_kw_dict[kw] = kwargs[kw]

        if style == "gaussian":
            return self.get_gaussian_footer(**other_kw_dict)

        elif style == "psi4":
            return self.get_psi4_footer(**other_kw_dict)

    def get_gaussian_header(self, return_warnings=False, **other_kw_dict):
        """write Gaussian09/16 input file header (up to charge mult)
        other_kw_dict is a dictionary with file positions (using GAUSSIAN_*)
        corresponding to options/keywords
        returns warnings if a certain feature is not available in Gaussian"""

        if self.job_types is not None:
            for job in self.job_types:
                job_dict = job.get_gaussian()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)

        warnings = []
        s = ""

        #processors, memory, and other link 0 stuff
        if self.processors is not None:
            s += "%%NProcShared=%i\n" % self.processors

        if self.memory is not None:
            s += "%%Mem=%iGB\n" % self.memory

        if GAUSSIAN_PRE_ROUTE in other_kw_dict:
            for key in other_kw_dict[GAUSSIAN_PRE_ROUTE]:
                s += "%%%s" % key
                if len(other_kw_dict[GAUSSIAN_PRE_ROUTE][key]) > 0:
                    s += "=%s" % ",".join(other_kw_dict[GAUSSIAN_PRE_ROUTE][key])

                if not s.endswith('\n'):
                    s += '\n'

        #start route line with functional
        s += "#n "
        if self.functional is not None:
            func, warning = self.functional.get_gaussian()
            if warning is not None:
                warnings.append(warning)
            s += "%s" % func
            if not self.functional.is_semiempirical and self.basis is not None:
                basis_info = self.basis.get_gaussian_basis_info()
                if self.geometry is not None:
                    #check basis elements to make sure no element is in two basis sets or left out of any
                    basis_warning = self.basis.check_for_elements([atom.element for atom in self.geometry.atoms])
                    if basis_warning is not None:
                        warnings.append(warning)
    
                if GAUSSIAN_ROUTE in basis_info:
                    s += "%s" % basis_info[GAUSSIAN_ROUTE]

            s += " "

        #add EmpiricalDispersion info
        if self.empirical_dispersion is not None:
            disp, warning = self.empirical_dispersion.get_gaussian()
            other_kw_dict = combine_dicts(other_kw_dict, disp)
            if warning is not None:
                warnings.append(warning)

        #add Integral(grid=X)
        if self.grid is not None:
            grid, warning = self.grid.get_gaussian()
            other_kw_dict = combine_dicts(other_kw_dict, grid)
            if warning is not None:
                warnings.append(warning)

        #add other route options
        #only one option can be specfied
        #e.g. for {'Integral':['grid=X', 'grid=Y']}, only grid=X will be used
        if GAUSSIAN_ROUTE in other_kw_dict.keys():
            #reverse order b/c then freq comes after opt
            for option in sorted(other_kw_dict[GAUSSIAN_ROUTE].keys(), key=probable_job_order):
                known_opts = []
                s += option
                if len(other_kw_dict[GAUSSIAN_ROUTE][option]) > 1 or \
                   (len(other_kw_dict[GAUSSIAN_ROUTE][option]) == 1 and \
                   ('=' in other_kw_dict[GAUSSIAN_ROUTE][option][0] or \
                    '(' in other_kw_dict[GAUSSIAN_ROUTE][option][0])):
                    opts = [opt.split('=')[0] for opt in other_kw_dict[GAUSSIAN_ROUTE][option]]

                    s += "=("
                    for x in other_kw_dict[GAUSSIAN_ROUTE][option]:
                        opt = x.split('=')[0]
                        if opt not in known_opts:
                            if len(known_opts) > 0:
                                s += ','
                            known_opts.append(opt)
                            s += x
                    s += ")"
                
                elif len(other_kw_dict[GAUSSIAN_ROUTE][option]) == 1:
                    s += "=%s" % other_kw_dict[GAUSSIAN_ROUTE][option][0]

                s += " "

        s += "\n\n"

        #add comment, removing any trailing newlines
        if GAUSSIAN_COMMENT in other_kw_dict:
            if len(other_kw_dict[GAUSSIAN_COMMENT]) > 0:
                s += "\n".join([x.rstrip() for x in other_kw_dict[GAUSSIAN_COMMENT]])
            else:
                s += "comment"

            if not s.endswith('\n'):
                s += '\n'

        s += "\n"

        #charge mult
        s += "%i %i\n" % (self.charge, self.multiplicity)

        if return_warnings:
            return s, warnings
        else:
            return s


    def get_gaussian_footer(self, return_warnings=False, **other_kw_dict):
        """write footer of gaussian input file"""
        
        if self.job_types is not None:
            for job in self.job_types:
                job_dict = job.get_gaussian()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)

        s = ""
        warnings = []

        #if functional is not semi emperical, basis set might be gen or genecp
        #get basis info (will be written after constraints)
        if self.functional is not None and not self.functional.is_semiempirical and self.basis is not None:
            basis_info = self.basis.get_gaussian_basis_info()
            basis_elements = self.basis.elements_in_basis
            #check if any element is in multiple basis sets
            #check to make sure all elements have a basis set
            if self.geometry is not None:
                basis_warning = self.basis.check_for_elements([atom.element for atom in self.geometry.atoms])
                if basis_warning is not None:
                    warnings.append(basis_warning)

        s += "\n"

        #bond, angle, and torsion constraints
        if GAUSSIAN_CONSTRAINTS in other_kw_dict:
            for constraint in other_kw_dict[GAUSSIAN_CONSTRAINTS]:
                s += constraint
                s += '\n'

            s += '\n'

        #write gen info
        if self.functional is not None and not self.functional.is_semiempirical:
            if GAUSSIAN_GEN_BASIS in basis_info:
                s += basis_info[GAUSSIAN_GEN_BASIS]
            
                s += "\n"

            if GAUSSIAN_GEN_ECP in basis_info:
                s += basis_info[GAUSSIAN_GEN_ECP]

        #post info e.g. for NBOREAD
        if GAUSSIAN_POST in other_kw_dict:
            for item in other_kw_dict[GAUSSIAN_POST]:
                s += item
                s += " "

            s += '\n'
        
        #new lines
        s += '\n\n'

        if return_warnings:
            return s, warnings
        else:
            return s

    def get_orca_header(self, return_warnings=False, **other_kw_dict):
        """get ORCA input file header
        other_kw_dict is a dictionary with file positions (using ORCA_*)
        corresponding to options/keywords
        returns file content and warnings e.g. if a certain feature is not available in ORCA
        returns str of header content
        if return_warnings, returns str, list(warning)"""
        
        if self.job_types is not None:
            for job in self.job_types:
                job_dict = job.get_orca()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)

        warnings = []

        #if functional isn't semi-empirical, get basis info to write later
        if not self.functional.is_semiempirical:
            basis_info = self.basis.get_orca_basis_info()
            if self.geometry is not None:
                struc_elements = set([atom.element for atom in self.geometry.atoms])
            
                warning = self.basis.check_for_elements(struc_elements)
                if warning is not None:
                    warnings.append(warning)

        else:
            basis_info = {}

        combined_dict = combine_dicts(other_kw_dict, basis_info)

        #get grid info
        if self.grid is not None:
            grid_info, warning = self.grid.get_orca()
            if warning is not None:
                warnings.append(warning)

            if any('finalgrid' in x.lower() for x in combined_dict[ORCA_ROUTE]):
                grid_info[ORCA_ROUTE].pop(1)

            combined_dict = combine_dicts(combined_dict, grid_info)

        #start building input file header
        s = ""

        #comment
        if ORCA_COMMENT in combined_dict:
            for comment in combined_dict[ORCA_COMMENT]:
                for line in comment.split('\n'):
                    s += "#%s\n" % line

        s += "!"
        #functional
        if self.functional is not None:
            func, warning = self.functional.get_orca()
            if warning is not None:
                warnings.append(warning)
            s += " %s" % func

        #dispersion
        if self.empirical_dispersion is not None:
            if not s.endswith(' '):
                s += " "

            #TODO make dispersion behave like grid, returning ({ORCA_ROUTE:['D2']}, None) or w/e
            dispersion, warning = self.empirical_dispersion.get_orca()
            if warning is not None:
                warnings.append(warning)

            s += "%s" % dispersion

        #add other route options
        if ORCA_ROUTE in combined_dict:
            if not s.endswith(' '):
                s += " "
                
                s += " ".join(combined_dict[ORCA_ROUTE])

        s += "\n"

        #procs
        if self.processors is not None:
            s += "%%pal\n    nprocs %i\nend\n" % self.processors

            #orca memory is per core, so only specify it if processors are specified
            if self.memory is not None:
                s += "%%MaxCore %i\n" % (int(1000 * self.memory / self.processors))

        #add other blocks
        if ORCA_BLOCKS in combined_dict:
            for kw in combined_dict[ORCA_BLOCKS]:
                if any(len(x) > 0 for x in combined_dict[ORCA_BLOCKS][kw]):
                    s += "%%%s\n" % kw
                    for opt in combined_dict[ORCA_BLOCKS][kw]:
                        s += "    %s\n" % opt
                    s += "end\n"

            s += "\n"

        #start of coordinate section - end of header
        s += "*xyz %i %i\n" % (self.charge, self.multiplicity)
        
        if return_warnings:
            return s, warnings
        else:
            return s

    def get_psi4_header(self, return_warnings=False, **other_kw_dict):
        """write Psi4 input file
        other_kw_dict is a dictionary with file positions (using PSI4_*)
        corresponding to options/keywords
        returns file content and warnings e.g. if a certain feature is not available in Psi4"""

        if self.job_types is not None:
            for job in self.job_types:
                job_dict = job.get_psi4()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)

        warnings = []

        #get basis info if functional is not semi empirical
        if not self.functional.is_semiempirical:
            basis_info = self.basis.get_psi4_basis_info('sapt' in self.functional.get_psi4()[0].lower())
            if self.geometry is not None:
                struc_elements = set([atom.element for atom in self.geometry.atoms])

                warning = self.basis.check_for_elements(struc_elements)
                if warning is not None:
                    warnings.append(warning)

            #aux basis sets might have a '%s' b/c the keyword to apply them depends on
            #the functional - replace %s with the appropriate thing for the functional
            for key in basis_info:
                for i in range(0, len(basis_info[key])):
                    if "%s" in basis_info[key][i]:
                        if 'cc' in self.functional.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace("%s", "cc")

                        elif 'dct' in self.functional.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace("%s", "dct")

                        elif 'mp2' in self.functional.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace("%s", "mp2")

                        elif 'sapt' in self.functional.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace("%s", "sapt")

                        elif 'scf' in self.functional.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace("%s", "scf")

                        elif 'ci' in self.functional.name.lower():
                            basis_info[key][i] = basis_info[key][i].replace("%s", "mcscf")

        else:
            basis_info = {}

        combined_dict = combine_dicts(other_kw_dict, basis_info)

        #start building input file header
        s = ""

        #comment
        if PSI4_COMMENT in combined_dict:
            for comment in combined_dict[PSI4_COMMENT]:
                for line in comment.split('\n'):
                    s += "#%s\n" % line

        #procs
        if self.processors is not None:
            s += "set_num_threads(%i)\n" % self.processors

        #mem
        if self.memory is not None:
            s += "memory %i GB\n" % self.memory

        #before geometry options e.g. basis {} or building a dft functional
        if PSI4_BEFORE_GEOM in combined_dict:
            if len(combined_dict[PSI4_BEFORE_GEOM]) > 0:
                for opt in combined_dict[PSI4_BEFORE_GEOM]:
                    s += opt
                    s += '\n'
                s += '\n'

        
        s += "molecule {\n"
        s += "%2i %i\n" % (self.charge, self.multiplicity)
        if PSI4_COORDINATES in combined_dict:
            for kw in combined_dict[self.PSI4_COORDINATES]:
                if len(combined_dict[self.PSI4_COORDINATES][kw]) > 0:
                    opt = combined_dict[self.PSI4_COORDINATES][kw][0]
                    if 'pubchem' in kw.lower() and not kw.strip().endswith(':'):
                        kw = kw.strip() + ':'
                    s += "%s %s\n" % (kw.strip(), opt)
                
                else:
                    s += "%s\n" % kw

        if return_warnings:
            return s, warnings
        else:
            return s

    def get_psi4_footer(self, return_warnings=False, **other_kw_dict):
        """get psi4 footer"""
        
        if self.job_types is not None:
            for job in self.job_types:
                job_dict = job.get_psi4()
                other_kw_dict = combine_dicts(other_kw_dict, job_dict)

        s = "}\n\n"
        warnings = []

        #grid
        if self.grid is not None:
            grid_info, warning = self.grid.get_psi4()
            if warning is not None:
                warnings.append(warning)
            other_kw_dict = combine_dicts(other_kw_dict, grid_info)
        
        #settings
        #a setting will only get added if its list has at least one item, but only the first item will be used
        if PSI4_SETTINGS in other_kw_dict and any(len(other_kw_dict[PSI4_SETTINGS][setting]) > 0 for setting in other_kw_dict[PSI4_SETTINGS]):
            s += "set {\n"
            for setting in other_kw_dict[PSI4_SETTINGS]:
                if len(other_kw_dict[PSI4_SETTINGS][setting]) > 0:
                    s += "    %-20s    %s\n" % (setting, other_kw_dict[PSI4_SETTINGS][setting][0])

            s += "}\n\n"
        
        if PSI4_OPTKING in other_kw_dict and any(len(other_kw_dict[PSI4_OPTKING][setting]) > 0 for setting in other_kw_dict[PSI4_OPTKING]):
            s += "set optking {\n"
            for setting in other_kw_dict[PSI4_OPTKING]:
                if len(other_kw_dict[PSI4_OPTKING][setting]) > 0:
                    s += "    %-20s    %s\n" % (setting, other_kw_dict[PSI4_OPTKING][setting][0])

            s += "}\n\n"

        #functional is functional name + dispersion if there is dispersion
        functional = self.functional.get_psi4()[0]
        if self.empirical_dispersion is not None:
            functional += self.empirical_dispersion.get_psi4()[0]

        #after job stuff - replace $FUNCTIONAL with functional
        if PSI4_BEFORE_JOB in other_kw_dict:
            for opt in other_kw_dict[PSI4_BEFORE_JOB]:
                if "$FUNCTIONAL" in opt:
                    opt = opt.replace("$FUNCTIONAL", "'%s'" % functional)

                s += opt
                s += '\n'
        
        #for each job, start with nrg = f('functional'
        #unless return_wfn=True, then do nrg, wfn = f('functional'
        if PSI4_JOB in other_kw_dict:
            for func in sorted(other_kw_dict[PSI4_JOB].keys(), key=probable_job_order):
                if any(['return_wfn' in kwarg and ('True' in kwarg or 'on' in kwarg) \
                        for kwarg in other_kw_dict[PSI4_JOB][func]]):
                    s += "nrg, wfn = %s('%s'" % (func, functional)
                else:
                    s += "nrg = %s('%s'" % (func, functional)
                
                known_kw = []
                for kw in other_kw_dict[PSI4_JOB][func]:
                    key = kw.split('=')[0].strip()
                    if key not in known_kw:
                        known_kw.append(key)
                        s += ", "
                        s += kw.replace("$FUNCTIONAL", "'%s'" % functional)
                
                s += ")\n"

        #after job stuff - replace $FUNCTIONAL with functional
        if PSI4_AFTER_JOB in other_kw_dict:
            for opt in other_kw_dict[PSI4_AFTER_JOB]:
                if "$FUNCTIONAL" in opt:
                    opt = opt.replace("$FUNCTIONAL", "'%s'" % functional)

                s += opt
                s += '\n'

        if return_warnings:
            return s, warnings
        else:
            return s


