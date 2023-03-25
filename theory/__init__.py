"""
Theory methods that construct headers, footers, or molecule specifications
can take some keyword arguments (i.e. `\*\*kwargs`).
For these methods, you can use the parameters below. 
The value should be as specified. 

:ORCA:

* ORCA_ROUTE: list(str)
* ORCA_BLOCKS: dict(list(str)) - keys are block names minus %
* ORCA_COORDINATES: ignored
* ORCA_COMMENT: list(str)

:Psi4:

* PSI4_SETTINGS: dict(setting_name: [value])
* PSI4_BEFORE_GEOM: list(str)
* PSI4_AFTER_JOB: list(str) $METHOD will be replaced with method name
* PSI4_COMMENT: list(str)
* PSI4_MOLECULE: dict(str:list(str)) e.g. {'symmetry': ['c1']}
* PSI4_COORDINATES: dict() with keys:
    'coords' - array of coordinates with one item for each atom
    'variables' - list of name (str), value (float), is_angstrom (bool) tuples
    this is ignored if using a SAPTMethod with a low-spin combination
    of monomers
* PSI4_JOB: dict(optimize/frequencies/etc: list(str $METHOD replaced w/ method))
* PSI4_OPTKING: dict(setting_name: [value])

:Gaussian:

* GAUSSIAN_PRE_ROUTE: dict(list(str)) - keys are link0 minus %
* GAUSSIAN_ROUTE: dict(list(str)) - e.g. {'opt': ['NoEigenTest', 'Tight']}
* GAUSSIAN_COORDINATES: list of coordinates and variables/constants
* GAUSSIAN_CONSTRAINTS: list(str)
* GAUSSIAN_GEN_BASIS: list(str) - only filled by BasisSet automatically when writing footer
* GAUSSIAN_GEN_ECP: list(str) - only filled by BasisSet automatically when writing footer
* GAUSSIAN_POST: list(str)
* GAUSSIAN_COMMENT: list(str)

:SQM:

* SQM_COMMENT: list(str)
* SQM_QMMM: dict()

:Q-Chem:

* QCHEM_MOLECULE: list(str) - $molecule section
* QCHEM_REM: dict(str:str) - $rem section
* QCHEM_COMMENT: list(str) - comments
* QCHEM_SETTINGS: dict(str:str) - all other sections

:xTB:

* XTB_CONTROL_BLOCKS: dict - stuff in xcontrol file
* XTB_COMMAND_LINE: dict(str:list) - command line stuff, values should be 
    emtpy lists if the command line flag (key) has no arguments

:CREST:

* CREST_COMMAND_LINE: crest command line - see XTB_COMMAND_LINE
"""

ORCA_ROUTE = "simple" # simple input
ORCA_BLOCKS = "blocks" #blocks
ORCA_COORDINATES = 3 # molecule (not used)
ORCA_COMMENT = "comments" #comments

PSI4_SETTINGS = "settings" # set { stuff }
PSI4_BEFORE_GEOM = "before_molecule" #before geometry - basis info goes here
PSI4_AFTER_JOB = "after_job" #after job
PSI4_COMMENT = "comments" #comments
PSI4_MOLECULE = "molecule" #molecule - used for symmetry etc.
PSI4_COORDINATES = "coordinates" # coordinate variables
PSI4_JOB = "job" #energy, optimize, etc
PSI4_OPTKING = "optking" #constraints go here
PSI4_BEFORE_JOB = "before_job" #before job stuff - e.g. auto_fragments
PSI4_SOLVENT = "pcm_solver" # psi4 interface to pcmsolver

GAUSSIAN_PRE_ROUTE = "link0" #can be used for things like %chk=some.chk
GAUSSIAN_ROUTE = "route" #route specifies most options, e.g. #n B3LYP/3-21G opt
GAUSSIAN_COORDINATES = "coordinates" #coordinate section
GAUSSIAN_CONSTRAINTS = "constraints" #constraints section (e.g. B 1 2 F)
GAUSSIAN_GEN_BASIS = "gen_basis" #gen or genecp basis section
GAUSSIAN_GEN_ECP = "gen_ecp" #genecp ECP section
GAUSSIAN_POST = "end_of_file" #after everything else (e.g. NBO options)
GAUSSIAN_COMMENT = "comments" #comment line after the route
GAUSSIAN_MM = "mm" #options for mm methods
GAUSSIAN_ONIOM = "oniom" #oniom options like embedcharge
GAUSSIAN_MM_PARAMS = "parameters" #path to file with additional or total mm parameters

SQM_COMMENT = "comments"
SQM_QMMM = "qmmm"

QCHEM_MOLECULE = "molecule" # $molecule
QCHEM_REM = "rem" # $rem
QCHEM_COMMENT = "comments" # $comment
QCHEM_SETTINGS = "section" # $*

XTB_CONTROL_BLOCKS = "xcontrol" # $blocks in xcontrol
XTB_COMMAND_LINE = "command_line" # xtb command line options

CREST_COMMAND_LINE = "command_line" # crest command line options

from AaronTools.theory.basis import BasisSet, Basis, ECP
from AaronTools.theory.emp_dispersion import EmpiricalDispersion
from AaronTools.theory.implicit_solvent import ImplicitSolvent
from AaronTools.theory.grid import IntegrationGrid
from AaronTools.theory.method import Method, SAPTMethod
from AaronTools.theory.job_types import *
from AaronTools.theory.theory import Theory
