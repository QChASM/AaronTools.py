ORCA_ROUTE = "simple" # simple input
ORCA_BLOCKS = "blocks" #blocks
ORCA_COORDINATES = 3 #molecule (not used)
ORCA_COMMENT = "comments" #comments

PSI4_SETTINGS = "settings" # set { stuff }
PSI4_BEFORE_GEOM = "before_molecule" #before geometry - basis info goes here
PSI4_AFTER_JOB = "after_job" #after job
PSI4_COMMENT = "comments" #comments
PSI4_COORDINATES = "molecule" #molecule - used for symmetry etc.
PSI4_JOB = "job" #energy, optimize, etc
PSI4_OPTKING = "optking" #constraints go here
PSI4_BEFORE_JOB = "before_job" #before job stuff - e.g. auto_fragments

GAUSSIAN_PRE_ROUTE = "link0" #can be used for things like %chk=some.chk
GAUSSIAN_ROUTE = "route" #route specifies most options, e.g. #n B3LYP/3-21G opt 
GAUSSIAN_COORDINATES = 3 #coordinate section - ignored
GAUSSIAN_CONSTRAINTS = "constraints" #constraints section (e.g. B 1 2 F)
GAUSSIAN_GEN_BASIS = "gen_basis" #gen or genecp basis section
GAUSSIAN_GEN_ECP = "gen_ecp" #genecp ECP section
GAUSSIAN_POST = "end_of_file" #after everything else (e.g. NBO options)
GAUSSIAN_COMMENT = "comments" #comment line after the route


from AaronTools.theory.theory import Theory
from AaronTools.theory.basis import BasisSet, Basis, ECP
from AaronTools.theory.emp_dispersion import EmpiricalDispersion
from AaronTools.theory.implicit_solvent import ImplicitSolvent
from AaronTools.theory.grid import IntegrationGrid
from AaronTools.theory.method import Method
from AaronTools.theory.job_types import *
