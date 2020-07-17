"""placeholder stuff until Aaron.Theory is moved"""

import os

from AaronTools.geometry import Geometry

KNOWN_SEMI_EMPIRICAL = ["AM1", "PM3", "PM6", "PM7", "HF-3c"]


class Theory:
    """a Theory object can be used to create an input file for different QM software
    The creation of a Theory object does not depend on the specific QM software - that is determined when the file is written
    valid initialization key words are:
    structure               -   AaronTools Geometry 
    charge                  -   total charge
    multiplicity            -   electronic multiplicity
    
    functional              -   Functional object
    basis                   -   BasisSet object
    constraints             -   dictionary of bond, angle, and torsional constraints (keys: atoms, bonds, angles, torsions)
                                    items are:
                                        atoms: [Atom]
                                        bonds: [[Atom, Atom]]
                                        angles: [[Atom, Atom, Atom]]
                                        torsions: [[Atom, Atom, Atom, Atom]]
                                    all keys must be present for footers (except atoms if writing gaussian)
    empirical_dispersion    -   EmpiricalDispersion object
    grid                    -   IntegrationGrid object
    
    memory                  -   allocated memory (GB)
    processors              -   allocated cores

    methods that construct headers and footers require 'other_kw_dict'
    keys of other_kw_dict are ORCA_*, PSI4_*, or GAUSSIAN_*
    ORCA_ROUTE: list(str)
    ORCA_BLOCKS: dict(list(str)) - keys are block names minus %
    ORCA_COORDINATES: ignored
    ORCA_COMMENT: list(str)

    PSI4_SETTINGS: dict(setting_name: [value])
    PSI4_BEFORE_GEOM: list(str)
    PSI4_AFTER_GEOM: list(str) - $FUNCTIONAL will be replaced with functional name
    PSI4_COMMENT: list(str)
    PSI4_COORDINATES: dict(str:list(str)) e.g. {'symmetry': ['c1']}
    PSI4_JOB: dict(optimize/frequencies/etc: list(str - $FUNCTIONAL replaced w/ functional))
    
    GAUSSIAN_PRE_ROUTE: dict(list(str)) - keys are link0 minus %
    GAUSSIAN_ROUTE: dict(list(str)) - e.g. {'opt': ['NoEigenTest', 'Tight']}
    GAUSSIAN_COORDINATES: ignored
    GAUSSIAN_CONSTRAINTS: ignored
    GAUSSIAN_GEN_BASIS: list(str)
    GAUSSIAN_GEN_ECP: list(str)
    GAUSSIAN_POST: list(str)
    GAUSSIAN_COMMENT: list(str)
"""
    
    ORCA_ROUTE = 1 # simple input
    ORCA_BLOCKS = 2 #blocks
    ORCA_COORDINATES = 3 #molecule (not used)
    ORCA_COMMENT = 4 #comments
    
    PSI4_SETTINGS = 1 # set { stuff }
    PSI4_BEFORE_GEOM = 2 #before geometry - basis info goes here
    PSI4_AFTER_GEOM = 3 #after geometry
    PSI4_COMMENT = 4 #comments
    PSI4_COORDINATES = 5 #molecule - used for symmetry etc.
    PSI4_JOB = 6 #energy, optimize, etc
    
    GAUSSIAN_PRE_ROUTE = 1 #can be used for things like %chk=some.chk
    GAUSSIAN_ROUTE = 2 #route specifies most options, e.g. #n B3LYP/3-21G opt 
    GAUSSIAN_COORDINATES = 3 #coordinate section
    GAUSSIAN_CONSTRAINTS = 4 #constraints section (e.g. B 1 2 F)
    GAUSSIAN_GEN_BASIS = 5 #gen or genecp basis section
    GAUSSIAN_GEN_ECP = 6 #genecp ECP section
    GAUSSIAN_POST = 7 #after everything else (e.g. NBO options)
    GAUSSIAN_COMMENT = 8 #comment line after the route

    ACCEPTED_INIT_KW = ['functional', \
                        'basis', \
                        'structure', \
                        'constraints', \
                        'memory', \
                        'processors', \
                        'empirical_dispersion', \
                        'charge', \
                        'multiplicity', \
                        'grid']

    def __init__(self, **kw):
        for key in self.ACCEPTED_INIT_KW:
            if key in kw:
                #print("%s in kw" % key)
                self.__setattr__(key, kw[key])
            else:
                #print("%s not in kw" % key)
                self.__setattr__(key, None)

    def make_header(self, geom, step, form='gaussian', other_kw_dict={}, **kwargs):
        """geom: Geometry
        step: float
        form: str, gaussian, orca, or psi4
        other_kw_dict: dict, keys are ORCA_*, PSI4_*, or GAUSSIAN_*"""

        self.structure = geom

        if form == "gaussian":
            other_kw_dict[self.GAUSSIAN_COMMENT] = ["step %.1f" % step]
            return self.get_gaussian_header(other_kw_dict)
        
        elif form == "orca":
            other_kw_dict[self.ORCA_COMMENT] = ["step %.1f" % step]
            return self.get_orca_header(other_kw_dict)
        
        elif form == "psi4":
            other_kw_dict[self.PSI4_COMMENT] = ["step %.1f" % step]
            return self.get_psi4_header(other_kw_dict)
    
    def make_footer(self, geom, step, form='gaussian', other_kw_dict={}):
        """geom: Geometry
        step: float (ignored)
        form: str, gaussian or psi4
        other_kw_dict: dict, keys are GAUSSIAN_*, ORCA_*, or PSI4_*
        """
        if form == "gaussian":
            return self.get_gaussian_footer(other_kw_dict)

        elif form == "psi4":
            return self.get_psi4_footer(other_kw_dict)

    def get_gaussian_header(self, other_kw_dict, return_warnings=False):
        """write Gaussian09/16 input file
        other_kw_dict is a dictionary with file positions (using GAUSSIAN_* int map)
        corresponding to options/keywords
        returns warnings if a certain feature is not available in Gaussian"""

        warnings = []
        s = ""

        if self.processors is not None:
            s += "%%NProcShared=%i\n" % self.processors

        if self.memory is not None:
            s += "%%Mem=%iGB\n" % self.memory

        if self.GAUSSIAN_PRE_ROUTE in other_kw_dict:
            for key in other_kw_dict[self.GAUSSIAN_PRE_ROUTE]:
                s += "%%%s" % key
                if len(other_kw_dict[self.GAUSSIAN_PRE_ROUTE][key]) > 0:
                    s += "=%s" % ",".join(other_kw_dict[self.GAUSSIAN_PRE_ROUTE][key])

                if not s.endswith('\n'):
                    s += '\n'

        #start route line with functional
        func, warning = self.functional.get_gaussian()
        if warning is not None:
            warnings.append(warning)

        s += "#n %s" % func
        if not self.functional.is_semiempirical:
            basis_info = self.basis.get_gaussian_basis_info()
            basis_elements = self.basis.elements_in_basis
            #check if any element is in multiple basis sets
            for element in basis_elements:
                if basis_elements.count(element) > 1:
                    warnings.append("%s is in basis set multiple times" % element)

            #check to make sure all elements have a basis set
            if self.structure is not None:
                struc_elements = set([atom.element for atom in self.structure.atoms])

                elements_wo_basis = []
                for ele in struc_elements:
                    if ele not in basis_elements:
                        elements_wo_basis.append(ele)

                if len(elements_wo_basis) > 0:
                    warnings.append("no basis set for %s" % ", ".join(elements_wo_basis))

            if self.GAUSSIAN_ROUTE in basis_info:
                s += "%s" % basis_info[self.GAUSSIAN_ROUTE]

        s += " "

        if self.empirical_dispersion is not None:
            disp, warning = self.empirical_dispersion.get_gaussian()
            other_kw_dict = combine_dicts(other_kw_dict, disp)
            if warning is not None:
                warnings.append(warning)

        if self.grid is not None:
            grid, warning = self.grid.get_gaussian()
            other_kw_dict = combine_dicts(other_kw_dict, grid)
            if warning is not None:
                warnings.append(warning)

        if self.GAUSSIAN_ROUTE in other_kw_dict:
            for option in other_kw_dict[self.GAUSSIAN_ROUTE]:
                known_opts = []
                s += option
                if len(other_kw_dict[self.GAUSSIAN_ROUTE][option]) > 1 or \
                   (len(other_kw_dict[self.GAUSSIAN_ROUTE][option]) == 1 and \
                   ('=' in other_kw_dict[self.GAUSSIAN_ROUTE][option][0] or \
                    '(' in other_kw_dict[self.GAUSSIAN_ROUTE][option][0])):
                    opts = [opt.split('=')[0] for opt in other_kw_dict[self.GAUSSIAN_ROUTE][option]]

                    s += "=("
                    for x in other_kw_dict[self.GAUSSIAN_ROUTE][option]:
                        opt = x.split('=')[0]
                        if opt not in known_opts:
                            if len(known_opts) > 0:
                                s += ','
                            known_opts.append(opt)
                            s += x
                    s += ")"
                
                elif len(other_kw_dict[self.GAUSSIAN_ROUTE][option]) == 1:
                    s += "=%s" % other_kw_dict[self.GAUSSIAN_ROUTE][option][0]

                s += " "

        s += "\n\n"

        if self.GAUSSIAN_COMMENT in other_kw_dict:
            if len(other_kw_dict[self.GAUSSIAN_COMMENT]) > 0:
                s += "\n".join([x.rstrip() for x in other_kw_dict[self.GAUSSIAN_COMMENT]])
            else:
                s += "comment"

            if not s.endswith('\n'):
                s += '\n'

        else:
            if self.comment is None:
                s += "comment\n"
            else:
                s += "%s\n" % self.comment

        s += "\n"

        s += "%i %i\n" % (self.charge, self.multiplicity)

        if return_warnings:
            return s, warnings
        else:
            return s


    def get_gaussian_footer(self, other_kw_dict, return_warnings=False):
        """write footer of gaussian input file"""
        s = ""
        warnings = []

        if not self.functional.is_semiempirical:
            basis_info = self.basis.get_gaussian_basis_info()
            basis_elements = self.basis.elements_in_basis
            #check if any element is in multiple basis sets
            for element in basis_elements:
                if basis_elements.count(element) > 1:
                    warnings.append("%s is in basis set multiple times" % element)

            #check to make sure all elements have a basis set
            if self.structure is not None:
                struc_elements = set([atom.element for atom in self.structure.atoms])

                elements_wo_basis = []
                for ele in struc_elements:
                    if ele not in basis_elements:
                        elements_wo_basis.append(ele)

                if len(elements_wo_basis) > 0:
                    warnings.append("no basis set for %s" % ", ".join(elements_wo_basis))

        s += "\n"

        if self.constraints is not None and self.structure is not None:
            for constraint in self.constraints['bonds']:
                atom1, atom2 = constraint
                ndx1 = self.structure.atoms.index(atom1) + 1
                ndx2 = self.structure.atoms.index(atom2) + 1
                s += "B %2i %2i F\n" % (ndx1, ndx2)

            for constraint in self.constraints['angles']:
                atom1, atom2, atom3 = constraint
                ndx1 = self.structure.atoms.index(atom1) + 1
                ndx2 = self.structure.atoms.index(atom2) + 1
                ndx3 = self.structure.atoms.index(atom3) + 1
                s += "A %2i %2i %2i F\n" % (ndx1, ndx2, ndx3)

            for constraint in self.constraints['torsions']:
                atom1, atom2, atom3, atom4 = constraint
                ndx1 = self.structure.atoms.index(atom1) + 1
                ndx2 = self.structure.atoms.index(atom2) + 1
                ndx3 = self.structure.atoms.index(atom3) + 1
                ndx4 = self.structure.atoms.index(atom4) + 1
                s += "D %2i %2i %2i %2i F\n" % (ndx1, ndx2, ndx3, ndx4)

            s += '\n'

        if not self.functional.is_semiempirical:
            if self.GAUSSIAN_GEN_BASIS in basis_info:
                s += basis_info[self.GAUSSIAN_GEN_BASIS]
            
                s += "\n"

            if self.GAUSSIAN_GEN_ECP in basis_info:
                s += basis_info[self.GAUSSIAN_GEN_ECP]
                
                s += '\n'

        if self.GAUSSIAN_POST in other_kw_dict:
            for item in other_kw_dict[self.GAUSSIAN_POST]:
                s += item
                s += " "

            s += '\n'

        s += '\n\n'

        if return_warnings:
            return s, warnings
        else:
            return s

    def get_orca_header(self, other_kw_dict, return_warnings=False):
        """get ORCA input file header
        other_kw_dict is a dictionary with file positions (using ORCA_* int map)
        corresponding to options/keywords
        returns file content and warnings e.g. if a certain feature is not available in ORCA
        returns str of header content
        if return_warnings, returns str, list(warning)"""
        
        warnings = []

        if not self.functional.is_semiempirical:
            basis_info = self.basis.get_orca_basis_info()
            if self.structure is not None:
                struc_elements = set([atom.element for atom in self.structure.atoms])
            
                warning = self.basis.check_for_elements(struc_elements)
                if warning is not None:
                    warnings.append(warning)

        else:
            basis_info = {}

        combined_dict = combine_dicts(other_kw_dict, basis_info)

        if self.grid is not None:
            grid_info, warning = self.grid.get_orca()
            if warning is not None:
                warnings.append(warning)

            if any('finalgrid' in x.lower() for x in combined_dict[self.ORCA_ROUTE]):
                grid_info[self.ORCA_ROUTE].pop(1)

            combined_dict = combine_dicts(combined_dict, grid_info)

        if self.constraints is not None and self.structure is not None:
            if 'geom' not in combined_dict[self.ORCA_BLOCKS]:
                combined_dict[self.ORCA_BLOCKS]['geom'] = []

            combined_dict[self.ORCA_BLOCKS]['geom'].append("constraints")
            for constraint in self.constraints['atoms']:
                atom1 = constraint
                ndx1 = self.structure.atoms.index(atom1)
                s = "    {C %2i C}" % (ndx1)
                combined_dict[self.ORCA_BLOCKS]['geom'].append(s)

            for constraint in self.constraints['bonds']:
                atom1, atom2 = constraint
                ndx1 = self.structure.atoms.index(atom1)
                ndx2 = self.structure.atoms.index(atom2)
                s = "    {B %2i %2i C}" % (ndx1, ndx2)
                combined_dict[self.ORCA_BLOCKS]['geom'].append(s)

            for constraint in self.constraints['angles']:
                atom1, atom2, atom3 = constraint
                ndx1 = self.structure.atoms.index(atom1)
                ndx2 = self.structure.atoms.index(atom2)
                ndx3 = self.structure.atoms.index(atom3)
                s = "    {A %2i %2i %2i C}" % (ndx1, ndx2, ndx3)
                combined_dict[self.ORCA_BLOCKS]['geom'].append(s)

            for constraint in self.constraints['torsions']:
                atom1, atom2, atom3, atom4 = constraint
                ndx1 = self.structure.atoms.index(atom1)
                ndx2 = self.structure.atoms.index(atom2)
                ndx3 = self.structure.atoms.index(atom3)
                ndx4 = self.structure.atoms.index(atom4)
                s = "    {D %2i %2i %2i %2i C}" % (ndx1, ndx2, ndx3, ndx4)
                combined_dict[self.ORCA_BLOCKS]['geom'].append(s)

            combined_dict[self.ORCA_BLOCKS]['geom'].append("end")

        s = ""

        if self.ORCA_COMMENT in combined_dict:
            for comment in combined_dict[self.ORCA_COMMENT]:
                for line in comment.split('\n'):
                    s += "#%s\n" % line

        s += "!"
        if self.functional is not None:
            func, warning = self.functional.get_orca()
            if warning is not None:
                warnings.append(warning)
            s += " %s" % func

        if self.empirical_dispersion is not None:
            if not s.endswith(' '):
                s += " "

            dispersion, warning = self.empirical_dispersion.get_orca()
            if warning is not None:
                warnings.append(warning)

            s += "%s" % dispersion

        if self.ORCA_ROUTE in combined_dict:
            if not s.endswith(' '):
                s += " "
                
                s += " ".join(combined_dict[self.ORCA_ROUTE])

        s += "\n"

        if self.processors is not None:
            s += "%%pal\n    nprocs %i\nend\n" % self.processors

            if self.memory is not None:
                s += "%%MaxCore %i\n" % (int(1000 * self.memory / self.processors))

        if self.ORCA_BLOCKS in combined_dict:
            for kw in combined_dict[self.ORCA_BLOCKS]:
                if any(len(x) > 0 for x in combined_dict[self.ORCA_BLOCKS][kw]):
                    s += "%%%s\n" % kw
                    for opt in combined_dict[self.ORCA_BLOCKS][kw]:
                        s += "    %s\n" % opt
                    s += "end\n"

            s += "\n"

        s += "*xyz %i %i\n" % (self.charge, self.multiplicity)
        
        if return_warnings:
            return s, warnings
        else:
            return s

    def get_psi4_header(self, other_kw_dict, fname=None):
        """write Psi4 input file
        other_kw_dict is a dictionary with file positions (using PSI4_* int map)
        corresponding to options/keywords
        returns file content and warnings e.g. if a certain feature is not available in Psi4"""

        warnings = []

        if not self.functional.is_semiempirical:
            basis_info = self.basis.get_psi4_basis_info('sapt' in self.functional.get_psi4()[0].lower())
            if self.structure is not None:
                struc_elements = set([atom.element for atom in self.structure.atoms])

                warning = self.basis.check_for_elements(struc_elements)
                if warning is not None:
                    warnings.append(warning)

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
        if self.grid is not None:
            grid_info, warning = self.grid.get_psi4()
            if warning is not None:
                warnings.append(warning)
            combined_dict = combine_dicts(combined_dict, grid_info)

        s = ""

        if self.PSI4_COMMENT in combined_dict:
            for comment in combined_dict[self.PSI4_COMMENT]:
                for line in comment.split('\n'):
                    s += "#%s\n" % line

        if self.processors is not None:
            s += "set_num_threads(%i)\n" % self.processors

        if self.memory is not None:
            s += "memory %i GB\n" % self.memory

        if self.PSI4_BEFORE_GEOM in combined_dict:
            if len(combined_dict[self.PSI4_BEFORE_GEOM]) > 0:
                for opt in combined_dict[self.PSI4_BEFORE_GEOM]:
                    s += opt
                    s += '\n'
                s += '\n'


        if return_warnings:
            return s, warnings
        else:
            return s

    def get_psi4_footer(self, other_kw_dict, return_warnings=False):
        """get psi4 footer"""
        s = ""
        warnings = []

        if self.PSI4_SETTINGS in other_kw_dict and any(len(other_kw_dict[self.PSI4_SETTINGS][setting]) > 0 for setting in other_kw_dict[self.PSI4_SETTINGS]):
            s += "set {\n"
            for setting in other_kw_dict[self.PSI4_SETTINGS]:
                if len(other_kw_dict[self.PSI4_SETTINGS][setting]) > 0:
                    s += "    %-20s    %s\n" % (setting, other_kw_dict[self.PSI4_SETTINGS][setting][0])

            s += "}\n\n"

        if self.constraints is not None:
            if len(self.constraints['atoms']) > 0 and self.structure is not None:
                s += "freeze_list = \"\"\"\n"
                for atom in self.constraints['atoms']:
                    s += "    %2i xyz\n" % (self.structure.atoms.index(atom) + 1)

                s += "\"\"\"\n"
                s += "    \n"

            s += "set optking {\n"

            if len(self.constraints['atoms']) > 0 and self.structure is not None:
                s += "    frozen_cartesian $freeze_list\n"

            if len(self.constraints['bonds']) > 0 and self.structure is not None:
                s += "    frozen_distance = (\"\n"
                for bond in self.constraints['bonds']:
                    atom1, atom2 = bond
                    s += "        %2i %2i\n" % (self.structure.atoms.index(atom1) + 1, self.structure.atoms.index(atom2) + 1)

                s += "    \")\n"

            if len(self.constraints['angles']) > 0 and self.structure is not None:
                s += "    frozen_bend = (\"\n"
                for angle in self.constraints['angles']:
                    atom1, atom2, atom3 = angle
                    s += "        %2i %2i %2i\n" % (self.structure.atoms.index(atom1) + 1, self.structure.atoms.index(atom2) + 1, self.structure.atoms.index(atom3) + 1)

                s += "    \")\n"

            if len(self.constraints['torsions']) > 0 and self.structure is not None:
                s += "    frozen_dihedral = (\"\n"
                for torsion in self.constraints['torsions']:
                    atom1, atom2, atom3, atom4 = torsion
                    s += "        %2i %2i %2i %2i\n" % (self.structure.atoms.index(atom1) + 1, \
                                                        self.structure.atoms.index(atom2) + 1, \
                                                        self.structure.atoms.index(atom3) + 1, \
                                                        self.structure.atoms.index(atom4) + 1)

                s += "    \")\n"

            s += "}\n\n"

        functional = self.functional.get_psi4()[0]
        if self.empirical_dispersion is not None:
            functional += self.empirical_dispersion.get_psi4()[0]

        if self.PSI4_JOB in other_kw_dict:
            for func in other_kw_dict[self.PSI4_JOB]:
                if any(['return_wfn' in kwarg and ('True' in kwarg or 'on' in kwarg) \
                        for kwarg in other_kw_dict[self.PSI4_JOB][func]]):
                    s += "nrg, wfn = %s('%s'" % (func, functional)
                else:
                    s += "nrg = %s('%s'" % (func, functional)
                
                known_kw = []
                for kw in other_kw_dict[self.PSI4_JOB][func]:
                    key = kw.split('=')[0].strip()
                    if key not in known_kw:
                        known_kw.append(key)
                        s += ", "
                        s += kw.replace("$FUNCTIONAL", "'%s'" % functional)
                
                s += ")\n"

        if self.PSI4_AFTER_GEOM in other_kw_dict:
            for opt in other_kw_dict[self.PSI4_AFTER_GEOM]:
                if "$FUNCTIONAL" in opt:
                    opt = opt.replace("$FUNCTIONAL", "'%s'" % functional)

                s += opt
                s += '\n'

        if return_warnings:
            return s, warnings
        else:
            return s


class Functional:
    """functional object
    used to ensure the proper keyword is used
    e.g.
    using Functional('PBE0') will use PBE1PBE in a gaussian input file"""
    def __init__(self, name, is_semiempirical):
        """name: str, functional name
        is_semiemperical: bool, basis set is not required"""
        self.name = name
        self.is_semiempirical = is_semiempirical

    def get_gaussian(self):
        """maps proper functional name to one Gaussian accepts"""
        if self.name == "ωB97X-D":
            return ("wB97XD", None)
        elif self.name == "Gaussian's B3LYP":
            return ("B3LYP", None)
        elif self.name == "B97X-D":
            return ("B97D", None)
        elif self.name.startswith("M06-"):
            return (self.name.replace("M06-", "M06", 1), None)
        
        elif self.name == "PBE0":
            return ("PBE1PBE", None)
        
        #methods available in ORCA but not Gaussian
        elif self.name == "𝜔ωB97X-D3":
            return ("wB97XD", "ωB97X-D3 is not available in Gaussian, switching to ωB97X-D2")
        elif self.name == "B3LYP":
            return ("B3LYP", "Gaussian's B3LYP uses a different LDA")
        
        else:
            return self.name.replace('ω', 'w'), None

    def get_orca(self):
        """maps proper functional name to one ORCA accepts"""
        if self.name == "ωB97X-D":
            return ("wB97X-D3", "ωB97X-D may refer to ωB97X-D2 or ωB97X-D3 - using the latter")
        elif self.name == "ωB97X-D3":
            return ("wB97X-D3", None)
        elif self.name == "B97-D":
            return ("B97-D2", "B97-D may refer to B97-D2 or B97-D3 - using the former")
        elif self.name == "Gaussian's B3LYP":
            return ("B3LYP/G", None)
        elif self.name == "M06-L":
            #why does M06-2X get a hyphen but not M06-L? 
            return ("M06L", None)
        
        else:
            return self.name.replace('ω', 'w'), None
    
    def get_psi4(self):
        """maps proper functional name to one Psi4 accepts"""
        return self.name.replace('ω', 'w'), None


class BasisSet:
    """used to more easily get basis set info for writing input files"""
    ORCA_AUX = ["C", "J", "JK", "CABS", "OptRI CABS"]
    PSI4_AUX = ["JK", "RI"]

    def __init__(self, basis, ecp=None):
        """basis: list(Basis) or None
        ecp: list(ECP) or None"""
        self.basis = basis
        self.ecp = ecp

    @property
    def elements_in_basis(self):
        elements = []
        if self.basis is not None:
            for basis in self.basis:
                elements.extend(basis.elements)
            
        return elements
    
    def get_gaussian_basis_info(self):
        info = {}

        if self.basis is not None:
            if all([basis == self.basis[0] for basis in self.basis]) and not self.basis[0].user_defined and self.ecp is None:
                info[Theory.GAUSSIAN_ROUTE] = "/%s" % Basis.map_gaussian09_basis(self.basis[0].name)
            else:
                if self.ecp is None:
                    info[Theory.GAUSSIAN_ROUTE] = "/gen"
                else:
                    info[Theory.GAUSSIAN_ROUTE] = "/genecp"
                    
                s = ""
                for basis in self.basis:
                    if len(basis.elements) > 0 and not basis.user_defined:
                        s += " ".join([ele for ele in basis.elements])
                        s += " 0\n"
                        s += Basis.map_gaussian09_basis(basis.get_basis_name())
                        s += "\n****\n"
                    
                for basis in self.basis:
                    if len(basis.elements) > 0:
                        if basis.user_defined:
                            if os.path.exists(basis.user_defined):
                                with open(basis.user_defined, "r") as f:
                                    lines = f.readlines()
                                
                                i = 0
                                while i < len(lines):
                                    test = lines[i].strip()
                                    if len(test) == 0 or test.startswith('!'):
                                        i += 1
                                        continue
                                    
                                    ele = test.split()[0]
                                    while i < len(lines):
                                        if ele in basis.elements:
                                            s += lines[i]
                                        
                                        if lines[i].startswith('****'):
                                            break

                                        i += 1
                                    
                                    i += 1

                            else:
                                s += "@%s\n" % basis.user_defined
                        
                    info[Theory.GAUSSIAN_GEN_BASIS] = s
                    
        if self.ecp is not None:
            s = ""
            for basis in self.ecp:
                if len(basis.elements) > 0 and not basis.user_defined:
                    s += " ".join([ele for ele in basis.elements])
                    s += " 0\n"
                    s += Basis.map_gaussian09_basis(basis.get_basis_name())
                    s += "\n"
                
            for basis in self.ecp:
                if len(basis.elements) > 0:
                    if basis.user_defined:
                        if os.path.exists(basis.user_defined):
                            with open(basis.user_defined, "r") as f:
                                lines = f.readlines()
                            
                                i = 0
                                while i < len(lines):
                                    test = lines[i].strip()
                                    if len(test) == 0 or test.startswith('!'):
                                        i += 1
                                        continue
                                    
                                    ele = test.split()[0]
                                    while i < len(lines):
                                        if ele in basis.elements:
                                            s += lines[i]
                                        
                                        if lines[i].startswith('****'):
                                            break

                                        i += 1
                                    
                                    i += 1

                        else:
                            s += "@%s\n" % basis.user_defined
                            
            info[Theory.GAUSSIAN_GEN_ECP] = s
            
            if self.basis is None:
                info[Theory.GAUSSIAN_ROUTE] = " Pseudo=Read"
            
        return info    
    
    def get_orca_basis_info(self):
        #TODO: warn if basis should be f12
        info = {Theory.ORCA_BLOCKS:{'basis':[]}, Theory.ORCA_ROUTE:[]}

        first_basis = []

        if self.basis is not None:
            for basis in self.basis:
                if len(basis.elements) > 0:
                    if basis.aux_type is None:
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = Basis.map_orca_basis(basis.get_basis_name())
                                info[Theory.ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                                
                            else:
                                s = "GTOName \"%s\"" % basis.user_defined
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                            
                        else:
                            for ele in basis.elements:
                                s = "newGTO            %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                        
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "C":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s/C" % Basis.map_orca_basis(basis.get_basis_name())
                                info[Theory.ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                        
                            else:
                                s = "AuxCGTOName \"%s\"" % basis.user_defined
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                                                    
                        else:
                            for ele in basis.elements:
                                s = "newAuxCGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s/C\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                            
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                    
                    elif basis.aux_type == "J":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s/J" % Basis.map_orca_basis(basis.get_basis_name())
                                info[Theory.ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                            
                            else:
                                s = "AuxJGTOName \"%s\"" % basis.user_defined
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                s = "newAuxJGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s/J\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined

                                info[Theory.ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "JK":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s/JK" % Basis.map_orca_basis(basis.get_basis_name())
                                info[Theory.ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                            
                            else:
                                s = "AuxJKGTOName \"%s\"" % basis.user_defined
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                        
                        else:
                            for ele in basis.elements:
                                s = "newAuxJKGTO       %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s/JK\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s-CABS" % Basis.map_orca_basis(basis.get_basis_name())
                                info[Theory.ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                            
                            else:
                                s = "CABSGTOName \"%s\"" % basis.user_defined
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                s = "newCABSGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s-CABS\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "OptRI CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s-OptRI" % Basis.map_orca_basis(basis.get_basis_name())
                                info[Theory.ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                                
                            else:
                                s = "CABSGTOName \"%s\"" % basis.user_defined
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                        
                        else:
                            for ele in basis.elements:
                                s = "newCABSGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s-OptRI\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                
                                info[Theory.ORCA_BLOCKS]['basis'].append(s)

        if self.ecp is not None:
            for basis in self.ecp:
                if len(basis.elements) > 0 and not basis.user_defined:
                    for ele in basis.elements:
                        s = "newECP            %-2s " % ele
                        s += "\"%s\" end" % Basis.map_orca_basis(basis.get_basis_name())
                    
                        info[Theory.ORCA_BLOCKS]['basis'].append(s)
 
                elif len(basis.elements) > 0 and basis.user_defined:
                    #TODO: check if this works
                    s = "GTOName \"%s\"" % basis.user_defined                            
            
                    info[Theory.ORCA_BLOCKS]['basis'].append(s)
            
        return info

    def get_psi4_basis_info(self, sapt=False):
        """sapt: bool, use df_basis_sapt instead of df_basis_scf for jk basis"""
        s = ""
        s2 = None
        s3 = None
        s4 = None

        first_basis = []

        if self.basis is not None:
            for basis in self.basis:
                if basis.aux_type not in first_basis:
                    first_basis.append(basis.aux_type)
                    if basis.aux_type is None or basis.user_defined:
                        s += "basis {\n"
                        s += "    assign    %s\n" % Basis.map_psi4_basis(basis.get_basis_name())
                        
                    elif basis.aux_type == "JK":
                        if sapt:
                            s4 = "df_basis_sapt {\n"
                        else:
                            s4 = "df_basis_scf {\n"
                        s4 += "    assign %s-jkfit\n" % Basis.map_psi4_basis(basis.get_basis_name())
                    
                    elif basis.aux_type == "RI":
                        s2 = "df_basis_%s {\n"
                        if basis.name.lower() == "sto-3g" or basis.name.lower() == "3-21g":
                            s2 += "    assign %s-rifit\n" % Basis.map_psi4_basis(basis.get_basis_name())
                        else:
                            s2 += "    assign %s-ri\n" % Basis.map_psi4_basis(basis.get_basis_name())

                else:
                    if basis.aux_type is None or basis.user_defined:
                        for ele in basis.elements:
                            s += "    assign %2s %s\n" % (ele, Basis.map_psi4_basis(basis.get_basis_name()))
                    
                    elif basis.aux_type == "JK":
                        for ele in basis.elements:
                            s2 += "    assign %2s %s-jkfit\n" % (ele, Basis.map_psi4_basis(basis.get_basis_name()))
                            
                    elif basis.aux_type == "RI":
                        for ele in basis.elements:
                            if basis.name.lower() == "sto-3g" or basis.name.lower() == "3-21g":
                                s2 += "    assign %2s %s-rifit\n" % (ele, Basis.map_psi4_basis(basis.get_basis_name()))
                            else:
                                s2 += "    assign %2s %s-ri\n" % (ele, Basis.map_psi4_basis(basis.get_basis_name()))
                                    
            if any(basis.user_defined for basis in self.basis):
                s3 = ""
                for basis in self.basis:
                    if basis.user_defined:
                        if os.path.exists(basis.user_defined):
                            s3 += "\n[%s]\n" % basis.name
                            with open(basis.user_defined, 'r') as f:
                                lines = [line.rstrip() for line in f.readlines() if len(line.strip()) > 0 and not line.startswith('!')]
                                s3 += '\n'.join(lines)
                                s3 += '\n\n'
    
            if s3 is not None:
                s += s3
    
            if len(s) > 0:
                s += "}"
            
            if s2 is not None:
                s2 += "}"
                
                s += "\n\n%s" % s2
                    
            if s4 is not None:
                s4 += "}"
                
                s += "\n\n%s" % s4
        
        if len(s) > 0:
            info = {Theory.PSI4_BEFORE_GEOM:[s]}
        else:
            info = {}

        return info

    def check_for_elements(self, elements):
        warning = ""
        if self.basis is not None:
            elements_without_basis = {}
            for basis in self.basis:
                if basis.aux_type not in elements_without_basis:
                    elements_without_basis[basis.aux_type] = elements.copy()
                    
                for element in basis.elements:
                    if element in elements_without_basis[basis.aux_type]:
                        elements_without_basis[basis.aux_type].remove(element)
            
            if any(len(elements_without_basis[aux]) != 0 for aux in elements_without_basis.keys()):
                for aux in elements_without_basis.keys():
                    if len(elements_without_basis[aux]) != 0:
                        if aux is not None and aux != "no":
                            warning += "%s ha%s no auxiliary %s basis; " % (", ".join(elements_without_basis[aux]), "s" if len(elements_without_basis[aux]) == 1 else "ve", aux)
                        else:
                            warning += "%s ha%s no basis; " % (", ".join(elements_without_basis[aux]), "s" if len(elements_without_basis[aux]) == 1 else "ve")
                            
                return warning.strip('; ')
            
            else:
                return None


class Basis:
    def __init__(self, name, elements, aux_type=None, user_defined=False):
        """
        name         -   basis set base name (e.g. 6-31G)
        elements     -   list of element symbols the basis set applies to
        user_defined -   file containing basis info/False for builtin basis sets
        """
        self.name = name
        self.elements = elements
        self.aux_type = aux_type
        self.user_defined = user_defined

    def __repr__(self):
        return "%s(%s)" % (self.get_basis_name(), " ".join(self.elements))

    def __eq__(self, other):
        if not isinstance(other, Basis):
            return False
        
        return self.get_basis_name() == other.get_basis_name()

    def get_basis_name(self):
        """returns basis set name"""
        #was originally going to allow specifying diffuse and polarizable
        name = self.name
            
        return name

    @staticmethod
    def map_gaussian09_basis(name):
        """returns the Gaussian09/16 name of the basis set
        currently just removes the hyphen from the Karlsruhe def2 ones"""
        if name.startswith('def2-'):
            return name.replace('def2-', 'def2', 1)
        else:
            return name    

    @staticmethod
    def map_orca_basis(name):
        """returns the ORCA name of the basis set
        currently doesn't do anything"""
        return name
    
    @staticmethod
    def map_psi4_basis(name):
        """returns the Psi4 name of the basis set
        currently doesn't do anything"""
        return name


class ECP(Basis):
    """ECP - aux info will be ignored"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        
    def __eq__(self, other):
        if not isinstance(other, ECP):
            return False
            
        return super().__eq__(other) 
 
 
class EmpiricalDispersion:
    """try to keep emerpical dispersion keywords and settings consistent across file types"""
    def __init__(self, name):
        self.name = name

    def get_gaussian(self):
        """Acceptable dispersion methods for Gaussian are:
        Grimme D2
        Grimme D3
        Becke-Johnson damped Grimme D3
        Petersson-Frisch
        
        Dispersion methods available in other software that will be modified are:
        Grimme D4
        undampened Grimme D3
        
        other methods will raise RuntimeError"""
        
        if self.name == "Grimme D2":
            return ({Theory.GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD2"]}}, None)
        elif self.name == "Zero-damped Grimme D3":
            return ({Theory.GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3"]}}, None)
        elif self.name == "Becke-Johnson damped Grimme D3":
            return ({Theory.GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3BJ"]}}, None)
        elif self.name == "Petersson-Frisch":
            return ({Theory.GAUSSIAN_ROUTE:{"EmpiricalDispersion":["PFD"]}}, None)
            
        #dispersions in ORCA but not Gaussian
        elif self.name == "Grimme D4":
            return ({Theory.GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3BJ"]}}, "Grimme's D4 has no keyword in Gaussian, switching to GD3BJ")
        elif self.name == "undampened Grimme D3":
            return ({Theory.GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3"]}}, "undampened Grimme's D3 is unavailable in Gaussian, switching to GD3")
        
        #unrecognized
        else:
            raise RuntimeError("unrecognized emperical dispersion: %s" % self.name)

    def get_orca(self):
        if self.name == "Grimme D2":
            return ("D2", None)
        elif self.name == "Zero-damped Grimme D3":
            return ("D3", None)
        elif self.name == "Becke-Johnson damped Grimme D3":
            return ("D3BJ", None)
        elif self.name == "Grimme D4":
            return ("D4", None)

    def get_psi4(self):
        if self.name == "Grimme D1":
            return ("-d1", None)        
        if self.name == "Grimme D2":
            return ("-d2", None)
        elif self.name == "Zero-damped Grimme D3":
            return ("-d3", None)
        elif self.name == "Becke-Johnson damped Grimme D3":
            return ("-d3bj", None)
        elif self.name == "Becke-Johnson damped modified Grimme D3":
            return ("-d3mbj", None)
        elif self.name == "Chai & Head-Gordon":
            return ("-chg", None)
        elif self.name == "Nonlocal Approximation":
            return ("-nl", None)
        elif self.name == "Pernal, Podeszwa, Patkowski, & Szalewicz":
            return ("-das2009", None)        
        elif self.name == "Podeszwa, Katarzyna, Patkowski, & Szalewicz":
            return ("-das2010", None)        
        elif self.name == "Řezác, Greenwell, & Beran":
            return ("dmp2", None)


class ImplicitSolvent:
    """this isn't really used
    solvents are added by directly using other_kw_dict or whatever"""
    KNOWN_GAUSSIAN_SOLVENTS = ["Water", 
                               "Acetonitrile", 
                               "Methanol", 
                               "Ethanol", 
                               "IsoQuinoline", 
                               "Quinoline", 
                               "Chloroform", 
                               "DiethylEther", 
                               "DichloroMethane", 
                               "DiChloroEthane", 
                               "CarbonTetraChloride", 
                               "Benzene", 
                               "Toluene", 
                               "ChloroBenzene", 
                               "NitroMethane", 
                               "Heptane", 
                               "CycloHexane", 
                               "Aniline", 
                               "Acetone", 
                               "TetraHydroFuran", 
                               "DiMethylSulfoxide", 
                               "Argon", 
                               "Krypton", 
                               "Xenon", 
                               "n-Octanol", 
                               "1,1,1-TriChloroEthane", 
                               "1,1,2-TriChloroEthane", 
                               "1,2,4-TriMethylBenzene", 
                               "1,2-DiBromoEthane", 
                               "1,2-EthaneDiol", 
                               "1,4-Dioxane", 
                               "1-Bromo-2-MethylPropane", 
                               "1-BromoOctane", 
                               "1-BromoPentane", 
                               "1-BromoPropane", 
                               "1-Butanol", 
                               "1-ChloroHexane", 
                               "1-ChloroPentane", 
                               "1-ChloroPropane", 
                               "1-Decanol", 
                               "1-FluoroOctane", 
                               "1-Heptanol", 
                               "1-Hexanol", 
                               "1-Hexene", 
                               "1-Hexyne", 
                               "1-IodoButane", 
                               "1-IodoHexaDecane", 
                               "1-IodoPentane", 
                               "1-IodoPropane", 
                               "1-NitroPropane", 
                               "1-Nonanol", 
                               "1-Pentanol", 
                               "1-Pentene", 
                               "1-Propanol", 
                               "2,2,2-TriFluoroEthanol", 
                               "2,2,4-TriMethylPentane", 
                               "2,4-DiMethylPentane", 
                               "2,4-DiMethylPyridine", 
                               "2,6-DiMethylPyridine", 
                               "2-BromoPropane", 
                               "2-Butanol", 
                               "2-ChloroButane", 
                               "2-Heptanone", 
                               "2-Hexanone", 
                               "2-MethoxyEthanol", 
                               "2-Methyl-1-Propanol", 
                               "2-Methyl-2-Propanol", 
                               "2-MethylPentane", 
                               "2-MethylPyridine", 
                               "2-NitroPropane", 
                               "2-Octanone", 
                               "2-Pentanone", 
                               "2-Propanol", 
                               "2-Propen-1-ol", 
                               "3-MethylPyridine", 
                               "3-Pentanone", 
                               "4-Heptanone", 
                               "4-Methyl-2-Pentanone", 
                               "4-MethylPyridine", 
                               "5-Nonanone", 
                               "AceticAcid", 
                               "AcetoPhenone", 
                               "a-ChloroToluene", 
                               "Anisole", 
                               "Benzaldehyde", 
                               "BenzoNitrile", 
                               "BenzylAlcohol", 
                               "BromoBenzene", 
                               "BromoEthane", 
                               "Bromoform", 
                               "Butanal", 
                               "ButanoicAcid", 
                               "Butanone", 
                               "ButanoNitrile", 
                               "ButylAmine", 
                               "ButylEthanoate", 
                               "CarbonDiSulfide", 
                               "Cis-1,2-DiMethylCycloHexane", 
                               "Cis-Decalin", 
                               "CycloHexanone", 
                               "CycloPentane", 
                               "CycloPentanol", 
                               "CycloPentanone", 
                               "Decalin-mixture", 
                               "DiBromoMethane", 
                               "DiButylEther", 
                               "DiEthylAmine", 
                               "DiEthylSulfide", 
                               "DiIodoMethane", 
                               "DiIsoPropylEther", 
                               "DiMethylDiSulfide", 
                               "DiPhenylEther", 
                               "DiPropylAmine", 
                               "E-1,2-DiChloroEthene", 
                               "E-2-Pentene", 
                               "EthaneThiol", 
                               "EthylBenzene", 
                               "EthylEthanoate", 
                               "EthylMethanoate", 
                               "EthylPhenylEther", 
                               "FluoroBenzene", 
                               "Formamide", 
                               "FormicAcid", 
                               "HexanoicAcid", 
                               "IodoBenzene", 
                               "IodoEthane", 
                               "IodoMethane", 
                               "IsoPropylBenzene", 
                               "m-Cresol", 
                               "Mesitylene", 
                               "MethylBenzoate", 
                               "MethylButanoate", 
                               "MethylCycloHexane", 
                               "MethylEthanoate", 
                               "MethylMethanoate", 
                               "MethylPropanoate", 
                               "m-Xylene", 
                               "n-ButylBenzene", 
                               "n-Decane", 
                               "n-Dodecane", 
                               "n-Hexadecane", 
                               "n-Hexane", 
                               "NitroBenzene", 
                               "NitroEthane", 
                               "n-MethylAniline", 
                               "n-MethylFormamide-mixture", 
                               "n,n-DiMethylAcetamide", 
                               "n,n-DiMethylFormamide", 
                               "n-Nonane", 
                               "n-Octane", 
                               "n-Pentadecane", 
                               "n-Pentane", 
                               "n-Undecane", 
                               "o-ChloroToluene", 
                               "o-Cresol", 
                               "o-DiChloroBenzene", 
                               "o-NitroToluene", 
                               "o-Xylene", 
                               "Pentanal", 
                               "PentanoicAcid", 
                               "PentylAmine", 
                               "PentylEthanoate", 
                               "PerFluoroBenzene", 
                               "p-IsoPropylToluene", 
                               "Propanal", 
                               "PropanoicAcid", 
                               "PropanoNitrile", 
                               "PropylAmine", 
                               "PropylEthanoate", 
                               "p-Xylene", 
                               "Pyridine", 
                               "sec-ButylBenzene", 
                               "tert-ButylBenzene", 
                               "TetraChloroEthene", 
                               "TetraHydroThiophene-s,s-dioxide", 
                               "Tetralin", 
                               "Thiophene", 
                               "Thiophenol", 
                               "trans-Decalin", 
                               "TriButylPhosphate", 
                               "TriChloroEthene", 
                               "TriEthylAmine", 
                               "Xylene-mixture", 
                               "Z-1,2-DiChloroEthene"]

    KNOWN_ORCA_SOLVENTS = ["Water",
                           "Acetonitrile", 
                           "Acetone", 
                           "Ammonia", 
                           "Ethanol", 
                           "Methanol", 
                           "CH2Cl2", 
                           "CCl4", 
                           "DMF", 
                           "DMSO", 
                           "Pyridine", 
                           "THF", 
                           "Chloroform", 
                           "Hexane", 
                           "Benzene", 
                           "CycloHexane",
                           "Octanol", 
                           "Toluene"]

    def __init__(self, name, solvent):
        self.name = name
        self.solvent = solvent


class IntegrationGrid:
    """used to try to keep integration grid settings more easily when writing different input files"""
    def __init__(self, name):
        """name: str, gaussian keyword (e.g. SuperFineGrid), ORCA keyword (e.g. Grid7) or '(radial, angular)'
        ORCA can only use ORCA grid keywords
        Gaussian can use its keywords and will try to use ORCA keywords
        Psi4 will use '(r, a)'"""
        self.name = name

    def get_gaussian(self):
        """gets gaussian integration grid info and a warning as tuple(dict, str or None)
        dict is of the form {Theory.GAUSSIAN_ROUTE:[x]}"""
        if self.name == "UltraFine":
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=UltraFine"]}}, None)
        elif self.name == "FineGrid":
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=FineGrid"]}}, None)
        elif self.name == "SuperFineGrid":
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=SuperFineGrid"]}}, None)

        #Grids available in ORCA but not Gaussian
        #uses n_rad from K-Kr as specified in ORCA 4.2.1 manual (section 9.3)
        #XXX: there's probably IOp's that can get closer
        elif self.name == "Grid 2":
            n_rad = 45
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%i110" % n_rad]}}, "Approximating ORCA Grid 2")
        elif self.name == "Grid 3":
            n_rad = 45
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%i194" % n_rad]}}, "Approximating ORCA Grid 3")
        elif self.name == "Grid 4":
            n_rad = 45
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%i302" % n_rad]}}, "Approximating ORCA Grid 4")
        elif self.name == "Grid 5":
            n_rad = 50
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%i434" % n_rad]}}, "Approximating ORCA Grid 5")
        elif self.name == "Grid 6":
            n_rad = 55
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%i590" % n_rad]}}, "Approximating ORCA Grid 6")
        elif self.name == "Grid 7":
            n_rad = 60
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%i770" % n_rad]}}, "Approximating ORCA Grid 7")

        else:
            return ({Theory.GAUSSIAN_ROUTE:{"Integral":["grid=%s" % self.name]}}, "grid may not be available in Gaussian")

    def get_orca(self):
        """translates grid to something ORCA accepts
        current just returns self.name"""
        #TODO: allow '(r, a)'
        return ({Theory.ORCA_ROUTE:[self.name, "Final%s" % self.name]}, None)

    def get_psi4(self):
        #TODO: allow keywords
        radial, spherical = [x.strip() for x in self.name[1:-1].split(', ')]
        return ({Theory.PSI4_SETTINGS:{'dft_radial_points':[radial], 'dft_spherical_points':[spherical]}}, None)

