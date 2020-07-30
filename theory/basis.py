import os

from AaronTools.theory import ORCA_ROUTE, ORCA_BLOCKS, \
                              \
                              PSI4_BEFORE_GEOM, \
                              \
                              GAUSSIAN_ROUTE, GAUSSIAN_GEN_BASIS, GAUSSIAN_GEN_ECP

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
        """returns dict used by get_gaussian_header/footer with basis info"""
        info = {}

        if self.basis is not None:
            if all([basis == self.basis[0] for basis in self.basis]) and not self.basis[0].user_defined and self.ecp is None:
                info[GAUSSIAN_ROUTE] = "/%s" % Basis.map_gaussian09_basis(self.basis[0].name)
            else:
                if self.ecp is None:
                    info[GAUSSIAN_ROUTE] = "/gen"
                else:
                    info[GAUSSIAN_ROUTE] = "/genecp"
                    
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
                        
                    info[GAUSSIAN_GEN_BASIS] = s
                    
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
                            
            info[GAUSSIAN_GEN_ECP] = s
            
            if self.basis is None:
                info[GAUSSIAN_ROUTE] = " Pseudo=Read"
            
        return info    
    
    def get_orca_basis_info(self):
        """return dict for get_orca_header"""
        #TODO: warn if basis should be f12
        info = {ORCA_BLOCKS:{'basis':[]}, ORCA_ROUTE:[]}

        first_basis = []

        if self.basis is not None:
            for basis in self.basis:
                if len(basis.elements) > 0:
                    if basis.aux_type is None:
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = Basis.map_orca_basis(basis.get_basis_name())
                                info[ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                                
                            else:
                                s = "GTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                            
                        else:
                            for ele in basis.elements:
                                s = "newGTO            %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                        
                                info[ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "C":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s/C" % Basis.map_orca_basis(basis.get_basis_name())
                                info[ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                        
                            else:
                                s = "AuxCGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                                                    
                        else:
                            for ele in basis.elements:
                                s = "newAuxCGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s/C\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                            
                                info[ORCA_BLOCKS]['basis'].append(s)
                    
                    elif basis.aux_type == "J":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s/J" % Basis.map_orca_basis(basis.get_basis_name())
                                info[ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                            
                            else:
                                s = "AuxJGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                s = "newAuxJGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s/J\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "JK":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s/JK" % Basis.map_orca_basis(basis.get_basis_name())
                                info[ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                            
                            else:
                                s = "AuxJKGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                        
                        else:
                            for ele in basis.elements:
                                s = "newAuxJKGTO       %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s/JK\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                
                                info[ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s-CABS" % Basis.map_orca_basis(basis.get_basis_name())
                                info[ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                            
                            else:
                                s = "CABSGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                s = "newCABSGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s-CABS\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                
                                info[ORCA_BLOCKS]['basis'].append(s)

                    elif basis.aux_type == "OptRI CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                s = "%s-OptRI" % Basis.map_orca_basis(basis.get_basis_name())
                                info[ORCA_ROUTE].append(s)
                                first_basis.append(basis.aux_type)
                                
                            else:
                                s = "CABSGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(s)
                                first_basis.append(basis.aux_type)
                        
                        else:
                            for ele in basis.elements:
                                s = "newCABSGTO        %-2s " % ele
                                
                                if not basis.user_defined:
                                    s += "\"%s-OptRI\" end" % Basis.map_orca_basis(basis.get_basis_name())
                                else:
                                    s += "\"%s\" end" % basis.user_defined
                                
                                info[ORCA_BLOCKS]['basis'].append(s)

        if self.ecp is not None:
            for basis in self.ecp:
                if len(basis.elements) > 0 and not basis.user_defined:
                    for ele in basis.elements:
                        s = "newECP            %-2s " % ele
                        s += "\"%s\" end" % Basis.map_orca_basis(basis.get_basis_name())
                    
                        info[ORCA_BLOCKS]['basis'].append(s)
 
                elif len(basis.elements) > 0 and basis.user_defined:
                    #TODO: check if this works
                    s = "GTOName \"%s\"" % basis.user_defined                            
            
                    info[ORCA_BLOCKS]['basis'].append(s)
            
        return info

    def get_psi4_basis_info(self, sapt=False):
        """sapt: bool, use df_basis_sapt instead of df_basis_scf for jk basis
        return dict for get_psi4_header"""
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
            info = {PSI4_BEFORE_GEOM:[s]}
        else:
            info = {}

        return info

    def check_for_elements(self, elements):
        """checks to make sure each element is in a basis set"""
        warning = ""
        #assume all elements aren't in a basis set, remove from the list if they have a basis
        #need to check each type of aux basis
        if self.basis is not None:
            elements_without_basis = {}
            for basis in self.basis:
                if basis.aux_type not in elements_without_basis:
                    elements_without_basis[basis.aux_type] = [str(e) for e in elements]
                    
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
        aux_type     -   str - ORCA: one of BasisSet.ORCA_AUX; Psi4: one of BasisSet.PSI4_AUX
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
        currently just adds hyphen to Karlsruhe basis if it isn't there"""
        if name.startswith('def2') and not name.startswith('def2-'):
            return name.replace('def2', 'def2-', 1)
        else:
            return name
    
    @staticmethod
    def map_psi4_basis(name):
        """returns the Psi4 name of the basis set
        currently just adds hyphen to Karlsruhe basis if it isn't there"""
        if name.startswith('def2') and not name.startswith('def2-'):
            return name.replace('def2', 'def2-', 1)
        else:
            return name


class ECP(Basis):
    """ECP - aux info will be ignored"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        
    def __eq__(self, other):
        if not isinstance(other, ECP):
            return False
            
        return super().__eq__(other) 
 

