"""
used for specifying basis information for a Theory()
"""

import os

from warnings import warn

from AaronTools.theory import (
    ORCA_ROUTE,
    ORCA_BLOCKS,
    PSI4_BEFORE_GEOM,
    GAUSSIAN_ROUTE,
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
)
from AaronTools.finders import AnyTransitionMetal, AnyNonTransitionMetal, NotAny
from AaronTools.const import ELEMENTS


class Basis:
    """
    has attributes:
    name          - same as initialization keyword
    elements      - same as initialization keyword
    aux_type      - same as initialization keyword
    elements      - list of element symbols for elements this basis applies to
                    updated with Basis.refresh_elements
                    Basis.refresh_elements is called when writing an input file
    ele_selection - list of finders used to determine which elements this basis applies to
    not_anys      - list of finders used to determine which elements this basis does not apply to
    """
    default_elements = [AnyTransitionMetal(), AnyNonTransitionMetal()]
    def __init__(self, name, elements=None, aux_type=None, user_defined=False):
        """
        name         -   basis set base name (e.g. 6-31G)
        elements     -   list of element symbols or finders to determine the basis set applies to
                         elements may also be 'tm' or 'all' to indicate any transition metal and
                         all elements, respectively
                         elements may start with '!' to exclude that element from the basis
                         for example, elements='!H' will apply to default elements, minus H
        aux_type     -   str - ORCA: one of BasisSet.ORCA_AUX; Psi4: one of BasisSet.PSI4_AUX
        user_defined -   path to file containing basis info from basissetexchange.org or similar
                         False for builtin basis sets
        """
        self.name = name

        if elements is None:
            self.elements = []
            self.ele_selection = self.default_elements
            self.not_anys = []
        else:
            #a list of elements or other identifiers was given
            #if it's an element with a ! in front, add that element to not_anys
            #otherwise, add the appropriate thing to ele_selection
            if not hasattr(elements, '__iter__') or isinstance(elements, str):
                elements = [elements]

            self.elements = elements
            ele_selection = []
            not_anys = []
            for ele in elements:
                not_any = False
                if isinstance(ele, str) and ele.startswith('!'):
                    ele = ele.lstrip('!')
                    not_any = True

                if ele.lower() == 'all':
                    if not_any:
                        not_anys.append(AnyTransitionMetal())
                        not_anys.append(AnyNonTransitionMetal())
                    else:
                        ele_selection.append(AnyTransitionMetal())
                        ele_selection.append(AnyNonTransitionMetal())
                elif ele.lower() == 'tm' and ele != "Tm":
                    if not_any:
                        ele_selection.append(AnyNonTransitionMetal())
                    else:
                        ele_selection.append(AnyTransitionMetal())
                elif isinstance(ele, str) and ele in ELEMENTS:
                    if not_any:
                        not_anys.append(ele)
                    else:
                        ele_selection.append(ele)
                else:
                    warn("element not known: %s" % repr(ele))

            if not ele_selection:
                #if only not_anys were given, fall back to the default elements
                ele_selection = self.default_elements

            self.ele_selection = ele_selection
            self.not_anys = not_anys

        self.aux_type = aux_type
        self.user_defined = user_defined

    def __repr__(self):
        return "%s(%s)" % (self.name, " ".join(self.elements))

    def refresh_elements(self, geometry):
        """sets self's elements for the geometry"""
        atoms = geometry.find(self.ele_selection, NotAny(*self.not_anys))
        elements = set([atom.element for atom in atoms])
        self.elements = elements

    @staticmethod
    def get_gaussian(name):
        """
        returns the Gaussian09/16 name of the basis set
        currently just removes the hyphen from the Karlsruhe def2 ones
        """
        if name.startswith('def2-'):
            return name.replace('def2-', 'def2', 1)

        return name

    @staticmethod
    def get_orca(name):
        """
        returns the ORCA name of the basis set
        currently just adds hyphen to Karlsruhe basis if it isn't there
        """
        if name.startswith('def2') and not name.startswith('def2-'):
            return name.replace('def2', 'def2-', 1)

        return name

    @staticmethod
    def get_psi4(name):
        """
        returns the Psi4 name of the basis set
        currently just adds hyphen to Karlsruhe basis if it isn't there
        """
        if name.startswith('def2') and not name.startswith('def2-'):
            return name.replace('def2', 'def2-', 1)

        return name


class ECP(Basis):
    """ECP - aux info will be ignored"""
    default_elements = AnyTransitionMetal()
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __eq__(self, other):
        if not isinstance(other, ECP):
            return False

        return super().__eq__(other)


class BasisSet:
    """used to more easily get basis set info for writing input files"""
    ORCA_AUX = ["C", "J", "JK", "CABS", "OptRI CABS"]
    PSI4_AUX = ["JK", "RI"]

    def __init__(self, basis=None, ecp=None):
        """
        basis: list(Basis), Basis, str, or None
        ecp: list(ECP) or None
        """
        if isinstance(basis, str):
            basis = self.parse_basis_str(basis, cls=Basis)
        elif isinstance(basis, Basis):
            basis = [basis]

        if isinstance(ecp, str):
            if ecp.split():
                ecp = self.parse_basis_str(ecp, cls=ECP)
            else:
                ecp = [ECP(ecp)]
        elif isinstance(ecp, ECP):
            ecp = [ecp]

        self.basis = basis
        self.ecp = ecp

    @property
    def elements_in_basis(self):
        """returns a list of elements in self's basis"""
        elements = []
        if self.basis is not None:
            for basis in self.basis:
                elements.extend(basis.elements)

        return elements

    @staticmethod
    def parse_basis_str(basis_str, cls=Basis):
        """
        parse basis set specification string and returns list(cls)
        cls should be Basis or ECP (or subclasses of these)
        basis info should have:
            - a list of elements before basis set name (e.g. C H N O)
                - other element keywords are 'tm' (transition metals) and 'all' (all elements)
                - can also put "!" before an element to exclude it from the basis set
            - auxilliary type before basis name (e.g. auxilliary C)
            - basis set name
            - path to basis set file right after basis set name if the basis is not builtin
                - path cannot contain spaces
        Example:
            "!H !tm def2-SVPD /home/CoolUser/basis_sets/def2svpd.gbs H def2-SVP Ir SDD
        """
        info = basis_str.split()
        i = 0
        basis_sets = []
        elements = []
        aux_type = None
        user_defined = False
        while i < len(info):
            if (
                    info[i].lstrip("!") in ELEMENTS or
                    any(
                        info[i].lower().lower().lstrip("!") == x for x in ["all", "tm"]
                    )
            ):
                elements.append(info[i])
            elif info[i].lower().startswith("aux"):
                try:
                    aux_type = info[i+1]
                    i += 1
                    if aux_type.lower() == "optri":
                        aux_type += " %s" % info[i+1]
                        i += 1
                except:
                    raise RuntimeError(
                        "error while parsing basis set string: %s\nfound \"aux\"" +
                        ", but no auxilliary type followed" % basis_str
                    )
            else:
                basis_name = info[i]
                try:
                    # TODO: allow spaces in paths
                    if os.path.exists(info[i+1]) or "\\" in info[i+1] or "/" in info[i+1]:
                        user_defined = info[i+1]
                        i += 1
                except:
                    pass

                if not elements:
                    elements = None

                basis_sets.append(
                    cls(
                        basis_name,
                        elements,
                        aux_type=aux_type,
                        user_defined=user_defined,
                    )
                )
                elements = []
                aux_type = None
                user_defined = False

            i += 1

        return basis_sets

    def add_ecp(self, ecp):
        """add ecp to this BasisSet
        ecp - ECP"""
        if self.ecp is None:
            self.ecp = []

        self.ecp.append(ecp)

    def add_basis(self, basis):
        """add basis to this BasisSet
        basis - Basis"""
        if self.basis is None:
            self.basis = []

        self.basis.append(basis)

    def refresh_elements(self, geometry):
        """evaluate element specifications for each basis and ecp to
        make them compatible with the supplied geometry"""
        if self.basis is not None:
            for basis in self.basis:
                basis.refresh_elements(geometry)

        if self.ecp is not None:
            for ecp in self.ecp:
                ecp.refresh_elements(geometry)

    def get_gaussian_basis_info(self):
        """returns dict used by get_gaussian_header/footer with basis info"""
        info = {}

        if self.basis is not None:
            #check if we need to use gen or genecp:
            #    -a basis set is user-defined (stored in an external file e.g. from the BSE)
            #    -multiple basis sets
            #    -an ecp
            if (
                    all([basis == self.basis[0] for basis in self.basis])
                    and not self.basis[0].user_defined and self.ecp is None
            ):
                info[GAUSSIAN_ROUTE] = "/%s" % Basis.get_gaussian(self.basis[0].name)
            else:
                if self.ecp is None or all(not ecp.elements for ecp in self.ecp):
                    info[GAUSSIAN_ROUTE] = "/gen"
                else:
                    info[GAUSSIAN_ROUTE] = "/genecp"

                out_str = ""
                #gaussian flips out if you specify basis info for an element that
                #isn't on the molecule, so make sure the basis set has an element
                for basis in self.basis:
                    if basis.elements and not basis.user_defined:
                        out_str += " ".join([ele for ele in basis.elements])
                        out_str += " 0\n"
                        out_str += Basis.get_gaussian(basis.name)
                        out_str += "\n****\n"

                for basis in self.basis:
                    if basis.elements:
                        if basis.user_defined:
                            if os.path.exists(basis.user_defined):
                                with open(basis.user_defined, "r") as f:
                                    lines = f.readlines()

                                i = 0
                                while i < len(lines):
                                    test = lines[i].strip()
                                    if not test or test.startswith('!'):
                                        i += 1
                                        continue

                                    ele = test.split()[0]
                                    while i < len(lines):
                                        if ele in basis.elements:
                                            out_str += lines[i]

                                        if lines[i].startswith('****'):
                                            break

                                        i += 1

                                    i += 1

                            #if the file does not exists, just insert the path as an @ file
                            else:
                                out_str += "@%s\n" % basis.user_defined

                    info[GAUSSIAN_GEN_BASIS] = out_str

        if self.ecp is not None:
            out_str = ""
            for basis in self.ecp:
                if basis.elements and not basis.user_defined:
                    out_str += " ".join([ele for ele in basis.elements])
                    out_str += " 0\n"
                    out_str += Basis.get_gaussian(basis.name)
                    out_str += "\n"

            for basis in self.ecp:
                if basis.elements:
                    if basis.user_defined:
                        if os.path.exists(basis.user_defined):
                            with open(basis.user_defined, "r") as f:
                                lines = f.readlines()

                                i = 0
                                while i < len(lines):
                                    test = lines[i].strip()
                                    if not test or test.startswith('!'):
                                        i += 1
                                        continue

                                    ele = test.split()[0]
                                    while i < len(lines):
                                        if ele in basis.elements:
                                            out_str += lines[i]

                                        if lines[i].startswith('****'):
                                            break

                                        i += 1

                                    i += 1

                        else:
                            out_str += "@%s\n" % basis.user_defined

            info[GAUSSIAN_GEN_ECP] = out_str

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
                if basis.elements:
                    if basis.aux_type is None:
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                out_str = Basis.get_orca(basis.name)
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = "GTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newGTO            %-2s " % ele

                                if not basis.user_defined:
                                    out_str += "\"%s\" end" % Basis.get_orca(basis.name)
                                else:
                                    out_str += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(out_str)

                    elif basis.aux_type.upper() == "C":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                out_str = "%s/C" % Basis.get_orca(basis.name)
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = "AuxCGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newAuxCGTO        %-2s " % ele

                                if not basis.user_defined:
                                    out_str += "\"%s/C\" end" % Basis.get_orca(basis.name)
                                else:
                                    out_str += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(out_str)

                    elif basis.aux_type.upper() == "J":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                out_str = "%s/J" % Basis.get_orca(basis.name)
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = "AuxJGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newAuxJGTO        %-2s " % ele

                                if not basis.user_defined:
                                    out_str += "\"%s/J\" end" % Basis.get_orca(basis.name)
                                else:
                                    out_str += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(out_str)

                    elif basis.aux_type.upper() == "JK":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                out_str = "%s/JK" % Basis.get_orca(basis.name)
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = "AuxJKGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newAuxJKGTO       %-2s " % ele

                                if not basis.user_defined:
                                    out_str += "\"%s/JK\" end" % Basis.get_orca(basis.name)
                                else:
                                    out_str += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(out_str)

                    elif basis.aux_type.upper() == "CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                out_str = "%s-CABS" % Basis.get_orca(basis.name)
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = "CABSGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newCABSGTO        %-2s " % ele

                                if not basis.user_defined:
                                    out_str += "\"%s-CABS\" end" % Basis.get_orca(basis.name)
                                else:
                                    out_str += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(out_str)

                    elif basis.aux_type.upper() == "OPTRI CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                out_str = "%s-OptRI" % Basis.get_orca(basis.name)
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = "CABSGTOName \"%s\"" % basis.user_defined
                                info[ORCA_BLOCKS]['basis'].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newCABSGTO        %-2s " % ele

                                if not basis.user_defined:
                                    out_str += "\"%s-OptRI\" end" % Basis.get_orca(basis.name)
                                else:
                                    out_str += "\"%s\" end" % basis.user_defined

                                info[ORCA_BLOCKS]['basis'].append(out_str)

        if self.ecp is not None:
            for basis in self.ecp:
                if basis.elements and not basis.user_defined:
                    for ele in basis.elements:
                        out_str = "newECP            %-2s " % ele
                        out_str += "\"%s\" end" % Basis.get_orca(basis.name)

                        info[ORCA_BLOCKS]['basis'].append(out_str)

                elif basis.elements and basis.user_defined:
                    #TODO: check if this works
                    out_str = "GTOName \"%s\"" % basis.user_defined

                    info[ORCA_BLOCKS]['basis'].append(out_str)

        return info

    def get_psi4_basis_info(self, sapt=False):
        """
        sapt: bool, use df_basis_sapt instead of df_basis_scf for jk basis
        return dict for get_psi4_header
        """
        out_str = ""
        out_str2 = None
        out_str3 = None
        out_str4 = None

        first_basis = []

        if self.basis is not None:
            for basis in self.basis:
                if basis.aux_type not in first_basis:
                    first_basis.append(basis.aux_type)
                    if basis.aux_type is None or basis.user_defined:
                        out_str += "basis {\n"
                        out_str += "    assign    %s\n" % Basis.get_psi4(basis.name)

                    elif basis.aux_type.upper() == "JK":
                        if sapt:
                            out_str4 = "df_basis_sapt {\n"
                        else:
                            out_str4 = "df_basis_scf {\n"

                        out_str4 += "    assign    %s-jkfit\n" % Basis.get_psi4(basis.name)

                    elif basis.aux_type.upper() == "RI":
                        out_str2 = "df_basis_%s {\n"
                        if basis.name.lower() == "sto-3g" or basis.name.lower() == "3-21g":
                            out_str2 += "    assign    %s-rifit\n" % Basis.get_psi4(basis.name)
                        else:
                            out_str2 += "    assign    %s-ri\n" % Basis.get_psi4(basis.name)

                else:
                    if basis.aux_type is None or basis.user_defined:
                        for ele in basis.elements:
                            out_str += "    assign %-2s %s\n" % (ele, Basis.get_psi4(basis.name))

                    elif basis.aux_type.upper() == "JK":
                        for ele in basis.elements:
                            out_str4 += "    assign %-2s %s-jkfit\n" % (
                                ele,
                                Basis.get_psi4(basis.name),
                            )

                    elif basis.aux_type.upper() == "RI":
                        for ele in basis.elements:
                            if basis.name.lower() == "sto-3g" or basis.name.lower() == "3-21g":
                                out_str2 += "    assign %-2s %s-rifit\n" % (
                                    ele,
                                    Basis.get_psi4(basis.name),
                                )
                            else:
                                out_str2 += "    assign %-2s %s-ri\n" % (
                                    ele,
                                    Basis.get_psi4(basis.name),
                                )

            if any(basis.user_defined for basis in self.basis):
                out_str3 = ""
                for basis in self.basis:
                    if basis.user_defined:
                        if os.path.exists(basis.user_defined):
                            out_str3 += "\n[%s]\n" % basis.name
                            with open(basis.user_defined, 'r') as f:
                                lines = [
                                    line.rstrip() for line in f.readlines()
                                    if line.strip() and not line.startswith('!')
                                ]
                                out_str3 += '\n'.join(lines)
                                out_str3 += '\n\n'

            if out_str3 is not None:
                out_str += out_str3

            if out_str:
                out_str += "}"

            if out_str2 is not None:
                out_str2 += "}"

                out_str += "\n\n%s" % out_str2

            if out_str4 is not None:
                out_str4 += "}"

                out_str += "\n\n%s" % out_str4

        if out_str:
            info = {PSI4_BEFORE_GEOM:[out_str]}
        else:
            info = {}

        return info

    def check_for_elements(self, geometry):
        """checks to make sure each element is in a basis set"""
        warning = ""
        #assume all elements aren't in a basis set, remove from the list if they have a basis
        #need to check each type of aux basis
        elements = list(set([str(atom.element) for atom in geometry.atoms]))
        if self.basis is not None:
            elements_without_basis = {}
            for basis in self.basis:
                if basis.aux_type not in elements_without_basis:
                    elements_without_basis[basis.aux_type] = [str(e) for e in elements]

                for element in basis.elements:
                    if element in elements_without_basis[basis.aux_type]:
                        elements_without_basis[basis.aux_type].remove(element)

            if any(elements_without_basis[aux] for aux in elements_without_basis.keys()):
                for aux in elements_without_basis.keys():
                    if elements_without_basis[aux]:
                        if aux is not None and aux != "no":
                            warning += "%s ha%s no auxiliary %s basis; " % (
                                ", ".join(elements_without_basis[aux]),
                                "s" if len(elements_without_basis[aux]) == 1 else "ve",
                                aux,
                            )

                        else:
                            warning += "%s ha%s no basis; " % (
                                ", ".join(elements_without_basis[aux]),
                                "s" if len(elements_without_basis[aux]) == 1 else "ve",
                            )

                return warning.strip('; ')

            return None
