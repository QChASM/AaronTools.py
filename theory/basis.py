"""
used for specifying basis information for a Theory()
"""

import os
from warnings import warn

from AaronTools.const import ELEMENTS
from AaronTools.finders import (
    AnyNonTransitionMetal,
    AnyTransitionMetal,
    NotAny,
)
from AaronTools.theory import (
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_BEFORE_GEOM,
)


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
            # a list of elements or other identifiers was given
            # if it's an element with a ! in front, add that element to not_anys
            # otherwise, add the appropriate thing to ele_selection
            if not hasattr(elements, "__iter__") or isinstance(elements, str):
                elements = [elements]

            self.elements = elements
            ele_selection = []
            not_anys = []
            for ele in elements:
                not_any = False
                if isinstance(ele, str) and ele.startswith("!"):
                    ele = ele.lstrip("!")
                    not_any = True

                if ele.lower() == "all":
                    if not_any:
                        not_anys.append(AnyTransitionMetal())
                        not_anys.append(AnyNonTransitionMetal())
                    else:
                        ele_selection.append(AnyTransitionMetal())
                        ele_selection.append(AnyNonTransitionMetal())
                elif ele.lower() == "tm" and ele != "Tm":
                    if not_any:
                        ele_selection.append(AnyNonTransitionMetal())
                    else:
                        ele_selection.append(AnyTransitionMetal())
                elif ele.lower() == "!tm" and ele != "!Tm":
                    if not_any:
                        ele_selection.append(AnyNonTransitionMetal())
                    else:
                        ele_selection.append(AnyNonTransitionMetal())
                elif isinstance(ele, str) and ele in ELEMENTS:
                    if not_any:
                        not_anys.append(ele)
                    else:
                        ele_selection.append(ele)
                else:
                    warn("element not known: %s" % repr(ele))

            if not ele_selection and not_anys:
                # if only not_anys were given, fall back to the default elements
                ele_selection = self.default_elements

            self.ele_selection = ele_selection
            self.not_anys = not_anys

        self.aux_type = aux_type
        self.user_defined = user_defined

    def __repr__(self):
        return "%s(%s)" % (self.name, " ".join(self.elements))

    def __lt__(self, other):
        if self.name < other.name:
            return True
        elif self.name == other.name and self.elements and other.elements:
            return self.elements[0] < other.elements[0]
        return False

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        if (
            self.get_gaussian(self.name).lower()
            != other.get_gaussian(other.name).lower()
        ):
            return False

        if self.aux_type != other.aux_type:
            return False

        for obj, obj2 in zip([self, other], [other, self]):
            for finder in obj.ele_selection:
                if isinstance(finder, str):
                    if finder not in obj2.ele_selection:
                        return False
                else:
                    for finder2 in obj2.ele_selection:
                        if repr(finder) == repr(finder2):
                            break
                    else:
                        return False

        return True

    def refresh_elements(self, geometry):
        """sets self's elements for the geometry"""
        atoms = geometry.find(self.ele_selection, NotAny(*self.not_anys))
        elements = set([atom.element for atom in atoms])
        self.elements = sorted(elements)

    @staticmethod
    def sanity_check_basis(name, program):
        import os.path
        from difflib import SequenceMatcher as seqmatch
        from re import IGNORECASE, match

        from AaronTools.const import AARONTOOLS
        from numpy import argsort, loadtxt

        warning = None

        if program.lower() == "gaussian":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS, "theory", "valid_basis_sets", "gaussian.txt"
                ),
                dtype=str,
            )
        elif program.lower() == "orca":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS, "theory", "valid_basis_sets", "orca.txt"
                ),
                dtype=str,
            )
        elif program.lower() == "psi4":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS, "theory", "valid_basis_sets", "psi4.txt"
                ),
                dtype=str,
            )
        else:
            raise NotImplementedError(
                "cannot validate basis names for %s" % program
            )

        if not any(
            # need to escape () b/c they aren't capturing groups, it's ccsd(t) or something
            match(
                "%s$"
                % (basis.replace("(", "\(").replace(")", "\)"))
                .replace("*", "\*")
                .replace("+", "\+"),
                name,
                flags=IGNORECASE,
            )
            for basis in valid
        ):
            warning = (
                "basis '%s' may not be available in %s\n" % (name, program)
                + "if this is incorrect, please submit a bug report at https://github.com/QChASM/AaronTools.py/issues"
            )

            # try to suggest alternatives that have similar names
            simm = [
                seqmatch(
                    lambda x: x in "-_()/", name.upper(), test_basis.upper()
                ).ratio()
                for test_basis in valid
            ]
            ndx = argsort(simm)[-5:][::-1]
            warning += "\npossible misspelling of:\n"
            warning += "\n".join([valid[i] for i in ndx])

        return warning

    @staticmethod
    def get_gaussian(name):
        """
        returns the Gaussian09/16 name of the basis set
        currently just removes the hyphen from the Karlsruhe def2 ones
        """
        if name.startswith("def2-"):
            return name.replace("def2-", "def2", 1)

        return name

    @staticmethod
    def get_orca(name):
        """
        returns the ORCA name of the basis set
        currently just adds hyphen to Karlsruhe basis if it isn't there
        """
        if name.startswith("def2") and not name.startswith("def2-"):
            return name.replace("def2", "def2-", 1)

        return name

    @staticmethod
    def get_psi4(name):
        """
        returns the Psi4 name of the basis set
        currently just adds hyphen to Karlsruhe basis if it isn't there
        """
        if name.startswith("def2") and not name.startswith("def2-"):
            return name.replace("def2", "def2-", 1)

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

    @staticmethod
    def sanity_check_basis(name, program):
        import os.path
        from difflib import SequenceMatcher as seqmatch
        from re import IGNORECASE, match

        from AaronTools.const import AARONTOOLS
        from numpy import argsort, loadtxt

        warning = None

        if program.lower() == "gaussian":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS,
                    "theory",
                    "valid_basis_sets",
                    "gaussian_ecp.txt",
                ),
                dtype=str,
            )
        elif program.lower() == "orca":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS, "theory", "valid_basis_sets", "orca_ecp.txt"
                ),
                dtype=str,
            )
        else:
            raise NotImplementedError(
                "cannot validate basis names for %s" % program
            )

        if not any(
            # need to escape () b/c they aren't capturing groups, it's ccsd(t) or something
            match(
                "%s$"
                % (basis.replace("(", "\(").replace(")", "\)"))
                .replace("*", "\*")
                .replace("+", "\+"),
                name,
                flags=IGNORECASE,
            )
            for basis in valid
        ):
            warning = (
                "basis '%s' may not be available in %s\n" % (name, program)
                + "if this is incorrect, please submit a bug report at https://github.com/QChASM/AaronTools.py/issues"
            )

            # try to suggest alternatives that have similar names
            simm = [
                seqmatch(
                    lambda x: x in "-_()/", name.upper(), test_basis.upper()
                ).ratio()
                for test_basis in valid
            ]
            ndx = argsort(simm)[-5:][::-1]
            warning += "\npossible misspelling of:\n"
            warning += "\n".join([valid[i] for i in ndx])

        return warning


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
        elif isinstance(basis, BasisSet):
            if ecp is None:
                ecp = basis.ecp
            basis = basis.basis

        if isinstance(ecp, str):
            if ecp.split():
                ecp = self.parse_basis_str(ecp, cls=ECP)
            else:
                ecp = [ECP(ecp)]
        elif isinstance(ecp, ECP):
            ecp = [ecp]
        elif isinstance(ecp, BasisSet):
            ecp = ecp.ecp

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
            if info[i].lstrip("!") in ELEMENTS or any(
                info[i].lower().lower().lstrip("!") == x for x in ["all", "tm"]
            ):
                elements.append(info[i])
            elif info[i].lower().startswith("aux"):
                try:
                    aux_type = info[i + 1]
                    i += 1
                    if aux_type.lower() == "optri":
                        aux_type += " %s" % info[i + 1]
                        i += 1
                except:
                    raise RuntimeError(
                        'error while parsing basis set string: %s\nfound "aux"'
                        + ", but no auxilliary type followed" % basis_str
                    )
            else:
                basis_name = info[i]
                try:
                    # TODO: allow spaces in paths
                    if (
                        # os thinks I have a file named "aux" somewhere on my computer
                        # I don't see it, but basis file names cannot start with 'aux'
                        os.path.exists(info[i + 1])
                        and not info[i + 1].lower().startswith("aux")
                    ) or os.sep in info[i + 1]:
                        user_defined = info[i + 1]
                        i += 1
                except Exception as e:
                    pass

                if not elements:
                    elements = None

                basis_sets.append(
                    cls(
                        basis_name,
                        elements=elements,
                        aux_type=aux_type,
                        user_defined=user_defined,
                    )
                )
                elements = []
                aux_type = None
                user_defined = False
            i += 1
        return basis_sets

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        if self.basis and other.basis:
            if len(self.basis) != len(other.basis):
                return False
            for b1, b2 in zip(sorted(self.basis), sorted(other.basis)):
                if b1 != b2:
                    return False
        else:
            if self.basis != other.basis:
                return False

        if self.ecp and other.ecp:
            if len(self.ecp) != len(other.ecp):
                return False
            for b1, b2 in zip(sorted(self.ecp), sorted(other.ecp)):
                if b1 != b2:
                    return False
        else:
            if bool(self.ecp) != bool(other.ecp):
                return False

        return True

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
        warnings = []

        if self.basis is not None:
            # check if we need to use gen or genecp:
            #    -a basis set is user-defined (stored in an external file e.g. from the BSE)
            #    -multiple basis sets
            #    -an ecp
            if (
                all([basis == self.basis[0] for basis in self.basis])
                and not self.basis[0].user_defined
                and self.ecp is None
            ):
                basis_name = Basis.get_gaussian(self.basis[0].name)
                warning = self.basis[0].sanity_check_basis(
                    basis_name, "gaussian"
                )
                if warning:
                    warnings.append(warning)
                info[GAUSSIAN_ROUTE] = "/%s" % basis_name
            else:
                if self.ecp is None or all(
                    not ecp.elements for ecp in self.ecp
                ):
                    info[GAUSSIAN_ROUTE] = "/gen"
                else:
                    info[GAUSSIAN_ROUTE] = "/genecp"

                out_str = ""
                # gaussian flips out if you specify basis info for an element that
                # isn't on the molecule, so make sure the basis set has an element
                for basis in self.basis:
                    if basis.elements and not basis.user_defined:
                        out_str += " ".join([ele for ele in basis.elements])
                        out_str += " 0\n"
                        basis_name = Basis.get_gaussian(basis.name)
                        warning = basis.sanity_check_basis(
                            basis_name, "gaussian"
                        )
                        if warning:
                            warnings.append(warning)
                        out_str += basis_name
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
                                    if not test or test.startswith("!"):
                                        i += 1
                                        continue

                                    ele = test.split()[0]
                                    while i < len(lines):
                                        if ele in basis.elements:
                                            out_str += lines[i]

                                        if lines[i].startswith("****"):
                                            break

                                        i += 1

                                    i += 1

                            # if the file does not exists, just insert the path as an @ file
                            else:
                                out_str += "@%s\n" % basis.user_defined

                info[GAUSSIAN_GEN_BASIS] = out_str

        if self.ecp is not None:
            out_str = ""
            for basis in self.ecp:
                if basis.elements and not basis.user_defined:
                    out_str += " ".join([ele for ele in basis.elements])
                    out_str += " 0\n"
                    basis_name = Basis.get_gaussian(basis.name)
                    warning = basis.sanity_check_basis(basis_name, "gaussian")
                    if warning:
                        warnings.append(warning)
                    out_str += basis_name
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
                                    if not test or test.startswith("!"):
                                        i += 1
                                        continue

                                    ele = test.split()[0]
                                    while i < len(lines):
                                        if ele in basis.elements:
                                            out_str += lines[i]

                                        if lines[i].startswith("****"):
                                            break

                                        i += 1

                                    i += 1

                        else:
                            out_str += "@%s\n" % basis.user_defined

            info[GAUSSIAN_GEN_ECP] = out_str

            if self.basis is None:
                info[GAUSSIAN_ROUTE] = " Pseudo=Read"

        return info, warnings

    def get_orca_basis_info(self):
        """return dict for get_orca_header"""
        # TODO: warn if basis should be f12
        info = {ORCA_BLOCKS: {"basis": []}, ORCA_ROUTE: []}
        warnings = []

        first_basis = []

        if self.basis is not None:
            for basis in self.basis:
                if basis.elements:
                    if basis.aux_type is None:
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                basis_name = Basis.get_orca(basis.name)
                                warning = Basis.sanity_check_basis(
                                    basis_name, "orca"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str = basis_name
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = 'GTOName "%s"' % basis.user_defined
                                info[ORCA_BLOCKS]["basis"].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newGTO            %-2s " % ele

                                if not basis.user_defined:
                                    basis_name = Basis.get_orca(basis.name)
                                    warning = Basis.sanity_check_basis(
                                        basis_name, "orca"
                                    )
                                    if warning:
                                        warnings.append(warning)
                                    out_str += '"%s" end' % basis_name
                                else:
                                    out_str += '"%s" end' % basis.user_defined

                                info[ORCA_BLOCKS]["basis"].append(out_str)

                    elif basis.aux_type.upper() == "C":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                basis_name = Basis.get_orca(basis.name) + "/C"
                                warning = Basis.sanity_check_basis(
                                    basis_name, "orca"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str = "%s" % basis_name
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = (
                                    'AuxCGTOName "%s"' % basis.user_defined
                                )
                                info[ORCA_BLOCKS]["basis"].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newAuxCGTO        %-2s " % ele

                                if not basis.user_defined:
                                    basis_name = (
                                        Basis.get_orca(basis.name) + "/C"
                                    )
                                    warning = Basis.sanity_check_basis(
                                        basis_name, "orca"
                                    )
                                    if warning:
                                        warnings.append(warning)

                                    out_str += '"%s" end' % basis_name
                                else:
                                    out_str += '"%s" end' % basis.user_defined

                                info[ORCA_BLOCKS]["basis"].append(out_str)

                    elif basis.aux_type.upper() == "J":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                basis_name = Basis.get_orca(basis.name) + "/J"
                                warning = Basis.sanity_check_basis(
                                    basis_name, "orca"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str = "%s" % basis_name
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = (
                                    'AuxJGTOName "%s"' % basis.user_defined
                                )
                                info[ORCA_BLOCKS]["basis"].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newAuxJGTO        %-2s " % ele

                                if not basis.user_defined:
                                    basis_name = (
                                        Basis.get_orca(basis.name) + "/J"
                                    )
                                    warning = Basis.sanity_check_basis(
                                        basis_name, "orca"
                                    )
                                    if warning:
                                        warnings.append(warning)

                                    out_str += '"%s" end' % basis_name
                                else:
                                    out_str += '"%s" end' % basis.user_defined

                                info[ORCA_BLOCKS]["basis"].append(out_str)

                    elif basis.aux_type.upper() == "JK":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                basis_name = Basis.get_orca(basis.name) + "/JK"
                                warning = Basis.sanity_check_basis(
                                    basis_name, "orca"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str = "%s" % basis_name
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = (
                                    'AuxJKGTOName "%s"' % basis.user_defined
                                )
                                info[ORCA_BLOCKS]["basis"].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newAuxJKGTO       %-2s " % ele

                                if not basis.user_defined:
                                    basis_name = (
                                        Basis.get_orca(basis.name) + "/JK"
                                    )
                                    warning = Basis.sanity_check_basis(
                                        basis_name, "orca"
                                    )
                                    if warning:
                                        warnings.append(warning)

                                    out_str += '"%s" end' % basis_name
                                else:
                                    out_str += '"%s" end' % basis.user_defined

                                info[ORCA_BLOCKS]["basis"].append(out_str)

                    elif basis.aux_type.upper() == "CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                basis_name = (
                                    Basis.get_orca(basis.name) + "-CABS"
                                )
                                warning = Basis.sanity_check_basis(
                                    basis_name, "orca"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str = "%s" % basis_name
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = (
                                    'CABSGTOName "%s"' % basis.user_defined
                                )
                                info[ORCA_BLOCKS]["basis"].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newCABSGTO        %-2s " % ele

                                if not basis.user_defined:
                                    basis_name = (
                                        Basis.get_orca(basis.name) + "-CABS"
                                    )
                                    warning = Basis.sanity_check_basis(
                                        basis_name, "orca"
                                    )
                                    if warning:
                                        warnings.append(warning)

                                    out_str += '"%s" end' % basis_name
                                else:
                                    out_str += '"%s" end' % basis.user_defined

                                info[ORCA_BLOCKS]["basis"].append(out_str)

                    elif basis.aux_type.upper() == "OPTRI CABS":
                        if basis.aux_type not in first_basis:
                            if not basis.user_defined:
                                basis_name = (
                                    Basis.get_orca(basis.name) + "-OptRI"
                                )
                                warning = Basis.sanity_check_basis(
                                    basis_name, "orca"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str = "%s" % basis_name
                                info[ORCA_ROUTE].append(out_str)
                                first_basis.append(basis.aux_type)

                            else:
                                out_str = (
                                    'CABSGTOName "%s"' % basis.user_defined
                                )
                                info[ORCA_BLOCKS]["basis"].append(out_str)
                                first_basis.append(basis.aux_type)

                        else:
                            for ele in basis.elements:
                                out_str = "newCABSGTO        %-2s " % ele

                                if not basis.user_defined:
                                    basis_name = (
                                        Basis.get_orca(basis.name) + "-OptRI"
                                    )
                                    warning = Basis.sanity_check_basis(
                                        basis_name, "orca"
                                    )
                                    if warning:
                                        warnings.append(warning)

                                    out_str += '"%s" end' % basis_name
                                else:
                                    out_str += '"%s" end' % basis.user_defined

                                info[ORCA_BLOCKS]["basis"].append(out_str)

        if self.ecp is not None:
            for basis in self.ecp:
                if basis.elements and not basis.user_defined:
                    for ele in basis.elements:
                        out_str = "newECP            %-2s " % ele
                        basis_name = Basis.get_orca(basis.name)
                        warning = basis.sanity_check_basis(basis_name, "orca")
                        if warning:
                            warnings.append(warning)
                        out_str += '"%s" end' % basis_name

                        info[ORCA_BLOCKS]["basis"].append(out_str)

                elif basis.elements and basis.user_defined:
                    # TODO: check if this works
                    out_str = 'GTOName "%s"' % basis.user_defined

                    info[ORCA_BLOCKS]["basis"].append(out_str)

        return info, warnings

    def get_psi4_basis_info(self, sapt=False):
        """
        sapt: bool, use df_basis_sapt instead of df_basis_scf for jk basis
        return dict for get_psi4_header
        """
        out_str = ""
        out_str2 = None
        out_str3 = None
        out_str4 = None
        warnings = []

        first_basis = []

        if self.basis is not None:
            for basis in self.basis:
                if basis.aux_type not in first_basis:
                    first_basis.append(basis.aux_type)
                    if basis.aux_type is None or basis.user_defined:
                        out_str += "basis {\n"
                        basis_name = Basis.get_psi4(basis.name)
                        warning = basis.sanity_check_basis(basis_name, "psi4")
                        if warning and not basis.user_defined:
                            warnings.append(warning)
                        out_str += "    assign    %s\n" % basis_name

                    elif basis.aux_type.upper() == "JK":
                        if sapt:
                            out_str4 = "df_basis_sapt {\n"
                        else:
                            out_str4 = "df_basis_scf {\n"

                        basis_name = Basis.get_psi4(basis.name) + "-jkfit"
                        warning = basis.sanity_check_basis(basis_name, "psi4")
                        if warning and not basis.user_defined:
                            warnings.append(warning)
                        out_str4 += "    assign    %s\n" % basis_name

                    elif basis.aux_type.upper() == "RI":
                        out_str2 = "df_basis_%s {\n"
                        if (
                            basis.name.lower() == "sto-3g"
                            or basis.name.lower() == "3-21g"
                        ):
                            basis_name = Basis.get_psi4(basis.name) + "-rifit"
                            warning = basis.sanity_check_basis(
                                basis_name, "psi4"
                            )
                            if warning and not basis.user_defined:
                                warnings.append(warning)

                            out_str2 += "    assign    %s\n" % basis_name
                        else:
                            basis_name = Basis.get_psi4(basis.name) + "-ri"
                            warning = basis.sanity_check_basis(
                                basis_name, "psi4"
                            )
                            if warning and not basis.user_defined:
                                warnings.append(warning)

                            out_str2 += "    assign    %s\n" % basis_name

                else:
                    if basis.aux_type is None or basis.user_defined:
                        basis_name = Basis.get_psi4(basis.name)
                        warning = basis.sanity_check_basis(basis_name, "psi4")
                        if warning and not basis.user_defined:
                            warnings.append(warning)
                        for ele in basis.elements:
                            out_str += "    assign %-2s %s\n" % (
                                ele,
                                basis_name,
                            )

                    elif basis.aux_type.upper() == "JK":
                        basis_name = Basis.get_psi4(basis.name) + "-jkfit"
                        warning = basis.sanity_check_basis(basis_name, "psi4")
                        if warning and not basis.user_defined:
                            warnings.append(warning)
                        for ele in basis.elements:
                            out_str4 += "    assign %-2s %s\n" % (
                                ele,
                                basis_name,
                            )

                    elif basis.aux_type.upper() == "RI":
                        basis_name = Basis.get_psi4(basis.name)
                        if (
                            basis_name.lower() == "sto-3g"
                            or basis_name.lower() == "3-21g"
                        ):
                            basis_name += "-rifit"
                        else:
                            basis_name += "-ri"
                        warning = basis.sanity_check_basis(basis_name, "psi4")
                        if warning and not basis.user_defined:
                            warnings.append(warning)

                        for ele in basis.elements:
                            if (
                                basis.name.lower() == "sto-3g"
                                or basis.name.lower() == "3-21g"
                            ):
                                out_str2 += "    assign %-2s %s-rifit\n" % (
                                    ele,
                                    basis_name,
                                )

            if any(basis.user_defined for basis in self.basis):
                out_str3 = ""
                for basis in self.basis:
                    if basis.user_defined:
                        if os.path.exists(basis.user_defined):
                            out_str3 += "\n[%s]\n" % basis.name
                            with open(basis.user_defined, "r") as f:
                                lines = [
                                    line.rstrip()
                                    for line in f.readlines()
                                    if line.strip()
                                    and not line.startswith("!")
                                ]
                                out_str3 += "\n".join(lines)
                                out_str3 += "\n\n"

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
            info = {PSI4_BEFORE_GEOM: [out_str]}
        else:
            info = {}

        return info, warnings

    def check_for_elements(self, geometry):
        """checks to make sure each element is in a basis set"""
        warning = ""
        # assume all elements aren't in a basis set, remove from the list if they have a basis
        # need to check each type of aux basis
        elements = list(set([str(atom.element) for atom in geometry.atoms]))
        if self.basis is not None:
            elements_without_basis = {}
            for basis in self.basis:
                if basis.aux_type not in elements_without_basis:
                    elements_without_basis[basis.aux_type] = [
                        str(e) for e in elements
                    ]

                for element in basis.elements:
                    if element in elements_without_basis[basis.aux_type]:
                        elements_without_basis[basis.aux_type].remove(element)

            if any(
                elements_without_basis[aux]
                for aux in elements_without_basis.keys()
            ):
                for aux in elements_without_basis.keys():
                    if elements_without_basis[aux]:
                        if aux is not None and aux != "no":
                            warning += "%s ha%s no auxiliary %s basis; " % (
                                ", ".join(elements_without_basis[aux]),
                                "s"
                                if len(elements_without_basis[aux]) == 1
                                else "ve",
                                aux,
                            )

                        else:
                            warning += "%s ha%s no basis; " % (
                                ", ".join(elements_without_basis[aux]),
                                "s"
                                if len(elements_without_basis[aux]) == 1
                                else "ve",
                            )

                return warning.strip("; ")

            return None
