"""
used for specifying basis information for a Theory()
"""

import os
import re
from warnings import warn

from AaronTools import addlogger
from AaronTools.const import ELEMENTS
from AaronTools.finders import (
    AnyNonTransitionMetal,
    AnyTransitionMetal,
    NotAny,
    ONIOMLayer
)
from AaronTools.theory import (
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_BEFORE_GEOM,
    PSI4_SETTINGS,
    QCHEM_REM,
    QCHEM_SETTINGS,
)


@addlogger
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
    ONIOM-only attributes:
    oniom_layer   - same as initialization keyword
    atom_selection - list of finders used to determine which atoms this basis applies to
    atoms         - list of atoms this basis applies to
                    updated with Bases.refresh_atoms
    default_notany_atoms - finder for atoms that are not in the given layer
    """

    LOG = None

    default_elements = [AnyTransitionMetal(), AnyNonTransitionMetal()]

    def __init__(self, name, elements=None, aux_type=None, user_defined=False, oniom_layer=None, atoms=None):
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
        ONIOM-only:
        oniom_layer  -   str - must be 'H', 'M', or 'L' if not None
        atoms        -   list of finders or 'tm' to determine what atoms the basis set applies to
        """
        self.name = name
        if oniom_layer is not None:
            oniom_layer = oniom_layer.capitalize()
            self.not_anys = [ONIOMLayer(layers=['H','M','L'].remove(self.oniom_layer))]

        self.oniom_layer = oniom_layer

        if elements is None and oniom_layer is None:
            self.elements = []
            self.ele_selection = self.default_elements
            self.not_anys = []

        elif elements is None and oniom_layer is not None and atoms is None:
            if self.oniom_layer not in ['H','M','L']:
                raise ValueError("oniom_layer must be either H, M, or L")
            self.atom_selection = ONIOMLayer(layers=self.oniom_layer)
            self.ele_selection = []

        elif elements is None and oniom_layer is not None and atoms is not None:
#            if not hasattr(atoms, "__iter__") or isinstance(atoms, str):
#                atoms = [atoms]
            self.atoms = atoms
            atom_selection = []
            for atom in atoms:
                not_any = False
                if isinstance(atom, str) and atom.startswith("!"):
                    atom = atom.lstrip("!")
                    not_any = True
                if atom.lower() == "all":
                    if not_any:
                        self.not_anys.append(AnyTransitionMetal())
                        self.not_anys.append(AnyNonTransitionMetal())
                    else:
                        atom_selection.append(AnyTransitionMetal())
                        atom_selection.append(AnyNonTransitionMetal())
                elif atom.lower() == "tm":
                    if not_any:
                        atom_selection.append(AnyNonTransitionMetal())
                        self.not_anys.append(AnyTransitionMetal())
                    else:
                        atom_selection.append(AnyTransitionMetal())
                        self.not_anys.append(AnyNonTransitionMetal())
                elif isinstance(atom, str) and atom.element in ELEMENTS:
                    if not_any:
                        self.not_anys.append(atom)
                    else:
                        atom_selection.append(atom)
                else:
                    warn("atom not known: %s" % repr(atom))
 
            self.atom_selection = atom_selection
            self.ele_selection = []

        elif elements is not None and oniom_layer is not None:
            raise ValueError("use atoms keyword to describe the basis set")

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
        try:
            return "%s(%s)%s" % (
                self.name,
                " ".join(self.elements),
                "" if not self.aux_type else "/%s" % self.aux_type
            )
        except AttributeError:
            return "%s(%s)" % (self.name, " ".join(self.oniom_layer))

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

    def copy(self):
        new_dict = dict()
        for key, value in self.__dict__.items():
            try:
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
        
        return self.__class__(**new_dict)

    def refresh_elements(self, geometry):
        """sets self's elements for the geometry"""
        atoms = geometry.find(self.ele_selection, NotAny(*self.not_anys))
        elements = set([atom.element for atom in atoms])
        self.elements = sorted(elements)

    def refresh_atoms(self, geometry):
        """sets self's atoms for the geometry"""
        excluded = []
        atoms = geometry.find(self.atom_selection, NotAny(*self.not_anys)) 
        for atom_exclude in NotAny(ONIOMLayer(layers=self.oniom_layer)):
            for atom in atoms:
                if atom == atom_exclude: excluded.append(atom)
        atoms = [atom for atom in atoms if atom not in excluded]
        self.atoms = atoms

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
        elif program.lower() == "qchem":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS, "theory", "valid_basis_sets", "qchem.txt"
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
        if name.startswith("def2") and not re.match("def2(?:-|$)", name):
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

        # pople basis sets don't have commas
        # e.g. 6-31G(d,p) -> 6-31G(d_p)
        if name.startswith("6-31") or name.startswith("3-21") or name.lower().startswith("sto"):
            name = name.replace(",", "_")

        return name

    @staticmethod
    def get_qchem(name):
        """
        returns the Psi4 name of the basis set
        currently just adds hyphen to Karlsruhe basis if it isn't there
        """
        if name.startswith("def2") and not name.startswith("def2-"):
            return name.replace("def2", "def2-", 1)

        return name


class ECP(Basis):
    """ECP - aux info will be ignored"""

    default_elements = (AnyTransitionMetal(), )

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
        elif program.lower() == "qchem":
            valid = loadtxt(
                os.path.join(
                    AARONTOOLS, "theory", "valid_basis_sets", "qchem_ecp.txt"
                ),
                dtype=str,
            )
        else:
            raise NotImplementedError(
                "cannot validate ECP names for %s" % program
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
                "ECP '%s' may not be available in %s\n" % (name, program)
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


@addlogger
class BasisSet:
    """used to more easily get basis set info for writing input files"""

    LOG = None

    ORCA_AUX = ["C", "J", "JK", "CABS", "OptRI CABS"]
    PSI4_AUX = [
        "JK", "RI", "DF SCF", "DF SAPT", "DF GUESS", "DF SAD", "DF MP2",
        "DF CC", "DF DCT", "DF MCSCF", "DF ELST"
    ]
    QCHEM_AUX = ["RI", "J", "K", "corr"]

    def __init__(self, basis=None, ecp=None, angular_momentum_type=None):
        """
        basis: list(Basis), Basis, str, or None
        ecp: list(ECP) or None
        angular_momentum_type: pure, cartesian, or None
        """
        self.angular_momentum_type = angular_momentum_type
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

        if basis is None:
            basis = []
        self.basis = basis
        if ecp is None:
            ecp = []
        self.ecp = ecp

    @property
    def elements_in_basis(self):
        """returns a list of elements in self's basis"""
        elements = []
        if self.basis is not None:
            for basis in self.basis:
                elements.extend(basis.elements)

        return elements

    @property
    def atoms_in_basis(self):
        """returns a list of atoms in self's basis"""
        atoms = []
        if self.atoms is not None:
            for basis in self.basis:
                atoms.extend(basis.atoms)

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
        ONIOM only:
            - high, medium, or low to describe the ONIOM layer before the list of atoms
            - a list of atoms that can be all, tm, or ! to exclude those. automatically excludes atoms outside of layer
        Example:
            "!H !tm def2-SVPD /home/CoolUser/basis_sets/def2svpd.gbs H def2-SVP Ir SDD
        """
        # split basis_str into words
        # if there are quotes, whatever is inside the quotes is a word
        # otherwise, whitespace determines words
        info = list()
        i = 0
        word = ""
        basis_str = " ".join(basis_str.splitlines())
        while i < len(basis_str):
            s = basis_str[i]
            for char in ["\"", "'"]:
                if s == char:
                    while i < len(basis_str):
                        i += 1
                        if basis_str[i] == char:
                            break
                        word += basis_str[i]
                    info.append(word)
                    word = ""
                    i += 1
                    break
            else:
                if word and s == " ":
                    info.append(word)
                    word = ""
                else:
                    word += s
                i += 1

        if word:
            info.append(word)

        i = 0
        basis_sets = []
        elements = []
        atoms = []
        aux_type = None
        oniom_layer = None
        user_defined = False
        while i < len(info):
            if info[i].lstrip("!") in ELEMENTS or any(
                info[i].lower().lower().lstrip("!") == x for x in ["all", "tm"]
            ):
                if oniom_layer is not None:
                    atoms.append(info[i])
                    elements = []
                else:
                    elements.append(info[i])
            elif info[i].lower().startswith("aux"):
                try:
                    aux_type = info[i + 1]
                    i += 1
                    if any(aux_type.lower() == x for x in ["df", "optri"]):
                        aux_type += " %s" % info[i + 1]
                        i += 1
                except:
                    raise RuntimeError(
                        'error while parsing basis set string: %s\nfound "aux"'
                        + ", but no auxilliary type followed" % basis_str
                    )
            elif info[i].lower() in {"high", "medium", "low"}:
                oniom_layer = list(info[i])[0].capitalize()
            else:
                basis_name = info[i]
                try:
                    # TODO: allow spaces in paths
                    if (
                        # os thinks I have a file named "aux" somewhere on my computer
                        # I don't see it, but basis file names cannot start with 'aux'
                        os.path.exists(info[i + 1])
                        and not info[i + 1].lower().startswith("aux")
                        and not info[i + 1].lower() in {"high", "medium", "low"}
                    ) or (
                        "/" in info[i + 1] or
                        "\\" in info[i + 1] or
                        os.sep in info[i + 1]
                    ):
                        user_defined = info[i + 1]
                        i += 1
                except IndexError:
                    pass

                if not elements:
                    elements = None

                if not atoms:
                    atoms = None

                basis_sets.append(
                    cls(
                        basis_name,
                        elements=elements,
                        atoms=atoms,
                        oniom_layer=oniom_layer,
                        aux_type=aux_type,
                        user_defined=user_defined,
                    )
                )
                elements = []
                atoms = []
                oniom_layer = None
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
    
    def copy(self):
        new_dict = dict()
        for key, value in self.__dict__.items():
            try:
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
        
        return self.__class__(**new_dict)

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

    def refresh_atoms(self,geometry):
        """evaluate atom specification for each basis to make them compatible with the specified  geometry"""
        if self.basis is not None:
            for basis in self.basis:
                basis.refresh_atoms(geometry)

    def get_gaussian_basis_info(self):
        """returns dict used by get_gaussian_header/footer with basis info"""
        info = {GAUSSIAN_ROUTE: {}}
        warnings = []

        # check if we need to use gen or genecp:
        #    -a basis set is user-defined (stored in an external file e.g. from the BSE)
        #    -multiple basis sets
        #    -an ecp
        if (
            self.basis and 
            all([basis == self.basis[0] for basis in self.basis]) and
            not self.basis[0].user_defined and
            not self.ecp
        ):
            # we expect these to be basis sets that are in gaussian
            # get the keyword gaussian uses for this basis set
            basis_name = Basis.get_gaussian(self.basis[0].name)
            # check to make sure the basis set is in gaussian
            # (or at least the list of basis sets that we know are in gaussian)
            warning = self.basis[0].sanity_check_basis(
                basis_name, "gaussian"
            )
            if warning:
                # if it's not, try to grab it from the BSE
                try:
                    import basis_set_exchange as bse
                    bse_data = bse.get_basis(
                        self.basis[0].name,
                        elements=self.basis[0].elements,
                        fmt="gaussian94",
                        header=False,
                    )
                    info[GAUSSIAN_GEN_BASIS] = bse_data
                    # make sure to switch to /gen
                    # TODO: check if we need /genecp
                    basis_name = "gen"
                except ModuleNotFoundError:
                    # no BSE module
                    warnings.append(warning)
                except KeyError:
                    # BSE doesn't have it either
                    warnings.append(warning)
            info[GAUSSIAN_ROUTE]["/%s" % basis_name] = []
        
        else:
            # basis set is split or user-defined (i.e. in a basis set file)
            # use /gen or /genecp 
            if not self.ecp or all(
                not ecp.elements for ecp in self.ecp
            ):
                info[GAUSSIAN_ROUTE]["/gen"] = []
            else:
                info[GAUSSIAN_ROUTE]["/genecp"] = []
                try:
                    info[GAUSSIAN_ROUTE].remove("/gen")
                except KeyError:
                    pass

            out_str = ""
            # gaussian can flips out if you specify basis info for an element that
            # isn't on the molecule, so make sure the basis set has an element
            for basis in self.basis:
                if basis.elements and not basis.user_defined:
                    using_bse = False
                    basis_name = Basis.get_gaussian(basis.name)
                    warning = basis.sanity_check_basis(
                        basis_name, "gaussian"
                    )
                    if warning:
                        # this basis set isn't in our list of basis sets that we
                        # know are in gaussian
                        # try to get it from the BSE
                        try:
                            import basis_set_exchange as bse
                            bse_data = bse.get_basis(
                                basis.name,
                                elements=basis.elements,
                                fmt="gaussian94",
                                header=False,
                            )
                            # basis data is stored in user_defined
                            basis.user_defined = bse_data
                            using_bse = True
                        except ModuleNotFoundError:
                            # BSE module not installed
                            warnings.append(warning)
                        except KeyError:
                            # BSE doesn't have this basis either
                            warnings.append(warning)
                    if not using_bse:
                        out_str += " ".join([ele for ele in basis.elements])
                        out_str += " 0\n"
                        out_str += basis_name
                        out_str += "\n****\n"

            for basis in self.basis:
                if basis.elements:
                    if basis.user_defined:
                        # lines are the contents of the basis set file
                        # either read them from the file (if it exists)
                        # or grab the BSE data we stored in user_defined (which will be multiline)
                        lines = []
                        if len(basis.user_defined.splitlines()) > 1:
                            lines = [s + "\n" for s in basis.user_defined.splitlines()]
                        elif os.path.exists(basis.user_defined):
                            with open(basis.user_defined, "r") as f:
                                lines = f.readlines()
                        else:
                            # if the file does not exists, just insert the path as an @ file
                            out_str += "@%s\n" % basis.user_defined

                        if lines:
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

            info[GAUSSIAN_GEN_BASIS] = out_str

        # process ECPs similar to how we did the regular basis sets
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

        # go through both ECPs and basis sets to add ECP data
        # if we grabbed a basis set from the BSE, it might have ECP data
        # thus, we need to go through self.basis again
        for basis in [*self.ecp, *self.basis]:
            elements = set(basis.elements)
            if not isinstance(basis, ECP):
                if not basis.user_defined:
                    # Basis does not have BSE data
                    continue
                # we want to prioritize ECPs, which are likely specifically requested,
                # over a regular basis file which might have ECP data for an element
                # in common with an ECP
                for ecp in self.ecp:
                    elements -= set(ecp.elements)
            if not elements:
                continue
            
            if basis.user_defined:
                lines = []
                if len(basis.user_defined.splitlines()) > 1:
                    lines = [s + "\n" for s in basis.user_defined.splitlines()]
                elif os.path.exists(basis.user_defined):
                    with open(basis.user_defined, "r") as f:
                        lines = f.readlines()
                else:
                    out_str += "@%s\n" % basis.user_defined

                if lines:
                    i = 0
                    while i < len(lines):
                        test = lines[i].strip()
                        if not test or test.startswith("!"):
                            i += 1
                            continue

                        match = re.search("([A-Z][a-z]?)-ECP", lines[i], re.IGNORECASE)
                        if match and match.group(1).capitalize() in elements:
                            if isinstance(basis, Basis):
                                info[GAUSSIAN_ROUTE]["/genecp"] = []
                                try:
                                    del info[GAUSSIAN_ROUTE]["/gen"]
                                except KeyError:
                                    pass
                            ele = match.group(1)
                            out_str += "%s      0\n" % ele.upper()
                            while i < len(lines):
                                out_str += lines[i]

                                i += 1
                                if i >= len(lines):
                                    break
                                test_data = lines[i].split()
                                if len(test_data) == 2 and test_data[1] == "0":
                                    break
                            i += 1

                        i += 1

            info[GAUSSIAN_GEN_ECP] = out_str

            if not self.basis:
                info[GAUSSIAN_ROUTE]["Pseudo=Read"] = []

        if self.angular_momentum_type:
            if self.angular_momentum_type.lower() == "pure":
                info[GAUSSIAN_ROUTE]["5D"] = []
                info[GAUSSIAN_ROUTE]["7F"] = []
            elif self.angular_momentum_type.lower() == "cartesian":
                info[GAUSSIAN_ROUTE]["6D"] = []
                info[GAUSSIAN_ROUTE]["10F"] = []
            else:
                warnings.append(
                    "unrecognized BasisSet.angular_momentum_type: %s" % self.angular_momentum_type
                )

        return info, warnings

    def get_orca_basis_info(self):
        """return dict for get_orca_header"""
        # TODO: warn if basis should be f12
        info = {ORCA_BLOCKS: {"basis": []}, ORCA_ROUTE: []}
        warnings = []

        first_basis = []

        for basis in self.basis:
            if basis.elements:
                if basis.aux_type is None:
                    if basis.aux_type not in first_basis:
                        if not basis.user_defined:
                            basis_name = Basis.get_orca(basis.name)
                            warning = Basis.sanity_check_basis(
                                basis_name, "orca"
                            )
                            use_bse = False
                            if warning:
                                try:
                                    import basis_set_exchange as bse
                                    bse_data = bse.get_basis(
                                        basis.name,
                                        elements=basis.elements,
                                    )                                    
                                    info[ORCA_BLOCKS]["basis"].append(_bse_to_orca_fmt(bse_data))
                                    use_bse = True
                                except ModuleNotFoundError:
                                    warnings.append(warning)
                                except KeyError:
                                    warnings.append(warning)
                            if not use_bse:
                                first_basis.append(basis.aux_type)
                                info[ORCA_ROUTE].append(basis_name)


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

        if self.angular_momentum_type and self.angular_momentum_type.lower() != "pure":
            warnings.append(
                "can only use pure angular momentum with ORCA basis sets"
            )

        return info, warnings

    def get_psi4_basis_info(self, sapt=False):
        """
        sapt: bool, use df_basis_sapt instead of df_basis_scf for jk basis
        return dict for get_psi4_header
        """
        out_str = dict()
        warnings = []

        if self.basis is not None:
            for basis in self.basis:
                aux_type = basis.aux_type
                basis_name = basis.get_psi4(basis.name)
                # JK and RI will try to guess what basis set is being requested
                # specifying "DF X" as the aux type will give more control
                # but the user will have to request -ri or -jkfit explicitly 
                if isinstance(aux_type, str):
                    aux_type = aux_type.upper()
                else:
                    aux_type = "BASIS"
                if aux_type == "JK":
                    aux_type = "df_basis_scf"
                    basis_name += "-jkfit"
                elif aux_type == "DF SCF":
                    aux_type = "df_basis_scf"
                elif aux_type == "DF SAPT":
                    aux_type = "df_basis_sapt"
                elif aux_type == "DF GUESS":
                    aux_type = "df_basis_guess"
                elif aux_type == "DF SAD":
                    aux_type = "df_basis_sad"
                elif aux_type == "DF MP2":
                    aux_type = "df_basis_mp2"
                elif aux_type == "DF DCT":
                    aux_type = "df_basis_dct"
                elif aux_type == "DF MCSCF":
                    aux_type = "df_basis_mcscf"
                elif aux_type == "DF CC":
                    aux_type = "df_basis_cc"
                elif aux_type == "DF ELST":
                    aux_type = "df_basis_elst"
                elif aux_type == "RI":
                    aux_type = "df_basis_%s"
                    if sapt:
                        aux_type = "df_basis_sapt"
                    if basis_name.lower() in ["sto-3g", "3-21g"]:
                        basis_name += "-rifit"
                    else:
                        basis_name += "-ri"

                if not basis.user_defined:
                    warning = basis.sanity_check_basis(basis_name, "psi4")
                    if warning:
                        try:
                            import basis_set_exchange as bse
                            bse_data = bse.get_basis(
                                basis.name,
                                fmt="psi4",
                                elements=basis.elements,
                                header=False,
                            )
                            basis.user_defined = bse_data
                        except ModuleNotFoundError:
                            warnings.append(warning)
                        except KeyError:
                            warnings.append(warning)

                if aux_type not in out_str:
                    out_str[aux_type] = "%s this_%s {\n" % (
                        aux_type.lower(),
                        aux_type.lower().replace(" ", "_")
                    )
                    out_str[aux_type] += "    assign    %s\n" % basis_name

                else:
                    for ele in basis.elements:
                        out_str[aux_type] += "    assign %-2s %s\n" % (
                            ele, basis_name
                        )

            if any(basis.user_defined for basis in self.basis):
                for basis in self.basis:
                    if basis.user_defined:
                        aux_type = basis.aux_type
                        if not aux_type:
                            aux_type = "BASIS"
                        aux_type = aux_type.upper()
                        if aux_type not in out_str:
                            out_str[aux_type] = "%s this_%s {\n" % (
                                aux_type.lower(),
                                aux_type.lower().replace(" ", "_")
                            )

                        out_str[aux_type] += "\n[%s]\n" % basis.name
                        if len(basis.user_defined.splitlines()) > 1:
                            lines = basis.user_defined.splitlines()
                        else:
                            with open(basis.user_defined, "r") as f:
                                lines = [
                                    line.rstrip()
                                    for line in f.readlines()
                                    if line.strip()
                                    and not line.startswith("!")
                                ]
                        out_str[aux_type] += "\n".join(lines)
                        out_str[aux_type] += "\n\n"

        s = "}\n\n".join(out_str.values())
        s += "}"

        info = {PSI4_BEFORE_GEOM: [s]}
        if self.angular_momentum_type:
            if self.angular_momentum_type.lower() == "pure":
                info[PSI4_SETTINGS] = {"puream": "true"}
            elif self.angular_momentum_type.lower() == "cartesian":
                info[PSI4_SETTINGS] = {"puream": "false"}

        return info, warnings

    def check_for_elements(self, geometry, count_ecps=False):
        """checks to make sure each element is in a basis set"""
        warning = ""
        # assume all elements aren't in a basis set, remove from the list if they have a basis
        # need to check each type of aux basis
        elements = list(set([str(atom.element) for atom in geometry.atoms]))
        if self.basis is not None:
            elements_without_basis = {None: elements.copy()}
            for basis in self.basis:
                if basis.aux_type not in elements_without_basis:
                    elements_without_basis[basis.aux_type] = [
                        str(e) for e in elements
                    ]

                for element in basis.elements:
                    if element in elements_without_basis[basis.aux_type]:
                        elements_without_basis[basis.aux_type].remove(element)

            if count_ecps and self.ecp:
                for basis in self.basis:
                    if basis.aux_type != None and basis.aux_type != "no":
                        continue
                    for ecp in self.ecp:
                        for element in ecp.elements:
                            print("removing", element)
                            if element in elements_without_basis[basis.aux_type]:
                                elements_without_basis[basis.aux_type].remove(element)

            for aux in elements_without_basis:
                try:
                    elements_without_basis[aux].remove("X")
                except ValueError:
                    pass

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

    def get_qchem_basis_info(self, geom):
        """returns dict used by get_qchem_header with basis info"""
        info = {QCHEM_REM: dict(), QCHEM_SETTINGS: dict()}
        warnings = []
        
        if self.basis:
            no_aux_basis = [basis for basis in self.basis if not basis.aux_type]
            other_basis = [basis for basis in self.basis if basis not in no_aux_basis]
            aux_j_basis = [basis for basis in other_basis if basis.aux_type.lower() == "j"]
            aux_k_basis = [basis for basis in other_basis if basis.aux_type.lower() == "k"]
            aux_corr_basis = [basis for basis in other_basis if basis.aux_type.lower() == "corr"]
            aux_ri_basis = [basis for basis in other_basis if basis.aux_type.lower() == "ri"]
        else:
            no_aux_basis = []
            aux_j_basis = []
            aux_k_basis = []
            aux_corr_basis = []
            aux_ri_basis = []

        for basis_list, label in zip(
            [no_aux_basis, aux_j_basis, aux_k_basis, aux_corr_basis, aux_ri_basis],
            ["BASIS", "AUX_BASIS_J", "AUX_BASIS_K", "AUX_BASIS_CORR", "AUX_BASIS"],
        ):
            if basis_list:
                # check if we need to use gen or mixed:
                #    -a basis set is user-defined (stored in an external file e.g. from the BSE)
                #    -multiple basis sets
                if (
                    all([basis == basis_list[0] for basis in basis_list])
                    and not basis_list[0].user_defined
                ):
                    basis_name = Basis.get_qchem(basis_list[0].name)
                    warning = basis_list[0].sanity_check_basis(
                        basis_name, "qchem"
                    )
                    if warning:
                        warnings.append(warning)
                    info[QCHEM_REM][label] = "%s" % basis_name
                elif not any(basis.user_defined for basis in basis_list):
                    info[QCHEM_REM][label] = "General"
                else:
                    info[QCHEM_REM][label] = "MIXED"
    
                if any(x == info[QCHEM_REM][label] for x in ["MIXED", "General"]):
                    out_str = ""
                    for basis in basis_list:
                        if basis.elements and not basis.user_defined:
                            if info[QCHEM_REM][label] == "General":
                                for ele in basis.elements:
    
                                    out_str += "%-2s 0\n    " % ele
                                    basis_name = Basis.get_qchem(basis.name)
                                    warning = basis.sanity_check_basis(
                                        basis_name, "qchem"
                                    )
                                    if warning:
                                        warnings.append(warning)
                                    out_str += basis_name
                                    out_str += "\n    ****\n    "
    
                            else:
                                atoms = geom.find(ele)
                                for atom in atoms:
                                    out_str += "%s %i\n    " % (atom.element, geom.atoms.index(atom) + 1)
                                    basis_name = Basis.get_qchem(basis.name)
                                    warning = basis.sanity_check_basis(
                                        basis_name, "qchem"
                                    )
                                    if warning:
                                        warnings.append(warning)
                                    out_str += basis_name
                                    out_str += "\n    ****\n    "
    
                    for basis in basis_list:
                        if basis.elements and basis.user_defined:
                            if os.path.exists(basis.user_defined):
                                with open(basis.user_defined, "r") as f:
                                    lines = f.readlines()
                            
                                for element in basis.elements:
                                    atoms = geom.find(element)
                                    for atom in atoms:
                                        i = 0
                                        while i < len(lines):
                                            test = lines[i].strip()
                                            if not test or test.startswith("!") or test.startswith("$"):
                                                i += 1
                                                continue
            
                                            ele = test.split()[0]
                                            if ele == atom.element:
                                                out_str += "%s %i\n" % (ele, geom.atoms.index(atom))
                                                i += 1
                                                while i < len(lines):
                                                    if ele == atom.element:
                                                        out_str += lines[i]
                
                                                    if lines[i].startswith("****"):
                                                        break
                
                                                    i += 1
            
                                            i += 1
            
                                    # if the file does not exists, just insert the path as an @ file
                                    else:
                                        warnings.append("file not found: %s" % basis.user_defined)
    
                    info[QCHEM_SETTINGS][label.lower()] = [out_str.strip()]

        if self.ecp is not None and any(ecp.elements for ecp in self.ecp):
            # check if we need to use gen:
            #    -a basis set is user-defined (stored in an external file e.g. from the BSE)
            if (
                all([ecp == self.ecp[0] for ecp in self.ecp])
                and not self.ecp[0].user_defined
                and not self.basis
            ):
                basis_name = ECP.get_qchem(self.ecp[0].name)
                warning = self.ecp[0].sanity_check_basis(
                    basis_name, "qchem"
                )
                if warning:
                    warnings.append(warning)
                if QCHEM_REM not in info:
                    info[QCHEM_REM] = {"ECP": "%s" % basis_name}
                else:
                    info[QCHEM_REM]["ECP"] = "%s" % basis_name

            elif not any(basis.user_defined for basis in self.basis):
                if QCHEM_REM not in info:
                    info[QCHEM_REM] = {"ECP": "General"}
                else:
                    info[QCHEM_REM]["ECP"] = "General"
            else:
                if QCHEM_REM not in info:
                    info[QCHEM_REM] = {"ECP": "MIXED"}
                else:
                    info[QCHEM_REM]["ECP"] = "MIXED"

            if any(x == info[QCHEM_REM]["ECP"] for x in ["MIXED", "General"]):
                out_str = ""
                for basis in self.ecp:
                    if basis.elements and not basis.user_defined:
                        if info[QCHEM_REM]["ECP"] == "General":
                            for ele in basis.elements:
                                out_str += "%-2s 0\n    " % ele
                                basis_name = ECP.get_qchem(basis.name)
                                warning = basis.sanity_check_basis(
                                    basis_name, "qchem"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str += basis_name
                                out_str += "\n    ****\n    "

                        else:
                            atoms = geom.find(element)
                            for atom in atoms:
                                out_str += "%s %i\n    " % (atom.element, geom.atoms.index(atom) + 1)
                                basis_name = ECP.get_qchem(basis.name)
                                warning = basis.sanity_check_basis(
                                    basis_name, "qchem"
                                )
                                if warning:
                                    warnings.append(warning)
                                out_str += basis_name
                                out_str += "\n    ****\n    "

                for ecp in self.ecp:
                    if ecp.elements:
                        if ecp.user_defined:
                            if os.path.exists(ecp.user_defined):
                                with open(ecp.user_defined, "r") as f:
                                    lines = f.readlines()
                        
                                for element in ecp.elements:
                                    atoms = geom.find(element)
                                    for atom in atoms:
                                        i = 0
                                        while i < len(lines):
                                            test = lines[i].strip()
                                            if not test or test.startswith("!") or test.startswith("$"):
                                                i += 1
                                                continue
        
                                            ele = test.split()[0]
                                            if ele == atom.element:
                                                out_str += "%s %i\n" % (ele, geom.atoms.index(atom))
                                                i += 1
                                                while i < len(lines):
                                                    if ele == atom.element:
                                                        out_str += lines[i]
            
                                                    if lines[i].startswith("****"):
                                                        break
            
                                                    i += 1
        
                                            i += 1
        
                            # if the file does not exists, just insert the path as an @ file
                            else:
                                warnings.append("file not found: %s" % ecp.user_defined)
                
                if QCHEM_SETTINGS not in info:
                    info[QCHEM_SETTINGS] = {"ecp": [out_str.strip()]}
                else:
                    info[QCHEM_SETTINGS]["ecp"] = [out_str.strip()]

        if self.angular_momentum_type:
            if self.angular_momentum_type.lower() == "pure":
                info[QCHEM_REM]["PURECART"] = ["1111"]
            elif self.angular_momentum_type.lower() == "cartesian":
                info[QCHEM_REM]["PURECART"] = ["2222"]
            else:
                warnings.append(
                    "unrecognized BasisSet.angular_momentum_type: %s" % self.angular_momentum_type
                )

        return info, warnings


def _bse_to_orca_fmt(data):
    out = ""
    momentum_symbols = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}
    for element in data["elements"]:
        symbol = ELEMENTS[int(element)]
        out += "  newGTO %s\n" % symbol
        for shell in data["elements"][element]["electron_shells"]:
            # for s=p orbitals presumably
            for i, angular_momentum in enumerate(shell["angular_momentum"]):
                out += "  %s  %i\n" % (momentum_symbols[angular_momentum], len(shell["exponents"]))
                for j, (coef, exp) in enumerate(zip(shell["coefficients"][i], shell["exponents"])):
                    out += "    %2i    %14s    %14s\n" % (j + 1, exp, coef)
        out += "  end\n"
    
    return out
