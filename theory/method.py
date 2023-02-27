"""methods (e.g. DFT functionals, coupled-cluster methods) for Theory()"""
import re

from AaronTools import addlogger

KNOWN_SEMI_EMPIRICAL = [
    "AM1",
    "PM3",
    "PM6",
    "PM7",
    "HF-3C",
    "PM3MM",
    "PDDG",
    "RM1",
    "MNDO",
    "PM3-PDDG",
    "MNDO-PDDG",
    "PM3-CARB1",
    "MNDO/d",
    "AM1/d",
    "DFTB2",
    "DFTB3",
    "AM1-D*",
    "PM6-D",
    "AM1-DH+",
    "PM6-DH+",
]

class Method:
    """functional object
    used to ensure the proper keyword is used
    e.g.
    using Functional('PBE0') will use PBE1PBE in a gaussian input file"""

    LOG = None

    def __init__(self, name, is_semiempirical=False):
        """
        name: str, functional name
        is_semiempirical: bool, basis set is not required
        """
        self.name = name
        self.is_semiempirical = is_semiempirical

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return (
            self.get_gaussian()[0].lower() == other.get_gaussian()[0].lower() and
            self.is_semiempirical == other.is_semiempirical
        )

    @staticmethod
    def sanity_check_method(name, program):
        """
        check to see if method is available in the specified program
        name - str, name of method
        program, str, gaussian, orca, psi4, or qchem
        """
        import os.path
        from difflib import SequenceMatcher as seqmatch
        from numpy import argsort, loadtxt
        from AaronTools.const import AARONTOOLS
        
        warning = None
        
        prefix = ""
        if program.lower() == "gaussian":
            valid = loadtxt(os.path.join(AARONTOOLS, "theory", "valid_methods", "gaussian.txt"), dtype=str)
            prefix = "(?:RO|R|U)?"
        elif program.lower() == "orca":
            valid = loadtxt(os.path.join(AARONTOOLS, "theory", "valid_methods", "orca.txt"), dtype=str)
        elif program.lower() == "psi4":
            valid = loadtxt(os.path.join(AARONTOOLS, "theory", "valid_methods", "psi4.txt"), dtype=str)
        elif program.lower() == "sqm":
            valid = loadtxt(os.path.join(AARONTOOLS, "theory", "valid_methods", "sqm.txt"), dtype=str)
        elif program.lower() == "qchem":
            valid = loadtxt(os.path.join(AARONTOOLS, "theory", "valid_methods", "qchem.txt"), dtype=str)
        elif program.lower() == "xtb":
            valid = loadtxt(os.path.join(AARONTOOLS, "theory", "valid_methods", "xtb.txt"), dtype=str)
        else:
            raise NotImplementedError("cannot validate method names for %s" % program)
        
        if not any(
            # need to escape () b/c they aren't capturing groups, it's ccsd(t) or something
            re.match(
                "%s%s$" % (prefix, method.replace("(", "\(").replace(")", "\)").replace("+", "\+")), name, flags=re.IGNORECASE
            ) for method in valid
        ):
            warning = "method '%s' may not be available in %s\n" % (name, program) + \
            "if this is incorrect, please submit a bug report at https://github.com/QChASM/AaronTools.py/issues"
            
            # try to suggest alternatives that have similar names
            simm = [
                seqmatch(
                    lambda x: x in "-_()/", name.upper(), test_method.upper()
                ).ratio() for test_method in valid
            ]
            ndx = argsort(simm)[-5:][::-1]
            warning += "\npossible misspelling of:\n"
            warning += "\n".join([valid[i] for i in ndx])

        return warning

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

    def get_gaussian(self):
        """maps proper functional name to one Gaussian accepts"""
        warning = None
        if self.name.lower() == "ωb97x-d" or self.name.lower() == "wb97x-d":
            return ("wB97XD", None)
        elif self.name == "Gaussian's B3LYP":
            return ("B3LYP", None)
        elif self.name.lower() == "b97-d":
            return ("B97D", None)
        elif self.name.lower().startswith("m06-"):
            return (self.name.upper().replace("M06-", "M06", 1), None)

        elif self.name.upper() == "PBE0":
            return ("PBE1PBE", None)

        #methods available in ORCA but not Gaussian
        elif self.name.lower() == "ωb97x-d3":
            return ("wB97XD", "ωB97X-D3 is not available in Gaussian, switching to ωB97X-D2")
        elif self.name.lower() == "b3lyp":
            return ("B3LYP", None)
        
        name = self.name.replace('ω', 'w')
        return name, warning

    def get_orca(self):
        """maps proper functional name to one ORCA accepts"""
        from AaronTools.theory import ORCA_ROUTE
        
        warnings = []
        if (
                self.name == "ωB97X-D" or
                any(
                    test == self.name.lower() for test in ["wb97xd", "wb97x-d"]
                )
        ):
            warnings.append("ωB97X-D may refer to ωB97X-D2 or ωB97X-D3 - using the latter")
            return ({ORCA_ROUTE: ["wB97X-D3"]}, warnings)
        elif self.name == "ωB97X-D3":
            return ({ORCA_ROUTE: ["wB97X-D3"]}, warnings)
        elif any(self.name.upper() == name for name in ["B97-D", "B97D"]):
            return ({ORCA_ROUTE: ["B97-D"]}, warnings)
        elif self.name == "Gaussian's B3LYP":
            return ({ORCA_ROUTE: ["B3LYP/G", "NoCosx"]}, warnings)
        elif self.name.upper() == "M06-L":
            return ({ORCA_ROUTE: ["M06L"]}, warnings)
        elif self.name.upper() == "M06-2X":
            return ({ORCA_ROUTE: ["M062X"]}, warnings)
        elif self.name.upper() == "PBE1PBE":
            return ({ORCA_ROUTE: ["PBE0"]}, warnings)

        name = self.name.replace('ω', 'w')
        warning = self.sanity_check_method(name, "orca")
        if warning:
            warnings.append(warning)
        return {ORCA_ROUTE: [name]}, warnings

    def get_psi4(self):
        """maps proper functional name to one Psi4 accepts"""
        if self.name.lower() == 'wb97xd':
            return "wB97X-D", None
        elif self.name.upper() == 'B97D':
            return ("B97-D", None)
        elif self.name.upper() == "PBE1PBE":
            return ("PBE0", None)
        elif self.name.upper() == "M062X":
            return ("M06-2X", None)
        elif self.name.upper() == "M06L":
            return ("M06-L", None)

        # the functionals havent been combined with dispersion yet, so
        # we aren't checking if the method is available

        return self.name.replace('ω', 'w'), None

    def get_sqm(self):
        """get method name that is appropriate for sqm"""
        return self.name

    def get_qchem(self):
        """maps proper functional name to one Psi4 accepts"""
        if re.match("[wω]b97[xm]?[^-xm]", self.name.lower()) and self.name.lower() != "wb97m(2)":
            name = re.match("([wω]?)b97([xm]?)([\S]+)", self.name.lower())
            return ("%sB97%s%s" % (
                    name.group(1) if name.group(1) else "",
                    name.group(2).upper() if name.group(2) else "",
                    "-%s" % name.group(3).upper() if name.group(3) else "",
                ),
                None
            )
        elif self.name.upper() == 'B97D':
            return ("B97-D", None)
        elif self.name.upper() == "M062X":
            return ("M06-2X", None)
        elif self.name.upper() == "M06L":
            return ("M06-L", None)

        # the functionals havent been combined with dispersion yet, so
        # we aren't checking if the method is available

        return self.name.replace('ω', 'w'), None

    def get_xtb(self):
        if self.name.lower() == "gfn-ff":
            return {"command_line": {"gfnff": []}}, None
        if self.name.lower() == "gfn1-xtb":
            return {"command_line": {"gfn": ["1"]}}, None
        if self.name.lower() == "gfn2-xtb":
            return {"command_line": {"gfn": ["2"]}}, None
        return {"xcontrol": {"gfn": ["method=%s" % self.name]}}, None

    def get_crest(self):
        if self.name.lower() == "gfn-ff":
            return {"command_line": {"gfnff": []}}, None
        if self.name.lower() == "gfn1-xtb":
            return {"command_line": {"gfn1": []}}, None
        if self.name.lower() == "gfn2-xtb":
            return {"command_line": {"gfn2": []}}, None
        if self.name.lower() == "gfn2-xtb//gfn-ff":
            return {"command_line": {"gfn2//gfnff": []}}, None
        return {"xcontrol": {"gfn": ["method=%s" % self.name]}}, None


class SAPTMethod(Method):
    """
    method used to differentiate between regular methods and sapt
    methods because the molecule will need to be split into monomers
    if using a sapt method, the geometry given to Theory or Geometry.write
    should have a 'components' attribute with each monomer being a coordinate
    the charge and multiplicity given to Theory should be a list, with the first
    item in each list being the overall charge/multiplicity and the subsequent items
    being the charge/multiplicity of the monomers (components)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)