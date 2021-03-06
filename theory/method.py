"""methods (e.g. DFT functionals, coupled-cluster methods) for Theory()"""

KNOWN_SEMI_EMPIRICAL = ["AM1", "PM3", "PM6", "PM7", "HF-3C", "PM3MM", "PDDG"]

class Method:
    """functional object
    used to ensure the proper keyword is used
    e.g.
    using Functional('PBE0') will use PBE1PBE in a gaussian input file"""
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

    def get_gaussian(self):
        """maps proper functional name to one Gaussian accepts"""
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

        return self.name.replace('ω', 'w'), None

    def get_orca(self):
        """maps proper functional name to one ORCA accepts"""
        if (
                self.name == "ωB97X-D" or
                any(
                    test == self.name.lower() for test in ["wb97xd", "wb97x-d"]
                )
        ):
            return ("wB97X-D3", "ωB97X-D may refer to ωB97X-D2 or ωB97X-D3 - using the latter")
        elif self.name == "ωB97X-D3":
            return ("wB97X-D3", None)
        elif any(self.name.upper() == name for name in ["B97-D", "B97D"]):
            return ("B97-D2", "B97-D may refer to B97-D2 or B97-D3 - using the former")
        elif self.name == "Gaussian's B3LYP":
            return ("B3LYP/G", None)
        elif self.name.upper() == "M06-L":
            return ("M06L", None)
        elif self.name.upper() == "M06-2X":
            return ("M062X", None)
        elif self.name.upper() == "PBE1PBE":
            return ("PBE0", None)

        return self.name.replace('ω', 'w'), None

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

        return self.name.replace('ω', 'w'), None


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