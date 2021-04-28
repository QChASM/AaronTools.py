"""used for specifying emperical dispersion for Theory() instances"""

from AaronTools.theory import GAUSSIAN_ROUTE, ORCA_ROUTE

class EmpiricalDispersion:
    """try to keep emerpical dispersion keywords and settings consistent across file types"""
    def __init__(self, name):
        """name can be (availability may vary):
            Grimme D2 (or D2, -D2, GD2)
            Zero-damped Grimme D3 (or D3, -D3, GD3)
            Becke-Johnson damped Grimme D3 (or D3BJ, -D3BJ, GD3BJ)
            Becke-Johnson damped modified Grimme D3 (or B3MBJ, -D3MBJ)
            Petersson-Frisch (or PFD)
            Grimme D4 (or D4, -D4, GD4)
            Chai & Head-Gordon (or CHG, -CHG)
            Nonlocal Approximation (or NL, NLA, -NL)
            Pernal, Podeszwa, Patkowski, & Szalewicz (or DAS2009, -DAS2009)
            Podeszwa, Katarzyna, Patkowski, & Szalewicz (or DAS2010, -DAS2010)
            Coupled-Cluster Doubles (or CCD)
            Řezác, Greenwell, & Beran (or DMP2)
            Coupled-Cluster Doubles + Řezác, Greenwell, & Beran (or (CCD)DMP2)

        or simply the keyword for the input file type you are using"""

        self.name = name

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        if self.name == other.name:
            return True
        for d in [
                ["grimme d2", "d2", "-d2", "gd2"],
                ["grimme d3", "d3", "-d3", "gd3"],
                ["becke-johnson damped grimme d3", "d3bj", "-d3bj", "gd3bj"],
                ["becke-johnson damped modified grimme d3", "d3mbj", "-d3mbj"],
                ["petersson-frisch", "pfd"],
                ["grimme d4", "d4", "-d4", "gd4"],
                ["nonlocal approximation", "nl", "nla", "-nl"],
                ["coupled-cluster doubles", "ccd"],
        ]:
            if self.name.lower() in d and other.name.lower() in d:
                return True
        
        return False

    def get_gaussian(self):
        """Acceptable dispersion methods for Gaussian are:
        Grimme D2
        Grimme D3
        Becke-Johnson damped Grimme D3
        Petersson-Frisch

        Dispersion methods available in other software that will be modified are:
        Grimme D4
        undampened Grimme D3"""

        if any(
                self.name.upper() == name for name in [
                    "GRIMME D2", "GD2", "D2", "-D2"
                ]
        ):
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD2"]}}, None)
        elif any(
                self.name.upper() == name for name in [
                    "ZERO-DAMPED GRIMME D3", "GRIMME D3", "GD3", "D3", "-D3", "D3ZERO"
                ]
        ):
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3"]}}, None)
        elif any(
                self.name.upper() == name for name in [
                    "BECKE-JOHNSON DAMPED GRIMME D3", "GD3BJ", "D3BJ", "-D3BJ"
                ]
        ):
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3BJ"]}}, None)
        elif any(
                self.name.upper() == name for name in [
                    "PETERSSON-FRISCH", "PFD"
                ]
        ):
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["PFD"]}}, None)

        #dispersions in ORCA but not Gaussian
        elif self.name == "Grimme D4":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "EmpiricalDispersion":["GD3BJ"]
                }
            }
            return (out_dict, "Grimme's D4 has no keyword in Gaussian, switching to GD3BJ")

        #unrecognized
        return (self.name, "unrecognized emperical dispersion: %s" % self.name)

    def get_orca(self):
        """Acceptable keywords for ORCA are:
        Grimme D2
        Zero-damped Grimme D3
        Becke-Johnson damped Grimme D3
        Grimme D4"""
        if any(
                self.name.upper() == name for name in [
                    "GRIMME D2", "GD2", "D2", "-D2"
                ]
        ):
            return ({ORCA_ROUTE:["D2"]}, None)
        elif any(
                self.name.upper() == name for name in [
                    "ZERO-DAMPED GRIMME D3", "GRIMME D3", "GD3", "D3", "-D3", "D3ZERO"
                ]
        ):
            return ({ORCA_ROUTE:["D3ZERO"]}, None)
        elif any(
                self.name.upper() == name for name in [
                    "BECKE-JOHNSON DAMPED GRIMME D3", "GD3BJ", "D3BJ", "-D3BJ"
                ]
        ):
            return ({ORCA_ROUTE:["D3BJ"]}, None)
        elif any(
                self.name.upper() == name for name in [
                    "GRIMME D4", "GD4", "D4", "-D4"
                ]
        ):
            return ({ORCA_ROUTE:["D4"]}, None)

        out_dict = {
            ORCA_ROUTE: [self.name]
        }
        return(out_dict, "unrecognized emperical dispersion: %s" % self.name)

    def get_psi4(self):
        """Acceptable keywords for Psi4 are:
        Grimme D1
        Grimme D2
        Zero-damped Grimme D3
        Becke-Johnson damped Grimme D3
        Chai & Head-Gordon
        Nonlocal Approximation
        Pernal, Podeszwa, Patkowski, & Szalewicz
        Podeszwa, Katarzyna, Patkowski, & Szalewicz
        Řezác, Greenwell, & Beran"""
        if any(
                self.name.upper() == name for name in [
                    "GRIMME D1", "GD1", "D1", "-D1"
                ]
        ):
            return ("-d1", None)
        elif any(
                self.name.upper() == name for name in [
                    "GRIMME D2", "GD2", "D2", "-D2"
                ]
        ):
            return ("-d2", None)
        elif any(
                self.name.upper() == name for name in [
                    "ZERO-DAMPED GRIMME D3", "GRIMME D3", "GD3", "D3", "-D3", "D3ZERO"
                ]
        ):
            return ("-d3", None)
        elif any(
                self.name.upper() == name for name in [
                    "BECKE-JOHNSON DAMPED GRIMME D3", "GD3BJ", "D3BJ", "-D3BJ"
                ]
        ):
            return ("-d3bj", None)
        elif any(
                self.name.upper() == name for name in [
                    "BECKE-JOHNSON DAMPED MODIFIED GRIMME D3", "GD3MBJ", "D3MBJ", "-D3MBJ"
                ]
        ):
            return ("-d3mbj", None)
        elif any(
                self.name.upper() == name for name in [
                    "CHAI & HEAD-GORDON", "CHG", "-CHG"
                ]
        ):
            return ("-chg", None)
        elif any(
                self.name.upper() == name for name in [
                    "NONLOCAL APPROXIMATION", "NL", "NLA", "-NL"
                ]
        ):
            return ("-nl", None)
        elif any(
                self.name.upper() == name for name in [
                    "PERNAL, PODESZWA, PATKOWSKI, & SZALEWICZ", "DAS2009", "-DAS2009"
                ]
        ):
            return ("-das2009", None)
        elif any(
                self.name.upper() == name for name in [
                    "PODESZWA, KATARZYNA, PATKOWSKI, & SZALEWICZ", "DAS2010", "-DAS2010"
                ]
        ):
            return ("-das2010", None)
        elif any(
                self.name.upper() == name for name in [
                    "COUPLED-CLUSTER DOUBLES", "CCD"
                ]
        ):
            return ("(ccd)", None)
        elif any(
                self.name.upper() == name for name in [
                    "ŘEZÁC, GREENWELL, & BERAN", "DMP2"
                ]
        ):
            return ("dmp2", None)
        elif any(
                self.name.upper() == name for name in [
                    "COUPLED-CLUSTER DOUBLES + ŘEZÁC, GREENWELL, & BERAN", "(CCD)DMP2"
                ]
        ):
            return ("(ccd)dmp2", None)
        else:
            return (self.name, "unrecognized emperical dispersion: %s" % self.name)
