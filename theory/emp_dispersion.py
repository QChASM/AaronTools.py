from AaronTools.theory import GAUSSIAN_ROUTE

class EmpiricalDispersion:
    """try to keep emerpical dispersion keywords and settings consistent across file types"""
    def __init__(self, name):
        """name can be (availability may vary):
            Grimme D2
            Zero-damped Grimme D3
            Becke-Johnson damped Grimme D3
            Becke-Johnson damped modified Grimme D3
            Petersson-Frisch
            Grimme D4
            Chai & Head-Gordon
            Nonlocal Approximation
            Pernal, Podeszwa, Patkowski, & Szalewicz
            Podeszwa, Katarzyna, Patkowski, & Szalewicz
            Řezác, Greenwell, & Beran
        
        or simply the keyword for the input file type you are using"""
        
        self.name = name

    def get_gaussian(self):
        """Acceptable dispersion methods for Gaussian are:
        Grimme D2
        Grimme D3
        Becke-Johnson damped Grimme D3
        Petersson-Frisch
        
        Dispersion methods available in other software that will be modified are:
        Grimme D4
        undampened Grimme D3"""
        
        if self.name == "Grimme D2":
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD2"]}}, None)
        elif self.name == "Zero-damped Grimme D3":
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3"]}}, None)
        elif self.name == "Becke-Johnson damped Grimme D3":
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3BJ"]}}, None)
        elif self.name == "Petersson-Frisch":
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["PFD"]}}, None)
            
        #dispersions in ORCA but not Gaussian
        elif self.name == "Grimme D4":
            return ({GAUSSIAN_ROUTE:{"EmpiricalDispersion":["GD3BJ"]}}, "Grimme's D4 has no keyword in Gaussian, switching to GD3BJ")

        #unrecognized
        else:
            return (self.name, "unrecognized emperical dispersion: %s" % self.name)

    def get_orca(self):
        """Acceptable keywords for ORCA are:
        Grimme D2
        Zero-damped Grimme D3"
        Becke-Johnson damped Grimme D3
        Grimme D4"""
        if self.name == "Grimme D2":
            return ("D2", None)
        elif self.name == "Zero-damped Grimme D3":
            return ("D3", None)
        elif self.name == "Becke-Johnson damped Grimme D3":
            return ("D3BJ", None)
        elif self.name == "Grimme D4":
            return ("D4", None)
        else:
            return (self.name, "unrecognized emperical dispersion: %s" % self.name)

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
        else:
            return (self.name, "unrecognized emperical dispersion: %s" % self.name)

