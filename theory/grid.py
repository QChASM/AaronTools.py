import re

from AaronTools.theory import GAUSSIAN_ROUTE, PSI4_SETTINGS, ORCA_ROUTE, ORCA_BLOCKS


class IntegrationGrid:
    """used to try to keep integration grid settings more easily when writing different input files"""
    def __init__(self, name):
        """name: str, gaussian keyword (e.g. SuperFineGrid), ORCA keyword (e.g. Grid7) or '(radial, angular)'
        ORCA can only use ORCA grid keywords
        Gaussian can use its keywords and will try to use ORCA keywords, and can use (\d+, \d+) or \d+
        Psi4 will use '(\d+, \d+)' and will try to use ORCA or Gaussian keywords"""
        self.name = name

    def get_gaussian(self):
        """gets gaussian integration grid info and a warning as tuple(dict, str or None)
        dict is of the form {GAUSSIAN_ROUTE:[x]}"""
        if self.name.lower() == "ultrafine":
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=UltraFine"]}}, None)
        elif self.name.lower() == "finegrid":
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=FineGrid"]}}, None)
        elif self.name.lower() == "superfinegrid":
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=SuperFineGrid"]}}, None)

        #Grids available in ORCA but not Gaussian
        #uses n_rad from K-Kr as specified in ORCA 4.2.1 manual (section 9.3)
        #XXX: there's probably IOp's that can get closer
        elif self.name.lower() == "grid2":
            n_rad = 45
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=%i110" % n_rad]}}, "Approximating ORCA Grid 2")
        elif self.name.lower() == "grid3":
            n_rad = 45
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=%i194" % n_rad]}}, "Approximating ORCA Grid 3")
        elif self.name.lower() == "grid4":
            n_rad = 45
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=%i302" % n_rad]}}, "Approximating ORCA Grid 4")
        elif self.name.lower() == "grid5":
            n_rad = 50
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=%i434" % n_rad]}}, "Approximating ORCA Grid 5")
        elif self.name.lower() == "grid6":
            n_rad = 55
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=%i590" % n_rad]}}, "Approximating ORCA Grid 6")
        elif self.name.lower() == "grid7":
            n_rad = 60
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=%i770" % n_rad]}}, "Approximating ORCA Grid 7")

        else:
            #grid format may be (int, int)
            #or just int
            match = re.match('\(\s*?(\d+)\s*?,\s*?(\d+)?\s*\)', self.name)
            match_digit = re.match('-?\d+?', self.name)
            if match:
                r = int(match.group(1))
                a = int(match.group(2))
                return({GAUSSIAN_ROUTE:{"Integral":["grid=%i%03i" % (r, a)]}}, None)
            elif match_digit:
                return({GAUSSIAN_ROUTE:{"Integral":["grid=%s" % self.name]}}, None)
            else:
                return ({GAUSSIAN_ROUTE:{"Integral":["grid=%s" % self.name]}}, "grid may not be available in Gaussian")

    def get_orca(self):
        """translates grid to something ORCA accepts
        returns tuple(dict(ORCA_ROUTE:[self.name]), None) or tuple(dict(ORCA_BLOCKS:{'method':[str]}), None)"""
        if self.name.lower() == "ultrafine":
            return ({ORCA_BLOCKS:{'method':['AngularGrid     Lebedev590', \
                                            #I did not check how close this IntAcc gets to the gaussian grids
                                            'IntAcc          4.0', \
                                           ]}},\
                     "approximating UltraFineGrid")
        elif self.name.lower() == "finegrid":
            return ({ORCA_BLOCKS:{'method':['AngularGrid     Lebedev302', \
                                            'IntAcc          4.0', \
                                           ]}},\
                     "approximating FineGrid")
        elif self.name.lower() == "superfinegrid":
            #radial is 175 for 1st row, 250 for later rows
            return ({ORCA_ROUTE:["Grid7", "FinalGrid7"]}, "could not set SuperFineGrid equivalent - using largest ORCA grid keyword")

        elif 'grid' in self.name.lower():
            #orca grid keyword
            return ({ORCA_ROUTE:[self.name, "Final%s" % self.name]}, None)


        else:
            #grid format may be (int, int)
            match = re.match('\(\s*?(\d+)\s*?,\s*?(\d+)?\s*\)', self.name)
            if match:
                r = int(match.group(1))
                a = int(match.group(2))
                int_acc = -((r / -5) + 2 - 8) / 3
                return ({ORCA_BLOCKS:{'method':['AngularGrid     Lebedev%i' % a, \
                                                'IntAcc          %.1f' % int_acc, \
                                               ]}},\
                        None)
            
            else:
                raise RuntimeError("could not determine acceptable Psi4 grid settings for %s" % self.name)

    def get_psi4(self):
        """returns ({PSI4_SETTINGS:{'dft_radial_points':['n'], 'dft_spherical_points':['m']}}, warning)"""
        if self.name.lower() == "ultrafine":
            return ({PSI4_SETTINGS:{'dft_radial_points':['99'], 'dft_spherical_points':['590']}}, None)
        elif self.name.lower() == "finegrid":
            return ({PSI4_SETTINGS:{'dft_radial_points':['75'], 'dft_spherical_points':['302']}}, None)
        elif self.name.lower() == "superfinegrid":
            #radial is 175 for 1st row, 250 for later rows
            return ({PSI4_SETTINGS:{'dft_radial_points':['250'], 'dft_spherical_points':['974']}}, "Approximating Gaussian SuperFineGrid")

        #uses radial from K-Kr as specified in ORCA 4.2.1 manual (section 9.3)
        elif self.name.lower() == "grid2":
            return ({PSI4_SETTINGS:{'dft_radial_points':['45'], 'dft_spherical_points':['110']}}, "Approximating ORCA Grid 2")
        elif self.name.lower() == "grid3":
            return ({PSI4_SETTINGS:{'dft_radial_points':['45'], 'dft_spherical_points':['194']}}, "Approximating ORCA Grid 3")
        elif self.name.lower() == "grid4":
            return ({PSI4_SETTINGS:{'dft_radial_points':['45'], 'dft_spherical_points':['302']}}, "Approximating ORCA Grid 4")
        elif self.name.lower() == "grid5":
            return ({PSI4_SETTINGS:{'dft_radial_points':['50'], 'dft_spherical_points':['434']}}, "Approximating ORCA Grid 5")
        elif self.name.lower() == "grid6":
            return ({PSI4_SETTINGS:{'dft_radial_points':['55'], 'dft_spherical_points':['590']}}, "Approximating ORCA Grid 6")
        elif self.name.lower() == "grid7":
            return ({PSI4_SETTINGS:{'dft_radial_points':['60'], 'dft_spherical_points':['770']}}, "Approximating ORCA Grid 7")

        else:
            #grid format may be (int, int)
            match = re.match('\(\s*?(\d+)\s*?,\s*?(\d+)?\s*\)', self.name)
            if match:
                r = int(match.group(1))
                a = int(match.group(2))
                return({PSI4_SETTINGS:{'dft_radial_points':[r], 'dft_spherical_points':[a]}}, None)
            
            else:
                raise RuntimeError("could not determine acceptable Psi4 grid settings for %s" % self.name)

