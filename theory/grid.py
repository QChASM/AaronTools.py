"""
used for specifying integration grids for DFT calculations
"""

import re
from AaronTools.theory import GAUSSIAN_ROUTE, PSI4_SETTINGS, ORCA_ROUTE, ORCA_BLOCKS


class IntegrationGrid:
    """
    used to try to keep integration grid settings more
    easily when writing different input files
    """
    def __init__(self, name):
        """
        name: str, Gaussian keyword (e.g. SuperFineGrid),
                   ORCA keyword (e.g. Grid7),
                   or "(radial, angular)"
        ORCA can only use ORCA grid keywords
        Gaussian can use its keywords and will try to use ORCA keywords, and
        can use "(radial, angular)" or radialangular
        Psi4 will use "(radial, angular)" and will try to use ORCA or Gaussian keywords
        """
        self.name = name

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        if self.name == other.name:
            return True
        return self.get_gaussian[0] == other.get_gaussian[0]

    def get_gaussian(self):
        """
        gets gaussian integration grid info and a warning as tuple(dict, str or None)
        dict is of the form {GAUSSIAN_ROUTE:[x]}
        """
        if self.name.lower() == "ultrafine":
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=UltraFine"]}}, None)

        elif self.name.lower() == "finegrid":
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=FineGrid"]}}, None)

        elif self.name.lower() == "superfinegrid":
            return ({GAUSSIAN_ROUTE:{"Integral":["grid=SuperFineGrid"]}}, None)

        # Grids available in ORCA but not Gaussian
        # uses n_rad from K-Kr as specified in ORCA 4.2.1 manual (section 9.3)
        elif self.name.lower() == "grid2":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=45110"]
                }
            }
            return (out_dict, "Approximating ORCA Grid 2")

        elif self.name.lower() == "grid3":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=45194"]
                }
            }
            return (out_dict, "Approximating ORCA Grid 3")

        elif self.name.lower() == "grid4":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=45302"]
                }
            }
            return (out_dict, "Approximating ORCA Grid 4")

        elif self.name.lower() == "grid5":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=50434"]
                }
            }
            return (out_dict, "Approximating ORCA Grid 5")

        elif self.name.lower() == "grid6":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=55590"]
                }
            }
            return (out_dict, "Approximating ORCA Grid 6")

        elif self.name.lower() == "grid7":
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=60770"]
                }
            }
            return (out_dict, "Approximating ORCA Grid 7")

        # grid format may be (int, int)
        # or just int
        match = re.match(r"\(\s*?(\d+)\s*?,\s*?(\d+)?\s*\)", self.name)
        match_digit = re.match(r"-?\d+?", self.name)
        if match:
            r_pts = int(match.group(1))
            a_pts = int(match.group(2))
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=%i%03i" % (r_pts, a_pts)]
                }
            }
            return (out_dict, None)

        elif match_digit:
            out_dict = {
                GAUSSIAN_ROUTE: {
                    "Integral":["grid=%s" % self.name]
                }
            }
            return (out_dict, None)

        out_dict = {
            GAUSSIAN_ROUTE: {
                "Integral":["grid=%s" % self.name]
            }
        }
        return (out_dict, "grid may not be available in Gaussian")

    def get_orca(self):
        """
        translates grid to something ORCA accepts
        returns tuple(dict(ORCA_ROUTE:[self.name]), None) or
        tuple(dict(ORCA_BLOCKS:{"method":[str]}), None)
        """
        if self.name.lower() == "ultrafine":
            out_dict = {
                ORCA_BLOCKS: {
                    "method": [
                        "AngularGrid     Lebedev590",
                        "IntAcc          4.0",
                    ]
                }
            }
            return (out_dict, "approximating UltraFineGrid")

        elif self.name.lower() == "finegrid":
            out_dict = {
                ORCA_BLOCKS: {
                    "method": [
                        "AngularGrid     Lebedev302",
                        "IntAcc          4.0",
                    ]
                }
            }

            return (out_dict, "approximating FineGrid")

        elif self.name.lower() == "superfinegrid":
            #radial is 175 for 1st row, 250 for later rows
            out_dict = {
                ORCA_ROUTE: [
                    "Grid7",
                    "FinalGrid7",
                ],
            }

            return (
                out_dict,
                "could not set SuperFineGrid equivalent - using largest ORCA grid keyword",
            )

        elif "grid" in self.name.lower():
            # orca grid keyword
            out_dict = {
                ORCA_ROUTE: [
                    self.name,
                    "Final%s" % self.name,
                ]
            }
            return (out_dict, None)

        # grid format may be (int, int)
        match = re.match(r"\(\s*?(\d+)\s*?,\s*?(\d+)?\s*\)", self.name)
        if match:
            r_pts = int(match.group(1))
            a_pts = int(match.group(2))
            int_acc = -((r_pts / -5) + 2 - 8) / 3
            out_dict = {
                ORCA_BLOCKS: {
                    "method": [
                        "AngularGrid     Lebedev%i" % a_pts,
                        "IntAcc          %.1f" % int_acc,
                    ]
                }
            }
            return (out_dict, None)

        raise RuntimeError(
            "could not determine acceptable Psi4 grid settings for %s" % self.name
        )

    def get_psi4(self):
        """
        returns ({PSI4_SETTINGS:{"dft_radial_points":["n"], "dft_spherical_points":["m"]}}, warning)
        """
        if self.name.lower() == "ultrafine":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": ["99"],
                    "dft_spherical_points": ["590"],
                }
            }
            return (out_dict, None)

        elif self.name.lower() == "finegrid":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": ["75"],
                    "dft_spherical_points": ["302"],
                }
            }
            return (out_dict, None)

        elif self.name.lower() == "superfinegrid":
            # radial is 175 for 1st row, 250 for later rows
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": ["250"],
                    "dft_spherical_points": ["974"],
                }
            }
            return (out_dict, "Approximating Gaussian SuperFineGrid")

        # uses radial from K-Kr as specified in ORCA 4.2.1 manual (section 9.3)
        elif self.name.lower() == "grid2":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points":["45"],
                    "dft_spherical_points": ["110"],
                }
            }
            return (out_dict, "Approximating ORCA Grid 2")

        elif self.name.lower() == "grid3":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": ["45"],
                    "dft_spherical_points":["194"],
                }
            }
            return (out_dict, "Approximating ORCA Grid 3")

        elif self.name.lower() == "grid4":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points":["45"],
                    "dft_spherical_points": ["302"],
                }
            }
            return (out_dict, "Approximating ORCA Grid 4")
 
        elif self.name.lower() == "grid5":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points":["50"],
                    "dft_spherical_points": ["434"],
                }
            }
            return (out_dict, "Approximating ORCA Grid 5")

        elif self.name.lower() == "grid6":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": ["55"],
                    "dft_spherical_points": ["590"],
                }
            }
            return (out_dict, "Approximating ORCA Grid 6")

        elif self.name.lower() == "grid7":
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": ["60"],
                    "dft_spherical_points": ["770"],
                }
            }
            return (out_dict, "Approximating ORCA Grid 7")

        # grid format may be (int, int)
        match = re.match(r"\(\s*?(\d+)\s*?,\s*?(\d+)?\s*\)", self.name)
        if match:
            r_pts = int(match.group(1))
            a_pts = int(match.group(2))
            out_dict = {
                PSI4_SETTINGS: {
                    "dft_radial_points": [r_pts],
                    "dft_spherical_points": [a_pts],
                }
            }
            return (out_dict, None)

        raise RuntimeError(
            "could not determine acceptable Psi4 grid settings for %s" % self.name
        )
