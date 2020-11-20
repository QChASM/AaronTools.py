from AaronTools.theory import (
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_BEFORE_GEOM,
    PSI4_JOB,
    PSI4_OPTKING,
    PSI4_SETTINGS,
)


class JobType:
    """parent class of all job types"""

    def __init__(self):
        pass

    def get_gaussian(self):
        """overwrite to return dict with GAUSSIAN_* keys"""
        pass

    def get_orca(self):
        """overwrite to return dict with ORCA_* keys"""
        pass

    def get_psi4(self):
        """overwrite to return dict with PSI4_* keys"""
        pass


class OptimizationJob(JobType):
    """optimization job"""

    def __init__(
        self, transition_state=False, constraints=None, geometry=None
    ):
        """use transition_state=True to do a TS optimization
        constraints - dict with 'atoms', 'bonds', 'angles' and 'torsions' as keys
                      constraints['atoms']: list(Atom) - atoms to constrain
                      constraints['bonds']: list(list(Atom, len=2)) - distances to constrain
                      constraints['angles']: list(list(Atom, len=3)) - 1-3 angles to constrain
                      constraints['torsions']: list(list(Atom, len=4)) - dihedral angles to constrain
        geometry    - Geoemtry, will be set when using an AaronTools FileWriter"""
        super().__init__()

        self.ts = transition_state
        self.constraints = constraints
        self.geometry = geometry

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE, GAUSSIAN_CONSTRAINTS"""
        if self.ts:
            out = {GAUSSIAN_ROUTE: {"Opt": ["ts"]}}
        else:
            out = {GAUSSIAN_ROUTE: {"Opt": []}}

        if self.constraints is not None and any(
            len(self.constraints[key]) > 0 for key in self.constraints.keys()
        ):
            out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")
            out[GAUSSIAN_CONSTRAINTS] = []

            if "atoms" in self.constraints:
                for atom in self.constraints["atoms"]:
                    ndx = self.geometry.atoms.index(atom) + 1
                    out[GAUSSIAN_CONSTRAINTS].append("%2i F" % ndx)

            if "bonds" in self.constraints:
                for constraint in self.constraints["bonds"]:
                    atom1, atom2 = constraint
                    ndx1 = self.geometry.atoms.index(atom1) + 1
                    ndx2 = self.geometry.atoms.index(atom2) + 1
                    out[GAUSSIAN_CONSTRAINTS].append(
                        "B %2i %2i F" % (ndx1, ndx2)
                    )

            if "angles" in self.constraints:
                for constraint in self.constraints["angles"]:
                    atom1, atom2, atom3 = constraint
                    ndx1 = self.geometry.atoms.index(atom1) + 1
                    ndx2 = self.geometry.atoms.index(atom2) + 1
                    ndx3 = self.geometry.atoms.index(atom3) + 1
                    out[GAUSSIAN_CONSTRAINTS].append(
                        "A %2i %2i %2i F" % (ndx1, ndx2, ndx3)
                    )

            if "torsions" in self.constraints:
                for constraint in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = constraint
                    ndx1 = self.geometry.atoms.index(atom1) + 1
                    ndx2 = self.geometry.atoms.index(atom2) + 1
                    ndx3 = self.geometry.atoms.index(atom3) + 1
                    ndx4 = self.geometry.atoms.index(atom4) + 1
                    out[GAUSSIAN_CONSTRAINTS].append(
                        "D %2i %2i %2i %2i F" % (ndx1, ndx2, ndx3, ndx4)
                    )

        return out

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE, ORCA_BLOCKS"""
        if self.ts:
            out = {ORCA_ROUTE: ["OptTS"]}
        else:
            out = {ORCA_ROUTE: ["Opt"]}

        if self.constraints is not None and any(
            len(self.constraints[key]) > 0 for key in self.constraints.keys()
        ):

            out[ORCA_BLOCKS] = {"geom": ["Constraints"]}
            if "atoms" in self.constraints:
                for constraint in self.constraints["atoms"]:
                    atom1 = constraint
                    ndx1 = self.geometry.atoms.index(atom1)
                    s = "    {C %2i C}" % (ndx1)
                    out[ORCA_BLOCKS]["geom"].append(s)

            if "bonds" in self.constraints:
                for constraint in self.constraints["bonds"]:
                    atom1, atom2 = constraint
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    s = "    {B %2i %2i C}" % (ndx1, ndx2)
                    out[ORCA_BLOCKS]["geom"].append(s)

            if "angles" in self.constraints:
                for constraint in self.constraints["angles"]:
                    atom1, atom2, atom3 = constraint
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    s = "    {A %2i %2i %2i C}" % (ndx1, ndx2, ndx3)
                    out[ORCA_BLOCKS]["geom"].append(s)

            if "torsions" in self.constraints:
                for constraint in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = constraint
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx4 = self.geometry.atoms.index(atom4)
                    s = "    {D %2i %2i %2i %2i C}" % (ndx1, ndx2, ndx3, ndx4)
                    out[ORCA_BLOCKS]["geom"].append(s)

            out[ORCA_BLOCKS]["geom"].append("end")

        return out

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB, PSI4_OPTKING, PSI4_BEFORE_GEOM"""
        if self.ts:
            out = {
                PSI4_JOB: {"optimize": []},
                PSI4_SETTINGS: {"opt_type": ["ts"]},
            }
        else:
            out = {PSI4_JOB: {"optimize": []}}

        # constraints
        if self.constraints is not None and any(
            [len(self.constraints[key]) > 0 for key in self.constraints.keys()]
        ):
            out[PSI4_OPTKING] = {}
            if (
                "atoms" in self.constraints
                and len(self.constraints["atoms"]) > 0
            ):
                s = ""
                if (
                    len(self.constraints["atoms"]) > 0
                    and self.geometry is not None
                ):
                    s += 'freeze_list = """\n'
                    for atom in self.constraints["atoms"]:
                        s += "    %2i xyz\n" % (
                            self.geometry.atoms.index(atom) + 1
                        )

                    s += '"""\n'
                    s += "    \n"

                out[PSI4_BEFORE_GEOM] = [s]

                out[PSI4_OPTKING]["frozen_cartesian"] = ["$freeze_list"]

            if "bonds" in self.constraints:
                if (
                    len(self.constraints["bonds"]) > 0
                    and self.geometry is not None
                ):
                    s = '("\n'
                    for bond in self.constraints["bonds"]:
                        atom1, atom2 = bond
                        s += "        %2i %2i\n" % (
                            self.geometry.atoms.index(atom1) + 1,
                            self.geometry.atoms.index(atom2) + 1,
                        )

                    s += '    ")\n'

                    out[PSI4_OPTKING]["frozen_distance"] = [s]

            if "angles" in self.constraints:
                if (
                    len(self.constraints["angles"]) > 0
                    and self.geometry is not None
                ):
                    s = '("\n'
                    for angle in self.constraints["angles"]:
                        atom1, atom2, atom3 = angle
                        s += "        %2i %2i %2i\n" % (
                            self.geometry.atoms.index(atom1) + 1,
                            self.geometry.atoms.index(atom2) + 1,
                            self.geometry.atoms.index(atom3) + 1,
                        )

                    s += '    ")\n'

                    out[PSI4_OPTKING]["frozen_bend"] = [s]

            if "torsions" in self.constraints:
                if (
                    len(self.constraints["torsions"]) > 0
                    and self.geometry is not None
                ):
                    s += '("\n'
                    for torsion in self.constraints["torsions"]:
                        atom1, atom2, atom3, atom4 = torsion
                        s += "        %2i %2i %2i %2i\n" % (
                            self.geometry.atoms.index(atom1) + 1,
                            self.geometry.atoms.index(atom2) + 1,
                            self.geometry.atoms.index(atom3) + 1,
                            self.geometry.atoms.index(atom4) + 1,
                        )

                    s += '    ")\n'

                    out[PSI4_OPTKING]["frozen_dihedral"] = [s]

        return out


class FrequencyJob(JobType):
    """frequnecy job"""

    def __init__(self, numerical=False, temperature=298.15):
        """temperature in K for thermochem info that gets printed in output file"""
        super().__init__()
        self.numerical = numerical
        self.temperature = temperature

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE"""
        out = {
            GAUSSIAN_ROUTE: {
                "Freq": ["temperature=%.2f" % float(self.temperature)]
            }
        }
        if self.numerical:
            out[GAUSSIAN_ROUTE]["Freq"].append("Numerical")

        return out

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        out = {ORCA_BLOCKS: {"freq": ["Temp    %.2f" % self.temperature]}}
        if self.numerical:
            out[ORCA_ROUTE] = ["NumFreq"]
        else:
            out[ORCA_ROUTE] = ["Freq"]

        return out

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        out = {
            PSI4_JOB: {"frequencies": []},
            PSI4_SETTINGS: {"T": ["%.2f" % self.temperature]},
        }
        if self.numerical:
            out[PSI4_JOB]["frequencies"].append('dertype="gradient"')

        return out


class SinglePointJob(JobType):
    """single point energy"""

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE"""
        return {GAUSSIAN_ROUTE: {"SP": []}}

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE: ["SP"]}

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        return {PSI4_JOB: {"energy": []}}


class ForceJob(JobType):
    """force/gradient job"""
    def __init__(self, numerical=False):
        super().__init__()
        self.numerical = numerical
    
    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE"""
        out = {GAUSSIAN_ROUTE:{"force":[]}}
        if self.numerical:
            out[GAUSSIAN_ROUTE]["force"].append("EnGrad")
        return out
    
    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE:["NumGrad" if self.numerical else "EnGrad"]}
    
    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        out = {PSI4_JOB:{"gradient":[]}}
        if self.numerical:
            out[PSI4_JOB]["gradient"].append("dertype='energy'")
        return out
        
