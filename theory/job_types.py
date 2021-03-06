"""various job types for Theory() instances"""

from AaronTools.theory import (
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_COORDINATES,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_BEFORE_GEOM,
    PSI4_JOB,
    PSI4_COORDINATES,
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
            self,
            transition_state=False,
            constraints=None,
            geometry=None,
    ):
        """use transition_state=True to do a TS optimization
        constraints - dict with keys:
                      **** available for ORCA, Gaussian, and Psi4 ****
                      'atoms': atom identifiers/finders - atoms to constrain
                      'bonds': list(atom idenifiers/finders) - distances to constrain
                               each atom identifier in the list should result in exactly 2 atoms
                      'angles': list(atom idenifiers/finders) - 1-3 angles to constrain
                                each atom identifier should result in exactly 3 atoms
                      'torsions': list(atom identifiers/finders) - constrained dihedral angles
                                  each atom identifier should result in exactly 4 atoms
                      **** available for Gaussian and Psi4 ****
                      'x': list(atom identifiers/finders) - constrain the x coordinate of
                           these atoms
                      similarly, 'y' and 'z' are also accepted
                      'xgroup': list(tuple(list(atom idenifiers), x_val, hold)) -
                            constrain the x coordinate of these atoms to be the same
                            x_val - set x-coordinate to this value
                            hold - hold this value constant during the optimization
                                   if 'hold' is omitted, the value will not be held
                                   constant during the optimization
                            e.g. 'xgroup':[("1-6", 0, False), ("13-24", 3.25, False)]
                            this will keep atoms 1-6 and 13-24 in parallel planes, while also
                            allowing those planes to move
                      'ygroup' and 'zgroup' are also available, with analagous options

                      *** NOTE ***
                      for Gaussian, 'bonds', 'angles', and 'torsions' constraints cannot be mixed
                      with 'x', 'y', 'z', 'xgroup', 'ygroup', or 'zgroup' constraints
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

        coords = self.geometry.coords.tolist()
        vars = []
        consts = []
        use_zmat = False

        group_count = 1

        if (
                self.constraints is not None and
                any(
                    self.constraints[key] for key in self.constraints.keys()
                )
        ):
            for key in self.constraints:
                if key not in [
                        "x",
                        "y",
                        "z",
                        "xgroup",
                        "ygroup",
                        "zgroup",
                        "atoms",
                        "bonds",
                        "angles",
                        "torsions",
                ]:
                    raise NotImplementedError(
                        "%s constraints cannot be generated for Gaussian" % key
                    )
            out[GAUSSIAN_CONSTRAINTS] = []

            if "x" in self.constraints and self.constraints["x"]:
                x_atoms = self.geometry.find(self.constraints["x"])
                for i, atom in enumerate(self.geometry.atoms):
                    if atom in x_atoms:
                        var_name = "x%i" % (i + 1)
                        consts.append((var_name, atom.coords[0]))
                        coords[i] = [var_name, coords[i][1], coords[i][2]]

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "y" in self.constraints and self.constraints["y"]:
                y_atoms = self.geometry.find(self.constraints["y"])
                for i, atom in enumerate(self.geometry.atoms):
                    if atom in y_atoms:
                        var_name = "y%i" % (i + 1)
                        consts.append((var_name, atom.coords[1]))
                        coords[i] = [coords[i][0], var_name, coords[i][2]]

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "z" in self.constraints and self.constraints["z"]:
                z_atoms = self.geometry.find(self.constraints["z"])
                for i, atom in enumerate(self.geometry.atoms):
                    if atom in z_atoms:
                        var_name = "z%i" % (i + 1)
                        consts.append((var_name, atom.coords[2]))
                        coords[i] = [coords[i][0], coords[i][1], var_name]

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "xgroup" in self.constraints:
                for constraint in self.constraints["xgroup"]:
                    if len(constraint) == 3:
                        finders, val, hold = constraint
                    else:
                        finders, val = constraint
                        hold = False
                    x_atoms = self.geometry.find(finders)
                    var_name = "gx%i" % group_count
                    group_count += 1
                    if hold:
                        consts.append([var_name, val])
                    else:
                        vars.append([var_name, val])
                    for i, atom in enumerate(self.geometry.atoms):
                        if atom in x_atoms:
                            coords[i] = [var_name, coords[i][1], coords[i][2]]

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "ygroup" in self.constraints:
                for constraint in self.constraints["ygroup"]:
                    if len(constraint) == 3:
                        finders, val, hold = constraint
                    else:
                        finders, val = constraint
                        hold = False
                    y_atoms = self.geometry.find(finders)
                    var_name = "gy%i" % group_count
                    group_count += 1
                    if hold:
                        consts.append([var_name, val])
                    else:
                        vars.append([var_name, val])
                    for i, atom in enumerate(self.geometry.atoms):
                        if atom in y_atoms:
                            coords[i] = [coords[i][0], var_name, coords[i][2]]

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "zgroup" in self.constraints:
                for constraint in self.constraints["zgroup"]:
                    if len(constraint) == 3:
                        finders, val, hold = constraint
                    else:
                        finders, val = constraint
                        hold = False
                    z_atoms = self.geometry.find(finders)
                    var_name = "gz%i" % group_count
                    group_count += 1
                    if hold:
                        consts.append([var_name, val])
                    else:
                        vars.append([var_name, val])
                    for i, atom in enumerate(self.geometry.atoms):
                        if atom in z_atoms:
                            coords[i] = [coords[i][0], coords[i][1], var_name]

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "atoms" in self.constraints and self.constraints["atoms"]:
                atoms = self.geometry.find(self.constraints["atoms"])
                for atom in atoms:
                    ndx = self.geometry.atoms.index(atom) + 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append("%2i F" % ndx)
                    else:
                        for j, coord in enumerate(coords[ndx - 1]):
                            if isinstance(coord, str):
                                var_name = coord
                                for k, var in enumerate(vars):
                                    if var[0] == coord and not var[0].startswith("g"):
                                        vars.pop(k)
                                        break
                                else:
                                    var_name = "%s%i" % (["x" ,"y", "z"][j], ndx)
                                    coords[ndx - 1][j] = var_name
                            else:
                                var_name = "%s%i" % (["x" ,"y", "z"][j], ndx)
                                coords[ndx - 1][j] = var_name
                            if not any(const[0] == var_name for const in consts):
                                consts.append([var_name, atom.coords[j]])

                if not use_zmat:
                    if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                        out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")

            if "bonds" in self.constraints:
                for constraint in self.constraints["bonds"]:
                    atom1, atom2 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1) + 1
                    ndx2 = self.geometry.atoms.index(atom2) + 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append(
                            "B %2i %2i F" % (ndx1, ndx2)
                        )
                    else:
                        raise NotImplementedError(
                            "cannot apply bond constraints when using Cartesian Z-Matrix, which" +
                            " is necessitated by x, y, or z constraints"
                        )

                if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                    out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")

            if "angles" in self.constraints:
                for constraint in self.constraints["angles"]:
                    atom1, atom2, atom3 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1) + 1
                    ndx2 = self.geometry.atoms.index(atom2) + 1
                    ndx3 = self.geometry.atoms.index(atom3) + 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append(
                            "A %2i %2i %2i F" % (ndx1, ndx2, ndx3)
                        )
                    else:
                        raise NotImplementedError(
                            "cannot apply angle constraints when using Cartesian Z-Matrix, which" +
                            " is necessitated by x, y, or z constraints"
                        )

                if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                    out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")

            if "torsions" in self.constraints:
                for constraint in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1) + 1
                    ndx2 = self.geometry.atoms.index(atom2) + 1
                    ndx3 = self.geometry.atoms.index(atom3) + 1
                    ndx4 = self.geometry.atoms.index(atom4) + 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append(
                            "D %2i %2i %2i %2i F" % (ndx1, ndx2, ndx3, ndx4)
                        )
                    else:
                        raise NotImplementedError(
                            "cannot apply torsion constraints when using Cartesian Z-Matrix," +
                            "which is necessitated by x, y, or z constraints"
                        )

                if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                    out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")

        if consts or vars:
            for i, coord in enumerate(coords):
                for j, ax in enumerate(["x", "y", "z"]):
                    if isinstance(coord[j], float):
                        var_name = "%s%i" % (ax, i + 1)
                        vars.append((var_name, coord[j]))
                        coord[j] = var_name

        if consts or vars:
            for coord in coords:
                coord.insert(0, 0)

        out[GAUSSIAN_COORDINATES] = {
            "coords": coords,
            "variables": vars,
            "constants": consts,
        }

        return out

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE, ORCA_BLOCKS"""
        if self.ts:
            out = {ORCA_ROUTE: ["OptTS"]}
        else:
            out = {ORCA_ROUTE: ["Opt"]}

        if (
                self.constraints is not None and
                any(
                    self.constraints[key] for key in self.constraints.keys()
                )
        ):
            for key in self.constraints:
                if key not in [
                        "atoms",
                        "bonds",
                        "angles",
                        "torsions",
                ]:
                    raise NotImplementedError(
                        "%s constraints cannot be generated for ORCA" % key
                    )
            out[ORCA_BLOCKS] = {"geom": ["Constraints"]}
            if "atoms" in self.constraints:
                for constraint in self.constraints["atoms"]:
                    atom1 = self.geometry.find(constraint)[0]
                    ndx1 = self.geometry.atoms.index(atom1)
                    out_str = "    {C %2i C}" % (ndx1)
                    out[ORCA_BLOCKS]["geom"].append(out_str)

            if "bonds" in self.constraints:
                for constraint in self.constraints["bonds"]:
                    atom1, atom2 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    out_str = "    {B %2i %2i C}" % (ndx1, ndx2)
                    out[ORCA_BLOCKS]["geom"].append(out_str)

            if "angles" in self.constraints:
                for constraint in self.constraints["angles"]:
                    atom1, atom2, atom3 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    out_str = "    {A %2i %2i %2i C}" % (ndx1, ndx2, ndx3)
                    out[ORCA_BLOCKS]["geom"].append(out_str)

            if "torsions" in self.constraints:
                for constraint in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx4 = self.geometry.atoms.index(atom4)
                    out_str = "    {D %2i %2i %2i %2i C}" % (ndx1, ndx2, ndx3, ndx4)
                    out[ORCA_BLOCKS]["geom"].append(out_str)

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

        coords = self.geometry.coords.tolist()
        vars = []
        group_count = 1

        freeze_str = ""
        freeze_str += "freeze_list = \"\"\"\n"
        add_freeze_list = False

        # constraints
        if (
                self.constraints is not None and
                any(
                    [self.constraints[key] for key in self.constraints.keys()]
                )
        ):
            for key in self.constraints:
                if key not in [
                        "x",
                        "y",
                        "z",
                        "xgroup",
                        "ygroup",
                        "zgroup",
                        "atoms",
                        "bonds",
                        "angles",
                        "torsions",
                ]:
                    raise NotImplementedError(
                        "%s constraints cannot be generated for Psi4" % key
                    )
            out[PSI4_OPTKING] = {}
            if (
                    "x" in self.constraints and
                    self.constraints["x"] and
                    self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["x"])
                for atom in atoms:
                    freeze_str += "    %2i x\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if (
                    "y" in self.constraints and
                    self.constraints["y"] and
                    self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["y"])
                for atom in atoms:
                    freeze_str += "    %2i y\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if (
                    "z" in self.constraints and
                    self.constraints["z"] and
                    self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["z"])
                for atom in atoms:
                    freeze_str += "    %2i z\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if (
                    "atoms" in self.constraints and
                    self.constraints["atoms"] and
                    self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["atoms"])
                for atom in atoms:
                    freeze_str += "    %2i xyz\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if "xgroup" in self.constraints:
                for constraint in self.constraints["xgroup"]:
                    if len(constraint) == 3:
                        finders, val, hold = constraint
                    else:
                        finders, val = constraint
                        hold = False
                    x_atoms = self.geometry.find(finders)
                    var_name = "gx%i" % group_count
                    group_count += 1
                    if hold:
                        add_freeze_list = True
                        for i, atom in enumerate(y_atoms):
                            freeze_str += "    %2i x\n" % (
                                self.geometry.atoms.index(atom) + 1
                            )
                            coords[i][0] = val
                    else:
                        vars.append([var_name, val, True])
                        for i, atom in enumerate(self.geometry.atoms):
                            if atom in x_atoms:
                                coords[i] = [coords[i][0], coords[i][1], var_name]

            if "ygroup" in self.constraints:
                for constraint in self.constraints["ygroup"]:
                    if len(constraint) == 3:
                        finders, val, hold = constraint
                    else:
                        finders, val = constraint
                        hold = False
                    y_atoms = self.geometry.find(finders)
                    var_name = "gy%i" % group_count
                    group_count += 1
                    if hold:
                        add_freeze_list = True
                        for i, atom in enumerate(y_atoms):
                            freeze_str += "    %2i y\n" % (
                                self.geometry.atoms.index(atom) + 1
                            )
                            coords[i][1] = val
                    else:
                        vars.append([var_name, val, True])
                        for i, atom in enumerate(self.geometry.atoms):
                            if atom in y_atoms:
                                coords[i] = [coords[i][0], coords[i][1], var_name]

            if "zgroup" in self.constraints:
                for constraint in self.constraints["zgroup"]:
                    if len(constraint) == 3:
                        finders, val, hold = constraint
                    else:
                        finders, val = constraint
                        hold = False
                    z_atoms = self.geometry.find(finders)
                    var_name = "gz%i" % group_count
                    group_count += 1
                    if hold:
                        add_freeze_list = True
                        for i, atom in enumerate(z_atoms):
                            freeze_str += "    %2i z\n" % (
                                self.geometry.atoms.index(atom) + 1
                            )
                            coords[i][2] = val
                    else:
                        vars.append([var_name, val, True])
                        for i, atom in enumerate(self.geometry.atoms):
                            if atom in z_atoms:
                                coords[i] = [coords[i][0], coords[i][1], var_name]

            if add_freeze_list:
                freeze_str += '"""\n'
                freeze_str += "    \n"
                out[PSI4_BEFORE_GEOM] = [freeze_str]
                out[PSI4_OPTKING]["frozen_cartesian"] = ["$freeze_list"]

            if "bonds" in self.constraints:
                if (
                        self.constraints["bonds"]
                        and self.geometry is not None
                ):
                    out_str = '("\n'
                    for bond in self.constraints["bonds"]:
                        atom1, atom2 = self.geometry.find(bond)
                        out_str += "        %2i %2i\n" % (
                            self.geometry.atoms.index(atom1) + 1,
                            self.geometry.atoms.index(atom2) + 1,
                        )

                    out_str += '    ")\n'

                    out[PSI4_OPTKING]["frozen_distance"] = [out_str]

            if "angles" in self.constraints:
                if (
                        self.constraints["angles"]
                        and self.geometry is not None
                ):
                    out_str = '("\n'
                    for angle in self.constraints["angles"]:
                        atom1, atom2, atom3 = self.geometry.find(angle)
                        out_str += "        %2i %2i %2i\n" % (
                            self.geometry.atoms.index(atom1) + 1,
                            self.geometry.atoms.index(atom2) + 1,
                            self.geometry.atoms.index(atom3) + 1,
                        )

                    out_str += '    ")\n'

                    out[PSI4_OPTKING]["frozen_bend"] = [out_str]

            if "torsions" in self.constraints:
                if (
                        self.constraints["torsions"]
                        and self.geometry is not None
                ):
                    out_str += '("\n'
                    for torsion in self.constraints["torsions"]:
                        atom1, atom2, atom3, atom4 = self.geometry.find(torsion)
                        out_str += "        %2i %2i %2i %2i\n" % (
                            self.geometry.atoms.index(atom1) + 1,
                            self.geometry.atoms.index(atom2) + 1,
                            self.geometry.atoms.index(atom3) + 1,
                            self.geometry.atoms.index(atom4) + 1,
                        )

                    out_str += '    ")\n'

                    out[PSI4_OPTKING]["frozen_dihedral"] = [out_str]

        if vars:
            out[PSI4_COORDINATES] = {
                "coords": coords,
                "variables": vars,
            }

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
