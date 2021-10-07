"""various job types for Theory() instances"""
import numpy as np

from AaronTools import addlogger
from AaronTools.theory import (
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_COORDINATES,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_BEFORE_GEOM,
    PSI4_COORDINATES,
    PSI4_JOB,
    PSI4_OPTKING,
    PSI4_SETTINGS,
    SQM_QMMM,
    QCHEM_REM,
    QCHEM_SETTINGS,
)
from AaronTools.utils.utils import range_list


class JobType:
    """
    parent class of all job types
    initialization keywords should be the same as attribute names
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self.get_psi4() == other.get_psi4()

    def get_gaussian(self):
        """overwrite to return dict with GAUSSIAN_* keys"""
        pass

    def get_orca(self):
        """overwrite to return dict with ORCA_* keys"""
        pass

    def get_psi4(self):
        """overwrite to return dict with PSI4_* keys"""
        pass
    
    def get_sqm(self):
        """overwrite to return a dict with SQM_* keys"""
        pass
    
    def get_qchem(self):
        """overwrite to return a dict with QCHEM_* keys"""
        pass

    @staticmethod
    def resolve_error(error, theory, exec_type, geometry=None):
        """returns a copy of theory or modifies theory to attempt
        to resolve an error
        theory will be modified if it is not possible for the current theory
        to work for any job
        if the error is specific to the molecule and settings, theory will
        be copied, modified, and returned
        raises NotImplementedError if this job type has no fix for
        the error code
        error: error code (e.g. SCF_CONV; see fileIO ERROR)
        theory: Theory() instance
        exec_type: software program (i.e. gaussian, orca, etc.)
        
        optional kwargs:
        geometry: Geometry(), structure might be adjusted slightly if
            there are close contacts
        """

        if error.upper() == "CLASH":
            # if there is a clash, rotate substituents to mitigate clashing
            if geometry:
                geom_copy = geometry.copy()
                bad_subs = geom_copy.remove_clash()
                if not bad_subs:
                    geometry.update_structure(geom_copy.coords)
                    return None

        if exec_type.lower() == "gaussian":
            if error.upper() == "SCF_CONV":
                # SCF convergence issue, try different SCF algorithm
                out_theory = theory.copy()
                out_theory.kwargs = combine_dicts(
                     {
                        GAUSSIAN_ROUTE: {"scf": ["xqc"]}
                    },
                    out_theory.kwargs
                )
                return out_theory

        if exec_type.lower() == "orca":
            if error.upper() == "SCF_CONV":
                # SCF convergence issue, orca recommends ! SlowConv
                # and increasing SCF iterations
                out_theory = theory.copy()
                out_theory.kwargs = combine_dicts(
                     {
                        ORCA_ROUTE: ["SlowConv"],
                        ORCA_BLOCKS: {"scf": ["MaxIter 500"]}
                    },
                    out_theory.kwargs
                )
                return out_theory
        
        raise NotImplementedError(
            "cannot fix %s errors for %s; check your input" % (error, exec_type)
        )


@addlogger
class OptimizationJob(JobType):
    """optimization job"""

    LOG = None

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

        self.transition_state = transition_state
        self.constraints = constraints
        self.geometry = geometry

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE, GAUSSIAN_CONSTRAINTS"""
        if self.transition_state:
            out = {GAUSSIAN_ROUTE: {"Opt": ["ts"]}}
        else:
            out = {GAUSSIAN_ROUTE: {"Opt": []}}

        coords = self.geometry.coords.tolist()
        vars = []
        consts = []
        use_zmat = False

        group_count = 1

        if self.constraints is not None and any(
            self.constraints[key] for key in self.constraints.keys()
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
                try:
                    atoms = self.geometry.find(self.constraints["atoms"])
                except LookupError as e:
                    self.LOG.warning(e)
                    atoms = []
                for atom in atoms:
                    ndx = self.geometry.atoms.index(atom) + 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append("%2i F" % ndx)
                    else:
                        for j, coord in enumerate(coords[ndx - 1]):
                            if isinstance(coord, str):
                                var_name = coord
                                for k, var in enumerate(vars):
                                    if var[0] == coord and not var[
                                        0
                                    ].startswith("g"):
                                        vars.pop(k)
                                        break
                                else:
                                    var_name = "%s%i" % (
                                        ["x", "y", "z"][j],
                                        ndx,
                                    )
                                    coords[ndx - 1][j] = var_name
                            else:
                                var_name = "%s%i" % (["x", "y", "z"][j], ndx)
                                coords[ndx - 1][j] = var_name
                            if not any(
                                const[0] == var_name for const in consts
                            ):
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
                            "cannot apply bond constraints when using Cartesian Z-Matrix, which"
                            + " is necessitated by x, y, or z constraints"
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
                            "cannot apply angle constraints when using Cartesian Z-Matrix, which"
                            + " is necessitated by x, y, or z constraints"
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
                            "cannot apply torsion constraints when using Cartesian Z-Matrix,"
                            + "which is necessitated by x, y, or z constraints"
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
        if self.transition_state:
            out = {ORCA_ROUTE: ["OptTS"]}
        else:
            out = {ORCA_ROUTE: ["Opt"]}

        if self.constraints is not None and any(
            self.constraints[key] for key in self.constraints.keys()
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
                    out_str = "    {D %2i %2i %2i %2i C}" % (
                        ndx1,
                        ndx2,
                        ndx3,
                        ndx4,
                    )
                    out[ORCA_BLOCKS]["geom"].append(out_str)

            out[ORCA_BLOCKS]["geom"].append("end")

        return out

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB, PSI4_OPTKING, PSI4_BEFORE_GEOM"""
        if self.transition_state:
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
        freeze_str += 'freeze_list = """\n'
        add_freeze_list = False

        # constraints
        if self.constraints is not None and any(
            [self.constraints[key] for key in self.constraints.keys()]
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
                "x" in self.constraints
                and self.constraints["x"]
                and self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["x"])
                for atom in atoms:
                    freeze_str += "    %2i x\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if (
                "y" in self.constraints
                and self.constraints["y"]
                and self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["y"])
                for atom in atoms:
                    freeze_str += "    %2i y\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if (
                "z" in self.constraints
                and self.constraints["z"]
                and self.geometry is not None
            ):
                add_freeze_list = True
                atoms = self.geometry.find(self.constraints["z"])
                for atom in atoms:
                    freeze_str += "    %2i z\n" % (
                        self.geometry.atoms.index(atom) + 1
                    )

            if (
                "atoms" in self.constraints
                and self.constraints["atoms"]
                and self.geometry is not None
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
                        for i, atom in enumerate(x_atoms):
                            freeze_str += "    %2i x\n" % (
                                self.geometry.atoms.index(atom) + 1
                            )
                            coords[i][0] = val
                    else:
                        vars.append([var_name, val, True])
                        for i, atom in enumerate(self.geometry.atoms):
                            if atom in x_atoms:
                                coords[i] = [
                                    coords[i][0],
                                    coords[i][1],
                                    var_name,
                                ]

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
                                coords[i] = [
                                    coords[i][0],
                                    coords[i][1],
                                    var_name,
                                ]

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
                                coords[i] = [
                                    coords[i][0],
                                    coords[i][1],
                                    var_name,
                                ]

            if add_freeze_list:
                freeze_str += '"""\n'
                freeze_str += "    \n"
                out[PSI4_BEFORE_GEOM] = [freeze_str]
                out[PSI4_OPTKING]["frozen_cartesian"] = ["$freeze_list"]

            if "bonds" in self.constraints:
                if self.constraints["bonds"] and self.geometry is not None:
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
                if self.constraints["angles"] and self.geometry is not None:
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
                if self.constraints["torsions"] and self.geometry is not None:
                    out_str += '("\n'
                    for torsion in self.constraints["torsions"]:
                        atom1, atom2, atom3, atom4 = self.geometry.find(
                            torsion
                        )
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

    def get_xcontrol(self, config, ref=None):
        """
        Generates xcontrol file constraints

        Returns: dict(xcontrol)
        """
        if ref is None:
            ref = "ref.xyz"
        xcontrol = ""
        # only put constraints in xcontrol file so this works with Crest also
        frozen = [i + 1 for i, a in enumerate(self.geometry) if bool(a.flag)]
        if frozen:
            frozen = range_list(frozen)
            xcontrol += "$fix\n"
            xcontrol += "  atoms: {}\n".format(frozen)
            xcontrol += "  freeze: {}\n".format(frozen)
        elif self.constraints:
            xcontrol += "$constrain\n"
            xcontrol += "  force constant={}\n".format(
                config["Job"].get("constrain_force", fallback="0.5")
            )
            xcontrol += "  reference={}\n".format(
                config["Job"].get("constrain_ref", fallback=ref)
            )
            constrained = set([])
            for bond in self.constraints.get("bonds", []):
                bond = self.geometry.find(bond)
                constrained.update(bond)
                xcontrol += "  distance: {},{},auto\n".format(
                    *(self.geometry.atoms.index(c) + 1 for c in bond)
                )
            for angle in self.constraints.get("angles", []):
                angle = self.geometry.find(angle)
                constrained.update(angle)
                xcontrol += "  angle: {},{},{},auto\n".format(
                    *(self.geometry.atoms.index(c) + 1 for c in angle)
                )
            for dihedral in self.constraints.get("torsions", []):
                dihedral = self.geometry.find(dihedral)
                constrained.update(dihedral)
                xcontrol += "  dihedral: {},{},{},{},auto\n".format(
                    *(self.geometry.atoms.index(c) + 1 for c in dihedral)
                )
            relaxed = {
                i + 1
                for i, a in enumerate(self.geometry.atoms)
                if a not in constrained
            }
            relaxed = range_list(relaxed)
            xcontrol += "$metadyn\n"
            xcontrol += "  atoms: {}\n".format(relaxed)
        xcontrol += "$end\n"
        return xcontrol

    def get_sqm(self):
        """returns a dict(), warnings for optimization jobs"""
        warnings = []
        if self.transition_state:
            warnings.append("cannot do TS optimization with sqm")
        
        if self.constraints:
            warnings.append("cannot constrain sqm optimization")
        
        return dict(), warnings

    def get_qchem(self):
        if self.transition_state:
            out = {QCHEM_REM: {"JOB_TYPE": "TS"}}
        else:
            out = {QCHEM_REM: {"JOB_TYPE": "OPT"}}
        
        # constraints
        if self.constraints is not None and any(
            [self.constraints[key] for key in self.constraints.keys()]
        ):
            out[QCHEM_SETTINGS] = {"opt": []}
            constraints = None
            fixed = None

            x_atoms = []
            y_atoms = []
            z_atoms = []
            xyz_atoms = []

            if "x" in self.constraints:
                x_atoms = self.geometry.find(self.constraints["x"])


            if "y" in self.constraints:
                y_atoms = self.geometry.find(self.constraints["y"])


            if "z" in self.constraints:
                z_atoms = self.geometry.find(self.constraints["z"])

            if "atoms" in self.constraints:
                xyz_atoms = self.geometry.find(self.constraints["atoms"])

            if any([x_atoms, y_atoms, z_atoms, xyz_atoms]):
                fixed = "FIXED"
                for atom in x_atoms:
                    fixed += "\n        %i X" % (self.geometry.atoms.index(atom) + 1)
                    if atom in xyz_atoms:
                        fixed += "YZ"
                        continue
                    if atom in y_atoms:
                        fixed += "Y"
                    if atom in z_atoms:
                        fixed += "Z"
                
                for atom in y_atoms:
                    if atom in x_atoms:
                        continue
                    fixed += "\n        %i " % (self.geometry.atoms.index(atom) + 1)
                    if atom in xyz_atoms:
                        fixed += "XYZ"
                        continue
                    fixed += "Y"
                    if atom in z_atoms:
                        fixed += "Z"
                
                for atom in y_atoms:
                    if atom in x_atoms or atom in y_atoms:
                        continue
                    fixed += "\n        %i " % (self.geometry.atoms.index(atom) + 1)
                    if atom in xyz_atoms:
                        fixed += "XYZ"
                        continue
                    fixed += "Z"

                for atom in xyz_atoms:
                    if any(atom in l for l in [x_atoms, y_atoms, z_atoms]):
                        continue
                    fixed += "\n        %i XYZ" % (self.geometry.atoms.index(atom) + 1)

            if "bonds" in self.constraints:
                if constraints is None:
                    constraints = "CONSTRAINT\n"
                for bond in self.constraints["bonds"]:
                    atom1, atom2 = self.geometry.find(bond)
                    constraints += "        STRE %2i %2i %9.5f\n" % (
                        self.geometry.atoms.index(atom1) + 1,
                        self.geometry.atoms.index(atom2) + 1,
                        atom1.dist(atom2),
                    )
            
            if "angles" in self.constraints:
                if constraints is None:
                    constraints = "CONSTRAINT\n"
                for angle in self.constraints["angles"]:
                    atom1, atom2, atom3 = self.geometry.find(angle)
                    constraints += "        STRE %2i %2i %2i %9.5f\n" % (
                        self.geometry.atoms.index(atom1) + 1,
                        self.geometry.atoms.index(atom2) + 1,
                        self.geometry.atoms.index(atom3) + 1,
                        np.rad2deg(atom2.angle(atom1, atom3)),
                    )
            
            if "torsions" in self.constraints:
                if constraints is None:
                    constraints = "CONSTRAINT\n"
                for angle in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = self.geometry.find(angle)
                    constraints += "        TORS %2i %2i %2i %2i %9.5f\n" % (
                        self.geometry.atoms.index(atom1) + 1,
                        self.geometry.atoms.index(atom2) + 1,
                        self.geometry.atoms.index(atom3) + 1,
                        self.geometry.atoms.index(atom4) + 1,
                        np.rad2deg(
                            self.geometry.dihedral(
                                atom1, atom2, atom3, atom4,
                            )
                        ),
                    )
            
            
            if fixed:
                fixed += "\n    ENDFIXED"
                out[QCHEM_SETTINGS]["opt"].append(fixed)

            if constraints:
                constraints += "    ENDCONSTRAINT"
                out[QCHEM_SETTINGS]["opt"].append(constraints)
        
        return out

    @staticmethod
    def resolve_error(error, theory, exec_type, geometry=None):
        """
        resolves optimization-specific errors
        errors resolved by JobType take priority

        optional kwargs:
        geometry: Geometry(), structure might be adjusted slightly if
            the software had an issue with generating internal coordinates
        """
        try:
            return super(OptimizationJob, OptimizationJob).resolve_error(
                error, theory, exec_type, geometry=geometry
            )
        except NotImplementedError:
            pass
        
        if exec_type.lower() == "gaussian":
            if error.upper() == "CONV_LINK":
                # optimization out of steps, add more steps
                out_theory = theory.copy()
                out_theory.kwargs = combine_dicts(
                     {GAUSSIAN_ROUTE: {"opt": ["MaxCycles=300"]}}, out_theory.kwargs,
                )
                return out_theory
            
            if error.upper() == "FBX":
                # FormBX error, just restart the job
                # adjusting the geometry slightly can help
                if geometry:
                    coords = geometry.coords
                    scale = 1e-3
                    coords += scale * np.random.random_sample - scale / 2
                    geometry.update_structure(coords)
                return None
            
            if error.upper() == "REDUND":
                # internal coordinate error, just restart the job
                if geometry:
                    coords = geometry.coords
                    scale = 1e-3
                    coords += scale * np.random.random_sample - scale / 2
                    geometry.update_structure(coords)
                return None                
        
        if exec_type.lower() == "orca":
            if error.upper() == "OPT_CONV":
                # optimization out of steps, add more steps
                out_theory = theory.copy()
                out_theory.kwargs = combine_dicts(
                     {ORCA_BLOCKS: {"geom": ["MaxIter 300"]}}, out_theory.kwargs,
                )
                return out_theory
        
        raise NotImplementedError(
            "cannot fix %s errors for %s; check your input" % (error, exec_type)
        )


class FrequencyJob(JobType):
    """frequnecy job"""

    def __init__(self, numerical=False, temperature=None):
        """
        temperature in K for thermochem info, defaults to 298.15 K
        """
        super().__init__()
        if temperature is None:
            temperature = 298.15
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

    def get_sqm(self):
        raise NotImplementedError("cannot build frequnecy job input for sqm")

    def get_qchem(self):
        out = {QCHEM_REM: {"JOB_TYPE": "Freq"}}
        if self.numerical:
            out[QCHEM_REM]["FD_DERIVATIVE_TYPE"] = "1"

        return out

    @staticmethod
    def resolve_error(error, theory, exec_type, geometry=None):
        """
        resolves frequnecy-specific errors
        errors resolved by JobType take priority
        """
        try:
            return super(FrequencyJob, FrequencyJob).resolve_error(
                error, theory, exec_type, geometry=geometry
            )
        except NotImplementedError:
            pass
        
        if exec_type.lower() == "orca":
            if error.upper() == "NUMFREQ":
                # analytical derivatives are not available
                for job in theory.job_type:
                    if isinstance(job, FrequencyJob):
                        job.numerical = True
                return None
        
        raise NotImplementedError(
            "cannot fix %s errors for %s; check your input" % (error, exec_type)
        )


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
    
    def get_sqm(self):
        """returns a dict with keys: SQM_QMMM"""
        return {SQM_QMMM: {"maxcyc": ["0"]}}

    def get_qchem(self):
        out = {QCHEM_REM: {"JOB_TYPE": "SP"}}
        
        return out


class ForceJob(JobType):
    """force/gradient job"""

    def __init__(self, numerical=False):
        super().__init__()
        self.numerical = numerical

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE"""
        out = {GAUSSIAN_ROUTE: {"force": []}}
        if self.numerical:
            out[GAUSSIAN_ROUTE]["force"].append("EnGrad")
        return out

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE: ["NumGrad" if self.numerical else "EnGrad"]}

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        out = {PSI4_JOB: {"gradient": []}}
        if self.numerical:
            out[PSI4_JOB]["gradient"].append("dertype='energy'")
        return out

    def get_qchem(self):
        out = {QCHEM_REM: {"JOB_TYPE": "Force"}}
        
        return out
