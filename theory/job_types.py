"""various job types for Theory() instances"""
import numpy as np

from AaronTools import addlogger
from AaronTools.const import ELEMENTS
from AaronTools.finders import FlaggedAtoms
from AaronTools.theory import (
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_COORDINATES,
    GAUSSIAN_ROUTE,
    GAUSSIAN_POST,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_BEFORE_GEOM,
    PSI4_COORDINATES,
    PSI4_JOB,
    PSI4_AFTER_JOB,
    PSI4_OPTKING,
    PSI4_SETTINGS,
    PSI4_BEFORE_JOB,
    SQM_QMMM,
    QCHEM_REM,
    QCHEM_SETTINGS,
    XTB_CONTROL_BLOCKS,
    XTB_COMMAND_LINE,
)
from AaronTools.utils.utils import range_list, combine_dicts


def job_from_string(name, **kwargs):
    """
    get a job name given a simple name
    
    :param str name:
    
        * "opt" or "conf" with ".ts", ".transition_state", ".change", and ".con" extensions

            * .ts and .transition_state indicate a transition state optimization
            * .con indicates a constrained optimization - "constraints" should
              be in kwargs and the value should be a dictionary conformable with 
              the keyword of OptimizationJob
        
        * "freq" with ".num" extensions
        
            * .num indicates a numerical frequnecy, as does kwargs["numerical"] = True
              kwargs can also have a "temperature" key
        
        * "sp" or "energy" or "single-point"
        * "force" or "gradient" with a ".num" extension
        
            * .num indicates a numerical frequnecy, as does kwargs["numerical"] = True

    :returns: job types for the given string
    """
    ext = None
    if "." in name:
        ext = name.split(".")[-1].lower()
  
    if name.lower().startswith("opt"):
        geom = kwargs.get("geometry", None)
        constraints = kwargs.get("constraints", None)

        if ext and (ext.startswith("ts") or ext.startswith("transition")):
            return OptimizationJob(transition_state=True, geometry=geom)
        
        if ext and ext.startswith("con") and constraints:
            return OptimizationJob(geometry=geom, constraints=constraints)
        
        if ext and ext.startswith("change"):
            return OptimizationJob(
                constraints={"atoms": FlaggedAtoms()}, geometry=geom
            )
        return OptimizationJob(geometry=geom)
    
    if name.lower().startswith("conf"):
        geom = kwargs.get("geometry", None)
        constraints = kwargs.get("constraints", None)
        
        if ext and ext.startswith("con") and constraints:
            return ConformerSearchJob(geometry=geom, constraints=constraints)

        return ConformerSearchJob(geometry=geom)
    
    if name.lower().startswith("freq"):
        numerical = kwargs.get("numerical", False)
        numerical = numerical or (ext and ext.startswith("num"))
        temperature = kwargs.get("temperature", 298.15)
        
        return FrequencyJob(numerical=numerical, temperature=temperature)
    
    if any(name.lower().startswith(x) for x in ["sp", "energy", "single-point"]):
        return SinglePointJob()
    
    if any(name.lower().startswith(x) for x in ["force", "gradient"]):
        numerical = kwargs.get("numerical", False)
        numerical = numerical or (ext and ext.startswith("num"))
        
        return ForceJob(numerical=numerical)
    
    if any(name.lower().startswith(x) for x in ["td"]):
        return TDDFTJob(kwargs.get("roots", 10))
    
    raise ValueError("cannot determine job type from string: %s" % name)


def suggested_fix(error):
    if error == "OPT_CONV":
        return "add more optimization cycles or compute Hessians during optimization"
    if error == "SCF_CONV":
        return "use a different SCF algorithm, increase SCF step damping, or increase the number of SCF iterations"
    if error == "FBX":
        return "restart the job"
    if error == "REDUND":
        return "restart the job"
    if error == "COORD":
        return "restart the job"
    if error == "INT_COORD":
        return "restart the job or optimize with Cartesian coordinates"
    if error == "CONSTR":
        return "remove some constraints"
    if error == "QUOTA":
        return "hard drive or job scratch directory is full"
    if error == "MEM":
        return "increase the RAM:CPU ratio"
    if error == "CLASH":
        return "some atoms are too close - move them apart"
    if error == "ICOORDS":
        return "this is typically just a warning and the optimization usually recovers"

    return None

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

        return self.__dict__ == other.__dict__

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

    def get_xtb(self):
        """overwrite to return a dict with XTB_* keys"""
        pass

    @staticmethod
    def resolve_error(error, theory, exec_type, geometry=None):
        """
        returns a copy of theory or modifies theory to attempt
        to resolve an error
        
        theory will be modified if it is not possible for the current theory
        to work for any job
        
        if the error is specific to the molecule and settings, theory will
        be copied, modified, and returned
        raises NotImplementedError if this job type has no fix for
        the error code
        
        :param str error: error code (e.g. SCF_CONV; see fileIO ERROR)
        :param Theory theory: Theory instance used when the error happened
        :param str exec_type: software program (i.e. gaussian, orca, etc.)
        
        :param Geometry geometry: (optional) structure might be adjusted slightly if
            there are close contacts
        """

        if error.upper() == "OMO_UMO_GAP":
            # small HOMO LUMO gap - don't look at the HOMO LUMO gap
            if exec_type.lower() == "gaussian":
                out_theory = theory.copy()
                out_theory.add_kwargs(
                    GAUSSIAN_ROUTE={
                        "IOp": ["8/11=1"], # only warn about small energy gap
                        "SCF": ["Tight"] # tighten convergence to try to offset error
                    }
                )
                return out_theory

        if error.upper() == "CLASH":
            # if there is a clash, rotate substituents to mitigate clashing
            if geometry:
                geom_copy = geometry.copy()
                bad_subs = geom_copy.remove_clash()
                if not bad_subs:
                    geometry.update_structure(geom_copy.coords)
                    return None

        if error.upper() == "SCF_CONV":
            if exec_type.lower() == "gaussian":
                # SCF convergence issue, try different SCF algorithm
                out_theory = theory.copy()
                out_theory.add_kwargs(
                   GAUSSIAN_ROUTE={"scf": ["xqc"]}
                )
                return out_theory

            if exec_type.lower() == "orca":
                # SCF convergence issue, orca recommends ! SlowConv
                # and increasing SCF iterations
                out_theory = theory.copy()
                out_theory.add_kwargs(
                    ORCA_ROUTE=["SlowConv"],
                    ORCA_BLOCKS={"scf": ["MaxIter 500"]}
                )
                return out_theory
        
            if exec_type.lower() == "psi4":
                out_theory = theory.copy()
                # if theory.charge < 0:
                #     # if the charge is negative, try adding two electrons
                #     # to get a guess that might work
                #     # as well as HF with a small basis set
                #     out_theory.kwargs = combine_dicts(
                #         {
                #             PSI4_BEFORE_JOB: {
                #                 [
                #                     "mol = get_active_molecule()",
                #                     "mol.set_molecular_charge(%i)" % (theory.charge + 2),
                #                     "nrg = energy($METHOD)",
                #                     "mol.set_molecular_charge(%i)" % theory.charge,
                #                 ]
                #             }
                #         },
                #         out_theory.kwargs,
                #     )
                # ^ this doesn't seem to help in general
                # do 500 iterations and dampen
                out_theory.kwargs = combine_dicts(
                    {
                        PSI4_SETTINGS: {
                            "damping_percentage": "15",
                            "maxiter": "500",
                        },
                    },
                    out_theory.kwargs,
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
        scans=None,
        geometry=None,
    ):
        """
        :param bool transition_state: request a transition state optimization
        :param dict constraints: keys are
        
            **available for ORCA, Gaussian, and Psi4**
            
            * 'atoms' - atom identifiers/finders - atoms to constrain
            * 'bonds' - list(atom idenifiers/finders) - distances to constrain each atom identifier
              in the list should result in exactly 2 atoms
            * 'angles' - list(atom idenifiers/finders) - 1-3 angles to constrain each atom identifier
              should result in exactly 3 atoms
            * 'torsions' - list(atom identifiers/finders) - constrained dihedral angles each atom
              identifier should result in exactly 4 atoms
            
            
            **available for Gaussian and Psi4**
            
            * 'x' - list(atom identifiers/finders) - constrain the x coordinate of these atoms. Similarly, 'y' and 'z' are also accepted.
            * 'xgroup' - list(tuple(list(atom idenifiers), x_val, hold)) -
              constrain the x coordinate of these atoms to be the same
                
                * x_val - set x-coordinate to this value
                * hold - hold this value constant during the optimization if 'hold' is omitted,
                  the value will not be held constant during the optimization
                
                  e.g. 'xgroup':[("1-6", 0, False), ("13-24", 3.25, False)]
                
                  this will keep atoms 1-6 and 13-24 in parallel planes, while also
                  allowing those planes to move
            
            * 'ygroup' and 'zgroup' are also available, with analagous options
    
            **NOTE**
            
            for Gaussian, 'bonds', 'angles', and 'torsions' constraints cannot be mixed
            with 'x', 'y', 'z', 'xgroup', 'ygroup', or 'zgroup' constraints
        
        :param Geometry geometry: will be set when using an AaronTools FileWriter"""
        super().__init__()

        self.transition_state = transition_state
        self.constraints = constraints
        self.scans = scans
        self.geometry = geometry

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE, GAUSSIAN_CONSTRAINTS"""
        if self.transition_state:
            out = {GAUSSIAN_ROUTE: {"Opt": ["ts"]}}
        else:
            out = {GAUSSIAN_ROUTE: {"Opt": []}}

        coords = dict()
        con_vars = []
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
                    "linear_angles",
                    "torsions",
                ]:
                    raise NotImplementedError(
                        "%s constraints cannot be generated for Gaussian" % key
                    )
            out[GAUSSIAN_CONSTRAINTS] = []

            dummies = np.cumsum([1 if atom.is_dummy else 0 for atom in self.geometry.atoms], dtype=int)

            if "x" in self.constraints and self.constraints["x"]:
                x_atoms = self.geometry.find(self.constraints["x"])
                for i, atom in enumerate(self.geometry.atoms):
                    if atom in x_atoms:
                        var_name = "x%i" % (i + 1 - dummies[i])
                        consts.append((var_name, atom.coords[0]))
                        coords.setdefault(i, self.geometry.coords[i].tolist())
                        coords[i][0] = var_name

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "y" in self.constraints and self.constraints["y"]:
                y_atoms = self.geometry.find(self.constraints["y"])
                for i, atom in enumerate(self.geometry.atoms):
                    if atom in y_atoms:
                        var_name = "y%i" % (i + 1 - dummies[i])
                        consts.append((var_name, atom.coords[1]))
                        coords.setdefault(i, self.geometry.coords[i].tolist())
                        coords[i][1] = var_name

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "z" in self.constraints and self.constraints["z"]:
                z_atoms = self.geometry.find(self.constraints["z"])
                for i, atom in enumerate(self.geometry.atoms):
                    if atom in z_atoms:
                        var_name = "z%i" % (i + 1 - dummies[i])
                        consts.append((var_name, atom.coords[2]))
                        coords.setdefault(i, self.geometry.coords[i].tolist())
                        coords[i][2] = var_name

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
                        con_vars.append([var_name, val])
                    for i, atom in enumerate(self.geometry.atoms):
                        if atom in x_atoms:
                            coords.setdefault(i, self.geometry.coords[i].tolist())
                            coords[i][0] = var_name

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
                        con_vars.append([var_name, val])
                    for i, atom in enumerate(self.geometry.atoms):
                        if atom in y_atoms:
                            coords.setdefault(i, self.geometry.coords[i].tolist())
                            coords[i][1] = var_name

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
                        con_vars.append([var_name, val])
                    for i, atom in enumerate(self.geometry.atoms):
                        if atom in z_atoms:
                            coords.setdefault(i, self.geometry.coords[i].tolist())
                            coords[i][2] = var_name

                if not use_zmat:
                    use_zmat = True
                    out[GAUSSIAN_ROUTE]["Opt"].append("Z-Matrix")

            if "atoms" in self.constraints and self.constraints["atoms"]:
                try:
                    atoms = self.geometry.find(self.constraints["atoms"])
                except LookupError as e:
                    self.LOG.warning(e)
                    atoms = []
                for i, atom in enumerate(atoms):
                    ndx = self.geometry.atoms.index(atom) + 1 - dummies[i]
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append("%2i F" % ndx)
                    else:
                        for j, coord in enumerate(coords[ndx - 1]):
                            if isinstance(coord, str):
                                var_name = coord
                                for k, var in enumerate(con_vars):
                                    if var[0] == coord and not var[
                                        0
                                    ].startswith("g"):
                                        con_vars.pop(k)
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
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
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
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx3 -= dummies[ndx3]
                    ndx3 += 1
                    if not use_zmat:
                        angle = atom2.angle(atom1, atom3)
                        # TODO: figure out proper tolerance for linear angles
                        if abs(angle - np.pi) < 5e-2 or abs(angle) < 5e-2:
                            out[GAUSSIAN_CONSTRAINTS].append(
                                "L %2i %2i %2i F" % (ndx1, ndx2, ndx3)
                            )
                        else:
                            out[GAUSSIAN_CONSTRAINTS].append(
                                "A %2i %2i %2i F" % (ndx1, ndx2, ndx3)
                            )
                    else:
                        raise NotImplementedError(
                            "cannot apply angle constraints when using Cartesian Z-Matrix, which"
                            + " is necessitated by x, y, or z constraints"
                        )

            if "linear_angles" in self.constraints:
                for constraint in self.constraints["linear_angles"]:
                    atoms = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atoms[0])
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atoms[1])
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atoms[2])
                    ndx3 -= dummies[ndx3]
                    ndx3 += 1
                    ndx4 = None
                    if len(atoms) == 4:
                        ndx4 = self.geometry.atoms.index(atoms[3])
                        ndx4 -= dummies[ndx3]
                        ndx4 += 1

                    if not use_zmat:
                        if ndx4 is None:
                            out[GAUSSIAN_CONSTRAINTS].append(
                                "L %2i %2i %2i F" % (ndx1, ndx2, ndx3)
                            )
                        else:
                            out[GAUSSIAN_CONSTRAINTS].append(
                                "L %2i %2i %2i %2i F" % (ndx1, ndx2, ndx3, ndx4)
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
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx3 -= dummies[ndx3]
                    ndx3 += 1
                    ndx4 = self.geometry.atoms.index(atom4)
                    ndx4 -= dummies[ndx4]
                    ndx4 += 1
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

        if self.scans is not None and any(
            self.scans[key] for key in self.scans.keys()
        ):
            for key in self.scans:
                if self.scans[key] and key not in [
                    "bonds",
                    "angles",
                    "torsions",
                ]:
                    raise NotImplementedError(
                        "%s scans cannot be generated for Gaussian" % key
                    )
            out.setdefault(GAUSSIAN_CONSTRAINTS, [])

            dummies = np.cumsum([1 if atom.is_dummy else 0 for atom in self.geometry.atoms], dtype=int)


            if "bonds" in self.scans:
                for constraint in self.scans["bonds"]:
                    steps, size = constraint[-2:]
                    atom1, atom2 = self.geometry.find(constraint[:-2])
                    ndx1 = self.geometry.atoms.index(atom1) 
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2) 
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append(
                            "B %2i %2i S %2i %f" % (ndx1, ndx2, steps, size)
                        )
                    else:
                        raise NotImplementedError(
                            "cannot apply bond scans when using Cartesian Z-Matrix, which"
                            + " is necessitated by x, y, or z constraints"
                        )

                if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                    out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")

            if "angles" in self.scans:
                for constraint in self.scans["angles"]:
                    steps, size = constraint[-2:]
                    atom1, atom2, atom3 = self.geometry.find(constraint[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx3 -= dummies[ndx3]
                    ndx3 += 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append(
                            "A %2i %2i %2i S %2i %f" % (ndx1, ndx2, ndx3, steps, size)
                        )
                    else:
                        raise NotImplementedError(
                            "cannot apply angle scans when using Cartesian Z-Matrix, which"
                            + " is necessitated by x, y, or z constraints"
                        )

                if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                    out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")

            if "torsions" in self.scans:
                for constraint in self.scans["torsions"]:
                    steps, size = constraint[-2:]
                    atom1, atom2, atom3, atom4 = self.geometry.find(constraint[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 -= dummies[ndx1]
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 -= dummies[ndx2]
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx3 -= dummies[ndx3]
                    ndx3 += 1
                    ndx4 = self.geometry.atoms.index(atom4)
                    ndx4 -= dummies[ndx4]
                    ndx4 += 1
                    if not use_zmat:
                        out[GAUSSIAN_CONSTRAINTS].append(
                            "D %2i %2i %2i %2i S %2i %f" % (ndx1, ndx2, ndx3, ndx4, steps, size)
                        )
                    else:
                        raise NotImplementedError(
                            "cannot apply torsion scans when using Cartesian Z-Matrix,"
                            + "which is necessitated by x, y, or z constraints"
                        )

                if "ModRedundant" not in out[GAUSSIAN_ROUTE]["Opt"]:
                    out[GAUSSIAN_ROUTE]["Opt"].append("ModRedundant")


        if consts or con_vars:
            for i, coord in enumerate(coords.values()):
                for j, ax in enumerate(["x", "y", "z"]):
                    if isinstance(coord[j], float):
                        var_name = "%s%i" % (ax, i + 1)
                        con_vars.append((var_name, coord[j]))
                        coord[j] = var_name

        if consts or con_vars:
            for coord in coords.values():
                coord.insert(0, 0)

        out[GAUSSIAN_COORDINATES] = {
            "coords": coords,
            "variables": con_vars,
            "constants": consts,
        }

        return out, []

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
            geom_constraints = "Constraints\n"
            if "atoms" in self.constraints:
                for constraint in self.constraints["atoms"]:
                    atom1 = self.geometry.find(constraint)[0]
                    ndx1 = self.geometry.atoms.index(atom1)
                    out_str = "        {C %2i C}\n" % (ndx1)
                    geom_constraints += out_str

            if "bonds" in self.constraints:
                for constraint in self.constraints["bonds"]:
                    atom1, atom2 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    out_str = "        {B %2i %2i C}\n" % (ndx1, ndx2)
                    geom_constraints += out_str
    
            if "angles" in self.constraints:
                for constraint in self.constraints["angles"]:
                    atom1, atom2, atom3 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    out_str = "        {A %2i %2i %2i C}\n" % (ndx1, ndx2, ndx3)
                    geom_constraints += out_str

            if "torsions" in self.constraints:
                for constraint in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx4 = self.geometry.atoms.index(atom4)
                    out_str = "        {D %2i %2i %2i %2i C}\n" % (
                        ndx1,
                        ndx2,
                        ndx3,
                        ndx4,
                    )
                    geom_constraints += out_str

            geom_constraints += "    end"
            out[ORCA_BLOCKS] = {"geom": [geom_constraints]}

        if self.scans is not None and any(
            self.scans[key] for key in self.scans.keys()
        ):
            for key in self.scans:
                if self.scans[key] and key not in [
                    "bonds",
                    "angles",
                    "torsions",
                ]:
                    raise NotImplementedError(
                        "%s scans cannot be generated for ORCA" % key
                    )
            geom_scans = []
            if "bonds" in self.scans:
                for scan in self.scans["bonds"]:
                    steps, size = scan[-2:]
                    atom1, atom2 = self.geometry.find(scan[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    start = atom1.dist(atom2)
                    stop = start + size * (steps - 1)
                    out_str = "Scan B %i %i [%s] end" % (
                        ndx1, ndx2, " ".join("%.3f" % x for x in np.linspace(start, stop, steps))
                    )
                    geom_scans.append(out_str)
    
            if "angles" in self.scans:
                for scan in self.scans["angles"]:
                    steps, size = scan[-2:]
                    atom1, atom2, atom3 = self.geometry.find(scan[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    start = np.rad2deg(atom2.angle(atom1, atom3))
                    stop = start + size * (steps - 1)
                    out_str = "Scan A %i %i %i [%s] end" % (
                        ndx1, ndx2, ndx3, " ".join("%.3f" % x for x in np.linspace(start, stop, steps))
                    )
                    geom_scans.append(out_str)

            if "torsions" in self.scans:
                for scan in self.scans["torsions"]:
                    steps, size = scan[-2:]
                    atom1, atom2, atom3, atom4 = self.geometry.find(scan[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx4 = self.geometry.atoms.index(atom4)
                    start = np.rad2deg(self.geometry.dihedral(atom1, atom2, atom3, atom4))
                    stop = start + size * (steps - 1)
                    out_str = "Scan D %i %i %i %i [%s] end" % (
                        ndx1, ndx2, ndx3, ndx4, " ".join("%.3f" % x for x in np.linspace(start, stop, steps))
                    )
                    geom_scans.append(out_str)
            
            print(geom_scans)
            out.setdefault(ORCA_BLOCKS, {})
            out[ORCA_BLOCKS].setdefault("geom", [])
            out[ORCA_BLOCKS]["geom"].extend(geom_scans)

        return out, []

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB, PSI4_OPTKING, PSI4_BEFORE_GEOM"""
        if self.transition_state:
            out = {
                PSI4_JOB: {"optimize": []},
                PSI4_SETTINGS: {"opt_type": ["ts"]},
            }
        else:
            out = {PSI4_JOB: {"optimize": []}}

        warnings = []

        coords = self.geometry.coords.tolist()
        vars = []
        group_count = 1

        freeze_str = ""
        freeze_str += 'freeze_list = """\n'
        add_freeze_list = False

        if self.scans and any(self.scans[key] for key in self.scans):
            raise NotImplementedError(
                "coordinate scans have not been implemented for the Psi4 input builder"
            )

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
                    out_str = '("\n'
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

        return out, warnings

    def get_xtb(self):
        """
        Generates xcontrol file constraints

        Returns: dict(fix, constrain, metadyn, cli)
        """
        out = dict()
        # only put constraints in xcontrol file so this works with Crest also
        if self.constraints and self.constraints.get("atoms", False):
            frozen = [
                self.geometry.atoms.index(a) + 1
                for a in self.geometry.find(self.constraints["atoms"])
            ]
            frozen = range_list(frozen)
            out["fix"] = []
            out["fix"].append("atoms: {}".format(frozen))
            out["fix"].append("freeze: {}".format(frozen))

        if self.constraints and self.constraints.get("bonds", False):
            out.setdefault("constrain", [])
            for bond in self.constraints["bonds"]:
                bond = self.geometry.find(bond)
                out["constrain"].append(
                    "distance: {},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in bond)
                    )
                )

        if self.scans and self.scans.get("bonds", False):
            out.setdefault("constrain", [])
            out.setdefault("scan", [])
            for bond in self.scans["bonds"]:
                a1, a2, steps, size = bond
                bond = self.geometry.find([a1, a2])
                out["constrain"].append(
                    "distance: {},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in bond)
                    )
                )
                start = bond[0].dist(bond[1])
                stop = start + (steps - 1) * size
                out["scan"].append(
                    "%i: %.3f, %.3f, %2i" % (
                        len(out["constrain"]), start, stop, steps,
                    )
                )

        if self.constraints and self.constraints.get("angles", False):
            out.setdefault("constrain", [])
            for angle in self.constraints["angles"]:
                angle = self.geometry.find(angle)
                out["constrain"].append(
                    "angle: {},{},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in angle)
                    )
                )

        if self.scans and self.scans.get("angles", False):
            out.setdefault("constrain", [])
            out.setdefault("scan", [])
            for angle in self.scans["angles"]:
                a1, a2, a3, steps, size = angle
                angle = self.geometry.find([a1, a2, a3])
                out["constrain"].append(
                    "angle: {},{},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in angle)
                    )
                )
                start = np.rad2deg(self.geometry.angle(*angle))
                stop = start + (steps - 1) * size
                out["scan"].append(
                    "%i: %.3f, %.3f, %2i" % (
                        len(out["constrain"]), start, stop, steps,
                    )
                )

        if self.constraints and self.constraints.get("torsions", False):
            out.setdefault("constrain", [])
            for dihedral in self.constraints["torsions"]:
                dihedral = self.geometry.find(dihedral)
                out["constrain"].append(
                    "dihedral: {},{},{},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in dihedral)
                    )
                )

        if self.scans and self.scans.get("torsions", False):
            out.setdefault("constrain", [])
            out.setdefault("scan", [])
            for dihedral in self.scans["torsions"]:
                a1, a2, a3, a4, steps, size = dihedral
                dihedral = self.geometry.find([a1, a2, a3, a4])
                start = np.rad2deg(self.geometry.dihedral(*dihedral))
                stop = start + (steps - 1) * size
                out["constrain"].append(
                    "dihedral: {},{},{},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in dihedral)
                    )
                )
                out["scan"].append(
                    "%i: %.3f, %.3f, %2i" % (
                        len(out["constrain"]), start, stop, steps,
                    )
                )

        out = {XTB_CONTROL_BLOCKS: out}
        
        if self.transition_state:
            out[XTB_COMMAND_LINE] = {"optts": []}
        else:
            out[XTB_COMMAND_LINE] = {"opt": []}
        
        return out, []

    def get_sqm(self):
        """returns a dict(), warnings for optimization jobs"""
        warnings = []
        if self.transition_state:
            warnings.append("cannot do TS optimization with sqm")
        
        if self.constraints:
            warnings.append("cannot constrain sqm optimization")
        
        return dict(), warnings

    def get_qchem(self):
        """returns dict for Q-Chem"""
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

            if "x" in self.constraints and self.constraints["x"]:
                x_atoms = self.geometry.find(self.constraints["x"])


            if "y" in self.constraints and self.constraints["y"]:
                y_atoms = self.geometry.find(self.constraints["y"])

            if "z" in self.constraints and self.constraints["z"]:
                z_atoms = self.geometry.find(self.constraints["z"])

            if "atoms" in self.constraints and self.constraints["atoms"]:
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
                    constraints += "        BEND %2i %2i %2i %9.5f\n" % (
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

        if self.scans is not None and any(
            self.scans[key] for key in self.scans.keys()
        ):
            for key in self.scans:
                if self.scans[key] and key not in [
                    "bonds",
                    "angles",
                    "torsions",
                ]:
                    raise NotImplementedError(
                        "%s scans cannot be generated for Gaussian" % key
                    )
            
            out.setdefault(QCHEM_SETTINGS, {})
            out[QCHEM_SETTINGS].setdefault("scans", [])

            if "bonds" in self.scans:
                for constraint in self.scans["bonds"]:
                    steps, size = constraint[-2:]
                    atom1, atom2 = self.geometry.find(constraint[:-2])
                    ndx1 = self.geometry.atoms.index(atom1) 
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2) 
                    ndx2 += 1
                    current = atom1.dist(atom2)
                    final = current + size * steps
                    out[QCHEM_SETTINGS]["scans"].append(
                        "stre %2i %2i %.4f %.4f %.4f" % (ndx1, ndx2, current, final, size)
                    )

            if "angles" in self.scans:
                for constraint in self.scans["angles"]:
                    steps, size = constraint[-2:]
                    atom1, atom2, atom3 = self.geometry.find(constraint[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx3 += 1
                    current = np.rad2deg(atom2.angle(atom1, atom3))
                    final = current + steps * size
                    out[QCHEM_SETTINGS]["scans"].append(
                        "bend %2i %2i %2i %.4f %.4f %.4f" % (ndx1, ndx2, ndx3, current, final, size)
                    )

            if "torsions" in self.scans:
                for constraint in self.scans["torsions"]:
                    steps, size = constraint[-2:]
                    atom1, atom2, atom3, atom4 = self.geometry.find(constraint[:-2])
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx1 += 1
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx2 += 1
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx3 += 1
                    ndx4 = self.geometry.atoms.index(atom4)
                    ndx4 += 1
                    current = np.rad2deg(self.geometry.dihedral(atom1, atom2, atom3, atom4))
                    final = current + steps * size
                    out[QCHEM_SETTINGS]["scans"].append(
                        "tors %2i %2i %2i %2i %.4f %.4f %.4f" % (ndx1, ndx2, ndx3, ndx4, current, final, size)
                    )

        return out, []

    @staticmethod
    def resolve_error(error, theory, exec_type, geometry=None):
        """
        resolves optimization-specific errors
        errors resolved by JobType take priority
        
        :param str error: error code from FileReader["error"] 
        :param Theory theory: theory used
        :param str exec_type: program name (gaussian, psi4, orca...)
        :param Geometry geometry: structure used when the error occured 
        
        Note that the geometry might be modified when resolving errors
        
        :rtype: None|Theory
        """
        try:
            return super(OptimizationJob, OptimizationJob).resolve_error(
                error, theory, exec_type, geometry=geometry
            )
        except NotImplementedError:
            pass
        
        if exec_type.lower() == "gaussian":
            if error.upper() == "OPT_CONV":
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
                    coords += scale * np.random.random_sample(coords.shape) - scale / 2
                    geometry.update_structure(coords)
                return None
            
            if error.upper() == "REDUND":
                # internal coordinate error, just restart the job
                if geometry:
                    coords = geometry.coords
                    scale = 1e-3
                    coords += scale * np.random.random_sample(coords.shape) - scale / 2
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
        
        if exec_type.lower() == "psi4":
            if error.upper() == "ICOORD":
                out_theory = theory.copy()
                out_theory.kwargs = combine_dicts(
                     {PSI4_OPTKING: {"opt_coordinates": "cartesian"}}, out_theory.kwargs,
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

        return out, []

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        out = {ORCA_BLOCKS: {"freq": ["Temp    %.2f" % self.temperature]}}
        if self.numerical:
            out[ORCA_ROUTE] = ["NumFreq"]
        else:
            out[ORCA_ROUTE] = ["Freq"]

        return out, []

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        out = {
            PSI4_JOB: {"frequencies": []},
            PSI4_SETTINGS: {"T": ["%.2f" % self.temperature]},
        }
        if self.numerical:
            out[PSI4_JOB]["frequencies"].append('dertype="gradient"')

        return out, []

    def get_sqm(self):
        raise NotImplementedError("cannot build frequnecy job input for sqm")

    def get_qchem(self):
        out = {QCHEM_REM: {"JOB_TYPE": "Freq"}}
        if self.numerical:
            out[QCHEM_REM]["FD_DERIVATIVE_TYPE"] = "1"

        return out, []

    def get_xtb(self):
        out = {
            XTB_COMMAND_LINE: {"hess": []},
            XTB_CONTROL_BLOCKS: {"thermo": ["temp=%.2f" % self.temperature]},
        }
        return out, []

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
        return {GAUSSIAN_ROUTE: {"SP": []}}, []

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE: ["SP"]}, []

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        return {PSI4_JOB: {"energy": []}}, []
    
    def get_sqm(self):
        """returns a dict with keys: SQM_QMMM"""
        return {SQM_QMMM: {"maxcyc": ["0"]}}, []

    def get_qchem(self):
        out = {QCHEM_REM: {"JOB_TYPE": "SP"}}
        
        return out, []


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
        return out, []

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE: ["NumGrad" if self.numerical else "EnGrad"]}, []

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        out = {PSI4_JOB: {"gradient": []}}
        if self.numerical:
            out[PSI4_JOB]["gradient"].append("dertype='energy'")
        return out, []

    def get_qchem(self):
        out = {QCHEM_REM: {"JOB_TYPE": "Force"}}
        
        return out, []

    def get_xtb(self):
        return {XTB_COMMAND_LINE: {"grad": []}}, []


class ConformerSearchJob(JobType):
    """conformer search (basically only for crest)"""
    
    def __init__(
        self,
        constraints=None,
        geometry=None,
        use_topology=True,
    ):
        """
        :param dict constraints:
            
            valid keys are:
            
            * 'atoms' - atom identifiers/finders - atoms to constrain
            * 'bonds' - list(atom idenifiers/finders) - distances to constrain each atom identifier in the list should result in exactly 2 atoms
            * 'angles' - list(atom idenifiers/finders) - 1-3 angles to constrain each atom identifier should result in exactly 3 atoms
            * 'torsions' - list(atom identifiers/finders) - constrained dihedral angles
                each atom identifier should result in exactly 4 atoms            
        
        :param Geometry geometry: will be set when using an AaronTools FileWriter
        """
        self.constraints = constraints
        self.geometry = geometry
        self.use_topology = use_topology

    def get_crest(self):
        out = dict()

        constrained = set([])
        if self.constraints and self.constraints.get("atoms", False):
            out.setdefault("constrain", [])
            atoms = self.geometry.find(self.constraints["atoms"])
            constrained.update(atoms)
            out["constrain"].append("atoms: " + ",".join(
                [str(self.geometry.atoms.index(a) + 1) for a in atoms]
            ))

        if self.constraints and self.constraints.get("bonds", False):
            out.setdefault("constrain", [])
            for bond in self.constraints["bonds"]:
                bond = self.geometry.find(bond)
                constrained.update(bond)
                out["constrain"].append(
                    "distance: {},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in bond)
                    )
                )

        if self.constraints and self.constraints.get("angles", False):
            out.setdefault("constrain", [])
            for angle in self.constraints["angles"]:
                angle = self.geometry.find(angle)
                constrained.update(angle)
                out["constrain"].append(
                    "angle: {},{},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in angle)
                    )
                )

        if self.constraints and self.constraints.get("torsions", False):
            out.setdefault("constrain", [])
            for dihedral in self.constraints["torsions"]:
                dihedral = self.geometry.find(dihedral)
                constrained.update(dihedral)
                out["constrain"].append(
                    "dihedral: {},{},{},{},auto".format(
                        *(self.geometry.atoms.index(c) + 1 for c in dihedral)
                    )
                )

        if out:
            out = {XTB_CONTROL_BLOCKS: out}

        if constrained:
            out[XTB_CONTROL_BLOCKS].setdefault("constrain", [])
            out[XTB_CONTROL_BLOCKS]["constrain"].append(
                "reference=original.xyz"
            )
            relaxed = {
                i + 1
                for i, a in enumerate(self.geometry.atoms)
                if a not in constrained
            }
            relaxed = range_list(relaxed)
            out[XTB_CONTROL_BLOCKS]["metadyn"] = ["  atoms: {}".format(relaxed)]
            # out[XTB_COMMAND_LINE] = {}
            # out[XTB_COMMAND_LINE]["cinp"] = ["{{ name }}.xc"]

        if self.use_topology:
            out.setdefault(XTB_COMMAND_LINE, {})
            out[XTB_COMMAND_LINE]["noreftopo"] = []

        return out, []

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE, ORCA_BLOCKS"""
        if self.use_topology:
            out = {ORCA_ROUTE: ["GOAT"]}
        else:
            out = {ORCA_ROUTE: ["GOAT-EXPLORE"]}


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
            geom_constraints = "Constraints\n"
            if "atoms" in self.constraints:
                for constraint in self.constraints["atoms"]:
                    atom1 = self.geometry.find(constraint)[0]
                    ndx1 = self.geometry.atoms.index(atom1)
                    out_str = "        {C %2i C}\n" % (ndx1)
                    geom_constraints += out_str

            if "bonds" in self.constraints:
                for constraint in self.constraints["bonds"]:
                    atom1, atom2 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    out_str = "        {B %2i %2i C}\n" % (ndx1, ndx2)
                    geom_constraints += out_str
    
            if "angles" in self.constraints:
                for constraint in self.constraints["angles"]:
                    atom1, atom2, atom3 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    out_str = "        {A %2i %2i %2i C}\n" % (ndx1, ndx2, ndx3)
                    geom_constraints += out_str

            if "torsions" in self.constraints:
                for constraint in self.constraints["torsions"]:
                    atom1, atom2, atom3, atom4 = self.geometry.find(constraint)
                    ndx1 = self.geometry.atoms.index(atom1)
                    ndx2 = self.geometry.atoms.index(atom2)
                    ndx3 = self.geometry.atoms.index(atom3)
                    ndx4 = self.geometry.atoms.index(atom4)
                    out_str = "        {D %2i %2i %2i %2i C}\n" % (
                        ndx1,
                        ndx2,
                        ndx3,
                        ndx4,
                    )
                    geom_constraints += out_str

            geom_constraints += "    end"
            out[ORCA_BLOCKS] = {"geom": [geom_constraints]}

        return out, []


class TDDFTJob(JobType):
    """TD-DFT job"""
    def __init__(self, roots, root_of_interest=0, compute_nacmes=False):
        """
        :param int roots: number of roots
        :param int root_of_interest: root for further computations (e.g. forces)
        :param bool compute_nacmes: request nonadiabatic coupling matrix elements
        """
        self.root_of_interest = root_of_interest
        self.roots = roots
        self.compute_nacmes = compute_nacmes
    
    def get_gaussian(self):
        out = dict()
        warnings = []
        out[GAUSSIAN_ROUTE] = {
            "TD": [
                "Root=%i" % self.root_of_interest,
                "NStates=%i" % self.roots,
            ]
        }
        if self.compute_nacmes:
            warnings.append(
                "nonadiabatic matrix elements are not supported for Gaussian"
            )
        return out, warnings
    
    def get_orca(self):
        out = dict()
        out[ORCA_BLOCKS] = {
            "TDDFT": [
                "IRoot %i" % self.root_of_interest,
                "NRoots %i" % self.roots,
            ]
        }
        
        if self.compute_nacmes:
            out[ORCA_BLOCKS]["TDDFT"].extend([
                "NAMCE True",
                "ETF True",
            ])
        return out, []
    
    def get_psi4(self):
        out = dict()
        warnings = []
        out[PSI4_BEFORE_GEOM] = [
            "from psi4.driver.procrouting.response.scf_response import tdscf_excitations"
        ]
        out[PSI4_SETTINGS] = {
            "save_jk": "true",
        }
        out[PSI4_JOB] = {"energy": ["return_wfn=True"]}
        out[PSI4_AFTER_JOB] = [
            "tdscf_excitations(wfn, states=%i)" % self.roots,
        ]
        
        if self.root_of_interest:
            warnings.append("only initial state being the ground state is supported")
        if self.compute_nacmes:
            warnings.append("nonadiabatic matrix elements are not supported for Psi4")
        return out, warnings
    
    def get_qchem(self):
        # TODO
        raise NotImplementedError("we currently don't support TD-DFT for Q-Chem")
    

class NMRJob(JobType):
    """NMR job"""
    def __init__(self, geometry=None, atoms=None, coupling_type=None):
        super().__init__()
        self.geometry = geometry
        self.atoms = atoms
        self.coupling_type = coupling_type
    
    def get_gaussian(self):
        out = {GAUSSIAN_ROUTE: {"NMR": []}}
        warnings = []
        if self.coupling_type == "spin-spin":
            out[GAUSSIAN_ROUTE]["NMR"].append("SpinSpin")
        elif self.coupling_type == "mixed":
            out[GAUSSIAN_ROUTE]["NMR"].append("Mixed")
        elif self.coupling_type:
            warnings.append("coupling_type '%s' might not be available" % self.coupling_type)
            out[GAUSSIAN_ROUTE]["NMR"].append(self.coupling_type)
        if self.atoms:
            out[GAUSSIAN_ROUTE]["NMR"].append("ReadAtoms")
            if all(x in ELEMENTS for x in self.atoms):
                out[GAUSSIAN_POST] = ["atoms=%s" % ",".join(self.atoms)]
                
            else:
                atoms = self.geometry.find(self.atoms)
                ndx = {a: str(i + 1) for i, a in enumerate(atoms)}
                out[GAUSSIAN_POST] = ["atoms=%s" % ",".join([ndx[a] for a in atoms])]
    
        return out, warnings
    
    def get_orca(self):
        out = {ORCA_ROUTE: ["NMR"]}
        out[ORCA_BLOCKS] = {"eprnmr": []}
        warnings = []
        if self.atoms:
            use_symbols = True
            if all(x in ELEMENTS for x in self.atoms):
                atoms = self.atoms
            else:
                atoms = self.geometry.find(self.atoms)
                use_symbols = False
            for i, a in enumerate(atoms):
                if use_symbols:
                    spec = "Nuclei = all %s " % a
                else:
                    spec = "Nuclei = %i " % (i + 1)
                kind = ["shift"]
                if self.coupling_type == "spin-spin":
                    kind.append("ssall")
                elif self.coupling_type:
                    warnings.append("coupling_type '%s' might not be available" % self.coupling_type)
                    kind = [self.coupling_type]
                spec += "{ %s }" % ", ".join(kind)
                out[ORCA_BLOCKS]["eprnmr"].append(spec)
        
        return out, warnings
        
