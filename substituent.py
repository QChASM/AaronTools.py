"""for fragments attached to a structure by one bond"""

#!/usr/bin/env python3
import json
import os
import re
import sys
from copy import deepcopy
from glob import glob
from warnings import warn

import numpy as np

from AaronTools.const import AARONLIB, AARONTOOLS, BONDI_RADII, VDW_RADII
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import perp_vector


class Substituent(Geometry):
    """
    Attributes:
        name
        atoms
        end         the atom substituent is connected to
        conf_num    number of conformers
        conf_angle  angle to rotate by to make next conformer
    """

    AARON_LIBS = os.path.join(AARONLIB, "Subs")
    BUILTIN = os.path.join(AARONTOOLS, "Substituents")
    CACHE_FILE = os.path.join(AARONLIB, "cache", "substituents.json")

    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        cache = {}
        cache["lengths"] = {}  # for storing number of atoms in each sub

    def __init__(
            self,
            sub=None,
            name=None,
            targets=None,
            end=None,
            conf_num=None,
            conf_angle=None,
            detect=True,
    ):
        """
        sub is either a file sub, a geometry, or an atom list
        """
        super().__init__()
        self.name = name
        self.atoms = []
        self.end = end
        self.conf_angle = conf_angle
        self.conf_num = conf_num
        self.comment = None
        if sub is None:
            return

        if isinstance(sub, (Geometry, list)):
            # we can create substituent object from fragment
            if isinstance(sub, Substituent):
                self.name = name if name else sub.name
                self.conf_num = conf_num if conf_num else sub.conf_num
                self.conf_angle = conf_angle if conf_angle else sub.conf_angle
                self.comment = sub.comment
            elif isinstance(sub, Geometry):
                self.name = name if name else sub.name
                self.conf_num = conf_num
                self.conf_angle = conf_angle
                self.comment = sub.comment
            else:
                self.name = name
                self.conf_num = conf_num
                self.conf_angle = conf_angle

            # save atom info
            if targets is None:
                try:
                    self.atoms = sub.atoms
                except AttributeError:
                    self.atoms = sub
            else:
                self.atoms = sub.find(targets)

            # detect sub and conformer info
            if detect and (not conf_num or not conf_angle):
                if not self.detect_sub():
                    LookupError(
                        "Substituent not found in library: " + str(self.name)
                    )
        else:
            # or we can create from file
            # find substituent xyz file
            fsub = None
            for lib in [Substituent.AARON_LIBS, Substituent.BUILTIN]:
                if not os.path.exists(lib):
                    continue
                for f in os.listdir(lib):
                    name, ext = os.path.splitext(f)
                    if not any(".%s" % x == ext for x in read_types):
                        continue
                    match = sub == name
                    if match:
                        fsub = os.path.join(lib, f)
                        break
                
                if fsub:
                    break

            # or assume we were given a file name instead
            if not fsub and ".xyz" in sub:
                fsub = sub
                sub = os.path.basename(sub).rstrip(".xyz")

            if fsub is None:
                match = re.search(r"^{X(.*)}$", sub)
                if match:
                    fsub = Geometry.from_string("Cl" + match.group(1))
                    fsub.coord_shift(-fsub.atoms[0].coords)
                    bond = fsub.bond(fsub.atoms[0], fsub.atoms[1])
                    x_axis = np.array([1.0, 0.0, 0.0])
                    rot_axis = np.cross(bond, x_axis)
                    if np.linalg.norm(rot_axis):
                        bond /= np.linalg.norm(bond)
                        rot_axis /= np.linalg.norm(rot_axis)
                        angle = np.arccos(np.dot(bond, x_axis))
                        fsub.rotate(rot_axis, angle)
                    self.atoms = fsub.atoms[1:]
                    self.refresh_connected()
                    self.name = match.group(1)
                    warn("Conformer info not loaded for" + sub)
                    return
                else:
                    raise RuntimeError(
                        "substituent name not recognized: %s" % sub
                    )

            # load in atom info
            from_file = FileReader(fsub)
            self.name = sub
            self.comment = from_file.comment
            self.atoms = from_file.atoms
            self.refresh_connected()
            if targets is not None:
                self.atoms = self.find(targets)

            # set conformer info
            conf_info = re.search(r"CF:(\d+),(\d+)", self.comment)
            if conf_info is not None:
                self.conf_num = int(conf_info.group(1))
                self.conf_angle = np.deg2rad(float(conf_info.group(2)))
            else:
                warn("Conformer info not loaded for" + f)

        if not self.name:
            self.name = "sub"
        if self.name == "sub" and end is not None:
            self.name += "-{}".format(end.name)

    def __lt__(self, other):
        if self.end < other.end and not other.end < self.end:
            return True
        if len(self.atoms) != len(other.atoms):
            return len(self.atoms) < len(other.atoms)
        for a, b in zip(
                self.reorder(start=self.atoms[0])[0],
                other.reorder(start=other.atoms[0])[0],
        ):
            if a < b and not b < a:
                return True
        return False

    @classmethod
    def from_string(
            cls,
            name,
            conf_num=None,
            conf_angle=None,
            form="smiles",
            debug=False,
            strict_use_rdkit=False,
    ):
        """
        creates a substituent from a string
        name        str     identifier for substituent
        conf_num    int     number of conformers expected for hierarchical conformer generation
        conf_angle  int     angle between conformers
        form        str     type of identifier (smiles, iupac)
        """
        # convert whatever format we"re given to smiles
        # then grab the structure from cactus site
        from AaronTools.finders import BondedTo

        accepted_forms = ["iupac", "smiles"]
        if form not in accepted_forms:
            raise NotImplementedError(
                "cannot create substituent given %s; use one of %s" % form,
                str(accepted_forms),
            )

        rad = re.compile(r"\[\S+?\]")
        elements = re.compile(r"[A-Z][a-z]?")

        if form == "smiles":
            smiles = name
        elif form == "iupac":
            smiles = cls.iupac2smiles(name)
        if debug:
            print("radical smiles:", smiles, file=sys.stderr)

        # radical atom is the first atom in []
        # charged atoms are also in []
        my_rad = None
        radicals = rad.findall(smiles)
        if radicals:
            for rad in radicals:
                if "." in rad:
                    my_rad = rad
                    break
                elif "+" not in rad and "-" not in rad:
                    my_rad = rad
                    break
        if my_rad is None:
            if radicals:
                warn(
                    "radical atom may be ambiguous, be sure to check output: %s"
                    % smiles
                )
                my_rad = radicals[0]
            else:
                raise RuntimeError(
                    "could not determine radical site on %s; radical site is expected to be in []"
                    % smiles
                )

        # construct a modified smiles string with (Cl) right after the radical center
        # keep track of the position of this added Cl
        # (use Cl instead of H b/c explicit H"s don"t always play nice with RDKit)
        pos1 = smiles.index(my_rad)
        pos2 = smiles.index(my_rad) + len(my_rad)
        previous_atoms = elements.findall(smiles[:pos1])
        rad_pos = len(previous_atoms)
        if "+" not in my_rad and "-" not in my_rad:
            mod_smiles = (
                smiles[:pos1]
                + re.sub(r"H\d+", "", my_rad[1:-1])
                + "(Cl)"
                + smiles[pos2:]
            )
        else:
            mod_smiles = (
                smiles[:pos1]
                + my_rad[:-1].rstrip("H")
                + "]"
                + "(Cl)"
                + smiles[pos2:]
            )
        mod_smiles = mod_smiles.replace(".", "")
        if debug:
            print("modified smiles:", mod_smiles, file=sys.stderr)
            print("radical position:", rad_pos, file=sys.stderr)

        # grab structure from cactus/RDKit
        geom = Geometry.from_string(
            mod_smiles, form="smiles", strict_use_rdkit=strict_use_rdkit
        )

        # the Cl we added is in the same position in the structure as in the smiles string
        rad = geom.atoms[rad_pos]
        added_Cl = [atom for atom in rad.connected if atom.element == "Cl"][0]

        # move the added H to the origin
        geom.coord_shift(-added_Cl.coords)

        # get the atom bonded to this H
        # also move the atom on H to the front of the atoms list to have the expected connectivity
        bonded_atom = geom.find(BondedTo(added_Cl))[0]
        geom.atoms = [bonded_atom] + [
            atom for atom in geom.atoms if atom != bonded_atom
        ]
        bonded_atom.connected.discard(added_Cl)

        # align the H-atom bond with the x-axis to have the expected orientation
        bond = deepcopy(bonded_atom.coords)
        bond /= np.linalg.norm(bond)
        x_axis = np.array([1.0, 0.0, 0.0])
        rot_axis = np.cross(x_axis, bond)
        if abs(np.linalg.norm(rot_axis)) > np.finfo(float).eps:
            rot_axis /= np.linalg.norm(rot_axis)
            angle = np.arccos(np.dot(bond, x_axis))
            geom.rotate(rot_axis, -angle)
        else:
            try:
                import rdkit
            except ImportError:
                # if the bonded_atom is already on the x axis, we will instead
                # rotate about the y axis by 180 degrees
                angle = np.pi
                geom.rotate(np.array([0.0, 1.0, 0.0]), -angle)

        out = cls(
            [atom for atom in geom.atoms if atom is not added_Cl],
            conf_num=conf_num,
            conf_angle=conf_angle,
            detect=False,
        )
        out.refresh_connected()
        out.refresh_ranks()
        return out

    def copy(self, end=None):
        """
        creates a new copy of the geometry
        parameters:
            atoms (list): defaults to all atoms
            name (str): defaults to NAME_copy
        """
        rv = super().copy()
        rv = Substituent(
            rv,
            end=end,
            conf_angle=self.conf_angle,
            conf_num=self.conf_num,
            detect=False,
        )
        if end is not None:
            rv.atoms[0].connected.add(rv.end)
        return rv

    @classmethod
    def list(cls, include_ext=False):
        """list substituents available from AaronTools or the user's library"""
        names = []
        for lib in [cls.AARON_LIBS, cls.BUILTIN]:
            if not os.path.exists(lib):
                continue
            for f in os.listdir(lib):
                name, ext = os.path.splitext(os.path.basename(f))
                if not any(".%s" % x == ext for x in read_types):
                    continue
                
                if name in names:
                    continue
                
                if include_ext:
                    names.append(name + ext)
                else:
                    names.append(name)

        return names

    def detect_sub(self):
        """
        detects conformer information for a substituent by searching the
        substituent library
        """
        sub_lengths = Substituent.cache["lengths"]
        found = False
        cache_changed = False

        # temporarily detach end from sub so the connectivity is same as
        # for the library substituent by itself
        test_sub = self.copy()
        test_sub.refresh_ranks()

        for lib in [Substituent.AARON_LIBS, Substituent.BUILTIN]:
            if not os.path.exists(lib):
                continue
            for filename in os.listdir(lib):
                name, ext = os.path.splitext(filename)
                if not any(".%s" % x == ext for x in read_types):
                    continue

                # test number of atoms against cache
                if (
                        name in sub_lengths
                        and len(test_sub.atoms) != sub_lengths[name]
                ):
                    continue
                
                # use Geometry until we've done all the checks we can do without
                # determining connectivity
                # (for performance reasons)
                init_ref = Geometry(
                    os.path.join(lib, name + ext),
                    refresh_connected=False,
                    refresh_ranks=False,
                )
                # add to cache
                sub_lengths[name] = len(init_ref.atoms)
                cache_changed = True
    
                # want same number of atoms
                if len(test_sub.atoms) != len(init_ref.atoms):
                    continue

                # same number of each element
                ref_eles = [atom.element for atom in init_ref.atoms]
                test_eles = [atom.element for atom in test_sub.atoms]
                ref_counts = {ele:ref_eles.count(ele) for ele in set(ref_eles)}
                test_counts = {ele:test_eles.count(ele) for ele in set(test_eles)}
                if ref_counts != test_counts:
                    continue

                ref_sub = Substituent(init_ref, detect=False)
                ref_sub.name = name
                ref_sub.refresh_connected()
                ref_sub.refresh_ranks()
    
                for a, b in zip(sorted(test_sub.atoms), sorted(ref_sub.atoms)):
                    # want correct elements
                    if a.element != b.element:
                        break
                    # and correct connections
                    if len(a.connected) != len(b.connected):
                        break
                    # and correct connected elements
                    failed = False
                    for i, j in zip(
                            sorted([aa.element for aa in a.connected]),
                            sorted([bb.element for bb in b.connected]),
                    ):
                        if i != j:
                            failed = True
                            break
                    if failed:
                        break
                else:
                    # if found, save name and conf info
                    self.name = ref_sub.name
                    self.comment = ref_sub.comment
                    conf_info = re.search(r"CF:(\d+),(\d+)", ref_sub.comment)
                    if conf_info is not None:
                        self.conf_num = int(conf_info.group(1))
                        self.conf_angle = np.deg2rad(float(conf_info.group(2)))
                    found = True
                    break

        # update cache
        if cache_changed:
            Substituent.cache["lengths"] = sub_lengths
            if not os.path.exists(os.path.dirname(Substituent.CACHE_FILE)):
                os.makedirs(os.path.dirname(Substituent.CACHE_FILE))

            with open(Substituent.CACHE_FILE, "w") as f:
                json.dump(Substituent.cache, f)

        return found

    def sterimol(self, parameter="L", return_vector=False, radii="bondi"):
        """
        returns sterimol parameter value for the specified parameter
        see Verloop, A. and Tipker, J. (1976), Use of linear free energy
        related and other parameters in the study of fungicidal
        selectivity. Pestic. Sci., 7: 379-390. 
        (DOI: 10.1002/ps.2780070410)
        
        return_vector: bool/returns tuple(vector start, vector end) instead
        parameter (str) can be:
            "L"
            "B1"
            "B5"
        radii: "bondi" - Bondi vdW radii
               "umn"   - vdW radii from Mantina, Chamberlin, Valero, Cramer, and Truhlar
        """
        from AaronTools.finders import BondedTo

        if self.end is None:
            raise RuntimeError(
                "cannot calculate sterimol values for substituents without end"
            )


        atom1 = self.find(BondedTo(self.end))[0]
        atom2 = self.end

        # print(atom1.name, atom2.name)

        L_axis = atom2.bond(atom1)
        L_axis /= np.linalg.norm(L_axis)

        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII

        param_value = None
        vector = None
        if parameter == "B1":
            from scipy.spatial import ConvexHull
            # for B1, we're going to use ConvexHull to find the minimum width
            # to do this, we're going to project the substituent in a plane
            # perpendicular to the L-axis and get a set of points along the 
            # vdw radii of the atoms
            # ConvexHull will take these points and figure out which ones
            # are on the outside (vertices)
            # we then just need to find the minimum bounding box of the hull
            points = []
            ndx = []
            test_v = atom2.bond(self.atoms[-1])
            # just grab a random vector perpendicular to the L-axis
            # it doesn't matter really
            ip_vector = perp_vector(L_axis)
            x_vec = np.cross(ip_vector, L_axis)
            x_vec /= np.linalg.norm(x_vec)
            basis = np.array([x_vec, ip_vector, L_axis]).T

        for i, atom in enumerate(self.atoms):
            test_v = atom2.bond(atom)

            if parameter == "L":
                test_L = (
                    np.dot(test_v, L_axis)
                    - atom1.dist(atom2)
                    + VDW_RADII[atom1.element]
                    + VDW_RADII[atom.element]
                )
                if param_value is None or test_L > param_value:
                    param_value = test_L
                    start = atom1.coords - VDW_RADII[atom1.element] * L_axis
                    vector = (start, start + param_value * L_axis)

            elif parameter == "B1":
                r1 = radii_dict[atom.element]

                new_coords = np.dot(test_v, basis)
                # in plane coordinates - z-axis is L-axis, which
                # we don't care about for B1
                ip_coords = new_coords[0:2]
                for x in np.linspace(0, 2 * np.pi, num=250):
                    ndx.append(i)
                    v = ip_coords + r1 * np.array([np.cos(x), np.sin(x)])
                    points.append(v)

            elif parameter == "B5":
                b = np.dot(test_v, L_axis)
                test_B5_v = test_v - (b * L_axis)
                test_B5 = np.linalg.norm(test_B5_v) + radii_dict[atom.element]
                if param_value is None or test_B5 > param_value:
                    param_value = test_B5
                    start = atom.coords - test_B5_v
                    if np.linalg.norm(test_B5_v) > 3 * np.finfo(float).eps:
                        perp_vec = test_B5_v
                    else:
                        v_n = test_v / np.linalg.norm(test_v)
                        perp_vec = v_n[::-1]
                        perp_vec -= np.dot(v_n, perp_vec) * v_n

                    end = start + test_B5 * (
                        perp_vec / np.linalg.norm(perp_vec)
                    )

                    vector = (start, end)

        if parameter == "B1":
            points = np.array(points)

            # import matplotlib.pyplot as plt
            hull = ConvexHull(points)

            # plt.plot(points[:, 0], points[:, 1], 'o')
            # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'ro')

            # ax = plt.gca()
            # ax.set_aspect('equal')
            
            min_b1 = None
            # go through each edge, find a vector perpendicular to the one
            # defined by the edge that passes through the origin
            # the length of the shortest of these vectors is B1
            for i in range(0, len(hull.vertices) - 1):
                v_ndx_0, v_ndx_1 = hull.vertices[i:i + 2]
                v = points[v_ndx_1] - points[v_ndx_0]
                v /= np.linalg.norm(v)
                b = points[v_ndx_0]
                t = np.dot(b, v)
                perp = b - t * v
                # plt.plot(perp[0], perp[1], '*')
                test_b1 = np.linalg.norm(perp)
                if min_b1 is None or test_b1 < min_b1:
                    # min_ndx = hull.vertices[i:i + 2]
                    min_b1 = test_b1
                    b1_atom = self.atoms[ndx[v_ndx_0]]
                    test_v = atom2.bond(b1_atom)
                    test_B1_v = test_v - (np.dot(test_v, L_axis) * L_axis)
                    start = b1_atom.coords - test_B1_v
                    end = x_vec * perp[0] + ip_vector * perp[1]
                    end += start
                    
            vector = (start, end)
            # print(np.linalg.norm(start - end), min_b1)
            param_value = min_b1
            
            # plt.plot(points[min_ndx,0], points[min_ndx,1], 'gx')
            
            # plt.show()

        if return_vector:
            return vector
        return param_value

    def align_to_bond(self, bond):
        """
        align substituent to a bond vector
        """
        bond /= np.linalg.norm(bond)
        x_axis = np.array([1.0, 0.0, 0.0])
        rot_axis = np.cross(x_axis, bond)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.dot(bond, x_axis))
        self.rotate(rot_axis, angle)

    def sub_rotate(self, angle=None, reverse=False):
        """
        rotates substituent about bond w/ rest of geometry
        :angle: in radians
        """
        if angle is None:
            angle = self.conf_angle
        if reverse:
            angle *= -1
        axis = self.atoms[0].bond(self.end)
        self.rotate(axis, angle, center=self.end)

    def rebuild(self):
        start = self.atoms.pop(0)
        super().rebuild()
        self.atoms = [start] + self.atoms
