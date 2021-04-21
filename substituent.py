#!/usr/bin/env python3
"""for fragments attached to a structure by one bond"""

import json
import os
import re
import sys
from copy import deepcopy

import numpy as np

from AaronTools import addlogger
from AaronTools.const import AARONLIB, AARONTOOLS, BONDI_RADII, VDW_RADII
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import perp_vector, boltzmann_average


@addlogger
class Substituent(Geometry):
    """
    Attributes:
        name
        atoms
        end         the atom substituent is connected to
        conf_num    number of conformers
        conf_angle  angle to rotate by to make next conformer
    """

    LOG = None
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
                    self.LOG.warning("Conformer info not loaded for" + sub)
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
                self.LOG.warning("Conformer info not loaded for" + f)

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

    @staticmethod
    def weighted_sterimol(substituents, energies, temperature, *args, **kwargs):
        """
        returns Boltzmann-averaged sterimol parameters for the substituents
        substituents - list of Substituent instances
        energies - numpy array, energy in kcal/mol; ith energy corresponds to ith substituent
        temperature - temperature in K
        *args, **kwargs - passed to Substituent.sterimol()
        """
        CITATION = "doi:10.1021/acscatal.8b04043"
        Substituent.LOG.citation(CITATION)
        values = {
            "B1": [],
            "B2": [],
            "B3": [],
            "B4": [],
            "B5": [],
            "L": [],
        }
        
        rv = dict()
        
        for sub in substituents:
            data = sub.sterimol(*args, **kwargs)
            for key in data.keys():
                values[key].append(data[key])
        
        for key in values.keys():
            values[key] = np.array(values[key])
            rv[key] = boltzmann_average(energies, values[key], temperature)
        
        return rv

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
                cls.LOG.warning(
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
                ref_counts = {
                    ele: ref_eles.count(ele) for ele in set(ref_eles)
                }
                test_counts = {
                    ele: test_eles.count(ele) for ele in set(test_eles)
                }
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

    def sterimol(self, return_vector=False, radii="bondi", old_L=False):
        """
        returns sterimol parameter values in a dictionary
        keys are B1, B2, B3, B4, B5, and L
        see Verloop, A. and Tipker, J. (1976), Use of linear free energy
        related and other parameters in the study of fungicidal
        selectivity. Pestic. Sci., 7: 379-390.
        (DOI: 10.1002/ps.2780070410)

        return_vector: bool/returns dict of tuple(vector start, vector end) instead
        radii: "bondi" - Bondi vdW radii
               "umn"   - vdW radii from Mantina, Chamberlin, Valero, Cramer, and Truhlar
        old_L: bool - True: use original L (ideal bond length between first substituent
                            atom and hydrogen + 0.40 angstrom
                      False: use AaronTools definition

        AaronTools' definition of the L parameter is different than the original
        STERIMOL program. In STERIMOL, the van der Waals radii of the substituent is
        projected onto a plane parallel to the bond between the molecule and the substituent.
        The L parameter is 0.40 Å plus the distance from the first substituent atom to the
        outer van der Waals surface of the projection along the bond vector. This 0.40 Å is
        a correction for STERIMOL using a hydrogen to represent the molecule, when a carbon
        would be more likely. In AaronTools the substituent is projected the same, but L is
        calculated starting from the van der Waals radius of the first substituent atom
        instead. This means AaronTools will give the same L value even if the substituent
        is capped with something besides a hydrogen. When comparing AaronTools' L values
        with STERIMOL (using the same set of radii for the atoms), the values usually
        differ by < 0.1 Å.
        """
        from AaronTools.finders import BondedTo
        from scipy.spatial import ConvexHull

        CITATION = "doi:10.1002/ps.2780070410"
        self.LOG.citation(CITATION)

        if self.end is None:
            raise RuntimeError(
                "cannot calculate sterimol values for substituents without end"
            )

        atom1 = self.find(BondedTo(self.end))[0]
        atom2 = self.end

        if old_L:
            from AaronTools.atoms import Atom, BondOrder
            bo = BondOrder
            key = bo.key(atom1, Atom(element="H"))
            dx = bo.bonds[key]["1.0"] + 0.4

        # print(atom1.name, atom2.name)

        L_axis = atom2.bond(atom1)
        L_axis /= np.linalg.norm(L_axis)

        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII

        B1 = None
        B2 = None
        B3 = None
        B4 = None
        B5 = None
        L = None
        vector = {"B1": None, "B2": None, "B3": None, "B4": None, "B5": None, "L": None}
        # for B1, we're going to use ConvexHull to find the minimum distance
        # from one face of a bounding box
        # to do this, we're going to project the substituent in a plane
        # perpendicular to the L-axis and get a set of points along the
        # vdw radii of the atoms
        # ConvexHull will take these points and figure out which ones
        # are on the outside (vertices)
        # we then just need to find the bounding box with the minimum distance
        # from L-axis to one side of the box
        points = []
        ndx = []
        # just grab a random vector perpendicular to the L-axis
        # it doesn't matter really
        ip_vector = perp_vector(L_axis)
        x_vec = np.cross(ip_vector, L_axis)
        x_vec /= np.linalg.norm(x_vec)
        basis = np.array([x_vec, ip_vector, L_axis]).T

        for i, atom in enumerate(self.atoms):
            test_v = atom2.bond(atom)

            # L
            if old_L:
                test_L = (
                    np.dot(test_v, L_axis)
                    - atom1.dist(atom2)
                    + dx
                    + radii_dict[atom.element]
                )
            else:
                # overlap with L-axis and vector from 1st substituent
                # atom to this atom plus the difference between the sum of
                # the vdw radii and bond length of the 1st substituent atom
                # and the molecule's atom
                # the distance from atom2 is subtracted off b/c our L
                # goes from one end of the VDW radii to the other
                # distance from atom2 is included in test_v, so we need
                # to subtract it off
                test_L = (
                    np.dot(test_v, L_axis)
                    - atom1.dist(atom2)
                    + radii_dict[atom1.element]
                    + radii_dict[atom.element]
                )
            if L is None or test_L > L:
                L = test_L
                if old_L:
                    start = atom1.coords - dx * L_axis
                    vector["L"] = (start, start + L * L_axis)
                else:
                    start = atom1.coords - radii_dict[atom1.element] * L_axis
                    vector["L"] = (start, start + L * L_axis)

            # B1-4 stuff - we come back to this later
            r1 = radii_dict[atom.element]

            new_coords = np.dot(test_v, basis)
            # in plane coordinates - z-axis is L-axis, which
            # we don't care about for B1
            ip_coords = new_coords[0:2]
            for x in np.linspace(0, 2 * np.pi, num=250):
                ndx.append(i)
                v = ip_coords + r1 * np.array([np.cos(x), np.sin(x)])
                points.append(v)

            # B5
            # find distance along L-axis, then subtract this from vector from
            # vector from molecule to this atom to get the B5 vector
            # add the atom's radius to get the full B5
            b = np.dot(test_v, L_axis)
            test_B5_v = test_v - (b * L_axis)
            test_B5 = np.linalg.norm(test_B5_v) + radii_dict[atom.element]
            if B5 is None or test_B5 > B5:
                B5 = test_B5
                start = atom.coords - test_B5_v
                if np.linalg.norm(test_B5_v) > 3 * np.finfo(float).eps:
                    perp_vec = test_B5_v
                else:
                    # this atom might be along the L-axis, in which case use
                    # any vector orthogonal to L-axis
                    v_n = test_v / np.linalg.norm(test_v)
                    perp_vec = perp_vector(L_axis)
                    perp_vec -= np.dot(v_n, perp_vec) * v_n

                end = start + test_B5 * (
                    perp_vec / np.linalg.norm(perp_vec)
                )

                vector["B5"] = (start, end)

        points = np.array(points)

        hull = ConvexHull(points)

        # import matplotlib.pyplot as plt
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # plt.plot(0, 0, 'kx')
        # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'ro')

        # ax = plt.gca()
        # ax.set_aspect('equal')

        # go through each edge, find a vector perpendicular to the one
        # defined by the edge that passes through the origin
        # the length of the shortest of these vectors is B1
        for i in range(0, len(hull.vertices) - 1):
            # the vertices of the hull are organized in a counterclockwise
            # direction, so neighboring indices define an edge
            v_ndx_0, v_ndx_1 = hull.vertices[i:i + 2]
            # find 'tangent' of the edge
            v = points[v_ndx_1] - points[v_ndx_0]
            v /= np.linalg.norm(v)
            # find normal to this edge by projecting a vector from the
            # L axis to one of the points on the edge
            b = points[v_ndx_0]
            t = np.dot(b, v)
            perp = b - t * v
            # plt.plot(perp[0], perp[1], 'g*')
            test_b1 = np.linalg.norm(perp)
            if B1 is None or test_b1 < B1:
                B1 = test_b1
                # figure out vector from L axis to represent B1
                b1_atom = self.atoms[ndx[v_ndx_0]]
                test_v = atom2.bond(b1_atom)
                test_B1_v = test_v - (np.dot(test_v, L_axis) * L_axis)
                start = b1_atom.coords - test_B1_v
                end = x_vec * perp[0] + ip_vector * perp[1]
                end += start
                vector["B1"] = (start, end)

        # figure out B2-4
        # these need to be sorted in increasing order
        # for now, they will just be Bpar for the one opposite B1
        # and Bperp1 and Bperp2 for the ones perpendicular to B1
        b1_norm = end - start
        b1_norm /= np.linalg.norm(b1_norm)
        b1_perp = np.cross(L_axis, b1_norm)
        b1_perp /= np.linalg.norm(b1_perp)
        Bpar = None
        Bperp1 = None
        Bperp2 = None
        for atom in self.atoms:
            test_v = atom2.bond(atom)
            b = np.dot(test_v, L_axis)
            test_B_v = test_v - (b * L_axis)
            test_par_vec = np.dot(test_B_v, b1_norm) * b1_norm
            test_par_vec -= radii_dict[atom.element] * b1_norm
            start = atom.coords - test_B_v
            end = start + test_par_vec

            test_Bpar = np.linalg.norm(end - start)
            if Bpar is None or test_Bpar > Bpar:
                Bpar = test_Bpar
                par_vec = (start, end)

            perp_vec = np.dot(test_B_v, b1_perp) * b1_perp
            if np.dot(test_B_v, b1_perp) > 0 or np.isclose(np.dot(b1_perp, test_B_v), 0):
                test_perp_vec1 = perp_vec + radii_dict[atom.element] * b1_perp
                end = start + test_perp_vec1
                test_Bperp1 = np.linalg.norm(end - start)
                if Bperp1 is None or test_Bperp1 > Bperp1:
                    Bperp1 = test_Bperp1
                    perp_vec1 = (start, end)
            if np.dot(test_B_v, b1_perp) < 0 or np.isclose(np.dot(b1_perp, test_B_v), 0):
                test_perp_vec2 = perp_vec - radii_dict[atom.element] * b1_perp
                end = start + test_perp_vec2
                test_Bperp2 = np.linalg.norm(end - start)
                if Bperp2 is None or test_Bperp2 > Bperp2:
                    Bperp2 = test_Bperp2
                    perp_vec2 = (start, end)

        # put B2-4 in order
        i = 0
        Bs = [Bpar, Bperp1, Bperp2]
        Bvecs = [par_vec, perp_vec1, perp_vec2]
        while Bs:
            max_b = max(Bs)
            n = Bs.index(max_b)
            max_v = Bvecs.pop(n)
            Bs.pop(n)

            if i == 0:
                B4 = max_b
                vector["B4"] = max_v
            elif i == 1:
                B3 = max_b
                vector["B3"] = max_v
            elif i == 2:
                B2 = max_b
                vector["B2"] = max_v
            i += 1

        params = {
            "B1": B1,
            "B2": B2,
            "B3": B3,
            "B4": B4,
            "B5": B5,
            "L": L,
        }

        # plt.plot(points[min_ndx,0], points[min_ndx,1], 'g*')

        # plt.show()

        if return_vector:
            return vector
        return params

    def align_to_bond(self, bond):
        """
        align substituent to a bond vector
        """
        bond /= np.linalg.norm(bond)
        x_axis = np.array([1.0, 0.0, 0.0])
        rot_axis = np.cross(x_axis, bond)
        if np.linalg.norm(rot_axis) > 1e-4:
            rot_axis /= np.linalg.norm(rot_axis)
        else:
            rot_axis = np.array([0.0, 1.0, 0.0])
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
