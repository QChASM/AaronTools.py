from collections import deque
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.spatial import distance_matrix

from AaronTools.utils.utils import proj, perp_vector, unique_combinations



def dist(coords_i, coords_j):
    return np.sqrt(sum((coords_i - coords_j)**2))


def e_ij(coords_i, coords_j):
    v_ij = coords_j - coords_i
    r_ij = np.sqrt(sum(v_ij**2))
    return v_ij / r_ij


class Coordinate:
    
    n_values = 1

    def value(self, coords):
        raise NotImplementedError
    
    def s_vector(self, coords):
        raise NotImplementedError
    

class CartesianCoordinate(Coordinate):

    n_values = 3
    
    def __init__(self, atom):
        self.atom = atom
    
    def __eq__(self, other):
        if not isinstance(other, CartesianCoordinate):
            return False
        
        return self.atom == other.atom
    
    def __repr__(self):
        return "Cartesian coordinate for atom %i" % self.atom

    def value(self, coords):
        return coords[self.atom]
    
    def s_vector(self, coords):
        s = np.zeros((3, 3 * len(coords)))
        for i in range(0, 3):
            s[i, 3 * self.atom + i] = 1
        return s


class Bond(Coordinate):
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2
    
    def __eq__(self, other):
        if not isinstance(other, Bond):
            return False
        
        if self.atom1 == other.atom1 and self.atom2 == other.atom2:
            return True

        if self.atom1 == other.atom2 and self.atom2 == other.atom1:
            return True
        
        return False

    def __repr__(self):
        return "bond between atom %i and %i" % (self.atom1, self.atom2)

    def value(self, coords):
        return dist(coords[self.atom1], coords[self.atom2])
    
    def s_vector(self, coords):
        s = np.zeros(3 * len(coords))
        e_12 = e_ij(coords[self.atom1], coords[self.atom2])
        s[3 * self.atom1 : 3 * self.atom1 + 3] = -e_12
        s[3 * self.atom2 : 3 * self.atom2 + 3] = e_12
        return s


class InverseBond(Coordinate):
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2
   
    def __eq__(self, other):
        if not isinstance(other, InverseBond):
            return False
        
        if self.atom1 == other.atom1 and self.atom2 == other.atom2:
            return True

        if self.atom1 == other.atom2 and self.atom2 == other.atom1:
            return True
        
        return False

    def __repr__(self):
        return "inverse bond between atom %i and %i" % (self.atom1, self.atom2)

    def value(self, coords):
        return 1 / dist(coords[self.atom1], coords[self.atom2])
    
    def s_vector(self, coords):
        s = np.zeros(3 * len(coords))
        e_12 = e_ij(coords[self.atom1], coords[self.atom2])
        s[3 * self.atom1 : 3 * self.atom1 + 3] = -e_12
        s[3 * self.atom2 : 3 * self.atom2 + 3] = e_12
        return s * self.value(coords)


class Angle(Coordinate):
    def __init__(self, atom1, atom2, atom3):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
    
    def __eq__(self, other):
        if not isinstance(other, Angle):
            return False
        
        if all(
            getattr(self, attr) == getattr(other, attr) for attr in [
                "atom1", "atom2", "atom3",
            ]
        ):
            return True
            
        if all(
            getattr(self, attr1) == getattr(other, attr2) for (attr1, attr2) in zip(
                ["atom1", "atom2", "atom3"],
                ["atom3", "atom2", "atom1"],
            )
        ):
            return True
        
        return False

    def __repr__(self):
        return "angle %i-%i-%i" % (
            self.atom1,
            self.atom2,
            self.atom3,
        )

    def value(self, coords):
        return Angle.angle(
            coords[self.atom1], coords[self.atom2], coords[self.atom3]
        )

    @staticmethod
    def angle(x1, x2, x3):
        a2 = sum((x1 - x2)**2)
        b2 = sum((x3 - x2)**2)
        c2 = sum((x1 - x3)**2)
        theta = np.arccos((c2 - a2 - b2) / (-2 * np.sqrt(a2 * b2)))
        if np.isnan(theta):
            return np.pi
        return theta

    def s_vector(self, coords):
        a = dist(coords[self.atom1], coords[self.atom2])
        b = dist(coords[self.atom3], coords[self.atom2])
        s = np.zeros(3 * len(coords))
        a_ijk = self.value(coords)
        e_21 = e_ij(coords[self.atom2], coords[self.atom1])
        e_23 = e_ij(coords[self.atom2], coords[self.atom3])
        s[3 * self.atom1 : 3 * self.atom1 + 3] = (np.cos(a_ijk) * e_21 - e_23) / (a * np.sin(a_ijk))
        s[3 * self.atom3 : 3 * self.atom3 + 3] = (np.cos(a_ijk) * e_23 - e_21) / (b * np.sin(a_ijk))
        s[3 * self.atom2 : 3 * self.atom2 + 3] = -s[3 * self.atom1 : 3 * self.atom1 + 3]
        s[3 * self.atom2 : 3 * self.atom2 + 3] -= s[3 * self.atom3 : 3 * self.atom3 + 3]
        return s


class LinearAngle(Coordinate):
    
    n_values = 2
    
    def __init__(self, atom1, atom2, atom3):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3

    def __eq__(self, other):
        if not isinstance(other, LinearAngle):
            return False
        
        if all(
            getattr(self, attr) == getattr(other, attr) for attr in [
                "atom1", "atom2", "atom3",
            ]
        ):
            return True
            
        if all(
            getattr(self, attr1) == getattr(other, attr2) for (attr1, attr2) in zip(
                ["atom1", "atom2", "atom3"],
                ["atom3", "atom2", "atom1"],
            )
        ):
            return True
        
        return False

    def __repr__(self):
        return "linear angle %i-%i-%i" % (
            self.atom1,
            self.atom2,
            self.atom3,
        )

    def value(self, coords, v=None):
        if v is None:
            v = perp_vector(e_ij(coords[self.atom1], coords[self.atom2]))
    
        v2 = np.cross(v, coords[self.atom1] - coords[self.atom3])
        v2 /= np.linalg.norm(v2)

        val1_a = Angle.angle(
            coords[self.atom1], 
            coords[self.atom2],
            coords[self.atom2] + v,
        )
        val1_b = Angle.angle(
            coords[self.atom3], 
            coords[self.atom2],
            coords[self.atom2] + v,
        )
        
        val2_a = Angle.angle(
            coords[self.atom1], 
            coords[self.atom2],
            coords[self.atom2] + v2,
        )
        val2_b = Angle.angle(
            coords[self.atom3], 
            coords[self.atom2],
            coords[self.atom2] + v2,
        )

        return np.array([val1_a + val1_b, val2_a + val2_b])

    def s_vector(self, coords, v=None):
        s = np.zeros((2, 3 * len(coords)))

        if v is None:
            v = perp_vector(e_ij(coords[self.atom1], coords[self.atom2]))
    
        v2 = np.cross(v, coords[self.atom1] - coords[self.atom3])
        v2 /= np.linalg.norm(v2)
        
        s[0, 3 * self.atom1 : 3 * self.atom1 + 3] = -v / dist(coords[self.atom1], coords[self.atom2])
        s[0, 3 * self.atom3 : 3 * self.atom3 + 3] = -v / dist(coords[self.atom2], coords[self.atom3])
        s[0, 3 * self.atom2 : 3 * self.atom2 + 3] -= s[0, 3 * self.atom1 : 3 * self.atom1 + 3]
        s[0, 3 * self.atom2 : 3 * self.atom2 + 3] -= s[0, 3 * self.atom3 : 3 * self.atom3 + 3]
        
        s[1, 3 * self.atom1 : 3 * self.atom1 + 3] = -v2 / dist(coords[self.atom1], coords[self.atom2])
        s[1, 3 * self.atom3 : 3 * self.atom3 + 3] = -v2 / dist(coords[self.atom2], coords[self.atom3])
        s[1, 3 * self.atom2 : 3 * self.atom2 + 3] -= s[0, 3 * self.atom1 : 3 * self.atom1 + 3]
        s[1, 3 * self.atom2 : 3 * self.atom2 + 3] -= s[0, 3 * self.atom3 : 3 * self.atom3 + 3]
        
        return s


class OutOfPlaneBend(Coordinate):
    def __init__(self, central_atom, planar_atoms):
        self.central_atom = central_atom
        self.planar_atoms = planar_atoms

    def __repr__(self):
        return "atom %i angle out of %i-%i-%i plane" % (self.central_atom, *self.planar_atoms)

    def value(self, coords):
        e_23 = e_ij(coords[self.central_atom], coords[self.planar_atoms[1]])
        e_24 = e_ij(coords[self.central_atom], coords[self.planar_atoms[2]])
        e_12 = e_ij(coords[self.planar_atoms[0]], coords[self.central_atom])
        v = np.cross(e_23, e_24)
        v /= np.linalg.norm(v)
        pv = proj(e_12, v)
        n = np.linalg.norm(pv)
        k = np.arccos(n)
        if np.isnan(k):
            return 0
        return np.pi / 2 - k
    
    def s_vector(self, coords):
        s = np.zeros(3 * len(coords))
        
        e_23 = e_ij(coords[self.central_atom], self.planar_atoms[1])
        e_24 = e_ij(coords[self.central_atom], self.planar_atoms[2])
        e_21 = e_ij(coords[self.central_atom], self.planar_atoms[0])
        
        r_23 = dist(coords[self.central_atom], self.planar_atoms[1])
        r_24 = dist(coords[self.central_atom], self.planar_atoms[2])
        r_21 = dist(coords[self.central_atom], self.planar_atoms[0])
        
        phi_i = Angle.angle(
            coords[self.planar_atoms[2]], 
            coords[self.central_atom],
            coords[self.planar_atoms[1]],
        )
        phi_k = Angle.angle(
            coords[self.planar_atoms[0]], 
            coords[self.central_atom],
            coords[self.planar_atoms[2]],
        )
        phi_l = Angle.angle(
            coords[self.planar_atoms[0]], 
            coords[self.central_atom],
            coords[self.planar_atoms[1]],
        )
        
        theta = self.value(coords)
        
        v = np.cross(e_23, e_24) / np.sin(phi_i)
        
        s[3 * self.planar_atoms[0] : 3 * self.planar_atoms[0] + 3] = 1 / r_21
        s[3 * self.planar_atoms[0] : 3 * self.planar_atoms[0] + 3] *= v
        s[3 * self.planar_atoms[0] : 3 * self.planar_atoms[0] + 3] /= np.cos(theta)
        s[3 * self.planar_atoms[0] : 3 * self.planar_atoms[0] + 3] -= np.tan(theta) * e_21
        
        s[3 * self.planar_atoms[1] : 3 * self.planar_atoms[1] + 3] = 1 / r_23
        s[3 * self.planar_atoms[1] : 3 * self.planar_atoms[1] + 3] *= v
        s[3 * self.planar_atoms[1] : 3 * self.planar_atoms[1] + 3] /= np.cos(theta) * np.sin(phi_i) ** 2
        s[3 * self.planar_atoms[1] : 3 * self.planar_atoms[1] + 3] *= np.cos(phi_i) * np.cos(phi_k) - np.cos(phi_l)
        
        s[3 * self.planar_atoms[2] : 3 * self.planar_atoms[2] + 3] = 1 / r_24
        s[3 * self.planar_atoms[2] : 3 * self.planar_atoms[2] + 3] *= v
        s[3 * self.planar_atoms[2] : 3 * self.planar_atoms[2] + 3] /= np.cos(theta) * np.sin(phi_i) ** 2
        s[3 * self.planar_atoms[2] : 3 * self.planar_atoms[2] + 3] *= np.cos(phi_i) * np.cos(phi_l) - np.cos(phi_k)
        
        for i in self.planar_atoms:
            s[3 * self.central_atom : 3 * self.central_atom + 3] -= s[3 * i : 3 * i + 3]
        
        return s


class Torsion(Coordinate):
    def __init__(self, group1, atom1, atom2, group2, improper=False):
        self.group1 = group1
        self.atom1 = atom1
        self.atom2 = atom2
        self.group2 = group2
        self.improper = improper
    
    def __repr__(self):
        out = "torsional angle ("
        out += ",".join(str(i) for i in self.group1)
        out += ")-%i-%i-(" % (self.atom1, self.atom2)
        out += ",".join(str(i) for i in self.group2)
        out += ")"
        return out
    
    def __eq__(self, other):
        if not isinstance(other, Torsion):
            return False

        print(self, other)

        if self.improper:
            if self.group2[0] != other.group2[0]:
                print("not equal")
                return False
            
            self_plane = set([self.atom1, self.atom2, *self.group1])
            other_plane = set([other.atom1, other.atom2, *other.group1])
            if self_plane != other_plane:
                print("not equal")
                return False
            
            print("equal")
            return True

        for a1 in self.group1:
            if (
                self.group1.count(a1) != other.group1.count(a1)
            ) or (
                self.group1.count(a1) != other.group2.count(a1)
            ):
                print("not equal")
                return False

        for a2 in self.group2:
            if (
                self.group2.count(a2) != other.group2.count(a2)
            ) or (
                self.group2.count(a2) != other.group1.count(a2)
            ):
                print("not equal")
                return False

        if self.atom1 == other.atom1 and self.atom2 == other.atom2:
            print("equal")
            return True

        if self.atom1 == other.atom2 and self.atom2 == other.atom1:
            print("equal")
            return True
        
        print("not equal")
        return False

    def value(self, coords):
        e_i1 = e_ij(coords[self.group1[0]], coords[self.atom1])
        e_12 = e_ij(coords[self.atom1], coords[self.atom2])
        e_2l = e_ij(coords[self.atom2], coords[self.group2[0]])
        
        v1 = np.cross(e_i1, e_12)
        v2 = np.cross(e_12, e_2l)
        angle = np.cross(v1, v2)

        angle = np.dot(angle, e_12)
        angle = np.arctan2(
            angle,
            np.dot(v1, v2),
        )
        
        return angle
    
    def s_vector(self, coords):
        s = np.zeros(3 * len(coords))

        e_12 = e_ij(coords[self.atom1], coords[self.atom2])
        r_12 = dist(coords[self.atom1], coords[self.atom2])
        for i in self.group1:
            e_i1 = e_ij(coords[i], coords[self.atom1])
            a_i12 = Angle.angle(coords[i], coords[self.atom1], coords[self.atom2])
            r_i1 = dist(coords[i], coords[self.atom1])
            
            s[3 * i : 3 * i + 3] = -1 / len(self.group1)
            s[3 * i : 3 * i + 3] *= np.cross(e_i1, e_12) / r_i1
            s[3 * i : 3 * i + 3] /= np.sin(a_i12) ** 2
            
            s[3 * self.atom1 : 3 * self.atom1 + 3] += (
                (r_12 - r_i1 * np.cos(a_i12)) / (r_i1 * r_12 * np.sin(a_i12) ** 2)
            ) * np.cross(e_i1, e_12) / len(self.group1)
            
            s[3 * self.atom2 : 3 * self.atom2 + 3] += (
                (np.cos(a_i12) / (r_12 * np.sin(a_i12) ** 2))
            ) * np.cross(e_i1, e_12) / len(self.group1)
        
        for l in self.group2:
            e_l2 = e_ij(coords[l], coords[self.atom2])
            a_l21 = Angle.angle(coords[l], coords[self.atom2], coords[self.atom1])
            r_l2 = dist(coords[l], coords[self.atom2])
            
            s[3 * l : 3 * l + 3] = -1 / len(self.group2)
            s[3 * l : 3 * l + 3] *= np.cross(e_l2, -e_12) / r_l2
            s[3 * l : 3 * l + 3] /= np.sin(a_l21) ** 2
            
            s[3 * self.atom2 : 3 * self.atom2 + 3] += (
                (r_12 - r_l2 * np.cos(a_l21)) / (r_l2 * r_12 * np.sin(a_l21) ** 2)
            ) * np.cross(e_l2, -e_12) / len(self.group2)
            
            s[3 * self.atom1 : 3 * self.atom1 + 3] += (
                np.cos(a_l21) / (r_12 * np.sin(a_l21) ** 2)
            ) * np.cross(e_l2, -e_12) / len(self.group2)
        
        return s


class InternalCoordinateSet:
    def __init__(
        self,
        geometry,
        use_improper_torsions=True,
        use_inverse_bonds=False,
    ):
        self.geometry = geometry.copy(copy_atoms=True)
        geometry = self.geometry
        self.coordinates = {
            "bonds": [],
            "inverse bonds": [],
            "angles": [],
            "linear angles": [],
            "torsions": [],
            "out of plane bends": [],
        }
        fragments = []
        start = perf_counter()
        for atom in geometry.atoms:
            if not any(atom in fragment for fragment in fragments):
                new_fragment = geometry.get_fragment(atom, stop=atom)
                fragments.append(new_fragment)
                print(len(new_fragment))
        
        # TODO: prioritize H-bonds
        while len(fragments) > 1:
            min_d = None
            closest_ndx = None
            for i, frag1 in enumerate(fragments):
                coords1 = geometry.coordinates(frag1)
                for frag2 in fragments[:i]:
                    coords2 = geometry.coordinates(frag2)
                    dist = distance_matrix(coords1, coords2)
                    this_closest = np.min(dist)
                    if min_d is None or this_closest < min_d:
                        k = dist.argmin()
                        min_d = this_closest
                        ndx = np.where(dist == np.min(dist))
                        print(ndx)
                        closest_ndx = (frag1, frag2, [ndx[0][0], ndx[1][0]])
            
            frag1, frag2, ndx = closest_ndx
            frag1[ndx[0]].connected.add(frag2[ndx[1]])
            frag2[ndx[1]].connected.add(frag1[ndx[0]])
            frag1.extend(frag2)
            fragments.remove(frag2)
        
        stop = perf_counter()
        self.determine_coordinates(
            self.geometry,
            use_improper_torsions=use_improper_torsions,
            use_inverse_bonds=use_inverse_bonds,
        )

    @property
    def n_dimensions(self):
        n = 0
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                n += coord.n_values
        return n
    
    def remove_equivalent_coords(self):
        """
        removes angles for which there are equivalent linear angles
        """
        remove_coords = []
        for i, coord1 in enumerate(self.coordinates["angles"]):
            if not isinstance(coord1, Angle):
                continue
            for coord2 in self.coordinates["linear angles"]:
                if coord1.atom2 != coord2.atom1:
                    continue
                if coord1.atom1 == coord2.atom1 and coord1.atom3 == coord3.atom3:
                    remove_coords.append(i)
                if coord1.atom3 == coord2.atom1 and coord1.atom1 == coord3.atom3:
                    remove_coords.append(i)
    
        for i in remove_coords[::-1]:
            self.coordinates["angles"].pop(i)
    
    def determine_coordinates(
        self,
        geometry,
        use_improper_torsions=True,
        use_inverse_bonds=False,
    ):
        """
        determines the (redundant) internal coordinate set for the
        given AaronTools geometry
        use_improper_torsions: use improper torsional angles instead of
                               out of plane bend angle
        """
        ndx = {a: i for i, a in enumerate(geometry.atoms)}
        added_coords = False
        BondClass = Bond
        bond_type = "bonds"
        if use_inverse_bonds:
            BondClass = InverseBond
            bond_type = "inverse bonds"
 
        for atom1 in geometry.atoms:
            # TODO: replace the planarity check with SVD/eigenvalue decomp
            # of the inner product of these atoms coordinates with themselves
            # a small singular value would indicate planarity
            # might have to do unit vectors instead of coordinates 
            # relative to atom1
            if len(atom1.connected) > 2:
                vsepr, _ = atom1.get_vsepr()
                if vsepr and "planar" in vsepr:
                    for trio in combinations(atom1.connected, 3):
                        trio_ndx = [ndx[a] for a in trio]
                        if not use_improper_torsions:
                            oop_bend = OutOfPlaneBend(
                                ndx[atom1], [ndx[a] for a in trio]
                            )
                        else:
                            oop_bend = Torsion(
                                [trio_ndx[0]], trio_ndx[1], trio_ndx[2], [ndx[atom1]],
                                improper=True,
                            )
                        if not any(coord == oop_bend for coord in self.coordinates["out of plane bends"]):
                            added_coords = True
                            self.coordinates["out of plane bends"].append(oop_bend)
                            print("added oop bend:")
                            print("\t", atom1.name)
                            print("\t", [a.name for a in trio])
                
            for atom2 in atom1.connected:
                new_bond = BondClass(ndx[atom1], ndx[atom2])
                if not any(coord == new_bond for coord in self.coordinates[bond_type]):
                    added_coords = True
                    self.coordinates[bond_type].append(new_bond)
                    print("new bond:")
                    print("\t", atom1.name)
                    print("\t", atom2.name)
                    
                    # we need to skip linear things when adding torsions
                    # consider allene:
                    #  H1        H4
                    #   \       /
                    #    C1=C2=C3
                    #   /       \
                    # H2         H3
                    # we don't want to define a torsion involving C1-C2-C3
                    # because those are linear
                    # we should instead add torsions for H1-C1-C3-H4 etc.
                    # also for C1-C2-C3, this should be a linear bend and not
                    # a regular angle
                    # if atom1 or atom2 is C2, keep branching out until
                    # we find an atom with bonds that aren't colinear
                    # do this for atom1 and atom2
                    nonlinear_atoms_1 = []
                    linear_atoms_1 = []
                    exclude_atoms = [atom1, atom2]
                    stack = deque(list(atom1.connected - set(exclude_atoms)))
                    next_stack = deque([])
                    while stack:
                        next_connected = stack.popleft()
                        connected = next_connected.connected - set(exclude_atoms)
                        next_stack.extend(connected)
                        if abs(atom1.angle(next_connected, atom2) - np.pi) < (np.pi / 12):
                            linear_atoms_1.append(next_connected)
                        else:
                            nonlinear_atoms_1.append(next_connected)
                        if not stack and not nonlinear_atoms_1:
                            stack = next_stack
                            next_stack = deque([])

                    nonlinear_atoms_2 = []
                    linear_atoms_2 = []
                    exclude_atoms.extend(nonlinear_atoms_1)
                    stack = deque(list(atom2.connected - set(exclude_atoms)))
                    next_stack = deque([])
                    while stack:
                        next_connected = stack.popleft()
                        connected = next_connected.connected - set(exclude_atoms)
                        next_stack.extend(connected)
                        if abs(atom2.angle(next_connected, atom1) - np.pi) < (np.pi / 12):
                            linear_atoms_2.append(next_connected)
                        else:
                            nonlinear_atoms_2.append(next_connected)
                        if not stack and not nonlinear_atoms_2:
                            stack = next_stack
                            next_stack = deque([])
                    
                    if nonlinear_atoms_1 and nonlinear_atoms_2:
                        print("non linear groups", nonlinear_atoms_1, nonlinear_atoms_2)
                        central_atom1 = atom1
                        central_atom2 = atom2
                        for atom in linear_atoms_1:
                            if atom in nonlinear_atoms_1[0].connected:
                                central_atom1 = atom
                        for atom in linear_atoms_2:
                            if atom in nonlinear_atoms_2[0].connected:
                                central_atom2 = atom
                        
                        for a1, a2 in unique_combinations(
                            nonlinear_atoms_1, nonlinear_atoms_2
                        ):
                        # print(a1, a2)
                        # print(central_atom1, central_atom2)
                        # new_torsion = Torsion(
                        #     [ndx[a] for a in nonlinear_atoms_1],
                        #     ndx[central_atom1],
                        #     ndx[central_atom2],
                        #     [ndx[a] for a in nonlinear_atoms_2],
                        # )
                            new_torsion = Torsion(
                                [ndx[a1]], ndx[central_atom1], ndx[central_atom2], [ndx[a2]],
                            )
                            if not any(new_torsion == coord for coord in self.coordinates["torsions"]):
                                added_coords = True
                                self.coordinates["torsions"].append(new_torsion)
                                print("new torsion:")
                                print("\t", [a.name for a in nonlinear_atoms_1])
                                print("\t", central_atom1.name)
                                print("\t", central_atom2.name)
                                print("\t", [a.name for a in nonlinear_atoms_2])

                    else:
                        print("not adding torsions - not enough bonds to either", atom1, atom2)

                    for atom3 in set(nonlinear_atoms_1).intersection(atom1.connected):
                        new_angle = Angle(ndx[atom3], ndx[atom1], ndx[atom2])
                        if not any(coord == new_angle for coord in self.coordinates["angles"]):
                            added_coords = True
                            self.coordinates["angles"].append(new_angle)
                            print("new angle:")
                            print("\t", atom3.name)
                            print("\t", atom1.name)
                            print("\t", atom2.name)
                    
                    for atom3 in set(linear_atoms_1).intersection(atom1.connected):
                        new_linear_angle = LinearAngle(ndx[atom3], ndx[atom1], ndx[atom2])
                        if not any(coord == new_linear_angle for coord in self.coordinates["linear angles"]):
                            added_coords = True
                            self.coordinates["linear angles"].append(new_linear_angle)
                            print("new linear angle:")
                            print("\t", atom3.name)
                            print("\t", atom1.name)
                            print("\t", atom2.name)
                    
                    for atom3 in set(nonlinear_atoms_2).intersection(atom2.connected):
                        new_angle = Angle(ndx[atom3], ndx[atom2], ndx[atom1])
                        if not any(coord == new_angle for coord in self.coordinates["angles"]):
                            added_coords = True
                            self.coordinates["angles"].append(new_angle)
                            print("new angle:")
                            print("\t", atom3.name)
                            print("\t", atom2.name)
                            print("\t", atom1.name)
                    
                    for atom3 in set(linear_atoms_2).intersection(atom2.connected):
                        new_linear_angle = LinearAngle(ndx[atom3], ndx[atom2], ndx[atom1])
                        if not any(coord == new_linear_angle for coord in self.coordinates["linear angles"]):
                            added_coords = True
                            self.coordinates["linear angles"].append(new_linear_angle)
                            print("new linear angle:")
                            print("\t", atom3.name)
                            print("\t", atom2.name)
                            print("\t", atom1.name)

        print("there are %i internal coordinates" % self.n_dimensions)
        print("there would be %i cartesian coordinates" % (3 * len(geometry.atoms)))
        # print(len(self.coordinates["torsions"]))
        # asdf
        return added_coords
    
    def B_matrix(self, coords):
        """
        returns the B matrix (B_ij = dq_i/dx_j)
        """
        B = np.zeros((self.n_dimensions, 3 * len(self.geometry.atoms)))
        i = 0
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                B[i: i + coord.n_values] = coord.s_vector(coords)
                i += coord.n_values
        return B
    
    def values(self, coords):
        """
        returns vector with the values of internal coordinates for the
        given Cartesian coordinates (coords)
        """
        q = np.zeros(self.n_dimensions)
        i = 0
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                q[i: i + coord.n_values] = coord.value(coords)
                i += coord.n_values
        return q
    
    def difference(self, coords1, coords2):
        """
        difference between internal coordinates for coords1 and coords2
        coords1 -> q1
        coords2 -> q2
        returns q2 - q1 after adjusting difference in torsions to account for phase
        """
        q1 = self.values(coords1)
        q2 = self.values(coords2)
        dq = q2 - q1
        return self.adjust_phase(dq)

    def distance_by_type(self, coords1, coords2, p=2, max_bond_l=None):
        out = dict()
        i = 0
        dq = self.difference(coords1, coords2)
        for coord_type in self.coordinates:
            out.setdefault(coord_type, 0)
            for coord in self.coordinates[coord_type]:
                if coord_type == "bonds" and max_bond_l and dq[i] < max_bond_l:
                    out[coord_type] += np.sum(dq[i : i + coord.n_values] ** p)
                else:
                    out[coord_type] += np.sum(dq[i : i + coord.n_values] ** p)
                i += coord.n_values

            out[coord_type] = out[coord_type] ** (1 / p)
        
        return out

    def distance_by_type_q(self, q1, q2, p=2, max_bond_l=None):
        out = {
            "bonds": 0,
            "angles": 0,
            "linear angles": 0,
            "torsions": 0,
            "out of plane bends": 0,
        }
        i = 0
        dq = self.adjust_phase(q2 - q1)
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                if coord_type == "bonds" and max_bond_l and dq[i] < max_bond_l:
                    out[coord_type] += np.sum(dq[i : i + coord.n_values] ** p)
                else:
                    out[coord_type] += np.sum(dq[i : i + coord.n_values] ** p)
                i += coord.n_values

            out[coord_type] = out[coord_type] ** (1 / p)
        
        return out

    def adjust_phase(self, q):
        i = 0
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                if isinstance(coord, Torsion):
                    q[i] = np.arcsin(np.sin(q[i]))
                i += coord.n_values
        return q

    def apply_change(
        self, coords, dq,
        use_delocalized=True,
        max_iterations=10,
        convergence=1e-6,
    ):
        """
        change coords (Cartesian Nx3 array) by the specified 
        amount in internal coordinates (dq)
        use_delocalized: if True, step in delocalized internal coordinates
                         if False, step in redundant internal coordinates
        max_iterations: number of allowed cycles to try to meet the dq
        convergence: end cycles if differences between actual step and dq
                     is less than this amount
        """
        xi = x0 = best_struc = np.reshape(coords, -1)
        ddq = np.zeros(len(dq))
        x0 = best_struc
        smallest_dq = None
        for i in range(0, max_iterations):
            B = self.B_matrix(np.reshape(x0, coords.shape))
            if use_delocalized:
                G = np.matmul(B, B.T)
                w, v = np.linalg.eigh(G)
                U = v[:, -(len(x0) - 6):]
                Bd = np.matmul(U.T, B)
                ds = np.dot(U.T, dq)
                x1 = x0 + np.dot(np.linalg.pinv(Bd), ds)
            else:
                B_pinv = np.linalg.pinv(B)
                x1 = x0 + np.dot(B_pinv, dq)

            ddq = dq - self.difference(
                np.reshape(x0, coords.shape),
                np.reshape(x1, coords.shape),
            )

            if use_delocalized:
                dds = np.matmul(U.T, ddq)
                togo = np.linalg.norm(dds)
            else:
                togo = np.linalg.norm(ddq)
            
            x0 = x1
            dq = ddq
    
            if smallest_dq is None or togo < smallest_dq:
                best_struc = x1
                smallest_dq = togo
        
            if togo < convergence:
                break

        if togo < convergence:
            return np.reshape(x1, coords.shape), togo
        
        return np.reshape(best_struc, coords.shape), smallest_dq
