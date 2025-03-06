import math
from collections import deque
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.spatial import distance_matrix, distance
from scipy.linalg import pinv

from AaronTools.utils.utils import proj, perp_vector, unique_combinations


def _xyzzy_cross(v1, v2):
    """quick cross product of two 3-D vectors"""
    # this tends to be faster than np.cross
    out = np.zeros(3)
    out[0] = v1[1] * v2[2] - v1[2] * v2[1]
    out[1] = v1[2] * v2[0] - v1[0] * v2[2]
    out[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return out


def dist(coords_i, coords_j):
    """returns the distance from coords_i to coords_j"""
    return np.sqrt(sum((coords_i - coords_j)**2))


def all_e_ij(coords, precomputed_dist=None):
    """returns all pairwise unit vectors for coords (Nx3 array)"""
    e_ij_mat = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    if precomputed_dist is None:
        dist = distance.squareform(distance.pdist(coords))
    else:
        dist = precomputed_dist
    e_ij_mat = e_ij_mat / dist[:, :, np.newaxis]

    return e_ij_mat


def e_ij(coords_i, coords_j):
    """returns unit vector from coords_i to coords_j"""
    v_ij = coords_j - coords_i
    r_ij = np.sqrt(sum(v_ij**2))
    return v_ij / r_ij


class Coordinate:
    """
    base class for all other coordinates
    
    Attributes:
    
        * n_values - how many values the coordinate has (most have 1)
    """
    n_values = 1

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        """
        override this function to return the value(s) of your coordinate
        given the cartesian coordinates
        :param np.ndarray coords: Nx3 array of the molecule's cartesian coordinates
        :param np.ndarray precomputed_dist: NxN array of pairwise distances between all atoms
        :param np.ndarray precomputed_e_ij: NxNx3 array of pairwise unit vectors pointing from each atom to each other atom
        """
        raise NotImplementedError
    
    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        """
        override this function to return the derivative of cartesian coordinates
        with respect to your coordinate(s) given the cartesian coordinates
        :param np.ndarray coords: Nx3 array of the molecule's cartesian coordinates
        :param np.ndarray precomputed_dist: NxN array of pairwise distances between all atoms
        :param np.ndarray precomputed_e_ij: NxNx3 array of pairwise unit vectors pointing from each atom to each other atom
        """
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
    
    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
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

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        if precomputed_dist is None:
            return dist(coords[self.atom1], coords[self.atom2])
        return precomputed_dist[self.atom1, self.atom2]
    
    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        s = np.zeros(3 * len(coords))
        if precomputed_e_ij is not None:
            e_12 = precomputed_e_ij[self.atom2, self.atom1]
        else:
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

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        if precomputed_dist is None:
            return 1 / dist(coords[self.atom1], coords[self.atom2])
        return 1 / precomputed_dist[self.atom1, self.atom2]
    
    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        s = np.zeros(3 * len(coords))
        if precomputed_e_ij is not None:
            e_12 = precomputed_e_ij[self.atom2, self.atom1]
        else:
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
        
        if self.atom2 != other.atom2:
            return False
        
        if self.atom1 == other.atom1 and self.atom3 == other.atom3:
            return True

        if self.atom1 == other.atom3 and self.atom3 == other.atom1:
            return True
        
        return False

    def __repr__(self):
        return "angle %i-%i-%i" % (
            self.atom1,
            self.atom2,
            self.atom3,
        )

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        if precomputed_dist is None:
            return Angle.angle(
                coords[self.atom1], coords[self.atom2], coords[self.atom3]
            )
        return Angle.angle_from_sq_distances(
            precomputed_dist[self.atom1, self.atom2] ** 2,
            precomputed_dist[self.atom3, self.atom2] ** 2,
            precomputed_dist[self.atom1, self.atom3] ** 2
        )

    @staticmethod
    def angle(x1, x2, x3):
        """
        calculate the angle given three points
        x2 is the middle point, and this calculates the angle between x2-x1 and x2-x3
        """
        a2 = sum((x1 - x2)**2)
        b2 = sum((x3 - x2)**2)
        c2 = sum((x1 - x3)**2)
        return Angle.angle_from_sq_distances(a2, b2, c2)
    
    # this function is for a little bit extra speed when distances are
    # already available
    @staticmethod
    def angle_from_sq_distances(a2, b2, c2):
        """
        calculate angle based on square distances
        c2 is the squared side length opposite the vertex this function calculates the angle of
        """
        theta = math.acos((c2 - a2 - b2) / (-2 * math.sqrt(a2 * b2)))
        # need to check nan in case there's numerical issues
        if math.isnan(theta):
            return math.pi
        return theta

    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        if precomputed_dist is None:
            a = dist(coords[self.atom1], coords[self.atom2])
            b = dist(coords[self.atom3], coords[self.atom2])
            a_ijk = self.value(
                coords,
                precomputed_dist=precomputed_dist,
                precomputed_e_ij=precomputed_e_ij
            )
        else:
            a = precomputed_dist[self.atom1, self.atom2]
            b = precomputed_dist[self.atom3, self.atom2]
            a_ijk = Angle.angle_from_sq_distances(
                precomputed_dist[self.atom1, self.atom2] ** 2,
                precomputed_dist[self.atom3, self.atom2] ** 2,
                precomputed_dist[self.atom1, self.atom3] ** 2
            )
        s = np.zeros(3 * len(coords))

        sin_a_ijk = math.sin(a_ijk)
        cos_a_ijk = math.cos(a_ijk)
        if precomputed_e_ij is not None:
            e_21 = precomputed_e_ij[self.atom1, self.atom2]
            e_23 = precomputed_e_ij[self.atom3, self.atom2]
        else:
            e_21 = e_ij(coords[self.atom2], coords[self.atom1])
            e_23 = e_ij(coords[self.atom2], coords[self.atom3])
        s[3 * self.atom1 : 3 * self.atom1 + 3] = (cos_a_ijk * e_21 - e_23) / (a * sin_a_ijk)
        s[3 * self.atom3 : 3 * self.atom3 + 3] = (cos_a_ijk * e_23 - e_21) / (b * sin_a_ijk)
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
        
        if self.atom2 != other.atom2:
            return False
        
        if self.atom1 == other.atom1 and self.atom3 == other.atom3:
            return True

        if self.atom1 == other.atom3 and self.atom3 == other.atom1:
            return True
        
        return False

    def __repr__(self):
        return "linear angle %i-%i-%i" % (
            self.atom1,
            self.atom2,
            self.atom3,
        )

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        v = e_ij(coords[self.atom1], coords[self.atom2])
        w = e_ij(coords[self.atom3], coords[self.atom2])
        return np.dot(v, w)
    
        v2 = _xyzzy_cross(v, coords[self.atom1] - coords[self.atom3])
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

    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        s = np.zeros((2, 3 * len(coords)))

        if v is None:
            v = perp_vector(e_ij(coords[self.atom1], coords[self.atom2]))
    
        v2 = _xyzzy_cross(v, coords[self.atom1] - coords[self.atom3])
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


class LinearAngle4(Coordinate):
    n_values = 1

    def __init__(self, atom1, atom2, atom3, atom4):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4
    
    def __eq__(self, other):
        if not isinstance(other, LinearAngle4):
            return False
        
        if self.atom2 != other.atom2:
            return False
        
        if (
            self.atom1 == other.atom1 and
            self.atom3 == other.atom3
        ):
            return True

        if (
            self.atom1 == other.atom3 and
            self.atom3 == other.atom1
        ):
            return True

        return False

    def __repr__(self):
        return "linear angle %i-%i-%i (%i)" % (
            self.atom1,
            self.atom2,
            self.atom3,
            self.atom4
        )

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        if precomputed_dist is None:
            return \
                Angle.angle(
                    coords[self.atom1], coords[self.atom2], coords[self.atom4]
                ) + Angle.angle(
                    coords[self.atom4], coords[self.atom2], coords[self.atom3]
                )
        
        return Angle.angle_from_sq_distances(
            precomputed_dist[self.atom1, self.atom2] ** 2,
            precomputed_dist[self.atom4, self.atom2] ** 2,
            precomputed_dist[self.atom1, self.atom4] ** 2
        ) + Angle.angle_from_sq_distances(
            precomputed_dist[self.atom4, self.atom2] ** 2,
            precomputed_dist[self.atom3, self.atom2] ** 2,
            precomputed_dist[self.atom4, self.atom3] ** 2
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

    def s_vector(self, coords, precomputed_e_ij=None):
        a = dist(coords[self.atom1], coords[self.atom2])
        b = dist(coords[self.atom4], coords[self.atom2])
        s = np.zeros(3 * len(coords))
        a_ijk = Angle.angle(coords[self.atom1], coords[self.atom2], coords[self.atom4])
        e_21 = e_ij(coords[self.atom2], coords[self.atom1])
        e_23 = e_ij(coords[self.atom2], coords[self.atom4])
        s[3 * self.atom1 : 3 * self.atom1 + 3] = (np.cos(a_ijk) * e_21 - e_23) / (a * np.sin(a_ijk))
        s[3 * self.atom4 : 3 * self.atom4 + 3] = (np.cos(a_ijk) * e_23 - e_21) / (b * np.sin(a_ijk))
        s[3 * self.atom2 : 3 * self.atom2 + 3] = -s[3 * self.atom1 : 3 * self.atom1 + 3]
        s[3 * self.atom2 : 3 * self.atom2 + 3] -= s[3 * self.atom4 : 3 * self.atom4 + 3]
        
        a = dist(coords[self.atom4], coords[self.atom2])
        b = dist(coords[self.atom3], coords[self.atom2])
        a_ijk = Angle.angle(coords[self.atom4], coords[self.atom2], coords[self.atom3])
        e_21 = e_ij(coords[self.atom2], coords[self.atom4])
        e_23 = e_ij(coords[self.atom2], coords[self.atom3])
        s[3 * self.atom4 : 3 * self.atom4 + 3] += (np.cos(a_ijk) * e_21 - e_23) / (a * np.sin(a_ijk))
        s[3 * self.atom3 : 3 * self.atom3 + 3] += (np.cos(a_ijk) * e_23 - e_21) / (b * np.sin(a_ijk))
        s[3 * self.atom2 : 3 * self.atom2 + 3] += -s[3 * self.atom4 : 3 * self.atom4 + 3]
        s[3 * self.atom2 : 3 * self.atom2 + 3] -= s[3 * self.atom3 : 3 * self.atom3 + 3]
        return s


class OutOfPlaneBend(Coordinate):
    def __init__(self, central_atom, planar_atoms):
        self.central_atom = central_atom
        self.planar_atoms = planar_atoms

    def __repr__(self):
        return "atom %i angle out of %i-%i-%i plane" % (self.central_atom, *self.planar_atoms)

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        e_23 = e_ij(coords[self.central_atom], coords[self.planar_atoms[1]])
        e_24 = e_ij(coords[self.central_atom], coords[self.planar_atoms[2]])
        e_12 = e_ij(coords[self.planar_atoms[0]], coords[self.central_atom])
        v = _xyzzy_cross(e_23, e_24)
        v /= np.linalg.norm(v)
        pv = proj(e_12, v)
        n = np.linalg.norm(pv)
        k = np.arccos(n)
        if np.isnan(k):
            return 0
        return np.pi / 2 - k
    
    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        s = np.zeros(3 * len(coords))
        
        if precomputed_e_ij is None:
            e_23 = precomputed_e_ij[coords[self.central_atom], self.planar_atoms[1]]
            e_24 = precomputed_e_ij[coords[self.central_atom], self.planar_atoms[2]]
            e_21 = precomputed_e_ij[coords[self.central_atom], self.planar_atoms[0]]
        else:
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
        
        v = _xyzzy_cross(e_23, e_24) / np.sin(phi_i)
        
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

        if self.improper:
            if self.group2[0] != other.group2[0]:
                return False
            
            self_plane = set([self.atom1, self.atom2, *self.group1])
            other_plane = set([other.atom1, other.atom2, *other.group1])
            if self_plane != other_plane:
                return False
            
            return True

        if (
            self.atom1 != other.atom1 and self.atom2 != other.atom2
        ) and (
            self.atom1 != other.atom2 and self.atom2 != other.atom1
        ):
            return False

        for a1 in self.group1:
            if a1 not in other.group1 or a1 not in other.group2:
                return False

        for a2 in self.group2:
            if a2 not in other.group2 or a2 not in other.group1:
                return False

        return True

    def value(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        if precomputed_e_ij is None:
            e_i1 = e_ij(coords[self.group1[0]], coords[self.atom1])
            e_12 = e_ij(coords[self.atom1], coords[self.atom2])
            e_2l = e_ij(coords[self.atom2], coords[self.group2[0]])
        else:
            e_i1 = precomputed_e_ij[self.atom1, self.group1[0]]
            e_12 = precomputed_e_ij[self.atom2, self.atom1]
            e_2l = precomputed_e_ij[self.group2[0], self.atom2]
        
        v1 = _xyzzy_cross(e_i1, e_12)
        v2 = _xyzzy_cross(e_12, e_2l)
        angle = _xyzzy_cross(v1, v2)

        angle = np.dot(angle, e_12)
        angle = math.atan2(
            angle,
            np.dot(v1, v2),
        )
        
        return angle
    
    def s_vector(self, coords, precomputed_dist=None, precomputed_e_ij=None):
        s = np.zeros(3 * len(coords))

        if precomputed_e_ij is not None:
            e_12 = precomputed_e_ij[self.atom2, self.atom1]
        else:
            e_12 = e_ij(coords[self.atom1], coords[self.atom2])
        
        if precomputed_dist is None:
            r_12 = dist(coords[self.atom1], coords[self.atom2])
        else:
            r_12 = precomputed_dist[self.atom1, self.atom2]
        
        for i in self.group1:
            if precomputed_e_ij is not None:
                e_i1 = precomputed_e_ij[self.atom1, i]
            else:
                e_i1 = e_ij(coords[i], coords[self.atom1])
            if precomputed_dist is None:
                a_i12 = Angle.angle(coords[i], coords[self.atom1], coords[self.atom2])
                r_i1 = dist(coords[i], coords[self.atom1])
            else:
                a_i12 = Angle.angle_from_sq_distances(
                    precomputed_dist[i, self.atom1] ** 2,
                    precomputed_dist[self.atom2, self.atom1] ** 2,
                    precomputed_dist[i, self.atom2] ** 2
                )
                r_i1 = precomputed_dist[i, self.atom1]
            
            sin_i12_squared = math.sin(a_i12) ** 2
            cos_i12 = math.cos(a_i12)
            e_i1_cross_12 = _xyzzy_cross(e_i1, e_12)
            
            s[3 * i : 3 * i + 3] = -1 / len(self.group1)
            s[3 * i : 3 * i + 3] *= e_i1_cross_12 / r_i1
            s[3 * i : 3 * i + 3] /= sin_i12_squared
            
            s[3 * self.atom1 : 3 * self.atom1 + 3] += (
                (r_12 - r_i1 * cos_i12) / (r_i1 * r_12 * sin_i12_squared)
            ) * e_i1_cross_12 / len(self.group1)
            
            s[3 * self.atom2 : 3 * self.atom2 + 3] += (
                (cos_i12 / (r_12 * sin_i12_squared))
            ) * e_i1_cross_12 / len(self.group1)
        
        for l in self.group2:
            if precomputed_e_ij is not None:
                e_l2 = precomputed_e_ij[self.atom2, l]
            else:
                e_l2 = e_ij(coords[l], coords[self.atom2])
            
            if precomputed_dist is None:
                a_l21 = Angle.angle(coords[l], coords[self.atom2], coords[self.atom1])
                r_l2 = dist(coords[l], coords[self.atom2])
            else:
                a_l21 = Angle.angle_from_sq_distances(
                    precomputed_dist[l, self.atom2] ** 2,
                    precomputed_dist[self.atom1, self.atom2] ** 2,
                    precomputed_dist[l, self.atom1] ** 2
                )
                r_l2 = precomputed_dist[l, self.atom2]
            
            sin_al21_squared = math.sin(a_l21) ** 2
            cos_al21 = math.cos(a_l21)
            e_l2_cross_12 = _xyzzy_cross(e_l2, -e_12)
            
            s[3 * l : 3 * l + 3] = -1 / len(self.group2)
            s[3 * l : 3 * l + 3] *= e_l2_cross_12 / r_l2
            s[3 * l : 3 * l + 3] /= sin_al21_squared
            
            s[3 * self.atom2 : 3 * self.atom2 + 3] += (
                (r_12 - r_l2 * cos_al21) / (r_l2 * r_12 * sin_al21_squared)
            ) * e_l2_cross_12 / len(self.group2)
            
            s[3 * self.atom1 : 3 * self.atom1 + 3] += (
                cos_al21 / (r_12 * sin_al21_squared)
            ) * e_l2_cross_12 / len(self.group2)
        
        return s


class InternalCoordinateSet:
    def __init__(
        self,
        geometry,
        use_improper_torsions=True,
        use_inverse_bonds=False,
        torsion_type="combine-similar",
        oop_type="none",
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
            torsion_type=torsion_type,
            oop_type=oop_type,
        )
        self.supplement_missing(geometry.coords)

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
                try:
                    if coord1.atom2 != coord2.atom2:
                        continue
                except AttributeError:
                    continue
                if coord1.atom1 == coord2.atom1 and coord1.atom3 == coord2.atom3:
                    remove_coords.append(i)
                if coord1.atom3 == coord2.atom1 and coord1.atom1 == coord2.atom3:
                    remove_coords.append(i)
    
        print("removing", len(remove_coords), "coordinates")
        for i in remove_coords[::-1]:
            self.coordinates["angles"].pop(i)
    
    def determine_coordinates(
        self,
        geometry,
        use_improper_torsions=True,
        use_inverse_bonds=False,
        torsion_type="combine-similar",
        oop_type="none",
    ):
        """
        determines the (redundant) internal coordinate set for the
        given AaronTools geometry
        use_improper_torsions: use improper torsional angles instead of
                               out of plane bend angle
        """
        coords = geometry.coords
        ndx = {a: i for i, a in enumerate(geometry.atoms)}
        added_coords = False
        BondClass = Bond
        bond_type = "bonds"
        if use_inverse_bonds:
            BondClass = InverseBond
            bond_type = "inverse bonds"
 
        ranks = geometry.canonical_rank(break_ties=False)
 
        for atom1 in geometry.atoms:
            # TODO: replace the planarity check with SVD/eigenvalue decomp
            # of the inner product of these atoms coordinates with themselves
            # a small singular value would indicate planarity
            # might have to do unit vectors instead of coordinates 
            # relative to atom1
            if oop_type != "none":
                if len(atom1.connected) > 2:
                    vsepr, _ = atom1.get_vsepr()
                    # print(atom1, vsepr)
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
                                # print("added oop bend:")
                                # print("\t", atom1.name)
                                # print("\t", [a.name for a in trio])
                
            for atom2 in atom1.connected:
                new_bond = BondClass(ndx[atom1], ndx[atom2])
                if not any(coord == new_bond for coord in self.coordinates[bond_type]):
                    added_coords = True
                    self.coordinates[bond_type].append(new_bond)
                    # print("new bond:")
                    # print("\t", atom1.name)
                    # print("\t", atom2.name)
                    
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
                        # print("non linear groups", nonlinear_atoms_1, nonlinear_atoms_2)
                        central_atom1 = atom1
                        central_atom2 = atom2
                        for atom in linear_atoms_1:
                            if atom in nonlinear_atoms_1[0].connected:
                                central_atom1 = atom
                        for atom in linear_atoms_2:
                            if atom in nonlinear_atoms_2[0].connected:
                                central_atom2 = atom
                        

                        if torsion_type == "combine-similar":
                            nonlinear_groups_1 = []
                            for atom in nonlinear_atoms_1:
                                for group in nonlinear_groups_1:
                                    if ranks[ndx[group[0]]] == ranks[ndx[atom]]:
                                        group.append(atom)
                                        break
                                else:
                                    nonlinear_groups_1.append([atom])
    
                            nonlinear_groups_2 = []
                            for atom in nonlinear_atoms_2:
                                for group in nonlinear_groups_2:
                                    if ranks[ndx[group[0]]] == ranks[ndx[atom]]:
                                        group.append(atom)
                                        break
                                else:
                                    nonlinear_groups_2.append([atom])
                            
                            for group1, group2 in unique_combinations(
                                nonlinear_groups_1, nonlinear_groups_2
                            ):
                                new_torsion = Torsion(
                                    [ndx[a] for a in group1],
                                    ndx[central_atom1],
                                    ndx[central_atom2],
                                    [ndx[a] for a in group2],
                                )
                                if not any(new_torsion == coord for coord in self.coordinates["torsions"]):
                                    added_coords = True
                                    self.coordinates["torsions"].append(new_torsion)
                        
                        elif torsion_type == "combine-all":
                            new_torsion = Torsion(
                                [ndx[a] for a in nonlinear_atoms_1],
                                ndx[central_atom1],
                                ndx[central_atom2],
                                [ndx[a] for a in nonlinear_atoms_2],
                            )
                            if not any(new_torsion == coord for coord in self.coordinates["torsions"]):
                                added_coords = True
                                self.coordinates["torsions"].append(new_torsion)

                        elif torsion_type == "all":
                            for a1, a2 in unique_combinations(
                                nonlinear_atoms_1, nonlinear_atoms_2
                            ):
                            # print(a1, a2)
                            # print(central_atom1, central_atom2)

                                new_torsion = Torsion(
                                    [ndx[a1]], ndx[central_atom1], ndx[central_atom2], [ndx[a2]],
                                )
                                if not any(new_torsion == coord for coord in self.coordinates["torsions"]):
                                    added_coords = True
                                    self.coordinates["torsions"].append(new_torsion)
                                    # print("new torsion:")
                                    # print("\t", [a.name for a in nonlinear_atoms_1])
                                    # print("\t", central_atom1.name)
                                    # print("\t", central_atom2.name)
                                    # print("\t", [a.name for a in nonlinear_atoms_2])
                        else:
                            raise NotImplementedError("torsion_type not known: %s" % torsion_type)

                    # else:
                    #     print("not adding torsions - not enough bonds to either", atom1, atom2)

                    for atom3 in set(nonlinear_atoms_1).intersection(atom1.connected):
                        if atom3 is atom2:
                            continue
                        new_angle = Angle(ndx[atom3], ndx[atom1], ndx[atom2])
                        if not any(coord == new_angle for coord in self.coordinates["angles"]):
                            added_coords = True
                            self.coordinates["angles"].append(new_angle)
                            # print("new angle:")
                            # print("\t", atom3.name)
                            # print("\t", atom1.name)
                            # print("\t", atom2.name)
                    
                    if linear_atoms_1:
                        added_angles = []
                        for atom3 in geometry.get_fragment(atom1, stop=atom2):
                            if atom3 in linear_atoms_1:
                                continue
                            if (
                                atom3 is atom1 or
                                atom3 is atom2
                            ):
                                continue
                            # print(1, atom2, atom1, atom3)
                            angle = atom1.angle(atom2, atom3)
                            if angle > 0.1 and angle < 0.9 * np.pi:
                                # print("angle is not linear", added_angles)
                                for angle in added_angles:
                                    if angle.atom2 != ndx[atom1]:
                                        continue
                                    a_ij = e_ij(coords[angle.atom2, :], coords[angle.atom1, :])
                                    a_jk = e_ij(coords[angle.atom2, :], coords[angle.atom3, :])
                                    va = _xyzzy_cross(a_ij, a_jk)
                                    b_ij = e_ij(atom1.coords, atom2.coords)
                                    b_jk = e_ij(atom1.coords, atom3.coords)
                                    vb = _xyzzy_cross(b_ij, b_jk)
                                    # print(va, vb)
                                    d = np.dot(va, vb)
                                    if abs(d) > 0.8 or abs(d) < 0.2:
                                        # print("angle in plane already added")
                                        break
                                    else:
                                        # print("angle is not in same plane as", angle, d)
                                        pass

                                else:
                                    new_angle = Angle(ndx[atom2], ndx[atom1], ndx[atom3])
                                    if not any(coord == new_angle for coord in self.coordinates["linear angles"]):
                                        added_coords = True
                                        self.coordinates["linear angles"].append(new_angle)
                                        added_angles.append(new_angle)
                                        # print("1 added linear angle half", new_angle)
                                    else:
                                        # print("angle exists")
                                        pass

                            else:
                                # print("angle is linear", angle)
                                pass

                    for atom3 in set(nonlinear_atoms_2).intersection(atom2.connected):
                        if atom3 is atom1:
                            continue
                        new_angle = Angle(ndx[atom3], ndx[atom2], ndx[atom1])
                        if not any(coord == new_angle for coord in self.coordinates["angles"]):
                            added_coords = True
                            self.coordinates["angles"].append(new_angle)
                            # print("new angle:")
                            # print("\t", atom3.name)
                            # print("\t", atom2.name)
                            # print("\t", atom1.name)
                    
                    if linear_atoms_2:
                        added_angles = []
                        for atom3 in geometry.get_fragment(atom2, stop=atom1):
                            if atom3 in linear_atoms_1:
                                continue
                            if (
                                atom3 is atom1 or
                                atom3 is atom2
                            ):
                                continue
                            # print(2, atom1, atom2, atom3)
                            angle = atom2.angle(atom1, atom3)
                            if angle > 0.1 and angle < 0.9 * np.pi:
                                for angle in added_angles:
                                    if angle.atom2 != ndx[atom2]:
                                        continue
                                    a_ij = e_ij(coords[angle.atom2, :], coords[angle.atom1, :])
                                    a_jk = e_ij(coords[angle.atom2, :], coords[angle.atom3, :])
                                    va = _xyzzy_cross(a_ij, a_jk)
                                    b_ij = e_ij(atom2.coords, atom1.coords)
                                    b_jk = e_ij(atom2.coords, atom3.coords)
                                    vb = _xyzzy_cross(b_ij, b_jk)
                                    d = np.dot(va, vb)
                                    # print(va, vb)
                                    if abs(d) > 0.8 or abs(d) < 0.2:
                                        # print("angle in plane already added")
                                        break
                                    else:
                                        # print("angle is not in same plane as", angle, d)
                                        pass
                                else:
                                    new_angle = Angle(ndx[atom1], ndx[atom2], ndx[atom3])
                                    if not any(coord == new_angle for coord in self.coordinates["linear angles"]):
                                        added_coords = True
                                        self.coordinates["linear angles"].append(new_angle)
                                        added_angles.append(new_angle)
                                        # print("2 added linear angle half", new_angle)
                                    else:
                                        # print("angle exists")
                                        pass

                            else:
                                # print("angle is linear", angle)
                                pass

        # print("there are %i internal coordinates" % self.n_dimensions)
        # print("there would be %i cartesian coordinates" % (3 * len(geometry.atoms)))
        # print(len(self.coordinates["torsions"]))
        # asdf
        return added_coords
    
    def supplement_missing(self, coords):
        """adds cartesian coordinates to atoms where the internal coordinates don't work well"""
        # try displacing every cartesian coordinate
        # transform the displacement from cartesian to internal
        # transform the internal back to cartesian
        # the original and transformed cartesian displacements
        # are aligned and compared
        # if they differ significantly, a cartesian coordinate
        # is added for that atom
        B = self.B_matrix(coords)
        B_inv = np.linalg.pinv(B)
        x_step = 1e-2
        dx = np.zeros(np.prod(coords.shape))
        bad_spots = set()
        for i in range(0, len(coords)):
            for j in range(0, 3):
                dx[3 * i + j] += x_step
                dq = np.matmul(B, dx)
                dx_back = np.matmul(B_inv, dq)
                dx_r = np.reshape(dx, (len(coords), 3))
                dx_back_r = np.reshape(dx_back, (len(coords), 3))
                dx_r_com = np.mean(dx_r, axis=0)
                dx_r -= dx_r_com
                dx_back_r_com = np.mean(dx_back_r, axis=0)
                dx_back_r -= dx_back_r_com
                H = np.matmul(dx_r.T, dx_back_r)
                u, w, vh = np.linalg.svd(H)
                e = np.eye(3)
                if np.linalg.det(u) * np.linalg.det(vh) < 0:
                    e[2, 2] = -1
                R = np.matmul(np.matmul(u, e), vh)
                dx_m = np.matmul(dx_r, R)
                diff = np.reshape(dx_back_r - dx_m, -1)
                dx[3 * i + j] = 0
                for k, x in enumerate(diff):
                    if abs(x) > x_step / 2:
                        # print(k // 3, k - 3 * (k // 3), x)
                        bad_spots.add(k // 3)
        
        for k in bad_spots:
            # print("adding cartesian for", k)
            self.coordinates.setdefault("cartesian", [])
            self.coordinates["cartesian"].append(CartesianCoordinate(k))
    
    def B_matrix(self, coords):
        """
        returns the B matrix (B_ij = dq_i/dx_j)
        """
        d = distance.squareform(distance.pdist(coords))
        e_ij_mat = all_e_ij(coords, precomputed_dist=d)

        B = np.zeros((self.n_dimensions, 3 * len(coords)))
        i = 0
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                B[i: i + coord.n_values] = coord.s_vector(
                    coords, precomputed_dist=d, precomputed_e_ij=e_ij_mat
                )
                i += coord.n_values
        return B
    
    def values(self, coords):
        """
        returns vector with the values of internal coordinates for the
        given Cartesian coordinates (coords)
        """
        d = distance.squareform(distance.pdist(coords))
        e_ij_mat = all_e_ij(coords, precomputed_dist=d)

        q = np.zeros(self.n_dimensions)
        i = 0
        for coord_type in self.coordinates:
            for coord in self.coordinates[coord_type]:
                q[i: i + coord.n_values] = coord.value(
                    coords, precomputed_dist=d, precomputed_e_ij=e_ij_mat
                )
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
                    if q[i] > 2 * np.pi:
                        q[i] -= (q[i] // (np.pi)) * np.pi
                    elif q[i] < -np.pi:
                        q[i] -= (q[i] // (np.pi)) * np.pi
                i += coord.n_values
        return q

    def apply_change_2(
        self, coords, dq,
        convergence=1e-10,
        max_iterations=100,
        step_limit=0.35,
        use_delocalized=False,
        debug=False,
    ):
        """
        alternative method to change internal coordinates
        This method aims to slowly approach the desired dq
        rather than move immediately by dq
        change coords (Cartesian Nx3 array) by the specified 
        amount in internal coordinates (dq)
        max_iterations: number of allowed cycles to try to meet the dq
        step_limit: if any cartesian component of the step exceeds this amount,
                    the step will be scaled down and the remaining difference
                    will (hopefully) be handled on the next iteration
                    Setting this option to e.g. 0.2 can improve stability for
                    larger changes
        convergence: end cycles if differences between actual step and dq
                     is less than this amount
        """
        ndx = dq.nonzero()[0]
        x0 = best_struc = np.reshape(coords, -1)
        current_q = self.values(coords)
        target_q = current_q + dq
        prev_dx = np.zeros(len(x0))
        dq = self.adjust_phase(target_q - current_q)
        if debug:
            with open("opt.xyz", "w") as f:
                f.write("%i\n0.0\n" % len(coords))
                for c in coords:
                    f.write("C   %.3f   %.3f   %.3f\n" % tuple(c))
        for i in range(0, max_iterations):
            # for bigger changes, use harmonic
            if max(np.abs(dq)) > 0.5:
                force = 0.5 * dq * (dq ** 2)

            # for smaller changes, use linear
            else:
                force = dq

            B = self.B_matrix(np.reshape(x0, coords.shape))
            if use_delocalized:
                G = np.matmul(B, B.T)
                w, v = np.linalg.eigh(G)
                U = v[:, -(len(x0) - 6):]
                Bd = np.matmul(U.T, B)
                ds = np.dot(U.T, force)
                dx = np.dot(np.linalg.pinv(Bd), ds)
            else:
                B_pinv = np.linalg.pinv(B)
                dx = np.dot(B_pinv, force)
            # dx = np.dot(B_pinv, force)
            max_step = max(np.abs(dx))
            if max_step > step_limit:
                dx *= step_limit / max_step

            # basically steepest descent with momentum
            x0 = x0 + 0.9 * dx + 0.1 * prev_dx
            prev_dx = dx

            current_coords = np.reshape(x0, coords.shape)
            if debug:
                with open("opt.xyz", "a") as f:
                    f.write("%i\n%i.0\n" % (len(current_coords), i + 1))
                    for c in current_coords:
                        f.write("C   %.3f   %.3f   %.3f\n" % tuple(c))

            current_q = self.values(current_coords)
            dq = self.adjust_phase(target_q - current_q)

            togo = np.linalg.norm(dq)
            if debug:
                print("iter", i, "error", togo)
            if togo < convergence:
                break
        
        return np.reshape(x0, coords.shape), togo

    def apply_change(
        self, coords, dq,
        use_delocalized=True,
        max_iterations=100,
        convergence=1e-10,
        step_limit=None,
        debug=False,
    ):
        """
        change coords (Cartesian Nx3 array) by the specified 
        amount in internal coordinates (dq)
        use_delocalized: if True, step in delocalized internal coordinates
                         if False, step in redundant internal coordinates
        max_iterations: number of allowed cycles to try to meet the dq
        step_limit: if any cartesian component of the step exceeds this amount,
                    the step will be scaled down and the remaining difference
                    will (hopefully) be handled on the next iteration
                    Setting this option to e.g. 0.2 can improve stability for
                    larger changes
        convergence: end cycles if differences between actual step and dq
                     is less than this amount
        """
        x0 = best_struc = np.reshape(coords, -1)
        ddq = np.zeros(len(dq))
        target_q = self.adjust_phase(self.values(coords) + dq)
        xq = dq.copy()
        smallest_dq = None
        for i in range(0, max_iterations):
            B = self.B_matrix(np.reshape(x0, coords.shape))
            if use_delocalized:
                G = np.matmul(B, B.T)
                w, v = np.linalg.eigh(G)
                U = v[:, -(len(x0) - 6):]
                Bd = np.matmul(U.T, B)
                ds = np.dot(U.T, xq)
                dx = np.dot(np.linalg.pinv(Bd), ds)
            else:
                B_pinv = np.linalg.pinv(B)
                dx = np.dot(B_pinv, xq)
            if step_limit is not None and max(np.absolute(dx)) > step_limit:
                dx = step_limit * dx / max(np.absolute(dx))
            x1 = x0 + dx

            ddq = self.adjust_phase(
                dq - self.difference(coords, np.reshape(x1, coords.shape))
            )

            if use_delocalized:
                dds = np.matmul(U.T, ddq)
            #     togo = np.linalg.norm(dds)
            # else:
            togo = np.linalg.norm(ddq)
            
            x0 = x1
            xq = ddq
    
            if smallest_dq is None or togo < smallest_dq:
                best_struc = x1
                smallest_dq = togo
            if debug:
                print("q -> r step", i, "error =", togo)
                print(xq)
                print(dq)
                print(self.difference(coords, np.reshape(x1, coords.shape)))
            if togo < convergence:
                break

        if togo < convergence:
            return np.reshape(x1, coords.shape), togo
        
        return np.reshape(best_struc, coords.shape), smallest_dq


class CartesianCoordinateSet(InternalCoordinateSet):
    def __init__(self, geometry):
        self.geometry = geometry.copy(copy_atoms=True)
        geometry = self.geometry
        self.coordinates = {
            "cartesian": [],
        }

        self.determine_coordinates(self.geometry)

    def determine_coordinates(self, geometry):
        for i, atom in enumerate(geometry.atoms):
            cart = CartesianCoordinate(i)
            if not any(x == cart for x in self.coordinates["cartesian"]):
                self.coordinates["cartesian"].append(cart)