"""detecting and forcing symmetry"""
import numpy as np

from scipy.spatial import distance_matrix

from AaronTools import addlogger
from AaronTools.utils.prime_numbers import Primes
from AaronTools.utils.utils import (
    rotation_matrix, mirror_matrix, proj, angle_between_vectors, perp_vector
)


class SymmetryElement:
    def __init__(self, order, center):
        self.order = order
        self.operation = np.identity(3)
        self.translation = center

    def perp_dist(self, coords):
        """distance from each coordinate perpendicular to this symmetry element"""
        return np.zeros(len(coords))

    def apply_operation(self, coords):
        """returns coords with the symmetry operation applied"""
        coords = coords - self.translation
        coords = np.matmul(coords, self.operation)
        coords += self.translation

        return coords

    def apply_operation_without_translation(self, coords):
        """
        returns coords with the symmetry operation applied but without
        translating the coordinates to or from self's center
        """
        coords = np.matmul(coords, self.operation)

        return coords

    def error(self, geom=None, tolerance=None, groups=None, coords=None):
        """
        error in this symmetry element for the given geometry
        either geom or coords and groups must be given
        if groups is not given and geom is, atoms will be grouped by element
        """
        if coords is None:
            coords = geom.coords

        full_coords2 = self.apply_operation(coords)
        error = 0

        # compute distances between the initial coords and
        # the coords after applying the symmetry operation
        # but only calculate distances for atoms that might
        # be symmetry-equivalent (i.e. in the same inital
        # group, which typically is based on what the atom's
        # neighbors are)
        if groups is not None:
            group_names = groups
        else:
            group_names = geom.elements
        for group in set(group_names):
            ndx = (group_names == group).nonzero()[0]
            coords1 = np.take(coords, ndx, axis=0)
            coords2 = np.take(full_coords2, ndx, axis=0)
            dist_mat = distance_matrix(coords1, coords2)

            perp_dist = self.perp_dist(coords1)
            # treat values less than 1 as 1 to avoid numerical nonsense
            perp_dist = np.maximum(perp_dist, np.ones(len(ndx)))

            error_mat = dist_mat / perp_dist
            min_d = max(np.min(error_mat, axis=1))
            if min_d > error:
                error = min_d

        return error
    
    def equivalent_positions(self, coords, groups):
        """
        return an array with the indices that are equivalent after
        applying this operation
        
        for example:
        ndx = element.equivalent_positions(geom.coords, groups)
        coords[ndx] should be equal to element.apply_operation(geom.coords)
        """
        eq_ndx = np.zeros(len(coords), dtype=int)
        init_partitions = dict()
        init_ndx = dict()
        
        for i, (coord, group) in enumerate(zip(coords, groups)):
            init_partitions.setdefault(group, [])
            init_partitions[group].append(coord)
            init_ndx.setdefault(group, [])
            init_ndx[group].append(i)

        for group in init_partitions:
            coords = init_partitions[group]
            new_coords = self.apply_operation(coords)
            
            dist = distance_matrix(coords, new_coords)
            closest_ndx = np.argmin(dist, axis=1)
            for i, (atom, ndx) in enumerate(zip(init_partitions[group], closest_ndx)):
                j = init_ndx[group][i]
                k = init_ndx[group][ndx]
                eq_ndx[j] = k
        
        return eq_ndx
    
    @property
    def trace(self):
        """trace of this symmetry element's matrix"""
        return np.trace(self.operation)


class Identity(SymmetryElement):
    def __init__(self):
        self.translation = np.zeros(3)
        self.operation = np.eye(3)

    def __repr__(self):
        return "E"

    def __lt__(self, other):
        return False


class ProperRotation(SymmetryElement):
    """proper rotation"""
    def __init__(self, center, axis, n, exp=1):
        self.order = n
        self.operation = rotation_matrix(
            2 * np.pi * exp / n,
            axis,
            renormalize=False,
        )
        self.translation = center
        self.axis = axis
        self.n = n
        self.exp = exp

    def __repr__(self):
        if self.exp > 1:
            return "C%i^%i (%5.2f %5.2f %5.2f)" % (
                self.n,
                self.exp,
                *self.axis,
            )
        return "C%i (%5.2f %5.2f %5.2f)" % (
            self.n,
            *self.axis,
        )

    def __lt__(self, other):
        if isinstance(other, Identity) or isinstance(other, InversionCenter):
            return False
        if isinstance(other, ProperRotation):
            if self.n == other.n:
                return self.exp > other.exp
            return self.n < other.n
        return False

    def perp_dist(self, coords):
        v = coords - self.translation
        n = np.dot(v, self.axis)
        p = np.outer(n, self.axis)
        return np.linalg.norm(v - p, axis=1)


class MirrorPlane(SymmetryElement):
    """mirror plane"""
    def __init__(self, center, axis, label=None):
        self.order = 2
        self.translation = center
        self.axis = axis
        self.operation = mirror_matrix(axis)
        self.label = label

    def __repr__(self):
        if self.label:
            return "sigma_%s (%5.2f %5.2f %5.2f)" % (self.label, *self.axis)
        return "sigma (%5.2f %5.2f %5.2f)" % tuple(self.axis)

    def __lt__(self, other):
        if not isinstance(other, MirrorPlane):
            return True
        if self.label and other.label:
            return self.label < other.label
        return True

    def perp_dist(self, coords):
        v = coords - self.translation
        return np.dot(v, self.axis[:, None]).flatten()


class InversionCenter(SymmetryElement):
    """inversion center"""
    def __init__(self, center):
        self.order = 2
        self.operation = -np.identity(3)
        self.translation = center

    def __lt__(self, other):
        if isinstance(other, Identity):
            return True
        return False

    def __repr__(self):
        return "i (%.2f %.2f %.2f)" % (
            *self.translation,
        )

    def perp_dist(self, coords):
        v = coords - self.translation
        return np.linalg.norm(v, axis=1)


class ImproperRotation(SymmetryElement):
    """improper rotation"""
    def __init__(self, center, axis, n, exp=1):
        self.order = n
        self.operation = np.matmul(
            rotation_matrix(
                2 * np.pi * exp / n,
                axis,
                renormalize=False,
            ),
            mirror_matrix(axis)
        )
        self.axis = axis
        self.translation = center
        self.n = n
        self.exp = exp

    def __repr__(self):
        if self.exp > 1:
            return "S%i^%i (%5.2f %5.2f %5.2f)" % (
                self.n,
                self.exp,
                *self.axis,
            )
        return "S%i (%5.2f %5.2f %5.2f)" % (
            self.n,
            *self.axis,
        )

    def __lt__(self, other):
        if (
                isinstance(other, Identity) or
                isinstance(other, ProperRotation) or
                isinstance(other, InversionCenter)
        ):
            return True
        if isinstance(other, ImproperRotation):
            if self.n == other.n:
                return self.exp > other.exp
            return self.n < other.n
        return False

    def perp_dist(self, coords):
        v = coords - self.translation
        n = np.dot(v, self.axis)
        p = np.outer(n, self.axis)
        ax_dist = np.linalg.norm(v - p, axis=1)

        sig_dist = np.dot(v, self.axis[:, None]).flatten()

        return np.minimum(ax_dist, sig_dist)


@addlogger
class PointGroup:

    LOG = None

    def __init__(
            self,
            geom,
            tolerance=0.1,
            max_rotation=6,
            rotation_tolerance=0.01,
            groups=None,
            center=None
    ):
        self.geom = geom
        self.center = center
        if self.center is None:
            self.center = geom.COM()
        self.elements = self.get_symmetry_elements(
            geom,
            tolerance=tolerance,
            max_rotation=max_rotation,
            groups=groups,
            rotation_tolerance=rotation_tolerance,
        )
        self.name = self.determine_point_group(
            rotation_tolerance=rotation_tolerance
        )

    def get_symmetry_elements(
            self,
            geom,
            tolerance=0.1,
            max_rotation=6,
            rotation_tolerance=0.01,
            groups=None,
    ):
        """
        determine what symmetry elements are valid for geom
        geom - Geometry()
        tolerance - maximum error for an element to be valid
        max_rotation - maximum n for Cn (Sn can be 2x this)
        rotation_tolerance - tolerance in radians for angle between
        axes to be for them to be considered parallel/antiparallel/orthogonal

        returns list(SymmetryElement)
        """
        CITATION = "doi:10.1002/jcc.22995"
        self.LOG.citation(CITATION)

        # atoms are grouped based on what they are bonded to
        # if there's not many atoms, don't bother splitting them up
        # based on ranks
        if groups is not None:
            atom_ids = np.array(groups)
            self.initial_groups = atom_ids
        else:
            atom_ids = np.array(
                geom.canonical_rank(
                    update=False,
                    break_ties=False,
                    invariant=False,
                )
            )
            self.initial_groups = atom_ids

        coords = geom.coords

        moments, axes = geom.get_principle_axes()
        axes = axes.T

        valid = [Identity()]

        degeneracy = np.ones(3, dtype=int)
        for i, m1 in enumerate(moments):
            for j, m2 in enumerate(moments):
                if i == j:
                    continue
                if np.isclose(m1, m2, rtol=tolerance, atol=tolerance):
                    degeneracy[i] += 1

        com = self.center

        inver = InversionCenter(com)
        error = inver.error(geom, tolerance, groups=atom_ids)
        if error <= tolerance:
            valid.append(inver)

        if any(np.isclose(m, 0, atol=1e-6) for m in moments):
            return valid

        ortho_to = []
        for vec, degen in zip(axes, degeneracy):
            if any(d > 1 for d in degeneracy) and degen == 1:
                ortho_to.append(vec)
            elif all(d == 1 for d in degeneracy):
                ortho_to.append(vec)

        # find vectors from COM to each atom
        # these might be proper rotation axes
        atom_axes = geom.coords - com

        # find vectors normal to each pair of atoms
        # these might be normal to a miror plane
        atom_pair_norms = []
        for i, v in enumerate(atom_axes):
            dv = atom_axes - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask2 = angles > rotation_tolerance
            mask3 = angles < np.pi - rotation_tolerance
            mask = np.logical_and(mask2, mask3)
            
            pair_n = np.cross(v, atom_axes[mask])
            norms = np.linalg.norm(pair_n, axis=1)
            pair_n = np.take(pair_n, np.nonzero(norms), axis=0)[0]
            norms = np.take(norms, np.nonzero(norms), axis=0)
            pair_n /= norms.T
            atom_pair_norms.extend(pair_n.tolist())
        
        atom_pair_norms = np.array(atom_pair_norms)

        # find vectors to the midpoints between each
        # pair of like atoms
        # these might be proper rotations
        atom_pairs = []
        for atom_id in set(atom_ids):
            ndx = (atom_ids == atom_id).nonzero()[0]
            subset_axes = np.take(atom_axes, ndx, axis=0)
            for i, v in enumerate(subset_axes):
                mask = np.ones(len(subset_axes), dtype=bool)
                mask[i] = False
                pair_v = subset_axes[mask] + v
                norms = np.linalg.norm(pair_v, axis=1)
                pair_v = np.take(pair_v, np.nonzero(norms), axis=0)[0]
                norms = np.take(norms, np.nonzero(norms), axis=0)
                pair_v /= norms.T
                atom_pairs.extend(pair_v.tolist())

        atom_pairs = np.array(atom_pairs)
        
        norms = np.linalg.norm(atom_axes, axis=1)
        # don't want axis for an atom that is at the COM (0-vector)
        atom_axes = np.take(atom_axes, np.nonzero(norms), axis=0)[0]

        # normalize
        norms = np.take(norms, np.nonzero(norms))
        atom_axes /= norms.T

        # s = ""
        # for v in atom_axes:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        # remove parallel/antiparallel axes for single atoms
        # print(atom_axes)
        mask = np.ones(len(atom_axes), dtype=bool)
        for i, v in enumerate(atom_axes):
            if not mask[i]:
                continue
            dv = atom_axes - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            # print(", ".join(["%.2f" % a for a in angles]))
            mask2 = angles > rotation_tolerance
            mask3 = angles < np.pi - rotation_tolerance
            mask[:i] *= np.logical_and(mask2, mask3)[:i]
            # print(mask)

        atom_axes = atom_axes[mask]

        # s = ""
        # for v in atom_axes:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        # remove parallel/antiparallel axes for pairs of atoms
        mask = np.ones(len(atom_pairs), dtype=bool)
        for i, v in enumerate(atom_pairs):
            if not mask[i]:
                continue
            dv = np.delete(atom_pairs, i, axis=0) - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))

            mask2 = angles > rotation_tolerance
            mask3 = angles < np.pi - rotation_tolerance
            mask4 = np.logical_and(mask2, mask3)
            mask[:i] *= mask4[:i]
            mask[i + 1:] *= mask4[i:]

        atom_pairs = atom_pairs[mask]

        # remove parallel/antiparallel norms for pairs of atoms
        mask = np.ones(len(atom_pair_norms), dtype=bool)
        for i, v in enumerate(atom_pair_norms):
            if not mask[i]:
                continue
            dv = np.delete(atom_pair_norms, i, axis=0) - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask2 = angles > rotation_tolerance
            mask3 = angles < np.pi - rotation_tolerance
            mask4 = np.logical_and(mask2, mask3)
            mask[:i] *= mask4[:i]
            mask[i + 1:] *= mask4[i:]

        atom_pair_norms = atom_pair_norms[mask]

        # s = ""
        # for v in atom_pair_norms:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        if len(atom_pairs):
            # remove axes for pairs of atoms that are parallel/antiparallel
            # to axes for single atoms
            mask = np.ones(len(atom_pairs), dtype=bool)
            for i, v in enumerate(atom_axes):
                dv = atom_pairs - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles > rotation_tolerance
                mask3 = angles < np.pi - rotation_tolerance
                mask *= np.logical_and(mask2, mask3)

            atom_pairs = atom_pairs[mask]

        if len(atom_pair_norms):
            # remove norms for pairs of atoms that are parallel/antiparallel
            # to axes for single atoms
            mask = np.ones(len(atom_pair_norms), dtype=bool)
            for i, v in enumerate(atom_axes):
                dv = atom_pair_norms - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles > rotation_tolerance
                mask3 = angles < np.pi - rotation_tolerance
                mask *= np.logical_and(mask2, mask3)

            atom_pair_norms = atom_pair_norms[mask]

        # s = ""
        # for v in atom_pair_norms:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        # remove axes for single atoms that are parallel/antiparallel
        # to moment of inertia axes
        mask = np.ones(len(atom_axes), dtype=bool)
        for i, v in enumerate(axes):
            dv = atom_axes - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask2 = angles > rotation_tolerance
            mask3 = angles < np.pi - rotation_tolerance
            mask *= np.logical_and(mask2, mask3)
        
        atom_axes = atom_axes[mask]

        # remove axes for pairs of atoms that are parallel/antiparallel
        # to moment of inertia axes
        if len(atom_pairs):
            mask = np.ones(len(atom_pairs), dtype=bool)
            for i, v in enumerate(axes):
                dv = atom_pairs - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles > rotation_tolerance
                mask3 = angles < np.pi - rotation_tolerance
                mask *= np.logical_and(mask2, mask3)

            atom_pairs = atom_pairs[mask]

        # remove norms for pairs of atoms that are parallel/antiparallel
        # to moment of inertia axes
        if len(atom_pair_norms):
            mask = np.ones(len(atom_pair_norms), dtype=bool)
            for i, v in enumerate(axes):
                dv = atom_pair_norms - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles > rotation_tolerance
                mask3 = angles < np.pi - rotation_tolerance
                mask *= np.logical_and(mask2, mask3)

            atom_pair_norms = atom_pair_norms[mask]

        # s = ""
        # for v in atom_pair_norms:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        # remove axes that are not orthogonal to moments of inertia axes
        if ortho_to:
            mask = np.ones(len(atom_axes), dtype=bool)
            pair_mask = np.ones(len(atom_pairs), dtype=bool)
            pair_mask_norms = np.ones(len(atom_pair_norms), dtype=bool)
            for v in ortho_to:
                dv = atom_axes - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask1 = abs(angles - np.pi / 2) < rotation_tolerance
                mask *= mask1

        
                if len(atom_pairs):
                    dv = atom_pairs - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    pair_mask = abs(angles - np.pi / 2) < rotation_tolerance
                    atom_pairs = atom_pairs[pair_mask]
        
                if len(atom_pair_norms):
                    dv = atom_pair_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    pair_mask_norms = abs(angles - np.pi / 2) < rotation_tolerance
                    atom_pair_norms = atom_pair_norms[pair_mask_norms]
        
            atom_axes = atom_axes[mask]
 
        for v in axes:
            mask = np.ones(len(atom_axes), dtype=bool)
            pair_mask = np.ones(len(atom_pairs), dtype=bool)
            pair_mask_norms = np.ones(len(atom_pair_norms), dtype=bool)
                
            dv = atom_axes - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask1 = angles > rotation_tolerance
            mask2 = angles < np.pi - rotation_tolerance
            mask *= np.logical_and(mask1, mask2)
            if len(atom_pairs):
                dv = atom_pairs - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                pair_mask1  = angles > rotation_tolerance
                pair_mask2 = angles < np.pi - rotation_tolerance
                pair_mask *= np.logical_and(pair_mask1, pair_mask2)
                atom_pairs = atom_pairs[pair_mask]

            if len(atom_pair_norms):
                dv = atom_pair_norms - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                atom_pair_norms1  = angles > rotation_tolerance
                atom_pair_norms2 = angles < np.pi - rotation_tolerance
                pair_mask_norms *= np.logical_and(atom_pair_norms1, atom_pair_norms2)
                atom_pair_norms = atom_pair_norms[pair_mask_norms]

        if len(atom_pairs) and len(atom_axes):
            for v in atom_pairs:
                mask = np.ones(len(atom_axes), dtype=bool)
                dv = atom_axes - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask1 = angles > rotation_tolerance
                mask2 = angles < np.pi - rotation_tolerance
                mask *= np.logical_and(mask1, mask2)
                atom_axes = atom_axes[mask]

        # s = ""
        # for v in ortho_to:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        checked_axes = 0
        # find proper rotations along the axes we've found:
        # * moments of inertia axes
        # * COM -> atom vectors
        # * COM -> midpoint of atom paris
        # also grab axes for checking mirror planes
        check_axes = []
        primes = dict()
        args = tuple([arg for arg in [axes, atom_axes, atom_pairs] if len(arg)])
        principal_axis = None
        for ax in np.concatenate(args):
            max_n = None
            found_n = []
            for n in range(2, max_rotation + 1):
                if n not in primes:
                    primes[n] = Primes.primes_below(n // 2)
                # print(n, primes[n])
                skip = False
                for prime in primes[n]:
                    if n % prime == 0 and prime not in found_n:
                        # print("skipping", n)
                        skip = True
                        break
                # if max_n and max_n % n != 0:
                #     # the highest order proper rotation axis must be
                #     # divisible by all other coincident axes
                #     continue
                # look for C5^2 stuff
                # for exp in range(1, 1 + n // 2):
                for exp in range(1, 2):
                    if exp > 1 and n % exp == 0:
                        # skip things like C4^2 b/c that's just C2
                        continue
                    # see if the error associated with the element is reasonable
                    rot = ProperRotation(com, ax, n, exp)
                    error = rot.error(tolerance, groups=atom_ids, coords=coords)
                    checked_axes += 1
                    if error <= tolerance:
                        # print(geom.atoms[i])
                        # s = ".arrow %f %f %f    " % tuple(com)
                        # end = com + 2 * ax
                        # s += "%f %f %f\n" % tuple(end)
                        # with open("test.bild", "a") as f:
                        #     f.write(s)
                        valid.append(rot)
                        if principal_axis is None or rot.n > principal_axis[0].n:
                            principal_axis = [rot]
                        elif principal_axis is not None and rot.n == principal_axis[0].n:
                            principal_axis.append(rot)
                        found_n.append(n)
                        if n > 2:
                            # for Cn n != 2, add an element that is the same
                            # except the axis of rotation is antiparallel
                            rot2 = ProperRotation(com, -ax, n, exp)
                            valid.append(rot2)
                        if not max_n:
                            max_n = n
                            check_axes.append(ax)
                    elif exp == 1:
                        # can't have Cn^y if you don't have Cn
                        break

        if degeneracy[0] == 3:
            # spherical top molecules need more checks related to C2 axes
            c2_axes = list(
                filter(
                    lambda ele: isinstance(ele, ProperRotation) and ele.n == 2,
                    valid,
                )
            )

            # TODO: replace with array operations like before
            for i, c2_1 in enumerate(c2_axes):
                for c2_2 in c2_axes[:i]:
                    test_axes = []
                    if len(c2_axes) == 3:
                        # T groups - check midpoint
                        for c2_3 in c2_axes[i:]:
                            axis = c2_1.axis + c2_2.axis + c2_3.axis
                            test_axes.append(axis)
                            axis = c2_1.axis + c2_2.axis - c2_3.axis
                            test_axes.append(axis)
                            axis = c2_1.axis - c2_2.axis + c2_3.axis
                            test_axes.append(axis)
                            axis = c2_1.axis - c2_2.axis - c2_3.axis
                            test_axes.append(axis)
                    else:
                        # O, I groups - check cross product
                        test_axes.append(np.cross(c2_1.axis, c2_2.axis))

                    for axis in test_axes:
                        norm = np.linalg.norm(axis)
                        if norm < 1e-5:
                            continue
                        axis /= norm
                        dup = False
                        for element in valid:
                            if isinstance(element, ProperRotation):
                                if 1 - abs(np.dot(element.axis, axis)) < rotation_tolerance:
                                    dup = True
                                    break
                        if dup:
                            continue
                        max_n = None
                        for n in range(max_rotation, 1, -1):
                            if max_n and max_n % n != 0:
                                continue
                            # for exp in range(1, 1 + n // 2):
                            for exp in range(1, 2):
                                if exp > 1 and n % exp == 0:
                                    continue
                                rot = ProperRotation(com, axis, n, exp)
                                checked_axes += 1
                                error = rot.error(tolerance, groups=atom_ids, coords=coords)
                                if error <= tolerance:
                                    if principal_axis is None or rot.n > principal_axis[0].n:
                                        principal_axis = [rot]
                                    elif principal_axis is not None and rot.n == principal_axis[0].n:
                                        principal_axis.append(rot)
                                    valid.append(rot)
                                    if not max_n:
                                        max_n = n
                                        check_axes.append(ax)
                                    if n > 2:
                                        rot2 = ProperRotation(com, -axis, n, exp)
                                        valid.append(rot2)
                                elif exp == 1:
                                    break

        # improper rotations
        # coincident with proper rotations and can be 1x or 2x
        # the order of the proper rotation
        for element in valid:
            if not isinstance(element, ProperRotation):
                continue

            if element.exp != 1:
                continue

            for x in [1, 2]:
                if x * element.n == 2:
                    # S2 is inversion - we already checked i
                    continue

                # for exp in range(1, 1 + (x * element.n) // 2):
                for exp in range(1, 2):
                    if exp > 1 and (x * element.n) % exp == 0:
                        continue

                    for element2 in valid:
                        if isinstance(element2, ImproperRotation):
                            angle = angle_between_vectors(element2.axis, element.axis)
                            if (
                                    element2.exp == exp and
                                    (
                                        angle < rotation_tolerance or
                                        angle > (np.pi - rotation_tolerance)
                                    ) and
                                    element2.n == x * element.n
                            ):
                                break
                    else:

                        imp_rot = ImproperRotation(
                            element.translation,
                            element.axis,
                            x * element.n,
                            exp,
                        )

                        error = imp_rot.error(tolerance, groups=atom_ids, coords=coords)

                        if error <= tolerance:
                            valid.append(imp_rot)
                            rot2 = ImproperRotation(
                                element.translation,
                                -element.axis,
                                x * element.n,
                                exp
                            )
                            valid.append(rot2)

                        elif exp == 1:
                            break

        c2_axes = list(
            filter(
                lambda ele: isinstance(ele, ProperRotation) and ele.n == 2 and ele.exp == 1,
                valid,
            )
        )
        c2_vectors = np.array([c2.axis for c2 in c2_axes])

        sigma_norms = []
        if bool(principal_axis) and len(c2_vectors) and principal_axis[0].n != 2:
            for ax in principal_axis:
                perp = np.cross(ax.axis, c2_vectors)
                norms = np.linalg.norm(perp, axis=1)
                mask = np.nonzero(norms)
                perp = perp[mask]
                norms = norms[mask]
                perp /= norms[:, None]
                sigma_norms.extend(perp)
            
            sigma_norms = np.array(sigma_norms)
            mask = np.ones(len(sigma_norms), dtype=bool)
            for i, v in enumerate(sigma_norms):
                if not mask[i]:
                    continue
                dv = np.delete(sigma_norms, i, axis=0) - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles > rotation_tolerance
                mask3 = angles < np.pi - rotation_tolerance
                mask4 = np.logical_and(mask2, mask3)
                mask[:i] *= mask4[:i]
                mask[i + 1:] *= mask4[i:]
            
            sigma_norms = sigma_norms[mask]

        c2_vectors = np.append(c2_vectors, [-c2.axis for c2 in c2_axes], axis=0)

        # mirror axes
        # for I, O - only check c2 axes
        if (
                degeneracy[0] != 3 or
                not c2_axes or
                (degeneracy[0] == 3 and len(c2_axes) == 3)
        ):
            if len(atom_pair_norms):
                mask = np.ones(len(atom_pair_norms), dtype=bool)
                for i, v in enumerate(axes):
                    dv = atom_pair_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                atom_pair_norms = atom_pair_norms[mask]

                mask = np.ones(len(atom_pair_norms), dtype=bool)
                for i, v in enumerate(check_axes):
                    dv = atom_pair_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                atom_pair_norms = atom_pair_norms[mask]
            
            if check_axes:
                check_axes = np.array(check_axes)
                mask = np.ones(len(check_axes), dtype=bool)
                for i, v in enumerate(axes):
                    dv = check_axes - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                check_axes = check_axes[mask]

                mask = np.ones(len(check_axes), dtype=bool)
                for i, v in enumerate(atom_axes):
                    dv = check_axes - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                check_axes = check_axes[mask]

                mask = np.ones(len(check_axes), dtype=bool)
                for i, v in enumerate(atom_pair_norms):
                    dv = check_axes - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                check_axes = check_axes[mask]
            
            if len(sigma_norms):
                sigma_norms = np.array(sigma_norms)
                mask = np.ones(len(sigma_norms), dtype=bool)
                for i, v in enumerate(axes):
                    dv = sigma_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    angles = np.nan_to_num(angles)
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                sigma_norms = sigma_norms[mask]

                mask = np.ones(len(sigma_norms), dtype=bool)
                for i, v in enumerate(atom_axes):
                    dv = sigma_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                sigma_norms = sigma_norms[mask]

                mask = np.ones(len(sigma_norms), dtype=bool)
                for i, v in enumerate(atom_pair_norms):
                    dv = sigma_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                sigma_norms = sigma_norms[mask]

                mask = np.ones(len(sigma_norms), dtype=bool)
                for i, v in enumerate(check_axes):
                    dv = sigma_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles > rotation_tolerance
                    mask3 = angles < np.pi - rotation_tolerance
                    mask *= np.logical_and(mask2, mask3)

                sigma_norms = sigma_norms[mask]

            # print("axes")
            # for ax in axes:
            #     print(ax)
            #     
            # print("atom_axes")
            # for ax in atom_axes:
            #     print(ax)
            #     
            # print("atom_pair_norms")
            # for ax in atom_pair_norms:
            #     print(ax)
            #     
            # print("check_axes")
            # for ax in check_axes:
            #     print(ax)
            #     
            # print("sigma_norms")
            # for ax in sigma_norms:
            #     print(ax)
                
            args = tuple(
                [arg for arg in [
                    axes, atom_axes, atom_pair_norms, check_axes, sigma_norms
                ] if len(arg)]
            )

            for ax in np.concatenate(args):
                mirror = MirrorPlane(com, ax)
                error = mirror.error(tolerance, groups=atom_ids, coords=coords)
                if error <= tolerance:
                    valid.append(mirror)
        else:
            for i, ax in enumerate(c2_axes):
                mirror = MirrorPlane(com, ax.axis)
                error = mirror.error(tolerance, groups=atom_ids, coords=coords)
                if error <= tolerance:
                    valid.append(mirror)

        min_atoms = None
        if principal_axis:
            for ele in valid:
                if not isinstance(ele, MirrorPlane):
                    continue

                for ax in principal_axis:
                    c2 = np.linalg.norm(ax.axis - ele.axis) ** 2
                    angle = np.arccos(-0.5 * (c2 - 2))
                    if angle < rotation_tolerance or angle > np.pi - rotation_tolerance:
                        # mirror plane normal is parallel to principal_axis
                        ele.label = "h"
                        break
                    elif abs(angle - np.pi / 2) < rotation_tolerance:
                        # mirror plane normal is perpendicular to principal_axis
                        ele.label = "v"
                
                if ele.label == "v":
                    # determine number of atoms in sigma_v in case we need to
                    # differentiate from sigma_d later
                    perp_dist = ele.perp_dist(geom.coords)
                    atoms_contained = sum([1 if abs(d) < tolerance else 0 for d in perp_dist])
                    if min_atoms is None or atoms_contained < min_atoms:
                        min_atoms = atoms_contained

        
        if c2_axes:
            # check for sigma_d
            # should be a sigma_v that bisects two C2 axes
            # if a molecule has different sigma_v planes that
            # bisect C2 axes and contain different numbers of
            # atoms, it is convention to label just the sigma_v
            # planes that contain the fewest atoms as sigma_d
            for ele in valid:
                if not isinstance(ele, MirrorPlane):
                    continue
                if principal_axis and ele.label != "v":
                    continue
                perp_dist = ele.perp_dist(geom.coords)
                atoms_contained = sum([1 if abs(d) < tolerance else 0 for d in perp_dist])
                if min_atoms is not None and atoms_contained != min_atoms:
                    continue
                perp_v = perp_vector(ele.axis)
                c2 = np.linalg.norm(c2_vectors - perp_v, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                for i, angle1 in enumerate(angles):
                    for j, angle2 in enumerate(angles[:i]):
                        if abs(np.dot(c2_vectors[i], c2_vectors[j])) == 1:
                            continue
                        if abs(angle1 - angle2) < rotation_tolerance:
                            ele.label = "d"
                            break
                    
                    if ele.label == "d":
                        break
            
        # s = ""
        # colors = ["black", "red", "blue", "green", "purple", "yellow", "cyan"]
        # for element in valid:
        #     # if isinstance(element, ProperRotation):
        #     #     s += ".color %s\n" % colors[element.n - 2]
        #     #     s += ".note C%i\n" % element.n
        #     #     s += ".arrow %f %f %f    " % tuple(com)
        #     #     end = com + 2 * np.sqrt(element.n) * element.axis
        #     #     s += "%f %f %f\n" % tuple(end)
        #     if isinstance(element, MirrorPlane):
        #         s += ".arrow %f %f %f    " % tuple(com)
        #         end = com + 2 * element.axis
        #         s += "%f %f %f\n" % tuple(end)
        #
        # with open("test.bild", "w") as f:
        #     f.write(s)

        return valid

    def determine_point_group(self, rotation_tolerance=0.01):
        """
        determines point group of self by looing at self.elements
        rotation_tolerance - tolerance in radians for axes to be
        considered parallel/antiparallel/orthogonal

        returns str for point group name
        """
        moments, axes = self.geom.get_principle_axes()
        linear = False
        if any(np.isclose(m, 0, atol=1e-6) for m in moments):
            linear = True

        if linear:
            if any(isinstance(ele, InversionCenter) for ele in self.elements):
                return "D_inf_h"
            else:
                return "C_inf_v"

        Cn = dict()
        has_inversion = False
        has_mirror = False
        has_sig_h = False
        has_sig_d = False
        has_sig_v = False
        for ele in self.elements:
            if isinstance(ele, ProperRotation):
                if ele.n not in Cn and ele.exp == 1:
                    Cn[ele.n] = []
                Cn[ele.n].append(ele)
            if isinstance(ele, InversionCenter):
                has_inversion = True
            if isinstance(ele, MirrorPlane):
                has_mirror = True
                if ele.label == "d":
                    has_sig_d = True
                elif ele.label == "h":
                    has_sig_h = True
                elif ele.label == "v":
                    has_sig_v = True

        if Cn:
            max_n = max(Cn.keys())

            if max_n == 5 and len(Cn[5]) >= 12:
                if has_mirror:
                    return "Ih"
                return "I"
            if max_n == 4 and len(Cn[4]) >= 6:
                if has_mirror:
                    return "Oh"
                return "O"
            if max_n == 3 and len(Cn[3]) >= 8:
                for ele in self.elements:
                    if has_sig_d:
                        return "Td"
                    if has_mirror:
                        return "Th"
                    return "T"

            n_sig_v = len([
                ele for ele in self.elements
                if isinstance(ele, MirrorPlane) and (ele.label == "v" or ele.label == "d")
            ])
            
            if n_sig_v > max_n:
                self.LOG.warning(
                    "%i-fold rotation found, but %i sigma_v planes found\n" % (
                        max_n,
                        n_sig_v,
                    ) + 
                    "you may need to increase the maximum order of proper rotational axes that are checked"
                )
            prin_ax = Cn[max_n][0]

            n_perp = 0
            if 2 in Cn:
                for c2 in Cn[2]:
                    angle = angle_between_vectors(
                        c2.axis,
                        prin_ax.axis,
                        renormalize=False
                    )
                    if abs(angle - np.pi / 2) < rotation_tolerance:
                        n_perp += 1 

            if n_perp >= max_n:
                if has_sig_h:
                    return "D%ih" % max_n

                if n_sig_v >= max_n:
                    return "D%id" % max_n

                return "D%i" % max_n

            if has_sig_h:
                return "C%ih" % max_n

            if n_sig_v >= max_n:
                return "C%iv" % max_n

            for ele in self.elements:
                if isinstance(ele, ImproperRotation) and ele.n == 2 * max_n:
                    return "S%i" % (2 * max_n)

            return "C%i" % max_n

        if has_mirror:
            return "Cs"

        if has_inversion:
            return "Ci"

        return "C1"

    @property
    def symmetry_number(self):
        """external symmetry number"""
        n = 1
        for ele in self.elements:
            if isinstance(ele, ProperRotation):
                if ele.n > n:
                    n = ele.n
            
            elif isinstance(ele, ImproperRotation):
                if ele.n / 2 > n:
                    n = ele.n
        
        if self.name.startswith("D"):
            n *= 2
        if self.name.startswith("T"):
            n *= 4
        if self.name.startswith("O"):
            n *= 6
        if self.name.startswith("I"):
            n *= 12
        return n
    
    def equivalent_positions(self):
        """returns a list of lists of atoms that are symmetry-equivalent"""
        equivs = [set([atom]) for atom in self.geom.atoms]
        init_coords = self.geom.coords
        init_partitions = dict()
        for atom, group in zip(self.geom.atoms, self.initial_groups):
            init_partitions.setdefault(group, [])
            init_partitions[group].append(atom)
        
        init_partition_coords = {
            group: self.geom.coordinates(atoms) for group, atoms in init_partitions.items()
        }
        
        for ele in self.elements:
            if isinstance(ele, Identity):
                continue
            for group in init_partitions:
                coords = init_partition_coords[group]
                new_coords = ele.apply_operation(coords)
                
                dist = distance_matrix(coords, new_coords)
                closest_ndx = np.argmin(dist, axis=1)
                for i, (atom, ndx) in enumerate(zip(init_partitions[group], closest_ndx)):
                    if i == ndx:
                        continue
                    j = self.geom.atoms.index(atom)
                    k = self.geom.atoms.index(init_partitions[group][ndx])
                    
                    equivs[j] = {*equivs[j], init_partitions[group][ndx]}
                    equivs[k] = {*equivs[k], atom}

        # for e in equivs:
        #     for atom in sorted(e, key=lambda a: int(a.name)):
        #         print(atom)
        #     print("----------------")
        
        out = []
        for eq in equivs:
            for group in out:
                if any(atom in group for atom in eq):
                    group.extend(eq)
                    break
            else:
                out.append(list(eq))

        return out

    @property
    def optically_chiral(self):
        """is this point group optically_chiral?"""
        found_mirror = False
        found_inversion = False
        found_improper = False
        for ele in self.elements:
            if isinstance(ele, MirrorPlane):
                # S1
                found_mirror = True
            elif isinstance(ele, InversionCenter):
                # S2
                found_inversion = True
            elif isinstance(ele, ImproperRotation):
                # Sn
                found_improper = True
        
        return not (found_mirror or found_inversion or found_improper)

    def idealize_geometry(self):
        """
        adjust coordinates of self.geom to better match this point group
        also re-determines point group and symmetry elements
        """
        com = self.geom.COM()
        coords = self.geom.coords
        centered_coords = coords - com
        out = np.zeros((len(self.geom.atoms), 3))
        n_ele = len(self.elements)
        principal_axis = None

        # try moving axes of elements to more ideal positions relative to
        # a principal axis
        # for example, some point groups have elements with axes that
        # are 30, 45, 60, and 90 degrees from a principal axis
        # if we find an element close to one of these angles, we move it
        # to that angle
        # many symmetry elements will also be regularly spaced around
        # the principal_axis
        # e.g. D6h has 6 C2 axes perpendicular to the C6, with each 
        # being 30 degrees apart
        # so this looks for C2 axes (and others) that are perpendicular
        # to the principal axis and looks for C2 axes that are
        # 2 * x * pi / N (order of principal axis) apart from one of
        # the C2 axes
        # it turns out none of that really helps, so if False
        if False:
        # if principal_axis:
            principal_axis.sort(
                key=lambda ele: ele.error(
                    geom=self.geom,
                    groups=self.initial_groups,
                )
            )
            for ax in principal_axis:
                non_ideal_axes = dict()
                for ele in sorted(self.elements):
                    if not hasattr(ele, "axis"):
                        continue
                    if ax is ele:
                        continue
                    dv = ax.axis - ele.axis
                    c2 = np.linalg.norm(dv) ** 2
                    angle = np.arccos(-0.5 * (c2 - 2))
                    if np.isclose(angle, 0):
                        continue
                    for n in range(1, ax.n):
                        test_angle = np.arccos(1 / np.sqrt(n))
                        if np.isclose(angle, np.pi / 2, atol=5e-3):
                            non_ideal_axes.setdefault(np.pi / 2, [])
                            perp_axis = np.cross(ax.axis, ele.axis)
                            perp_axis = np.cross(perp_axis, ax.axis)
                            perp_axis /= np.linalg.norm(perp_axis)
                            if np.dot(perp_axis, ele.axis) < 0:
                                perp_axis *= -1
                            ele.axis = perp_axis
                            non_ideal_axes[np.pi / 2].append(ele)
                            continue
                            
                        if np.isclose(angle, test_angle, atol=5e-3):
                            non_ideal_axes.setdefault(test_angle, [])
                            non_ideal_axes[test_angle].append(ele)
                        if n == 1:
                            continue

                        test_angle = np.arccos(1. / n)
                        if any(np.isclose(a, test_angle) for a in non_ideal_axes.keys()):
                            continue
                        if np.isclose(angle, test_angle, atol=5e-3):
                            non_ideal_axes.setdefault(test_angle, [])
                            non_ideal_axes[test_angle].append(ele)

                for angle in non_ideal_axes:
                    # print(angle)
                    prop_rots = dict()
                    improp_rots = dict()
                    mirror = []
                    for ele in non_ideal_axes[angle]:
                        # print("\t", ele)
                        if isinstance(ele, MirrorPlane):
                            mirror.append(ele)
                        elif isinstance(ele, ProperRotation):
                            prop_rots.setdefault(ele.n, [])
                            prop_rots[ele.n].append(ele)
                        elif isinstance(ele, ImproperRotation):
                            improp_rots.setdefault(ele.n, [])
                            improp_rots[ele.n].append(ele)
                    
                    for n in prop_rots:
                        prop_rots[n].sort(
                            key=lambda ele: ele.error(
                                geom=self.geom,
                                groups=self.initial_groups,
                            )
                        )

                    for n in improp_rots:
                        improp_rots[n].sort(
                            key=lambda ele: ele.error(
                                geom=self.geom,
                                groups=self.initial_groups,
                            )
                        )
                    
                    mirror.sort(
                        key=lambda ele: ele.error(
                            geom=self.geom,
                            groups=self.initial_groups,
                        )
                    )
                    
                    for i in range(1, 2 * ax.n):
                        test_angle = i * np.pi / (2 * ax.n)
                        mat = rotation_matrix(test_angle, ax.axis)
                        for n in improp_rots:
                            new_v = np.matmul(mat, improp_rots[n][0].axis)
                            for ele in improp_rots[n][1:]:
                                if np.isclose(np.dot(new_v, ele.axis), 1):
                                    ele.axis = new_v
                                    break
                            else:
                                print("no C%i axis at angle" % n, test_angle)
                        
                        for n in improp_rots:
                            new_v = np.matmul(mat, improp_rots[n][0].axis)
                            for ele in improp_rots[n][1:]:
                                if np.isclose(np.dot(new_v, ele.axis), 1):
                                    ele.axis = new_v
                                
                        for mir in mirror:
                            new_v = np.matmul(mat, mirror[0].axis)
                            for ele in mirror[1:]:
                                if np.isclose(np.dot(new_v, ele.axis), 1):
                                    ele.axis = new_v
        
        # apply each operation and average the coordinates of the
        # equivalent positions
        max_n = 0
        for ele in self.elements:
            equiv = ele.equivalent_positions(coords, self.initial_groups)
            out += ele.apply_operation_without_translation(
                centered_coords[equiv]
            ) / n_ele

            if isinstance(ele, ProperRotation):
                if ele.n > max_n:
                    max_n = ele.n
        
        self.geom.coords = out + com
        self.elements = self.get_symmetry_elements(
            self.geom,
            max_rotation=max_n,
            groups=self.initial_groups,
        )

        self.determine_point_group()

    def total_error(self, return_max=False):
        tot_error = 0
        max_error = 0
        max_ele = None
        for ele in self.elements:
            error = ele.error(geom=self.geom, groups=self.initial_groups)
            tot_error += error
            if error > max_error:
                max_error = error
                max_ele = ele
        
        if return_max:
            return tot_error, max_error, max_ele
        return tot_error