"""detecting and forcing symmetry"""
import numpy as np

from scipy.spatial import distance_matrix

from AaronTools import addlogger
from AaronTools.utils.utils import rotation_matrix, mirror_matrix, proj, angle_between_vectors


class SymmetryElement:
    def __init__(self, order, center):
        self.order = order
        self.operation = np.identity(3)
        self.translation = center

    def perp_dist(self, coords):
        """distance from each coordinate perpendicular to this symmetry element"""
        return np.zeros(len(coords))

    def apply_operation(self, coords):
        """returns a geometry with the symmetry operation applied"""
        coords = coords - self.translation
        coords = np.matmul(coords, self.operation)
        coords += self.translation

        return coords

    def error(self, geom, tolerance=None, groups=None):
        """error in this symmetry element for the given geometry"""
        coords = geom.coords

        full_coords2 = self.apply_operation(coords)
        error = 0

        # if isinstance(self, ProperRotation) and self.n == 3 and self.exp == 1:
        #     with open("asdf.xyz", "w") as f:
        #         s = "%i\n\n" % len(geom.atoms)
        #         for atom, coord in zip(geom.atoms, full_coords2):
        #             s += "%-2s   %7.5f   %7.5f  %7.5f\n" % (
        #                 atom.element,
        #                 *coord,
        #             )
        #         f.write(s)

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


class Identity(SymmetryElement):
    def __init__(self):
        pass

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
    def __init__(self, center, axis):
        self.order = 2
        self.translation = center
        self.axis = axis
        self.operation = mirror_matrix(axis)

    def __repr__(self):
        return "sigma (%5.2f %5.2f %5.2f)" % tuple(self.axis)

    def __lt__(self, other):
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
    ):
        self.geom = geom
        self.center = geom.COM()
        self.elements = self.get_symmetry_elements(
            geom, tolerance=tolerance, max_rotation=max_rotation
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
        error = inver.error(geom, tolerance)
        if error <= tolerance:
            valid.append(inver)

        if any(np.isclose(m, 0) for m in moments):
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

        # atoms are grouped based on what they are bonded to
        atom_ids = np.array([a.get_neighbor_id() for a in geom.atoms])

        # find vectors normal to each pair of atoms
        # these might be normal to a miror plane
        atom_pair_norms = []
        for i, v in enumerate(atom_axes):
            dv = atom_axes - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask2 = angles < rotation_tolerance
            mask3 = angles > np.pi - rotation_tolerance
            mask = np.invert(np.logical_or(mask2, mask3))
            
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
        # for v in atom_pair_norms:
        #     s += ".arrow   %f %f %f   " % tuple(com)
        #     end = com + 2 * v
        #     s += "%f %f %f\n" % tuple(end)
        # with open("test2.bild", "w") as f:
        #     f.write(s)

        # remove parallel/antiparallel axes for single atoms
        mask = np.ones(len(atom_axes), dtype=bool)
        for i, v in enumerate(atom_axes):
            if not mask[i]:
                continue
            dv = atom_axes - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask2 = angles < rotation_tolerance
            mask3 = angles > np.pi - rotation_tolerance
            mask *= np.invert(np.logical_or(mask2, mask3))

        atom_axes = atom_axes[mask]

        # remove parallel/antiparallel axes for pairs of atoms
        mask = np.ones(len(atom_pairs), dtype=bool)
        for i, v in enumerate(atom_pairs):
            if not mask[i]:
                continue
            dv = np.delete(atom_pairs, i, axis=0) - v
            c2 = np.linalg.norm(dv, axis=1) ** 2
            angles = np.arccos(-0.5 * (c2 - 2))
            mask2 = angles < rotation_tolerance
            mask3 = angles > np.pi - rotation_tolerance
            mask4 = np.invert(np.logical_or(mask2, mask3))
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
            mask2 = angles < rotation_tolerance
            mask3 = angles > np.pi - rotation_tolerance
            mask4 = np.invert(np.logical_or(mask2, mask3))
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
                mask2 = angles < rotation_tolerance
                mask3 = angles > np.pi - rotation_tolerance
                mask *= np.invert(np.logical_or(mask2, mask3))

            atom_pairs = atom_pairs[mask]

        if len(atom_pair_norms):
            # remove norms for pairs of atoms that are parallel/antiparallel
            # to axes for single atoms
            mask = np.ones(len(atom_pair_norms), dtype=bool)
            for i, v in enumerate(atom_axes):
                dv = atom_pair_norms - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles < rotation_tolerance
                mask3 = angles > np.pi - rotation_tolerance
                mask *= np.invert(np.logical_or(mask2, mask3))

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
            mask2 = angles < rotation_tolerance
            mask3 = angles > np.pi - rotation_tolerance
            mask *= np.invert(np.logical_or(mask2, mask3))

        atom_axes = atom_axes[mask]

        # remove axes for pairs of atoms that are parallel/antiparallel
        # to moment of inertia axes
        if len(atom_pairs):
            mask = np.ones(len(atom_pairs), dtype=bool)
            for i, v in enumerate(axes):
                dv = atom_pairs - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles < rotation_tolerance
                mask3 = angles > np.pi - rotation_tolerance
                mask *= np.invert(np.logical_or(mask2, mask3))

            atom_pairs = atom_pairs[mask]

        # remove norms for pairs of atoms that are parallel/antiparallel
        # to moment of inertia axes
        if len(atom_pair_norms):
            mask = np.ones(len(atom_pair_norms), dtype=bool)
            for i, v in enumerate(axes):
                dv = atom_pair_norms - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask2 = angles < rotation_tolerance
                mask3 = angles > np.pi - rotation_tolerance
                mask *= np.invert(np.logical_or(mask2, mask3))

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
            for v in ortho_to:
                dv = atom_axes - v
                c2 = np.linalg.norm(dv, axis=1) ** 2
                angles = np.arccos(-0.5 * (c2 - 2))
                mask1 = abs(angles - np.pi / 2) < rotation_tolerance
                mask2 = angles < rotation_tolerance
                mask3 = angles > np.pi - rotation_tolerance
                mask *= np.invert(
                    np.logical_or(mask1, mask2),
                    mask3,
                )
        
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
                    pair_mask = abs(angles - np.pi / 2) < rotation_tolerance
                    atom_pair_norms = atom_pair_norms[pair_mask]
        
            atom_axes = atom_axes[mask]

        # s = ""
        # for v in atom_pair_norms:
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
        args = tuple([arg for arg in [axes, atom_axes, atom_pairs] if len(arg)])
        for ax in np.concatenate(args):
            max_n = None
            for n in range(max_rotation, 1, -1):
                if max_n and max_n % n != 0:
                    # the highest order proper rotation axis must be
                    # divisible by all other coincident axes
                    continue
                # look for C5^2 stuff
                # for exp in range(1, 1 + n // 2):
                for exp in range(1, 2):
                    if exp > 1 and n % exp == 0:
                        # skip things like C4^2 b/c that's just C2
                        continue
                    # see if the error associated with the element is reasonable
                    rot = ProperRotation(com, ax, n, exp)
                    error = rot.error(geom, tolerance, groups=atom_ids)
                    checked_axes += 1
                    if error <= tolerance:
                        # print(geom.atoms[i])
                        # s = ".arrow %f %f %f    " % tuple(com)
                        # end = com + 2 * ax
                        # s += "%f %f %f\n" % tuple(end)
                        # with open("test.bild", "a") as f:
                        #     f.write(s)
                        valid.append(rot)
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
                    axes = []
                    if len(c2_axes) == 3:
                        # T groups - check midpoint
                        for c2_3 in c2_axes[i:]:
                            axis = c2_1.axis + c2_2.axis + c2_3.axis
                            axes.append(axis)
                            axis = c2_1.axis + c2_2.axis - c2_3.axis
                            axes.append(axis)
                            axis = c2_1.axis - c2_2.axis + c2_3.axis
                            axes.append(axis)
                            axis = c2_1.axis - c2_2.axis - c2_3.axis
                            axes.append(axis)
                    else:
                        # O, I groups - check cross product
                        axes.append(np.cross(c2_1.axis, c2_2.axis))

                    for axis in axes:
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
                                error = rot.error(geom, tolerance, groups=atom_ids)
                                if error <= tolerance:
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

                        error = imp_rot.error(geom, tolerance, groups=atom_ids)

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
                lambda ele: isinstance(ele, ProperRotation) and ele.n == 2,
                valid,
            )
        )

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
                    mask2 = angles < rotation_tolerance
                    mask3 = angles > np.pi - rotation_tolerance
                    mask *= np.invert(np.logical_or(mask2, mask3))

                atom_pair_norms = atom_pair_norms[mask]

                mask = np.ones(len(atom_pair_norms), dtype=bool)
                for i, v in enumerate(check_axes):
                    dv = atom_pair_norms - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles < rotation_tolerance
                    mask3 = angles > np.pi - rotation_tolerance
                    mask *= np.invert(np.logical_or(mask2, mask3))

                atom_pair_norms = atom_pair_norms[mask]

            if check_axes:
                check_axes = np.array(check_axes)
                mask = np.ones(len(check_axes), dtype=bool)
                for i, v in enumerate(axes):
                    dv = check_axes - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles < rotation_tolerance
                    mask3 = angles > np.pi - rotation_tolerance
                    mask *= np.invert(np.logical_or(mask2, mask3))

                check_axes = check_axes[mask]

                mask = np.ones(len(check_axes), dtype=bool)
                for i, v in enumerate(atom_axes):
                    dv = check_axes - v
                    c2 = np.linalg.norm(dv, axis=1) ** 2
                    angles = np.arccos(-0.5 * (c2 - 2))
                    mask2 = angles < rotation_tolerance
                    mask3 = angles > np.pi - rotation_tolerance
                    mask *= np.invert(np.logical_or(mask2, mask3))

                check_axes = check_axes[mask]

            args = tuple(
                [arg for arg in [
                    axes, atom_axes, atom_pair_norms, check_axes
                ] if len(arg)]
            )
            for ax in np.concatenate(args):
                mirror = MirrorPlane(com, ax)
                error = mirror.error(geom, tolerance, groups=atom_ids)
                if error <= tolerance:
                    valid.append(mirror)
        else:
            for element in c2_axes:
                mirror = MirrorPlane(com, element.axis)
                error = mirror.error(geom, tolerance, groups=atom_ids)
                if error <= tolerance:
                    valid.append(mirror)

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
        if any(np.isclose(m, 0) for m in moments):
            linear = True

        if linear:
            if any(isinstance(ele, InversionCenter) for ele in self.elements):
                return "D_inf_h"
            else:
                return "C_inf_v"

        Cn = dict()
        has_inversion = False
        has_mirror = False
        for ele in self.elements:
            if isinstance(ele, ProperRotation):
                if ele.n not in Cn:
                    Cn[ele.n] = []
                Cn[ele.n].append(ele)
            if isinstance(ele, InversionCenter):
                has_inversion = True
            if isinstance(ele, MirrorPlane):
                has_mirror = True

        for n in sorted(Cn.keys(), reverse=True):
            if len(Cn[n]) > 2 and n != 2:
                if any(isinstance(ele, InversionCenter) for ele in self.elements):
                    if 5 in Cn.keys():
                        if has_inversion:
                            return "Ih"
                        return "I"

                    if has_inversion:
                        return "Oh"
                    return "O"

                if not has_mirror:
                    return "T"
                if has_inversion:
                    return "Th"
                return "Td"

        if Cn:
            has_sig_h = False
            n_sig_v = 0

            max_n = max(Cn.keys())
            prin_ax = Cn[max_n][0]

            for ele in self.elements:
                if isinstance(ele, MirrorPlane):
                    angle = angle_between_vectors(
                        ele.axis, prin_ax.axis, renormalize=False
                    )
                    if angle < rotation_tolerance or abs(np.pi - angle) < rotation_tolerance:
                        has_sig_h = True
                        continue

                    if abs(angle - np.pi / 2) < rotation_tolerance:
                        n_sig_v += 1

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

            if n_perp == max_n:
                if has_sig_h:
                    return "D%ih" % max_n

                if n_sig_v == max_n:
                    return "D%id" % max_n

                return "D%i" % max_n

            if has_sig_h:
                return "C%ih" % max_n

            if n_sig_v == max_n:
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
