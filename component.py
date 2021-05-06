"""For more complicated geometry manipulation and complex building"""
import os
import re
from glob import glob

import numpy as np

from AaronTools.const import AARONLIB, AARONTOOLS, BONDI_RADII, VDW_RADII
from AaronTools.fileIO import read_types
from AaronTools.finders import BondedTo, CloserTo
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import get_filename


class Component(Geometry):
    """
    Attributes:
    :name:          str
    :comment:       str
    :atoms:         list(Atom)
    :other:         dict()
    :substituents:  list(Substituent) substituents detected
    :backbone:      list(Atom) the backbone atoms
    :key_atoms:     list(Atom) the atoms used for mapping
    """

    AARON_LIBS = os.path.join(AARONLIB, "Ligands")
    BUILTIN = os.path.join(AARONTOOLS, "Ligands")

    def __init__(
        self,
        structure,
        name="",
        comment=None,
        tag=None,
        to_center=None,
        key_atoms=None,
    ):
        """
        comp is either a file, a geometry, or an atom list
        """
        super().__init__()
        self.name = name
        self.comment = comment
        self.other = {}
        self.substituents = None
        self.backbone = None
        self.key_atoms = []

        if isinstance(structure, str) and not os.access(structure, os.R_OK):
            for ext in read_types:
                if structure.endswith(".%s" % ext):
                    structure = structure[: -(1 + len(ext))]

            for lib in [Component.AARON_LIBS, Component.BUILTIN]:
                if not os.path.exists(lib):
                    continue
                flig = None
                for f in os.listdir(lib):
                    name, ext = os.path.splitext(f)
                    if not any(".%s" % x == ext for x in read_types):
                        continue

                    match = structure == name
                    if match:
                        flig = os.path.join(lib, f)
                        break

                if flig:
                    break
            else:
                raise FileNotFoundError(
                    "Cannot find ligand in library:", structure
                )
            structure = flig
        super().__init__(structure, name, comment)

        if tag is not None:
            for a in self.atoms:
                a.add_tag(tag)

        self.other = self.parse_comment()
        try:
            self.key_atoms = self.find("key")
        except LookupError:
            if "key_atoms" in self.other:
                self.key_atoms = [
                    self.atoms[i] for i in self.other["key_atoms"]
                ]
        if key_atoms is not None:
            self.key_atoms = self.find(key_atoms)
        for a in self.key_atoms:
            a.tags.add("key")
        self.detect_backbone(to_center)
        self.rebuild()

    def __lt__(self, other):
        if len(self) != len(other):
            return len(self) < len(other)
        for a, b in zip(sorted(self.atoms), sorted(other.atoms)):
            if a < b:
                return True
        return False

    @classmethod
    def list(
        cls,
        name_regex=None,
        coordinating_elements=None,
        denticity=None,
        include_ext=False,
    ):
        names = []
        for lib in [cls.AARON_LIBS, cls.BUILTIN]:
            if not os.path.exists(lib):
                continue
            for f in os.listdir(lib):
                name, ext = os.path.splitext(f)
                if not any(".%s" % x == ext for x in read_types):
                    continue

                if name in names:
                    continue

                name_ok = True
                elements_ok = True
                denticity_ok = True

                if (
                    name_regex is not None
                    and re.search(name_regex, name, re.IGNORECASE) is None
                ):
                    name_ok = False

                if coordinating_elements is not None:
                    geom = Geometry(
                        os.path.join(lib, name + ext),
                        refresh_connected=False,
                        refresh_ranks=False,
                    )
                    # geom = cls(name)
                    elements = [
                        geom.atoms[i].element for i in geom.other["key_atoms"]
                    ]
                    if not all(
                        elements.count(x) == coordinating_elements.count(x)
                        for x in coordinating_elements
                    ) or not all(
                        coordinating_elements.count(x) == elements.count(x)
                        for x in elements
                    ):
                        elements_ok = False

                if denticity is not None:
                    geom = cls(name)
                    if len(geom.find("key")) != denticity:
                        denticity_ok = False

                if name_ok and elements_ok and denticity_ok:
                    if include_ext:
                        names.append(name + ext)
                    else:
                        names.append(name)

        return names

    def c2_symmetric(self, to_center=None, tolerance=0.1):
        """determine if center-key atom axis is a C2 axis"""
        # determine ranks
        ranks = self.canonical_rank(
            update=False,
            break_ties=False,
            invariant=False,
        )
        # remove the rank of atoms that are along the c2 axis
        ranks_off_c2_axis = []
        if to_center is None:
            center = np.zeros(3)
        else:
            center = self.COM(to_center)
        
        v = self.COM(self.key_atoms) - center
        v /= np.linalg.norm(v)
        
        for atom, rank in zip(self.atoms, ranks):
            dist_along_v = np.dot(atom.coords - center, v)
            if abs(np.linalg.norm(atom.coords - center) - dist_along_v) < tolerance:
                continue
            
            ranks_off_c2_axis.append(rank)
        
        return all([ranks.count(x) % 2 == 0 for x in set(ranks_off_c2_axis)])

    def copy(self, atoms=None, name=None, comment=None):
        rv = super().copy()
        return Component(rv)

    def rebuild(self):
        sub_atoms = []
        for sub in sorted(self.substituents):
            tmp = [sub.atoms[0]]
            tmp += sorted(sub.atoms[1:])
            for t in tmp:
                if t in sub_atoms:
                    continue
                if self.backbone and t in self.backbone:
                    continue
                sub_atoms += [t]
        if self.backbone is None:
            self.backbone = [a for a in self.atoms if a not in sub_atoms]
        self.backbone = sorted(self.backbone)
        self.atoms = self.backbone + sub_atoms

    def get_frag_list(self, targets=None, max_order=None):
        """
        find fragments connected by only one bond
        (both fragments contain no overlapping atoms)
        """
        if targets:
            atoms = self.find(targets)
        else:
            atoms = self.atoms
        frag_list = []
        for i, a in enumerate(atoms[:-1]):
            for b in atoms[i + 1 :]:
                if b not in a.connected:
                    continue

                frag_a = self.get_fragment(a, b)
                frag_b = self.get_fragment(b, a)
                if sorted(frag_a) == sorted(frag_b):
                    continue

                if len(frag_a) == 1 and frag_a[0].element == "H":
                    continue
                if len(frag_b) == 1 and frag_b[0].element == "H":
                    continue

                if max_order is not None and a.bond_order(b) > max_order:
                    continue

                if (frag_a, a, b) not in frag_list:
                    frag_list += [(frag_a, a, b)]
                if (frag_b, b, a) not in frag_list:
                    frag_list += [(frag_b, b, a)]
        return frag_list

    def detect_backbone(self, to_center=None):
        """
        Detects backbone and substituents attached to backbone
        Will tag atoms as 'backbone' or by substituent name

        :to_center:   the atoms connected to the metal/active center
        """
        # we must remove any tags already made
        for a in self.atoms:
            a.tags.discard("backbone")
        self.backbone = []
        if self.substituents is not None:
            for sub in self.substituents:
                for a in sub.atoms:
                    a.tags.discard(sub.name)
        self.substituents = []

        # get all possible fragments connected by one bond
        frag_list = self.get_frag_list()

        # get atoms connected to center
        if to_center is not None:
            to_center = self.find(to_center)
        else:
            try:
                to_center = self.find("key")
            except LookupError:
                to_center = []
            try:
                center = self.find("center")
                to_center += list(c.connected for c in center)
            except LookupError:
                center = []

        new_tags = {}  # hold atom tag options until assignment determined
        subs_found = {}  # for testing which sub assignment is best
        sub_atoms = set([])  # holds atoms assigned to substituents
        for frag_tup in sorted(frag_list, key=lambda x: len(x[0])):
            frag, start, end = frag_tup
            if frag[0] != start:
                frag = self.reorder(start=start, targets=frag)[0]
            # if frag contains atoms from to_center, it's part of backbone
            is_backbone = False
            for a in frag:
                if to_center and a in to_center:
                    is_backbone = True
                    break
            # skip substituent assignment if part of backbone
            if is_backbone:
                continue
            # try to find fragment in substituent library
            try:
                sub = Substituent(frag, end=end)
            except LookupError:
                continue
            if not to_center and len(frag) > len(self.atoms) - len(sub_atoms):
                break
            # save atoms and tags if found
            sub_atoms = sub_atoms.union(set(frag))
            subs_found[sub.name] = len(sub.atoms)
            for a in sub.atoms:
                if a in new_tags:
                    new_tags[a] += [sub.name]
                else:
                    new_tags[a] = [sub.name]
            # save substituent
            self.substituents += [sub]

        # tag substituents
        for a in new_tags:
            tags = new_tags[a]
            if len(tags) > 1:
                # if multiple substituent assignments possible,
                # want to keep the largest one (eg: tBu instead of Me)
                sub_length = []
                for t in tags:
                    sub_length += [subs_found[t]]
                max_length = max(sub_length)
                if max_length < 0:
                    max_length = min(sub_length)
                keep = sub_length.index(max_length)
                a.add_tag(tags[keep])
            else:
                a.add_tag(tags[0])

        # tag backbone
        for a in set(self.atoms) - set(sub_atoms):
            a.add_tag("backbone")
            self.backbone += [a]
        if not self.backbone:
            self.backbone = None
        return

    def capped_backbone(self, to_center=None, as_copy=True):
        if as_copy:
            comp = self.copy()
        else:
            comp = self

        if comp.backbone is None:
            comp.detect_backbone()

        subs = []
        for sub in comp.substituents:
            subs += [comp.remove_fragment(sub.atoms, sub.end, ret_frag=True)]

        if as_copy:
            comp.substituents = None
            return comp, subs
        else:
            return subs

    def minimize_sub_torsion(self, geom=None, all_frags=False):
        """
        rotates substituents to minimize LJ potential
        geom: calculate LJ potential between self and another geometry-like
              object, instead of just within self
        """
        if geom is None:
            geom = self

        if self.substituents is None:
            self.detect_backbone()

        return super().minimize_sub_torsion(geom, all_frags)

    def sub_rotate(self, start, angle=None):
        start = self.find_exact(start)[0]
        for sub in self.substituents:
            if sub.atoms[0] == start:
                break
        end = sub.end
        if angle is None:
            angle = sub.conf_angle
        if not angle:
            return
        self.change_dihedral(
            start, end, angle, fix=4, adjust=True, as_group=True
        )

    def cone_angle(
        self, center=None, method="exact", return_cones=False, radii="umn"
    ):
        """
        returns cone angle in degrees
        center - Atom() that this component is coordinating
                 used as the apex of the cone
        method (str) can be:
            'Tolman' - Tolman cone angle for asymmetric ligands 
                       See J. Am. Chem. Soc. 1974, 96, 1, 53–60 (DOI: 10.1021/ja00808a009)
                       NOTE: this does not make assumptions about the geometry
                       NOTE: only works with monodentate and bidentate ligands
            'exact' - cone angle from Allen et. al. 
                      See Bilbrey, J.A., Kazez, A.H., Locklin, J. and Allen, W.D. 
                      (2013), Exact ligand cone angles. J. Comput. Chem., 34: 
                      1189-1197. (DOI: 10.1002/jcc.23217)
        return_cones - return cone apex, center of base, and base radius list
                       the sides of the cones will be 5 angstroms long
                       for Tolman cone angles, multiple cones will be returned, one for
                       each substituent coming off the coordinating atom
        radii: 'bondi' - Bondi vdW radii
               'umn'   - vdW radii from Mantina, Chamberlin, Valero, Cramer, and Truhlar
               dict() with elements as keys and radii as values
        """
        if method.lower() == "tolman":
            CITATION = "doi:10.1021/ja00808a009"
        elif method.lower() == "exact":
            CITATION = "doi:10.1002/jcc.23217"
        self.LOG.citation(CITATION)

        key = self.find("key")

        center = self.find_exact(center)[0]

        L_axis = self.COM(key) - center.coords
        L_axis /= np.linalg.norm(L_axis)

        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII

        # list of cone data for printing bild file or w/e
        cones = []

        if method.lower() == "tolman":
            total_angle = 0
            if len(key) > 2:
                raise NotImplementedError(
                    "Tolman cone angle not implemented for tridentate or more ligands"
                )

            elif len(key) == 2:
                key1, key2 = key
                try:
                    bridge_path = self.shortest_path(key1, key2)
                except LookupError:
                    bridge_path = False

            for key_atom in key:
                L_axis = key_atom.coords - center.coords
                L_axis /= np.linalg.norm(L_axis)
                bonded_atoms = self.find(BondedTo(key_atom))
                for bonded_atom in bonded_atoms:
                    frag = self.get_fragment(bonded_atom, key_atom)

                    use_bridge = False
                    if any(k in frag for k in key):
                        # fragment on bidentate ligands that connects to
                        # the other coordinating atom
                        k = self.find(frag, key)[0]
                        # the bridge might be part of a ring (e.g. BPY)
                        # to avoid double counting the bridge, check if the
                        # first atom in the fragment is the first atom on the
                        # path from one key atom to the other
                        if frag[0] in bridge_path:
                            use_bridge = True

                    if use_bridge:
                        # angle between one L-M bond and L-M-L bisecting vector
                        tolman_angle = center.angle(k, key_atom) / 2

                    else:
                        tolman_angle = None

                        # for bidentate ligands with multiple bridges across, only use atoms that
                        # are closer to the key atom we are looking at right now
                        if len(key) == 2:
                            if bridge_path:
                                if key_atom is key1:
                                    other_key = key2
                                else:
                                    other_key = key1
                                frag = self.find(
                                    frag,
                                    CloserTo(
                                        key_atom, other_key, include_ties=True
                                    ),
                                )

                        # some ligands like DuPhos have rings on the phosphorous atom
                        # we only want ones that are closer to the the substituent end
                        frag = self.find(frag, CloserTo(bonded_atom, key_atom))

                        # Geometry(frag).write(outfile="frag%s.xyz" % bonded_atom.name)

                        for atom in frag:
                            beta = np.arcsin(
                                radii_dict[atom.element] / atom.dist(center)
                            )
                            v = center.bond(atom) / center.dist(atom)
                            c = np.linalg.norm(v - L_axis)
                            test_angle = beta + np.arccos((c ** 2 - 2) / -2)

                            if (
                                tolman_angle is None
                                or test_angle > tolman_angle
                            ):
                                tolman_angle = test_angle

                    scale = 5 * np.cos(tolman_angle)

                    cones.append(
                        (
                            center.coords + scale * L_axis,
                            center.coords,
                            scale * abs(np.tan(tolman_angle)),
                        )
                    )

                    total_angle += 2 * tolman_angle / len(bonded_atoms)

            if return_cones:
                return np.rad2deg(total_angle), cones

            return np.rad2deg(total_angle)

        elif method.lower() == "exact":
            beta = np.zeros(len(self.atoms), dtype=float)

            test_one_atom_axis = None
            max_beta = None
            for i, atom in enumerate(self.atoms):
                beta[i] = np.arcsin(
                    radii_dict[atom.element] / atom.dist(center)
                )
                if max_beta is None or beta[i] > max_beta:
                    max_beta = beta[i]
                    test_one_atom_axis = center.bond(atom)

            # check to see if all other atoms are in the shadow of one atom
            # e.g. cyano, carbonyl
            overshadowed_list = []
            for i, atom in enumerate(self.atoms):
                rhs = beta[i]

                if (
                    np.dot(center.bond(atom), test_one_atom_axis)
                    / (center.dist(atom) * np.linalg.norm(test_one_atom_axis))
                    <= 1
                ):
                    rhs += np.arccos(
                        np.dot(center.bond(atom), test_one_atom_axis)
                        / (
                            center.dist(atom)
                            * np.linalg.norm(test_one_atom_axis)
                        )
                    )
                lhs = max_beta
                if lhs >= rhs:
                    # print(atom, "is overshadowed")
                    overshadowed_list.append(atom)
                    break

            # all atoms are in the cone - we're done
            if len(overshadowed_list) == len(self.atoms):
                scale = 5 * np.cos(max_beta)

                cones.append(
                    (
                        center.coords + scale * test_one_atom_axis,
                        center.coords,
                        scale
                        * abs(
                            np.linalg.norm(test_one_atom_axis)
                            * np.tan(max_beta)
                        ),
                    )
                )

                if return_cones:
                    return np.rad2deg(2 * max_beta), cones

                return np.rad2deg(2 * max_beta)

            overshadowed_list = []
            for i, atom1 in enumerate(self.atoms):
                for j, atom2 in enumerate(self.atoms[:i]):
                    rhs = beta[i]

                    if (
                        np.dot(center.bond(atom1), center.bond(atom2))
                        / (center.dist(atom1) * center.dist(atom2))
                        <= 1
                    ):
                        rhs += np.arccos(
                            np.dot(center.bond(atom1), center.bond(atom2))
                            / (center.dist(atom1) * center.dist(atom2))
                        )
                    lhs = beta[j]
                    if lhs >= rhs:
                        overshadowed_list.append(atom1)
                        break
            # winow list to ones that aren't in the shadow of another
            atom_list = [
                atom for atom in self.atoms if atom not in overshadowed_list
            ]

            # check pairs of atoms
            max_a = None
            aij = None
            bij = None
            cij = None
            for i, atom1 in enumerate(atom_list):
                ndx_i = self.atoms.index(atom1)
                for j, atom2 in enumerate(atom_list[:i]):
                    ndx_j = self.atoms.index(atom2)
                    beta_ij = np.arccos(
                        np.dot(center.bond(atom1), center.bond(atom2))
                        / (atom1.dist(center) * atom2.dist(center))
                    )

                    test_alpha = (beta[ndx_i] + beta[ndx_j] + beta_ij) / 2
                    if max_a is None or test_alpha > max_a:
                        max_a = test_alpha
                        mi = center.bond(atom1)
                        mi /= np.linalg.norm(mi)
                        mj = center.bond(atom2)
                        mj /= np.linalg.norm(mj)

                        aij = np.sin(
                            0.5 * (beta_ij + beta[ndx_i] - beta[ndx_j])
                        ) / np.sin(beta_ij)
                        bij = np.sin(
                            0.5 * (beta_ij - beta[ndx_i] + beta[ndx_j])
                        ) / np.sin(beta_ij)
                        cij = 0
                        norm = (
                            aij * mi
                            + bij * mj
                            + cij * np.cross(mi, mj) / np.sin(bij)
                        )

            # r = 0.2 * np.tan(max_a)
            # print(
            #     ".cone %.3f %.3f %.3f   0.0 0.0 0.0   %.3f open" % (
            #         0.2 * norm[0], 0.2 * norm[1], 0.2 * norm[2], r
            #     )
            # )

            overshadowed_list = []
            rhs = max_a
            for atom in atom_list:
                ndx_i = self.atoms.index(atom)
                lhs = beta[ndx_i] + np.arccos(
                    np.dot(center.bond(atom), norm) / center.dist(atom)
                )
                # this should be >=, but there can be numerical issues
                if rhs > lhs or np.isclose(rhs, lhs):
                    overshadowed_list.append(atom)

            # the cone fits all atoms, we're done
            if len(overshadowed_list) == len(atom_list):
                scale = 5 * np.cos(max_a)

                cones.append(
                    (
                        center.coords + (scale * norm),
                        center.coords,
                        scale * abs(np.tan(max_a)),
                    )
                )

                if return_cones:
                    return np.rad2deg(2 * max_a), cones

                return np.rad2deg(2 * max_a)

            centroid = self.COM()
            c_vec = centroid - center.coords
            c_vec /= np.linalg.norm(c_vec)

            min_alpha = None
            c = 0
            for i, atom1 in enumerate(atom_list):
                for j, atom2 in enumerate(atom_list[:i]):
                    for k, atom3 in enumerate(atom_list[i + 1 :]):
                        c += 1
                        ndx_i = self.atoms.index(atom1)
                        ndx_j = self.atoms.index(atom2)
                        ndx_k = self.atoms.index(atom3)
                        # print(atom1.name, atom2.name, atom3.name)

                        mi = center.bond(atom1)
                        mi /= np.linalg.norm(center.dist(atom1))
                        mj = center.bond(atom2)
                        mj /= np.linalg.norm(center.dist(atom2))
                        mk = center.bond(atom3)
                        mk /= np.linalg.norm(center.dist(atom3))

                        gamma_ijk = np.dot(mi, np.cross(mj, mk))

                        # M = np.column_stack((mi, mj, mk))

                        # N = gamma_ijk * np.linalg.inv(M)

                        N = np.column_stack(
                            (
                                np.cross(mj, mk),
                                np.cross(mk, mi),
                                np.cross(mi, mj),
                            )
                        )

                        u = np.array(
                            [
                                np.cos(beta[ndx_i]),
                                np.cos(beta[ndx_j]),
                                np.cos(beta[ndx_k]),
                            ]
                        )

                        v = np.array(
                            [
                                np.sin(beta[ndx_i]),
                                np.sin(beta[ndx_j]),
                                np.sin(beta[ndx_k]),
                            ]
                        )

                        P = np.dot(N.T, N)

                        A = np.dot(u.T, np.dot(P, u))
                        B = np.dot(v.T, np.dot(P, v))
                        C = np.dot(u.T, np.dot(P, v))

                        D = gamma_ijk ** 2

                        # beta_ij = np.dot(center.bond(atom1), center.bond(atom2))
                        # beta_ij /= atom1.dist(center) * atom2.dist(center)
                        # beta_ij = np.arccos(beta_ij)
                        # beta_jk = np.dot(center.bond(atom2), center.bond(atom3))
                        # beta_jk /= atom2.dist(center) * atom3.dist(center)
                        # beta_jk = np.arccos(beta_jk)
                        # beta_ik = np.dot(center.bond(atom1), center.bond(atom3))
                        # beta_ik /= atom1.dist(center) * atom3.dist(center)
                        # beta_ik = np.arccos(beta_ik)
                        #
                        # D = 1 - np.cos(beta_ij) ** 2 - np.cos(beta_jk) ** 2 - np.cos(beta_ik) ** 2
                        # D += 2 * np.cos(beta_ik) * np.cos(beta_jk) * np.cos(beta_ij)
                        # this should be equal to the other D

                        t1 = (A - B) ** 2 + 4 * C ** 2
                        t2 = 2 * (A - B) * (A + B - 2 * D)
                        t3 = (A + B - 2 * D) ** 2 - 4 * C ** 2

                        w_lt = (-t2 - np.sqrt(t2 ** 2 - 4 * t1 * t3)) / (
                            2 * t1
                        )
                        w_gt = (-t2 + np.sqrt(t2 ** 2 - 4 * t1 * t3)) / (
                            2 * t1
                        )

                        alpha1 = np.arccos(w_lt) / 2
                        alpha2 = (2 * np.pi - np.arccos(w_lt)) / 2
                        alpha3 = np.arccos(w_gt) / 2
                        alpha4 = (2 * np.pi - np.arccos(w_gt)) / 2

                        for alpha in [alpha1, alpha2, alpha3, alpha4]:
                            if alpha < max_a:
                                continue

                            if min_alpha is not None and alpha >= min_alpha:
                                continue

                            lhs = (
                                A * np.cos(alpha) ** 2 + B * np.sin(alpha) ** 2
                            )
                            lhs += 2 * C * np.sin(alpha) * np.cos(alpha)
                            if not np.isclose(lhs, D):
                                continue

                            # print(lhs, D)

                            p = np.dot(
                                N, u * np.cos(alpha) + v * np.sin(alpha)
                            )
                            norm = p / gamma_ijk

                            for atom in atom_list:
                                ndx = self.atoms.index(atom)
                                rhs = beta[ndx]
                                d = np.dot(
                                    center.bond(atom), norm
                                ) / center.dist(atom)
                                if abs(d) < 1:
                                    rhs += np.arccos(d)
                                if not alpha >= rhs:
                                    break
                            else:
                                if min_alpha is None or alpha < min_alpha:
                                    # print("min_alpha set", alpha)
                                    min_alpha = alpha
                                    min_norm = norm
                                    # r = 2 * np.tan(min_alpha)
                                    # print(
                                    #     ".cone %.3f %.3f %.3f   0.0 0.0 0.0   %.3f open" % (
                                    #         2 * norm[0], 2 * norm[1], 2 * norm[2], r
                                    #     )
                                    # )

            scale = 5 * np.cos(min_alpha)

            cones.append(
                (
                    center.coords + scale * min_norm,
                    center.coords,
                    scale * abs(np.tan(min_alpha)),
                )
            )

            if return_cones:
                return np.rad2deg(2 * min_alpha), cones

            return np.rad2deg(2 * min_alpha)

        else:
            raise NotImplementedError(
                "cone angle type is not implemented: %s" % method
            )
