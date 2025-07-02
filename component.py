"""For more complicated geometry manipulation and complex building"""
import os
import re

import numpy as np

from scipy.spatial import distance_matrix

from AaronTools.const import (
    AARONTOOLS,
    BONDI_RADII,
    ELEMENTS,
    VDW_RADII,
)
from AaronTools.fileIO import read_types
from AaronTools.finders import BondedTo, CloserTo, NotAny
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.decorators import classproperty
from AaronTools.utils.utils import (
    perp_vector,
    lebedev_sphere,
    fibonacci_sphere,
)


class Component(Geometry):
    """
    class for parts of a Geometry (e.g. ligands)
    Attributes:
    
    * name           str
    * comment        str
    * atoms          list(Atom)
    * other          dict()
    * substituents   list(Substituent) substituents detected
    * backbone       list(Atom) the backbone atoms
    * key_atoms      list(Atom) the atoms used for mapping
    
    """

    BUILTIN = os.path.join(AARONTOOLS, "Ligands")
    FROM_SUBSTITUENTS = set([])

    def __init__(
        self,
        structure,
        name="",
        comment=None,
        tag=None,
        to_center=None,
        key_atoms=None,
        detect_backbone=True,
        **kwargs,
    ):
        """
        comp is either a file, a geometry, or an atom list
        """
        super().__init__(**kwargs)
        self.name = name
        self.comment = comment
        self.other = {}
        self.substituents = []
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
                try:
                    structure = Substituent(structure)
                    Component.FROM_SUBSTITUENTS.add(structure.name)
                    self.__init__(structure, comment="K:1")
                    return
                except Exception:
                    raise FileNotFoundError(
                        "Cannot find ligand in library:", structure
                    )
            structure = flig
        super().__init__(structure, name, comment, **kwargs)

        if tag is not None:
            for a in self.atoms:
                a.add_tag(tag)

        self.other = self.parse_comment()
        if key_atoms is not None:
            # print("looking for key atoms:", key_atoms)
            self.key_atoms = self.find(key_atoms)
            # print("found key atoms:", self.key_atoms)
        else:
            self.key_atoms = self.find("key")
            if not self.key_atoms:
                if "key_atoms" in self.other:
                    self.key_atoms = [
                        self.atoms[i] for i in self.other["key_atoms"]
                    ]

        for a in self.key_atoms:
            a.tags.add("key")
        if detect_backbone:
            self.detect_backbone(to_center)
        self.rebuild()

    def __lt__(self, other):
        if len(self) != len(other):
            return len(self) < len(other)
        for a, b in zip(sorted(self.atoms), sorted(other.atoms)):
            if a < b:
                return True
        return False

    @classproperty
    def AARON_LIBS(cls):
        from AaronTools.const import AARONLIB
        return os.path.join(AARONLIB, "Ligands")

    @classmethod
    def list(
        cls,
        name_regex=None,
        coordinating_elements=None,
        denticity=None,
        include_ext=False,
    ):
        """
        returns a list of ligand names in the library

        :param str name_regex: regex identifier/name for a library to search
        :param list(Atom) coordinating_elements: if specified, only lists ligands with included coordinated elements
        :param int denticity: if specified, lists ligands with included denticity
        :param bool include_ext: includes extensions on each item if True
        """
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

        return names + sorted(cls.FROM_SUBSTITUENTS)

    def c2_symmetric(self, to_center=None, tolerance=0.1):
        """
        determine if center-key atom axis is a C2 axis
        
        :param list(Atom) to_center: arg. for Geometry.COM(), atoms connected to center
        :param float tolerance: buffer for consideration of axis symmetry

        :returns: True if axis is C2, otherwise False
        :rtype: boolean
        """
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
            if (
                abs(np.linalg.norm(atom.coords - center) - dist_along_v)
                < tolerance
            ):
                continue

            ranks_off_c2_axis.append(rank)

        return all([ranks.count(x) % 2 == 0 for x in set(ranks_off_c2_axis)])

    def sterimol(self, to_center=None, bisect_L=False, **kwargs):
        """
        calculate ligand sterimol parameters for the ligand
        
        :param Atom to_center: atom the ligand is coordinated to
        :param bool bisect_L: L axis will bisect (or analogous for higher denticity
            ligands) the L-M-L angle
            
            Default - center to centroid of key atoms
        :param kwargs: - arguments passed to Geometry.sterimol
        """
        if to_center is not None:
            center = self.find(to_center)
        else:
            center = self.find(
                [BondedTo(atom) for atom in self.key_atoms], NotAny(self.atoms)
            )

        if len(center) != 1:
            raise TypeError(
                "wrong number of center atoms specified;\n"
                "expected 1, got %i" % len(center)
            )
        center = center[0]

        if bisect_L:
            L_axis = np.zeros(3)
            for atom in self.key_atoms:
                v = center.bond(atom)
                v /= np.linalg.norm(v)
                v /= len(self.key_atoms)
                L_axis += v
        else:
            L_axis = self.COM(self.key_atoms) - center.coords
            L_axis /= np.linalg.norm(L_axis)

        return super().sterimol(L_axis, center, self.atoms, **kwargs)

    def copy(self, atoms=None, name=None, comment=None):
        """
        creates a new copy of the geometry

        :param list(Atom) atoms: atoms to copy defaults to all atoms
        :param str name: defaults to NAME_copy
        :param str comment: comment to add to the copy, defaults to self's comment

        :returns: copy of self
        :rtype: Geometry
        """
        rv = super().copy()
        return Component(rv)

    def rebuild(self):
        sub_atoms = []
        if self.substituents:
            for sub in sorted(self.substituents):
                tmp = [sub.atoms[0]]
                tmp += sorted(sub.atoms[1:])
                for t in tmp:
                    if t in sub_atoms:
                        continue
                    if self.backbone and t in self.backbone:
                        continue
                    sub_atoms += [t]
            sub_atoms_set = set(sub_atoms)
            if self.backbone is None:
                self.backbone = [a for a in self.atoms if a not in sub_atoms_set]
            self.backbone = sorted(self.backbone)
            self.atoms = self.backbone + sub_atoms
        else:
            self.backbone = self.atoms.copy()

    def get_frag_list(self, targets=None, max_order=None):
        """
        find fragments connected by only one bond
        (both fragments contain no overlapping atoms)

        :param list(Atom) targets: all fragments must include targets if specified
        :param int max_order: maximum order/length for fragments

        :returns: all fragments under specified conditions
        :rtype: list(Geometry)
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
                if len(frag_a) == len(frag_b) and sorted(
                    frag_a, key=lambda x: ELEMENTS.index(x.element)
                ) == sorted(frag_b, key=lambda x: ELEMENTS.index(x.element)):
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

        :param list(Atom) to_center:   the atoms connected to the metal/active center
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
            to_center = self.find("key")
            center = self.find("center")
            to_center += list(c.connected for c in center)

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

    def minimize_sub_torsion(self, geom=None, **kwargs):
        """
        rotates substituents to minimize LJ potential
        
        :param None|Geometry geom: calculate LJ potential between self and another geometry-like
            object, instead of just within self
        """
        if geom is None:
            geom = self

        if self.substituents is None:
            self.detect_backbone()

        return super().minimize_sub_torsion(geom, **kwargs)

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
        self,
        center=None,
        method="exact",
        return_cones=False,
        return_individual=False,
        radii="umn",
    ):
        """
        returns cone angle in degrees
        
        :param Atom center: that this component is coordinating
            used as the apex of the cone
        :param str method: can be:
            
            * 'Tolman' - Tolman cone angle for unsymmetric ligands
              
              See J. Am. Chem. Soc. 1974, 96, 1, 53–60 (DOI: 10.1021/ja00808a009)
               
              :NOTE: this does not make assumptions about the geometry
               
            * 'exact' - cone angle from Allen et. al.
              
              See Bilbrey, J.A., Kazez, A.H., Locklin, J. and Allen, W.D.
              (2013), Exact ligand cone angles. J. Comput. Chem., 34:
              1189-1197. (DOI: 10.1002/jcc.23217)
        :param bool return_cones: return cone apex, center of base, and base radius list
            the sides of the cones will be 5 angstroms long
            
            for Tolman cone angles, multiple cones will be returned, one for
            each substituent coming off the coordinating atom
        :param bool return_individual: return the angles for each individual
            Tolman cone
        :param str|dict radii:
            * 'bondi' - Bondi vdW radii
            * 'umn'   - vdW radii from Mantina, Chamberlin, Valero, Cramer, and Truhlar
            * dict() with elements as keys and radii as values
        """
        from AaronTools.utils.utils import xyzzy_cross
        if method.lower() == "tolman":
            CITATION = "doi:10.1021/ja00808a009"
        elif method.lower() == "exact":
            CITATION = "doi:10.1002/jcc.23217"
        self.LOG.citation(CITATION)
        
        center = self.find_exact(center)[0]

        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII

        # list of cone data for printing bild file or w/e
        cones = []

        if method.lower() == "tolman":
            key = self.find("key")
            
            L_axis = self.COM(key) - center.coords
            L_axis /= np.linalg.norm(L_axis)

            total_angle = 0
            all_cones = {"substituents": [], "bridges": []}
            bridges = dict()
            if len(key) > 1:
                for i, key1 in enumerate(key):
                    bridges.setdefault(key1, [])
                    for key2 in key[:i]:
                        bridges.setdefault(key2, [])
                        try:
                            bridge = self.shortest_path(key1, key2, avoid=center)
                            if any(k in bridge for k in key if k not in (key1, key2)):
                                continue
                            bridges[key1].append(bridge)
                            bridges[key2].append(bridge)
                        except LookupError:
                            pass

            for key_atom in key:
                L_axis = key_atom.coords - center.coords
                L_axis /= np.linalg.norm(L_axis)
                bonded_atoms = self.find(BondedTo(key_atom))
                if not bonded_atoms:
                    continue
                for bonded_atom in bonded_atoms:
                    frag = self.get_fragment(bonded_atom, key_atom)

                    use_bridge = False
                    if any(k in frag for k in key):
                        # fragment on bidentate ligands that connects to
                        # the other coordinating atom
                        ks = self.find(frag, key)
                        k = sorted(
                            ks,
                            key=lambda x: len(self.shortest_path(x, key_atom)),
                            reverse=False,
                        )[0]
                        # the bridge might be part of a ring (e.g. BPY)
                        # to avoid double counting the bridge, check if the
                        # first atom in the fragment is the first atom on the
                        # path from one key atom to the other
                        if any(frag[0] in bridge_path for bridge_path in bridges[key_atom]):
                            use_bridge = True

                    if use_bridge:
                        # angle between one L-M bond and L-M-L bisecting vector
                        tolman_angle = center.angle(k, key_atom) / 2
                        all_cones["bridges"].append(np.rad2deg(tolman_angle))

                    else:
                        tolman_angle = None

                        # for bidentate ligands with multiple bridges across, only use atoms that
                        # are closer to the key atom we are looking at right now
                        if len(key) > 1:
                            if bridges[key_atom]:
                                if key_atom is key1:
                                    other_key = key2
                                else:
                                    other_key = key1
                                closer_atoms = (
                                    CloserTo(key_atom, other_key, include_ties=True)
                                    for other_key in key if other_key is not key_atom
                                )
                                frag = self.find(
                                    frag, *closer_atoms,
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

                        all_cones["substituents"].append(np.rad2deg(tolman_angle))

                    scale = 5 * np.cos(tolman_angle)

                    cones.append(
                        (
                            center.coords + scale * L_axis,
                            center.coords,
                            scale * abs(np.tan(tolman_angle)),
                        )
                    )

                    total_angle += 2 * tolman_angle / len(bonded_atoms)

            if not return_individual and not return_cones:
                out = np.rad2deg(total_angle)
            else:
                out = [np.rad2deg(total_angle)]
                if return_cones:
                    out.append(cones)
                
                if return_individual:
                    out.append(all_cones)

            return out

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
            ndx = {a: i for i, a in enumerate(self.atoms)}
            for i, atom1 in enumerate(atom_list):
                ndx_i = ndx[atom1]
                for j, atom2 in enumerate(atom_list[:i]):
                    ndx_j = ndx[atom2]
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
                        max_a_norm = (
                            aij * mi
                            + bij * mj
                            + cij * xyzzy_cross(mi, mj) / np.sin(bij)
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
                ndx_i = ndx[atom]
                lhs = beta[ndx_i] + np.arccos(
                    np.dot(center.bond(atom), max_a_norm) / center.dist(atom)
                )
                # this should be >=, but there can be numerical issues
                if rhs > lhs or np.isclose(rhs, lhs):
                    overshadowed_list.append(atom)

            # the cone fits all atoms, we're done
            if len(overshadowed_list) == len(atom_list):
                scale = 5 * np.cos(max_a)

                cones.append(
                    (
                        center.coords + (scale * max_a_norm),
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
                        ndx_i = ndx[atom1]
                        ndx_j = ndx[atom2]
                        ndx_k = ndx[atom3]
                        # print(atom1.name, atom2.name, atom3.name)

                        mi = center.bond(atom1)
                        mi /= np.linalg.norm(center.dist(atom1))
                        mj = center.bond(atom2)
                        mj /= np.linalg.norm(center.dist(atom2))
                        mk = center.bond(atom3)
                        mk /= np.linalg.norm(center.dist(atom3))

                        gamma_ijk = np.dot(mi, xyzzy_cross(mj, mk))

                        # M = np.column_stack((mi, mj, mk))

                        # N = gamma_ijk * np.linalg.inv(M)

                        N = np.column_stack(
                            (
                                xyzzy_cross(mj, mk),
                                xyzzy_cross(mk, mi),
                                xyzzy_cross(mi, mj),
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
                            if np.isnan(alpha):
                                continue

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
                                ndx_i = self.atoms.index(atom)
                                rhs = beta[ndx_i]
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

            if min_alpha is None:
                min_alpha = max_a
                # min_norm = max_a_norm
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

    def solid_angle(
        self,
        center,
        radii="umn",
        grid=5810,
        return_solid_cone=False,
    ):
        """
        calculate the solid angle of a ligand
        
        :param Atom center: atoms or point to denote the center of the sphere
        :param str|dict radii: "umn", "bondi", or a dictionary with elements as
            the keys and radii as the values
        :param int grid: number of points in lebedev grid
        :param bool return_solid_cone: return solid ligand cone angle (degrees)
            instead of solid angle (steradians)
        """
        # we calculate the solid angle by projecting each atom's
        # radius onto a unit sphere around the center
        # the radii of the shadow of the atoms depends on
        # the original radius and the distance from the center

        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII

        radii_list = np.array([
            radii_dict[atom.element] for atom in self.atoms
        ])

        if not isinstance(center, np.ndarray):
            center = np.mean(self.coordinates(center), axis=0)
        
        coords = self.coords
        shifted_coords = coords - center
        
        dx2 = np.sum(shifted_coords * shifted_coords, axis=1)
        dist = np.sqrt(dx2)

        # X is the height of each atom's cone that touches the unit sphere
        # divided by the distance to the atom
        X = np.sqrt(dx2 - radii_list ** 2) / dx2
        if np.any(np.isnan(X)):
            return 4 * np.pi
        # adjusted_coords are the points inside the unit circle (shifted to be
        # centered at the origin) that are at the center of the base of each cone
        adjusted_coords = X[:, np.newaxis] * shifted_coords
        # H is the radius of each atom's cone on the unit sphere 
        H = radii_list / dist

        int_grid, weights = lebedev_sphere(
            radius=1, center=np.zeros(3), num=grid,
        )

        dist = distance_matrix(int_grid, adjusted_coords)

        # figure out which points are inside a shadow and
        # sum the corresponding weights
        mask = np.any(dist - H[np.newaxis, :] <= 0, axis=1)
        fraction = sum(weights[mask])

        solid_angle = 4 * np.pi * fraction
        if not return_solid_cone:
            return solid_angle
         
        ligand_solid_angle = 2 * np.arccos(
            1 - (solid_angle / (2 * np.pi))
        )
        return np.rad2deg(ligand_solid_angle)
