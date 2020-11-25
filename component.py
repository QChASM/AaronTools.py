"""For more complicated geometry manipulation and complex building"""
import os
import re
from glob import glob

from AaronTools.const import AARONLIB, AARONTOOLS
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent


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

    AARON_LIBS = os.path.join(AARONLIB, "Ligands", "*.xyz")
    BUILTIN = os.path.join(AARONTOOLS, "Ligands", "*.xyz")

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
        self.name = name
        self.comment = comment
        self.other = {}
        self.substituents = None
        self.backbone = None
        self.key_atoms = []

        if isinstance(structure, str) and not os.access(structure, os.R_OK):
            if structure.endswith(".xyz"):
                structure = structure[:-4]
            for f in glob(Component.AARON_LIBS) + glob(Component.BUILTIN):
                match = structure + ".xyz" == os.path.basename(f)
                if match:
                    structure = f
                    break
            else:
                raise FileNotFoundError(
                    "Cannot find ligand in library:", structure
                )
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
    def list(cls, name_regex=None, coordinating_elements=None):
        names = []
        for f in glob(cls.AARON_LIBS) + glob(cls.BUILTIN):
            name = os.path.splitext(os.path.basename(f))[0]
            name_ok = True
            elements_ok = True

            if (
                name_regex is not None
                and re.search(name_regex, name, re.IGNORECASE) is None
            ):
                name_ok = False

            if coordinating_elements is not None:
                geom = cls(name)
                elements = [atom.element for atom in geom.find("key")]
                if not all(
                    elements.count(x) == coordinating_elements.count(x)
                    for x in coordinating_elements
                ) or not all(
                    coordinating_elements.count(x) == elements.count(x)
                    for x in elements
                ):
                    elements_ok = False

            if name_ok and elements_ok:
                names.append(name)

        return names

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
