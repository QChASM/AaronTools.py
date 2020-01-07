"""For more complicated geometry manipulation and complex building"""
import os
import re
from glob import glob

from AaronTools.const import AARONLIB, QCHASM
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
    BUILTIN = os.path.join(QCHASM, "AaronTools", "Ligands", "*.xyz")

    def __init__(
        self,
        structure,
        name="",
        comment="",
        tag=None,
        to_center=None,
        key_atoms=None,
        refresh_connected=False,
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

        super().__init__(structure, name, comment, refresh_connected)

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

    def __lt__(self, other):
        for a, b in zip(self.atoms, other.atoms):
            if float(a) != float(b):
                return float(a) < float(b)
        return False

    def copy(self, atoms=None, name=None, comment=None):
        rv = super().copy()
        return Component(rv)

    def rebuild(self):
        atoms = self.backbone
        for sub in sorted(self.substituents):
            for a in sub.atoms:
                if a not in atoms:
                    atoms += [a]
        self.atoms = atoms

    def substitute(self, sub, target, attached_to=None):
        """
        substitutes fragment containing `target` with substituent `sub`
        if end provided, this is the atom where the substituent is attached
        if end==None, replace the smallest fragment containing `target`
        """
        if not isinstance(sub, Substituent):
            sub = Substituent(sub)

        Geometry.substitute(self, sub, target, attached_to)
        self.detect_backbone()

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
            for b in atoms[i + 1:]:
                if b not in a.connected:
                    continue

                frag_a = self.get_fragment(a, b)
                frag_b = self.get_fragment(b, a)

                if len(frag_a) == 1 and frag_a[0].element == "H":
                    continue
                if len(frag_b) == 1 and frag_b[0].element == "H":
                    continue

                if max_order is not None and a.bond_order(b) > max_order:
                    continue

                if len(set(frag_a) & set(frag_b)) == 0:
                    if frag_a not in frag_list:
                        frag_list += [(frag_a, a, b)]
                    if frag_b not in frag_list:
                        frag_list += [(frag_b, b, a)]
        return frag_list

    def detect_backbone(self, to_center=None):
        """
        Detects backbone and substituents attached to backbone
        Will tag atoms as 'backbone' or by substituent name

        :to_center:   the atoms connected to the metal/active center
        """
        # handle the case in which we want to refresh the backbone
        # we must remove any tags already made
        refresh_connected = True
        for a in self.atoms:
            if refresh_connected and a.connected:
                refresh_connected = False
            a.tags.discard("backbone")
        if refresh_connected:
            self.refresh_connected()
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
            to_center = self.find_exact(to_center)
        else:
            try:
                to_center = self.find("key")
            except LookupError:
                to_center = []
            try:
                center = self.find("center")
            except LookupError:
                center = []
            to_center += list(c.connected for c in center)

        new_tags = {}  # hold atom tag options until assignment determined
        subs_found = {}  # for testing which sub assignment is best
        sub_atoms = set([])  # holds atoms assigned to substituents
        for frag_tup in frag_list:
            frag, start, end = frag_tup
            if frag[0] != start:
                frag = (
                    [start]
                    + frag[: frag.index(start)]
                    + frag[frag.index(start) + 1:]
                )
            # if frag contains atoms from to_center, it's part of backbone
            if to_center:
                is_backbone = False
                for a in frag:
                    if a in to_center:
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
                    if t.startswith("sub-"):
                        sub_length += [-subs_found[t]]
                    else:
                        sub_length += [subs_found[t]]
                max_length = max(sub_length)
                if max_length < 0:
                    max_length = min(sub_length)
                keep = sub_length.index(max_length)
                a.add_tag(tags[keep])
            else:
                a.add_tag(tags[0])

        # tag backbone
        for a in self.atoms:
            if a in sub_atoms:
                continue
            a.add_tag("backbone")
            self.backbone += [a]
        self.rebuild()
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

        targets = {}
        for sub in self.substituents:
            try:
                targets[len(sub.atoms)] += [sub]
            except KeyError:
                targets[len(sub.atoms)] = [sub]

        # minimize torsion for each substituent
        # largest to smallest
        for k in sorted(targets.keys(), reverse=True):
            for sub in targets[k]:
                axis = sub.atoms[0].bond(sub.end)
                center = sub.end
                self.minimize_torsion(sub.atoms, axis, center, geom)
                if all_frags:
                    for frag, a, b in self.get_frag_list(
                        targets=sub.atoms, max_order=1
                    ):
                        axis = a.bond(b)
                        center = b.coords
                        self.minimize_torsion(frag, axis, center, geom)

