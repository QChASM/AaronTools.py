"""For more complicated geometry manipulation and complex building"""
import numpy as np
import itertools as it

from warnings import warn
from copy import deepcopy

from AaronTools.fileIO import FileReader
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

    def __init__(self, structure, name='', comment='',
                 tag=None, to_center=None):
        """
        comp is either a file, a geometry, or an atom list
        """
        self.name = name
        self.comment = comment
        self.other = {}
        self.substituents = None
        self.backbone = None
        self.key_atoms = []

        super().__init__(structure, name, comment)

        if tag is not None:
            for a in self.atoms:
                a.add_tag(tag)

        self.other = self.parse_comment()
        try:
            self.key_atoms = self.find('key')
        except LookupError:
            if 'key_atoms' in self.other:
                self.key_atoms = [self.atoms[i]
                                  for i in self.other['key_atoms']]

        self.refresh_connected(rank=False)
        self.detect_backbone(to_center)

    def copy(self, atoms=None, name=None, comment=None):
        if atoms is None:
            atoms = deepcopy(self.atoms)
        if name is None:
            name = self.name
        if comment is None:
            comment = self.comment
        return Component(atoms, name, comment)

    def substitute(self, sub, target, attached_to=None):
        """
        substitutes fragment containing `target` with substituent `sub`
        if end provided, this is the atom where the substituent is attached
        if end==None, replace the smallest fragment containing `target`
        """
        # set up substituent object
        if not isinstance(sub, Substituent):
            sub = Substituent(sub)
        # tag substituent atoms with substituent name
        for a in sub.atoms:
            a.add_tag(sub.name)

        # determine target and atoms defining connection bond
        target = self.find(target)

        # attached_to is provided or is the atom giving the
        # smallest target fragment
        if attached_to is not None:
            attached_to = self.find_exact(attached_to)
        else:
            smallest_frag = None
            smallest_attached_to = None
            # get all possible connection points
            attached_to = set()
            for t in target:
                attached_to = attached_to | (t.connected - set(target))
            # find smallest fragment
            for e in attached_to:
                frag = self.get_fragment(target, e)
                if smallest_frag is None or len(frag) < len(smallest_frag):
                    smallest_frag = frag
                    smallest_attached_to = e
            attached_to = [smallest_attached_to]
        if len(attached_to) != 1:
            raise NotImplementedError(
                "Can only replace substituents with one point of attachment")
        attached_to = attached_to[0]
        sub.end = attached_to

        # determine which atom of target fragment is connected to attached_to
        sub_attach = attached_to.connected & set(target)
        if len(sub_attach) > 1:
            raise NotImplementedError(
                "Can only replace substituents with one point of attachment")
        if len(sub_attach) < 1:
            raise LookupError("attached_to atom not connected to targets")
        sub_attach = sub_attach.pop()

        # manipulate substituent geometry; want sub.atoms[0] -> sub_attach
        #   attached_to == sub.end
        #   sub_attach will eventually be sub.atoms[0]
        # move attached_to to the origin
        shift = attached_to.coords.copy()
        self.coord_shift(-1*shift)
        # align substituent to current bond
        bond = self.bond(attached_to, sub_attach)
        sub.align_to_bond(bond)
        # shift geometry back and shift substituent to appropriate place
        self.coord_shift(shift)
        sub.coord_shift(shift)

        # tag and update name for sub atoms
        for s in sub.atoms:
            s.add_tag(sub.name)
            s.name = sub_attach.name + "." + s.name

        # remove old substituent
        removed_sub = self.remove_fragment(
            sub_attach, attached_to, add_H=False)
        if len(removed_sub) == 1 and removed_sub[0].element == 'H':
            pass
        else:
            try:
                removed_sub = Substituent(removed_sub)
            except LookupError:
                removed_sub = Substituent(
                    removed_sub, conf_angle=0, conf_num=1)
            try:
                self.substituents.remove(removed_sub)
            except ValueError:
                warn(
                    "Requested substituent to replace was not automatically "
                    + "detected upon loading of XYZ file.")
        self -= sub_attach
        attached_to.connected.discard(sub_attach)

        # add new substituent
        self += sub
        self.substituents += [sub]
        attached_to.connected.add(sub.atoms[0])

        # fix bond distance
        self.change_distance(attached_to, sub.atoms[0], as_group=True, fix=1)
        # fix connection info
        self.refresh_connected(rank=False)

    def get_frag_list(self, targets=None, max_order=None):
        """
        find fragments connected by only one bond
        (both fragments contain no overlapping atoms)
        """
        self.refresh_connected(rank=False)
        if targets:
            atoms = self.find(targets)
        else:
            atoms = self.atoms
        frag_list = []
        for i, a in enumerate(atoms[:-1]):
            for b in atoms[i+1:]:
                if b not in a.connected:
                    continue

                frag_a = self.get_fragment(a, b)
                frag_b = self.get_fragment(b, a)

                if len(frag_a) == 1 and frag_a[0].element == 'H':
                    continue
                if len(frag_b) == 1 and frag_b[0].element == 'H':
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
        if self.backbone is not None:
            for a in self.backbone:
                a.tags.discard('backbone')
            self.backbone = None
        if self.substituents is not None:
            for sub in self.substituents:
                for a in sub.atoms:
                    a.tags.discard(sub.name)
            self.substituents = []
        self.substituents = []

        # get all possible fragments connected by one bond
        frag_list = self.get_frag_list()

        # get atoms connected to center
        if to_center is not None:
            to_center = self.find_exact(to_center)
        else:
            try:
                to_center = self.find(['key', 'center'])
            except LookupError:
                pass

        new_tags = {}  # hold atom tag options until assignment determined
        subs_found = {}  # for testing which sub assignment is best
        sub_atoms = []  # holds atoms assigned to substituents
        for frag_tup in frag_list:
            frag, start, end = frag_tup
            if frag[0] != start:
                frag = ([start]
                        + frag[:frag.index(start)]
                        + frag[frag.index(start) + 1:])
            # if frag contains atoms from to_center, it's part of backbone
            if to_center is not None:
                is_backbone = False
                for a in frag:
                    if a in to_center:
                        is_backbone = True
                        break
                if is_backbone:
                    # skip substituent assignment if part of backbone
                    continue
            # try to find fragment in substituent library
            try:
                sub = Substituent(frag, end=end)
            except LookupError:
                continue
            # save atoms and tags if found
            sub_atoms += sub.atoms
            subs_found[sub.name] = len(sub.atoms)
            for a in sub.atoms:
                try:
                    new_tags[a] += [sub.name]
                except KeyError:
                    new_tags[a] = [sub.name]
            # save substituent
            try:
                self.substituents += [sub]
            except TypeError:
                self.substituents = [sub]

        # tag substituents
        for a in new_tags:
            tags = new_tags[a]
            if len(tags) > 1:
                # if multiple substituent assignments possible,
                # want to keep the largest one (eg: tBu instead of Me)
                sub_length = [subs_found[t] for t in tags]
                max_length = max(sub_length)
                keep = sub_length.index(max_length)
                a.add_tag(tags[keep])
            else:
                a.add_tag(tags[0])

        # tag backbone
        backbone = []
        for a in self.atoms:
            if a in sub_atoms:
                continue
            a.add_tag('backbone')
            backbone += [a]
        self.backbone = sorted(backbone)

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

    def minimize_sub_torsion(self, geom=None):
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
                for frag, a, b in self.get_frag_list(targets=sub.atoms,
                                                     max_order=1):
                    axis = a.bond(b)
                    center = b.coords
                    self.minimize_torsion(frag, axis, center, geom)
