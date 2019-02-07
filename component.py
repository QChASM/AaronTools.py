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
        name
        comment
        atoms
        other
        substituents
        backbone
        key_atoms (for ligand only)
    """

    def __init__(self, comp, targets=None, tag=None, name='', comment=''):
        """
        comp is either a file, a geometry, or an atom list
        """
        self.name = name
        self.comment = comment
        self.other = {}
        self.substituents = None
        self.backbone = None
        self.key_atoms = []

        if isinstance(comp, (Geometry, list)):
            # we can create object from fragment
            # save atom info
            if targets is None:
                try:
                    self.atoms = comp.atoms
                    if name == '':
                        self.name = comp.name
                    if comment == '':
                        self.comment = comp.comment
                except AttributeError:
                    self.atoms = comp
            else:
                self.atoms = comp.find(targets)

        else:
            # or we can create from file
            # load in atom info
            from_file = FileReader(comp)
            self.name = comp
            self.comment = from_file.comment
            self.atoms = from_file.atoms
            if targets is not None:
                self.atoms = self.find(targets)

        if tag is not None:
            for a in self.atoms:
                a.add_tag(tag)
        self.other = self.parse_comment()
        if 'key_atoms' in self.other:
            self.key_atoms = self.other['key_atoms']

        self.refresh_connected()
        self.detect_backbone()

    def copy(self, atoms=None, name=None, comment=None):
        if name is None:
            name = self.name + "_copy"
        if comment is None:
            comment = deepcopy(self.comment)
        rv = super().copy(atoms, name, comment)
        rv = Component(rv)
        return rv

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
            removed_sub = Substituent(removed_sub)
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
        self.refresh_connected()

    def get_frag_list(self, targets=None, max_order=None):
        """
        find fragments connected by only one bond
        (both fragments contain no overlapping atoms)
        """
        if targets is None:
            atoms = self.atoms
        else:
            atoms = self.find(targets)
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
        Parameters:
            to_center   the atoms connected to the metal/active center
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
        # smallest to largest
        for k in sorted(targets.keys()):
            for sub in targets[k]:
                axis = sub.atoms[0].bond(sub.end)
                center = sub.end
                self.minimize_torsion(sub.atoms, axis, center, geom)

    def make_conformer(self, targets=None):
        """
        Yields conformers made by rotating substituents.
        Returns:
            updated component and a code indicating which substituents
            were rotated
            eg: new_conf, (1, 0, 2)
        Parameters:
            skip - atom list which, if any found in a substituent, will
                skip rotation of that entire substituent
            targets - a list of substituents. If provided, will only
                rotate the substituents in `target`
        """
        # validate targets
        if targets is None:
            targets = self.substituents
        tmp = list(targets)
        for t in targets:
            if t.conf_num == 1 or t.conf_angle == 0.0:
                tmp.remove(t)
                continue
            if t not in self.substituents:
                tmp.remove(t)
                continue
        targets = list(tmp)
        if len(targets) == 0:
            yield None, None
            return

        # yeild conformers
        max_confs = max([t.conf_num for t in targets])
        max_confs = list(range(max_confs+1))
        for confs in it.product(max_confs, repeat=len(targets)):
            for i, c in enumerate(confs):
                if c != 0:
                    targets[i].sub_rotate()
                    for a in targets[i].atoms:
                        a.flag = False
            yield(self.name.split('_')[-1], confs)
