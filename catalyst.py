import numpy as np
import itertools as it
from copy import deepcopy

from AaronTools.const import TMETAL
from AaronTools.geometry import Geometry
from AaronTools.component import Component
from AaronTools.substituent import Substituent


class Catalyst(Geometry):
    """
    Attributes:
        name
        comment
        atoms
        other
        center      the metal center or active center
        components  holds ligands and substrates
        conf_num    conformer number
    """

    def __init__(self, fname='', name='', comment='', atoms=None):
        self.center = None
        self.components = None
        self.conf_num = 0
        Geometry.__init__(self, fname, name, comment, atoms)
        if isinstance(fname, str) and fname == '' and atoms is None:
            return
        self.other = self.parse_comment()
        self.detect_components()
        self.conf_num = 1

    def copy(self, atoms=None, name=None, comment=None):
        if name is None:
            name = self.name + "_copy"
        if comment is None:
            comment = deepcopy(self.comment)
        rv = super().copy(atoms, name, comment=self.comment)
        rv = Catalyst(rv)
        return rv

    def rebuild(self):
        atoms = []
        for comp in self.components['substrate']:
            atoms += comp.atoms
        if self.center:
            atoms += self.center
        for comp in self.components['ligand']:
            atoms += comp.atoms
        self.atoms = atoms
        self.refresh_connected()

    def detect_components(self, debug=False):
        self.components = {}
        self.center = []

        # get center
        lig_assigned = False
        center_max = None
        for a in self.atoms:
            if 'ligand' not in a.tags and a.element in TMETAL.keys():
                # detect transition metal center
                if a not in self.center:
                    self.center += [a]
                a.add_tag('center')
            if 'center' in a.tags:
                # center provided by comment line in xyz file
                if a not in self.center:
                    self.center += [a]
            if not lig_assigned and 'ligand' in a.tags:
                lig_assigned = True
            if a in self.center:
                if center_max is None or center_max < float(a):
                    center_max = float(a)

        if not lig_assigned and len(self.center) < 1:
            raise IOError(
                "Non-transition metal centered catalysts must either have "
                + "active centers or ligand atoms specified in the comment "
                + "line of the XYZ input file")

        # label ligand and substrate
        lig = []
        subst = []
        for a in self.atoms:
            if lig_assigned:
                if 'ligand' in a.tags:
                    lig += [a]
            elif float(a) > center_max:
                a.add_tag('ligand')
                lig += [a]

        for a in self.atoms:
            if 'ligand' not in a.tags and 'center' not in a.tags:
                a.add_tag('substrate')
                subst += [a]
        if debug:
            print("lig", [a.name for a in lig])
            print("sub", [a.name for a in subst])
            print("center", [a.name for a in self.center])

        # get components
        if len(self.center) > 0:
            self.components['ligand'] = self.detect_fragments(lig)
            self.components['substrate'] = self.detect_fragments(subst)
        else:
            self.components['ligand'] = self.detect_fragments(lig, subst)
            self.components['substrate'] = self.detect_fragments(subst, lig)
        # rename
        for lig in self.components['ligand']:
            lig.name = self.name + '_lig-{}'.format(lig.atoms[0].name)
        for sub in self.components['substrate']:
            sub.name = self.name + '_sub-{}'.format(sub.atoms[0].name)
        return

    def detect_fragments(self, targets, avoid=None):
        """
        Returns a list of Geometries in which the connection to other
        atoms in the larger geometry must go through the center atoms
        eg: L1--C--L2 will give two fragments, L1 and L2
            (  /
            L1/
        """
        rv = []
        found = set([])
        if avoid is None:
            avoid = self.center
        for a in targets:
            if a in found:
                continue
            if a in avoid:
                continue
            frag = self.get_fragment(a, avoid)
            frag = Component(frag)
            for f in frag.atoms:
                found.add(f)
            rv += [frag]
        return rv

    def map_ligand(self, ligands, old_keys):
        """
        Maps new ligand according to key_map
        Parameters:
            ligand      the name of a ligand in the ligand library
            old_keys    the key atoms of the old ligand to map to
        """
        def get_rotation(old_axis, new_axis):
            w = np.cross(old_axis, new_axis)
            angle = np.dot(old_axis, new_axis)
            angle /= np.linalg.norm(old_axis)
            angle /= np.linalg.norm(new_axis)
            angle = np.arccos(angle)
            return w, -1*angle

        def map_1_key(self, ligand, old_key, new_key):
            # align new key to old key
            shift = new_key.bond(old_key)
            ligand.coord_shift(shift)
            # rotate ligand
            targets = old_key.connected - set(self.center)
            old_axis = self.COM(targets=targets) - old_key.coords
            new_axis = ligand.COM(targets=new_key.connected) - new_key.coords
            w, angle = get_rotation(old_axis, new_axis)
            ligand.rotate(w, angle, center=new_key)
            return ligand

        def map_2_key(old_ligand, ligand, old_keys, new_keys):
            # align COM of key atoms
            shift = old_ligand.COM(targets=old_keys) - \
                ligand.COM(targets=new_keys)
            ligand.coord_shift(shift)

            # rotate for best overlap
            old_axis = old_keys[0].bond(old_keys[1])
            new_axis = new_keys[0].bond(new_keys[1])
            w, angle = get_rotation(old_axis, new_axis)
            ligand.rotate(w, angle, center=ligand.COM(new_keys))

            # bend around key axis
            old_vec = old_ligand.COM(targets=old_ligand.backbone)
            old_vec -= old_ligand.COM(targets=old_keys)

            new_vec = ligand.COM(targets=ligand.backbone)
            new_vec -= old_ligand.COM(targets=old_keys)

            w, angle = get_rotation(old_vec, new_vec)
            ligand.rotate(w, angle, center=old_ligand.COM(old_keys))
            return

        def map_rot_frag(frag, a, b, ligand, old_key, new_key):
            old_vec = old_key.coords - b.coords
            new_vec = new_key.coords - b.coords
            axis, angle = get_rotation(old_vec, new_vec)
            ligand.rotate(b.bond(a), -1*angle, targets=frag, center=b.coords)

        def map_more_key(self, old_ligand, ligand, old_keys, new_keys):
            # backbone fragments separated by rotatable bonds
            frag_list = ligand.get_frag_list(ligand.backbone, max_order=1)

            # get key atoms on each side of rotatable bond
            key_count = {}
            for frag, a, b in frag_list:
                tmp = []
                for i in frag:
                    if i not in ligand.key_atoms:
                        continue
                    tmp += [i]
                try:
                    key_count[len(tmp)] += [(frag, a, b)]
                except KeyError:
                    key_count[len(tmp)] = [(frag, a, b)]

            partial_map = False
            for k in sorted(key_count.keys(), reverse=True):
                if k == 2 and not partial_map:
                    frag, a, b = key_count[k][0]
                    ok = []
                    nk = []
                    for i, n in enumerate(new_keys):
                        if n not in frag:
                            continue
                        ok += [old_keys[i]]
                        nk += [n]
                    map_2_key(old_ligand, ligand, ok, nk)
                    partial_map = True
                    continue
                if k == 1 and not partial_map:
                    frag, a, b = key_count[k][0]
                    for i, n in enumerate(new_keys):
                        if n not in frag:
                            continue
                        map_1_key(self, ligand, n, old_keys[i])
                        partial_map = True
                        break
                    continue
                if k == 1 and partial_map:
                    for frag, a, b in key_count[k]:
                        for i, n in enumerate(new_keys):
                            if n not in frag:
                                continue
                            map_rot_frag(frag, a, b, ligand, old_keys[i], n)
                            break
            return

        old_keys = self.find(old_keys)

        if not hasattr(ligands, '__iter__') or isinstance(ligands, str):
            ligands = [ligands]

        new_keys = []
        for ligand in ligands:
            if not isinstance(ligand, Component):
                ligand = Component(ligand)
            new_keys += ligand.key_atoms

        if len(old_keys) != len(new_keys):
            raise ValueError("Cannot map ligand. "
                             + "Differing number of key atoms. "
                             + "Old keys: " +
                             ",".join([i.name for i in old_keys]) + "; "
                             + "New keys: " +
                             ",".join([i.name for i in new_keys])
                             )

        old_ligands = []
        for k in old_keys:
            for c in self.components['ligand']:
                if k in c.atoms:
                    old_ligands += [c]

        start = 0
        end = None
        for i, ligand in enumerate(ligands):
            end = start + len(ligand.key_atoms)
            if len(ligand.key_atoms) == 1:
                map_1_key(self, ligand, old_keys[start], new_keys[start])
            elif len(ligand.key_atoms) == 2:
                map_2_key(old_ligands[start], ligand,
                          old_keys[start:end], new_keys[start:end])
            else:
                map_more_key(
                    self, old_ligands[start], ligand,
                    old_keys[start:end], new_keys[start:end])

            for l in ligand.atoms:
                l.name = old_keys[start].name + '.' + l.name
            start = end

        # remove old
        for ol in old_ligands:
            try:
                self.components['ligand'].remove(ol)
            except ValueError:
                continue

        # add new
        for ligand in ligands:
            self.components['ligand'] += [ligand]
        self.rebuild()

    def substitute(self, sub, target, attached_to=None):
        if not isinstance(sub, Substituent):
            sub = Substituent(sub)
        for comp in it.chain.from_iterable(self.components.values()):
            try:
                target = comp.find_exact(target)
            except LookupError:
                continue

            comp.substitute(sub, target, attached_to)
            self.rebuild()
            self.minimize_torsion(sub.atoms, self.bond(
                sub.atoms[0], sub.end), sub.end)
            break

    def minimize(self):
        """
        Rotates substituents in each component to minimize LJ_energy.
        Different from Component.minimize_sub_torsion() in that it minimizes
        with respect to the entire catalyst instead of just the component
        """
        components = self.components['ligand'] + self.components['substrate']
        for comp in components:
            if comp.substituents is None:
                comp.detect_backbone()
            targets = {}
            for sub in comp.substituents:
                if len(sub.atoms):
                    continue
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
                    self.minimize_torsion(sub.atoms, axis, center)
