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
        :name: str
        :comment: str
        :atoms: [Atom]
        :other: dict
        :center: [Atom] - the metal center or active center
        :components: {'ligand': [Component], 'substrate': [Component]}
        :conf_num: int - the conformer number
        :conf_spec: {Substituent.end: int(rotation_number), list(skip_rots)}
    """

    def __init__(self, structure='', name='', comment='', conf_num=1):
        self.center = None
        self.components = None
        self.conf_num = conf_num
        self.conf_spec = {}

        Geometry.__init__(self, structure, name, comment)

        if isinstance(structure, str) and structure == '':
            return

        self.other = self.parse_comment()
        self.detect_components()
        for sub in self.get_substituents():
            # self.conf_spec[sub.end] holds:
            # (current rotation number, [rotation numbers to skip] or 'all')
            self.conf_spec[sub.end] = [1, []]

    def find_substituent(self, end, for_confs=True):
        """
        Finds a substituent based on a given atom (matches end == sub.end)

        :end: the atom the substituent is connected to
        :for_confs: if true(default), only consider substituents that need to
            be rotated to generate conformers
        """
        end = self.find(end)[0]
        for sub in self.get_substituents(for_confs):
            if sub.end == end:
                return sub
        else:
            msg = "Could not find substituent connected to {}"
            raise LookupError(msg.format(end.name))

    def get_substituents(self, for_confs=True):
        """
        Returns list of all substituents found on all components

        :for_confs: if true (default), returns only substituents that need to
            be rotated to generate conformers
        """
        rv = []
        for comp in it.chain(self.components['ligand'],
                             self.components['substrate']):
            for sub in comp.substituents:
                if for_confs and (sub.conf_num is None or sub.conf_num <= 1):
                    continue
                rv += [sub]
        return rv

    def copy(self, atoms=None, name=None, comment=None):
        if atoms is None:
            atoms = deepcopy(self.atoms)

        if name is None:
            name = self.name + "_copy"
        elif name == '':
            name = self.name

        if comment is None:
            comment = deepcopy(self.comment)

        return Catalyst(atoms, name, comment)

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
        self.fix_comment()

    def fix_comment(self):
        new_comment = ''

        # center
        if self.center:
            new_comment += 'C:'
            for c in self.center:
                new_comment += "{},".format(self.atoms.index(c))
            else:
                new_comment = new_comment[:-1]

        # ligand and key atoms
        if self.components['ligand']:
            ids = []
            new_comment += ' K:'
            for lig in self.components['ligand']:
                lowest_id = None
                for l in lig.atoms:
                    ids += [self.atoms.index(l) + 1]
                    if lowest_id is None or lowest_id > ids[-1]:
                        lowest_id = ids[-1]
                keys = []
                for k in lig.key_atoms:
                    keys += [self.atoms.index(k) + 1]
                for k in sorted(keys):
                    new_comment += '{},'.format(k - lowest_id + 1)
                else:
                    new_comment = new_comment[:-1] + ';'
            new_comment = new_comment[:-1]

            new_comment += ' L:'
            last_i = None
            for i in sorted(ids):
                if last_i is None:
                    new_comment += '{}-'.format(i)
                elif last_i != i - 1:
                    new_comment += '{},{}-'.format(last_i, i)
                    if i == ids[-1]:
                        new_comment = new_comment[:-1]
                elif i == ids[-1]:
                    new_comment += str(i)
                last_i = i

        # constrained bonds
        constrained = set([])
        for a in self.atoms:
            if a.constraint:
                for c in a.constraint:
                    constrained.add(tuple(sorted([a, c])))
        if constrained:
            new_comment += ' F:'
            for cons in constrained:
                ids = [self.atoms.index(cons[0]) + 1]
                ids += [self.atoms.index(cons[1]) + 1]
                new_comment += '{}-{};'.format(*sorted(ids))
            else:
                new_comment = new_comment[:-1]

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
            msg = "Non-transition metal centered catalysts must either have " \
                + "active centers or ligand atoms specified in the comment " \
                + "line of the XYZ input file"
            raise IOError(msg)

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

        # label key atoms:
        for i, a in enumerate(lig):
            if 'key_atoms' not in self.other:
                break
            if i in self.other['key_atoms']:
                a.add_tag('key')
        else:
            del self.other['key_atoms']

        # get components
        if len(self.center) > 0:
            self.components['ligand'] = self.detect_fragments(lig)
            self.components['substrate'] = self.detect_fragments(subst)
        else:
            self.components['ligand'] = self.detect_fragments(lig, subst)
            self.components['substrate'] = self.detect_fragments(subst, lig)
        # rename
        for i, lig in enumerate(self.components['ligand']):
            name = self.name + '_lig-{}'.format(lig[0].name)
            self.components['ligand'][i] = Component(lig, name)
        for i, sub in enumerate(self.components['substrate']):
            name = self.name + '_sub-{}'.format(sub[0].name)
            self.components['substrate'][i] = Component(sub, name)

        self.refresh_connected()
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
            for f in frag:
                found.add(f)
                for c in self.center:
                    if f in c.connected:
                        f.add_tag('key')
            rv += [frag]
        return rv

    def map_ligand(self, ligands, old_keys):
        """
        Maps new ligand according to key_map
        Parameters:
        :ligand:    the name of a ligand in the ligand library
        :old_keys:  the key atoms of the old ligand to map to
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

        def map_2_key(old_ligand, ligand, old_keys, new_keys, rev_ang=False):
            # align COM of key atoms
            center = old_ligand.COM(targets=old_keys)
            shift = old_ligand.COM(targets=old_keys) - \
                ligand.COM(targets=new_keys)
            ligand.coord_shift(shift)

            # rotate for best overlap
            old_axis = old_keys[0].bond(old_keys[1])
            new_axis = new_keys[0].bond(new_keys[1])
            w, angle = get_rotation(old_axis, new_axis)
            ligand.rotate(w, angle, center=center)

            # bend around key axis
            old_con = set([])
            for k in old_keys:
                for c in k.connected:
                    old_con.add(c)
            old_vec = old_ligand.COM(targets=old_con) - center

            new_con = set([])
            for k in new_keys:
                for c in k.connected:
                    new_con.add(c)
            new_vec = ligand.COM(targets=new_con) - center

            w, angle = get_rotation(old_vec, new_vec)
            if rev_ang:
                angle = -angle
            ligand.rotate(old_axis, -angle, center=center)

        def map_rot_frag(frag, a, b, ligand, old_key, new_key):
            old_vec = old_key.coords - b.coords
            new_vec = new_key.coords - b.coords
            axis, angle = get_rotation(old_vec, new_vec)
            ligand.rotate(b.bond(a), -1*angle, targets=frag, center=b.coords)

            for c in new_key.connected:
                con_frag = ligand.get_fragment(new_key, c)
                if len(con_frag) > len(frag):
                    continue
                old_vec = self.COM(targets=old_key.connected)
                old_vec -= old_key.coords
                new_vec = ligand.COM(targets=new_key.connected)
                new_vec -= new_key.coords
                axis, angle = get_rotation(old_vec, new_vec)
                ligand.rotate(c.bond(new_key), -1*angle,
                              targets=con_frag, center=new_key.coords)

        def map_more_key(self, old_ligand, ligand, old_keys, new_keys):
            # backbone fragments separated by rotatable bonds
            frag_list = ligand.get_frag_list(ligand.backbone, max_order=1)

            # get key atoms on each side of rotatable bonds
            key_count = {}
            for frag, a, b in frag_list:
                tmp = []
                for i in frag:
                    if i not in ligand.key_atoms:
                        continue
                    tmp += [i]
                if len(tmp) not in key_count:
                    key_count[len(tmp)] = [(frag, a, b)]
                else:
                    key_count[len(tmp)] += [(frag, a, b)]

            partial_map = False
            mapped_frags = []
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
                    map_2_key(old_ligand, ligand, ok, nk, rev_ang=True)
                    partial_map = True
                    mapped_frags += [frag]
                    continue
                if k == 1 and not partial_map:
                    frag, a, b = key_count[k][0]
                    for i, n in enumerate(new_keys):
                        if n not in frag:
                            continue
                        map_1_key(self, ligand, n, old_keys[i])
                        partial_map = True
                        mapped_frags += [frag]
                        break
                    continue
                if k == 1 and partial_map:
                    for frag, a, b in key_count[k]:
                        for i, n in enumerate(new_keys):
                            if n not in frag:
                                continue
                            map_rot_frag(frag, a, b, ligand, old_keys[i], n)
                            mapped_frags += [frag]
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
                             + "Old keys: "
                             + ",".join([i.name for i in old_keys]) + "; "
                             + "New keys: "
                             + ",".join([i.name for i in new_keys])
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
                ligand.write('lig1')
                map_2_key(old_ligands[start], ligand,
                          old_keys[start:end], new_keys[start:end])
                ligand.write('lig2')
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

    def next_conformer(self):
        for end, (conf_num, skip) in sorted(self.conf_spec.items()):
            sub = self.find_substituent(end)
            if skip == 'all' or conf_num in skip:
                if conf_num == sub.conf_num:
                    self.conf_spec[end][0] = 1
                else:
                    self.conf_spec[end][0] += 1
                sub.sub_rotate()
                continue
            if conf_num == sub.conf_num:
                self.conf_spec[end][0] = 1
                sub.sub_rotate()
                continue
            self.conf_spec[end][0] += 1
            sub.sub_rotate()
            return True
        else:
            return False

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
