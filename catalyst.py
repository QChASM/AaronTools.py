import itertools as it
from copy import deepcopy
from warnings import warn

import numpy as np

from AaronTools.component import Component
from AaronTools.const import TMETAL
from AaronTools.geometry import Geometry
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
    :conf_spec: {Substituent.start: int(rotation_number), list(skip_rots)}
    """

    def __init__(
        self,
        structure="",
        name="",
        comment="",
        components=None,
        refresh_connected=True,
    ):
        self.center = None
        self.components = components

        Geometry.__init__(self, structure, name, comment, refresh_connected)

        if isinstance(structure, str) and structure == "":
            return

        self.other = self.parse_comment()
        self.detect_components()

    def find_substituent(self, start, for_confs=True):
        """
        Finds a substituent based on a given atom (matches start==sub.atoms[0])

        :start: the first atom of the subsituent, where it connects to sub.end
        :for_confs: if true(default), only consider substituents that need to
            be rotated to generate conformers
        """
        start = self.find(start)[0]
        for sub in self.get_substituents(for_confs):
            if sub.atoms[0] == start:
                return sub
        else:
            msg = "Could not find substituent starting at atom {}"
            raise LookupError(msg.format(start.name))

    def get_substituents(self, for_confs=True):
        """
        Returns list of all substituents found on all components

        :for_confs: if true (default), returns only substituents that need to
            be rotated to generate conformers
        """
        rv = []
        if self.components is None:
            self.detect_components()
        for comp in it.chain(
            self.components["ligand"], self.components["substrate"]
        ):
            if comp.substituents is None:
                comp.detect_backbone()
            for sub in comp.substituents:
                if for_confs and (sub.conf_num is None or sub.conf_num <= 1):
                    continue
                rv += [sub]
        return rv

    def copy(self, atoms=None, name=None, comment=None):
        self.fix_comment()
        rv = super().copy()
        rv = Catalyst(rv, refresh_connected=False)
        return rv

    def rebuild(self):
        atoms = []
        for comp in sorted(self.components["substrate"]):
            comp.rebuild()
            atoms += comp.atoms
        if self.center:
            atoms += self.center
        for comp in sorted(self.components["ligand"]):
            comp.rebuild()
            atoms += comp.atoms
        self.atoms = atoms
        self.fix_comment()

    def write(self, name=None, style="xyz", *args, **kwargs):
        """
        write geometry to a file
        See fileIO.FileWriter for more details

        :name:  (str) defaults to self.name
        :style: (str) defaults to xyz
        """
        self.rebuild()
        return super().write(name, style, *args, **kwargs)

    def fix_comment(self):
        new_comment = ""
        # center
        if self.center:
            new_comment += "C:"
            for c in self.center:
                new_comment += "{},".format(self.atoms.index(c) + 1)
            else:
                new_comment = new_comment[:-1]

        # ligand and key atoms
        if self.components["ligand"]:
            # key atoms
            ids = []
            new_comment += " K:"
            subtract = 0
            for lig in self.components["ligand"]:
                subtract += len(lig.atoms)
                for a in lig.atoms:
                    ids += [self.atoms.index(a) + 1]
            subtract = len(self.atoms) - subtract
            for lig in self.components["ligand"]:
                keys = []
                for k in lig.key_atoms:
                    keys += [self.atoms.index(k) + 1]
                for k in sorted(set(keys)):
                    new_comment += "{},".format(k - subtract)
                else:
                    new_comment = new_comment[:-1] + ";"
            new_comment = new_comment[:-1]
            # ligand atoms
            new_comment += " L:"
            last_i = None
            for i in sorted(ids):
                if last_i is None:
                    new_comment += "{}-".format(i)
                elif last_i != i - 1:
                    new_comment += "{},{}-".format(last_i, i)
                    if i == ids[-1]:
                        new_comment = new_comment[:-1]
                elif i == ids[-1]:
                    new_comment += str(i)
                last_i = i

        # constrained bonds
        constrained = self.get_constraints()
        if constrained:
            new_comment += " F:"
            for cons in constrained:
                ids = [cons[0] + 1]
                ids += [cons[1] + 1]
                new_comment += "{}-{};".format(*sorted(ids))
            else:
                new_comment = new_comment[:-1]

        # save new comment (original comment still in self.other)
        self.comment = new_comment

    def detect_components(self, debug=False):
        self.components = {}
        self.center = []

        # get center
        lig_assigned = False
        center_max = None
        for a in self.atoms:
            if "ligand" not in a.tags and a.element in TMETAL.keys():
                # detect transition metal center
                if a not in self.center:
                    self.center += [a]
                a.add_tag("center")
            if "center" in a.tags:
                # center provided by comment line in xyz file
                if a not in self.center:
                    self.center += [a]
            if not lig_assigned and "ligand" in a.tags:
                lig_assigned = True
            if a in self.center:
                if center_max is None or center_max < float(a):
                    center_max = float(a)

        if not lig_assigned and len(self.center) < 1:
            msg = (
                "Non-transition metal centered catalysts must either have "
                + "active centers or ligand atoms specified in the comment "
                + "line of the XYZ input file"
            )
            raise IOError(msg)

        # label ligand and substrate
        lig = []
        subst = []
        for a in self.atoms:
            if lig_assigned:
                if "ligand" in a.tags:
                    lig += [a]
            elif float(a) > center_max:
                a.add_tag("ligand")
                lig += [a]

        for a in self.atoms:
            if "ligand" not in a.tags and "center" not in a.tags:
                a.add_tag("substrate")
                subst += [a]
        if debug:
            print("lig", [a.name for a in lig])
            print("sub", [a.name for a in subst])
            print("center", [a.name for a in self.center])

        # label key atoms:
        for i, a in enumerate(lig):
            if "key_atoms" not in self.other:
                break
            if i in self.other["key_atoms"]:
                a.add_tag("key")
        else:
            del self.other["key_atoms"]

        # get components
        if len(self.center) > 0:
            self.components["ligand"] = self.detect_fragments(lig)
            self.components["substrate"] = self.detect_fragments(subst)
        else:
            self.components["ligand"] = self.detect_fragments(lig, subst)
            self.components["substrate"] = self.detect_fragments(subst, lig)
        # rename
        for i, lig in enumerate(self.components["ligand"]):
            name = self.name + "_lig-{}".format(lig[0].name)
            self.components["ligand"][i] = Component(
                lig, name, refresh_connected=False
            )
        for i, sub in enumerate(self.components["substrate"]):
            name = self.name + "_sub-{}".format(sub[0].name)
            self.components["substrate"][i] = Component(
                sub, name, refresh_connected=False
            )
        self.rebuild()
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
                        f.add_tag("key")
            rv += [frag]
        return rv

    def map_ligand(self, ligands, old_keys, minimize=True):
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
            # occasionally there will be some round-off errors,
            # so let's fix those before we take arccos
            if angle > 1 + 10 ** -12 or angle < -1 - 10 ** -12:
                # and check to make sure we aren't covering something
                # more senister up...
                raise ValueError("Bad angle value for arccos():", angle)
            elif angle > 1:
                angle = 1.0
            elif angle < -1:
                angle = -1.0
            angle = np.arccos(angle)
            return w, -1 * angle

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
            shift = center - ligand.COM(targets=new_keys)
            ligand.coord_shift(shift)

            # bend around key axis
            old_walk = old_ligand.short_walk(*old_keys)
            if len(old_walk) == 2:
                old_con = set([])
                for k in old_keys:
                    for c in k.connected:
                        old_con.add(c)
                old_vec = old_ligand.COM(targets=old_con) - center
            else:
                old_vec = old_ligand.COM(targets=old_walk[1:-1]) - center

            new_walk = ligand.short_walk(*new_keys)
            if len(new_walk) == 2:
                new_con = set([])
                for k in new_keys:
                    for c in k.connected:
                        new_con.add(c)
                new_vec = ligand.COM(targets=new_con) - center
            else:
                new_vec = ligand.COM(targets=new_walk[1:-1]) - center

            w, angle = get_rotation(old_vec, new_vec)
            if rev_ang:
                angle = -angle
            ligand.rotate(w, angle, center=center)

            # rotate for best overlap
            old_axis = old_keys[0].bond(old_keys[1])
            new_axis = new_keys[0].bond(new_keys[1])
            w, angle = get_rotation(old_axis, new_axis)
            ligand.rotate(w, angle, center=center)

        def map_rot_frag(frag, a, b, ligand, old_key, new_key):
            old_vec = old_key.coords - b.coords
            new_vec = new_key.coords - b.coords
            axis, angle = get_rotation(old_vec, new_vec)
            ligand.rotate(b.bond(a), -1 * angle, targets=frag, center=b.coords)

            for c in new_key.connected:
                con_frag = ligand.get_fragment(new_key, c)
                if len(con_frag) > len(frag):
                    continue
                old_vec = self.COM(targets=old_key.connected)
                old_vec -= old_key.coords
                new_vec = ligand.COM(targets=new_key.connected)
                new_vec -= new_key.coords
                axis, angle = get_rotation(old_vec, new_vec)
                ligand.rotate(
                    c.bond(new_key),
                    -1 * angle,
                    targets=con_frag,
                    center=new_key.coords,
                )

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

        if isinstance(ligands, (str, Geometry)):
            ligands = [ligands]

        new_keys = []
        for i, ligand in enumerate(ligands):
            if not isinstance(ligand, Component):
                ligand = Component(ligand)
                ligands[i] = ligand
            ligand.refresh_connected()
            new_keys += ligand.key_atoms

        if len(old_keys) != len(new_keys):
            raise ValueError(
                "Cannot map ligand. "
                + "Differing number of key atoms. "
                + "Old keys: "
                + ",".join([i.name for i in old_keys])
                + "; "
                + "New keys: "
                + ",".join([i.name for i in new_keys])
            )

        old_ligands = []
        for k in old_keys:
            for c in self.components["ligand"]:
                if k in c.atoms:
                    old_ligands += [c]

        start = 0
        end = None
        for i, ligand in enumerate(ligands):
            end = start + len(ligand.key_atoms)
            if len(ligand.key_atoms) == 1:
                map_1_key(self, ligand, old_keys[start], new_keys[start])
            elif len(ligand.key_atoms) == 2:
                map_2_key(
                    old_ligands[start],
                    ligand,
                    old_keys[start:end],
                    new_keys[start:end],
                )
            else:
                map_more_key(
                    self,
                    old_ligands[start],
                    ligand,
                    old_keys[start:end],
                    new_keys[start:end],
                )

            for l in ligand.atoms:
                l.name = old_keys[start].name + "." + l.name
            start = end

        # remove old
        for ol in old_ligands:
            try:
                self.components["ligand"].remove(ol)
            except ValueError:
                continue
            for atom in self.atoms:
                if atom.connected & set(ol.atoms):
                    atom.connected = atom.connected - set(ol.atoms)

        # add new
        for ligand in ligands:
            self.components["ligand"] += [ligand]
            for sub in ligand.substituents:
                if sub.conf_num is None or sub.conf_num <= 1:
                    continue
        self.rebuild()
        self.remove_clash()
        if minimize:
            self.minimize()

    def substitute(self, sub, target, attached_to=None, minimize=True):
        if not isinstance(sub, Substituent):
            sub = Substituent(sub)
        for key, val in self.components.items():
            for comp in val:
                try:
                    target = comp.find_exact(target)
                except LookupError:
                    # keep looking
                    continue

                # found! do substitution and return
                comp.substitute(sub, target, attached_to)
                # update tags
                for a in comp.atoms:
                    a.add_tag(key)
                self.rebuild()
                self.detect_components()
                self.refresh_ranks()
                if minimize:
                    self.minimize()
                return sub

    def next_conformer(self, conf_spec, skip_spec={}):
        """
        Generates the next possible conformer

        :conf_spec: {sub_start_number: conf_number}
        :skip_spec: {sub_start_number: [skip_numbers]}


        Returns:
            conf_spec if there are still more conformers
            {} if there are no more conformers to generate
        """
        for start, conf_num in sorted(conf_spec.items()):
            sub = self.find_substituent(start)
            # skip conformer if signalled it's a repeat
            skip = skip_spec.get(start, [])
            if skip == "all" or conf_num == 0 or conf_num in skip:
                if conf_num == sub.conf_num:
                    conf_spec[start] = 1
                else:
                    conf_spec[start] += 1
                continue
            # reset conf if we hit max conf #
            if conf_num == sub.conf_num:
                sub.sub_rotate()
                conf_spec[start] = 1
                continue
            # perform rotation
            sub.sub_rotate()
            conf_spec[start] += 1
            self.remove_clash()
            # continue if the same as cf1
            angle = int(np.rad2deg((conf_spec[start] - 1) * sub.conf_angle))
            if angle != 360 and angle != 0:
                return conf_spec
            else:
                continue
        else:
            # we are done now
            return {}

    def make_conformer(self, conf_spec):
        """
        Returns:
            conf_spec, True if conformer generated (allowed by conf_spec),
            conf_spec, False if not allowed or invalid

        :conf_spec: dictionary of the form
            {sub_start_number: (conf_number, [skip_numbers])}
        """
        original = self.copy()
        for start, conf_num in conf_spec.items():
            current, skip = conf_spec[start]
            # skip if flagged a repeat
            if conf_num in skip or skip == "all":
                self = original
                return conf_spec, False
            sub = self.find_substituent(start)
            # validate conf_spec
            if conf_num > sub.conf_num:
                self = original
                warn(
                    "Bad conformer number given:",
                    sub.name,
                    conf_num,
                    ">",
                    sub.conf_num,
                )
                return conf_spec, False
            if conf_num > current:
                n_rot = conf_num - current - 1
                for _ in range(n_rot):
                    conf_spec[start][0] += 1
                    sub.rotate()
            elif conf_num < current:
                n_rot = current - conf_num - 1
                for _ in range(n_rot):
                    conf_spec[start][0] -= 1
                    sub.rotate(reverse=True)
        return conf_spec, True

    def remove_clash(self, sub_list=None):
        def get_clash(sub, scale):
            """
            Returns: np.array(bend_axis) if clash found, False otherwise
            """
            clashing = []
            for atom in self.atoms:
                if atom in sub.atoms or atom == sub.end:
                    continue
                threshold = atom._radii
                for sub_atom in sub.atoms:
                    threshold += sub_atom._radii
                    threshold *= scale
                    dist = atom.dist(sub_atom)
                    if dist < threshold or dist < 0.8:
                        clashing += [(atom, threshold - dist)]
            if not clashing:
                return False
            rot_axis = sub.atoms[0].bond(sub.end)
            vector = np.array([0, 0, 0], dtype=float)
            for a, w in clashing:
                vector += a.bond(sub.end) * w
            bend_axis = np.cross(rot_axis, vector)
            return bend_axis

        bad_subs = []  # substituents for which releif not found
        # bend_angles = [8, -16, 32, -48, 68, -88]
        # bend_back = np.deg2rad(20)
        bend_angles = [8, 8, 8, 5, 5, 5]
        bend_back = []
        rot_angles = [8, -16, 32, -48]
        rot_back = np.deg2rad(16)
        scale = 0.75  # for scaling distance threshold

        if sub_list is None:
            sub_list = sorted(self.get_substituents())
            try_twice = True
        else:
            scale = 0.65
            sub_list = sorted(sub_list, reverse=True)
            try_twice = False

        for i, b in enumerate(bend_angles):
            bend_angles[i] = -np.deg2rad(b)
        for i, r in enumerate(rot_angles):
            rot_angles[i] = np.deg2rad(r)

        for sub in sub_list:
            b, r = 0, 0  # bend_angle, rot_angle index counters
            bend_axis = get_clash(sub, scale)
            if bend_axis is False:
                continue
            else:
                # try just rotating first
                while r < len(rot_angles):
                    # try rotating
                    if r < len(rot_angles):
                        sub.sub_rotate(rot_angles[r])
                        r += 1
                    if get_clash(sub, scale) is False:
                        break
                else:
                    sub.sub_rotate(rot_back)
                    r = 0
            bend_axis = get_clash(sub, scale)
            while b < len(bend_angles) and bend_axis is not False:
                bend_back += [bend_axis]
                # try bending
                if b < len(bend_angles):
                    sub.rotate(bend_axis, bend_angles[b], center=sub.end)
                    b += 1
                bend_axis = get_clash(sub, scale)
                if bend_axis is False:
                    break
                while r < len(rot_angles):
                    # try rotating
                    if r < len(rot_angles):
                        sub.sub_rotate(rot_angles[r])
                        r += 1
                    if get_clash(sub, scale) is False:
                        break
                else:
                    sub.sub_rotate(rot_back)
                    r = 0
            else:
                # bend back to original if cannot automatically remove
                # the clash, add to bad_sub list
                bend_axis = get_clash(sub, scale)
                if bend_axis is False:
                    break
                for bend_axis in bend_back:
                    sub.rotate(bend_axis, -bend_angles[0], center=sub.end)
                bad_subs += [sub]

        # try a second time just in case other subs moved out of the way enough
        # for the first subs encountered to work now
        if try_twice and len(bad_subs) > 0:
            bad_subs = self.remove_clash(bad_subs)
        return bad_subs

    def minimize(self):
        """
        Rotates substituents in each component to minimize LJ_energy.
        Different from Component.minimize_sub_torsion() in that it minimizes
        with respect to the entire catalyst instead of just the component
        """
        targets = {}
        for sub in self.get_substituents(for_confs=True):
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
