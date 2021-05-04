"""For storing, manipulating, and measuring molecular structures"""
import itertools
import re
import ssl
from collections import deque
from copy import deepcopy

import numpy as np
from scipy.spatial import distance_matrix

import AaronTools
import AaronTools.utils.utils as utils
from AaronTools import addlogger
from AaronTools.atoms import Atom
from AaronTools.config import Config
from AaronTools.const import BONDI_RADII, D_CUTOFF, ELEMENTS, TMETAL, VDW_RADII
from AaronTools.fileIO import FileReader, FileWriter
from AaronTools.finders import Finder
from AaronTools.utils.prime_numbers import Primes

COORD_THRESHOLD = 0.2
CACTUS_HOST = "https://cactus.nci.nih.gov"
OPSIN_HOST = "https://opsin.ch.cam.ac.uk"

DEFAULT_CONFIG = Config(quiet=True)

if not DEFAULT_CONFIG["DEFAULT"].getboolean("local_only"):
    import urllib.parse
    from urllib.error import HTTPError
    from urllib.request import urlopen


@addlogger
class Geometry:
    """
    Attributes:
        name
        comment
        atoms
        other
        _iter_idx
    """

    # AaronTools.addlogger decorator will add logger to this class attribute
    LOG = None
    # decorator uses this to set log level (defaults to WARNING if None)
    # LOGLEVEL = "INFO"
    # add to this dict to override log level for specific functions
    # keys are log level, values are lists of function names
    # LOGLEVEL_OVERRIDE = {"DEBUG": "find"}

    Primes()

    def __init__(
        self,
        structure="",
        name="",
        comment="",
        components=None,
        refresh_connected=True,
        refresh_ranks=True,
    ):
        """
        :structure: can be a Geometry(), a FileReader(), a file name, or a
            list of atoms
        :name: str
        :comment: str
        :components: list of AaronTools.component.Component() or None
        :refresh_connected: usually True - determine connectivity
        :refresh_ranks: usually True - rank atoms, only False when loading from database
        """
        super().__setattr__("_hashed", False)
        self.name = name
        self.comment = comment
        self.atoms = []
        self.center = None
        self.components = components
        self.other = {}
        self._iter_idx = None
        self._sigmat = None
        self._epsmat = None

        if isinstance(structure, Geometry):
            # new from geometry
            self.atoms = structure.atoms
            if not name:
                self.name = structure.name
            if not comment:
                self.comment = structure.comment
            return
        elif isinstance(structure, FileReader):
            # get info from FileReader object
            from_file = structure
        elif isinstance(structure, str) and structure:
            # parse file
            from_file = FileReader(structure)
        elif hasattr(structure, "__iter__") and structure:
            for a in structure:
                if not isinstance(a, Atom):
                    raise TypeError
            else:
                # list of atoms supplied
                self.atoms = structure
                if refresh_connected:
                    # SEQCROW sometimes uses refresh_connected=False to keep
                    # the connectivity the same as what's on screen
                    self.refresh_connected()
                if refresh_ranks:
                    self.refresh_ranks()
                return
        else:
            return

        # only get here if we were given a file reader object or a file name
        self.name = from_file.name
        self.comment = from_file.comment
        self.atoms = from_file.atoms
        self.other = self.parse_comment()
        if refresh_connected:
            # some file types contain connectivity info (e.g. sd) - might not want
            # to overwrite that
            self.refresh_connected()
        if refresh_ranks:
            self.refresh_ranks()
        return

    # class methods
    @classmethod
    def iupac2smiles(cls, name):
        if DEFAULT_CONFIG["DEFAULT"].getboolean("local_only"):
            raise PermissionError(
                "Converting IUPAC to SMILES failed. External network lookup disallowed."
            )
        # opsin seems to be better at iupac names with radicals
        url_smi = "{}/opsin/{}.smi".format(
            OPSIN_HOST, urllib.parse.quote(name)
        )

        try:
            smiles = (
                urlopen(url_smi, context=ssl.SSLContext())
                .read()
                .decode("utf8")
            )
        except HTTPError:
            raise RuntimeError(
                "%s is not a valid IUPAC name or https://opsin.ch.cam.ac.uk is down"
                % name
            )
        return smiles

    @classmethod
    def from_string(cls, name, form="smiles", strict_use_rdkit=False):
        """get Geometry from string
        form=iupac -> iupac to smiles from opsin API
                       --> form=smiles
        form=smiles -> structure from cactvs API/RDKit
        """

        def get_cactus_sd(smiles):
            if DEFAULT_CONFIG["DEFAULT"].getboolean("local_only"):
                raise PermissionError(
                    "Cannot retrieve structure from {}. External network lookup disallowed.".format(
                        CACTUS_HOST
                    )
                )
            url_sd = "{}/cgi-bin/translate.tcl?smiles={}&format=sdf&astyle=kekule&dim=3D&file=".format(
                CACTUS_HOST, urllib.parse.quote(smiles)
            )
            s_sd_get = urlopen(url_sd, context=ssl.SSLContext())
            msg, status = s_sd_get.msg, s_sd_get.status
            if msg != "OK":
                cls.LOG.error(
                    "Issue contacting %s for SMILES lookup (status: %s)",
                    CACTUS_HOST,
                    status,
                )
                raise IOError
            s_sd_get = s_sd_get.read().decode("utf8")
            try:
                tmp_url = re.search(
                    'User-defined exchange format file: <a href="(.*)"',
                    s_sd_get,
                ).group(1)
            except AttributeError as err:
                if re.search("You entered an invalid SMILES", s_sd_get):
                    cls.LOG.error(
                        "Invalid SMILES encountered: %s (consult %s for syntax help)",
                        smiles,
                        "https://cactus.nci.nih.gov/translate/smiles.html",
                    )
                    exit(1)
                raise IOError(err)
            new_url = "{}{}".format(CACTUS_HOST, tmp_url)
            s_sd = (
                urlopen(new_url, context=ssl.SSLContext())
                .read()
                .decode("utf8")
            )
            return s_sd

        if DEFAULT_CONFIG["DEFAULT"].getboolean("local_only"):
            strict_use_rdkit = True
        accepted_forms = ["iupac", "smiles"]
        if form not in accepted_forms:
            raise NotImplementedError(
                "cannot create substituent given %s; use one of %s" % form,
                str(accepted_forms),
            )

        if form == "smiles":
            smiles = name
        elif form == "iupac":
            smiles = cls.iupac2smiles(name)

        try:
            import rdkit.Chem.AllChem as rdk

            m = rdk.MolFromSmiles(smiles)
            if m is None and not strict_use_rdkit:
                s_sd = get_cactus_sd(smiles)
            elif m:
                mh = rdk.AddHs(m)
                rdk.EmbedMolecule(mh, randomSeed=0x421C52)
                s_sd = rdk.MolToMolBlock(mh)
            else:
                raise RuntimeError(
                    "Could not load {} with RDKit".format(smiles)
                )
        except ImportError:
            s_sd = get_cactus_sd(smiles)

        try:
            f = FileReader((name, "sd", s_sd))
            is_sdf = True
        except ValueError:
            # for some reason, CACTUS is giving xyz files instead of sdf...
            is_sdf = False
            try:
                f = FileReader((name, "xyz", s_sd))
            except ValueError:
                cls.LOG.error("Error loading geometry:\n %s", s_sd)
                raise

        return cls(f, refresh_connected=not is_sdf)

    @classmethod
    def get_coordination_complexes(
        cls,
        center,
        ligands,
        shape,
        c2_symmetric=None,
        minimize=False,
        session=None,
    ):
        """
        get all unique coordination complexes
        uses templates from Inorg. Chem. 2018, 57, 17, 10557–10567

        center - str, element of center atom
        ligands - list of ligand names in the ligand library
        shape - str, e.g. octahedral - see Atom.get_shape
        c2_symmetric - list of bools, specifies which of the bidentate ligands are C2-symmetric
                       if this list is as long as the ligands list, the nth item corresponds
                       to the nth ligand
                       otherwise, the nth item indicate the symmetry of the nth bidentate ligand
        minimize - bool, use minimize=True when mapping ligands (see Geometry.map_ligand)

        returns a list of cls containing all unique coordination complexes and the
        general formula of the complexes
        """
        import os.path
        from AaronTools.atoms import BondOrder
        from AaronTools.const import AARONTOOLS
        from AaronTools.component import Component
        
        if c2_symmetric is None:
            c2_symmetric = []
            for lig in ligands:
                comp = Component(lig)
                if not len(comp.key_atoms) == 2:
                    c2_symmetric.append(False)
                    continue
                c2_symmetric.append(comp.c2_symmetric())
        
        bo = BondOrder()
        
        # create a geometry with the specified shape
        # change the elements from dummy atoms to something else
        start_atoms = Atom.get_shape(shape)
        n_coord = len(start_atoms) - 1
        start_atoms[0].element = center
        start_atoms[0].reset()
        for atom in start_atoms[1:]:
            start_atoms[0].connected.add(atom)
            atom.connected.add(start_atoms[0])
            atom.element = "B"
            atom.reset()
        geom = cls(start_atoms, refresh_connected=False)

        # we'll need to determine the formula of the requested complex
        # monodentate ligands are a, b, etc
        # symmetric bidentate are AA, BB, etc
        # asymmetric bidentate are AB, CD, etc
        # ligands are sorted monodentate, followed by symmetric bidentate, followed by
        # asymmetric bidentate, then by decreasing count
        # e.g., Ca(CO)2(ACN)4 is Ma4b2
        alphabet = "abcdefghi"
        symmbet = ["AA", "BB", "CC", "DD"]
        asymmbet = ["AB", "CD", "EF", "GH"]
        monodentate_names = []
        symm_bidentate_names = []
        asymm_bidentate_names = []
        
        n_bidentate = 0
        # determine types of ligands
        for i, lig in enumerate(ligands):
            comp = Component(lig)
            if len(comp.key_atoms) == 1:
                monodentate_names.append(lig)
            elif len(comp.key_atoms) == 2:
                if len(ligands) == len(c2_symmetric):
                    c2 = c2_symmetric[i]
                else:
                    c2 = c2_symmetric[n_bidentate]
                n_bidentate += 1
                if c2:
                    symm_bidentate_names.append(lig)
                else:
                    asymm_bidentate_names.append(lig)
            else:
                # tridentate or something
                raise NotImplementedError(
                    "can only attach mono- and bidentate ligands: %s (%i)" % (
                        lig, len(comp.key_atoms)
                    )
                )
        
        coord_num = (
            len(monodentate_names) + 
            2 * (
                len(symm_bidentate_names) + len(asymm_bidentate_names)
            )
        )
        if coord_num != n_coord:
            raise RuntimeError(
                "coordination number (%i) does not match sum of ligand denticity (%i)" % (
                    n_coord, coord_num
                )
            )
        
        # start putting formula together
        cc_type = "M"
        this_name = center
        # sorted by name count is insufficient when there's multiple monodentate ligands
        # with the same count (e.g. Ma3b3)
        # add the index in the library to offset this

        monodentate_names = sorted(
            monodentate_names,
            key=lambda x: 10000 * monodentate_names.count(x) + Component.list().index(x),
            reverse=True,
        )
        for i, mono_lig in enumerate(
            sorted(
                set(monodentate_names),
                key=lambda x: 10000 * monodentate_names.count(x) + Component.list().index(x),
                reverse=True,
            )
        ):
            cc_type += alphabet[i]
            this_name += "(%s)" % mono_lig
            if monodentate_names.count(mono_lig) > 1:
                cc_type += "%i" % monodentate_names.count(mono_lig)
                this_name += "%i" % monodentate_names.count(mono_lig)

        symm_bidentate_names = sorted(
            symm_bidentate_names,
            key=lambda x: 10000 * symm_bidentate_names.count(x) + Component.list().index(x),
            reverse=True
        )
        for i, symbi_lig in enumerate(sorted(
                set(symm_bidentate_names),
                key=lambda x: 10000 * symm_bidentate_names.count(x) + Component.list().index(x),
                reverse=True
            )
        ):
            cc_type += "(%s)" % symmbet[i]
            this_name += "(%s)" % symbi_lig
            if symm_bidentate_names.count(symbi_lig) > 1:
                cc_type += "%i" % symm_bidentate_names.count(symbi_lig)
                this_name += "%i" % symm_bidentate_names.count(symbi_lig)
        asymm_bidentate_names = sorted(
            asymm_bidentate_names,
            key=lambda x: 10000 * asymm_bidentate_names.count(x) + Component.list().index(x),
            reverse=True
        )
        for i, asymbi_lig in enumerate(
            sorted(
                set(asymm_bidentate_names),
                key=lambda x: 10000 * asymm_bidentate_names.count(x) + Component.list().index(x),
                reverse=True
            )
        ):
            cc_type += "(%s)" % asymmbet[i]
            this_name += "(%s)" % asymbi_lig
            if asymm_bidentate_names.count(asymbi_lig) > 1:
                cc_type += "%i" % asymm_bidentate_names.count(asymbi_lig)
                this_name += "%i" % asymm_bidentate_names.count(asymbi_lig)

        # load the key atoms for ligand mapping from the template file
        libdir = os.path.join(
            AARONTOOLS,
            "coordination_complex",
            shape,
            cc_type
        )
        if not os.path.exists(libdir):
            raise RuntimeError("no templates for %s %s" % (cc_type, shape))

        geoms = []
        for f in os.listdir(libdir):
            mappings = np.loadtxt(os.path.join(libdir, f), dtype=str, delimiter=",", ndmin=2)
            
            point_group, subset = f.rstrip(".csv").split("_")            
            # for each possible structure, create a copy of the original template shape
            # attach ligands in the order they would appear in the formula
            for i, mapping in enumerate(mappings):
                geom_copy = geom.copy()
                geom_copy.center = [geom_copy.atoms[0]]
                geom_copy.components = [Component([atom]) for atom in geom_copy.atoms[1:]]
            
                start = 0
                for lig in monodentate_names:
                    key = mapping[start]
                    start += 1
                    comp = Component(lig)
                    d = 2.5
                    # adjust distance to key atoms to what they should be for the new ligand
                    try:
                        d = bo.bonds[bo.key(center, comp.key_atoms[0])]["1.0"]
                    except KeyError:
                        pass
                    geom_copy.change_distance(
                        geom_copy.atoms[0],
                        key,
                        dist=d,
                        fix=1
                    )
                    # attach ligand
                    geom_copy.map_ligand(comp, key, minimize=minimize)
                    for key in comp.key_atoms:
                        geom_copy.atoms[0].connected.add(key)
                        key.connected.add(geom_copy.atoms[0])

                for lig in symm_bidentate_names:
                    keys = mapping[start:start + 2]
                    start += 2
                    comp = Component(lig)
                    for old_key, new_key in zip(keys, comp.key_atoms):
                        d = 2.5
                        try:
                            d = bo.bonds[bo.key(center, new_key)]["1.0"]
                        except KeyError:
                            pass
                        geom_copy.change_distance(
                            geom_copy.atoms[0],
                            old_key,
                            dist=d,
                            fix=1,
                            as_group=False
                        )
                    geom_copy.map_ligand(comp, keys, minimize=minimize)
                    for key in comp.key_atoms:
                        geom_copy.atoms[0].connected.add(key)
                        key.connected.add(geom_copy.atoms[0])

                for lig in asymm_bidentate_names:
                    keys = mapping[start:start + 2]
                    start += 2
                    comp = Component(lig)
                    for old_key, new_key in zip(keys, comp.key_atoms):
                        d = 2.5
                        try:
                            d = bo.bonds[bo.key(center, new_key)]["1.0"]
                        except KeyError:
                            pass
                        geom_copy.change_distance(
                            geom_copy.atoms[0],
                            old_key,
                            dist=d,
                            fix=1,
                            as_group=False
                        )
                    geom_copy.map_ligand(comp, keys, minimize=minimize)
                    for key in comp.key_atoms:
                        geom_copy.atoms[0].connected.add(key)
                        key.connected.add(geom_copy.atoms[0])

                geom_copy.name = "%s-%i_%s_%s" % (this_name, i + 1, point_group, subset)
                geoms.append(geom_copy)
        
        return geoms, cc_type

    @staticmethod
    def weighted_percent_buried_volume(geometries, energies, temperature, *args, **kwargs):
        """
        Boltzmann-averaged percent buried volume
        geometries - list of Geometry instances
        energies - numpy array, energy in kcal/mol; ith energy corresponds to ith substituent
        temperature - temperature in K
        *args, **kwargs - passed to Geometry.percent_buried_volume()
        """
        values = []
        
        for geom in geometries:
            values.append(geom.percent_buried_volume(*args, **kwargs))
        
        rv = utils.boltzmann_average(
            energies,
            np.array(values),
            temperature,
        )
        
        return rv
        
    # attribute access
    def _stack_coords(self, atoms=None):
        """
        Generates a N x 3 coordinate matrix for atoms
        Note: the matrix rows are copies of, not references to, the
            Atom.coords objects. Run Geometry.update_geometry(matrix) after
            using this method to save changes.
        """
        if atoms is None:
            atoms = self.atoms
        else:
            atoms = self.find(atoms)
        rv = np.zeros((len(atoms), 3), dtype=float)
        for i, a in enumerate(atoms):
            rv[i] = a.coords[:]
        return rv

    @property
    def elements(self):
        """ returns list of elements composing the atoms in the geometry """
        return np.array([a.element for a in self.atoms])

    @property
    def coords(self):
        """
        array of coordinates (read only)
        """
        return self.coordinates()

    def coordinates(self, atoms=None):
        """
        returns N x 3 coordinate matrix for requested atoms
            (defaults to all atoms)
        """
        if atoms is None:
            return self._stack_coords()
        return self._stack_coords(atoms)

    # utilities
    def __str__(self):
        s = ""
        for a in self:
            s += a.__str__() + "\n"
        return s

    def __repr__(self):
        """ string representation """
        s = ""
        for a in self:
            s += a.__repr__() + "\n"
        return s

    def __eq__(self, other):
        """
        two geometries equal if:
            same number of atoms
            same numbers of elements
            coordinates of atoms similar
        """
        if id(self) == id(other):
            return True
        if len(self.atoms) != len(other.atoms):
            return False

        self_eles = [atom.element for atom in self.atoms]
        other_eles = [atom.element for atom in other.atoms]
        self_counts = {ele: self_eles.count(ele) for ele in set(self_eles)}
        other_counts = {ele: other_eles.count(ele) for ele in set(other_eles)}
        if self_counts != other_counts:
            return False

        rmsd = self.RMSD(other, sort=False)
        return rmsd < COORD_THRESHOLD

    def __add__(self, other):
        if isinstance(other, Atom):
            other = [other]
        elif not isinstance(other, list):
            other = other.atoms
        self.atoms += other
        return self

    def __sub__(self, other):
        if isinstance(other, Atom):
            other = [other]
        elif not isinstance(other, list):
            other = other.atoms
        for o in other:
            self.atoms.remove(o)
        for a in self.atoms:
            if a.connected & set(other):
                a.connected = a.connected - set(other)
        return self

    def __iter__(self):
        self._iter_idx = -1
        return self

    def __next__(self):
        if self._iter_idx + 1 < len(self.atoms):
            self._iter_idx += 1
            return self.atoms[self._iter_idx]
        raise StopIteration

    def __len__(self):
        return len(self.atoms)

    def __setattr__(self, attr, val):
        if attr == "_hashed" and not val:
            raise RuntimeError("can only set %s to True" % attr)

        if not self._hashed or (self._hashed and attr != "atoms"):
            super().__setattr__(attr, val)
        else:
            raise RuntimeError(
                "cannot change atoms attribute of HashableGeometry"
            )

    def __hash__(self):
        # hash depends on atom elements, connectivity, order, and coordinates
        # reorient using principle axes
        coords = self.coords
        coords -= self.COM()
        mat = np.matmul(coords.T, coords)
        vals = np.linalg.svd(mat, compute_uv=False)

        t = [int(v * 3) for v in vals]
        for atom, coord in zip(self.atoms, coords):
            # only use the first 3 decimal places of coordinates b/c numerical issues
            t.append(int(atom.get_neighbor_id()))
            if not atom._hashed:
                atom.connected = frozenset(atom.connected)
                atom.coords.setflags(write=False)
                atom._hashed = True
            # make sure atoms don't move
            # if atoms move, te hash value could change making it impossible to access
            # items in a dictionary with this instance as the key

        if not self._hashed:
            self.LOG.warning(
                "Geometry `%s` has been hashed and will no longer be editable.\n"
                "Use Geometry.copy to get an editable duplicate of this instance",
                self.name,
            )
            self.atoms = tuple(self.atoms)
            self._hashed = True

        return hash(tuple(t))

    def tag(self, tag, targets=None):
        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)
        for atom in targets:
            atom.tags.add(tag)

    def write(self, name=None, *args, **kwargs):
        """
        Write geometry to a file

        :name: defaults to self.name
        :style: defaults to xyz

        *args and **kwargs for styles:
            xyz
                :append: True/False
            com
                :step:
                :theory: the Theory specification object
                :**kwargs: Additional route arguments in key=val format
        """
        tmp = self.name
        if name is not None:
            self.name = name
        out = FileWriter.write_file(self, *args, **kwargs)
        self.name = tmp
        if out is not None:
            return out

    def copy(self, atoms=None, name=None, comment=None, copy_atoms=True):
        """
        creates a new copy of the geometry
        parameters:
            atoms (list): defaults to all atoms
            name (str): defaults to NAME_copy
        """
        if name is None:
            name = self.name
        if comment is None:
            comment = self.comment
        atoms = self._fix_connectivity(atoms)
        if hasattr(self, "components") and self.components is not None:
            self.fix_comment()
        if copy_atoms:
            return Geometry([a.copy() for a in atoms], name, comment)
        return Geometry(atoms, name, comment)

    def parse_comment(self):
        """
        Saves auxillary data found in comment line
        """
        if not self.comment:
            return {}
        rv = {}
        # constraints
        match = re.search("F:([0-9;-]+)", self.comment)
        if match is not None:
            rv["constraint"] = []
            for a in self.atoms:
                a.constraint = set([])
            match = match.group(1).split(";")
            for m in match:
                if m == "":
                    continue
                m = m.split("-")
                m = [int(i) for i in m]
                if len(m) == 2:
                    for i, j in zip(m[:-1], m[1:]):
                        a = self.find(str(i))[0]
                        b = self.find(str(j))[0]
                        a.constraint.add((b, a.dist(b)))
                        b.constraint.add((a, b.dist(a)))
                rv["constraint"] += [m]
        # active centers
        match = re.search("C:([0-9,]+)", self.comment)
        if match is not None:
            rv["center"] = []
            match = match.group(1).split(",")
            for m in match:
                if m == "":
                    continue
                a = self.atoms[int(m) - 1]
                a.add_tag("center")
                rv["center"] += [a]
        # ligand
        match = re.search("L:([0-9;,-]+)", self.comment)
        if match is not None:
            rv["ligand"] = []
            match = match.group(1).split(";")
            for submatch in match:
                tmp = []
                for m in submatch.split(","):
                    if m == "":
                        continue
                    if "-" not in m:
                        a = self.atoms[int(m) - 1]
                        tmp += [a]
                        continue
                    m = m.split("-")
                    for i in range(int(m[0]) - 1, int(m[1])):
                        try:
                            a = self.atoms[i]
                        except IndexError:
                            continue
                        tmp += [a]
                rv["ligand"] += [tmp]
        # key atoms
        match = re.search("K:([0-9,;]+)", self.comment)
        if match is not None:
            rv["key_atoms"] = []
            match = match.group(1).split(";")
            for m in match:
                if m == "":
                    continue
                m = m.split(",")
                for i in m:
                    if i == "":
                        continue
                    rv["key_atoms"] += [int(i) - 1]
        self.other = rv
        return rv

    def fix_comment(self):
        if not hasattr(self, "components"):
            return
        elif self.components is None:
            self.detect_components()
        new_comment = ""
        # center
        if self.center:
            new_comment += "C:"
            for c in self.center:
                new_comment += "{},".format(self.atoms.index(c) + 1)
            else:
                new_comment = new_comment[:-1]

        # key atoms
        new_comment += " K:"
        for frag in sorted(self.components):
            tmp = ""
            for key in sorted(frag.key_atoms):
                tmp += "{},".format(self.atoms.index(key) + 1)
            if tmp:
                new_comment += tmp[:-1] + ";"
        if new_comment[-3:] == " K:":
            new_comment = new_comment[:-3]
        else:
            new_comment = new_comment[:-1]

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

        # components
        if self.components:
            new_comment += " L:"
            for lig in sorted(self.components):
                ids = sorted([1 + self.atoms.index(a) for a in lig])
                tmp = []
                for i in ids:
                    if i == ids[0]:
                        tmp = [i]
                        continue
                    if i == tmp[-1] + 1:
                        tmp += [i]
                    elif len(tmp) == 1:
                        new_comment += "{},".format(tmp[0])
                        tmp = [i]
                    else:
                        new_comment += "{}-{},".format(tmp[0], tmp[-1])
                        tmp = [i]
                if len(tmp) == 1:
                    new_comment += "{},".format(tmp[0])
                else:
                    new_comment += "{}-{},".format(tmp[0], tmp[-1])
                new_comment = new_comment[:-1] + ";"
            new_comment = new_comment[:-1]

        # save new comment (original comment still in self.other)
        self.comment = new_comment

    def _flag(self, flag, targets=None):
        """
        freezes targets if <flag> is True,
        relaxes targets if <flag> is False
        """
        if isinstance(targets, Config):
            if targets._changed_list is not None:
                targets = targets._changed_list
            else:
                raise RuntimeError(
                    "Substitutions/Mappings requested, but not performed"
                )
        if targets is not None:
            targets = self.find(targets)
        else:
            targets = self.atoms
        for a in targets:
            a.flag = flag
        return

    def freeze(self, targets=None):
        """
        freezes atoms in the geometry
        """
        self._flag(True, targets)

    def relax(self, targets=None):
        """
        relaxes atoms in the geometry
        """
        self._flag(False, targets)

    def get_constraints(self, as_index=True):
        rv = {}
        for i, a in enumerate(self.atoms[:-1]):
            if not a.constraint:
                continue
            for j, b in enumerate(self.atoms[i:]):
                for atom, dist in a.constraint:
                    if b == atom:
                        if as_index:
                            rv[(i, i + j)] = dist
                        else:
                            rv[(a, b)] = dist
                        break
        return rv

    def get_connectivity(self):
        rv = []
        for atom in self.atoms:
            rv += [atom.connected]
        return rv

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

    def detect_substituents(self):
        """sets self.substituents to a list of substituents"""
        from AaronTools.substituent import Substituent

        # TODO: allow detection of specific substituents only
        #       -check fragment length and elements against
        #        that of the specified substituent
        # copy-pasted from Component.detect_backbone, but
        # removed stuff that refers to the center/backbone

        if not hasattr(self, "substituents") or self.substituents is None:
            self.substituents = []

        frag_list = self.get_frag_list()

        new_tags = {}  # hold atom tag options until assignment determined
        subs_found = {}  # for testing which sub assignment is best
        sub_atoms = set([])  # holds atoms assigned to substituents
        for frag_tup in sorted(frag_list, key=lambda x: len(x[0])):
            frag, start, end = frag_tup
            if frag[0] != start:
                frag = self.reorder(start=start, targets=frag)[0]

            # try to find fragment in substituent library
            try:
                sub = Substituent(frag, end=end)
            except LookupError:
                continue

            # substituents with more than half of self's atoms are ignored
            if len(frag) > len(self.atoms) - len(frag):
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
                    sub_length += [subs_found[t]]
                max_length = max(sub_length)
                if max_length < 0:
                    max_length = min(sub_length)
                keep = sub_length.index(max_length)
                a.add_tag(tags[keep])
            else:
                a.add_tag(tags[0])

    def find(self, *args, debug=False):
        """
        finds atom in geometry
        Parameters:
            *args are tags, names, elements, or a Finder subclass
            args=(['this', 'that'], 'other') will find atoms for which
                ('this' || 'that') && 'other' == True
        Returns:
            [Atom()] or []
        Raises:
            LookupError when the tags/names provided do not exist
            However, it will return empty list if valid tag/names were provided
                but were screened out using the && argument form
        """

        def _find(arg):
            """ find a single atom """
            # print(arg)
            if isinstance(arg, Atom):
                # print("atom")
                return [arg]

            rv = []
            if isinstance(arg, Finder):
                # print("finder")
                rv += arg.get_matching_atoms(self.atoms, self)

            name_str = re.compile("^(\*|\d)+(\.?\*|\.\d+)*$")
            if isinstance(arg, str) and name_str.match(arg) is not None:
                # print("name")
                test_name = arg.replace(".", "\.")
                test_name = test_name.replace("*", "(\.?\d+\.?)*")
                test_name = re.compile("^" + test_name + "$")
                # this is a name
                for a in self.atoms:
                    if test_name.search(a.name) is not None:
                        rv += [a]

            elif isinstance(arg, str) and len(arg.split(",")) > 1:
                # print("comma list")
                list_style = arg.split(",")
                if len(list_style) > 1:
                    for i in list_style:
                        if len(i.split("-")) > 1:
                            rv += _find_between(i)
                        else:
                            rv += _find(i)

            elif (
                isinstance(arg, str)
                and len(arg.split("-")) > 1
                and not re.search("[A-Za-z]", arg)
            ):
                # print("range list")
                rv += _find_between(arg)

            elif isinstance(arg, str) and arg in ELEMENTS:
                # print("element")
                # this is an element
                for a in self.atoms:
                    if a.element == arg:
                        rv += [a]
            else:
                # print("tag")
                # this is a tag
                for a in self.atoms:
                    if arg in a.tags:
                        rv += [a]
            return rv

        def _find_between(arg):
            """ find sequence of atoms """

            def _name2ints(name):
                name = name.split(".")
                return [int(i) for i in name]

            a1, a2 = tuple(arg.split("-"))
            a1 = _find(a1)[0]
            a2 = _find(a2)[0]

            rv = []
            for a in self.atoms:
                # keep if a.name is between a1.name and a2.name
                test_name = _name2ints(a.name)
                a1_name = _name2ints(a1.name)
                a2_name = _name2ints(a2.name)

                for tn, a1n, a2n in zip(test_name, a1_name, a2_name):
                    if tn < a1n:
                        # don't want if test atom sorts before a1
                        break
                    if tn > a2n:
                        # don't want if test atom sorts after a2
                        break
                else:
                    rv += _find(a)
            return rv

        if len(args) == 1:
            if isinstance(args[0], tuple):
                args = args[0]
        rv = []
        for a in args:
            if hasattr(a, "__iter__") and not isinstance(a, str):
                # argument is a list of sub-arguments
                # OR condition
                tmp = []
                for i in a:
                    tmp += _find(i)
                rv += [tmp]
            else:
                rv += [_find(a)]

        # error if no atoms found (no error if AND filters out all found atoms)
        if len(rv) == 1:
            if len(rv[0]) == 0:
                raise LookupError(
                    "Could not find atom: %s on\n%s\n%s"
                    % ("; ".join([str(x) for x in args]), self.name, str(self))
                )
            return rv[0]

        # exclude atoms not fulfilling AND requirement
        tmp = []
        for i in rv[0]:
            good = True
            for j in rv[1:]:
                if i not in j:
                    good = False
            if good:
                tmp += [i]
        return tmp

    def find_exact(self, *args):
        """
        finds exactly the same number of atoms as arguments used.
        Raises LookupError if wrong number of atoms found
        """
        rv = []
        err = "Wrong number of atoms found: "
        is_err = False
        for arg in args:
            try:
                a = self.find(arg)
            except LookupError:
                a = []

            if len(a) != 1:
                is_err = True
                err += "{} (found {}), ".format(arg, len(a))
            else:
                rv += a

        if is_err:
            err = err[:-2]
            raise LookupError(err)
        return tuple(rv)

    def _fix_connectivity(self, atoms=None, copy=True):
        """
        for fixing the connectivity for a set of atoms when grabbing
        a fragment or copying atoms, ensures atom references are sane

        :atoms: the atoms to fix connectivity for; connections to atoms
            outside of this list are severed in the resulting list
        :copy: perform a deepcopy of the atom list
        """
        if atoms is None:
            atoms = self.atoms
        else:
            atoms = self.find(atoms)

        connectivity = []
        for a in atoms:
            connectivity += [
                [atoms.index(i) for i in a.connected if i in atoms]
            ]
        if copy:
            atoms = [a.copy() for a in atoms]
        for a, con in zip(atoms, connectivity):
            a.connected = set([])
            for c in con:
                a.connected.add(atoms[c])

        return atoms

    def refresh_connected(self, targets=None, threshold=None):
        """
        reset connected atoms
        atoms are connected if their distance from each other is less than
            the sum of their covalent radii plus a threshold
        """
        # clear current connectivity
        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)
        old_connectivity = []
        for a in targets:
            old_connectivity += [a.connected]
            a.connected = set([])

        D = distance_matrix(self.coordinates(targets), self.coordinates(targets))

        # determine connectivity
        for i, a in enumerate(targets):
            for j, b in enumerate(targets[:i]):
                if a.dist_is_connected(b, D[i, j], threshold):
                    a.connected.add(b)
                    b.connected.add(a)

    def refresh_ranks(self, invariant=True):
        rank = self.canonical_rank(invariant=invariant)
        for a, r in zip(self.atoms, rank):
            a._rank = r
        return

    def canonical_rank(
        self, heavy_only=False, break_ties=True, update=True, invariant=True
    ):
        """
        determine canonical ranking for atoms
        invariant: bool - if True, use invariant described in J. Chem. Inf. Comput. Sci. 1989, 29, 2, 97–101
                          (DOI: 10.1021/ci00062a008)
                          if False, use neighbor IDs

        algorithm described in J. Chem. Inf. Model. 2015, 55, 10, 2111–2120 
        (DOI: 10.1021/acs.jcim.5b00543)
        """
        CITATION = "doi:10.1021/ci00062a008"
        self.LOG.citation(CITATION)
        CITATION = "doi:10.1021/acs.jcim.5b00543"
        self.LOG.citation(CITATION)

        primes = Primes.list(len(self.atoms))
        atoms = []
        ranks = []

        # using the catalyst's center can make it difficult
        # to compare C2 symmetric ligands
        # center = list(filter(lambda x: "center" in x.tags, self))
        # if center:
        #     center = self.COM(targets=center)
        # else:
        center = self.COM()

        def neighbors_rank(ranks):
            # partitions key is product of rank and neighbors' rank
            # use prime numbers for product so products are distinct
            # eg: primes[2]*primes[2] != primes[1]*primes[4]
            partitions = {}
            for i, a in enumerate(atoms):
                key = primes[ranks[i]]
                for b in a.connected:
                    if b in atoms:
                        key *= primes[ranks[atoms.index(b)]]
                partitions.setdefault(ranks[i], {})
                partitions[ranks[i]].setdefault(key, [])
                partitions[ranks[i]][key] += [i]
            return update_ranks(ranks, partitions)

        def update_ranks(ranks, partitions):
            new_ranks = ranks.copy()
            for rank, key_dict in partitions.items():
                if len(key_dict) == 1:
                    continue
                for key in sorted(key_dict.keys()):
                    for idx in key_dict[key]:
                        new_ranks[idx] = rank
                    rank += len(key_dict[key])
            return new_ranks

        def tie_break(ranks):
            """
            Uses atom angles around COM -> shared_atom axis to break ties[
            Does not break symmetry (eg: pentane carbons still [0, 2, 4, 2, 0]
            because C2 and C4 are ~180 deg apart relative to COM-C5 axis)
            """

            def get_angle(vi, vj, norm):
                dot = np.dot(vi, vj)
                cross = np.cross(vi, vj)
                det = np.dot(norm, cross)
                rv = np.arctan2(det, dot)
                return round(rv, 1)

            def get_start(connected, center, norm):
                # if we can, use the COM of tied atoms as reference 0-deg
                start = self.COM(targets=[atoms[c] for c in connected])
                start -= center
                if np.linalg.norm(np.cross(start, norm)) > 1e-2:
                    return start
                # if there's one atom that is closest/farthest to center,
                # use that as start
                start_min = None, None
                start_max = None, None
                for c in connected:
                    dist = np.linalg.norm(atoms[c].coords - center)
                    if start_min[0] is None or dist < start_min[1]:
                        start_min = [c], dist
                    elif dist == start_min[1]:
                        start_min = start_min[0] + [c], dist
                    if start_max[0] is None or dist < start_max[1]:
                        start_max = [c], dist
                    elif dist == start_max[1]:
                        start_max = start_max[0] + [c], dist
                if len(start_min[0]) == 1:
                    start = atoms[start_min[0][0]].coords - center
                    return start
                if len(start_max[0]) == 1:
                    start = atoms[start_max[0][0]].coords - center
                    return start
                # otherwise, try to use COM of equally close/far atoms
                if len(start_min[0]) < len(connected):
                    start = self.COM(targets=[atoms[c] for c in start_min[0]])
                    start -= center
                    if np.linalg.norm(np.cross(start, norm)) > 1e-2:
                        return start
                if len(start_max[0]) < len(connected):
                    start = self.COM(targets=[atoms[c] for c in start_max[0]])
                    start -= center
                    if np.linalg.norm(np.cross(start, norm)) > 1e-2:
                        return start
                # if all else fails, just use the first atom I guess...
                return atoms[connected[0]].coords - center

            partitions = {}
            for i, rank in enumerate(ranks):
                partitions.setdefault(rank, {})
                partitions[rank].setdefault(rank, [])
                partitions[rank][rank] += [i]

            new_partitions = partitions.copy()

            # norm = self.get_principle_axes()
            # norm = norm[1][:, 0] - center
            for rank, rank_dict in partitions.items():
                idx_list = rank_dict[rank]
                if len(idx_list) == 1:
                    continue
                # split ties into groups connected to same atom
                groups = {}
                for i in idx_list[:-1]:
                    a = atoms[i]
                    for j in idx_list[1:]:
                        b = atoms[j]
                        connected = a.connected & b.connected
                        if len(connected) == 1:
                            k = connected.pop()
                            if k in atoms:
                                k = atoms.index(k)
                            else:
                                continue
                            groups.setdefault(k, set([i]))
                            groups[k] |= set([j])
                # atoms in each group sorted in counter clockwise order
                # around axis centered at shared atom and orthogonal to COM
                for shared_idx, connected in groups.items():
                    connected = sorted(connected)
                    start = atoms[shared_idx].coords - center
                    norm = np.cross(start, center)
                    angles = {}
                    for c in connected:
                        this = atoms[c].coords - center
                        angle = get_angle(start, this, norm)
                        angles.setdefault(angle, [])
                        angles[angle] += [c]
                    if len(angles) == 1 and atoms[shared_idx].connected - set(
                        [atoms[c] for c in connected]
                    ):
                        tmp_center = self.COM(
                            atoms[shared_idx].connected
                            - set([atoms[c] for c in connected])
                        )
                        start = atoms[shared_idx].coords - tmp_center
                        norm = np.cross(start, tmp_center)
                        angles = {}
                        for c in connected:
                            this = atoms[c].coords - tmp_center
                            angle = get_angle(start, this, norm)
                            angles.setdefault(angle, [])
                            angles[angle] += [c]
                    for i, angle in enumerate(sorted(angles.keys())):
                        new_partitions[rank].setdefault(rank + i, [])
                        new_partitions[rank][rank + i] += angles[angle]
                        for idx in angles[angle]:
                            if idx in new_partitions[rank][rank]:
                                new_partitions[rank][rank].remove(idx)
            return update_ranks(ranks, new_partitions)

        # rank all atoms the same initially
        for a in self.atoms:
            if heavy_only and a.element == "H":
                continue
            atoms += [a]
            ranks += [0]

        # partition and re-rank using invariants
        partitions = {}
        for i, a in enumerate(atoms):
            if invariant:
                id = a.get_invariant()
            else:
                id = a.get_neighbor_id()
            partitions.setdefault(id, [])
            partitions[id] += [i]
        new_rank = 0
        for key in sorted(partitions.keys()):
            idx_list = partitions[key]
            for idx in idx_list:
                ranks[idx] = new_rank
            new_rank += len(idx_list)

        # re-rank using neighbors until no change
        for i in range(500):
            new_ranks = neighbors_rank(ranks)
            if ranks == new_ranks:
                break
            ranks = new_ranks
        else:
            self.LOG.warning(
                "Max cycles reached in canonical sorting (neighbor-ranks)"
            )

        # break ties using spatial positions
        # AND update neighbors until no change
        if break_ties:
            for i in range(500):
                new_ranks = tie_break(ranks)
                new_ranks = neighbors_rank(new_ranks)
                if ranks == new_ranks:
                    break
                ranks = new_ranks
            else:
                self.LOG.warning(
                    "Max cycles reached in canonical sorting (tie-breaking)"
                )

        return ranks

    def reorder(
        self,
        start=None,
        targets=None,
        heavy_only=False,
    ):
        """
        Returns:
            list(ordered_targets), list(non_targets)

        Depth-first reorder of atoms based on canonical ranking
        """

        if not targets:
            targets = self.atoms
        else:
            targets = self.find(targets)
        if heavy_only:
            targets = [t for t in targets if t.element != "H"]
        non_targets = [a for a in self.atoms if a not in targets]

        # get starting atom
        if not start:
            order = [sorted(targets)[0]]
        else:
            order = sorted(self.find(start))
        start = sorted(order)
        stack = []
        for s in start:
            stack += sorted(s.connected)
        atoms_left = set(targets) - set(order) - set(stack)
        while len(stack) > 0:
            this = stack.pop()
            if heavy_only and this.element == "H":
                continue
            if this in order:
                continue
            order += [this]
            connected = this.connected & atoms_left
            atoms_left -= set(connected)
            stack += sorted(connected)

            if not stack and atoms_left:
                stack += [sorted(atoms_left)[0]]
                atoms_left -= set(stack)

        return order, non_targets

    def rebuild(self):
        atoms = []
        if self.components:
            if self.center:
                atoms += self.center
            for comp in sorted(self.components):
                comp.rebuild()
                atoms += comp.atoms
            self.atoms = atoms
        self.fix_comment()
        self.refresh_ranks()

    def detect_components(self, center=None):
        from AaronTools.component import Component

        self.components = []
        if center is None:
            self.center = []
        else:
            self.center = self.find(center)

        # get center
        if not self.center:
            for a in self.atoms:
                if a.element in TMETAL.keys():
                    # detect transition metal center
                    if a not in self.center:
                        self.center += [a]
                    a.add_tag("center")
                if "center" in a.tags:
                    # center provided by comment line in xyz file
                    if a not in self.center:
                        self.center += [a]

        # label key atoms:
        for i, a in enumerate(self.atoms):
            if "key_atoms" not in self.other:
                break
            if i in self.other["key_atoms"]:
                a.add_tag("key")
        else:
            del self.other["key_atoms"]

        # get components
        self.components = self.detect_fragments(self.atoms)
        # rename
        for i, frag in enumerate(self.components):
            name = self.name + ".{:g}".format(
                min(
                    [
                        float(a.name)
                        if utils.is_num(a.name)
                        else frag.index(a)
                        for a in frag
                    ]
                )
            )
            self.components[i] = Component(frag, name)
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

        def add_tags(frag):
            for f in frag:
                found.add(f)
                for c in self.center:
                    if f in c.connected:
                        f.add_tag("key")

        if avoid is None and self.center:
            avoid = self.center
        found = set([])
        rv = []

        if "ligand" in self.other:
            for ligand in self.other["ligand"]:
                frag = set(self.find(ligand))
                frag -= found
                add_tags(frag)
                rv += [sorted(frag)]
                found.union(frag)

        for a in targets:
            if a in found:
                continue
            if avoid:
                if a in avoid:
                    continue
                frag = set(self.get_fragment(a, avoid))
                frag -= found
                add_tags(frag)
                rv += [sorted(frag)]
                found.union(frag)
            else:
                frag = set([a])
                queue = a.connected.copy()
                while queue:
                    this = queue.pop()
                    if this in frag:
                        continue
                    frag.add(this)
                    queue = queue.union(this.connected)
                frag -= found
                add_tags(frag)
                rv += [sorted(frag)]
                found = found.union(frag)
        return rv

    def shortest_path(self, atom1, atom2, avoid=None):
        """
        Uses Dijkstra's algorithm to find shortest path between atom1 and atom2
        avoid: atoms to avoid on the path
        """
        a1 = self.find(atom1)[0]
        a2 = self.find(atom2)[0]
        if avoid is None:
            path = utils.shortest_path(self, a1, a2)
        else:
            avoid = self.find(avoid)
            graph = [
                [
                    self.atoms.index(j)
                    for j in i.connected
                    if j in self.atoms and j not in avoid
                ]
                if i not in avoid
                else []
                for i in self.atoms
            ]
            path = utils.shortest_path(
                graph, self.atoms.index(a1), self.atoms.index(a2)
            )
        if not path:
            raise LookupError(
                "could not determine best path between {} and {}".format(
                    atom1, atom2
                )
            )
        return [self.atoms[i] for i in path]

    # geometry measurement
    def bond(self, a1, a2):
        """ takes two atoms and returns the bond vector """
        a1, a2 = self.find_exact(a1, a2)
        return a1.bond(a2)

    def angle(self, a1, a2, a3=None):
        """returns a1-a2-a3 angle"""
        a1, a2, a3 = self.find_exact(a1, a2, a3)
        return a2.angle(a1, a3)

    def dihedral(self, a1, a2, a3, a4):
        """measures dihedral angle of a1 and a4 with respect to a2-a3 bond"""
        a1, a2, a3, a4 = self.find_exact(a1, a2, a3, a4)

        b12 = a1.bond(a2)
        b23 = a2.bond(a3)
        b34 = a3.bond(a4)

        dihedral = np.cross(np.cross(b12, b23), np.cross(b23, b34))
        dihedral = np.dot(dihedral, b23) / np.linalg.norm(b23)
        dihedral = np.arctan2(
            dihedral, np.dot(np.cross(b12, b23), np.cross(b23, b34))
        )

        return dihedral

    def COM(self, targets=None, heavy_only=False, mass_weight=True):
        """
        calculates center of mass of the target atoms
        returns a vector from the origin to the center of mass
        parameters:
            targets (list) - the atoms to use in calculation (defaults to all)
            heavy_only (bool) - exclude hydrogens (defaults to False)
        """
        # get targets
        if targets:
            targets = self.find(targets)
        else:
            targets = list(self.atoms)
        # screen hydrogens if necessary
        if heavy_only:
            targets = [a for a in targets if a.element != "H"]

        coords = self.coordinates(targets)
        if mass_weight:
            total_mass = 0
            for i in range(0, len(coords)):
                coords[i] *= targets[i].mass()
                total_mass += targets[i].mass()

        # COM = (1/M) * sum(m * r) = sum(m*r) / sum(m)
        center = np.mean(coords, axis=0)

        if mass_weight and total_mass:
            return center * len(targets) / total_mass
        return center

    def RMSD(
        self,
        ref,
        align=False,
        heavy_only=False,
        sort=False,
        targets=None,
        ref_targets=None,
        debug=False,
        weights=None,
        ref_weights=None,
    ):
        """
        calculates the RMSD between two geometries
        Returns: rmsd (float)

        :ref: (Geometry) the geometry to compare to
        :align: (bool) if True (default), align self to other;
            if False, just calculate the RMSD
        :heavy_only: (bool) only use heavy atoms (default False)
        :targets: (list) the atoms in `self` to use in calculation
        :ref_targets: (list) the atoms in the reference geometry to use
        :sort: (bool) canonical sorting of atoms before comparing
        :debug: returns RMSD and Geometry([ref_targets]), Geometry([targets])
        :weights: (list(float)) weights to apply to targets
        :ref_weights: (list(float)) weights to apply to ref_targets
        """

        def _RMSD(ref, other):
            """
            ref and other are lists of atoms
            returns rmsd, angle, vector
                rmsd (float)
                angle (float) angle to rotate by
                vector (np.array(float)) the rotation axis
            """
            matrix = np.zeros((4, 4), dtype=np.float64)
            for i, a in enumerate(ref):
                pt1 = a.coords
                try:
                    pt2 = other[i].coords
                except IndexError:
                    break
                matrix += utils.quat_matrix(pt2, pt1)

            eigenval, eigenvec = np.linalg.eigh(matrix)
            val = eigenval[0]
            vec = eigenvec.T[0]

            if val > 0:
                # val is the SD
                rmsd = np.sqrt(val / len(ref))
            else:
                # negative numbers that are basically zero, like -1e-16
                rmsd = 0

            # sometimes it freaks out if the coordinates are right on
            # top of each other and gives overly large rmsd/rotation
            # I think this is a numpy precision problem, may want to
            # try scipy.linalg to see if that helps?
            tmp = sum(
                [
                    np.linalg.norm(a.coords - b.coords) ** 2
                    for a, b in zip(ref, other)
                ]
            )
            tmp = np.sqrt(tmp / len(ref))
            if tmp < rmsd:
                rmsd = tmp
                vec = np.array([0, 0, 0])
            return rmsd, vec

        # get target atoms
        tmp = targets
        if targets is not None:
            targets = self.find(targets)
        else:
            targets = self.atoms
        if ref_targets is not None:
            ref_targets = ref.find(ref_targets)
        elif tmp is not None:
            ref_targets = ref.find(tmp)
        else:
            ref_targets = ref.atoms

        # screen out hydrogens if heavy_only requested
        if heavy_only:
            targets = [a for a in targets if a.element != "H"]
            ref_targets = [a for a in ref_targets if a.element != "H"]

        this = Geometry([t.copy() for t in targets])
        ref = Geometry([r.copy() for r in ref_targets])
        if weights is not None:
            for w, a in zip(weights, this.atoms):
                a.coords *= w

        if ref_weights is not None:
            for w, a in zip(ref_weights, ref.atoms):
                a.coords *= w

        # align center of mass to origin
        com = this.COM()
        ref_com = ref.COM()

        this.coord_shift(-com)
        ref.coord_shift(-ref_com)

        # try current ordering
        min_rmsd = _RMSD(ref.atoms, this.atoms)
        # try canonical ordering
        if sort:
            this.atoms = this.reorder()[0]
            ref.atoms = ref.reorder()[0]
            this_ranks = [a._rank for a in this.atoms]
            ref_ranks = [a._rank for a in ref.atoms]
            if any(this_ranks.count(r) > 1 for r in this_ranks) or any(
                ref_ranks.count(r) > 1 for r in ref_ranks
            ):
                # if there are atoms with the same rank, align both ref and this
                # to their principle axes and use the distance between the atoms
                # in the structures to determine the order
                this_atoms = []
                ref_atoms = [a for a in ref.atoms]
                _, this_axes = this.get_principle_axes()
                _, ref_axes = ref.get_principle_axes()
                this_coords = this.coords - this.COM()
                ref_coords = ref.coords - ref.COM()
                # align this to ref using the matrix that aligns this's principle axes
                # to ref's principle axes
                H = np.dot(ref_coords.T, this_coords)
                u, s, vh = np.linalg.svd(H, compute_uv=True)
                d = 1.0
                if np.linalg.det(np.matmul(vh.T, u.T)) < 0:
                    d = 1.0
                m = np.diag([1.0, 1.0, d])
                R = np.matmul(vh.T, m)
                R = np.matmul(R, u.T)

                this_coords = np.dot(this_coords, R)

                # find the closest atom in this to the closest atom in ref
                dist = distance_matrix(ref_coords, this_coords)
                for i, r_a in enumerate(ref.atoms):
                    min_dist = None
                    match = None
                    for j, t_a in enumerate(this.atoms):
                        if t_a.element != r_a.element:
                            continue
                        if t_a in this_atoms:
                            continue

                        if min_dist is None or dist[i, j] < min_dist:
                            min_dist = dist[i, j]
                            match = t_a
                    if match is not None:
                        this_atoms.append(match)
                    else:
                        ref_atoms.remove(r_a)

                # if we didn't find any matches or not all atoms are matched,
                # use the original order
                # otherwise, use the order determined by distances
                if len(ref_atoms) == len(this_atoms) and ref_atoms:
                    res = _RMSD(ref_atoms, this_atoms)
                else:
                    res = _RMSD(ref.atoms, this.atoms)

            else:
                res = _RMSD(ref.atoms, this.atoms)

            if res[0] < min_rmsd[0]:
                min_rmsd = res

        rmsd, vec = min_rmsd

        # return rmsd
        if not align:
            if debug:
                return this, ref, rmsd
            else:
                return rmsd
        # or update geometry and return rmsd
        self.coord_shift(-com)
        if np.linalg.norm(vec) > 0:
            self.rotate(vec)
        self.coord_shift(ref_com)
        if debug:
            return this, ref, rmsd
        else:
            return rmsd

    def get_near(self, ref, dist, by_bond=False, include_ref=False):
        """
        Returns: list of atoms within a distance or number of bonds of a
            reference point, line, plane, atom, or list of atoms

        :ref: the point (eg: [0, 0, 0]), line (eg: ['*', 0, 0]), plane
            (eg: ['*', '*', 0]), atom, or list of atoms
        :dist: the distance threshold or number of bonds away threshold, is an
            inclusive upper bound (uses `this <= dist`)
        :by_bond: if true, `dist` is interpreted as the number of bonds away
            instead of distance in angstroms
            NOTE: by_bond=True means that ref must be an atom or list of atoms
        :include_ref: if Atom or list(Atom) given as ref, include these in the
            returned list, (default=False, do not include ref in returned list)
        """
        if dist < 0:
            raise ValueError(
                "Distance or number of bonds threshold must be positive"
            )
        if not hasattr(ref, "iter") and isinstance(ref, Atom):
            ref = [ref]
        rv = []

        # find atoms within number of bonds away
        if by_bond:
            dist_err = "by_bond=True only applicable for integer bonds away"
            ref_err = (
                "by_bond=True only applicable for ref of type Atom() or "
                "list(Atom())"
            )
            if int(dist) != dist:
                raise ValueError(dist_err)
            for r in ref:
                if not isinstance(r, Atom):
                    raise TypeError(ref_err)
            stack = set(ref)
            rv = set([])
            while dist > 0:
                dist -= 1
                new_stack = set([])
                for s in stack:
                    rv = rv.union(s.connected)
                    new_stack = new_stack.union(s.connected)
                stack = new_stack
            if not include_ref:
                rv = rv - set(ref)
            return sorted(rv)

        # find atoms within distance
        if isinstance(ref, Atom):
            ref = [ref.coords]
        elif isinstance(ref, list):
            new_ref = []
            just_nums = []
            for r in ref:
                if isinstance(r, Atom):
                    new_ref += [r.coords]
                elif isinstance(r, list):
                    new_ref += [r]
                else:
                    just_nums += [r]
            if len(just_nums) % 3 != 0:
                raise ValueError(
                    "coordinates (or wildcards) must be passed in sets of "
                    "three: [x, y, z]"
                )
            else:
                while len(just_nums) > 0:
                    new_ref += [just_nums[-3:]]
                    just_nums = just_nums[:-3]
        mask = [False, False, False]
        for r in new_ref:
            for i, x in enumerate(r):
                if x == "*":
                    mask[i] = True
                    r[i] = 0
            for a in self.atoms:
                coords = a.coords.copy()
                for i, x in enumerate(mask):
                    if x:
                        coords[i] = 0
                if np.linalg.norm(np.array(r, dtype=float) - coords) <= dist:
                    rv += [a]
        if not include_ref:
            for r in ref:
                if isinstance(r, Atom) and r in rv:
                    rv.remove(r)
        return rv

    def get_principle_axes(self, mass_weight=True, center=None):
        """
        Return: [principal moments], [principle axes]
        """
        if center is None:
            COM = self.COM(mass_weight=mass_weight)
        else:
            COM = center
        I_CM = np.zeros((3, 3))
        for a in self:
            if mass_weight:
                mass = a.mass()
            else:
                mass = 1
            coords = a.coords - COM
            I_CM[0, 0] += mass * (coords[1] ** 2 + coords[2] ** 2)
            I_CM[1, 1] += mass * (coords[0] ** 2 + coords[2] ** 2)
            I_CM[2, 2] += mass * (coords[0] ** 2 + coords[1] ** 2)
            I_CM[0, 1] -= mass * (coords[0] * coords[1])
            I_CM[0, 2] -= mass * (coords[0] * coords[2])
            I_CM[1, 2] -= mass * (coords[1] * coords[2])
        I_CM[1, 0] = I_CM[0, 1]
        I_CM[2, 0] = I_CM[0, 2]
        I_CM[2, 1] = I_CM[1, 2]

        return np.linalg.eigh(I_CM)

    def LJ_energy(self, other=None, use_prev_params=False):
        """
        computes LJ energy using autodock parameters
        use_prev_params - use same sigma/epsilon as the last time LJ_energy was called
                          useful for methods that make repetitive LJ_energy calls, like
                          minimize_torsion
        """

        if (
            use_prev_params
            and self._sigmat is not None
            and self._sigmat.shape != (len(self), len(other))
        ):
            sigmat = self._sigmat
            epsmat = self._epsmat
        else:
            sigmat = np.array(
                [[a.rij(b) for a in self.atoms] for b in self.atoms]
            )
            epsmat = np.array(
                [[a.eij(b) for a in self.atoms] for b in self.atoms]
            )

        if other is None or other is self:
            D = distance_matrix(self.coords, self.coords)
            np.fill_diagonal(D, 1)

        else:
            if hasattr(other, "coords"):
                D = distance_matrix(self.coords, other.coords)
                other = other.atoms

                sigmat = np.array(
                    [[a.rij(b) for a in other] for b in self.atoms]
                )
                epsmat = np.array(
                    [[a.eij(b) for a in other] for b in self.atoms]
                )
            else:
                D = distance_matrix(
                    self.coords, np.array([a.coords for a in other])
                )

        self._sigmat = sigmat
        self._epsmat = epsmat

        repmat = sigmat / D
        repmat = repmat ** 2
        repmat = repmat ** 3
        attmat = repmat ** 2

        if other is None or other is self:
            nrgmat = np.tril(epsmat * (attmat - repmat), -1)
        else:
            nrgmat = epsmat * (attmat - repmat)

        return np.sum(nrgmat)

    def examine_constraints(self, thresh=None):
        """
        Determines if constrained atoms are too close/ too far apart
        Returns: (atom1, atom2, flag) where flag is 1 if atoms too close,
            -1 if atoms to far apart (so one can multiply a distance to change
            by the flag and it will adjust in the correct direction)
        """
        rv = []
        if thresh is None:
            thresh = D_CUTOFF
        constraints = self.get_constraints()
        # con of form (atom_name_1, atom_name_2, original_distance)
        for con in constraints:
            if len(con) != 2:
                continue
            dist = self.atoms[con[0]].dist(self.atoms[con[1]])
            if dist - constraints[con] > thresh:
                # current > constraint: atoms too far apart
                # want to move closer together
                rv += [(con[0], con[1], -1)]
            elif constraints[con] - dist > thresh:
                # constraint > current: atoms too close together
                # want to move farther apart
                rv += [(con[0], con[1], 1)]
        return rv

    def compare_connectivity(self, ref, thresh=None, return_idx=False):
        """
        Compares connectivity of self relative to ref
        Returns: broken, formed
            :broken: set of atom name pairs for which a bond broke
            :formed: set of atom name pairs for which a bond formed

        :ref: the structure to compare to (str(path), FileReader, or Geometry)
            ref.atoms should be in the same order as self.atoms
        :thresh: allow for connectivity changes as long as the difference
            between bond distances is below a threshold
        :by_name: if True (default) lookup atoms by name, otherwise compare
            based on index in atom list
        """
        broken = set([])
        formed = set([])
        if not isinstance(ref, Geometry):
            ref = Geometry(ref)

        not_found = set(self.atoms)
        for i, r in enumerate(ref.atoms):
            s = self.find(r.name)[0]
            not_found.remove(s)

            conn = set(self.find(i.name)[0] for i in r.connected)
            if not conn ^ s.connected:
                continue
            for c in conn - s.connected:
                if thresh is not None:
                    dist = r.dist(ref.find(c.name)[0]) - s.dist(c)
                    if abs(dist) <= thresh:
                        continue
                if return_idx:
                    broken.add(tuple(sorted([i, self.atoms.index(c)])))
                else:
                    broken.add(tuple(sorted([s.name, c.name])))
            for c in s.connected - conn:
                if thresh is not None:
                    dist = r.dist(ref.find(c.name)[0]) - s.dist(c)
                    if abs(dist) <= thresh:
                        continue
                if return_idx:
                    broken.add(tuple(sorted([i, self.atoms.index(c)])))
                else:
                    formed.add(tuple(sorted([s.name, c.name])))
        return broken, formed

    def percent_buried_volume(
        self,
        center=None,
        targets=None,
        radius=3.5,
        radii="umn",
        scale=1.17,
        exclude=None,
        method="lebedev",
        rpoints=20,
        apoints=1454,
        min_iter=25,
        basis=None,
    ):
        """
        calculates % buried volume (%V_bur) using Monte-Carlo or Gauss-Legendre/Lebedev integration
        see Organometallics 2008, 27, 12, 2679–2681 (DOI: 10.1021/om8001119) for details

        center  - center atom(s) or np.array of coordinates
                  if more than one atom is specified, the sphere will be centered on
                  the centroid between the atoms
        targets - atoms to use in calculation, defaults to all non-center if there
                  is only one center, otherwise all atoms
        radius  - sphere radius around center atom
        radii   - "umn" or "bondi", VDW radii to use
                  can also be a dict() with atom symbols as the keys and
                  their respective radii as the values
        scale   - scale VDW radii by this
        method  - integration method (MC or lebedev)
        rpoints - number of radial shells for Lebedev integration
        apoints - number of angular points for Lebedev integration
        min_iter - minimum number of iterations for MC integration
                   each iteration is a batch of 3000 points
                   iterations will continue beyond min_iter if the volume has not converged
        basis - change of basis matrix
                will cause %Vbur to be returned as a tuple for different quadrants (I, II, III, IV)
        """
        # NOTE - it would be nice to multiprocess the MC integration (or
        #        split up the shells for the Lebedev integration, but...
        #        python's multiprocessing doesn't let you spawn processes
        #        outside of the __name__ == '__main__' context

        # determine center if none was specified
        if center is None:
            if self.center is None:
                self.detect_components()
            center = self.center
            center_coords = self.COM(center)

        else:
            try:
                center = self.find(center)
                center_coords = self.COM(center)
            except LookupError:
                # assume an array was given
                center_coords = center

        # determine atoms if none were specified
        if targets is None:
            if center is None:
                targets = self.atoms
            else:
                if len(center) == 1:
                    targets = [
                        atom for atom in self.atoms if atom not in center
                    ]
                else:
                    targets = [atom for atom in self.atoms]
        else:
            targets = self.find(targets)

        # VDW radii to use
        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        else:
            raise RuntimeError(
                "received %s for radii, must be umn or bondi" % radii
            )

        # list of scaled VDW radii for each atom that's close enough to
        # the center of the sphere
        radius_list = []
        atoms_within_radius = []

        # determine which atom's radii extend within the sphere
        # reduces the number of distances we need to calculate
        # also determine innermost and outermost atom edges (minr and maxr)
        # so we can skip integration shells that don't contain atoms
        minr = radius
        maxr = 0.0
        for atom in targets:
            if exclude is not None and atom in exclude:
                continue
            d = np.linalg.norm(center_coords - atom.coords)
            inner_edge = d - scale * radii_dict[atom.element]
            outer_edge = inner_edge + 2 * scale * radii_dict[atom.element]
            if inner_edge < radius:
                atoms_within_radius.append(atom)
                if inner_edge < minr:
                    minr = inner_edge
                if outer_edge > maxr:
                    maxr = outer_edge
        maxr = min(maxr, radius)
        if minr < 0:
            minr = 0

        # sort atoms based on their distance to the center
        # this makes is so we usually break out of looping over the atoms faster
        atoms_within_radius.sort(
            key=lambda a, c=center_coords: np.linalg.norm(a.coords - c)
        )

        for atom in atoms_within_radius:
            radius_list.append(scale * radii_dict[atom.element])

        coords = self.coordinates(atoms_within_radius)

        # Monte-Carlo integration
        if method.lower() == "mc":
            if basis is None:
                prev_vol = cur_vol = 0
                buried_points = 0

            else:
                prev_vol = np.zeros(4)
                cur_vol = np.zeros(4)
                buried_points = np.zeros(4)
                tot_points = np.zeros(4)

            n_samples = 3000
            dV = []
            i = 0
            # determine %V_bur
            # do at least 75000 total points, but keep going until
            # the last 5 changes are all less than 1e-4
            while i < min_iter or not (
                all(dv < 2e-4 for dv in dV[-5:]) and np.mean(dV[-5:]) < 1e-4
            ):
                i += 1
                # get a random point uniformly distributed inside the sphere
                # only sample points between minr and maxr because maybe that makes
                # things converge faster
                r = (maxr - minr) * np.random.uniform(0, 1, n_samples) ** (
                    1 / 3
                )
                r += minr
                z = np.random.uniform(-1, 1, n_samples)

                theta = np.arcsin(z) + np.pi / 2
                phi = np.random.uniform(0, 2 * np.pi, n_samples)

                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z *= r

                xyz = np.array([x, y, z]).T
                if basis is not None:
                    # determine what quadrant this point is in, add it to the appropriate bin
                    map_xyz = np.dot(xyz, basis)
                    for p in map_xyz:
                        if p[0] > 0 and p[1] > 0:
                            tot_points[0] += 1
                        elif p[0] <= 0 and p[1] > 0:
                            tot_points[1] += 1
                        elif p[0] <= 0 and p[1] <= 0:
                            tot_points[2] += 1
                        else:
                            tot_points[3] += 1
                xyz += center_coords
                # see if the point is inside of any atom's
                # scaled VDW radius
                D = distance_matrix(xyz, coords)
                for k, d_row in enumerate(D):
                    for j, (d, r) in enumerate(zip(d_row, radius_list)):
                        if d < r:
                            if basis is None:
                                buried_points += 1
                            else:
                                # determine what quadrant the point is in
                                if map_xyz[k, 0] > 0 and map_xyz[k, 1] > 0:
                                    buried_points[0] += 1
                                elif map_xyz[k, 0] <= 0 and map_xyz[k, 1] > 0:
                                    buried_points[1] += 1
                                elif map_xyz[k, 0] <= 0 and map_xyz[k, 1] <= 0:
                                    buried_points[2] += 1
                                else:
                                    buried_points[3] += 1
                            break

                if basis is None:
                    cur_vol = float(buried_points) / (float(i * n_samples))
                    dV.append(abs(cur_vol - prev_vol))
                    prev_vol = cur_vol
                else:
                    cur_vol = np.divide(buried_points, tot_points) / 4
                    dV.append(abs(sum(cur_vol) - sum(prev_vol)))
                    prev_vol = cur_vol

            between_v = cur_vol * (maxr ** 3 - minr ** 3)
            tot_v = radius ** 3
            return 100 * between_v / tot_v

        # default to Gauss-Legendre integration over Lebedev spheres
        else:
            # grab radial grid points and weights for range (minr, maxr)
            rgrid, rweights = utils.gauss_legendre_grid(
                start=minr, stop=maxr, num=rpoints
            )
            # grab Lebedev grid for unit sphere at origin
            agrid, aweights = utils.lebedev_sphere(
                radius=1, center=np.zeros(3), num=apoints
            )

            # value of integral (without 4 pi r^2) for each shell
            if basis is not None:
                shell_values = np.zeros((4, rpoints))
            else:
                shell_values = np.zeros(rpoints)
            # loop over radial shells

            for i, rvalue in enumerate(rgrid):
                # collect non-zero weights in inside_weights, then sum after looping over shell
                if basis is not None:
                    inside_weights = np.zeros((4, apoints))
                else:
                    inside_weights = np.zeros(apoints)
                # scale grid point to radius and shift to center
                agrid_r = agrid * rvalue
                if basis is not None:
                    map_agrid_r = np.dot(agrid_r, basis)
                agrid_r += center_coords
                D = distance_matrix(agrid_r, coords)
                for j, (d_row, aweight) in enumerate(zip(D, aweights)):
                    # add weight if the point is inside of any atom's
                    # scaled VDW radius
                    for d, r in zip(d_row, radius_list):
                        if d < r:
                            if basis is not None:
                                # determine what quadrant the point is in for steric map stuff
                                if (
                                    map_agrid_r[j, 0] > 0
                                    and map_agrid_r[j, 1] > 0
                                ):
                                    inside_weights[0][j] = aweight
                                elif (
                                    map_agrid_r[j, 0] < 0
                                    and map_agrid_r[j, 1] > 0
                                ):
                                    inside_weights[1][j] = aweight
                                elif (
                                    map_agrid_r[j, 0] < 0
                                    and map_agrid_r[j, 1] < 0
                                ):
                                    inside_weights[2][j] = aweight
                                else:
                                    inside_weights[3][j] = aweight
                            else:
                                inside_weights[j] = aweight

                            break

                    # save integral over current shell (without 4 pi r^2)
                    if basis is not None:
                        for k in range(0, 4):
                            shell_values[k][i] = np.sum(inside_weights[k])
                    else:
                        shell_values[i] = np.sum(inside_weights)

            if basis is not None:
                # return a list of buried volume in each quadrant
                return [
                    300
                    * np.dot(shell_values[k] * rgrid ** 2, rweights)
                    / (radius ** 3)
                    for k in range(0, 4)
                ]
            else:
                # return buried volume
                return (
                    300
                    * np.dot(shell_values * rgrid ** 2, rweights)
                    / (radius ** 3)
                )

    def steric_map(
        self,
        center=None,
        key_atoms=None,
        radii="umn",
        radius=3.5,
        oop_vector=None,
        ip_vector=None,
        return_basis=False,
        num_pts=100,
        shape="circle",
    ):
        """
        returns x, y, z, min_alt, max_alt or x, y, z, min_alt, max_alt, basis, atoms if return_basis is True
        x - x coordinates for grid
        y - y coordinates for grid
        z - altitude levels; points where no atoms are will be -1000
        min_alt - minimum altitude (above -1000)
        max_alt - maximum altitute
        basis - basis to e.g. reorient structure with np.dot(self.coords, basis)
        atoms - list of atoms that are in the steric map
        a contour plot can be created with this data - see stericMap.py command line script

        parameters:
        center - atom, list of atoms, or array specifiying the origin
        key_atoms - list of ligand key atoms. Atoms on these ligands will be in the steric map.
        radii - "umn", "bondi", or dict() specifying the VDW radii to use
        oop_vector - None or array specifying the direction out of the plane of the steric map
                     if None, oop_vector is determined using the average vector from the key
                     atoms to the center atom
        ip_vector - None or array specifying a vector in the plane of the steric map
                    if None, ip_vector is determined as the plane of best fit through the
                    key_atoms and the center
        return_basis - whether or not to return a change of basis matrix
        num_pts - number of points along x and y axis to use
        shape - "circle" or "square"
        """

        # determine center if none was specified
        if center is None:
            if self.center is None:
                self.detect_components()
            center = self.center
            center_coords = self.COM(center)

        else:
            try:
                center = self.find(center)
                center_coords = self.COM(center)
            except LookupError:
                # assume an array was given
                center_coords = center

        # VDW radii to use
        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        else:
            raise RuntimeError(
                "received %s for radii, must be umn or bondi" % radii
            )

        if key_atoms is None:
            key_atoms = []
            if self.components is None:
                self.detect_components()

            for comp in self.components:
                key_atoms.extend(comp.key_atoms)

        else:
            key_atoms = self.find(key_atoms)

        targets = []
        for key in key_atoms:
            if key not in targets:
                if isinstance(center, Atom) or (
                    hasattr(center, "__iter__")
                    and all(isinstance(a, Atom) for a in center)
                ):
                    targets.extend(self.get_fragment(key, center))

                else:
                    targets.extend(self.get_all_connected(key))

        if oop_vector is None:
            oop_vector = np.zeros(3)
            for atom in key_atoms:
                oop_vector += center_coords - atom.coords

        oop_vector /= np.linalg.norm(oop_vector)

        if ip_vector is None:
            if len(key_atoms) == 1:
                ip_vector = utils.perp_vector(oop_vector)
                x_vec = np.cross(ip_vector, oop_vector)
            else:
                coords = [atom.coords for atom in key_atoms]
                coords.append(center_coords)
                coords = np.array(coords)
                ip_vector = utils.perp_vector(coords)
                x_vec = np.cross(ip_vector, oop_vector)
                x_vec /= np.linalg.norm(x_vec)
                ip_vector = -np.cross(x_vec, oop_vector)

        else:
            x_vec = np.cross(ip_vector, oop_vector)

        basis = np.array([x_vec, ip_vector, oop_vector]).T
        coords = self.coordinates(targets) - center_coords
        new_coords = np.dot(coords, basis)
        dist_ip = distance_matrix(new_coords[:, 0:2], [np.zeros(2)])[:, 0]
        atoms_within_radius = []
        radius_list = []
        for i, atom in enumerate(targets):
            if (
                shape == "circle"
                and dist_ip[i] - radii_dict[atom.element] < radius
            ):
                atoms_within_radius.append(atom)
                radius_list.append(radii_dict[atom.element])
            elif (
                shape == "square"
                and dist_ip[i] - radii_dict[atom.element] < np.sqrt(2) * radius
            ):
                atoms_within_radius.append(atom)
                radius_list.append(radii_dict[atom.element])

        atom_coords = np.dot(
            self.coordinates(atoms_within_radius) - center_coords, basis
        )

        x = np.linspace(-radius, radius, num=num_pts)
        y = np.linspace(-radius, radius, num=num_pts)
        z = -1000 * np.ones((num_pts, num_pts))
        max_alt = None
        min_alt = None
        for i in range(0, num_pts):
            for j in range(0, num_pts):
                if shape == "circle" and x[i] ** 2 + y[j] ** 2 > radius ** 2:
                    continue
                for k in range(0, len(atoms_within_radius)):
                    w = np.sqrt(
                        (x[i] - atom_coords[k][0]) ** 2
                        + (y[j] - atom_coords[k][1]) ** 2
                    )
                    if w < radius_list[k]:
                        h = np.sqrt(radius_list[k] ** 2 - w ** 2)
                        alt = atom_coords[k, 2] + h
                        # matplotlib mirrors z?
                        if alt > z[j, i]:
                            z[j, i] = alt
                            if max_alt is None or alt > max_alt:
                                max_alt = alt

                            if min_alt is None or alt < min_alt:
                                min_alt = alt

        if return_basis:
            return x, y, z, min_alt, max_alt, basis, atoms_within_radius
        return x, y, z, min_alt, max_alt

    def sterimol(
            self,
            L_axis,
            start_atom,
            targets,
            L_func=None,
            return_vector=False,
            radii="bondi",
    ):
        """
        returns sterimol parameter values in a dictionary
        keys are B1, B2, B3, B4, B5, and L
        see Verloop, A. and Tipker, J. (1976), Use of linear free energy
        related and other parameters in the study of fungicidal
        selectivity. Pestic. Sci., 7: 379-390.
        (DOI: 10.1002/ps.2780070410)

        return_vector: bool/returns dict of tuple(vector start, vector end) instead
        radii: "bondi" - Bondi vdW radii
               "umn"   - vdW radii from Mantina, Chamberlin, Valero, Cramer, and Truhlar

        AaronTools' definition of the L parameter is different than the original
        STERIMOL program. In STERIMOL, the van der Waals radii of the substituent is
        projected onto a plane parallel to the bond between the molecule and the substituent.
        The L parameter is 0.40 Å plus the distance from the first substituent atom to the
        outer van der Waals surface of the projection along the bond vector. This 0.40 Å is
        a correction for STERIMOL using a hydrogen to represent the molecule, when a carbon
        would be more likely. In AaronTools the substituent is projected the same, but L is
        calculated starting from the van der Waals radius of the first substituent atom
        instead. This means AaronTools will give the same L value even if the substituent
        is capped with something besides a hydrogen. When comparing AaronTools' L values
        with STERIMOL (using the same set of radii for the atoms), the values usually
        differ by < 0.1 Å.
        """
        from AaronTools.finders import BondedTo
        from scipy.spatial import ConvexHull

        CITATION = "doi:10.1002/ps.2780070410"
        self.LOG.citation(CITATION)

        targets = self.find(targets)
        start = self.find(start_atom)
        if len(start) != 1:
            raise TypeError(
                "start must be exactly 1 atom, %i found for %s" % (
                    len(start),
                    repr(start_atom),
                )
            )
        start = start[0]

        L_axis /= np.linalg.norm(L_axis)

        if not L_func:
            def L_func(atom, start, radius, L_axis):
                test_v = start.bond(atom)
                test_L = np.dot(test_v, L_axis) + radius
                vec = (start.coords, start.coords + test_L * L_axis)
                return test_L, vec

        if isinstance(radii, dict):
            radii_dict = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII

        B1 = None
        B2 = None
        B3 = None
        B4 = None
        B5 = None
        L = None
        vector = {
            "B1": None,
            "B2": None,
            "B3": None,
            "B4": None,
            "B5": None,
            "L": None
        }
        # for B1, we're going to use ConvexHull to find the minimum distance
        # from one face of a bounding box
        # to do this, we're going to project the substituent in a plane
        # perpendicular to the L-axis and get a set of points along the
        # vdw radii of the atoms
        # ConvexHull will take these points and figure out which ones
        # are on the outside (vertices)
        # we then just need to find the bounding box with the minimum distance
        # from L-axis to one side of the box
        points = []
        ndx = []
        # just grab a random vector perpendicular to the L-axis
        # it doesn't matter really
        ip_vector = utils.perp_vector(L_axis)
        x_vec = np.cross(ip_vector, L_axis)
        x_vec /= np.linalg.norm(x_vec)
        basis = np.array([x_vec, ip_vector, L_axis]).T

        num_pts = 360
        b1_points = np.array([
                [np.cos(x), np.sin(x)] for x in np.linspace(
                0,
                2 * np.pi,
                num=num_pts,
                )
            ]
        )
        std_ndx = np.ones(num_pts, dtype=int)

        for i, atom in enumerate(targets):
            test_v = start.bond(atom)

            # L
            test_L, L_vec = L_func(
                atom,
                start,
                radii_dict[atom.element],
                L_axis
            )

            if L is None or test_L > L:
                L = test_L
                vector["L"] = L_vec

            # B1-4 stuff - we come back to this later
            r1 = radii_dict[atom.element]
            new_coords = np.dot(test_v, basis)
            # in plane coordinates - z-axis is L-axis, which
            # we don't care about for B1
            ip_coords = new_coords[0:2]
            ndx.extend((i * std_ndx).tolist())
            grid = r1 * b1_points
            grid += ip_coords
            points.extend(grid)

            # B5
            # find distance along L-axis, then subtract this from vector from
            # vector from molecule to this atom to get the B5 vector
            # add the atom's radius to get the full B5
            b = np.dot(test_v, L_axis)
            test_B5_v = test_v - (b * L_axis)
            test_B5 = np.linalg.norm(test_B5_v) + radii_dict[atom.element]
            if B5 is None or test_B5 > B5:
                B5 = test_B5
                start_x = atom.coords - test_B5_v
                if np.linalg.norm(test_B5_v) > 3 * np.finfo(float).eps:
                    perp_vec = test_B5_v
                else:
                    # this atom might be along the L-axis, in which case use
                    # any vector orthogonal to L-axis
                    v_n = test_v / np.linalg.norm(test_v)
                    perp_vec = utils.perp_vector(L_axis)
                    perp_vec -= np.dot(v_n, perp_vec) * v_n

                end = start_x + test_B5 * (
                    perp_vec / np.linalg.norm(perp_vec)
                )

                vector["B5"] = (start_x, end)

        points = np.array(points)

        hull = ConvexHull(points)

        # import matplotlib.pyplot as plt
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # plt.plot(0, 0, 'kx')
        # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'ro')

        # ax = plt.gca()
        # ax.set_aspect('equal')

        # go through each edge, find a vector perpendicular to the one
        # defined by the edge that passes through the origin
        # the length of the shortest of these vectors is B1
        for i in range(0, len(hull.vertices) - 1):
            # the vertices of the hull are organized in a counterclockwise
            # direction, so neighboring indices define an edge
            v_ndx_0, v_ndx_1 = hull.vertices[i:i + 2]
            # find 'tangent' of the edge
            v = points[v_ndx_1] - points[v_ndx_0]
            v /= np.linalg.norm(v)
            # find normal to this edge by projecting a vector from the
            # L axis to one of the points on the edge
            b = points[v_ndx_0]
            t = np.dot(b, v)
            perp = b - t * v
            # plt.plot(perp[0], perp[1], 'g*')
            test_b1 = np.linalg.norm(perp)
            if B1 is None or test_b1 < B1:
                B1 = test_b1
                # figure out vector from L axis to represent B1
                b1_atom = targets[ndx[v_ndx_0]]
                test_v = start.bond(b1_atom)
                test_B1_v = test_v - (np.dot(test_v, L_axis) * L_axis)
                start_x = b1_atom.coords - test_B1_v
                end = x_vec * perp[0] + ip_vector * perp[1]
                end += start_x
                vector["B1"] = (start_x, end)

        # figure out B2-4
        # these need to be sorted in increasing order
        # for now, they will just be Bpar for the one opposite B1
        # and Bperp1 and Bperp2 for the ones perpendicular to B1
        b1_norm = end - start_x
        b1_norm /= np.linalg.norm(b1_norm)
        b1_perp = np.cross(L_axis, b1_norm)
        b1_perp /= np.linalg.norm(b1_perp)
        Bpar = None
        Bperp1 = None
        Bperp2 = None
        for atom in targets:
            test_v = start.bond(atom)
            b = np.dot(test_v, L_axis)
            test_B_v = test_v - (b * L_axis)
            test_par_vec = np.dot(test_B_v, b1_norm) * b1_norm
            test_par_vec -= radii_dict[atom.element] * b1_norm
            start_x = atom.coords - test_B_v
            end = start_x + test_par_vec

            test_Bpar = np.linalg.norm(end - start_x)
            if Bpar is None or test_Bpar > Bpar:
                Bpar = test_Bpar
                par_vec = (start_x, end)

            perp_vec = np.dot(test_B_v, b1_perp) * b1_perp
            if np.dot(test_B_v, b1_perp) > 0 or np.isclose(np.dot(b1_perp, test_B_v), 0):
                test_perp_vec1 = perp_vec + radii_dict[atom.element] * b1_perp
                end = start_x + test_perp_vec1
                test_Bperp1 = np.linalg.norm(end - start_x)
                if Bperp1 is None or test_Bperp1 > Bperp1:
                    Bperp1 = test_Bperp1
                    perp_vec1 = (start_x, end)
            if np.dot(test_B_v, b1_perp) < 0 or np.isclose(np.dot(b1_perp, test_B_v), 0):
                test_perp_vec2 = perp_vec - radii_dict[atom.element] * b1_perp
                end = start_x + test_perp_vec2
                test_Bperp2 = np.linalg.norm(end - start_x)
                if Bperp2 is None or test_Bperp2 > Bperp2:
                    Bperp2 = test_Bperp2
                    perp_vec2 = (start_x, end)

        # put B2-4 in order
        i = 0
        Bs = [Bpar, Bperp1, Bperp2]
        Bvecs = [par_vec, perp_vec1, perp_vec2]
        while Bs:
            max_b = max(Bs)
            n = Bs.index(max_b)
            max_v = Bvecs.pop(n)
            Bs.pop(n)

            if i == 0:
                B4 = max_b
                vector["B4"] = max_v
            elif i == 1:
                B3 = max_b
                vector["B3"] = max_v
            elif i == 2:
                B2 = max_b
                vector["B2"] = max_v
            i += 1

        params = {
            "B1": B1,
            "B2": B2,
            "B3": B3,
            "B4": B4,
            "B5": B5,
            "L": L,
        }

        # plt.plot(points[min_ndx,0], points[min_ndx,1], 'g*')

        # plt.show()

        if return_vector:
            return vector
        return params


    # geometry manipulation
    def append_structure(self, structure):
        from AaronTools.component import Component

        if not isinstance(structure, Geometry):
            structure = Component(structure)
        if not self.components:
            self.detect_components()
        self.components += [structure]
        self.rebuild()

    def update_geometry(self, structure):
        """
        Replace current coords with those from :structure:

        :structure: a file name, atom list, Geometry or np.array() of shape Nx3
        """
        if isinstance(structure, np.ndarray):
            coords = structure
            elements = None
        else:
            atoms = Geometry(structure).atoms
            elements = [a.element for a in atoms]
            coords = [a.coords for a in atoms]
        if len(coords) != len(self.atoms):
            raise RuntimeError(
                "Updated geometry has different number of atoms"
            )
        for i, row in enumerate(coords):
            if elements is not None and elements[i] != self.atoms[i].element:
                raise RuntimeError(
                    "Updated coords atom order doesn't seem to match original "
                    "atom order. Stopping..."
                )
            self.atoms[i].coords = row
        self.refresh_connected()
        return

    def get_all_connected(self, target):
        """returns a list of all elements on the target atom's monomer"""

        def _get_all_connected(geom, target, avoid):
            atoms = [target]
            for atom in target.connected:
                if atom not in avoid:
                    new_avoid = avoid + [target]
                    atoms.extend(
                        [
                            x
                            for x in _get_all_connected(geom, atom, new_avoid)
                            if x not in atoms
                        ]
                    )

            return atoms

        target = self.find(target)[0]
        atoms = _get_all_connected(self, target, [])

        return atoms

    def get_fragment(
        self, start, stop=None, as_object=False, copy=False, biggest=False
    ):
        """
        Returns:
            [Atoms()] if as_object == False
            Geometry() if as_object == True

        :start: the atoms to start on
        :stop: the atom(s) to avoid
            stop=None will try all possibilities and return smallest fragment
        :as_object: return as list (default) or Geometry object
        :copy: whether or not to copy the atoms before returning the list;
            copy will automatically fix connectivity information
        :biggest: if stop=None, will return biggest possible fragment instead of smallest
        """
        start = self.find(start)
        if stop is None:
            best = None
            for stop in itertools.chain(*[s.connected for s in start]):
                frag = self.get_fragment(start, stop, as_object, copy)
                if (
                    best is None
                    or (len(frag) < len(best) and not biggest)
                    or (len(frag) > len(best) and biggest)
                ):
                    best = frag
            return best
        stop = self.find(stop)

        stack = deque(start)
        frag = start
        while len(stack) > 0:
            connected = stack.popleft()
            connected = connected.connected - set(stop) - set(frag)
            stack.extend(sorted(connected))
            frag += sorted(connected)

        if as_object:
            return self.copy(atoms=frag, comment="")
        if copy:
            return self._fix_connectivity(frag, copy=True)
        return frag

    def remove_fragment(self, start, avoid=None, add_H=True):
        """
        Removes a fragment of the geometry
        Returns:
            (list) :start: + the removed fragment

        :start: the atom of the fragment to be removed that attaches to the
            rest of the geometry
        :avoid: the atoms :start: is attached to that should be avoided
        :add_H: default is to change :start: to H and update bond lengths, but
            add_H=False overrides this behaviour

        """
        start = self.find(start)
        if avoid is not None:
            avoid = self.find(avoid)
        frag = self.get_fragment(start, avoid)[len(start) :]
        self -= frag
        rv = start + frag

        # replace start with H
        if add_H:
            for a in start:
                a.element = "H"
                a._set_radii()
                self.change_distance(a, a.connected - set(frag), fix=2)
        return rv

    def coord_shift(self, vector, targets=None):
        """
        shifts the coordinates of the target atoms by a vector
        parameters:
            vector (np.array) - the shift vector
            targets (list) - the target atoms to shift (default to all)
        """
        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float64)
        for t in targets:
            t.coords += vector
        return

    def change_distance(
        self, a1, a2, dist=None, adjust=False, fix=0, as_group=True
    ):
        """For setting/adjusting bond length between atoms
        Parameters:
            a1_arg - the first atom
            a2_arg - the second atom
            dist - the distance to change by/to. Default is to set the bond
                   lenght to that determined by RADII
            adjust - default is to set the bond length to `dist`, adjust=True
                     indicates the current bond length should be adjusted by
                     `dist`
            fix - default is to move both a1 and a2 by half of `dist`, fix=1
                  will move only a2 and fix=2 will move only a1
            as_group - default is to move the fragments connected to a1 and a2
                       as well, as_group=False will only move the requested
                       atom(s)
        """
        a1, a2 = self.find_exact(a1, a2)

        # determine new bond length
        if isinstance(dist, str):
            dist = float(dist)
        if dist is None:
            if hasattr(a1, "_radii") and hasattr(a2, "_radii"):
                new_dist = a1._radii + a2._radii
            elif not hasattr(a1, "_radii"):
                self.LOG.warning("no radii for %s", a1)
                return
            elif not hasattr(a2, "_radii"):
                self.LOG.warning("no radii for %s", a2)
                return
        elif adjust:
            new_dist = a1.dist(a2) + dist
        else:
            new_dist = dist
        dist = a1.dist(a2)

        # adjustment vector for each atom
        adj_a1 = (new_dist - dist) * a2.bond(a1) / a2.dist(a1)
        adj_a2 = (new_dist - dist) * a1.bond(a2) / a1.dist(a2)
        if fix == 0:
            adj_a1 /= 2
            adj_a2 /= 2
        elif fix == 1:
            adj_a1 = None
        elif fix == 2:
            adj_a2 = None
        else:
            raise ValueError(
                "Bad parameter `fix` (should be 0, 1, or 2):", fix
            )

        # get atoms to adjust
        if as_group:
            a1 = self.get_fragment(a1, a2)
            a2 = self.get_fragment(a2, a1)
        else:
            a1 = [a1]
            a2 = [a2]

        # translate atom(s)
        for i in a1:
            if adj_a1 is None:
                break
            i.coords += adj_a1
        for i in a2:
            if adj_a2 is None:
                break
            i.coords += adj_a2

        return

    def rotate_fragment(self, start, avoid, angle):
        """rotates the all atoms on the 'start' side of the
        start-avoid bond about the bond vector by angle"""
        start = self.find(start)[0]
        avoid = self.find(avoid)[0]
        shift = start.coords
        self.coord_shift(-shift)
        self.rotate(
            start.bond(avoid),
            angle=angle * 180 / np.pi,
            targets=self.get_fragment(start, avoid),
        )
        self.coord_shift(shift)

    def rotate(self, w, angle=None, targets=None, center=None):
        """
        rotates target atoms by an angle about an axis

        :w: (np.array) - the axis of rotation (doesnt need to be unit vector)
            or a quaternion (angle not required then)
        :angle: (float) - the angle by which to rotate (in radians)
        :targets: (list) - the atoms to rotate (defaults to all)
        :center: (Atom or list) - if provided, the atom (or COM of a list)
            will be centered at the origin before rotation, then shifted
            back after rotation
        """
        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)

        # shift geometry to place center atom at origin
        if center is not None:
            if not (
                hasattr(center, "__len__")
                and all(isinstance(x, float) for x in center)
            ):
                tmp = self.find(center)
                if len(tmp) > 1:
                    center = deepcopy(self.COM(tmp))
                else:
                    center = deepcopy(tmp[0].coords)
            else:
                center = deepcopy(center)
            self.coord_shift(-1 * center)

        if not isinstance(w, np.ndarray):
            w = np.array(w, dtype=np.double)

        if angle is not None and len(w) == 3:
            w = w / np.linalg.norm(w)
            q = np.hstack(([np.cos(angle / 2)], w * np.sin(angle / 2)))
        elif len(w) != 4:
            raise TypeError(
                """Vector `w` must be either a rotation vector (len 3)
                or a quaternion (len 4). Angle parameter required if `w` is a
                rotation vector"""
            )
        else:
            q = w

        q /= np.linalg.norm(q)
        qs = q[0]
        qv = q[1:]

        xyz = self.coordinates(targets)
        xprod = np.cross(qv, xyz)
        qs_xprod = 2 * qs * xprod
        qv_xprod = 2 * np.cross(qv, xprod)

        xyz += qs_xprod + qv_xprod
        for t, coord in zip(targets, xyz):
            t.coords = coord

        if center is not None:
            self.coord_shift(center)

    def change_angle(
        self,
        a1,
        a2,
        a3,
        angle,
        radians=True,
        adjust=False,
        fix=0,
        as_group=True,
    ):
        """For setting/adjusting angle between atoms
        Parameters:
            a1 - the first atom
            a2 - the second atom (vertex)
            a3 - the third atom
            angle - the angle to change by/to
            radians - default units are radians, radians=False uses degrees
            adjust - default is to set the angle to `angle`, adjust=True
                indicates the current angle should be adjusted by `angle`
            fix - default is to move both a1 and a3 by half of `angle`, fix=1
                will move only a3 and fix=3 will move only a1
            as_group - default is to move the fragments connected to a1 and a3
                as well, as_group=False will only move the requested atom(s)
        """
        try:
            a1, a2, a3 = self.find([a1, a2, a3])
        except ValueError:
            raise LookupError(
                "Bad atom request: {}, {}, {}".format(a1, a2, a3)
            )

        # get rotation vector
        v1 = a2.bond(a1)
        v2 = a2.bond(a3)
        w = np.cross(v1, v2)
        w = w / np.linalg.norm(w)

        # determine rotation angle
        if not radians:
            angle = np.deg2rad(angle)
        if not adjust:
            angle -= a2.angle(a1, a3)

        # get target fragments
        a1_frag = self.get_fragment(a1, a2)
        a3_frag = self.get_fragment(a3, a2)

        # shift a2 to origin
        self.coord_shift(-a2.coords, a1_frag)
        self.coord_shift(-a2.coords, a3_frag)

        # perform rotation
        if fix == 0:
            angle /= 2
            self.rotate(w, -angle, a1_frag)
            self.rotate(w, angle, a3_frag)
        elif fix == 1:
            self.rotate(w, angle, a3_frag)
        elif fix == 3:
            self.rotate(w, -angle, a1_frag)
        else:
            raise ValueError("fix must be 0, 1, 3 (supplied: {})".format(fix))

        # shift a2 back to original location
        self.coord_shift(a2.coords, a1_frag)
        self.coord_shift(a2.coords, a3_frag)

    def change_dihedral(self, *args, **kwargs):
        """
        For setting/adjusting dihedrals

        *args
        :a1: the first atom
        :a2: the second atom
        :a3: the third atom (optional for adjust=True if as_group=True)
        :a4: the fourth atom (optional for adjust=True if as_group=True)
        :dihedral: the dihedral to change by/to

        **kwargs
        :fix: default is to move both a1 and a4 by half of `dihedral`,
            fix=1 will move only a4 and fix=4 will move only a1
        :adjust: default is to set the dihedral to `dihedral`, adjust=True
            indicates the current dihedral should be adjusted by `dihedral`
        :as_group: default is to move the fragments connected to a1 and a3
            as well, as_group=False will only move the requested atom(s)
        :radians: default units are degrees, radians=True to use radians
        """
        fix = kwargs.get("fix", 0)
        adjust = kwargs.get("adjust", False)
        as_group = kwargs.get("as_group", True)
        radians = kwargs.get("radians", False)
        left_over = set(kwargs.keys()) - set(
            ["fix", "adjust", "as_group", "radians"]
        )
        if left_over:
            raise SyntaxError(
                "Unused **kwarg(s) provided: {}".format(left_over)
            )
        # get atoms
        count = len(args)
        if count == 3 and adjust:
            # we can just define the bond to rotate about, as long as we are
            # adjusting, not setting, the whole fragments on either side
            as_group = True
            a2, a3 = self.find_exact(*args[:2])
            dihedral = args[2]
            try:
                a1 = next(iter(a2.connected - set([a2, a3])))
            except StopIteration:
                a1 = next(iter(set(self.atoms) - set([a2, a3])))
            try:
                a4 = next(iter(a3.connected - set([a1, a2, a3])))
            except StopIteration:
                a4 = next(iter(set(self.atoms) - set([a1, a2, a3])))
        elif count != 5:
            raise TypeError(
                "Number of atom arguments provided insufficient to define "
                + "dihedral"
            )
        else:
            a1, a2, a3, a4 = self.find_exact(*args[:4])
            dihedral = args[4]
        # get fragments
        if as_group:
            a2_frag = self.get_fragment(a2, a3)[1:]
            a3_frag = self.get_fragment(a3, a2)[1:]
            if any(atom in a2_frag for atom in a3_frag):
                self.LOG.warning(
                    "changing dihedral that is part of a ring: %s %s", a2, a3
                )
        else:
            a2_frag = [a1]
            a3_frag = [a4]

        # fix units
        if not radians:
            dihedral = np.deg2rad(dihedral)
        # get adjustment
        if not adjust:
            dihedral -= self.dihedral(a1, a2, a3, a4)

        # rotate fragments
        if not a2_frag and not a3_frag:
            raise RuntimeError(
                "Cannot change dihedral, no fragments to target for rotation"
            )
        if not a2_frag and fix == 0:
            fix = 1
        if not a3_frag and fix == 0:
            fix = 4
        if fix == 0:
            dihedral /= 2
            self.rotate(a2.bond(a3), -dihedral, a2_frag, center=a2)
            self.rotate(a2.bond(a3), dihedral, a3_frag, center=a3)
        elif fix == 1:
            self.rotate(a2.bond(a3), dihedral, a3_frag, center=a3)
        elif fix == 4:
            self.rotate(a2.bond(a3), -dihedral, a2_frag, center=a2)
        else:
            raise ValueError(
                "`fix` must be 0, 1, or 4 (supplied: {})".format(fix)
            )

    def minimize_sub_torsion(self, geom=None, all_frags=False, increment=30):
        """rotate substituents to try to minimize LJ potential
        geom: calculate LJ potential between self and another geometry-like
              object, instead of just within self
        all_frags: minimize rotatable bonds on substituents
        """
        # minimize torsion for each substituent

        if not hasattr(self, "substituents") or self.substituents is None:
            self.detect_substituents()

        for i, sub in enumerate(sorted(self.substituents, reverse=True)):
            axis = sub.atoms[0].bond(sub.end)
            center = sub.end
            self.minimize_torsion(
                sub.atoms, axis, center, geom, increment=increment
            )
            if all_frags:
                for frag, a, b in self.get_frag_list(
                    targets=sub.atoms, max_order=1
                ):
                    axis = a.bond(b)
                    center = b.coords
                    self.minimize_torsion(frag, axis, center, geom)

    def minimize_torsion(self, targets, axis, center, geom=None, increment=5):
        """
        Rotate :targets: to minimize the LJ potential

        :targets: the target atoms to rotate
        :axis: the axis by which to rotate
        :center: where to center before rotation
        :geom: calculate LJ potential between self and another geometry-like
            object, instead of just within self
        """
        targets = Geometry(
            self.find(targets),
            refresh_connected=False,
            refresh_ranks=False,
        )

        if geom is None or geom is self:
            from AaronTools.finders import NotAny

            try:
                geom = Geometry(
                    self.find(NotAny(targets)),
                    refresh_connected=False,
                    refresh_ranks=False,
                )
            except LookupError:
                return

        E_min = None
        angle_min = None
        # copied an reorganized some stuff from Geometry.rotate for
        # performance reasons
        if hasattr(center, "__iter__") and all(
            isinstance(x, float) for x in center
        ):
            center_coords = center
        else:
            center_coords = self.COM(center)

        axis = axis / np.linalg.norm(axis)
        q = np.hstack(
            (
                [np.cos(np.deg2rad(increment) / 2)],
                axis * np.sin(np.deg2rad(increment) / 2),
            )
        )
        q /= np.linalg.norm(q)
        qs = q[0]
        qv = q[1:]

        # rotate targets by increment and save lowest energy
        angle = 0
        xyz = targets.coords
        for inc in range(0, 360, increment):
            angle += increment

            xyz -= center_coords
            xprod = np.cross(qv, xyz)
            qs_xprod = 2 * qs * xprod
            qv_xprod = 2 * np.cross(qv, xprod)

            xyz += qs_xprod + qv_xprod
            xyz += center_coords
            for t, coord in zip(targets.atoms, xyz):
                t.coords = coord

            energy = targets.LJ_energy(other=geom, use_prev_params=True)

            if E_min is None or energy < E_min:
                E_min = energy
                angle_min = angle

        # rotate to min angle
        self.rotate(
            axis, np.deg2rad(angle_min - angle), targets=targets, center=center_coords
        )

        return

    def substitute(self, sub, target, attached_to=None, minimize=False):
        """
        substitutes fragment containing `target` with substituent `sub`
        if attached_to is provided, this is the atom where the substituent is attached
        if attached_to==None, replace the smallest fragment containing `target`
        minimize - bool, rotate sub to lower LJ potential
        """
        from AaronTools.component import Component

        # set up substituent
        if not isinstance(sub, AaronTools.substituent.Substituent):
            sub = AaronTools.substituent.Substituent(sub)
        sub.refresh_connected()
        # determine target and atoms defining connection bond
        target = self.find(target)
        # if we have components, do the substitution to the component
        # otherwise, just do it on self
        geom = self
        if hasattr(self, "components") and self.components is not None:
            for comp in self.components:
                if target in comp:
                    geom = comp
                    break

        # attached_to is provided or is the atom giving the
        # smallest target fragment
        if attached_to is not None:
            attached_to = geom.find_exact(attached_to)
        else:
            smallest_frag = None
            smallest_attached_to = None
            # get all possible connection points
            attached_to = set()
            for t in target:
                attached_to = attached_to | (t.connected - set(target))
            # find smallest fragment
            for e in attached_to:
                frag = geom.get_fragment(target, e)
                if smallest_frag is None or len(frag) < len(smallest_frag):
                    smallest_frag = frag
                    smallest_attached_to = e
            attached_to = [smallest_attached_to]
        if len(attached_to) != 1:
            raise NotImplementedError(
                "Can only replace substituents with one point of attachment"
            )
        attached_to = attached_to[0]
        sub.end = attached_to

        # determine which atom of target fragment is connected to attached_to
        sub_attach = attached_to.connected & set(target)
        if len(sub_attach) > 1:
            raise NotImplementedError(
                "Can only replace substituents with one point of attachment"
            )
        if len(sub_attach) < 1:
            raise LookupError("attached_to atom not connected to targets")
        sub_attach = sub_attach.pop()

        # manipulate substituent geometry; want sub.atoms[0] -> sub_attach
        #   attached_to == sub.end
        #   sub_attach will eventually be sub.atoms[0]
        # move attached_to to the origin
        shift = np.array([x for x in attached_to.coords])
        geom.coord_shift(-1 * shift)
        # align substituent to current bond
        bond = geom.bond(attached_to, sub_attach)
        sub.align_to_bond(bond)
        # shift geometry back and shift substituent to appropriate place
        geom.coord_shift(shift)
        sub.coord_shift(shift)

        # tag and update name for sub atoms
        for i, s in enumerate(sub.atoms):
            s.add_tag(sub.name)
            if i > 0:
                s.name = sub_attach.name + "." + s.name
            else:
                s.name = sub_attach.name

        # add first atoms of new substituent where the target atoms were
        # add the rest of the new substituent at the end
        old = geom.get_fragment(target, attached_to)
        for i, a in enumerate(old):
            if i == len(sub.atoms):
                break
            geom.atoms.insert(geom.atoms.index(old[i]), sub.atoms[i])
            sub.atoms[i].name = old[i].name
        else:
            if len(sub.atoms) > len(old):
                geom += sub.atoms[i + 1 :]
        # remove old substituent
        geom -= old
        attached_to.connected.discard(sub_attach)

        # fix connections (in lieu of geom.refresh_connected(), since clashing may occur)
        attached_to.connected.add(sub.atoms[0])
        sub.atoms[0].connected.add(attached_to)

        # fix bond distance
        geom.change_distance(attached_to, sub.atoms[0], as_group=True, fix=1)

        # clean up changes
        if isinstance(geom, Component):
            self.substituents += [sub]
            self.detect_backbone(to_center=self.backbone)
            self.rebuild()
        self.refresh_ranks()
        if minimize:
            self.minimize_torsion(sub.atoms, bond, shift)
        return sub

    def find_substituent(self, start, for_confs=False):
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
            if for_confs:
                for sub in self.get_substituents(for_confs=not for_confs):
                    if sub.atoms[0] == start:
                        return None
            msg = "Could not find substituent starting at atom {}."
            raise LookupError(msg.format(start.name))

        if not hasattr(self, "substituents") or self.substituents is None:
            self.substituents = []

        self.substituents.append(sub)

    def get_substituents(self, for_confs=True):
        """
        Returns list of all substituents found on all components

        :for_confs: if true (default), returns only substituents that need to
            be rotated to generate conformers
        """
        rv = []
        if self.components is None:
            self.detect_components()
        for comp in self.components:
            if comp.substituents is None:
                comp.detect_backbone()
            for sub in comp.substituents:
                if for_confs and (sub.conf_num is None or sub.conf_num <= 1):
                    continue
                rv += [sub]
        return rv

    def ring_substitute(
        self, targets, ring_fragment, minimize=False, flip_walk=False
    ):
        """
        take ring, reorient it, put it on self and replace targets with atoms
        on the ring fragment
        ring_fragment - Ring instance or name of ring in the library
        minimize - try other rings with the same name (appended with a number)
                   in the library to see if they fit better
        flip_walk - also flip the rings when minimizing to see if that fits better
        """

        def attach_short(geom, walk, ring_fragment):
            """for when walk < end, rmsd and remove end[1:-1]"""
            # align ring's end to geom's walk
            ring_fragment.RMSD(
                geom,
                align=True,
                targets=ring_fragment.end,
                ref_targets=walk,
                sort=False,
            )

            ring_waddle(geom, targets, [walk[1], walk[-2]], ring_fragment)

            for atom in ring_fragment.end[1:-1]:
                for t in atom.connected:
                    if t not in ring_fragment.end:
                        ring_fragment.remove_fragment(t, atom, add_H=False)
                        ring_fragment -= t

                ring_fragment -= atom

            geom.remove_fragment([walk[0], walk[-1]], walk[1:-1], add_H=False)
            geom -= [walk[0], walk[-1]]

            walk[1].connected.add(ring_fragment.end[0])
            walk[-2].connected.add(ring_fragment.end[-1])
            ring_fragment.end[-1].connected.add(walk[-2])
            ring_fragment.end[0].connected.add(walk[1])
            ring_fragment.end = walk[1:-1]
            geom.atoms.extend(ring_fragment.atoms)
            geom.refresh_ranks()

        def ring_waddle(geom, targets, walk_end, ring):
            """adjusted the new bond lengths by moving the ring in a 'waddling' motion
            pivot on one end atom to adjust the bond lenth of the other, then do
            the same with the other end atom"""
            if hasattr(ring.end[0], "_radii") and hasattr(
                walk_end[0], "_radii"
            ):
                d1 = ring.end[0]._radii + walk_end[0]._radii
            else:
                d1 = ring.end[0].dist(walk_end[0])

            v1 = ring.end[-1].bond(walk_end[0])
            v2 = ring.end[-1].bond(ring.end[0])

            v1_n = np.linalg.norm(v1)
            v2_n = np.linalg.norm(v2)

            target_angle = np.arccos(
                (d1 ** 2 - v1_n ** 2 - v2_n ** 2) / (-2.0 * v1_n * v2_n)
            )
            current_angle = ring.end[-1].angle(ring.end[0], walk_end[0])
            ra = target_angle - current_angle

            rv = np.cross(v1, v2)

            ring.rotate(rv, ra, center=ring.end[-1])

            if hasattr(ring.end[-1], "_radii") and hasattr(
                walk_end[-1], "_radii"
            ):
                d1 = ring.end[-1]._radii + walk_end[-1]._radii
            else:
                d1 = ring.end[-1].dist(walk_end[-1])

            v1 = ring.end[0].bond(walk_end[-1])
            v2 = ring.end[0].bond(ring.end[-1])

            v1_n = np.linalg.norm(v1)
            v2_n = np.linalg.norm(v2)

            target_angle = np.arccos(
                (d1 ** 2 - v1_n ** 2 - v2_n ** 2) / (-2.0 * v1_n * v2_n)
            )
            current_angle = ring.end[0].angle(ring.end[-1], walk_end[-1])
            ra = target_angle - current_angle

            rv = np.cross(v1, v2)

            ring.rotate(rv, ra, center=ring.end[0])

        def clashing(geom, ring):
            from AaronTools.finders import NotAny

            geom_coords = geom.coordinates(NotAny(ring.atoms))
            dist_mat = distance_matrix(geom_coords, ring.coords)
            if np.any(dist_mat < 0.75):
                return True

            return False

        from AaronTools.ring import Ring

        if not isinstance(ring_fragment, Ring):
            ring_fragment = Ring(ring_fragment)

        targets = self.find(targets)
        # we want to keep atom naming conventions consistent with regular substitutions
        for atom in ring_fragment.atoms:
            atom.name = "{}.{}".format(targets[0].name, atom.name)

        # find a path between the targets
        walk = self.shortest_path(*targets)
        if len(ring_fragment.end) != len(walk):
            ring_fragment.find_end(len(walk), start=ring_fragment.end)

        if len(walk) == len(ring_fragment.end) and len(walk) != 2:
            if not minimize:
                attach_short(self, walk, ring_fragment)
            else:
                # to minimize, check VSEPR on self's atoms attached to targets
                # lower deviation is better
                # do this for the original ring and also try flipping the ring
                # ring is flipped by reversing walk
                # check for other rings in the library with ring.\d+
                # e.g. cyclohexane.2
                vsepr1, _ = walk[1].get_vsepr()
                vsepr2, _ = walk[-2].get_vsepr()
                geom = self.copy()
                test_walk = [
                    geom.atoms[i]
                    for i in [self.atoms.index(atom) for atom in walk]
                ]
                frag = ring_fragment.copy()
                attach_short(geom, test_walk, frag)
                new_vsepr1, score1 = test_walk[1].get_vsepr()
                new_vsepr2, score2 = test_walk[-2].get_vsepr()
                new_vsepr3, score3 = frag.end[0].get_vsepr()
                new_vsepr4, score4 = frag.end[-1].get_vsepr()

                score = score1 + score2 + score3 + score4

                min_diff = score
                min_ring = 0

                # print("%s score: %.3f" % (ring_fragment.name, score))

                if flip_walk:
                    geom = self.copy()
                    test_walk = [
                        geom.atoms[i]
                        for i in [self.atoms.index(atom) for atom in walk]
                    ][::-1]
                    frag = ring_fragment.copy()
                    attach_short(geom, test_walk, frag)
                    new_vsepr1, score1 = test_walk[1].get_vsepr()
                    new_vsepr2, score2 = test_walk[-2].get_vsepr()
                    new_vsepr3, score3 = frag.end[0].get_vsepr()
                    new_vsepr4, score4 = frag.end[-1].get_vsepr()

                    score = score1 + score2 + score3 + score4

                    if score < min_diff and not clashing(geom, frag):
                        min_ring = 1
                        min_diff = score

                    # print("flipped %s score: %.3f" % (ring_fragment.name, score))

                # check other rings in library
                # for these, flip the ring end instead of walk
                for ring_name in Ring.list():
                    if re.search("%s\.\d+" % ring_fragment.name, ring_name):
                        test_ring_0 = Ring(ring_name)

                        geom = self.copy()
                        test_walk = [
                            geom.atoms[i]
                            for i in [self.atoms.index(atom) for atom in walk]
                        ]
                        if len(test_ring_0.end) != len(walk):
                            test_ring_0.find_end(
                                len(walk), start=test_ring_0.end
                            )

                        frag = test_ring_0.copy()
                        attach_short(geom, test_walk, frag)
                        new_vsepr1, score1 = test_walk[1].get_vsepr()
                        new_vsepr2, score2 = test_walk[-2].get_vsepr()
                        new_vsepr3, score3 = frag.end[0].get_vsepr()
                        new_vsepr4, score4 = frag.end[-1].get_vsepr()

                        score = score1 + score2 + score3 + score4

                        if score < min_diff and not clashing(geom, frag):
                            min_ring = test_ring_0
                            min_diff = score

                        # print("%s score: %.3f" % (ring_name, score))

                        if flip_walk:
                            test_ring_1 = Ring(ring_name)
                            test_ring_1.end.reverse()

                            geom = self.copy()
                            test_walk = [
                                geom.atoms[i]
                                for i in [
                                    self.atoms.index(atom) for atom in walk
                                ]
                            ]
                            if len(test_ring_0.end) != len(walk):
                                test_ring_0.find_end(
                                    len(walk), start=test_ring_0.end
                                )

                            frag = test_ring_1.copy()
                            attach_short(geom, test_walk, frag)
                            new_vsepr1, score1 = test_walk[1].get_vsepr()
                            new_vsepr2, score2 = test_walk[-2].get_vsepr()
                            new_vsepr3, score3 = frag.end[0].get_vsepr()
                            new_vsepr4, score4 = frag.end[-1].get_vsepr()

                            score = score1 + score2 + score3 + score4

                            # print("flipped %s score: %.3f" % (ring_name, score))

                            if score < min_diff and not clashing(geom, frag):
                                min_ring = test_ring_1
                                min_diff = score

                if not isinstance(min_ring, Ring) and min_ring == 0:
                    walk = self.shortest_path(*targets)
                    attach_short(self, walk, ring_fragment)
                elif not isinstance(min_ring, Ring) and min_ring == 1:
                    walk = self.shortest_path(*targets)[::-1]
                    attach_short(self, walk, ring_fragment)
                else:
                    walk = self.shortest_path(*targets)
                    attach_short(self, walk, min_ring)

        elif not walk[1:-1]:
            raise ValueError(
                "insufficient information to close ring - selected atoms are bonded to each other: %s"
                % (" ".join(str(a) for a in targets))
            )

        else:
            raise ValueError(
                "this ring is not appropriate to connect\n%s\nand\n%s:\n%s\nspacing is %i; expected %i"
                % (
                    targets[0],
                    targets[1],
                    ring_fragment.name,
                    len(ring_fragment.end),
                    len(walk),
                )
            )
        # AaronJr needs to know this when relaxing changes
        return ring_fragment.atoms

    def change_element(
        self, target, new_element, adjust_bonds=False, adjust_hydrogens=False, hold_steady=None
    ):
        """change the element of an atom on self
        target              - target atom
        new_element         - str:  element of new atom
        adjust_bonds        - bool: adjust distance to bonded atoms
        adjust_hydrogens    - bool: try to add or remove hydrogens and guess how many
                                    hydrogens to add or remove
                              tuple(int, str): remove specified number of hydrogens and
                                               set the geometry to the specified shape
                                               (see Atom.get_shape for a list of shapes)
        hold_steady         - atom: atom bonded to target that will be held steady when
                                    adjusting bonds; Default - longest fragment
        """

        def get_corresponding_shape(target, shape_object, frags):
            """
            returns shape object, but where shape_object.atoms are lined up with
            target.connected as much as possible
            """
            shape_object.coord_shift(
                target.coords - shape_object.atoms[0].coords
            )
            if len(frags) == 0:
                return shape_object

            # to minimize changes to the structure, things are aligned to the largest fragment
            max_frag = sorted(frags, key=len, reverse=True)[0]
            angle = target.angle(shape_object.atoms[1], max_frag[0])
            v1 = target.bond(max_frag[0])
            v2 = shape_object.atoms[0].bond(shape_object.atoms[1])
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            rv = np.cross(v1, v2)

            # could have numerical issues
            # avoid those, but check to see if the angle is 180 degrees
            if abs(np.linalg.norm(rv)) < 10 ** -3 or abs(angle) < 10 ** -3:
                if np.dot(v1, v2) == -1:
                    rv = np.array([v1[2], v1[0], v1[1]])
                    shape_object.rotate(rv, np.pi, center=target)
                    angle = 0

            if abs(np.linalg.norm(rv)) > 10 ** -3 and abs(angle) > 10 ** -3:
                shape_object.rotate(rv, -angle, center=target)

            # rotate about the vector from the center (target atom) to the first
            # atom in shape (which will be the largest fragment) to get the shape object
            # as lined up with the rest of the connected atoms as possible
            rv = target.bond(shape_object.atoms[1])
            min_dev = None
            min_angle = 0
            inc = 5
            angle = 0
            while angle < 360:
                angle += inc
                shape_object.rotate(rv, np.deg2rad(inc), center=target)

                previous_positions = [0]
                dev = 0
                for j, frag in enumerate(sorted(frags, key=len, reverse=True)):
                    if j == 0:
                        continue
                    v1 = target.bond(frag[0])
                    max_overlap = None
                    corresponding_position = None
                    for i, position in enumerate(shape_object.atoms[1:]):
                        if i in previous_positions:
                            continue
                        v2 = shape_object.atoms[0].bond(position)
                        d = np.dot(v1, v2)
                        if max_overlap is None or d > max_overlap:
                            max_overlap = d
                            corresponding_position = i

                    if corresponding_position is None:
                        continue

                    previous_positions.append(corresponding_position)
                    dev += (
                        max_overlap
                        - (
                            np.linalg.norm(frag[0].coords)
                            * np.linalg.norm(
                                shape_object.atoms[
                                    corresponding_position + 1
                                ].coords
                            )
                        )
                    ) ** 2

                if min_dev is None or dev < min_dev:
                    min_dev = dev
                    min_angle = angle

            shape_object.rotate(rv, np.deg2rad(min_angle), center=target)

            return shape_object

        target = self.find(target)
        if len(target) > 1:
            raise RuntimeError(
                "only one atom's element can be changed at a time (%i attempted)"
                % len(target)
            )
        else:
            target = target[0]

        # new_atom is only used to determine how many H's to add
        new_atom = Atom(
            element=new_element, name=target.name, coords=target.coords
        )

        if adjust_hydrogens is True:
            # try to determine how many hydrogens to add based on how many hydrogens are currently
            # bonded and what the saturation of the atoms is
            # e.g. C(H3) -> N(H?)
            # C's saturation is 4, it's missing one
            # N's saturation is 3, it should be missing one: N(H2)
            if hasattr(target, "_saturation") and hasattr(
                new_atom, "_saturation"
            ):
                change_hydrogens = new_atom._saturation - target._saturation
                new_shape = None
            else:
                raise RuntimeError(
                    "H adjust requested, but saturation is not known for %s"
                    % ", ".join(
                        [
                            atom.element
                            for atom in [target, new_atom]
                            if not hasattr(atom, "_saturation")
                        ]
                    )
                )

        elif isinstance(adjust_hydrogens, tuple):
            # tuple of (change in hydrogens, vsepr shape) was given
            change_hydrogens, new_shape = adjust_hydrogens

        else:
            # no change was requested, only the element will change
            # and maybe bond lengths
            change_hydrogens = 0
            new_shape = None

        if change_hydrogens != 0 or new_shape is not None:
            # if we're removing hydrogens, check if we have enough to remove
            if change_hydrogens < 0:
                n_hygrogens = sum(
                    [1 for atom in target.connected if atom.element == "H"]
                )
                if n_hygrogens + change_hydrogens < 0:
                    raise RuntimeError(
                        "cannot remove %i hydrogens from an atom with %i hydrogens"
                        % (abs(change_hydrogens), n_hygrogens)
                    )

            # get vsepr geometry
            old_shape, score = target.get_vsepr()

            if new_shape is None:
                shape = old_shape
                if hasattr(new_atom, "_connectivity"):
                    new_connectivity = new_atom._connectivity
                else:
                    new_connectivity = None

                # if we're changing the number of hydrogens, but no shape was specified,
                # we will remove hydrogens one by one and see what shape we end up with
                # shape changes are based on rules like adding a bond to a trigonal planar
                # atom will cause it to be tetrahedral (e.g. carbocation gaining a hydride)
                for i in range(0, abs(change_hydrogens)):
                    shape = Atom.new_shape(
                        shape, new_connectivity, np.sign(change_hydrogens)
                    )
                    if shape is None:
                        raise RuntimeError(
                            "shape changed from %s to None" % old_shape
                        )

                new_shape = shape

            shape_object = Geometry(Atom.get_shape(new_shape))

            if (
                len(shape_object.atoms[1:]) - len(target.connected)
                != change_hydrogens
            ):
                # insufficient to change the shape
                # we cannot delete random fragments
                raise RuntimeError(
                    "number of positions changed by %i, but a change of %i hydrogens was attempted"
                    % (
                        len(shape_object.atoms[1:]) - len(target.connected),
                        change_hydrogens,
                    )
                )

            # get each branch off of the target atom
            frags = [
                self.get_fragment(atom, target) for atom in target.connected
            ]

            if new_shape != old_shape or change_hydrogens == 0:
                if change_hydrogens < 0:
                    # remove extra hydrogens
                    shape_object = get_corresponding_shape(
                        target, shape_object, frags
                    )
                    removed_Hs = 1
                    while removed_Hs <= abs(change_hydrogens):
                        H_atom = [
                            atom
                            for atom in target.connected
                            if atom.element == "H"
                        ][0]
                        self -= H_atom
                        removed_Hs += 1

                # get fragments after hydrogens have been removed
                frags = [
                    self.get_fragment(atom, target)
                    for atom in target.connected
                ]

                # align the shape object
                # the shape object's first atom will be on top of target
                # the second will be lined up with the largest fragment on target
                # the rest are rotated to minimize deviation from the remaining groups
                shape_object = get_corresponding_shape(
                    target, shape_object, frags
                )

                # ring detection - remove ring fragments because those are more difficult to adjust
                remove_frags = []
                for i, frag1 in enumerate(frags):
                    for frag2 in frags[i + 1 :]:
                        dups = [atom for atom in frag2 if atom in frag1]
                        if len(dups) != 0:
                            remove_frags.append(frag1)
                            remove_frags.append(frag2)

                # add Hs if needed
                if change_hydrogens > 0:
                    # determine which connected atom is occupying which position on the shape
                    shape_object = get_corresponding_shape(
                        target, shape_object, frags
                    )

                    positions = []
                    for j, frag in enumerate(
                        sorted(frags, key=len, reverse=True)
                    ):
                        v2 = target.bond(frag[0])
                        max_overlap = None
                        position = None
                        for i, pos in enumerate(shape_object.atoms[1:]):
                            v1 = shape_object.atoms[0].bond(pos)
                            if i in positions:
                                continue

                            d = np.dot(v1, v2)

                            if max_overlap is None or d > max_overlap:
                                max_overlap = d
                                position = i

                        positions.append(position)

                    # add hydrogens to positions that are not occupied
                    for open_position in [
                        i + 1
                        for i in range(0, len(shape_object.atoms[1:]))
                        if i not in positions
                    ]:
                        # add one because the 0th "position" of the shape is the central atom
                        H_atom = Atom(
                            element="H",
                            coords=shape_object.atoms[open_position].coords,
                            name=str(len(self.atoms) + 1),
                        )

                        self.change_distance(target, H_atom, fix=1)
                        self += H_atom
                        target.connected.add(H_atom)
                        H_atom.connected.add(target)
                        frags.append([H_atom])

                # for each position on the new idealized geometry, find the fragment
                # that corresponds to it the best
                # reorient that fragment to match the idealized geometry

                previous_positions = []
                frag_atoms = []
                first_frag = None
                if hold_steady:
                    hold_steady = self.find(hold_steady)
                for j, frag in enumerate(sorted(frags, key=len, reverse=True)):
                    # print(j, frag)
                    if j == 0 or (hold_steady and frag[0] in hold_steady):
                        # skip the first fragment
                        # that's already aligned with one of the atoms on shape_object
                        first_frag = frag
                        continue
                    frag_atoms.extend(frag)
                    v1 = target.bond(frag[0])
                    max_overlap = None
                    corresponding_position = None
                    for i, position in enumerate(shape_object.atoms[2:]):
                        if i in previous_positions:
                            continue
                        v2 = shape_object.atoms[0].bond(position)
                        d = np.dot(v1, v2)
                        # determine by max. bond overlap
                        if max_overlap is None or d > max_overlap:
                            max_overlap = d
                            corresponding_position = i

                    previous_positions.append(corresponding_position)
                    corresponding_position += 2

                    v1 = target.bond(frag[0])
                    v1 /= np.linalg.norm(v1)
                    v2 = shape_object.atoms[0].bond(
                        shape_object.atoms[corresponding_position]
                    )
                    v2 /= np.linalg.norm(v2)

                    rv = np.cross(v1, v2)

                    if np.linalg.norm(rv) < 10 ** -3:
                        continue

                    c = np.linalg.norm(v1 - v2)

                    if abs((c ** 2 - 2.0) / -2.0) >= 1:
                        continue

                    angle = np.arccos((c ** 2 - 2.0) / -2.0)

                    self.rotate(rv, angle, targets=frag, center=target)

                # rotate the normal of this atom to be parallel to the largest group
                # this makes changing to trigonal planar look cleaner
                # don't do this if it isn't planar
                if "planar" in new_shape:
                    for frag in frags:
                        if first_frag and frag is first_frag:
                            stop = frag[0]
                            other_vsepr = stop.get_vsepr()[0]
                            if isinstance(other_vsepr, str) and "planar" in other_vsepr:
                                min_torsion = None
                                for atom in target.connected:
                                    if atom is stop:
                                        continue

                                    for a4 in stop.connected:
                                        if a4 is target:
                                            continue
                                        torsion = self.dihedral(atom, target, stop, a4)
                                        # print("checking", atom, a4, torsion, min_torsion)
                                        if min_torsion is None or abs(torsion) < abs(min_torsion):
                                            min_torsion = torsion
            
                                if min_torsion is not None and abs(min_torsion) > 1e-2:
                                    angle = min_torsion
                                    targs = []
                                    self.write(outfile="test_ele.xyz")
                                    for f in frags:
                                        if f is frag:
                                            continue
                                        targs.extend(f)

                                    self.rotate(
                                        target.bond(stop),
                                        angle,
                                        targets=targs,
                                        center=target,
                                    )

                    for frag in frags:
                        if first_frag and frag is not first_frag:
                            stop = frag[0]
                            if len(stop.connected) > 1:
                                other_vsepr = stop.get_vsepr()[0]
                                if isinstance(other_vsepr, str) and "planar" in other_vsepr:
                                    min_torsion = None
                                    for atom in target.connected:
                                        if atom is stop:
                                            continue
                                        for atom2 in stop.connected:
                                            if atom2 is target:
                                                continue
                                            torsion = self.dihedral(
                                                atom,
                                                target,
                                                stop,
                                                atom2
                                            )
                                            if min_torsion is None or abs(torsion) < abs(min_torsion):
                                                min_torsion = torsion

                                    if min_torsion is not None and abs(min_torsion) > 1e-2:
                                        angle = -1 * min_torsion
                                        targs = frag
                                        
                                        self.rotate(
                                            target.bond(stop),
                                            angle,
                                            targets=targs,
                                            center=target,
                                        )

        self.refresh_ranks()

        target.element = new_element

        # these methods are normally called when an atom is instantiated
        target._set_radii()
        target._set_connectivity()
        target._set_saturation()

        # fix bond lengths if requested
        # try to guess the bond order based on saturation
        if adjust_bonds:
            from AaronTools.atoms import BondOrder
            bo = BondOrder()
            
            if hold_steady:
                hold_steady = self.find(hold_steady)
            frags = [
                self.get_fragment(atom, target) for atom in target.connected
            ]
            target_bo = 1
            if hasattr(target, "_saturation"):
                target_bo = max(1 + target._saturation - len(target.connected), 1)

            for i, frag in enumerate(sorted(frags, key=len, reverse=True)):
                frag_bo = 1
                if hasattr(frag[0], "_saturation"):
                    frag_bo = max(1 + frag[0]._saturation - len(frag[0].connected), 1)
                
                expected_bo = "%.1f" % float(min(frag_bo, target_bo))
                # print(expected_bo)
                key = bo.key(frag[0], target)
                try:
                    expected_dist = bo.bonds[key][expected_bo]
                except KeyError:
                    expected_dist = None
                
                if hold_steady:
                    self.change_distance(
                        target,
                        frag[0],
                        as_group=True,
                        dist=expected_dist,
                        fix=2 if frag[0] in hold_steady else 1,
                    )
                else:
                    self.change_distance(
                        target,
                        frag[0],
                        as_group=True,
                        dist=expected_dist,
                        fix=2 if i == 0 else 1,
                    )

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
            targets = [
                atom for atom in self.center if atom.is_connected(old_key)
            ]
            if len(targets) > 0:
                new_axis = shift - new_key.coords
            else:
                targets = old_key.connected - set(self.center)
                new_axis = (
                    ligand.COM(targets=new_key.connected) - new_key.coords
                )

            if not targets:
                old_axis = old_key.coords - self.COM(self.center)
            else:
                old_axis = self.COM(targets=targets) - old_key.coords
            w, angle = get_rotation(old_axis, new_axis)
            ligand.rotate(w, angle, center=new_key)
            return ligand

        def map_2_key(old_ligand, ligand, old_keys, new_keys, rev_ang=False):

            # align COM of key atoms
            center = old_ligand.COM(targets=old_keys)
            shift = center - ligand.COM(targets=new_keys)
            ligand.coord_shift(shift)
            remove_centers = []

            old_walk = False
            # bend around key axis
            try:
                old_walk = old_ligand.shortest_path(*old_keys)

            except (LookupError, ValueError):
                # for some ferrocene ligands, AaronTools misidentifies the Fe
                # as another metal center
                # we'll remove any centers that are on the path between the key atoms
                # also, sometimes the ligand atoms don't have the center in their connected
                # attribute, even though the center has the ligand atoms in its
                # connected attribute, so refresh_connected
                self.refresh_connected(targets=self.center)
                # print("old keys:", old_keys)
                # print("old ligand:\n", old_ligand)
                stop=[
                    atom
                    for atom in old_keys[0].connected
                    if atom not in old_ligand.atoms
                ]
                if stop:
                    frag = old_ligand.get_fragment(
                        old_keys[0],
                        stop=stop,
                    )
                    if all(atom in frag for atom in old_keys):
                        old_walk = self.shortest_path(
                            *old_keys,
                            avoid=[
                                a
                                for a in self.center
                                if any(k.is_connected(a) for k in old_keys)
                            ]
                        )
                        remove_centers = [c for c in self.center if c in old_walk]
                
                else:
                    old_walk = [a for a in old_keys]

            if old_walk and len(old_walk) == 2:
                old_con = set([])
                for k in old_keys:
                    for c in k.connected:
                        old_con.add(c)
                if old_con:
                    old_vec = old_ligand.COM(targets=old_con) - center
                else:
                    old_vec = center

            elif old_walk:
                old_vec = old_ligand.COM(targets=old_walk[1:-1]) - center
            else:
                old_vec = np.zeros(3)
                for atom in old_keys:
                    if any(
                        bonded_atom not in old_ligand.atoms
                        for bonded_atom in atom.connected
                    ):
                        v = atom.coords - self.COM(
                            targets=[
                                bonded_atom
                                for bonded_atom in atom.connected
                                if bonded_atom not in old_ligand.atoms
                            ]
                        )
                        v /= np.linalg.norm(v)
                        old_vec += v
                        # print(atom)

                # print("vec:", old_vec)
                old_vec /= np.linalg.norm(old_vec)

            new_walk = ligand.shortest_path(*new_keys)
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

            return remove_centers

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
            frag_list = ligand.get_frag_list(max_order=1)
            # ligand.write("ligand")

            remove_centers = []

            # get key atoms on each side of rotatable bonds
            key_count = {}
            for frag, a, b in frag_list:
                n_keys = []
                for i in frag:
                    if i not in ligand.key_atoms:
                        continue
                    n_keys += [i]
                if len(n_keys) < 1 or len(n_keys) > 2:
                    continue
                if a in ligand.key_atoms or b in ligand.key_atoms:
                    continue
                if utils.same_cycle(ligand, a, b):
                    continue
                if len(n_keys) not in key_count:
                    key_count[len(n_keys)] = [(frag, a, b)]
                else:
                    key_count[len(n_keys)] += [(frag, a, b)]

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
                    remove_centers.extend(
                        map_2_key(old_ligand, ligand, ok, nk)
                    )
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

            return remove_centers

        if not self.components:
            self.detect_components()

        # find old and new keys
        old_keys = self.find(old_keys)
        if isinstance(ligands, (str, Geometry)):
            ligands = [ligands]
        new_keys = []
        for i, ligand in enumerate(ligands):
            if not isinstance(ligand, AaronTools.component.Component):
                ligand = AaronTools.component.Component(ligand)
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

        old_ligand = []
        remove_components = []
        for k in old_keys:
            for i, c in enumerate(self.components):
                if k in c.atoms and k not in old_ligand:
                    old_ligand.extend(c.atoms)
                    remove_components.append(i)

        for i in remove_components:
            self.components.pop(i)
            for j in range(0, len(remove_components)):
                if remove_components[j] > i:
                    remove_components[j] -= 1

        old_ligand = Geometry(old_ligand)

        start = 0
        end = None
        remove_centers = []
        for i, ligand in enumerate(ligands):
            end = start + len(ligand.key_atoms)
            if len(ligand.key_atoms) == 1:
                map_1_key(self, ligand, old_keys[start], new_keys[start])
            elif len(ligand.key_atoms) == 2:
                remove_centers.extend(
                    map_2_key(
                        old_ligand,
                        ligand,
                        old_keys[start:end],
                        new_keys[start:end],
                    )
                )
            else:
                remove_centers.extend(
                    map_more_key(
                        self,
                        old_ligand,
                        ligand,
                        old_keys[start:end],
                        new_keys[start:end],
                    )
                )

            for a in ligand.atoms:
                a.name = old_keys[start].name + "." + a.name
                a.add_tag("ligand")
            start = end

        for atom in self.atoms:
            if atom.connected & set(old_ligand.atoms):
                atom.connected = atom.connected - set(old_ligand.atoms)

        # remove extraneous centers, i.e. from ferrocene ligands
        for rc in remove_centers:
            self.center.remove(rc)

        # add new
        for ligand in ligands:
            self.components += [ligand]
        rv = ligands

        self.rebuild()
        # rotate monodentate to relieve clashing
        for ligand in self.components:
            if len(ligand.key_atoms) == 1:
                targets = ligand.atoms
                key = ligand.key_atoms[0]
                if self.center:
                    start = self.COM(self.center)
                    end = key.coords
                else:
                    start = key.coords
                    end = self.COM(key.connected)
                axis = end - start
                self.minimize_torsion(targets, axis, center=key, increment=10)

        self.remove_clash()
        if minimize:
            self.minimize()

        self.refresh_ranks()

        return rv

    def remove_clash(self, sub_list=None):
        def get_clash(sub, scale):
            """
            Returns: np.array(bend_axis) if clash found, False otherwise
            """

            clashing = []
            D = distance_matrix(self.coords, sub.coords)
            for i, atom in enumerate(self.atoms):
                if atom in sub.atoms or atom == sub.end:
                    continue
                threshold = atom._radii
                for j, sub_atom in enumerate(sub.atoms):
                    threshold += sub_atom._radii
                    threshold *= scale
                    dist = D[i, j]
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
                self.LOG.warning(
                    "Bad conformer number given: {} {} > {}".format(
                        sub.name, conf_num, sub.conf_num
                    )
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
