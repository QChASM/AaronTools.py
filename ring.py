#!/usr/bin/env python3

import itertools
import os
import re
from glob import glob

from AaronTools.const import AARONLIB, AARONTOOLS
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry


class Ring(Geometry):
    """
    Attributes:
        name
        atoms
        end
    """

    AARON_LIBS = os.path.join(AARONLIB, "Rings")
    BUILTIN = os.path.join(AARONTOOLS, "Rings")

    def __init__(self, frag, name=None, end=None):
        """
        frag is either a file sub, a geometry, or an atom list
        name is a name
        end is a list of atoms that defines which part of the ring is not part of the fragment
        """

        super().__init__()
        if isinstance(frag, (Geometry, list)):
            # we can create ring object from a geometry
            if isinstance(frag, Ring):
                self.name = name if name else frag.name
                self.end = end if end else frag.end
            elif isinstance(frag, Geometry):
                self.name = name if name else frag.name
                self.end = end if end else None
            else:
                self.name = name

            try:
                self.atoms = frag.atoms
            except AttributeError:
                self.atoms = frag

        else:  # or we can create from file
            # find ring xyz file
            fring = None
            for lib in [Ring.AARON_LIBS, Ring.BUILTIN]:
                if not os.path.exists(lib):
                    continue
                for f in os.listdir(lib):
                    name, ext = os.path.splitext(f)
                    if not any(".%s" % x == ext for x in read_types):
                        continue
                    match = frag == name
                    if match:
                        fring = os.path.join(lib, f)
                        break
                
                if fring:
                    break
            # or assume we were given a file name instead
            if not fring and ".xyz" in frag:
                fring = frag
                frag = os.path.basename(frag).rstrip(".xyz")

            if fring is None:
                raise RuntimeError("ring name not recognized: %s" % frag)

            # load in atom info
            from_file = FileReader(fring)
            self.name = frag
            self.comment = from_file.comment
            self.atoms = from_file.atoms
            self.refresh_connected()

            end_info = re.search("E:(\d+)", self.comment)
            if end_info is not None:
                self.end = [
                    self.find(end)[0]
                    for end in re.findall("\d+", self.comment)
                ]
            else:
                self.end = None

    @classmethod
    def from_string(cls, name, end_length, end_atom=None, form="smiles"):
        """create ring fragment from string
        name        str         identifier for ring
        end_length  int         number of atoms in ring end
        end_atom    identifiers identifier for ring end
        form        str         type of identifier (smiles, iupac)
        """

        ring = Geometry.from_string(name, form)
        if end_atom is not None and end_length is not None:
            ring = cls(ring)
            end_atom = ring.find(end_atom)[0]
            ring.find_end(end_length, end_atom)
            return ring

        elif end_length is not None:
            ring = cls(ring)
            ring.find_end(end_length)
            return ring

        else:
            return cls(ring, name=name)

    @classmethod
    def list(cls, include_ext=False):
        names = []
        for lib in [cls.AARON_LIBS, cls.BUILTIN]:
            if not os.path.exists(lib):
                continue
            for f in os.listdir(lib):
                name, ext = os.path.splitext(os.path.basename(f))
                if not any(".%s" % x == ext for x in read_types):
                    continue
                
                if name in names:
                    continue
                
                if include_ext:
                    names.append(name + ext)
                else:
                    names.append(name)

        return names

    def copy(self):
        dup = super().copy()
        dup.end = dup.find([atom.name for atom in self.end])
        return dup

    def find_end(self, path_length, start=[]):
        """finds a path around self that is path_length long and starts with start"""
        def linearly_connected(atom_list):
            """returns true if every atom in atom_list is connected to another atom in
            the list without backtracking"""
            # start shouldn't be end
            if atom_list[0] == atom_list[-1]:
                return False

            # first and second atoms should be bonded
            elif atom_list[0] not in atom_list[1].connected:
                return False

            # last and second to last atoms should be bonded
            elif atom_list[-1] not in atom_list[-2].connected:
                return False

            # all other atoms should be conneced to exactly 2 atoms
            elif any(
                [
                    sum([atom1 in atom2.connected for atom2 in atom_list]) != 2
                    for atom1 in atom_list[1:-1]
                ]
            ):
                return False

            # first two atoms should only be connected to one atom unless they are connected to each other
            elif (
                sum(
                    [
                        sum([atom_list[0] in atom.connected])
                        + sum([atom_list[-1] in atom.connected])
                        for atom in atom_list
                    ]
                )
                > 2
                and atom_list[0] not in atom_list[-1].connected
            ):

                return False

            else:
                return True

        self.end = None

        if start:
            start_atoms = self.find(start)
        else:
            start_atoms = []

        usable_atoms = []
        for atom in self.atoms:
            if atom not in start_atoms:
                if hasattr(atom, "_connectivity"):
                    if atom._connectivity > 1:
                        usable_atoms.append(atom)
                else:
                    usable_atoms.append(atom)

        for path in itertools.permutations(
            usable_atoms, path_length - len(start_atoms)
        ):
            full_path = start_atoms + list(path)
            if linearly_connected(full_path) or path_length == 1:
                self.end = list(full_path)
                break

        if self.end is None:
            raise LookupError(
                "unable to find %i long path starting with %s around %s"
                % (path_length, start, self.name)
            )

        else:
            self.comment = "E:" + ",".join(
                [str(self.atoms.index(a) + 1) for a in self.end]
            )
