#!/usr/bin/env python3

import itertools
import os
import re
from glob import glob
from urllib.request import urlopen
from urllib.error import HTTPError

from AaronTools.const import AARONLIB, QCHASM
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry


class RingFragment(Geometry):
    """
    Attributes:
        name
        atoms
        end
    """

    AARON_LIBS = os.path.join(AARONLIB, "RingFrags", "*.xyz")
    BUILTIN = os.path.join(QCHASM, "AaronTools", "RingFragments", "*.xyz")

    def __init__(
        self,
        frag,
        name=None,
        end=None
    ):
        """
        frag is either a file sub, a geometry, or an atom list
        """

        if isinstance(frag, (Geometry, list)):
            # we can create substituent object from fragment
            if isinstance(frag, RingFragment):
                self.name = name if name else frag.name
                self.end = end if end else frag.end 
            elif isinstance(frag, Geometry):
                self.name = name if name else frag.name
                self.end = end if end else frag.end 
            else:
                self.name = name

            try:
                self.atoms = frag.atoms
            except AttributeError:
                self.atoms = frag
        
        else:  # or we can create from file
            # find substituent xyz file
            fsub = None
            for f in glob(RingFragment.AARON_LIBS) + glob(RingFragment.BUILTIN):
                match = frag + ".xyz" == os.path.basename(f)
                if match:
                    fsub = f
                    break
            # or assume we were given a file name instead
            if not fsub and ".xyz" in frag:
                fsub = frag
                frag = os.path.basename(frag).rstrip(".xyz")

            if fsub is None:
                raise RuntimeError("substituent name not recognized: %s" % fsub)

            # load in atom info
            from_file = FileReader(fsub)
            self.name = frag
            self.comment = from_file.comment
            self.atoms = from_file.atoms
            self.refresh_connected(rank=False)

            end_info = re.search("E:(\d+)", self.comment)
            if end_info is not None:
                self.end = [self.find(end)[0] for end in re.findall('\d+', self.comment)]
            else:
                self.end = None

    def find_end(self, path_length, start=[]):
        """finds a path around self that is path_length long and starts with start"""
        
        def linearly_connected(atom_list):
            """returns true if every atom in atom_list is connected to another atom in 
            the list without backtracking"""
            #start shouldn't be end
            if atom_list[0] == atom_list[-1]:
                return False

            #first and second atoms should be bonded
            elif atom_list[0] not in atom_list[1].connected:
                return False

            #last and second to last atoms should be bonded
            elif atom_list[-1] not in atom_list[-2].connected:
                return False
            
            #all other atoms should be conneced to exactly 2 atoms
            elif any([sum([atom1 in atom2.connected for atom2 in atom_list]) != 2 for atom1 in atom_list[1:-1]]):
                return False

            #first two atoms should only be connected to one atom
            elif sum([sum([atom_list[0] in atom.connected]) + \
                      sum([atom_list[-1] in atom.connected]) for atom in atom_list]) \
                        > 2:
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
            if atom not in start:
                if hasattr(atom, '_connectivity'):
                    if atom._connectivity > 1:
                        usable_atoms.append(atom)
                else:
                    usable_atoms.append(atom)

        for path in itertools.permutations(usable_atoms, path_length - len(start_atoms)):
            full_path = start_atoms + list(path)
            if linearly_connected(full_path):
                self.end = list(path)
                break

        if self.end is None:
            raise LookupError("unable to find %i long path starting with %s around %s" % (path_length, start, self.name))

    @classmethod
    def from_string(cls, name, end=None, form='smiles'):
        """create ring fragment from string"""

        ring = Geometry.from_string(name, form)
        if end is not None:
            if isinstance(end, int):
                end = RingFragment.find_end(ring, end)
                return RingFragment(ring, name=name)
            elif isinstance(end, list):
                return RingFragment(ring, end=end, name=name)
            else:
                raise ValueError("expected int or list for 'end' in 'from_string', got %s", str(end))
        else:
            return RingFragment(ring, name=name)

