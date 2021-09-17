import numpy as np
from AaronTools.atoms import Atom
from AaronTools.const import ATOM_TYPES, CONNECTIVITY, EIJ, ELEMENTS, MASS, RADII, RIJ, ATOM_TYPES

class OniomAtom(Atom):
    def __init__(self, element="", coords=[], flag=False, name="", tags=[], layer="", atomtype="", charge=""):
        super().__init__(element="", coords=[], flag=False, name="", tags=[])

        element = str(element).strip().capitalize()
        if element == "":
            self.element = element
            self._radii = None
            self._connectivity = None
        elif element in ELEMENTS:
            self.element = element
            self._set_radii()
            self._set_connectivity()
        else:
            raise ValueError("Unknown element detected:", element)

        self.coords = np.array(coords, dtype=float)
        self.flag = bool(flag)
        self.name = str(name).strip()
        try:
            self.index = int(self.name)
        except ValueError:
            pass

        if hasattr(tags, "__iter__") and not isinstance(tags, str):
            self.tags = set(tags)
        else:
            self.tags = set([tags])

        self.connected = set([])
        self.constraint = set([])
        self._rank = None
        charge=str(charge).strip()
        if charge == "":
            pass
        else:
            self.charge = float(charge)

        atomtype = str(atomtype).strip()
        layer=str(layer).strip().capitalize()
        if layer == "":
            self.layer=layer
        elif layer not in ['H', 'L', 'M']:
            raise ValueError("Incorrect symbol for layer: " + layer)
        else:
            self.layer=layer

        if atomtype == "":
            self.atomtype=atomtype
        elif atomtype not in ATOM_TYPES:
            self.atomtype=atomtype
            #raise ValueError("Atom type " + self.atomtype + " not in reference")
        else:
            self.atomtype=atomtype

    def __repr__(self):
        s = ""
        s += "{:>3s}  ".format(self.element)
        for c in self.coords:
            s += "{: 13.8f} ".format(c)
        s += " {: 2d}  ".format(-1 if self.flag else 0)
        s += "{}".format(self.name)
        s += "   {}".format(self.layer)
        try:
            s += "   {}".format(self.atomtype)
        except AttributeError:
            pass
        try:
            s += "   {}".format(self.charge)
        except AttributeError:
            pass
        try:
            s += "   {}".format(self.tags)
        except AttributeError:
            pass
        return s

    def get_layer(self):
        if self.layer not in ['H', 'L', 'M']:
            raise ValueError("Layer set to High. Incorrect symbol for layer: " + self.layer)
        else:
            return self.layer

    def get_charge(self):
        return self.charge

    def get_at(self):
        return self.atomtype
        if self.atomtype not in ATOM_TYPES:
            raise ValueError("Atom type " + self.atomtype + " not in reference")

    def change_layer(self, newlayer):
        self.layer = str(newlayer).strip().capitalize()
 
    def change_atomtype(self, newtype):
        self.atomtype = str(newtype).strip()
 
    def change_charge(self, newcharge):
        self.charge[0] = float(newcharge)

    def __gt__(self, other):
        if not isinstance(other, OniomAtom):
            raise TypeError("cannot compare atoms of different type")
        if self.layer == "H" and self.layer != other.layer:
            return True
        elif self.layer == "M" and other.layer == "L":
            return True
        else:
            return False

class ChargeSum:
    def __init__(self):
        pass

    @classmethod
    def get(cls, atomlist):
        """sums charges in list of atoms of OniomAtom type."""
        totalcharge=[]
        if isinstance(atomlist, (list, tuple, np.ndarray)):
            for atom in atomlist:
                if isinstance(atom, OniomAtom):
                    totalcharge=np.append(totalcharge, atom.charge)
                else:
                    raise TypeError("each atom in list must be an OniomAtom")
            return np.sum(totalcharge)
        else:
            raise TypeError("function only accepts a list, tuple, or array of atoms")

class OniomSanity:
    def __init__(self):
        pass
    @classmethod
    def check_charges(cls, atomlist, targetcharge, threshold=""):
        """returns True if sum of charges is equal to expected value within threshold"""
        if threshold == "":
            threshold=float(10**(-1))
        else:
            threshold=float(threshold)
        targetcharge = float(targetcharge)
        return abs(ChargeSum.get(atomlist)-targetcharge) < threshold

    @classmethod
    def check_types(cls,atomlist):
        """returns True if atomtypes in atomlist are all in reference"""
        counter = 0
        for atom in atomlist:
            if atom.get_at():
                counter+=1
        return counter == len(atomlist)

    @classmethod
    def check_layers(cls, atomlist):
        """returns True if layers are defined correctly as either H, L, or M"""
        counter=0
        for atom in atomlist:
            if atom.get_layer():
                counter+=1
        return counter==len(atomlist)

