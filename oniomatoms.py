import numpy as np
from AaronTools.atoms import Atom
from AaronTools.const import ATOM_TYPES, CONNECTIVITY, EIJ, ELEMENTS, MASS, RADII, RIJ, ATOM_TYPES

class OniomAtom(Atom):
    def __init__(self, element="", coords=[], flag=False, name="", tags=[], layer="", atomtype="", charge="", link_info={}, atom=None):
        super().__init__(element="", coords=[], flag=False, name="", tags=[])

        if atom != None and type(atom) == Atom:
            self.element = atom.element
            self.coords = atom.coords
            self.flag = atom.flag
            self.name = atom.name
            self.tags = atom.tags
            self.charge = atom.charge

        element = str(element).strip().capitalize()
        if element == "" and atom == None:
            self.element = element
            self._radii = None
            self._connectivity = None
        elif element in ELEMENTS:
            self.element = element
            self._set_radii()
            self._set_connectivity()
        elif element == "" and atom != None:
            pass
        else:
            raise ValueError("Unknown element detected:", element)

        if atom == None:
            self.coords = np.array(coords, dtype=float)
        if atom != None and coords != []:
            self.coords = np.array(coords, dtype=float)
        if atom == None:
            self.flag = bool(flag)
        if atom == None or (atom != None and name != ""):
            self.name = str(name).strip()
        try:
            self.index = int(self.name)
        except ValueError:
            pass

        if hasattr(tags, "__iter__") and not isinstance(tags, str) and atom == None:
            self.tags = tags
        elif atom != None:
            self.add_tag(tags)
        else:
            self.tags = set([tags])

        self.link_info = link_info

        if atom == None:
            self.connected = set([])
            self.constraint = set([])
            self._rank = None

        if atom != None:
            if atom.connected == []:
                self.connected = set([])
            else:
                self.connected = atom.connected
            if atom.constraint == []:
                self.constraint == set([])
            else:
                self.contraint = atom.constraint
            if atom._rank == None:
                self._rank = None
            else:
                self._rank = atom._rank 

        charge=str(charge).strip()
        if charge == "":
            pass
        else:
            self.charge = float(charge)

        atomtype = str(atomtype).strip()
        layer=str(layer).strip().capitalize()
        if layer == "" and atom==None:
            self.layer=layer
        elif layer == "" and atom != None:
            if hasattr(atom, "layer"):
                self.layer = atom.layer
            else:
                self.layer = layer
        elif layer not in ['H', 'L', 'M']:
            raise ValueError("Incorrect symbol for layer: " + layer)
        else:
            self.layer=layer

        if atomtype == "" and atom==None:
            self.atomtype=atomtype
        elif atomtype == "" and atom != None:
            pass
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

    def __gt__(self, other):
        """ sorts by the layer the atom is in, with atoms in High layer considered greater than those in Medium and Low"""
        if self.layer == "H" and other.layer != "H":
            return True
        elif self.layer == "M" and other.layer == "L":
            return True
        else:
            return False

    @classmethod
    def from_atom(cls, atom):
        """creates a new OniomAtom object from an existing Atom or OniomAtom object"""
        if not isinstance(atom, Atom):
            raise ValueError("atom must be an Atom or OniomAtom object")

        oniom_atom = OniomAtom(element=atom.element, coords = atom.coords)
        for attr in ("atomtype", "layer", "tags", "flag", "name", "charge"):
            if hasattr(atom, attr):
                oniom_atom.attr = atom.attr
        return oniom_atom

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

