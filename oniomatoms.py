import numpy as np
from AaronTools.atoms import Atom
from AaronTools.const import ATOM_TYPES, CONNECTIVITY, EIJ, ELEMENTS, MASS, RADII, RIJ

class OniomAtom(Atom):
    def __init__(self, element="", coords=[], flag=False, name="", tags=[], layer=None, atomtype=None, charge=None, link_info=None, res = "", atom=None):
        super().__init__(element="", coords=[], flag=False, name="", tags=[])

        atomtype = str(atomtype).strip()
        layer=str(layer).strip().upper()

        if atom != None and isinstance(atom, Atom):
            self.element = atom.element
            self.coords = atom.coords
            self.flag = atom.flag
            self.name = atom.name
            self.tags = atom.tags
            if hasattr(atom, "layer") and not layer:
                self.layer = atom.layer
            else:
                self.layer = layer
            if hasattr(atom, "charge") and not charge:
                self.charge = atom.charge
            else:
                self.charge = float(charge)
            if hasattr(atom, "atomtype") and not atomtype:
                self.atomtype = atom.atomtype
            else:
                self.atomtype = atomtype
            if hasattr(atom, "link_info") and link_info == None:
                self.link_info = atom.link_info
            elif not hasattr(atom, "link_info") and link_info == None:
                self.link_info = dict()
            else:
                self.link_info = link_info
            if coords:
                self.coords = np.array(coords, dtype=float)
            if name:
                self.name = str(name).strip()
            if tags:
                self.add_tag(tags)
            if atom.connected == []:
                self.connected = set([])
            else:
                self.connected = atom.connected # change the connected atoms to OniomAtom type
            if atom.constraint == []:
                self.constraint == set([])
            else:
                self.contraint = atom.constraint
            if atom._rank == None:
                self._rank = None
            else:
                self._rank = atom._rank
            if hasattr(atom, "res") and not res:
                self.res = atom.res
            else:
                self.res = res

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
            self.flag = bool(flag)
            self.name = str(name).strip()
            if hasattr(tags, "__iter__") and not isinstance(tags, str) and atom == None:
                self.tags = tags
            else:
                self.tags = set([tags])
            if link_info == None:
                self.link_info = dict()
            else:
                self.link_info = link_info
            self.connected = set([])
            self.constraint = set([])
            self._rank = None
            self.res = res
            #charge=str(charge).strip()
            if charge == "" or charge == None:
                pass
            else:
                self.charge = float(str(charge).strip())
            self.layer=layer
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
        if not isinstance(other, OniomAtom):
            raise TypeError("cannot compare atoms of different type")

        if self.layer == "H" and other.layer != "H":
            return True
        elif self.layer == "M" and other.layer not in ("H", "M"):
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
            if hasattr(atom,"atomtype"):
                counter+=1
                if atom.atomtype in ATOM_TYPES:
                    continue
                else:
                    return False
        return counter == len(atomlist)

    @classmethod
    def check_layers(cls, atomlist):
        """returns True if layers are defined correctly as either H, L, or M"""
        counter=0
        for atom in atomlist:
            if hasattr(atom,"layer"):
                counter+=1
                if atom.layer in ("H", "M", "L"):
                    continue
                else:
                    return False
        return counter==len(atomlist)

