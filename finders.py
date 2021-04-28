"""finders are used by Geometry.find to locate atoms in a more general way"""
import numpy as np


class Finder:
    def get_matching_atoms(self, atoms, geometry=None):
        """overwrite with function that returns list(Atom) of the atoms that
        match your Finder's criteria
        geometry is an optional argument that could be used to e.g. find
        atoms a certain number of bonds"""
        pass


class BondsFrom(Finder):
    """exact number of bonds from specified atom
    avoid: bonding path cannot pass through these atoms"""
    def __init__(self, atom, number_of_bonds, avoid=None):
        super().__init__()

        self.central_atom = atom
        self.n_bonds = number_of_bonds
        self.avoid = avoid

    def __repr__(self):
        return "atoms %i bonds of %s" % (self.n_bonds, self.central_atom)

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        for atom in atoms:
            try:
                path = geometry.shortest_path(atom, self.central_atom, avoid=self.avoid)
            except LookupError:
                continue

            if len(path) - 1 == self.n_bonds:
                matching_atoms.append(atom)

        return matching_atoms


class WithinBondsOf(BondsFrom):
    """within a specified number of bonds from the atom"""
    def __init__(self, atom, number_of_bonds):
        super().__init__(atom, number_of_bonds)

    def __repr__(self):
        return "atoms %i bonds of %s" % (self.n_bonds, self.central_atom)

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        for atom in atoms:
            try:
                path = geometry.shortest_path(atom, self.central_atom)
            except LookupError:
                continue

            if len(path) - 1 <= self.n_bonds and len(path) > 1:
                matching_atoms.append(atom)

        return matching_atoms


class BondedTo(Finder):
    """returns all atoms that are bonded to the specified atom"""
    def __init__(self, atom):
        super().__init__()

        self.atom = atom

    def __repr__(self):
        return "atoms bonded to %s" % self.atom

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        return [atom for atom in atoms if atom in self.atom.connected]


class WithinRadiusFromPoint(Finder):
    """within a specified radius of a point"""
    def __init__(self, point, radius):
        super().__init__()

        self.point = np.array(point)
        self.radius = radius
    
    def __repr__(self):
        return "atoms within %.2f angstroms of (%.2f, %.2f, %.2f)" % (self.radius, *self.point)

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        matching_atoms = []
        for atom in atoms:
            coords = atom.coords
            d = np.linalg.norm(coords - self.point)
            if d < self.radius:
                matching_atoms.append(atom)

        return matching_atoms


class WithinRadiusFromAtom(Finder):
    """within a specified radius of a point"""
    def __init__(self, atom, radius):
        super().__init__()

        self.atom = atom
        self.radius = radius
    
    def __repr__(self):
        return "atoms within %.2f angstroms of %s" % (self.radius, self.atom)

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        matching_atoms = []
        for atom in atoms:
            coords = atom.coords
            d = self.atom.dist(atom)
            if d < self.radius:
                matching_atoms.append(atom)

        return matching_atoms
    

class NotAny(Finder):
    """atoms not matching specifiers/Finders"""
    def __init__(self, *args):
        """args can be any number of Finders and/or other atom specifiers (tags, elements, etc.)"""
        super().__init__()

        self.critera = args

    def __repr__(self):
        return "not any of: %s" % ", ".join([str(x) for x in self.critera])

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that do not match any of the critera"""
        unmatching_atoms = []
        for criterion in self.critera:
            try:
                unmatch = geometry.find(criterion)
                unmatching_atoms.extend(unmatch)
            
            except LookupError:
                pass

        return [atom for atom in atoms if atom not in set(unmatching_atoms)]


class AnyTransitionMetal(Finder):
    """any atoms that are transition metals"""
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "any transition metal"

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that are metals"""
        from AaronTools.const import TMETAL
        return [atom for atom in atoms if atom.element in TMETAL]


class AnyNonTransitionMetal(NotAny):
    """any atoms that are not transition metals"""
    def __init__(self):
        super().__init__(AnyTransitionMetal())

    def __repr__(self):
        return "any non-transition metal"


class HasAttribute(Finder):
    """all atoms with the specified attribute"""
    def __init__(self, attribute):
        super().__init__()

        self.attribute_name = attribute

    def __repr__(self):
        return "atoms with the '%s' attribute" % self.attribute_name

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that have the attribute"""
        return [atom for atom in atoms if hasattr(atom, self.attribute_name)]


class VSEPR(Finder):
    """atoms with the specified VSEPR geometry
    see Atom.get_shape for a list of valid vsepr_geometry strings"""
    def __init__(self, vsepr_geometry):
        super().__init__()
        
        self.vsepr = vsepr_geometry
    
    def __repr__(self):
        return "atoms with %s shape" % self.vsepr

    def get_matching_atoms(self, atoms, geometry=None):
        matching_atoms = []
        for atom in atoms:
            out = atom.get_vsepr()
            if out is not None:
                shape, score = atom.get_vsepr()
                if shape == self.vsepr and score < 0.5:
                    matching_atoms.append(atom)
        
        return matching_atoms


class BondedElements(Finder):
    """atoms bonded to the specified neighboring elements
    if match_exact=True (default), elements must match exactly 
    e.g. BondedElements('C') will find
    atoms bonded to only one carbon and nothing else"""
    def __init__(self, *args, match_exact=True):
        super().__init__()
        
        self.elements = list(args)
        self.match_exact = match_exact
        
    def __repr__(self):
        if len(self.elements) == 0:
            return "atoms bonded to nothing"
        elif len(self.elements) == 1:
            return "atoms bonded to %s" % self.elements[0]
        else:
            return "atoms bonded to %s and %s" % (", ".join(self.elements[:-1]), self.elements[-1])
    
    def get_matching_atoms(self, atoms, geometry=None):
        matching_atoms = []
        if self.match_exact:
            ref = "".join(sorted(self.elements))
        else:
            ref = self.elements
        
        for atom in atoms:
            if self.match_exact:
                ele_list = [a.element for a in [ele for ele in atom.connected]]
                test = "".join(sorted(ele_list))
                if ref == test:
                    matching_atoms.append(atom)
            
            else:
                bonded_eles = [bonded_atom.element for bonded_atom in atom.connected]
                if all([ele in bonded_eles for ele in self.elements]):
                    matching_atoms.append(atom)
        
        return matching_atoms


class NumberOfBonds(Finder):
    """atoms with the specified number of bonds"""
    def __init__(self, num_bonds):
        super().__init__()
        
        self.num_bonds = num_bonds
    
    def __repr__(self):
        return "atoms with %i bonds" % self.num_bonds
    
    def get_matching_atoms(self, atoms, geometry=None):
        return [atom for atom in atoms if len(atom.connected) == self.num_bonds]


class ChiralCentres(Finder):
    """chiral centers
    atoms with a non-planar VSEPR geometry with all bonded groups
    being distinct
    for rings, looks for a set of unique canonical ranks for atoms that 
    are all the same number of bonds away from one atom"""
    #IUPAC spelling 
    def __init__(self, RS_only=False):
        """RS_only: bool  - if True, do not identify chiral centers that are chiral because they
                            are connected to multiple chiral fragments with the same chirality
                            this corresponds to R/S centers
                            False will include r/s centers as well
        """
        super().__init__()
        self.cip = RS_only

    def __repr__(self):
        return "chiral centers"

    def get_matching_atoms(self, atoms, geometry):
        from AaronTools.geometry import Geometry

        matching_atoms = []

        # b/c they are connected to chiral fragments
        geometry.refresh_ranks()
        chiral_atoms_changed = True
        ranks = geometry.canonical_rank(break_ties=False, update=False)
        frags = []
        for atom in geometry.atoms:
            frags.append([])
            for bonded_atom in atom.connected:
                frags[-1].append(geometry.get_fragment(bonded_atom, atom, as_object=False))

        #need to do multiple passes b/c sometimes atoms are chiral
        k = 0
        while chiral_atoms_changed:
            chiral_atoms_changed = False
            k += 1
            #skip atoms we've already found
            for ndx, atom in enumerate(atoms):
                if atom in matching_atoms:
                    continue

                #can't be chiral with 2 bonds
                if len(atom.connected) < 3:
                    continue

                #planar vsepr don't get checked
                vsepr = atom.get_vsepr()
                if vsepr is not None:
                    shape, score = vsepr
                    if shape in  ['trigonal planar', 't shaped', 'sqaure planar']:
                        continue

                chiral = True
                for i, frag1 in enumerate(frags[ndx]):
                    #get the ranks of the atoms in this fragment
                    ranks_1 = [ranks[geometry.atoms.index(atom)] for atom in frag1]
                    for j, frag2 in enumerate(frags[ndx][:i]):
                        same = True
                        
                        ranks_2 = [ranks[geometry.atoms.index(atom)] for atom in frag2]
                        
                        if len(frag1) != len(frag2):
                            same = False
                            continue

                        for a, b in zip(sorted(ranks_1), sorted(ranks_2)):
                            # want correct elements
                            if a != b:
                                same = False
                                break
                           
                        for a, b in zip(sorted(frag1), sorted(frag2)):
                            # and other chiral atoms
                            if not self.cip and a in matching_atoms and b in matching_atoms:
                                #use RMSD to see if they have the same handedness
                                a_connected = sorted(a.connected)
                                b_connected = sorted(b.connected)
                                a_targets = [a] + list(a_connected)
                                b_targets = [b] + list(b_connected)
                                if geometry.RMSD(geometry, targets=a_targets, ref_targets=b_targets, sort=False, align=False) < 0.1:
                                    same = False
                                    break

                            # and correct connected elements
                            for i, j in zip(
                                sorted([aa.element for aa in a.connected]),
                                sorted([bb.element for bb in b.connected]),
                            ):
                                if i != j:
                                    same = False
                                    break

                        
                        ring_atoms = [bonded_atom for bonded_atom in atom.connected if bonded_atom in frag1 and bonded_atom in frag2]
                        if len(ring_atoms) > 0:
                            #this is a ring
                            #look at the rank of all atoms that are n bonds away from this atom
                            #if the ranks are ever all different, this is a chiral center
                            n_bonds = 1
                            acceptable_nbonds = True
                            while acceptable_nbonds:
                                try:
                                    atoms_within_nbonds = geometry.find(BondsFrom(atom, n_bonds))
                                    nbonds_ranks = [ranks[geometry.atoms.index(a)] for a in atoms_within_nbonds]
                                    if all([nbonds_ranks.count(r) == 1 for r in nbonds_ranks]):
                                        same = False
                                        acceptable_nbonds = False
                                    elif not self.cip:
                                        #need to find things in the ring that are chiral b/c of other chiral centers
                                        for i, atom1 in enumerate(atoms_within_nbonds):
                                            for j, atom2 in enumerate(atoms_within_nbonds[i+1:]):
                                                k = j + i + 1
                                                if nbonds_ranks[i] == nbonds_ranks[k]:
                                                    a_connected = sorted(atom1.connected)
                                                    b_connected = sorted(atom2.connected)
                                                    a_targets = [atom1] + list(a_connected)
                                                    b_targets = [atom2] + list(b_connected)
                                                    if geometry.RMSD(geometry, targets=a_targets, ref_targets=b_targets, sort=False, align=False) < 0.1:
                                                        same = False
                                                        break
                                        if not same:
                                            break

                                    n_bonds += 1
                                except LookupError:
                                    acceptable_nbonds = False

                            if not same:
                                break


                        if same:
                            chiral = False
                            break

                if chiral:
                    chiral_atoms_changed = True
                    matching_atoms.append(atom)
        
        return matching_atoms

#alternative spelling
ChiralCenters = ChiralCentres


class FlaggedAtoms(Finder):
    """
    atoms with a non-zero flag
    """
    # useful for finding constrained atoms
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "flagged atoms"

    def get_matching_atoms(self, atoms, geometry):
        return [atom for atom in atoms if atom.flag]


class CloserTo(Finder):
    """
    atoms closer to atom1 than atom2 (based on bonds, not actual distance)
    """
    def __init__(self, atom1, atom2, include_ties=False):
        super().__init__()
        
        self.atom1 = atom1
        self.atom2 = atom2
        self.include_ties = include_ties
    
    def __repr__(self):
        return "atoms closer to %s than %s" % (self.atom1, self.atom2)
    
    def get_matching_atoms(self, atoms, geometry):
        matching_atoms = []
        for atom in atoms:
            if atom is self.atom1 and atom is not self.atom2:
                matching_atoms.append(atom)
                continue
                
            try:
                d1 = len(geometry.shortest_path(self.atom1, atom))
            except LookupError:
                d1 = False
            
            try:
                d2 = len(geometry.shortest_path(self.atom2, atom))
            except LookupError:
                d2 = False
            
            if d1 is not False and d2 is not False and d1 <= d2:
                if self.include_ties:
                    matching_atoms.append(atom)
                elif d1 < d2:
                    matching_atoms.append(atom)
            
            elif d1 is not False and d2 is False:
                matching_atoms.append(atom)
        
        return matching_atoms


class AmideCarbon(Finder):
    """
    amide carbons
    trigonal planar carbons bonded to a linear oxygen and a
    nitrogen with 3 bonds
    """
    def __repr__(self):
        return "amide carbons"

    def get_matching_atoms(self, atoms, geometry):
        from AaronTools.atoms import BondOrder
        matching_atoms = []
        
        carbons = geometry.find("C", VSEPR("trigonal planar"))
        oxygens = geometry.find("O", VSEPR("linear 1"))
        nitrogens = geometry.find("N", NumberOfBonds(3))
        for carbon in carbons:
            if (
                    any(atom in oxygens for atom in carbon.connected)
                    and any(atom in nitrogens for atom in carbon.connected)
            ):
                matching_atoms.append(carbon)
        
        return matching_atoms


class Bridgehead(Finder):
    """
    bridgehead atoms
    can specify ring sizes that the atoms bridge
    """
    def __init__(self, ring_sizes=None, match_exact=False):
        """
        ring_sizes  - list of int, size of rings (e.g. [6, 6] for atoms that bridge
                      two 6-membered rings)
                      not specifying yields bridgehead atoms for any ring size
        match_exact - bool, if True, return atoms only bridging the specified rings
                      if False, the ring_sizes is taken as a minimum (e.g. 
                      ring_size=[6, 6], match_exact=False would also yield atoms
                      bridging three 6-membered rings or two six-membered rings and
                      a five-membered ring)
        """
        self.ring_sizes = ring_sizes
        self.match_exact = match_exact
    
    def __repr__(self):
        if self.ring_sizes:
            return "atoms that bridge %s-member rings" % " or ".join([str(x) for x in self.ring_sizes])
        else:
            return "bridgehead atoms"

    def get_matching_atoms(self, atoms, geometry):
        matching_atoms = []
        for atom1 in atoms:
            matching = True
            if self.ring_sizes:
                unfound_rings = [x for x in self.ring_sizes]
            n_rings = 0
            for i, atom2 in enumerate(atom1.connected):
                for atom3 in atom1.connected[i:]:
                    try:
                        path = geometry.shortest_path(atom2, atom3, avoid=atom1)
                        n_rings += 1
                        if self.ring_sizes:
                            ring_size = len(path) + 1
                            if ring_size in unfound_rings:
                                unfound_rings.remove(ring_size)
                            elif self.match_exact:
                                matching = False
                                break
                    except LookupError:
                        pass

                if not matching:
                    break

            if self.ring_sizes and not unfound_rings and matching:
                matching_atoms.append(atom1)
            elif n_rings > 1 and not self.ring_sizes:
                matching_atoms.append(atom1)
        
        return matching_atoms
