"""finders are used by Geometry.find to locate atoms in a more general way"""
import sys
import inspect
from collections import deque

import numpy as np
from AaronTools import addlogger
from AaronTools.oniomatoms import OniomAtom

def get_class(name):
    """returns the finder class with the given name"""
    for obj_name, obj in inspect.getmembers(sys.modules[__name__]):
        if obj_name == name and inspect.isclass(obj):
            return obj
    raise ValueError("no finder named %s in AaronTools.finders" % name)


class Finder:
    def get_matching_atoms(self, atoms, geometry=None):
        """
        overwrite with function that returns list(Atom) of the atoms that
        match your Finder's criteria
        geometry is an optional argument that could be used to e.g. find
        atoms a certain number of bonds
        """
        pass


class BondsFrom(Finder):
    """
    exact number of bonds from specified atom
    avoid: bonding path cannot pass through these atoms
    """
    def __init__(self, central_atom, number_of_bonds, avoid=None):
        super().__init__()

        self.central_atom = central_atom
        self.number_of_bonds = number_of_bonds
        self.avoid = avoid

    def __repr__(self):
        return "atoms %i bonds from %s" % (self.number_of_bonds, self.central_atom)

    def get_matching_atoms(self, atoms, geometry): # TODO are atoms and geometry needed input?
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        stack = deque([self.central_atom])
        next_stack = deque([])
        frag = [self.central_atom]
        n_bonds = 1
        while stack and self.number_of_bonds:
            next_connected = stack.popleft()
            connected = next_connected.connected - set(frag)
            frag += connected
            next_stack.extend(connected)
            if not stack:
                if n_bonds == self.number_of_bonds:
                    return list(next_stack)
                n_bonds += 1
                stack = next_stack
                next_stack = deque([])

        return matching_atoms


class WithinBondsOf(BondsFrom):
    """within a specified number of bonds from the atom"""
    def __init__(self, central_atom, number_of_bonds, **kwargs):
        super().__init__(central_atom, number_of_bonds)

    def __repr__(self):
        return "atoms within %i bonds of %s" % (self.number_of_bonds, self.central_atom)

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        stack = deque([self.central_atom])
        next_stack = deque([])
        frag = [self.central_atom]
        n_bonds = 1
        while stack and self.number_of_bonds:
            next_connected = stack.popleft()
            connected = next_connected.connected - set(frag)
            if n_bonds < self.number_of_bonds:
                next_stack.extend(connected)
            matching_atoms.extend(connected)
            frag += connected
            if not stack:
                n_bonds += 1
                stack = next_stack
                next_stack = deque([])

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
        try:
            return [atom for atom in atoms if atom in self.atom.connected]
        except AttributeError:
            pass


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
        keep = np.arange(0, len(atoms), dtype=int)
        coords = np.array([atom.coords for atom in atoms])
        coords -= self.point
        
        mask = np.where(coords[:, 0] < self.radius)
        coords = coords[mask]
        keep = keep[mask]
        
        mask = np.where(coords[:, 1] < self.radius)
        coords = coords[mask]
        keep = keep[mask]
        
        mask = np.where(coords[:, 2] < self.radius)
        coords = coords[mask]
        keep = keep[mask]
        
        dist = np.linalg.norm(coords, axis=1)
        mask = np.where(dist < self.radius)
        keep = keep[mask]
        
        matching_atoms = [atoms[k] for k in keep]

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
        keep = np.arange(0, len(atoms), dtype=int)
        coords = np.array([atom.coords for atom in atoms])
        coords -= self.atom.coords
        
        mask = np.where(coords[:, 0] < self.radius)
        coords = coords[mask]
        keep = keep[mask]
        
        mask = np.where(coords[:, 1] < self.radius)
        coords = coords[mask]
        keep = keep[mask]
        
        mask = np.where(coords[:, 2] < self.radius)
        coords = coords[mask]
        keep = keep[mask]
        
        dist = np.linalg.norm(coords, axis=1)
        mask = np.where(dist < self.radius)
        keep = keep[mask]
        
        matching_atoms = [atoms[k] for k in keep]

        return matching_atoms


class NotAny(Finder):
    """atoms not matching specifiers/Finders"""
    def __init__(self, *critera, **kwargs):
        """critera can be any number of Finders and/or other atom specifiers (tags, elements, etc.)"""
        super().__init__()

        if not critera and "critera" in kwargs:
            critera = kwargs["critera"]
        if len(critera) == 1:
            if isinstance(critera[0], tuple):
                critera = critera[0]
        self.critera = critera

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
    def __init__(self, *a, **kw):
        super().__init__(AnyTransitionMetal())

    def __repr__(self):
        return "any non-transition metal"


class HasAttribute(Finder):
    """all atoms with the specified attribute"""
    def __init__(self, attribute_name):
        super().__init__()

        self.attribute_name = attribute_name

    def __repr__(self):
        return "atoms with the '%s' attribute" % self.attribute_name

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that have the attribute"""
        return [atom for atom in atoms if hasattr(atom, self.attribute_name)]


@addlogger
class VSEPR(Finder):
    """
    atoms with the specified VSEPR geometry
    
    see Atom.get_shape for a list of valid vsepr strings
    """
    LOG = None
    def __init__(self, vsepr, cutoff=0.5):
        super().__init__()
        
        self.vsepr = vsepr
        if any(vsepr == x for x in ["triangular cupola", "heptagonal bipyramidal"]):
            self.LOG.warning(
                "triangular cupola and heptagonal bipyramidal cannot be distinguished"
            )
        self.cutoff = cutoff

    def __repr__(self):
        return "atoms with %s shape" % self.vsepr

    def get_matching_atoms(self, atoms, geometry=None):
        matching_atoms = []
        for atom in atoms:
            out = atom.get_vsepr()
            if out is not None:
                shape, score = atom.get_vsepr()
                if shape == self.vsepr and score < self.cutoff:
                    matching_atoms.append(atom)
        
        return matching_atoms


class BondedElements(Finder):
    """
    atoms bonded to the specified neighboring elements
    
    if match_exact=True (default), elements must match exactly 
    
    e.g. BondedElements('C') will find
    atoms bonded to only one carbon and nothing else
    """
    def __init__(self, *args, match_exact=True, **kwargs):
        super().__init__()
        
        if not args and "elements" in kwargs:
            args = kwargs["elements"]
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
    """
    chiral centers
    
    atoms with a non-planar VSEPR geometry with all bonded groups
    being distinct
    
    for rings, looks for a set of unique canonical ranks for atoms that 
    are all the same number of bonds away from one atom
    """
    #IUPAC spelling 
    def __init__(self, RS_only=False):
        """RS_only: bool  - if True, do not identify chiral centers that are chiral because they
                            are connected to multiple chiral fragments with the same chirality
                            this corresponds to R/S centers
                            False will include r/s centers as well
        """
        super().__init__()
        self.RS_only = RS_only

    def __repr__(self):
        return "chiral centers"

    def get_matching_atoms(self, atoms, geometry):
        from AaronTools.geometry import Geometry
        from AaronTools.symmetry import PointGroup
        # from time import perf_counter
        # 
        # start = perf_counter()
        matching_atoms = []

        # b/c they are connected to chiral fragments
        geometry.refresh_ranks()
        chiral_atoms_changed = True
        ranks = geometry.canonical_rank(break_ties=False, update=False, invariant=True)
        frags = []
        properly_shaped_atoms = []
        for atom in geometry.atoms:
            if len(atom.connected) < 3:
                continue
            vsepr, _ = atom.get_vsepr()
            if vsepr in ['trigonal planar', 't shaped', 'sqaure planar']:
                continue
            properly_shaped_atoms.append(atom)
            frags.append([])
            single_atoms = dict()
            for bonded_atom in atom.connected:
                frag = geometry.get_fragment(bonded_atom, atom, as_object=False)
                frags[-1].append(frag)
                # keep track of single atom fragments to more quickly
                # eliminate atoms that aren't chiral
                if len(frag) == 1:
                    if frag[0].element in single_atoms:
                        single_atoms[frag[0].element] += 1
                        if single_atoms[frag[0].element] >= len(atom.connected) / 2:
                            frags.pop(-1)
                            properly_shaped_atoms.pop(-1)
                            break
                    else:
                        single_atoms[frag[0].element] = 1

        # print(properly_shaped_atoms)

        # need to do multiple passes b/c sometimes atoms are chiral
        # because of other chiral centers
        k = 0
        while chiral_atoms_changed:
            chiral_atoms_changed = False
            k += 1
            #skip atoms we've already found
            for ndx, atom in enumerate(properly_shaped_atoms):
                if atom in matching_atoms:
                    continue

                neighbor_ranks = [
                    ranks[geometry.atoms.index(bonded_atom)]
                    for bonded_atom in atom.connected
                ]
                
                # first iteration should only look for centers that are chiral
                # because the fragments are different
                if k == 1 and len(atom.connected) <= 4 and all(
                    neighbor_ranks.count(rank) == 1 for rank in neighbor_ranks
                ):
                    matching_atoms.append(atom)
                    chiral_atoms_changed = True
                elif k == 1 and len(atom.connected) > 4:
                    test_geom = Geometry(
                        [atom, *atom.connected], refresh_ranks=False, refresh_connected=False
                    )
                    groups = [ranks[geometry.atoms.index(a)] for a in test_geom.atoms]
                    pg = PointGroup(test_geom, groups=groups, center=atom.coords)
                    print(pg.name)
                    if pg.name == "C1":
                        matching_atoms.append(atom)
                        chiral_atoms_changed = True
                # more iterations should only look for centers that are
                # chiral because of the presence of other chiral centers
                elif k > 1 and all(
                    neighbor_ranks.count(rank) <= len(atom.connected) / 2 for rank in neighbor_ranks
                ):
                    chiral = True
                    for i, frag1 in enumerate(frags[ndx]):
                        #get the ranks of the atoms in this fragment
                        ranks_1 = [ranks[geometry.atoms.index(atom)] for atom in frag1]
                        for frag2 in frags[ndx][:i]:
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

                                # correct connected elements
                                for o, l in zip(
                                    sorted([aa.element for aa in a.connected]),
                                    sorted([bb.element for bb in b.connected]),
                                ):
                                    if o != l:
                                        same = False
                                        break
    
                                if a is b:
                                    break
    
                                if not self.RS_only and a in matching_atoms and b in matching_atoms:
                                    #use RMSD to see if they have the same handedness
                                    a_connected = sorted(a.connected)
                                    b_connected = sorted(b.connected)
                                    a_targets = [a] + list(a_connected)
                                    b_targets = [b] + list(b_connected)
                                    if geometry.RMSD(
                                        geometry,
                                        targets=a_targets,
                                        ref_targets=b_targets,
                                        sort=False,
                                        align=False,
                                    ) < 0.1:
                                        same = False
                                        break

                            # I'm not sure why this code was here...
                            # ring_atoms = [
                            #     bonded_atom for bonded_atom in atom.connected
                            #     if bonded_atom in frag1 and bonded_atom in frag2
                            # ]
                            # if len(ring_atoms) > 0:
                            #     #this is a ring
                            #     #look at the rank of all atoms that are n bonds away from this atom
                            #     #if the ranks are ever all different, this is a chiral center
                            #     n_bonds = 1
                            #     acceptable_nbonds = True
                            #     while acceptable_nbonds:
                            #         try:
                            #             atoms_within_nbonds = geometry.find(BondsFrom(atom, n_bonds))
                            #             nbonds_ranks = [
                            #                 ranks[geometry.atoms.index(a)] for a in atoms_within_nbonds
                            #             ]
                            #             if all(nbonds_ranks.count(r) == 1 for r in nbonds_ranks):
                            #                 same = False
                            #                 acceptable_nbonds = False
                            #             elif not self.RS_only:
                            #                 # need to find things in the ring that are chiral
                            #                 # b/c of other chiral centers
                            #                 for n, atom1 in enumerate(atoms_within_nbonds):
                            #                     for m, atom2 in enumerate(atoms_within_nbonds[n+1:]):
                            #                         p = m + n + 1
                            #                         if nbonds_ranks[n] == nbonds_ranks[p]:
                            #                             a_connected = sorted(atom1.connected)
                            #                             b_connected = sorted(atom2.connected)
                            #                             a_targets = [atom1] + list(a_connected)
                            #                             b_targets = [atom2] + list(b_connected)
                            #                             if geometry.RMSD(
                            #                                 geometry,
                            #                                 targets=a_targets,
                            #                                 ref_targets=b_targets,
                            #                                 sort=False,
                            #                                 align=False,
                            #                             ) < 0.1:
                            #                                 same = False
                            #                                 break
                            #                     if not same:
                            #                         break
                            # 
                            #             n_bonds += 1
                            #         except LookupError:
                            #             acceptable_nbonds = False
                            # 
                            #     if not same:
                            #         break
    
                            if same:
                                chiral = False
                                break

                    if chiral:
                        chiral_atoms_changed = True
                        matching_atoms.append(atom)
        
            if self.RS_only:
                break
        # stop = perf_counter()
        # print("took %.3fs" % (stop - start))
        
        return matching_atoms

#alternative spelling
ChiralCenters = ChiralCentres


class FlaggedAtoms(Finder):
    """
    atoms with a non-zero flag
    """
    # useful for finding constrained atoms
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
        matching_atoms = set([])
        wrong_atoms = set([])

        # stackJ is everything that is N bonds away from atomJ
        # but stack1 excludes things that are closer to atom2
        # once we run out of stackJ, everything that's left
        # is close to atom2
        stack1 = deque([self.atom1])
        stack2 = deque([self.atom2])
        visited1 = set([self.atom1])
        visited2 = set([self.atom2])
        while len(stack1) > 0:
            group1_atoms = set()
            # go through each atom in stack1 and add in things it's
            # bonded to but that we haven't visited yet
            for a in stack1:
                group1_atoms.update((a.connected - visited1))
            
            group1_atoms.discard(visited1)
            visited1.update(group1_atoms)

            group2_atoms = set()
            for a in stack2:
                group2_atoms.update((a.connected - visited2))

            group2_atoms.discard(visited2)
            visited2.update(group2_atoms)

            # next stack is whatever we just visited
            stack1 = group1_atoms
            stack2 = group2_atoms

            # matching atoms are ones from stack1 that haven't been visited in stack2 yet
            matching_atoms.update((group1_atoms - visited2))
            if self.include_ties:
                # unless we are including ties
                matching_atoms.update(group1_atoms & group2_atoms)    

        return list(matching_atoms)


class IsElement(Finder):
    """all atoms of the specified element"""
    def __init__(self, element):
        super().__init__()

        self.element = element

    def __repr__(self):
        return "atoms of the element '%s'" % self.element

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms of that element"""
        return [atom for atom in atoms if atom.element == self.element]

class OfType(Finder):
    """
    all atoms of the specified GAFF atom type
    
    if ignore_metals = True (default), bonding with metals will not count towards VSEPR shapes
    """
    def __init__(self, atomtype, ignore_metals=True):
        super().__init__()

        self.atomtype = atomtype.capitalize()
        if self.atomtype in {'Br', 'Cl'}:
            self.element = self.atomtype
        else:
            self.split_type = list(self.atomtype)
            self.element = self.split_type[0]
        self.ignore_metals = ignore_metals

    def __repr__(self):
        return "atoms of the gaff atomtype '%s'" % self.atomtype

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are of the given atom type"""
#        geom = geometry.copy()
        if self.ignore_metals == True:
            metals = []
            from AaronTools.const import TMETAL
            for i, atom in enumerate(geometry.atoms):
                atom.index = i
                if atom.element in TMETAL:
                    metals.append(atom)

        geometry - metals
        atoms = geometry.atoms

        atoms = [atom for atom in atoms if atom.element == self.element]

        class CustomError(Exception):
            pass

        shapes = {'C1': ['linear 1', 'linear 2'],
                  'C2': ['trigonal planar', 'bent 2 planar'],
                  'C3': ['trigonal pyramidal', 'tetrahedral'],
                   'C': ['trigonal planar'],
                  'Ha': ['linear 1'],
                  'Hc': ['linear 1'],
                  'N1': ['linear 1', 'linear 2'],
                  'N2': ['bent 2 planar', 'bent 2 tetrahedral'],
                  'N3': ['trigonal pyramidal', 'bent 3 tetrahedral'],
                  'N4': ['tetrahedral'],
                  'Na': ['trigonal planar'],
                  'S4': ['trigonal planar'],
                  'S6': ['tetrahedral'],
                  'P3': ['trigonal pyramidal'],
                  'P4': ['trigonal planar'],
                  'P5': ['tetrahedral'],
                  'Ca': ['trigonal planar', 'bent 2 planar'],
                   'N': ['trigonal planar'],
                  'Nh': ['trigonal planar'],
                  'Os': ['bent 2 tetrahedral', 'bent 2 planar']}

        """helper functions"""
        def is_carbonyl(atom):
            """returns True if atom is carbonyl carbon"""
            for connected in atom.connected:
                if connected.element == 'O' and connected in BondedElements(atom.element).get_matching_atoms(atoms):
                    carbonyl = True
                    break
                else:
                    carbonyl = False
            return carbonyl

        def is_carboxyl(atom):
            """returns True if atom is carboxyl carbon"""
            o_counter = 0
            if is_carbonyl(atom):
                for connected in atom.connected:
                    if connected.element == 'O':
                        o_counter += 1
                if o_counter == 2:
                    return True
            else:
                return False

        def is_water(atom):
            if atom.element == 'O':
                h_counter = 0
                for connected in atom.connected:
                    if connected.element == 'H': h_counter +=1
                if h_counter == 2: return True
                else: return False
            else: return False

        def is_amide(atom):
            if atom.element == 'N':
                for connected in atom.connected:
                    if is_carbonyl(connected):
                        return True
                        break
                    else: return False
            else: return False

        matching_atoms = []
        if self.atomtype in {'F', 'Cl', 'Br', 'I'}: 
            for atom in IsElement(self.atomtype).get_matching_atoms(atoms): matching_atoms.append(atom)
        elif self.split_type[0] == 'H' and self.atomtype not in {'Ha', 'Hc'}:
            if self.split_type[1] in {'o','w'}:
                for atom in BondedElements('O').get_matching_atoms(atoms):
                    for connected in atom.connected:
                        if self.atomtype == 'Hw' and is_water(connected): matching_atoms.append(atom)
                        elif self.atomtype == 'Ho' and not is_water(connected): matching_atoms.append(atom)
            else:
                for atom in BondedElements(self.split_type[1].capitalize()).get_matching_atoms(atoms): matching_atoms.append(atom)
        elif self.atomtype in {'O', 'S2', 'P2'}:
            for atom in BondedElements('C').get_matching_atoms(atoms): matching_atoms.append(atom)
        elif self.atomtype in {'C', 'C2', 'Ca', 'Na', 'Nh', 'Ha', 'Hc','N'}:
            aromatics, charge, fused = geometry.get_aromatic_atoms(return_rings=False)
            for shape in shapes.get(self.atomtype):
                for atom in VSEPR(shape).get_matching_atoms(atoms):
                    if self.atomtype == 'Ca' and atom in aromatics and not is_carbonyl(atom): matching_atoms.append(atom)
                    elif self.atomtype == 'C2' and atom not in aromatics and not is_carbonyl(atom): matching_atoms.append(atom)
                    elif self.atomtype == 'C' and is_carbonyl(atom): matching_atoms.append(atom)
                    elif self.atomtype == 'N' and is_amide(atom) and atom not in aromatics: matching_atoms.append(atom)
                    elif self.atomtype == 'Na' and charge == 1 and atom.element == 'N' and not is_carboxyl(atom) and len(matching_atoms) == 0 and atom in aromatics: matching_atoms.append(atom)
                    elif self.atomtype == 'Na' and atom not in aromatics and not is_carboxyl(atom) and not is_amide(atom): matching_atoms.append(atom)
                    elif self.atomtype in {'Na','Nh'} and charge == 1 and atom.element == 'N' and len(matching_atoms) > 0 and atom in aromatics:
                        raise CustomError("Indistinguishable nitrogens in aromatic ring")
                    elif self.atomtype == 'Nh' and charge == 0 and atom.element == 'N' and atom in aromatics: matching_atoms.append(atom)
                    elif self.atomtype in {'Hc', 'Ha'}:
                        for connected in atom.connected:
                            if self.atomtype == 'Ha' and connected.element == 'C' and connected in aromatics: matching_atoms.append(atom)
                            elif self.atomtype == 'Hc' and connected.element == 'C' and connected not in aromatics: matching_atoms.append(atom)
        elif self.atomtype in {'Oh', 'Os', 'Sh', 'Ss','Ow'}:
            for shape in shapes.get('Os'):
                for atom in VSEPR(shape,cutoff=0.7).get_matching_atoms(atoms):
                    counter = 0
                    for connected in atom.connected:
                        if self.split_type[1] == 'h' and connected.element == 'H' and not is_water(atom): matching_atoms.append(atom)
                        elif self.split_type[1] == 'w' and is_water(atom) and atom not in matching_atoms: matching_atoms.append(atom)
                        elif self.split_type[1] == 's' and connected.element != 'H': counter +=1
                    if counter == 2: matching_atoms.append(atom)
        elif self.atomtype == 'No':
            for atom in IsElement('N').get_matching_atoms(atoms):
                if is_carboxyl(atom): matching_atoms.append(atom)
        else:
            for shape in shapes.get(self.atomtype):
                for atom in VSEPR(shape).get_matching_atoms(atoms): matching_atoms.append(atom)
        matching_atoms = [match for match in matching_atoms if match.element == self.element]

        #add metals back into geometry and reorder
        geometry = geometry + metals
        new_atoms = [0]* len(geometry.atoms)
        for atom in geometry.atoms:
            new_atoms[atom.index] = atom
        geometry.atoms = new_atoms

        return matching_atoms

class Aromatics(Finder):
    """all atoms in aromatic rings"""
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "atoms that are in aromatic rings"

    def get_matching_atoms(self, atoms, geometry):
        aromatics, charge, fused = geometry.get_aromatic_atoms(return_rings=False)
        return aromatics

class ONIOMLayer(Finder):
    """all atoms in a given ONIOM layer or list of ONIOM layers"""
    def __init__(self, layers=""):
        super().__init__()

        self.layers = layers
        if isinstance(layers, list):
            for layer in self.layers: 
                if layer.capitalize() not in ['H', 'M', 'L']:
                    raise ValueError("layer must be H, M, or L")

    def __repr__(self):
        return "atoms in the ONIOM layer '%s'" % self.layers

    def get_matching_atoms(self, atoms, geometry=None):
        matching_atoms = []
        for atom in atoms:
            if isinstance(self.layers, list):
                try:
                    for layer in self.layers:
                        if atom.layer.capitalize() == layer.capitalize(): matching_atoms.append(atom)
                except AttributeError:
                    pass
                    #print("ONIOMlayer only accepts OniomAtom type atoms")
            else:
                try:
                    if atom.layer.capitalize() == self.layer.capitalize(): matching_atoms.append(atom)
                except AttributeError:
                    pass #print("ONIOMlayer only accepts OniomAtom type atoms")
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
        :param None|list(int) ring_sizes" list of int, size of rings (e.g. [6, 6] for atoms that bridge
            two 6-membered rings)
            
            not specifying yields bridgehead atoms for any ring size
        :param bool match_exact: bool, if True, return atoms only bridging the specified rings
            if False, the ring_sizes is taken as a minimum (e.g.
            ring_size=[6, 6], match_exact=False would also yield atoms
            bridging three 6-membered rings or two six-membered rings and
            a five-membered ring)
        """
        self.ring_sizes = ring_sizes
        self.match_exact = match_exact

    def __repr__(self):
        if self.ring_sizes:
            return "bridgeheads of %s-member rings" % " or ".join([str(x) for x in self.ring_sizes])
        return "bridgehead atoms"

    def get_matching_atoms(self, atoms, geometry):
        matching_atoms = []
        for atom1 in atoms:
            matching = True
            if self.ring_sizes:
                unfound_rings = list(self.ring_sizes)
            n_rings = 0
            for i, atom2 in enumerate(atom1.connected):
                for atom3 in list(atom1.connected)[:i]:
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


class SpiroCenters(Finder):
    """
    atom in two different rings with no other common atoms
    """
    def __init__(self, ring_sizes=None, match_exact=False):
        """
        :param None|list(int) ring_sizes: list of int, size of rings (e.g. [6, 6] for atoms that bridge
            two 6-membered rings)
            
            not specifying yields bridgehead atoms for any ring size
        :param bool match_exact: if True, return atoms only bridging the specified rings
            if False, the ring_sizes is taken as a minimum (e.g.
            ring_size=[6, 6], match_exact=False would also yield atoms
            bridging three 6-membered rings or two six-membered rings and
            a five-membered ring)
        """
        self.ring_sizes = ring_sizes
        self.match_exact = match_exact

    def __repr__(self):
        if self.ring_sizes:
            return "atoms in different %s-member rings" % " or ".join(
                [str(x) for x in self.ring_sizes]
            )
        return "spiro atoms"

    def get_matching_atoms(self, atoms, geometry):
        matching_atoms = []
        for atom1 in atoms:
            matching = True
            if self.ring_sizes:
                unfound_rings = list(self.ring_sizes)
            n_rings = 0
            rings = []
            for i, atom2 in enumerate(atom1.connected):
                for atom3 in list(atom1.connected)[:i]:
                    try:
                        path = geometry.shortest_path(atom2, atom3, avoid=atom1)
                        for ring in rings:
                            if any(atom in path for atom in ring):
                                continue
                        rings.append(path)
                    except LookupError:
                        pass
            for i, ring in enumerate(rings):
                bad_ring = False
                for ring2 in rings[:i]:
                    if any(atom in ring for atom in ring2):
                        bad_ring = True
                        break
                if bad_ring:
                    continue
                n_rings += 1
                if self.ring_sizes:
                    ring_size = len(path) + 1
                    if ring_size in unfound_rings:
                        unfound_rings.remove(ring_size)
                    elif self.match_exact:
                        matching = False
                        break

                if not matching:
                    break

            if self.ring_sizes and not unfound_rings and matching:
                matching_atoms.append(atom1)
            elif n_rings > 1 and not self.ring_sizes:
                matching_atoms.append(atom1)

        return matching_atoms

class Residue(Finder):
    """all atoms in a given residue"""
    def __init__(self, residue):
        super().__init__()
        
        if not isinstance(residue, str):
            residue = str(residue)
        self.residue=residue

    def get_matching_atoms(self, geometry):
        matching_atoms = []
        for atom in geometry.atoms:
            if isinstance(atom, OniomAtom):
                break
            else:
                raise AttributeError("atoms in % geometry have no attribute 'residue'" % geometry.name)
        for atom in geometry.atoms:
            if atom.residue == self.residue:
                matching_atoms.append(atom)
        return matching_atoms
