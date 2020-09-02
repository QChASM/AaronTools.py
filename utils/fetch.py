import re
import numpy as np

from AaronTools.fileIO import FileReader
from AaronTools.finders import BondedTo
from AaronTools.geometry import Geometry
from AaronTools.ring import Ring
from AaronTools.substituent import Substituent

from copy import deepcopy

from urllib.request import urlopen
from urllib.error import HTTPError

def from_string(name, form='smiles'):
    """get Geometry from string
    form=iupac -> iupac to smiles from opsin API
                   --> form=smiles
    form=smiles -> structure from cactvs API"""

    accepted_forms = ['iupac', 'smiles']

    if form not in accepted_forms:
        raise NotImplementedError("cannot create substituent given %s; use one of %s" % form, str(accepted_forms))

    if form == 'smiles':
        smiles = name
    elif form == 'iupac':
        #opsin seems to be better at iupac names with radicals
        url_smi = "https://opsin.ch.cam.ac.uk/opsin/%s.smi" % name

        try:
            smiles = urlopen(url_smi).read().decode('utf8')
        except HTTPError:
           raise RuntimeError("%s is not a valid IUPAC name or https://opsin.ch.cam.ac.uk is down" % name)

    url_sd = "https://cactus.nci.nih.gov/chemical/structure/%s/file?format=sdf" % smiles
    s_sd = urlopen(url_sd).read().decode('utf8')
    f = FileReader((name, "sd", s_sd))
    return Geometry(f)

def substituent_from_string(name, form='smiles'):
    """
    creates a substituent from a string
    name    str     identifier for substituent
    form    str     type of identifier (smiles, iupac)
    """
    #convert whatever format we're given to smiles
    #then grab the structure from cactus site

    accepted_forms = ['iupac', 'smiles']

    if form not in accepted_forms:
        raise NotImplementedError("cannot create substituent given %s; use one of %s" % form, str(accepted_forms))

    rad = re.compile('\[\S+?\]')
    elements = re.compile('[A-Z][a-z]?')

    if form == 'smiles':
        smiles = name
    elif form == 'iupac':
        #opsin seems to be better at iupac names with radicals
        url_smi = "https://opsin.ch.cam.ac.uk/opsin/%s.smi" % name

        try:
            smiles = urlopen(url_smi).read().decode('utf8')
        except HTTPError:
           raise RuntimeError("%s is not a valid IUPAC name or https://opsin.ch.cam.ac.uk is down" % name)

    #radical atom is the first atom in []
    #charged atoms are also in []
    my_rad = None
    radicals = rad.findall(smiles)
    if radicals:
        for rad in radicals:
            if '+' not in rad and '-' not in rad:
                my_rad = rad
                break

    if my_rad is None:
        if radicals:
            warn("radical atom may be ambiguous, be sure to check output: %s" % smiles)
            my_rad = radicals[0]
        else:
            raise RuntimeError("could not determine radical site on %s; radical site is expected to be in []" % smiles)

    #construct a modified smiles string with (H) right after the radical center
    #keep track of the position of this added H
    pos1 = smiles.index(my_rad)
    pos2 = smiles.index(my_rad)+len(my_rad)
    previous_atoms = elements.findall(smiles[:pos1])
    rad_pos = len(previous_atoms)
    if '+' not in my_rad and '-' not in my_rad:
        mod_smiles = smiles[:pos1] + re.sub(r'H\d+', '', my_rad[1:-1]) + smiles[pos2:]
    else:
        mod_smiles = smiles[:pos1] + my_rad[:-1].rstrip('H') + ']' + '(H)' + smiles[pos2:]
    
    #fix triple bond url
    mod_smiles = mod_smiles.replace('#', '%23')

    #grab structure from cactus
    geom = from_string(mod_smiles, form='smiles')

    #the H we added is in the same position in the structure as in the smiles string
    rad = geom.atoms[rad_pos]
    added_H = [atom for atom in rad.connected if atom.element == 'H'][0]

    #move the added H to the origin
    geom.coord_shift(-added_H.coords)

    #get the atom bonded to this H
    #also move the atom on H to the front of the atoms list to have the expected connectivity
    bonded_atom = geom.find(BondedTo(added_H))[0]
    geom.atoms = [bonded_atom] + [atom for atom in geom.atoms if atom != bonded_atom]

    #align the H-atom bond with the x-axis to have the expected orientation
    bond = deepcopy(bonded_atom.coords)
    bond /= np.linalg.norm(bond)
    x_axis = np.array([1.0, 0.0, 0.0])
    rot_axis = np.cross(x_axis, bond)
    if np.linalg.norm(rot_axis) > np.finfo(float).eps:
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.dot(bond, x_axis))
        geom.rotate(rot_axis, -angle)

    return Substituent([atom for atom in geom.atoms if atom != added_H])
    
def ring_from_string(name, end=None, form='smiles'):
    """create ring fragment from string
    name    str         identifier for ring
    end     AtomSpec    atom specifiers for ring walk direction
    form    str         type of identifier (smiles, iupac)
    """

    ring = from_string(name, form)
    if end is not None:
        if isinstance(end, int):
            ring = Ring(ring)
            ring.find_end(end)
            return ring
        elif isinstance(end, list):
            return Ring(ring, end=end, name=name)
        else:
            raise ValueError("expected int or list for 'end' in 'from_string', got %s", str(end))

    else:
        return Ring(ring, name=name)

