#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.const import RADII

libaddsub_parser = argparse.ArgumentParser(description='add a substituent to your personal library', \
                    formatter_class=argparse.RawTextHelpFormatter)
libaddsub_parser.add_argument('infile', metavar='input file', \
                                type=str, \
                                default=None, \
                                help='a coordinate file')

libaddsub_parser.add_argument('-n', '--name', \
                                type=str, \
                                required=False, \
                                default=None, \
                                dest='name', \
                                help="""Name of substituent being added to the library
if no name is given, the substituent will be printed to STDOUT""")

libaddsub_parser.add_argument('-t', '--target', \
                                type=str, \
                                nargs=1, \
                                required=True, \
                                dest='target', \
                                help='substituent atom connected to the rest of the molecule (1-indexed)')

libaddsub_parser.add_argument('-a', '--avoid', \
                                type=str, \
                                nargs=1, \
                                required=True, \
                                dest='avoid', \
                                help='atom on the molecule that is connected to the substituent')

libaddsub_parser.add_argument('-c', '--conf', \
                                type=int, \
                                nargs=2, \
                                required=True, \
                                metavar=('CONFORMERS', 'ANGLE'), \
                                dest='confangle', \
                                help="number of conformers and the rotation angle (degrees) used to generate each conformer")

args = libaddsub_parser.parse_args()

infile = args.infile
name = args.name
n_confs = args.confangle[0]
angle = args.confangle[1]

if n_confs < 1:
    raise RuntimeError("conformers cannot be < 1")

geom = Geometry(infile)

target = geom.find(args.target[0])[0]
avoid = geom.find(args.avoid[0])[0]

geom.coord_shift(-avoid.coords)

sub = geom.get_fragment(target, avoid, as_object=True)

x_axis = np.array([1., 0., 0.])
n = np.linalg.norm(target.coords)
vb = target.coords/n
d = np.linalg.norm(vb - x_axis)

theta = np.arccos((d**2 - 2) / -2)

vx = np.cross(vb, x_axis)

sub.rotate(vx, theta)

if target.element in RADII:
    goal_dist = RADII['C'] + RADII[target.element]
    sub.coord_shift(goal_dist - target.coords[0])
else:
    raise RuntimeWarning("No bond radius for %s, distance will not be adjusted" % target.element)

sub.comment = "CF:%i,%i" % (n_confs, angle)

if name is None:
    print(sub.write(outfile=False))
else:
    sub_file = os.path.join(os.path.dirname(Substituent.AARON_LIBS), name + '.xyz')
    if os.path.exists(sub_file):
        overwrite = input("%s already exists.\nWould you like to overwrite it? (YES/no)\n" % sub_file)
        if overwrite != "YES":
            print("%s to overwrite, not overwriting" % overwrite)
            sys.exit(0)
    
    sub.write(outfile=sub_file)
