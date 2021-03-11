#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

libaddsub_parser = argparse.ArgumentParser(
    description="add a substituent to your personal library",
    formatter_class=argparse.RawTextHelpFormatter,
)

libaddsub_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    default=None,
    help="a coordinate file",
)

libaddsub_parser.add_argument(
    "-n",
    "--name",
    type=str,
    required=False,
    default=None,
    dest="name",
    help="Name of substituent being added to the library\n" +
    "if no name is given, the substituent will be printed to STDOUT",
)

libaddsub_parser.add_argument(
    "-s",
    "--substituent-atom",
    type=str,
    nargs="+",
    required=True,
    dest="target",
    help="substituent atom connected to the rest of the molecule (1-indexed)",
)

libaddsub_parser.add_argument(
    "-a",
    "--attached-to",
    type=str,
    nargs="+",
    required=True,
    dest="avoid",
    help="atom on the molecule that is connected to the substituent",
)

libaddsub_parser.add_argument(
    "-c",
    "--conf",
    type=int,
    nargs=2,
    required=True,
    metavar=("CONFORMERS", "ANGLE"),
    dest="confangle",
    help="number of conformers and the rotation angle (degrees) used to generate each conformer",
)

args = libaddsub_parser.parse_args()

n_confs, angle = args.confangle

if n_confs < 1:
    raise RuntimeError("conformers cannot be < 1")

geom = Geometry(args.infile)
geom.coord_shift(-geom.COM(args.avoid))
sub = geom.get_fragment(args.target, args.avoid, as_object=True)

target = geom.COM(args.target)
x_axis = np.array([1.0, 0.0, 0.0])
n = np.linalg.norm(target)
vb = target / n
d = np.linalg.norm(vb - x_axis)
theta = np.arccos((d ** 2 - 2) / -2)
vx = np.cross(vb, x_axis)
sub.rotate(vx, theta)

sub.comment = "CF:%i,%i" % (n_confs, angle)

if args.name is None:
    print(sub.write(outfile=False))
else:
    sub_file = os.path.join(
        os.path.dirname(Substituent.AARON_LIBS), args.name + ".xyz"
    )
    if os.path.exists(sub_file):
        overwrite = input(
            "%s already exists.\nWould you like to overwrite it? (yes/NO)\n"
            % sub_file
        )
        if overwrite.lower() not in ["yes", "y"]:
            print("not overwriting")
            sys.exit(0)

    sub.write(outfile=sub_file)
