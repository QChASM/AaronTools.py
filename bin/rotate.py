#!/usr/bin/env python3

import argparse
import os
import sys
from warnings import warn

import numpy as np
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import get_filename, glob_files

rotate_parser = argparse.ArgumentParser(
    description="rotate a fragment or molecule's coordinates",
    formatter_class=argparse.RawTextHelpFormatter,
)

rotate_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

rotate_parser.add_argument(
    "-if",
    "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin",
)

rotated_atoms = rotate_parser.add_argument_group(
    "rotated atoms (default is all atoms)"
)
rot_atoms = rotated_atoms.add_mutually_exclusive_group(required=False)
rot_atoms.add_argument(
    "-t",
    "--target",
    type=str,
    default=None,
    required=False,
    dest="targets",
    metavar="targets",
    help="atoms to rotate (1-indexed)\n"
    + "comma- and/or hyphen-separated list\n"
    + "hyphens denote a range of atoms; commas separate individual atoms or ranges",
)

rot_atoms.add_argument(
    "-f",
    "--fragment",
    type=str,
    default=None,
    required=False,
    dest="fragment",
    metavar="target",
    help="rotate fragment containing target",
)

rotate_parser.add_argument(
    "-c",
    "--center",
    default=None,
    required=False,
    dest="center",
    metavar="targets",
    help="translate the centroid of the specified atoms to the\norigin before rotating",
)

defined_vector = rotate_parser.add_argument_group("define vector")
def_vector = defined_vector.add_mutually_exclusive_group(required=True)
def_vector.add_argument(
    "-v",
    "--vector",
    type=float,
    nargs=3,
    default=None,
    required=False,
    dest="vector",
    metavar=("x", "y", "z"),
    help="rotate about the vector from the origin to (x, y, z)",
)

def_vector.add_argument(
    "-b",
    "--bond",
    type=str,
    nargs=2,
    default=None,
    dest="bond",
    metavar=("a1", "a2"),
    help="rotate about the vector from atom a1 to atom a2 (1-indexed)",
)

def_vector.add_argument(
    "-x",
    "--axis",
    type=str,
    default=None,
    required=False,
    dest="axis",
    choices=["x", "y", "z"],
    help="rotate about specified axis",
)

def_vector.add_argument(
    "-g",
    "--group",
    type=str,
    default=None,
    required=False,
    dest="group",
    metavar="targets",
    help="rotate about axis from origin (or center specified with '--center')\n" +
    "to the centroid of the specified atoms",
)

def_vector.add_argument(
    "-p",
    "--perpendicular",
    type=str,
    default=None,
    required=False,
    dest="perp",
    metavar="targets",
    help="rotate about a vector orthogonal to the plane of best fit containing targets",
)

rotate_parser.add_argument(
    "-a",
    "--angle",
    type=float,
    default=None,
    required=None,
    dest="angle",
    metavar="angle",
    help="angle of rotation (in degrees by default)",
)

rotate_parser.add_argument(
    "-r",
    "--radians",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="radians",
    help="use when angle is specified in radians instead of degrees",
)

rotate_parser.add_argument(
    "-n",
    "--number",
    type=int,
    default=None,
    required=False,
    dest="num",
    metavar="num",
    help="when angle is specified, rotate num times by angle\n" +
    "when angle is not specified, rotate 360/num degrees num times",
)

rotate_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE, $AXIS, $ANGLE will be replaced with the name of the\n" +
    "input file, rotation axis, and angle or rotation, respectively\nDefault: stdout",
)

args = rotate_parser.parse_args()

if args.angle is None and args.num is None:
    raise ValueError("must specified one of ('--angle', '--number')")
elif args.num is None and args.angle is not None:
    args.num = 1
    args.angle = args.angle
elif args.num is not None and args.angle is None:
    args.num = args.num
    args.angle = 360.0 / args.num
elif args.num is not None and args.angle is not None:
    args.num = args.num
    args.angle = args.angle

if not args.radians:
    args.angle = np.deg2rad(args.angle)

for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f))
        else:
            infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)
    center = args.center

    if args.fragment is not None:
        targets = geom.get_all_connected(args.fragment)

    else:
        targets = args.targets

    if args.vector is not None:
        vector = args.vector

    elif args.bond is not None:
        a1 = geom.find(args.bond[0])[0]
        a2 = geom.find(args.bond[1])[0]
        vector = a1.bond(a2)
        if center is None:
            warn(
                "center set to the coordinates of atom %s; " +
                "using --center/-c none will override this"
                % a1.name
            )
            center = a1

    elif args.axis is not None:
        vector = np.zeros(3)
        vector[["x", "y", "z"].index(args.axis)] = 1.0

    elif args.group is not None:
        vector = geom.COM(targets=args.group)
        if center is not None:
            vector -= geom.COM(targets=center)

    elif args.perp is not None:
        if len(geom.find(args.perp)) < 3:
            raise RuntimeError(
                "must specify at least three atoms to --perpendicular/-p"
            )

        xyz = geom.coordinates(args.perp) - geom.COM(args.perp)
        R = np.dot(xyz.T, xyz)
        u, s, vh = np.linalg.svd(R, compute_uv=True)
        vector = u[:, -1]
        if center is None:
            warn(
                "center set to the centroid of atoms %s; using --center/-c none will override this"
                % args.perp
            )
            center = geom.COM(args.perp)

    if args.center is not None and args.center.lower() == "none":
        center = None

    rotated_geoms = []
    for i in range(0, args.num):
        geom.rotate(vector, args.angle, targets=targets, center=center)

        if args.outfile is not False:
            outfile = args.outfile
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(f))
            outfile = outfile.replace(
                "$AXIS", ".".join(["%.3f" % x for x in vector])
            )
            outfile = outfile.replace(
                "$ANGLE",
                str.zfill("%.2f" % np.rad2deg(args.angle * (i + 1)), 6),
            )
            parent_dir = os.path.dirname(outfile)
            if not os.path.isdir(parent_dir) and parent_dir != "":
                os.makedirs(parent_dir)

        else:
            outfile = args.outfile

        s = geom.write(append=True, outfile=outfile)
        if not outfile:
            print(s)
