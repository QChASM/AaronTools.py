#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

translate_parser = argparse.ArgumentParser(
    description="move atoms along a vector",
    formatter_class=argparse.RawTextHelpFormatter
)

translate_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

translate_parser.add_argument(
    "-if", "--input-format",
    type=str,
    choices=read_types,
    default=None,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

target_specifier = translate_parser.add_argument_group("atoms to move")
target_specifier.add_argument(
    "-t", "--targets",
    type=str,
    default=None,
    required=False,
    dest="targets",
    metavar="atoms",
    help="move atoms with specified indices\n"
    "Default: whole structure"
)

target_specifier.add_argument(
    "-f", "--fragment",
    type=str,
    default=None,
    required=False,
    dest="fragments",
    metavar="atoms",
    help="move fragments containing specified atoms\n"
    "Default: whole structure"
)

translate_parser.add_argument(
    "-ct", "--center-targets",
    type=str,
    default=None,
    required=False,
    dest="targets",
    metavar="targets",
    help="target atoms for -com or -cent arguments\n" +
    "comma (,) and/or hyphen (-) separated list\n" +
    "hyphens denote a range of atoms\n" +
    "commas separate individual atoms or ranges\n" +
    "default: whole structure"
)

translate_mode = translate_parser.add_argument_group(
    "translation mode (default: move centroid to origin)"
)
trans_modes = translate_mode.add_mutually_exclusive_group(required=True)
trans_modes.add_argument(
    "-v", "--vector",
    type=float,
    nargs=3,
    default=None,
    required=False,
    dest="vector",
    metavar=("x", "y", "z"),
    help="translate in direction of this vector\n" +
    "vector is normalized when --distance/-d is used"
)

translate_parser.add_argument(
    "-d", "--distance",
    type=float,
    default=None,
    required=False,
    dest="distance",
    help="distance translated - only applies to --vector/-v"
)

trans_modes.add_argument(
    "-dest", "--destination",
    type=float,
    nargs=3,
    default=None,
    required=False,
    dest="dest",
    metavar=("x", "y", "z"),
    help="translate fragment to a point"
)

center_parser = translate_parser.add_argument_group("center (default: centroid)")
center_option = center_parser.add_mutually_exclusive_group(required=False)
center_option.add_argument(
    "-com", "--center-of-mass",
    action="store_const",
    const=True,
    default=None,
    required=False,
    dest="com",
    help="translate the center of mass of the targets to the destination"
)

center_option.add_argument(
    "-cent", "--centroid",
    action="store_const",
    const=True,
    default=None,
    required=False,
    dest="cent",
    help="translate the centroid of the targets to the destination"
)

translate_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = translate_parser.parse_args()

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

    # targets to move
    targets = []
    if args.targets is not None:
        targets.extend(geom.find(args.targets))
    
    if args.fragments is not None:
        for atom in geom.find(args.fragments):
            frag_atoms = geom.get_all_connected(atom)
            targets.extend([frag_atom for frag_atom in frag_atoms if frag_atom not in targets])
    
    if not targets:
        targets = None

    # start with com or centroid where it is
    if args.com or not args.cent:
        com = geom.COM(targets=targets, mass_weight=True)
        start = com

    if args.cent:
        cent = geom.COM(targets=targets, mass_weight=False)
        start = cent

    # find where we are moving com or centroid to
    if args.dest is not None:
        # destination was specified
        destination = np.array(args.dest)
    elif args.vector is not None:
        # direction was specified
        if args.distance is not None:
            # magnitute was specified
            v = np.array(args.vector)
            v /= np.linalg.norm(v)
            destination = start + args.distance * v
        else:
            destination = start + np.array(args.vector)
    else:
        # nothing was specified - move to origin
        destination = np.zeros(3)

    translate_vector = destination - start

    geom.coord_shift(translate_vector, targets=targets)

    if args.outfile:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        geom.write(
            append=True,
            outfile=outfile,
        )
    else:
        print(geom.write(outfile=False))
