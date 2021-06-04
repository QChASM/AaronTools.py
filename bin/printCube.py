#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

cube_parser = argparse.ArgumentParser(
    description="print a cube file for a molecular orbital",
    formatter_class=argparse.RawTextHelpFormatter
)

cube_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="an FCHK file or ORCA output with MO's"
)

cube_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

info = cube_parser.add_mutually_exclusive_group(required=False)
info.add_argument(
    "-mo", "--molecular-orbital",
    dest="mo_ndx",
    default=None,
    help="index of molecular orbital to print (0-indexed)\n"
    "Default: highest occupied MO in the ground state"
)
info.add_argument(
    "-ao", "--atomic-orbital",
    dest="ao_ndx",
    default=None,
    type=int,
    help="index of atomic orbital to print (0-indexed)"
)

cube_parser.add_argument(
    "-s", "--spacing",
    type=float,
    dest="spacing",
    default=0.35,
    help="spacing between points in the cube file\n"
    "Default: 0.35",
)

cube_parser.add_argument(
    "-p", "--padding",
    type=float,
    dest="padding",
    default=4,
    help="extra space around the molecule\n"
    "Default: 4"
)

cube_parser.add_argument(
    "-xyz", "--standard-axes",
    action="store_true",
    dest="xyz",
    default=False,
    help="use x, y, and z axes to define the directions\n"
    "Default: determine directions using SVD"
)

args = cube_parser.parse_args()

if args.mo_ndx and args.mo_ndx.isnumeric():
    args.mo_ndx = int(args.mo_ndx)

for f in glob_files(args.infile):
    if isinstance(f, str):
        infile = FileReader(f, just_geom=False)
    elif len(sys.argv) >= 1:
         infile = FileReader(("from stdin", "fchk", f), just_geom=False)

    geom = Geometry(infile, refresh_connected=False, refresh_ranks=False)

    out = geom.write(
        outfile=False,
        orbitals=infile.other["orbitals"],
        padding=args.padding,
        mo=args.mo_ndx,
        ao=args.ao_ndx,
        spacing=args.spacing,
        style="cube",
        xyz=args.xyz,
    )
    
    if not args.outfile:
        print(out)
    else:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        with open(outfile, "w") as f:
            f.write(out)

