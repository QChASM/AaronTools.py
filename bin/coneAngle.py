#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.component import Component
from AaronTools.finders import AnyTransitionMetal
from AaronTools.fileIO import FileReader, read_types

cone_parser = argparse.ArgumentParser(
    description="calculate ligand cone angles",
    formatter_class=argparse.RawTextHelpFormatter
)

cone_parser = argparse.ArgumentParser(
    description="print Gaussian, ORCA, or Psi4 input file",
    formatter_class=argparse.RawTextHelpFormatter
)

cone_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

cone_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

cone_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin",
)

cone_parser.add_argument(
    "-k", "--key-atoms",
    type=str,
    default=None,
    dest="key_atoms",
    help="indices of ligand coordinating atoms you are calculating\n" +
    "the cone angle of (1-indexed)",
)

cone_parser.add_argument(
    "-c", "--center",
    type=str,
    default=AnyTransitionMetal(),
    dest="center",
    help="index of complex's center atom (1-indexed)\nDefault: transition metals",
)

cone_parser.add_argument(
    "-m", "--method",
    type=lambda x: x.capitalize(),
    choices=["Tolman", "Exact"],
    default="exact",
    dest="method",
    help="cone angle type\n" +
    "Tolman: Tolman's method for asymmetric mono- and bidentate ligands\n" +
    "        see doi 10.1021/ja00808a009\n" +
    "Exact: (Default) Allen's method for an all-encompassing cone\n" +
    "       see doi 10.1002/jcc.23217",
)

cone_parser.add_argument(
    "-r", "--vdw-radii",
    default="umn",
    choices=["umn", "bondi"],
    dest="radii",
    help="VDW radii to use in calculation\nDefault: umn",
)

cone_parser.add_argument(
    "-b", "--cone-bild",
    action="store_true",
    default=False,
    dest="print_cones",
    help="print Chimera bild file containing cones",
)

args = cone_parser.parse_args()

s = ""

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None))
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f))
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)

    ligand = geom.get_fragment(args.key_atoms, stop=args.center)

    comp = Component(ligand, key_atoms=args.key_atoms)

    angle = comp.cone_angle(
        center=geom.find(args.center),
        method=args.method,
        radii=args.radii,
        return_cones=args.print_cones,
    )

    if args.print_cones:
        angle, cones = angle

    if len(args.infile) > 1:
        s += "%20s:\t" % f

    s += "%4.1f\n" % angle
    
    if args.print_cones:
        s += ".transparency 0.5\n"
        for cone in cones:
            print(cone)
            apex, base, radius = cone
            s += ".cone   %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f   %.3f open\n" % (
                *apex, *base, radius
            )

if not args.outfile:
    print(s.rstrip())
else:
    with open(args.outfile, "a") as f:
        f.write(s.rstrip())
