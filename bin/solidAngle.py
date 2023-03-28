#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.component import Component
from AaronTools.finders import AnyTransitionMetal
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import glob_files

# from cProfile import Profile

solid_parser = argparse.ArgumentParser(
    description="print ligand solid angle",
    formatter_class=argparse.RawTextHelpFormatter
)

solid_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

solid_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

solid_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin",
)

solid_parser.add_argument(
    "-k", "--key-atoms",
    type=str,
    default=None,
    dest="key_atoms",
    help="indices of ligand coordinating atoms you are calculating\n" +
    "the cone angle of (1-indexed)",
)

solid_parser.add_argument(
    "-c", "--center",
    type=str,
    default=AnyTransitionMetal(),
    dest="center",
    help="index of complex's center atom (1-indexed)\nDefault: transition metals",
)

solid_parser.add_argument(
    "-p", "--points",
    type=int,
    choices=[110, 194, 302, 590, 974, 1454, 2030, 2702, 5810],
    default=5810,
    dest="points",
    help="number of angular points for integration\n" +
    "lower values are faster, but at the cost of accuracy\nDefault: 5810"
)

solid_parser.add_argument(
    "-sca", "--solid-cone-angle",
    action="store_true",
    default=False,
    dest="solid_cone",
    help="print solid ligand cone angles instead of solid angles",
)

solid_parser.add_argument(
    "-vdw", "--vdw-radii",
    default="umn",
    choices=["umn", "bondi"],
    dest="radii",
    help="VDW radii to use in calculation\n" + 
    "umn: main group vdw radii from J. Phys. Chem. A 2009, 113, 19, 5806–5812\n" +
    "    (DOI: 10.1021/jp8111556)\n" + 
    "    transition metals are crystal radii from Batsanov, S.S. Van der Waals\n" +
    "    Radii of Elements. Inorganic Materials 37, 871–885 (2001).\n" +
    "    (DOI: 10.1023/A:1011625728803)\n" + 
    "bondi: radii from J. Phys. Chem. 1964, 68, 3, 441–451 (DOI: 10.1021/j100785a001)\n" +
    "Default: umn",
)


args = solid_parser.parse_args()

# profile = Profile()
# profile.enable()

s = ""

# args.radii = {
#     "P": 1.8,
#     "H": 1.2,
#     "C": 1.7,
#     "N": 1.55,
#     "O": 1.52,
#     "F": 1.47,
#     "S": 1.8,
#     "Cl": 1.75,
#     "Fe": 2.0,
# }

for f in glob_files(args.infile, parser=solid_parser):
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

    geom = Geometry(infile, refresh_ranks=False)

    ligand = geom.get_fragment(args.key_atoms, stop=args.center)

    comp = Component(
        ligand,
        key_atoms=args.key_atoms,
        detect_backbone=False,
        refresh_ranks=False,
        refresh_connected=False,
    )

    angle = comp.solid_angle(
        center=geom.find(args.center),
        radii=args.radii,
        return_solid_cone=args.solid_cone,
        grid=args.points,
    )
    
    if len(args.infile) > 1:
        s += "%20s:\t" % f

    if args.solid_cone:
        s += "%4.2f\n" % angle
    else:
        s += "%4.3f\n" % angle
    
if not args.outfile:
    print(s.rstrip())
else:
    with open(args.outfile, "a") as f:
        f.write(s.rstrip())

# profile.disable()
# profile.print_stats()