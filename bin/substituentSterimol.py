#!/usr/bin/env python3

import argparse
import sys

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import get_filename

sterimol_parser = argparse.ArgumentParser(
    description="calculate B1, B5, and L sterimol parameters for substituents",
    formatter_class=argparse.RawTextHelpFormatter
)

sterimol_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

sterimol_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input\nxyz is assumed if input is stdin"
)

sterimol_parser.add_argument(
    "-s", "--substituent-atom",
    type=str,
    required=True,
    dest="targets",
    help="substituent atom\n" +
    "1-indexed position of the starting position of the\n" +
    "substituent of which you are calculating sterimol\nparameters"
)

sterimol_parser.add_argument(
    "-a", "--attached-to",
    type=str,
    required=True,
    dest="avoid",
    help="non-substituent atom\n" +
    "1-indexed position of the starting position of the atom\n" +
    "connected to the substituent of which you are calculating\n" +
    "sterimol parameters"
)

sterimol_parser.add_argument(
    "-r", "--radii",
    type=str,
    default="bondi",
    choices=["bondi", "umn"],
    dest="radii",
    help="van der Waals radii - Bondi or Truhlar et. al.\nDefault: bondi"
)

sterimol_parser.add_argument(
    "-v", "--vector",
    action="store_true",
    required=False,
    dest="vector",
    help="print Chimera bild file for vectors instead of\nparameter values"
)

sterimol_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = sterimol_parser.parse_args()

s = ""
if not args.vector:
    s += "B1\tB5\tL\tfile\n"

for infile in args.infile:
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format, infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(("from stdin", args.input_format, infile))
        else:
            f = FileReader(("from stdin", "xyz", infile))

    geom = Geometry(f)
    target = args.targets
    avoid = args.avoid
    end = geom.find(avoid)[0]
    frag = geom.get_fragment(target, stop=end)
    sub = Substituent(frag, end=end, detect=False)
    b1 = sub.sterimol("B1", return_vector=args.vector, radii=args.radii)
    b5 = sub.sterimol("B5", return_vector=args.vector, radii=args.radii)
    l = sub.sterimol("L", return_vector=args.vector, radii=args.radii)
    if args.vector:
        start, end = b1
        s += ".color black\n"
        s += ".note Sterimol B1\n"
        s += ".arrow %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f\n" % (*start, *end)
        start, end = b5
        s += ".color red\n"
        s += ".note Sterimol B5\n"
        s += ".arrow %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f\n" % (*start, *end)
        start, end = l
        s += ".color blue\n"
        s += ".note Sterimol L\n"
        s += ".arrow %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f\n" % (*start, *end)
    else:
        s += "%.2f\t%.2f\t%.2f\t%s\n" % (b1, b5, l, infile)

if not args.outfile:
    print(s)
else:
    with open(
            args.outfile.replace("$INFILE", get_filename(infile)),
            "w"
    ) as f:
        f.write(s)
