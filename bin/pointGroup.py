import sys
import argparse

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.symmetry import PointGroup

pg_parser = argparse.ArgumentParser(
    description="print point group",
    formatter_class=argparse.RawTextHelpFormatter
)

pg_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

pg_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

pg_parser.add_argument(
    "-t", "--tolerance",
    default=0.1,
    type=float,
    dest="tolerance",
    help="tolerance for determining if a symmetry element is valid\n"
    "for the input structure(s)\nDefault: 0.1"
)

pg_parser.add_argument(
    "-a", "--angle-tolerance",
    default=0.01,
    type=float,
    dest="rotation_tolerance",
    help="tolerance for determining if two axes are coincident or orthogonal"
    "\nDefault: 0.01"
)

pg_parser.add_argument(
    "-n", "--max-n",
    default=6,
    type=int,
    dest="max_n",
    help="max. order for proper rotation axes (improper rotations can be 2x this)"
    "\nDefault: 6"
)

pg_parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    default=False,
    dest="print_eles",
    help="print all symmetry elements",
)

args = pg_parser.parse_args()

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=False)

    geom = Geometry(infile)
    pg = PointGroup(
        geom,
        tolerance=args.tolerance,
        rotation_tolerance=args.rotation_tolerance,
        max_rotation=args.max_n,
    )
    print(f, pg.name)
    if args.print_eles:
        for ele in sorted(pg.elements, reverse=True):
            print("\t%s" % repr(ele))
