#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename

xyz_parser = argparse.ArgumentParser(
    description="print structure in xyz format",
    formatter_class=argparse.RawTextHelpFormatter
)

xyz_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

xyz_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

xyz_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

xyz_parser.add_argument(
    "-c", "--comment",
    type=str,
    default="",
    required=False,
    dest="comment",
    help="comment line"
)

args = xyz_parser.parse_args()

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f))
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile, refresh_connected=False, refresh_ranks=False)
    if args.comment:
        geom.comment = args.comment
    else:
        geom.comment = f


    if not args.outfile:
        print(geom.write(outfile=False))
    else:
        geom.write(
            append=True,
            outfile=args.outfile.replace("$INFILE", get_filename(f))
        )
