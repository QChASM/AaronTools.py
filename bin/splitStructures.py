#!/usr/bin/env python3

import os
import sys
import argparse

import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

split_parser = argparse.ArgumentParser(
    description="print all structures from a given file",
    formatter_class=argparse.RawTextHelpFormatter
)

split_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

split_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "$N will be replaced with the number of the structure\n" +
    "Default: print to stdout"
)

split_parser.add_argument(
    "-p", "--include-parent-dir",
    action="store_true",
    default=False,
    dest="include_parent",
    help="include parent directory in $INFILE\n" +
    "Default: False",
)

split_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

args = split_parser.parse_args()

for f in glob_files(args.infile, parser=split_parser):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None), get_all=True)
        else:
            infile = FileReader(f, get_all=True)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f), get_all=True)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), get_all=True)

    zeros = max(1, int(np.log10(len(infile.all_geom) + 1) + 1))
    n_fmt = "%%0%ii" % zeros
    for i, struc in enumerate(infile.all_geom):
        geom = Geometry(
            struc["atoms"],
            refresh_connected=False,
            refresh_ranks=False,
        )

        if not args.outfile:
            print(geom.write(outfile=False))
        else:
            outfile = args.outfile
            outfile = outfile.replace(
                "$INFILE",
                get_filename(f, include_parent_dir=args.include_parent)
            )
            outfile = outfile.replace("$N", n_fmt % (i + 1))
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            geom.write(outfile=outfile)
