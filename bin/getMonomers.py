#!/usr/bin/env python3

import sys
from os.path import dirname
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import (
    get_filename,
    glob_files,
    get_outfile,
)


monomer_parser = argparse.ArgumentParser(
    description="split a structure into monomers",
    formatter_class=argparse.RawTextHelpFormatter
)

monomer_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

monomer_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "$INDIR will be replaced with the directory of the input file\n" +
    "$MONOMER will be replaced with the index of the monomer\n" +
    "Default: stdout"
)

monomer_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

monomer_parser.add_argument(
    "-a", "--append",
    action="store_true",
    default=False,
    required=False,
    dest="append",
    help="append structures to output file if it already exists\nDefault: false"
)

args = monomer_parser.parse_args()

for f in glob_files(args.infile, parser=monomer_parser):
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

    geom = Geometry(infile, refresh_ranks=False)
    
    for i, monomer in enumerate(sorted(geom.get_monomers(), key=len)):
        monomer = Geometry(monomer)
        if not args.outfile:
            print(monomer.write(outfile=False))
        else:
            outfile = get_outfile(
                args.outfile,
                INFILE=get_filename(f, include_parent_dir="$INDIR" not in args.outfile),
                INDIR=dirname(f),
                MONOMER=str(i + 1),
            )
            monomer.write(append=args.append, outfile=outfile)
