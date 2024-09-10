#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import glob_files

combiner_parser = argparse.ArgumentParser(
        description="combine monomers into a supermolecule (e.g. dimer, trimer, etc).  Conserves monomer atom positions and order. Intended to build non-bonded complexes so does not check for clashes, etc.",
        formatter_class=argparse.RawTextHelpFormatter
)

combiner_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

combiner_parser.add_argument(
    "-a", "--append",
    action="store_true",
    default=False,
    required=False,
    dest="append",
    help="append structure to output file if it already exists\nDefault: false"
)

combiner_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "Default: stdout"
)

combiner_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

args = combiner_parser.parse_args()

outGeom = Geometry()

for f in glob_files(args.infile, parser = combiner_parser):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f), get_all=True)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), get_all=True)

    if len(infile.all_geom) == 0:
        geom = Geometry(infile, refresh_ranks=False)
        outGeom += geom
    else: # account for multiple structures from stdin
        geom_list = []
        for struc in infile.all_geom:
            outGeom += Geometry(struc["atoms"])

if not args.outfile:
    print(outGeom.write(outfile=False))
else:
    outfile = args.outfile
    outGeom.write(append=args.append, outfile=outfile)
