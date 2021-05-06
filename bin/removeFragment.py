#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from AaronTools.geometry import Geometry
from AaronTools.fileIO import read_types, FileReader
from AaronTools.utils.utils import glob_files, get_filename


remove_frag_parser = argparse.ArgumentParser(
    description="remove a fragment from a molecule",
    formatter_class=argparse.RawTextHelpFormatter,
)

remove_frag_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

remove_frag_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin",
)

remove_frag_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout",
)

remove_frag_parser.add_argument(
    "-t",
    "--targets",
    type=str,
    required=True,
    dest="target",
    help="fragment atom connected to the rest of the molecule (1-indexed)",
)

remove_frag_parser.add_argument(
    "-k",
    "--keep-group",
    type=str,
    required=False,
    default=None,
    dest="avoid",
    help="atom on the molecule that is connected to the fragment being removed\nDefault: longest fragment",
)

remove_frag_parser.add_argument(
    "-a",
    "--add-hydrogen",
    action="store_true",
    required=False,
    default=False,
    dest="add_H",
    help="add hydrogen to cap where the fragment was removed",
)

args = remove_frag_parser.parse_args()

for infile in glob_files(args.infile):
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format[0], infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(("from stdin", args.input_format[0], infile))
        else:
            f = FileReader(("from stdin", "xyz", infile))

    geom = Geometry(f)
    
    for atom in geom.find(args.target):
        if atom in geom.atoms:
            geom.remove_fragment(atom, avoid=args.avoid, add_H=args.add_H)
            if not args.add_H:
                geom -= atom
    
    if args.outfile:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(infile))
        geom.write(append=False, outfile=outfile)
    else:
        s = geom.write(outfile=False)
        print(s)
