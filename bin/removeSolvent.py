#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from AaronTools.geometry import Geometry
from AaronTools.fileIO import read_types, FileReader
from AaronTools.utils.utils import (
    get_filename,
    glob_files,
    get_outfile,
)

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
    "$INDIR will be replaced with the directory of the input file\n" +
    "Default: stdout",
)

remove_frag_parser.add_argument(
    "-s",
    "--solvent",
    type=str,
    required=True,
    dest="solvent_mol",
    help="solvent molecule to remove",
)


args = remove_frag_parser.parse_args()

sol = Geometry(args.solvent_mol)
sol_elements = sol.element_counts()

for infile in glob_files(args.infile, parser=remove_frag_parser):
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

    monomers = geom.get_monomers()
    keep_monomers = []
    for monomer in monomers:
        if len(monomer) != len(sol.atoms):
            keep_monomers.append(monomer)
    
    if args.outfile:
        outfile = get_outfile(
            args.outfile,
            INFILE=get_filename(infile, include_parent_dir="$INDIR" not in args.outfile),
            INDIR=os.path.dirname(infile),
        )
        geom.write(append=False, outfile=outfile)
    else:
        s = geom.write(outfile=False)
        print(s)
