#!/usr/bin/env python3

import sys
from os.path import dirname
import argparse
import logging

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files, get_outfile


ring_conf_parser = argparse.ArgumentParser(
    description="measure or modify distance between atoms",
    formatter_class=argparse.RawTextHelpFormatter
)

ring_conf_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

ring_conf_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

ring_conf_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    metavar="output destination",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "$INDIR will be replaced with the directory of the input file\n" +
    "Default: stdout"
)

ring_conf_parser.add_argument(
    "-t", "--targets",
    type=str,
    default=None,
    dest="targets",
    metavar="targets",
    help="list of atoms in rings, for which conformers will be checked\n" +
    "Default: all atoms",
)

ring_conf_parser.add_argument(
    "-u", "--uncommon",
    action="store_true",
    default=False,
    dest="uncommon",
    help="include uncommon conformers (e.g. boat cyclohexanes)",
)

ring_conf_parser.add_argument(
    "-m", "--minimize",
    action="store_true",
    default=False,
    dest="minimize",
    help="rotate substituents to reduce steric clashing",
)

ring_conf_parser.add_argument(
    "-a", "--append",
    action="store_true",
    default=False,
    dest="append",
    help="append structures to output file",
)

args = ring_conf_parser.parse_args()

s = ""

for f in glob_files(args.infile, parser=ring_conf_parser):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f))
        else:
            infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)
    conformers = Geometry.ring_conformers(geom, targets=args.targets, include_uncommon=args.uncommon)

    for i, conf in enumerate(conformers):
        if args.minimize:
            conf.detect_substituents()
            conf.minimize_sub_torsion()
        if args.outfile:
            outfile = args.outfile
            if isinstance(f, str): # apply substitutions if a file path was given as input
                outfile = get_outfile(
                    args.outfile,
                    INFILE=get_filename(f, include_parent_dir="$INDIR" not in args.outfile),
                    INDIR=dirname(f),
                    i=str(i + 1)
                )
            conf.write(
                outfile=outfile,
                append=args.append or "$i" not in args.outfile,
            )
        else:
            s = conf.write(outfile=False)
            s += "\n"

if args.outfile is None:
    print(out)
    
