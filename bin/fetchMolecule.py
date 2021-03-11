#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry

fetch_parser = argparse.ArgumentParser(
    description="print structure in xyz format",
    formatter_class=argparse.RawTextHelpFormatter
)

fetch_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination \nDefault: stdout"
)

input_format = fetch_parser.add_argument_group("Identifier format")
style = input_format.add_mutually_exclusive_group(required=True)
style.add_argument(
    "-s", "--smiles",
    type=str,
    default=None,
    required=False,
    dest="smiles",
    help="SMILES notation for molecule"
)

style.add_argument(
    "-i", "--iupac",
    type=str,
    default=None,
    required=False,
    dest="iupac",
    help="IUPAC name of molecule"
)

args = fetch_parser.parse_args()

if args.smiles is not None:
    geom = Geometry.from_string(args.smiles, form="smiles")
elif args.iupac is not None:
    geom = Geometry.from_string(args.iupac, form="iupac")

if not args.outfile:
    print(geom.write(outfile=False))
else:
    geom.write(
        append=True, 
        outfile=args.outfile
    )
