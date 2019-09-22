#!/usr/bin/env python3

import sys
import os
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileWriter, FileReader, read_types

xyz_parser = argparse.ArgumentParser(description='print structure in xyz format', \
    formatter_class=argparse.RawTextHelpFormatter)
xyz_parser.add_argument('infile', metavar='input file', \
                         type=str, \
                         nargs='*', \
                         default=[sys.stdin], \
                         help='a coordinate file')

xyz_parser.add_argument('-o', '--output', \
                        type=str, \
                        nargs=1, \
                        default=[False], \
                        required=False, \
                        dest='outfile', \
                        help='output destination \nDefault: stdout')

xyz_parser.add_argument('-if', '--input-format', \
                        type=str, \
                        nargs=1, \
                        default=None, \
                        dest='input_format', \
                        choices=read_types, \
                        help="file format of input - required if input is stdin")

xyz_parser.add_argument('-c', '--comment', \
                        type=str, \
                        nargs=1, \
                        default=[''], \
                        required=False, \
                        dest='comment', \
                        help='comment line')

args = xyz_parser.parse_args()

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(('from stdin', args.input_format[0], f))
        else:
            if len(sys.argv) >= 1:
                xyz_parser.print_help()
                raise RuntimeError("when no input file is given, stdin is read and a format must be specified")

    geom = Geometry(infile)
    if args.comment[0]:
        geom.comment = args.comment[0]
    else:
        geom.comment=f

    s = FileWriter.write_xyz(geom, append=True, outfile=args.outfile[0])
    if not args.outfile[0]:
        print(s)