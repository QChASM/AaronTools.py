#!/usr/bin/env python3

import sys
import os
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from warnings import warn

element_parser = argparse.ArgumentParser(description='change and element and/or adjust the VSEPR geometry at a site', \
    formatter_class=argparse.RawTextHelpFormatter)
element_parser.add_argument('infile', metavar='input file', \
                             type=str, \
                             nargs='*', \
                             default=[sys.stdin], \
                             help='a coordinate file')

element_parser.add_argument('-o', '--output', \
                            type=str, \
                            nargs=1, \
                            default=[False], \
                            required=False, \
                            dest='outfile', \
                            help='output destination \nDefault: stdout')

element_parser.add_argument('-if', '--input-format', \
                            type=str, \
                            nargs=1, \
                            default=None, \
                            dest='input_format', \
                            choices=read_types, \
                            help="file format of input - xyz is assumed if input is stdin")

element_parser.add_argument('-t', '--targets', \
                            type=str, \
                            nargs=1, \
                            required=True, \
                            dest='targets', \
                            help='comma- or hyphen-separated list of target atoms')

element_parser.add_argument('-e', '--element', \
                            required=True, \
                            dest='element', \
                            help='symbol for new element')

element_parser.add_argument('-b', '--fix-bonds', \
                            action='store_true', \
                            required=False, \
                            dest='fix_bonds', \
                            help='adjust bond lengths for the new element')

element_parser.add_argument('-c', '--change-hydrogens', \
                            nargs='?', \
                            required=False, \
                            default=[None], \
                            type=int, \
                            dest='change_hs', \
                            metavar=('N',), \
                            help='change the number of hydrogens by the specified amount\n' + \
                                 'Specify nothing to automatically determine how many hydrogens\n' + \
                                 'to add or remove. If nothing is specified, the new geometry will\n' + \
                                 'also be determined automatically.')

element_parser.add_argument('-g', '--geometry', \
                            nargs=1, \
                            type=str, \
                            default=[None], \
                            dest='geometry', \
                            required=False, \
                            help='specify the geometry to use with the new element\nif the argument is not supplied, the geometry will remain the same as the previous element\'s')

args = element_parser.parse_args()

element = args.element
fix_bonds = args.fix_bonds

if args.change_hs is None:
    adjust_structure = True
    if args.geometry[0] is not None:
        warn("a geometry was specified, but geometry is determined automatically with the supplied arguments")
else:
    if isinstance(args.change_hs, int):
        adjust_hs = args.change_hs
    else:
        adjust_hs = 0

    new_vsepr = args.geometry[0]

    if adjust_hs == 0 and new_vsepr is None:
        adjust_structure = False
    else:
        adjust_structure = (adjust_hs, new_vsepr)

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
                infile = FileReader(('from stdin', 'xyz', f))

    geom = Geometry(infile)

    targets = geom.find(args.targets[0])
    geom.change_element(targets, element, adjust_bonds=fix_bonds, adjust_hydrogens=adjust_structure)

    s = geom.write(append=True, outfile=args.outfile[0])
    if not args.outfile[0]:
        print(s)
