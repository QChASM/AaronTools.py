#!/usr/bin/env python3

import sys
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types

def two_atoms_and_a_float(vals):
    """check to see if argument is two ints and a float"""
    err = argparse.ArgumentTypeError("error with atom/distance specification: %s\n \
        expected -s/-c int int float\n \
        example: -c 5 6 +9001" % ' '.join(vals))
    out = []
    if len(vals) != 3:
        raise err
    for v in vals[:-1]:
        try:
            out.append(int(v)-1)
        except ValueError:
            raise err

        if int(v) != float(v):
            raise err

    try:
        out.append(float(vals[-1]))
    except ValueError:
        raise err

    return out
      

bond_parser = argparse.ArgumentParser(description='measure or modify distance between atoms', \
    formatter_class=argparse.RawTextHelpFormatter)
bond_parser.add_argument('infile', metavar='input file', \
                         type=str, \
                         nargs='*', \
                         default=[sys.stdin], \
                         help='a coordinate file')

bond_parser.add_argument('-if', '--input-format', \
                        type=str, \
                        nargs=1, \
                        default=None, \
                        choices=read_types, \
                        dest='input_format', \
                        help="file format of input - xyz is assumed if input is stdin")

bond_parser.add_argument('-m', '--measure', metavar=('atom1', 'atom2'),\
                        action='append', \
                        type=int, \
                        nargs=2, \
                        default=[], \
                        required=False, \
                        dest='measure', \
                        help='measure and print distance between atoms (1-indexed)')

bond_parser.add_argument('-c', '--change', metavar=('atom1', 'atom2', 'increment'),\
                        action='append', \
                        type=str, \
                        nargs=3, \
                        default=[], \
                        required=False, \
                        dest='change', \
                        help='change distance by the amount specified')

bond_parser.add_argument('-s', '--set', metavar=('atom1', 'atom2', 'distance'),\
                        action='append', \
                        type=str, \
                        nargs=3, \
                        default=[], \
                        required=False, \
                        dest='set_ang', \
                        help='set distance to the amount specified')

bond_parser.add_argument('-o', '--output', \
                        type=str, \
                        nargs=1, \
                        default=[False], \
                        required=False, \
                        dest='outfile', \
                        metavar='output destination', \
                        help='output destination\nDefault: stdout')

args = bond_parser.parse_args()

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
            infile = FileReader(('from stdin', 'xyz', f))

    geom = Geometry(infile)

    #set bond to specified value
    for bond in args.set_ang:
        vals = two_atoms_and_a_float(bond)
        a1 = geom.atoms[vals[0]]
        a2 = geom.atoms[vals[1]]
        geom.change_distance(a1, a2, dist=vals[2], adjust=False)

    #change bond by specified amount
    for bond in args.change:
        vals = two_atoms_and_a_float(bond)
        a1 = geom.atoms[vals[0]]
        a2 = geom.atoms[vals[1]]
        geom.change_distance(a1, a2, dist=vals[2], adjust=True)
    
    #print specified bonds
    out = ''
    for bond in args.measure:
        a1 = geom.atoms[bond[0]-1]
        a2 = geom.atoms[bond[1]-1]
        val = a1.dist(a2)
        out += "%f\n" % val

    out = out.rstrip()

    if len(args.set_ang) + len(args.change) > 0:
        s = geom.write(append=True, outfile=args.outfile[0])
        if not args.outfile[0]:
            print(s)
    
    if len(args.measure) > 0:
        if not args.outfile[0]:
            print(out)
        else:
            with open(args.outfile[0], 'w') as f:
                f.write(out)

