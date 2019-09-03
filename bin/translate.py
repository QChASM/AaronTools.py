#!/usr/bin/env python3

import sys
import os
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileWriter, FileReader

def range2int(s):
    """split on ',' and turn '-' into a range
    range2int(['1,3-5,7']) returns [0, 2, 3, 4, 6]
    returns None if input is None"""
    if s is None:
        return s

    if isinstance(s, list):
        range_str = ','.join(s)
    elif isinstance(s, str):
        range_str = s

    out = []
    c = range_str.split(',')
    for v in c:
        n = v.split('-')
        if len(n) == 2:
            out.extend([i for i in range(int(n[0])-1, int(n[1]))])
        else:
            for i in n:
                out.append(int(i)-1)

    return out

translate_parser = argparse.ArgumentParser(description="translate a fragment or molecule's coordinates", \
    formatter_class=argparse.RawTextHelpFormatter)
translate_parser.add_argument('infile', metavar='input file', \
                         type=str, \
                         nargs='*', \
                         default=[sys.stdin], \
                         help='a coordinate file')

translate_parser.add_argument('-if', '--input-format', \
                        type=str, \
                        nargs=1, \
                        default=None, \
                        dest='input_format', \
                        help="file format of input - required if input is stdin")

translate_parser.add_argument('-t', '--targets',\
                        type=str, \
                        nargs=1, \
                        default=None, \
                        required=False, \
                        dest='targets', \
                        metavar='targets', \
                        help='target atoms on input (1-indexed)\n' + \
                        'comma (,) and/or hyphen (-) separated list\n' + \
                        'hyphens denote a range of atoms\n' + \
                        'commas separate individual atoms or ranges\n' + \
                        'Default: whole structure')

translate_parser.add_argument('-f', '--fragment',\
                        type=str, \
                        nargs=1, \
                        default=None, \
                        required=False, \
                        dest='fragment', \
                        metavar='targets', \
                        help='fragment to move (default: whole structure)')

translate_parser.add_argument('-v', '--vector',\
                        type=float, \
                        nargs=3, \
                        default=None, \
                        required=False, \
                        dest='vector', \
                        metavar=('x', 'y', 'z'), \
                        help='translate in direction of this vector - vector is normalized when "-d" is used')

translate_parser.add_argument('-d', '--distance',\
                        type=float, \
                        nargs=1, \
                        default=None, \
                        required=False, \
                        dest='distance', \
                        help='translate targets to a point')

translate_parser.add_argument('-dest', '--destination',\
                        type=float, \
                        nargs=3, \
                        default=None, \
                        required=False, \
                        dest='dest', \
                        metavar=('x', 'y', 'z'), \
                        help='translate fragment to a point \nDefault: COM of targets')

translate_parser.add_argument('-com', '--center-of-mass',\
                        action='store_const', \
                        const=True, \
                        default=None, \
                        required=False, \
                        dest='com', \
                        help='translate the center of mass of the targets to the destination')

translate_parser.add_argument('-cent', '--centroid',\
                        action='store_const', \
                        const=True, \
                        default=None, \
                        required=False, \
                        dest='cent', \
                        help='translate the centroid of the targets to the destination')

translate_parser.add_argument('-o', '--output', \
                        type=str, \
                        nargs=1, \
                        default=[False], \
                        required=False, \
                        dest='outfile', \
                        help='output destination\nDefault: stdout')

args = translate_parser.parse_args()

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
            translate_parser.print_help()
            raise RuntimeError("when no input file is given, stdin is read and a format must be specified")

    geom = Geometry(infile)

    #targets = None means whole geometry is used
    if args.targets is not None:
        targets = [geom.atoms[i] for i in range2int(args.targets[0])]
    else:
        targets = None

    #fragment = None means whole geometry is moved
    if args.fragment is not None:
        fragment = [geom.atoms[i] for i in range2int(args.fragment[0])]
    else:
        fragment = None

    #start with com or centroid where it is
    if args.com or not args.cent:
        com = geom.COM(targets=targets, mass_weight=True)
        start = com
    
    if args.cent:
        cent = geom.COM(targets=targets, mass_weight=False)
        start = cent

    #find where we are moving com or centroid to
    if args.dest is not None:
        #destination was specified
        destination = np.array(args.dest)
    elif args.vector is not None:
        #direction was specified
        if args.distance is not None:
            #magnitute was specified
            v = np.array(vector)
            v /= np.linalg.norm(v)
            destination = start + args.distance[0]*v
        else:
            destination = start + np.array(vector)
    else:
        #nothing was specified - move to origin
        destination = np.zeros(3)

    translate_vector = destination - start
    
    geom.coord_shift(translate_vector, targets=fragment)

    s = FileWriter.write_xyz(geom, append=True, outfile=args.outfile[0])
    if not args.outfile[0]:
        print(s)
