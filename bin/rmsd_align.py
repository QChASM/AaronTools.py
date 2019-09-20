#!/usr/bin/env python3

import sys
import os
import numpy as np
import argparse
from warnings import warn
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileWriter, FileReader, read_types

def range2int(s):
    """split on ',' and turn '-' into a range
    range2int(['1,3-5,7']) returns [0, 2, 3, 4, 6]
    returns None if input is None"""
    if s is None:
        return s

    if isinstance(s, list):
        range_str = ','.join(s)

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

rmsd_parser = argparse.ArgumentParser(description='align structure to reference', \
    formatter_class=argparse.RawTextHelpFormatter)
rmsd_parser.add_argument('infile', metavar='input file', \
                         type=str, \
                         nargs='*', \
                         default=[sys.stdin], \
                         help='a coordinate file')

rmsd_parser.add_argument('-if', '--input-format', \
                        type=str, \
                        nargs=1, \
                        default=None, \
                        choices=read_types, \
                        dest='input_format', \
                        help="file format of input - required if input is stdin")

rmsd_parser.add_argument('-r', '--reference' ,\
                        type=str, \
                        nargs=1, \
                        default=None, \
                        dest='ref', \
                        help="reference structure")

rmsd_parser.add_argument('-it', '--input-targets',\
                        type=str, \
                        nargs=1, \
                        default=None, \
                        required=False, \
                        dest='in_target', \
                        metavar='targets', \
                        help='target atoms on input (1-indexed)\n' + \
                        'comma (,) and/or hyphen (-) separated list\n' + \
                        'hyphens denote a range of atoms\n' + \
                        'commas separate individual atoms or ranges\n' + \
                        'Default: whole structure')

rmsd_parser.add_argument('-rt', '--ref-targets',\
                        type=str, \
                        nargs=1, \
                        default=None, \
                        required=False, \
                        dest='ref_target', \
                        metavar='targets', \
                        help='target atoms on reference (1-indexed)')

rmsd_parser.add_argument('-o', '--output', \
                        type=str, \
                        nargs=1, \
                        default=[False], \
                        required=False, \
                        dest='outfile', \
                        help='output destination\nDefault: stdout')

args = rmsd_parser.parse_args()

if args.ref is not None:
    ref_geom = Geometry(args.ref[0])
else:
    rmsd_parser.print_help()
    raise RuntimeError("reference geometry was not specified")

if bool(args.in_target) ^ bool(args.ref_target):
    warn("targets may need to be specified for both input and reference")

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
            rmsd_parser.print_help()
            raise RuntimeError("when no input file is given, stdin is read and a format must be specified")

    geom = Geometry(infile)

    #get a list of atom targets
    if args.in_target is None:
        t = None
    else:
        t = geom.find(args.in_target)

    if args.ref_target is None:
        r = None
    else:
        r = ref_geom.find(args.ref_target)

    #align
    rmsd = geom.RMSD(ref_geom, align=True, targets=t, ref_targets=r)

    geom.comment = "rmsd = %f" % rmsd

    s = FileWriter.write_xyz(geom, append=True, outfile=args.outfile[0])
    if not args.outfile[0]:
        print(s)
