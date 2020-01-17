#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np

from AaronTools.ring import Ring

libaddring_parser = argparse.ArgumentParser(description='add a ring fragment to your personal library', \
                    formatter_class=argparse.RawTextHelpFormatter)
libaddring_parser.add_argument('infile', metavar='input file', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                help='a coordinate file')

libaddring_parser.add_argument('-n', '--name', \
                                type=str, \
                                nargs=1, \
                                required=False, \
                                default=[None], \
                                dest='name', \
                                help="""Name of ring being added to the library
if no name is given, the ring will be printed to STDOUT""")

libaddring_parser.add_argument('-w', '--walk', \
                                type=str, \
                                nargs=1, \
                                required=True, \
                                dest='walk', \
                                help="""comma-separated list of atoms to define 
the direction the ring is traversed (1-indexed)""")

args = libaddring_parser.parse_args()

name = args.name[0]
infile = args.infile[0]
walk = args.walk[0]

ring = Ring(infile)

walk_atoms = walk.split(',')
ring.find_end(len(walk_atoms), walk_atoms)

ring.comment = "E:%s" % walk

if name is None:
    print(ring.write(outfile=False))
else:
    ring_lib = Ring.AARON_LIBS
    ring_file = os.path.join(os.path.dirname(Ring.AARON_LIBS), name + '.xyz')
    if os.path.exists(ring_file):
        overwrite = input("%s already exists.\nWould you like to overwrite it? (YES/no)\n" % ring_file)
        if overwrite != "YES":
            print("%s to overwrite, not overwriting" % overwrite)
            sys.exit(0)
    
    ring.write(outfile=ring_file)
