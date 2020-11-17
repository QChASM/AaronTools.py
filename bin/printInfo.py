#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types

info_parser = argparse.ArgumentParser(description='print information in Gaussian, ORCA, or Psi4 output files', \
    formatter_class=argparse.RawTextHelpFormatter)
info_parser.add_argument('infile', metavar='input file', 
                         type=str, 
                         nargs='*', 
                         default=[sys.stdin], 
                         help='a coordinate file'
)

info_parser.add_argument('-o', '--output', 
                        type=str, 
                        default=False, 
                        required=False, 
                        dest='outfile', 
                        help='output destination \nDefault: stdout'
)

info_parser.add_argument('-if', '--input-format', 
                        type=str, 
                        default=None, 
                        dest='input_format', 
                        choices=read_types, 
                        help="file format of input - xyz is assumed if input is stdin"
)

info_parser.add_argument('-ls', '--list', 
                         action="store_true", 
                         default=False, 
                         required=False, 
                         dest='list', 
                         help='list info categories and exit',
)
info_parser.add_argument('-i', '--info', 
                        type=str, 
                        default=[], 
                        action="append", 
                        required=False, 
                        dest='info', 
                        help='information to print\n' + \
                             'Default is all info'
)

args = info_parser.parse_args()

s = ""

np.set_printoptions(precision=5)

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(('from stdin', args.input_format[0], f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(('from stdin', 'xyz', f), just_geom=False)

    if args.list:
        s += "info in %s:\n" % f
        for key in infile.other.keys():
            s += "\t%s\n" % key
    
    else:
        s += "%s:\n" % f
        missing_keys = [key for key in args.info if key not in infile.other.keys()]
        if len(missing_keys) > 0:
            s += "\nmissing some info: %s\n" % ", ".join(missing_keys)
            
        for key in infile.other.keys():
            if args.info == [] or any(key.lower() == info.lower() for info in args.info):
                if isinstance(infile.other[key], str):
                    s += "\t%-30s=\t%s\n" % (key, infile.other[key])
                elif isinstance(infile.other[key], bool):
                    s += "\t%-30s =\t%s\n" % (key, str(infile.other[key]))
                elif isinstance(infile.other[key], int):
                    s += "\t%-30s =\t%i\n" % (key, infile.other[key])
                elif isinstance(infile.other[key], float):
                    s += "\t%-30s =\t%.8f\n" % (key, infile.other[key])
                elif isinstance(infile.other[key], list):
                    s += "\t%-30s =\t%s\n" % (key, ", ".join([str(x) for x in infile.other[key]]))
                elif isinstance(infile.other[key], np.ndarray):
                    s += "\t%-30s =\n" % key
                    for line in str(infile.other[key]).splitlines():
                        s += "\t\t%s\n" % line
                
                if key == "frequency":
                    s += "\t%-30s =\n" % key
                    for mode in infile.other[key].data:
                        s += "\t\t%.5f\n" % mode.frequency

if not args.outfile:
    print(s)
else:
    with open(args.outfile, "w") as f:
        f.write(s)
