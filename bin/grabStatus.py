#!/usr/bin/env python3

import sys
import numpy as np
import argparse

from AaronTools.comp_output import CompOutput
from AaronTools.fileIO import FileReader


stat_parser = argparse.ArgumentParser(description='prints status of optimization job', \
    formatter_class=argparse.RawTextHelpFormatter)
stat_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default = [sys.stdin], \
                            help='input optimization file (i.e. Gaussian output where "opt" was specified)')

stat_parser.add_argument('-if', '--input-format', \
                        type=str, \
                        nargs=1, \
                        default=None, \
                        dest='input_format', \
                        choices=['log', 'out', 'dat'], \
                        help="file format of input - required if input is stdin")

stat_parser.add_argument('-o', '--output', \
                            type=str, \
                            nargs='+', \
                            default=None, \
                            required=False, \
                            dest='outfile', \
                            help='output destination\nDefault: stdout')

args = stat_parser.parse_args()

s = ""

header_vals = [None]

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
            stat_parser.print_help()
            raise RuntimeError("when no input file is given, stdin is read and a format must be specified")

    co = CompOutput(infile)
    if not all(x in header_vals for x in co.gradient.keys()) or not all(x in co.gradient for x in header_vals):
        header_vals = [x for x in sorted(co.gradient.keys())]
        header = "                      Filename  " + "  ".join(["%14s" % crit for crit in header_vals])
        header += '\n'
        s += header
    
    s += "%30s" % f

    for crit in header_vals:
        col = "%.2e/%s" % (float(co.gradient[crit]['value']), 'YES' if co.gradient[crit]['converged'] else 'NO')
        s += "  %14s" % col

    if "error" in infile.other and infile.other["error"] is not None and infile.other["error"] != "UNKNOWN":
        s += "  %s" % infile.other["error_msg"]
    elif len(header_vals) == 0:
        s += "  no progress found"
    elif co.finished:
        s += "  finished"
    else:
        s += "  not finished"

    s += '\n'
    

if args.outfile is None:
    print(s)
else:
    with open(args.outfile, 'a') as f:
        f.write(s)
