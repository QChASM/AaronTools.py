#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import glob_files

info_parser = argparse.ArgumentParser(
    description="print information in Gaussian, ORCA, or Psi4 output files",
    formatter_class=argparse.RawTextHelpFormatter
)

info_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

info_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination \nDefault: stdout"
)

info_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

info_parser.add_argument(
    "-ls", "--list",
    action="store_true",
    default=False,
    required=False,
    dest="list",
    help="list info categories and exit",
)

info_parser.add_argument(
    "-i", "--info",
    type=str,
    default=[],
    action="append",
    required=False,
    dest="info",
    help="information to print\n" +
    "Default is all info"
)

info_parser.add_argument(
    "-csv",
    "--csv-format",
    nargs="?",
    default=False,
    choices=("comma", "semicolon", "tab", "space"),
    required=False,
    dest="csv",
    help="print info in CSV format with the specified separator\n" +
    "Default: do not print in CSV format",
)

args = info_parser.parse_args()

if args.csv is None:
    args.csv = "comma"

if args.csv:
    if args.csv == "comma":
        sep = ","
    elif args.csv == "tab":
        sep = "\t"
    elif args.csv == "semicolon":
        sep = ";"
    else:
        sep = " "

s = ""

np.set_printoptions(precision=5)

for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=False)

    if args.list:
        s += "info in %s:\n" % f
        for key in infile.other.keys():
            s += "\t%s\n" % key

    else:
        s += "%s:\n" % f
        missing_keys = [key for key in args.info if key not in infile.other.keys()]
        if missing_keys:
            s += "\nmissing some info: %s\n" % ", ".join(missing_keys)

        for key in infile.other.keys():
            if args.info == [] or any(key.lower() == info.lower() for info in args.info):
                if isinstance(infile.other[key], str):
                    if args.csv:
                        s += "\"%s\"%s%s\n" % (key, sep, infile.other[key])
                    else:
                        s += "\t%-30s=\t%s\n" % (key, infile.other[key])
                elif isinstance(infile.other[key], bool):
                    if args.csv:
                        s += "\"%s\"%s%s\n" % (key, sep, infile.other[key])
                    else:
                        s += "\t%-30s =\t%s\n" % (key, str(infile.other[key]))
                elif isinstance(infile.other[key], int):
                    if args.csv:
                        s += "\"%s\"%s%i\n" % (key, sep, infile.other[key])
                    else:
                        s += "\t%-30s =\t%i\n" % (key, infile.other[key])
                elif isinstance(infile.other[key], float):
                    if args.csv:
                        s += "\"%s\"%s%.8f\n" % (key, sep, infile.other[key])
                    else:
                        s += "\t%-30s =\t%.8f\n" % (key, infile.other[key])
                elif isinstance(infile.other[key], list):
                    if args.csv:
                        s += "\"%s\"%s%s\n" % (
                            key, sep, sep.join([str(x) for x in infile.other[key]])
                        )
                    else:
                        s += "\t%-30s =\t%s\n" % (
                            key, ", ".join([str(x) for x in infile.other[key]])
                        )
                elif isinstance(infile.other[key], np.ndarray):
                    if args.csv:
                        s += "\"%s\"%s" % (key, sep)
                        vectorized = np.reshape(infile.other[key], (infile.other[key].size,))
                        if isinstance(vectorized[0], float):
                            s += sep.join(["%11.8f" % x for x in vectorized])
                        else:
                            s += sep.join([str(x) for x in vectorized])
                        s += "\n"
                    else:
                        s += "\t%-30s =\n" % key
                        for line in str(infile.other[key]).splitlines():
                            s += "\t\t%s\n" % line

if not args.outfile:
    print(s.strip())
else:
    with open(args.outfile, "a") as f:
        f.write(s.strip())
