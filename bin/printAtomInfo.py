#!/usr/bin/env python3

import sys
import argparse
import re
import numpy as np

from AaronTools.fileIO import FileReader, read_types
from AaronTools.finders import AnyTransitionMetal, AnyNonTransitionMetal
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import glob_files

info_parser = argparse.ArgumentParser(
    description="print information about atoms in Gaussian, ORCA, or Psi4 output files",
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
    "-t", "--targets",
    type=str,
    default=[AnyTransitionMetal(), AnyNonTransitionMetal()],
    dest="targets",
    help="print info from target atoms",
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

for f in glob_files(args.infile, parser=info_parser):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=False)

    if args.list:
        s += "info in %s:\n" % f
        for key in infile.other.keys():
            val = infile.other[key]
            if not isinstance(val, str) and hasattr(val, "__iter__") and len(val) == len(infile.atoms):
                s += "\t%s\n" % key

    else:
        geom = Geometry(infile)
        try:
            atoms = geom.find(args.targets)
        except LookupError:
            print("%s not found in %s" % (args.targets, f))
            continue
        ndx_list = [geom.atoms.index(atom) for atom in atoms]
        s += "%s:\n" % f
        missing_keys = [
            key for key in args.info if not any(
                re.search(key, data_key, flags=re.IGNORECASE) for data_key in infile.other.keys()
            )
        ]
        if missing_keys:
            s += "\nmissing some info: %s\n" % ", ".join(missing_keys)

        for key in infile.other.keys():
            if args.info == [] or any(re.search(info, key, flags=re.IGNORECASE) for info in args.info):
                val = infile.other[key]
                if not isinstance(val, str) and hasattr(val, "__iter__") and len(val) == len(infile.atoms):
                    if args.csv:
                        s += "\"%s\"%s%s\n" % (
                            key, sep, sep.join([str(x) for i, x in enumerate(val) if i in ndx_list])
                        )
                    else:
                        s += "\t%-30s =\t%s\n" % (
                            key, ", ".join([str(x) for i, x in enumerate(val) if i in ndx_list])
                        )

if not args.outfile:
    print(s.strip())
else:
    with open(args.outfile, "a") as f:
        f.write(s.strip())
