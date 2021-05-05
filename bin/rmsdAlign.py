#!/usr/bin/env python3

import sys
import argparse
from warnings import warn
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

def range2int(s):
    """split on "," and turn "-" into a range
    range2int(["1,3-5,7"]) returns [0, 2, 3, 4, 6]
    returns None if input is None"""
    if s is None:
        return s

    if isinstance(s, list):
        range_str = ",".join(s)

    out = []
    c = range_str.split(",")
    for v in c:
        n = v.split("-")
        if len(n) == 2:
            out.extend([i for i in range(int(n[0])-1, int(n[1]))])
        else:
            for i in n:
                out.append(int(i)-1)

    return out

rmsd_parser = argparse.ArgumentParser(
    description="align structure to reference",
    formatter_class=argparse.RawTextHelpFormatter
)

rmsd_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

rmsd_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

rmsd_parser.add_argument(
    "-r", "--reference",
    type=str,
    default=None,
    dest="ref",
    help="reference structure"
)

rmsd_parser.add_argument(
    "-it", "--input-targets",
    type=str,
    default=None,
    required=False,
    dest="in_target",
    metavar="targets",
    help="target atoms on input (1-indexed)\n" +
    "comma (,) and/or hyphen (-) separated list\n" +
    "hyphens denote a range of atoms\n" +
    "commas separate individual atoms or ranges\n" +
    "Default: whole structure"
)

rmsd_parser.add_argument(
    "-rt", "--ref-targets",
    type=str,
    default=None,
    required=False,
    dest="ref_target",
    metavar="targets",
    help="target atoms on reference (1-indexed)"
)

output_options = rmsd_parser.add_argument_group("output options")
output_format = output_options.add_mutually_exclusive_group(required=False)
output_format.add_argument(
    "-v", "--value",
    action="store_true",
    required=False,
    dest="value_only",
    help="print RMSD only"
)

output_format.add_argument(
    "-csv", "--comma-seperated",
    action="store_true",
    required=False,
    dest="csv",
    help="print output in CSV format"
)

output_options.add_argument(
    "-d", "--delimiter",
    type=str,
    default="comma",
    dest="delimiter",
    choices=["comma", "semicolon", "tab", "space"],
    help="CSV delimiter"
)

rmsd_parser.add_argument(
    "-s", "--sort",
    action="store_true",
    required=False,
    dest="sort",
    help="sort atoms"
)

rmsd_parser.add_argument(
    "-n", "--non-hydrogen",
    action="store_true",
    required=False,
    dest="heavy",
    help="ignore hydrogen atoms"
)

output_options.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = rmsd_parser.parse_args()

if args.ref is not None:
    ref_geom = Geometry(args.ref)
else:
    rmsd_parser.print_help()
    raise RuntimeError("reference geometry was not specified")

if bool(args.in_target) ^ bool(args.ref_target):
    warn("targets may need to be specified for both input and reference")

if args.csv:
    if args.delimiter == "comma":
        delim = ","
    elif args.delimiter == "space":
        delim = " "
    elif args.delimiter == "semicolon":
        delim = ";"
    elif args.delimiter == "tab":
        delim = "\t"

    header = delim.join(["reference", "geometry", "RMSD"])

    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(header + "\n")
    else:
        print(header)


for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f))
        else:
            infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)

    # align
    rmsd = geom.RMSD(
        ref_geom,
        align=True,
        targets=args.in_target,
        ref_targets=args.ref_target,
        heavy_only=args.heavy,
        sort=args.sort
    )

    geom.comment = "rmsd = %f" % rmsd

    if not args.value_only and not args.csv:
        if args.outfile:
            outfile = args.outfile
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(f))
            geom.write(append=True, outfile=outfile)
        else:
            print(geom.write(outfile=False))

    elif args.value_only:
        if args.outfile:
            outfile = args.outfile
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(f))
            with open(outfile, "a") as f:
                f.write("%f\n" % rmsd)

        else:
            print("%f" % rmsd)

    elif args.csv:
        s = delim.join([ref_geom.name, geom.name, "%f" % rmsd])
        if args.outfile:
            outfile = args.outfile
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(f))
            with open(outfile, "a") as f:
                f.write(s + "\n")
        else:
            print(s)
