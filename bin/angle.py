#!/usr/bin/env python3

import sys
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

def three_atoms_and_a_float(vals):
    """check to see if argument is three ints and a float"""
    err = argparse.ArgumentTypeError(
        "error with atom/angle specification: %s\n" % " ".join(vals) +
        "expected -s/-c int int int float\n" +
        "example: -c 4 5 6 +180"
    )
    out = []
    if len(vals) != 4:
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


angle_parser = argparse.ArgumentParser(
    description="measure or modify 1-2-3 angles",
    formatter_class=argparse.RawTextHelpFormatter
)
angle_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

angle_parser.add_argument(
    "-if",
    "--input-format",
    type=str,
    nargs=1,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

angle_parser.add_argument(
    "-m",
    "--measure",
    metavar=("atom1", "center", "atom3"),
    action="append",
    type=int,
    nargs=3,
    default=[],
    required=False,
    dest="measure",
    help="measure and print 1-2-3 angle (1-indexed)"
)

angle_parser.add_argument(
    "-c",
    "--change",
    metavar=("atom1", "center", "atom3", "increment"),
    action="append",
    type=str,
    nargs=4,
    default=[],
    required=False,
    dest="change",
    help="change 1-2-3 angle by the amount specified"
)

angle_parser.add_argument(
    "-s",
    "--set",
    metavar=("atom1", "center", "atom3", "angle"),
    action="append",
    type=str,
    nargs=4,
    default=[],
    required=False,
    dest="set_ang",
    help="set 1-2-3 angle to the amount specified"
)

angle_parser.add_argument(
    "-r",
    "--radians",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="radians",
    help="work with radians instead of degrees"
)

angle_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    metavar="output destination",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = angle_parser.parse_args()

for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None))
        else:
            infile = FileReader(f)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f))
        else:
            infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)

    #set angle to specified value
    for angle in args.set_ang:
        vals = three_atoms_and_a_float(angle)
        a1 = geom.find(str(vals[0]))[0]
        a2 = geom.find(str(vals[1]))[0]
        a3 = geom.find(str(vals[2]))[0]
        geom.change_angle(a1, a2, a3, vals[3], radians=args.radians, adjust=False)

    #change angle by specified amount
    for angle in args.change:
        vals = three_atoms_and_a_float(angle)
        a1 = geom.find(str(vals[0]))[0]
        a2 = geom.find(str(vals[1]))[0]
        a3 = geom.find(str(vals[2]))[0]
        geom.change_angle(a1, a2, a3, vals[3], radians=args.radians, adjust=True)

    #print specified angles
    out = ""
    for angle in args.measure:
        a1 = geom.find(str(angle[0]))[0]
        a2 = geom.find(str(angle[1]))[0]
        a3 = geom.find(str(angle[2]))[0]
        val = geom.angle(a1, a2, a3)
        if not args.radians:
            val *= 180 / np.pi

        out += "%f\n" % val

    out = out.rstrip()

    if len(args.set_ang) + len(args.change) > 0:
        if args.outfile:
            outfile = args.outfile
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(f))
            geom.write(append=True, outfile=outfile)
        else:
            print(geom.write(outfile=False))

    if len(args.measure) > 0:
        if not args.outfile:
            print(out)
        else:
            outfile = args.outfile
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(f))
            with open(outfile, "a") as f:
                f.write(out)