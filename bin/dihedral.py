#!/usr/bin/env python3

import sys
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

def four_atoms_and_a_float(vals):
    """check to see if argument is four numbers and a float"""
    err = argparse.ArgumentTypeError(
        "error with atom/dihedral specification: %s\n" % " ".join(vals) +
        "expected -s/-c int int int int float\n" +
        "example: -c 4 5 6 7 1.337"
    )
    out = []
    if len(vals) != 5:
        raise err
    for v in vals[:-1]:
        try:
            out.append(int(v))
        except ValueError:
            raise err

        if int(v) != float(v):
            raise err

    try:
        out.append(float(vals[-1]))
    except ValueError:
        raise err

    return out


dihedral_parser = argparse.ArgumentParser(
    description="measure or modify torsional angles",
    formatter_class=argparse.RawTextHelpFormatter
)

dihedral_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

dihedral_parser.add_argument(
    "-if", "--input-format",
    type=str,
    choices=read_types,
    default=None,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

dihedral_parser.add_argument(
    "-m", "--measure",
    metavar=("atom1", "atom2", "atom3", "atom4"),
    action="append",
    type=str,
    nargs=4,
    default=[],
    required=False,
    dest="measure",
    help="measure and print the torsional angle (1-indexed)"
)

dihedral_parser.add_argument(
    "-c", "--change",
    metavar=("atom1", "atom2", "atom3", "atom4", "increment"),
    action="append",
    type=str,
    nargs=5,
    default=[],
    required=False,
    dest="change",
    help="change torsional angle by the amount specified"
)

dihedral_parser.add_argument(
    "-s", "--set",
    metavar=("atom1", "atom2", "atom3", "atom4", "angle"),
    action="append",
    type=str,
    nargs=5,
    default=[],
    required=False,
    dest="set_ang",
    help="set dihedral to the amount specified"
)

dihedral_parser.add_argument(
    "-r", "--radians",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="radians",
    help="work with radians instead of degrees"
)

dihedral_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    metavar="output destination",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = dihedral_parser.parse_args()

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

    # set dihedral to specified value
    for dihedral in args.set_ang:
        vals = four_atoms_and_a_float(dihedral)
        a1 = geom.find(str(vals[0]))[0]
        a2 = geom.find(str(vals[1]))[0]
        a3 = geom.find(str(vals[2]))[0]
        a4 = geom.find(str(vals[3]))[0]
        geom.change_dihedral(a1, a2, a3, a4, vals[4], radians=args.radians, adjust=False, as_group=True)

    #change dihedral by specified amount
    for dihedral in args.change:
        vals = four_atoms_and_a_float(dihedral)
        a1 = geom.find(str(vals[0]))[0]
        a2 = geom.find(str(vals[1]))[0]
        a3 = geom.find(str(vals[2]))[0]
        a4 = geom.find(str(vals[3]))[0]
        geom.change_dihedral(a1, a2, a3, a4, vals[4], radians=args.radians, adjust=True, as_group=True)

    #print specified dihedrals
    if len(args.infile) > 1:
        out = "%s\t" % f
    else:
        out = ""
    for dihedral in args.measure:
        a1 = geom.find(dihedral[0])[0]
        a2 = geom.find(dihedral[1])[0]
        a3 = geom.find(dihedral[2])[0]
        a4 = geom.find(dihedral[3])[0]
        val = geom.dihedral(a1, a2, a3, a4)
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