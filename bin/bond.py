#!/usr/bin/env python3

import sys
from os.path import dirname
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files, get_outfile

def two_atoms_and_a_float(vals):
    """check to see if argument is two ints and a float"""
    err = argparse.ArgumentTypeError(
        "error with atom/distance specification: %s\n" % " ".join(vals) +
        "expected -s/-c int int float\nexample: -c 5 6 +9001"
    )
    out = []
    if len(vals) != 3:
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


bond_parser = argparse.ArgumentParser(
    description="measure or modify distance between atoms",
    formatter_class=argparse.RawTextHelpFormatter
)

bond_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

bond_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

bond_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    metavar="output destination",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "$INDIR will be replaced with the directory of the input file\n" +
    "Default: stdout"
)

bond_parser.add_argument(
    "-m", "--measure",
    metavar=("atom1", "atom2"),
    action="append",
    type=str,
    nargs=2,
    default=[],
    required=False,
    dest="measure",
    help="measure and print distance between atoms (1-indexed)"
)

bond_parser.add_argument(
    "-c", "--change",
    metavar=("atom1", "atom2", "increment"),
    action="append",
    type=str,
    nargs=3,
    default=[],
    required=False,
    dest="change",
    help="change distance by the amount specified"
)

bond_parser.add_argument(
    "-s", "--set", metavar=("atom1", "atom2", "distance"),
    action="append",
    type=str,
    nargs=3,
    default=[],
    required=False,
    dest="set_dist",
    help="set distance to the amount specified"
)

bond_parser.add_argument(
    "-a", "--append",
    action="store_true",
    default=False,
    help="append structures to output file",
)

args = bond_parser.parse_args()

for f in glob_files(args.infile, parser=bond_parser):
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

    geom = Geometry(infile, refresh_connected=args.set_dist or args.change, refresh_ranks=False)

    #set bond to specified value
    for bond in args.set_dist:
        vals = two_atoms_and_a_float(bond)
        a1 = geom.atoms[vals[0]]
        a2 = geom.atoms[vals[1]]
        geom.change_distance(a1, a2, dist=vals[2], adjust=False)

    #change bond by specified amount
    for bond in args.change:
        vals = two_atoms_and_a_float(bond)
        a1 = geom.atoms[vals[0]]
        a2 = geom.atoms[vals[1]]
        geom.change_distance(a1, a2, dist=vals[2], adjust=True)

    #print specified bonds
    out = ""
    for bond in args.measure:
        a1, a2 = geom.find(bond)
        val = a1.dist(a2)
        out += "%f\n" % val

    if len(args.set_dist) + len(args.change) > 0:
        if args.outfile:
            outfile = args.outfile
            if isinstance(f, str):
                outfile = get_outfile(
                    args.outfile,
                    INFILE=get_filename(f, include_parent_dir="$INDIR" not in args.outfile),
                    INDIR=dirname(f),
                )
            geom.write(append=args.append, outfile=outfile)
        else:
            print(geom.write(outfile=False))

    if len(args.measure) > 0:
        if args.outfile:
            outfile = args.outfile
            if isinstance(f, str): # apply substitutions if a file path was given as input
                outfile = get_outfile(
                    args.outfile,
                    INFILE=get_filename(f, include_parent_dir="$INDIR" not in args.outfile),
                    INDIR=dirname(f),
                )
            with open(outfile, "a" if args.append else "w") as f:
                f.write(out)
        else:
            out = out.rstrip()
            print(out)
