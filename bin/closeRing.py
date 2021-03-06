#!/usr/bin/env python3

import argparse
from sys import stdin

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.ring import Ring
from AaronTools.utils.utils import get_filename

ring_parser = argparse.ArgumentParser(
    description="close rings on a geometry",
    formatter_class=argparse.RawTextHelpFormatter
)

ring_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[stdin],
    help="a coordinate file"
)

ring_parser.add_argument(
    "-ls", "--list",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="list_avail",
    help="list available rings"
)

ring_parser.add_argument(
    "-if", "--input-format",
    type=str,
    nargs=1,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

ring_parser.add_argument(
    "-r", "--ring",
    metavar=("atom1", "atom2", "ring"),
    type=str,
    nargs=3,
    action="append",
    default=None,
    required=False,
    dest="substitutions",
    help="substitution instructions \n" +
    "atom1 and atom2 specify the position to add the new ring"
)

ring_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = ring_parser.parse_args()

if args.list_avail:
    s = ""
    for i, name in enumerate(sorted(Ring.list())):
        s += "%-20s" % name
        # if (i + 1) % 3 == 0:
        if (i + 1) % 1 == 0:
            s += "\n"

    print(s.strip())
    exit(0)

for infile in args.infile:
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format, infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(("from stdin", args.input_format, stdin))
        else:
            f = FileReader(("from stdin", "xyz", stdin))

    geom = Geometry(f)

    targets = {}

    for sub_info in args.substitutions:
        atom1 = geom.find(sub_info[0])[0]
        atom2 = geom.find(sub_info[1])[0]
        ring = sub_info[2]

        ring_geom = Ring(ring)

        key = (atom1, atom2)
        if key in targets:
            targets[key].append(ring_geom)
        else:
            targets[key] = [ring_geom]

    for key in targets:
        for ring_geom in targets[key]:
            geom.ring_substitute(list(key), ring_geom)

    if args.outfile:
        geom.write(
            append=True,
            outfile=args.outfile.replace("$INFILE", get_filename(infile))
        )
    else:
        print(geom.write(outfile=False))
