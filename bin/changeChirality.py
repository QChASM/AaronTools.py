#!/usr/bin/env python3

import argparse
import os

from sys import stdin
from warnings import warn

from AaronTools.fileIO import FileReader, read_types
from AaronTools.finders import ChiralCenters, Bridgehead, SpiroCenters, NotAny
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import glob_files


changechiral_parser = argparse.ArgumentParser(
    description="change handedness of chiral centers",
    formatter_class=argparse.RawTextHelpFormatter
)

changechiral_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[stdin],
    help="a coordinate file"
)

changechiral_parser.add_argument(
    "-ls", "--list-chiral",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="list_info",
    help="list information on detected chiral centers"
)

changechiral_parser.add_argument(
    "-if", "--input-format",
    type=str,
    nargs=1,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

changechiral_parser.add_argument(
    "-o", "--output-destination",
    type=str,
    default=None,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$i in the filename will be replaced with a number\n" +
    "if a directory is given, default is \"diastereomer-$i.xyz\" in \n" +
    "that directory\n" +
    "Default: stdout"
)

changechiral_parser.add_argument(
    "-t", "--targets",
    type=str,
    default=None,
    action="append",
    required=False,
    dest="targets",
    help="comma- or hyphen-seperated list of chiral centers to invert (1-indexed)\n" +
    "Chiral centers must have at least two fragments not in a ring\n" +
    "Detected chiral centers are atoms that:\n" +
    "    - have > 2 bonds\n" +
    "    - have a non-planar VSEPR shape\n" +
    "    - each connected fragment is distinct or is a spiro center\n" +
    "Default: change chirality of any detected chiral centers"
)

changechiral_parser.add_argument(
    "-d", "--diastereomers",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="combos",
    help="print all diastereomers for selected chiral centers"
)

changechiral_parser.add_argument(
    "-m", "--minimize",
    action="store_true",
    default=False,
    dest="minimize",
    help="rotate substituents to mitigate steric clashing",
)


args = changechiral_parser.parse_args()

s = ""

for infile in glob_files(args.infile, parser=changechiral_parser):
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format[0], infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(("from stdin", args.input_format[0], stdin))
        else:
            f = FileReader(("from stdin", "xyz", stdin))

    geom = Geometry(f)

    target_list = []
    if args.targets is None:
        try:
            chiral_centers = geom.find(ChiralCenters())
            spiro_centers = geom.find(SpiroCenters(), chiral_centers)
            bridge_centers = geom.find(chiral_centers, Bridgehead(), NotAny(spiro_centers))
            target_list = [t for t in chiral_centers if t not in bridge_centers]
        except LookupError as e:
            warn(str(e))
    else:
        for targ in args.targets:
            target_list.extend(geom.find(targ))

    if args.list_info:
        if len(args.infile) > 1:
            s += "%s\n" % infile
        s += "Target\tElement\n"
        for targ in target_list:
            s += "%-2s\t%-2s\n" % (targ.name, targ.element)
        if infile is not args.infile[-1]:
            s += "\n"
        continue

    geom.substituents = []
    if args.combos:
        # this stuff is copy-pasted from makeConf, so it's a bit overkill
        # for getting all diastereomers, as each chiral center can only
        # have 2 options instead of the random number of rotamers
        # a substituent can have
        diastereomers = Geometry.get_diastereomers(geom, minimize=args.minimize)
        
        for i, diastereomer in enumerate(diastereomers):
            if args.outfile is None:
                s += diastereomer.write(outfile=False)
                s += "\n"
            else:
                if os.path.isdir(os.path.expanduser(args.outfile)):
                    outfile = os.path.join(
                        os.path.expanduser(args.outfile),
                        "diastereomer-%i.xyz" % (i + 1)
                    )
    
                else:
                    outfile = args.outfile.replace("$i", str(i + 1))
    
                diastereomer.write(outfile=outfile, append="$i" not in args.outfile)

    else:
        for targ in target_list:
            geom.change_chirality(targ)

        if args.minimize:
            geom.minimize_sub_torsion(increment=15)

        if args.outfile is None:
            s += geom.write(outfile=False)
            s += "\n"
        else:
            geom.write(outfile=args.outfile)


if args.outfile is None or args.list_info:
    print(s[:-1])
