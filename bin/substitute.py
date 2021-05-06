#!/usr/bin/env python3

import argparse
import sys

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import get_filename, glob_files

substitute_parser = argparse.ArgumentParser(
    description="replace an atom or substituent with another",
    formatter_class=argparse.RawTextHelpFormatter,
)
substitute_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

substitute_parser.add_argument(
    "-ls",
    "--list",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="list_avail",
    help="list available substituents",
)

substitute_parser.add_argument(
    "-if",
    "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin",
)

substitute_parser.add_argument(
    "-s",
    "--substitute",
    metavar="n=substituent",
    type=str,
    action="append",
    default=None,
    required=False,
    dest="substitutions",
    help="substitution instructions \n"
    + "n is the 1-indexed position of the starting position of the\n"
    + "substituent you are replacing\n"
    + "a substituent name prefixed by iupac: or smiles: (e.g. iupac:acetyl\n"
    + "or smiles:O=[N.]=O) will create the substituent from the\n"
    + "corresponding identifier",
)

substitute_parser.add_argument(
    "-m",
    "--minimize",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="mini",
    help="rotate substituents to try to minimize LJ energy",
)

substitute_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

args = substitute_parser.parse_args()

if args.list_avail:
    s = ""
    for i, name in enumerate(sorted(Substituent.list())):
        s += "%-20s" % name
        # if (i + 1) % 3 == 0:
        if (i + 1) % 1 == 0:
            s += "\n"

    print(s.strip())
    sys.exit(0)

for infile in glob_files(args.infile):
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format, infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(("from stdin", args.input_format, infile))
        else:
            f = FileReader(("from stdin", "xyz", infile))

    geom = Geometry(f)

    target_list = []
    for sub in args.substitutions:
        ndx_targets = sub.split("=")[0]
        target_list.append(geom.find(ndx_targets))

    for i, sub in enumerate(args.substitutions):
        ndx_target = target_list[i]
        sub_name = "=".join(sub.split("=")[1:])

        for target in ndx_target:
            if sub_name.lower().startswith("iupac:"):
                sub_name = ":".join(sub_name.split(":")[1:])
                sub = Substituent.from_string(sub_name, form="iupac")
            elif sub_name.lower().startswith("smiles:"):
                sub_name = ":".join(sub_name.split(":")[1:])
                sub = Substituent.from_string(sub_name, form="smiles")
            else:
                sub = Substituent(sub_name)

            # replace old substituent with new substituent
            geom.substitute(sub, target, minimize=args.mini)
            geom.refresh_connected()

    if args.outfile:
        outfile = args.outfile
        if "INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(infile))
        geom.write(
            append=True,
            outfile=outfile,
        )
    else:
        print(geom.write(outfile=False))
