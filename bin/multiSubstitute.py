#!/usr/bin/env python3
"""basically cat_screen from the perl version"""

import argparse
import sys

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import get_filename, glob_files

substitute_parser = argparse.ArgumentParser(
    description="add or modify substituents with permutations",
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
    metavar="i,j,k,...=substituent1,substituent2,...",
    type=str,
    action="append",
    default=None,
    required=False,
    dest="substitutions",
    help="substitution instructions \n"
    + "i, j, k are the 1-indexed position of the starting position of the\n"
    + "substituent(s) you are replacing\n"
    + "a substituent name prefixed by iupac: or smiles: (e.g. iupac:acetyl\n"
    + "or smiles:O=[N.]=O) will create the substituent from the\n"
    + "corresponding identifier\n"
    + "substituents can be separated by commas to use all combinations",
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
    "$SUBSTITUTIONS will be replaced with the substitution info\n" +
    "Default: stdout"
)

args = substitute_parser.parse_args()

if args.list_avail:
    sub_list = ""
    for i, name in enumerate(sorted(Substituent.list())):
        sub_list += "%-20s" % name
        # if (i + 1) % 3 == 0:
        if (i + 1) % 1 == 0:
            sub_list += "\n"

    print(sub_list.strip())
    sys.exit(0)

for infile in glob_files(args.infile, parser=substitute_parser):
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
    sub_names = []
    # number of geometries is the product of the number of substituents
    # that are to be placed at each site
    total_geoms = 1
    # mod_array is used to determine which substituent combination to
    # use for the ith structure
    mod_array = []
    for sub in args.substitutions:
        ndx_targets = sub.split("=")[0]
        target_list.append([a.name for a in geom.find(ndx_targets)])
        sub_name = "=".join(sub.split("=")[1:])
        sub_names.append([x.strip() for x in sub_name.split(",")])
        total_geoms *= len(sub_names[-1])
        mod_array = [len(sub_names[-1]) * x for x in mod_array]
        mod_array.append(1)

    for i in range(0, total_geoms):
        new_geom = geom.copy()

        sub_str = ""

        # go through each substitution request combination
        for j, (ndx, sub_list) in enumerate(zip(target_list, sub_names)):
            sub_ndx = int(i / mod_array[j]) % len(sub_list)
            for target in ndx:
                sub_name = sub_list[sub_ndx]
                if sub_name.lower().startswith("iupac:"):
                    sub_name = ":".join(sub_name.split(":")[1:])
                    sub = Substituent.from_string(sub_name, form="iupac")
                elif sub_name.lower().startswith("smiles:"):
                    sub_name = ":".join(sub_name.split(":")[1:])
                    sub = Substituent.from_string(sub_name, form="smiles")
                else:
                    sub = Substituent(sub_name)

                # replace old substituent with new substituent
                new_geom.substitute(sub, target, minimize=args.mini)
                new_geom.refresh_connected()
            sub_str += "_%s=%s" % (",".join(ndx), sub_name)

        sub_str = sub_str.strip("_")

        if args.outfile:
            outfile = args.outfile
            # only append if the file name is probably not unique
            do_append = True
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(infile))
                do_append = False
            if "$SUBSTITUTIONS" in outfile:
                outfile = outfile.replace("$SUBSTITUTIONS", sub_str)
                do_append = False
            new_geom.write(
                append=do_append,
                outfile=outfile,
            )
        else:
            print(new_geom.write(outfile=False))
