#!/usr/bin/env python3

import sys
import argparse
from warnings import warn

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

vsepr_choices = [
        "point",
        "linear_1",
        "linear_2",
        "bent_2_tetrahedral",
        "bent_2_planar",
        "trigonal_planar",
        "bent_3_tetrahedral",
        "t_shaped",
        "tetrahedral",
        "sawhorse",
        "seesaw",
        "square_planar",
        "trigonal_pyramidal",
        "trigonal_bipyramidal",
        "square_pyramidal",
        "pentagonal",
        "hexagonal",
        "trigonal_prismatic",
        "pentagonal_pyramidal",
        "octahedral",
        "capped_octahedral",
        "hexagonal_pyramidal",
        "pentagonal_bipyramidal",
        "capped_trigonal_prismatic",
        "heptagonal",
        "hexagonal_bipyramidal",
        "heptagonal_pyramidal",
        "octagonal",
        "square_antiprismatic",
        "trigonal_dodecahedral",
        "capped_cube",
        "biaugmented_trigonal_prismatic",
        "cubic",
        "elongated_trigonal_bipyramidal",
        "capped_square_antiprismatic",
        "enneagonal",
        "heptagonal_bipyramidal",
        "hula-hoop",
        "triangular_cupola",
        "tridiminished_icosahedral",
        "muffin",
        "octagonal_pyramidal",
        "tricapped_trigonal_prismatic",
]

element_parser = argparse.ArgumentParser(
    description="change an element and/or adjust the VSEPR geometry",
    formatter_class=argparse.RawTextHelpFormatter
)

element_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

element_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

element_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

element_parser.add_argument(
    "-e", "--element",
    metavar="target=element",
    type=str,
    action="append",
    required=True,
    dest="targets",
    help="element to change into"
)

element_parser.add_argument(
    "-b", "--fix-bonds",
    action="store_true",
    required=False,
    dest="fix_bonds",
    help="adjust bond lengths for the new element"
)

element_parser.add_argument(
    "-c", "--change-hydrogens",
    nargs="?",
    required=False,
    default=[None],
    type=int,
    dest="change_hs",
    metavar=("N",),
    help="change the number of hydrogens by the specified amount\n" +
    "Specify nothing to automatically determine how many hydrogens\n" +
    "to add or remove. If nothing is specified, the new geometry will\n" +
    "also be determined automatically."
)

element_parser.add_argument(
    "-g", "--geometry",
    type=str,
    default=None,
    dest="geometry",
    choices=vsepr_choices,
    required=False,
    help="specify the geometry to use with the new element\n" +
    "if the argument is not supplied, the geometry will remain the same as\n" +
    "the previous element's, unless necessitated by an increase in hydrogens",
)

args = element_parser.parse_args()

fix_bonds = args.fix_bonds

if args.change_hs is None:
    adjust_structure = True
    if args.geometry is not None:
        warn(
            "a geometry was specified, but geometry is determined automatically\n" +
            "with the supplied arguments"
        )
else:
    if isinstance(args.change_hs, int):
        adjust_hs = args.change_hs
    else:
        adjust_hs = 0

    new_vsepr = args.geometry.replace("_", " ")

    if adjust_hs == 0 and new_vsepr is None:
        adjust_structure = False
    else:
        adjust_structure = (adjust_hs, new_vsepr)

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
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)

    target_list = []
    for sub in args.targets:
        ndx_targets = sub.split("=")[0]
        target_list.append(geom.find(ndx_targets))

    for i, target in enumerate(target_list):
        element = args.targets[i].split("=")[1]
        # changeElement will only change one at a time
        for single_target in target:
            geom.change_element(
                single_target,
                element,
                adjust_bonds=fix_bonds,
                adjust_hydrogens=adjust_structure
            )

    if args.outfile:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        geom.write(append=True, outfile=outfile)
    else:
        print(geom.write(outfile=False))
