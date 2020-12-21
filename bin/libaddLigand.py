#!/usr/bin/env python3

import os
import argparse
import sys

from AaronTools.geometry import Geometry
from AaronTools.component import Component

libaddlig_parser = argparse.ArgumentParser(
    description='add a ligand to your personal library',
    formatter_class=argparse.RawTextHelpFormatter
)

libaddlig_parser.add_argument(
    'infile', metavar='input file',
    type=str,
    default=None,
    help='a coordinate file'
)

libaddlig_parser.add_argument(
    '-n', '--name',
    type=str,
    required=False,
    default=None,
    dest='name',
    help="Name of ligand being added to the library\n" +
    "if no name is given, the ligand will be printed to STDOUT"
)

args = libaddlig_parser.parse_args()

cat = Geometry(args.infile)

if cat.center is None:
    cat.detect_components()

#move center to origin
cat.coord_shift(-cat.COM(targets=cat.center))

ligands = cat.components
lig_index = 0

#if there's multiple ligands, ask which one we should add
if len(ligands) > 1:
    print("multiple ligands found:")
    for i, ligand in enumerate(ligands):
        print("ligand %i:" % i)
        print(ligand.write(outfile=False))

    lig_index = int(input("What is the number of the ligand you would like to add?\n"))

new_ligand = ligands[lig_index]
new_ligand.comment = "K:" + ",".join(
    [str(new_ligand.atoms.index(key) + 1) for key in new_ligand.key_atoms]
) + ';'

if args.name is None:
    print(new_ligand.write(outfile=False))
else:
    lig_file = os.path.join(os.path.dirname(Component.AARON_LIBS), args.name + '.xyz')
    if os.path.exists(lig_file):
        overwrite = input(
            "%s already exists.\nWould you like to overwrite it? (YES/no)\n" % lig_file
        )
        if overwrite != "YES":
            print("%s to overwrite, not overwriting" % overwrite)
            sys.exit(0)

    new_ligand.write(outfile=lig_file)
