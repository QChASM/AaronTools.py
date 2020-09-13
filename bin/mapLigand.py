#!/usr/bin/env python3

import argparse
from sys import exit, stdin

from AaronTools.component import Component
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry

maplig_parser = argparse.ArgumentParser(
    description="replace a ligand on an organometallic system",
    formatter_class=argparse.RawTextHelpFormatter,
)
maplig_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[stdin],
    help="a coordinate file",
)

maplig_parser.add_argument(
    "-ls",
    "--list",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="list_avail",
    help="list available ligands",
)

maplig_parser.add_argument(
    "-if",
    "--input-format",
    type=str,
    nargs=1,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin",
)

maplig_parser.add_argument(
    "-l",
    "--ligand",
    metavar="[n[,m...]]=ligand|ligand",
    type=str,
    nargs=1,
    default=None,
    required=False,
    dest="ligand",
    help="ligand used to replace the current one\n" + \
         "n[,m...] are the 1-indexed positions of the coordinating atoms of the\n" + \
         "ligand that is being replaced\n" + \
         "if these indices are not provided, they will the guessed",
)

maplig_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\nDefault: stdout",
)

args = maplig_parser.parse_args()

if args.list_avail:
    s = ""
    for i, name in enumerate(sorted(Component.list())):
        s += "%-35s" % name
        if (i + 1) % 3 == 0:
       # if (i + 1) % 1 == 0:
            s += "\n"

    print(s.strip())
    exit(0)

for infile in args.infile:
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

    cat = Geometry(f)
    for lig_info in args.ligand:
        # TODO: change this if to a regex
        if '=' in lig_info:
            key = lig_info.split("=")[0]
            lig = "=".join(lig_info.split("=")[1:])
            cat.map_ligand(lig, key)
        else:
            ligand = Component(lig_info)
            old_key_atoms = []
            j = 0
            while len(old_key_atoms) < len(ligand.key_atoms):
                if j >= len(cat.components["ligand"]):
                    raise RuntimeError(
                        "new ligand appears to have a higher denticity than old ligands combined"
                    )
                else:
                    old_key_atoms.extend(cat.components["ligand"][j].key_atoms)
                    j += 1

            if len(old_key_atoms) % len(ligand.key_atoms) == 0:
                k = 0
                ligands = []
                while k != len(old_key_atoms):
                    k += len(ligand.key_atoms)
                    ligands.append(ligand.copy())

                cat.map_ligand(ligands, old_key_atoms)

    s = cat.write(append=False, outfile=args.outfile)
    if not args.outfile:
        print(s)
