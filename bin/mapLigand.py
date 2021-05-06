#!/usr/bin/env python3

import argparse
import sys

from AaronTools.component import Component
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import get_filename, glob_files

def get_matching_ligands(name):
    name_info = name
    coordinating_elements = None
    denticity = None
    if isinstance(name_info, str):
        if (
                "name:" not in name_info.lower() and
                "elements:" not in name_info.lower() and
                "denticity:" not in name_info.lower()
        ):
            name_info = "^%s$" % name_info
        else:
            lig_info = name.split(":")
            for i, info in enumerate(lig_info):
                if info.lower() == "name" and i+1 < len(lig_info):
                    name_info = name_info.replace("%s:%s" % (info, lig_info[i+1]), "")
                    name_info = lig_info[i+1]

                elif info.lower() == "elements" and i+1 < len(lig_info):
                    name_info = name_info.replace("%s:%s" % (info, lig_info[i+1]), "")
                    coordinating_elements = lig_info[i+1].split(",")

                elif info.lower() == "denticity" and i+1 < len(lig_info):
                    name_info = name_info.replace("%s:%s" % (info, lig_info[i+1]), "")
                    denticity = int(lig_info[i+1])

            if not name_info:
                name_info = None

    return Component.list(
        name_regex=name_info,
        coordinating_elements=coordinating_elements,
        denticity=denticity
    )


maplig_parser = argparse.ArgumentParser(
    description="replace a ligand on an organometallic system",
    formatter_class=argparse.RawTextHelpFormatter,
)

maplig_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

maplig_parser.add_argument(
    "-ls",
    "--list",
    nargs="?",
    const=None,
    default=False,
    required=False,
    dest="list_avail",
    metavar="elements:X[,Y...] | name:RegEx",
    help="list available ligands\n" +
    "elements:X[,Y] can be used to only list ligands that coordinate\n" +
    "with the specified elements - must match exactly\n" +
    "name:RegEx can be used to only list ligands with names matching\n" +
    "the supplied regular expression - matches are case-insensitive",
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
    metavar="[n[,m...]]=ligand | ligand",
    type=str,
    default=None,
    required=False,
    dest="ligand",
    help="ligand used to replace the current one\n" +
    "n[,m...] are the 1-indexed positions of the coordinating atoms of the\n" +
    "ligand that is being replaced\n" +
    "if these indices are not provided, they will the guessed\n" +
    "elements:X[,Y] or name:RegEx can be used in place of ligand\n" +
    "to swap ligands matching these criteria (see --list option)",
)

maplig_parser.add_argument(
    "-c",
    "--center",
    required=False,
    default=None,
    dest="center",
    help="catalyst center the ligand is bonded to\nDefault: any transition metal"
)

maplig_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$LIGAND will be replaced with ligand name\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout",
)

args = maplig_parser.parse_args()

if args.list_avail is not False:
    s = ""
    for i, name in enumerate(sorted(get_matching_ligands(args.list_avail))):
        s += "%-35s" % name
        if (i + 1) % 3 == 0:
        # if (i + 1) % 1 == 0:
            s += "\n"

    print(s.strip())
    sys.exit(0)

for infile in glob_files(args.infile):
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format[0], infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(("from stdin", args.input_format[0], infile))
        else:
            f = FileReader(("from stdin", "xyz", infile))

    cat = Geometry(f)
    
    if args.center:
        cat.detect_components(center=args.center)
    
    # TODO: change this if to a regex
    if '=' in args.ligand:
        key_atoms = args.ligand.split("=")[0]
        lig_names = [l for l in "=".join(args.ligand.split("=")[1:]).split(",")]

    else:
        lig_names = [l for l in "=".join(args.ligand.split("=")[1:]).split(",")]
        key_atoms = []

    for lig_names in [get_matching_ligands(lig_name) for lig_name in lig_names]:
        for lig_name in lig_names:
            ligands = [Component(lig_name)]
            cat_copy = cat.copy()

            if key_atoms != []:
                key = cat_copy.find(key_atoms)
            else:
                key = []

            original_ligands = [l for l in ligands]

            lig_keys = sum([len(ligand.key_atoms) for ligand in ligands])
            while len(key) > lig_keys:
                ligands.extend(l.copy() for l in original_ligands)
                lig_keys += sum([len(ligand.key_atoms) for ligand in original_ligands])

            j = 0
            while len(key) < sum([len(ligand.key_atoms) for ligand in ligands]):
                if j >= len(cat.components["ligand"]):
                    raise RuntimeError(
                        "new ligand appears to have a higher denticity than old ligands combined"
                    )

                key.extend(cat.components["ligand"][j].key_atoms)
                j += 1

            cat_copy.map_ligand(ligands, key)

            if args.outfile:
                if "$INFILE" in args.outfile:
                    outfile = args.outfile.replace("$INFILE", get_filename(infile))
                outfile = outfile.replace("$LIGAND", lig_name)
                cat_copy.write(append=False, outfile=outfile)
            else:
                s = cat_copy.write(outfile=False)
                print(s)
