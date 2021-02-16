#!/usr/bin/env python3

import os
import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename

unique_parser = argparse.ArgumentParser(
    description="determine which structures are unique",
    formatter_class=argparse.RawTextHelpFormatter
)

unique_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

unique_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

unique_parser.add_argument(
    "-t", "--rmsd-tolerance",
    type=float,
    default=0.15,
    dest="tol",
    help="RMSD tolerance for structures with the same chemical formula\n" +
    "to be considered unique"
)

unique_parser.add_argument(
    "-d", "--directory",
    type=str,
    dest="directory",
    default=False,
    help="put structures in specified directory\nDefault: don't output structures",
)

args = unique_parser.parse_args()

# dictionary of structures, which will be ordered by number of atoms, elements, etc.
structures = {}

for f in args.infile:
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

    n_atoms = len(geom.atoms)

    if n_atoms not in structures:
        structures[n_atoms] = {}

    element_list = [atom.element for atom in geom.atoms]
    elements = sorted(list(set(element_list)))
    s = ""
    for ele in elements:
        s += "%s%i" % (ele, element_list.count(ele))

    if not any(s == key for key in structures[n_atoms].keys()):
        structures[n_atoms][s] = []
    
    dup = False
    for group in structures[n_atoms][s]:
        for struc, _ in group:
            rmsd = geom.RMSD(struc, sort=True)
            if rmsd < args.tol:
                dup = True
                group.append((geom, rmsd))
                break
        if dup:
            break

    if not dup:
        structures[n_atoms][s].append([(geom, 0)])

s = ""
unique = 0
total = 0
for n_atoms in structures:
    for formula in structures[n_atoms]:
        formula_unique = len(structures[n_atoms][formula])
        unique += formula_unique
        s += "%s\n" % formula

        for group in structures[n_atoms][formula]:
            total += len(group)
            if args.directory:
                dir_name = os.path.join(
                    args.directory,
                    formula,
                    get_filename(group[0][0].name, include_parent_dir=False),
                )
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                for geom, rmsd in group:
                    geom.comment="RMSD from %s = %.4f" % (
                        get_filename(group[0][0].name, include_parent_dir=False),
                        rmsd,
                    )
                    geom.write(
                        outfile=os.path.join(
                            args.directory,
                            formula,
                            get_filename(group[0][0].name, include_parent_dir=False),
                            get_filename(geom.name, include_parent_dir=False) + ".xyz"
                        ),
                    )
            if len(group) > 1:
                if len(group) == 2:
                    s += "there is %i structure identical to %s:\n" % (
                        len(group) - 1,
                        group[0][0].name,
                    )
                else:
                    s += "there are %i structures identical to %s:\n" % (
                        len(group) - 1,
                        group[0][0].name,
                    )
                for geom, rmsd in group[1:]:
                    s += "\t%s (RMSD = %.3f)\n" % (geom.name, rmsd)
            else:
                s += "there are no other structures identical to %s\n" % group[0][0].name
           
            s += "\n"

        s += "-----\n\n"

s += "there were %i input structures\n" % total
s += "in total, there are %i unique structures\n" % unique
print(s)

