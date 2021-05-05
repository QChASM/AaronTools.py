#!/usr/bin/env python3

import os
import sys
import argparse

import numpy as np

from AaronTools.const import UNIT
from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

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
    "to be considered unique\nDefault:0.15"
)

unique_parser.add_argument(
    "-e", "--energy-filter",
    metavar="KCAL/MOL",
    nargs="?",
    default=False,
    type=float,
    dest="energy_filter",
    help="only compare structures with similar energy\n" +
    "structures without an energy are always compared\n" +
    "Default: compare regardless of energy",
)

unique_parser.add_argument(
    "-m", "--mirror",
    action="store_true",
    default=False,
    dest="mirror",
    help="also mirror structures when comparing",
)

unique_parser.add_argument(
    "-d", "--directory",
    type=str,
    dest="directory",
    default=False,
    help="put structures in specified directory\nDefault: don't output structures",
)

args = unique_parser.parse_args()

if args.energy_filter is None:
    args.energy_filter = 0.2

mirror_mat = np.eye(3)
mirror_mat[0][0] *= -1

# dictionary of structures, which will be ordered by number of atoms, elements, etc.
structures = {}

for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format, f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=False)

    geom = Geometry(infile)
    geom.other = infile.other

    if args.mirror:
        geom_mirrored = geom.copy()
        geom_mirrored.update_geometry(np.dot(geom.coords, mirror_mat))

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
        for struc, _, _ in group:
            if args.energy_filter and "energy" in geom.other and "energy" in struc.other:
                d_nrg = UNIT.HART_TO_KCAL * abs(
                    geom.other["energy"] - struc.other["energy"]
                ) 
                if d_nrg > args.energy_filter:
                    continue
            rmsd = geom.RMSD(struc, sort=True)
            if rmsd < args.tol:
                dup = True
                group.append((geom, rmsd, False))
                break

            if args.mirror:
                rmsd2 = geom_mirrored.RMSD(struc, sort=True)
                if rmsd2 < args.tol:
                    dup = True
                    group.append(geom, rmsd2, True)
        if dup:
            break

    if not dup:
        structures[n_atoms][s].append([(geom, 0, False)])

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
                for geom, rmsd, _ in group:
                    geom.comment="RMSD from %s = %.4f" % (
                        get_filename(group[0][0].name, include_parent_dir=False),
                        rmsd,
                    )
                    if args.energy_filter and all("energy" in g.other for g in [geom, group[0][0]]):
                        d_nrg = UNIT.HART_TO_KCAL * (geom.other["energy"] - group[0][0].other["energy"])
                        geom.comment += "  energy from %s = %.1f kcal/mol" % (
                            get_filename(group[0][0].name, include_parent_dir=False),
                            d_nrg,
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
                    s += "there is %i structure similar to %s:\n" % (
                        len(group) - 1,
                        group[0][0].name,
                    )
                else:
                    s += "there are %i structures similar to %s:\n" % (
                        len(group) - 1,
                        group[0][0].name,
                    )
                for geom, rmsd, mirrored in group[1:]:
                    if not mirrored:
                        s += "\t%s (RMSD = %.3f)" % (geom.name, rmsd)
                    else:
                        s += "\t%s (mirrored) (RMSD = %.3f)" % (geom.name, rmsd)
                    
                    if args.energy_filter and all("energy" in g.other for g in [geom, group[0][0]]):
                        d_nrg = UNIT.HART_TO_KCAL * (geom.other["energy"] - group[0][0].other["energy"])
                        s += " (dE = %.1f kcal/mol)" % (
                            d_nrg,
                        )

            else:
                s += "there are no other structures identical to %s\n" % group[0][0].name
           
            s += "\n"

        s += "-----\n\n"

s += "there were %i input structures\n" % total
s += "in total, there are %i unique structures\n" % unique
print(s)

