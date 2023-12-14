#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from AaronTools.geometry import Geometry
from AaronTools.fileIO import read_types, FileReader
from AaronTools.utils.utils import (
    get_filename,
    glob_files,
    get_outfile,
)

remove_frag_parser = argparse.ArgumentParser(
    description="remove a fragment from a molecule",
    formatter_class=argparse.RawTextHelpFormatter,
)

remove_frag_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

remove_frag_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin",
)

remove_frag_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "$INDIR will be replaced with the directory of the input file\n" +
    "Default: stdout",
)

remove_frag_parser.add_argument(
    "-s",
    "--solvent",
    type=str,
    required=True,
    dest="solvent_mol",
    help="solvent molecule to remove",
)


args = remove_frag_parser.parse_args()

sol = Geometry(args.solvent_mol)
sol_elements = sol.element_counts()

for infile in glob_files(args.infile, parser=remove_frag_parser):
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

    geom = Geometry(f, refresh_ranks=False)

    monomers = [Geometry(m, refresh_connected=False, refresh_ranks=False) for m in geom.get_monomers()]
    keep_monomers = []
    sol_elements = sol.element_counts()
    sol_ranks = sol.canonical_rank(break_ties=False, update=False, invariant=False)
    sorted_sol_atoms = [x for _, x in sorted(zip(sol_ranks, sol.atoms), key=lambda pair: pair[0])]
    keep_mol = Geometry()
    for monomer in monomers:
        if len(monomer) != len(sol.atoms):
            keep_mol.atoms.extend(monomer.atoms)
            continue

        if monomer.element_counts() != sol_elements:
            keep_mol.atoms.extend(monomer.atoms)
            continue

        frag_ranks = monomer.canonical_rank(break_ties=False, update=False, invariant=False)
        sorted_frag_atoms = [x for _, x in sorted(zip(frag_ranks, monomer.atoms), key=lambda pair: pair[0])]
        for a, b in zip(sorted_frag_atoms, sorted_sol_atoms):
            if a.element != b.element:
                keep_mol.atoms.extend(monomer.atoms)
                break

            if len(a.connected) != len(b.connected):
                keep_mol.atoms.extend(monomer.atoms)
                break

            for j, k in zip(
                sorted([aa.element for aa in a.connected]),
                sorted([bb.element for bb in b.connected]),
            ):
                if j != k:
                    keep_mol.atoms.extend(monomer.atoms)
                    break

    if args.outfile:
        outfile = get_outfile(
            args.outfile,
            INFILE=get_filename(infile, include_parent_dir="$INDIR" not in args.outfile),
            INDIR=os.path.dirname(infile),
        )
        keep_mol.write(append=False, outfile=outfile)
    else:
        s = keep_mol.write(outfile=False)
        print(s)
