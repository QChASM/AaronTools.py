#!/usr/bin/env python3

import argparse
import os

from AaronTools.geometry import Geometry
from AaronTools.component import Component

vsepr_choices = [
        "tetrahedral",
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
]

coord_comp_parser = argparse.ArgumentParser(
    description="build coordination complexes using templates from Inorg. Chem. 2018, 57, 17, 10557–10567",
    formatter_class=argparse.RawTextHelpFormatter,
)

coord_comp_parser.add_argument(
    "-l",
    "--ligands",
    type=str,
    nargs="*",
    required=True,
    # choices=Component.list(),
    dest="ligands",
    help="list of ligands to attach to the coordination complex\n" +
    "see `mapLigand.py --list` for a list of available ligands",
)

coord_comp_parser.add_argument(
    "-c2",
    "--c2-symmetric",
    type=lambda x: x.lower() in ["yes", "true", "t", "arr", "y", "aye", "yeah"],
    nargs="*",
    default=None,
    required=False,
    dest="c2_symmetric",
    help="list of true/false corresping to --ligands to denote which bidentate\n" +
    "ligands are C2-symmetric\nDefault: try to determine if bidentate ligands are C2-symmetric",
)

coord_comp_parser.add_argument(
    "-g",
    "--coordination-geometry",
    choices=vsepr_choices,
    required=True,
    dest="shape",
    help="coordination geometry of central atom"
)
coord_comp_parser.add_argument(
    "-c",
    "--center-atom",
    required=True,
    metavar="element",
    dest="center",
    help="central atom for coordination complexes"
)

coord_comp_parser.add_argument(
    "-m",
    "--minimize",
    action="store_true",
    default=False,
    required=False,
    dest="minimize",
    help="try to relax ligands to minimize steric clashing\nDefault: False",
)

coord_comp_parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    metavar="output destination",
    dest="outdir",
    help="output directory\n" +\
    "Filenames will match the detected generic formula and\n" +
    "include the point group and subset from the reference\n"
    "noted above\n"
    "Subsets with primes (e.g. A' and A'') are not distinguished",
)

args = coord_comp_parser.parse_args()

geoms, formula = Geometry.get_coordination_complexes(
    args.center,
    args.ligands,
    args.shape.replace("_", " "),
    c2_symmetric=args.c2_symmetric,
    minimize=args.minimize,
)

print("formula is %s" % formula)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

for geom in geoms:
    geom.write(outfile=os.path.join(args.outdir, geom.name + ".xyz"))

print("wrote", len(geoms), "structures to", args.outdir)