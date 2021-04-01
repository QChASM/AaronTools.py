#!/usr/bin/env python3

import argparse
import os

from AaronTools.geometry import Geometry
from AaronTools.component import Component


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
    help="list corresping to --ligands to denote which bidentate\n" +
    "ligands are C2-symmetric\nDefault: no ligands are treated as symmetric",
)

coord_comp_parser.add_argument(
    "-g",
    "--coordination-geometry",
    required=True,
    dest="shape",
    help="coordination geometry of central atom"
)
coord_comp_parser.add_argument(
    "-c",
    "--center-atom",
    required=True,
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
    help="output directory",
)

args = coord_comp_parser.parse_args()

geoms, formula = Geometry.get_coordination_complexes(
    args.center,
    args.ligands,
    args.shape,
    c2_symmetric=args.c2_symmetric,
    minimize=args.minimize,
)

print("formula is %s" % formula)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

for geom in geoms:
    geom.write(outfile=os.path.join(args.outdir, geom.name + ".xyz"))

print("wrote", len(geoms), "structures to", args.outdir)