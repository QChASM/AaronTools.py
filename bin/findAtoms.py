#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.finders import *
from AaronTools.utils.utils import glob_files

vsepr_choices = [
    "linear_1",
    "linear_2",
    "bent_2_planar",
    "bent_2_tetrahedral",
    "trigonal_planar",
    "bent_3_tetrahedral",
    "t_shaped",
    "tetrahedral",
    "sawhorse",
    "square_planar",
    "trigonal_bipyriamidal",
    "square_pyramidal",
    "octahedral",
]

find_parser = argparse.ArgumentParser(
    description="find atoms matching a description and return the list of indices",
    formatter_class=argparse.RawTextHelpFormatter
)

find_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

find_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin",
)


find_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "Default: stdout"
)

find_parser.add_argument(
    "-d",
    "--delimiter",
    required=False,
    dest="delim",
    default="comma",
    choices=["comma", "semicolon", "tab", "space"],
    help="delimiter for output atom indices"
)

find_parser.add_argument(
    "-e", "--element",
    type=str,
    action="append",
    default=[],
    required=False,
    dest="elements",
    help="element symbol",
)

find_parser.add_argument(
    "-n", "--index",
    type=str,
    action="append",
    default=[],
    required=False,
    dest="ndx",
    help="1-index position of atoms in the input file\n"
    "may hyphen separated to denote a range\n"
    "ranges and individual indices may be comma-separated",
)

find_parser.add_argument(
    "-bf", "--bonds-from",
    type=str,
    nargs=2,
    action="append",
    default=[],
    required=False,
    metavar=("BONDS", "NUM"),
    dest="bonds_from",
    help="find atoms BONDS (integer) bonds away from atom NDX",
)

find_parser.add_argument(
    "-wb", "--within-bonds",
    type=str,
    nargs=2,
    action="append",
    default=[],
    required=False,
    metavar=("BONDS", "NDX"),
    dest="within_bonds",
    help="find atoms within BONDS (integer) bonds from atom NDX",
)

find_parser.add_argument(
    "-bt", "--bonded-to",
    type=str,
    action="append",
    default=[],
    required=False,
    metavar="NDX",
    dest="bonded_to",
    help="find atoms bonded to atom NDX",
)

find_parser.add_argument(
    "-pd", "--point-distance",
    type=float,
    nargs=4,
    action="append",
    default=[],
    required=False,
    metavar=("X", "Y", "Z", "R"),
    dest="point_dist",
    help="find atoms within R Angstroms of (X, Y, Z)",
)

find_parser.add_argument(
    "-ad", "--atom-distance",
    type=str,
    nargs=2,
    action="append",
    default=[],
    required=False,
    metavar=("NDX", "R"),
    dest="atom_dist",
    help="find atoms within R Angstroms of atom NDX",
)

find_parser.add_argument(
    "-tm", "--transition-metal",
    action="store_true",
    default=False,
    required=False,
    dest="tmetal",
    help="find any elements in the d-block, up to the Actinides",
)

find_parser.add_argument(
    "-mg", "--main-group",
    action="store_true",
    default=False,
    required=False,
    dest="main_group",
    help="find any main group element (including H)",
)

find_parser.add_argument(
    "-v", "--vsepr",
    type=str,
    action="append",
    default=[],
    required=False,
    choices=vsepr_choices,
    metavar="SHAPE",
    dest="vsepr",
    help="find atoms with the specified VSEPR shape\n" +
    "shape can be:\n%s" % "".join(
        s + ", " if (sum(len(x) for x in vsepr_choices[:i])) % 40 < 21 else s + ",\n"
        for i, s in enumerate(vsepr_choices)
    ).strip().strip(","),
)

find_parser.add_argument(
    "-nb", "--number-of-bonds",
    type=int,
    action="append",
    default=[],
    required=False,
    dest="num_neighbors",
    help="find atoms with the specified number of bonds",
)

find_parser.add_argument(
    "-ct", "--closer-to",
    nargs=2,
    metavar=("THIS_NDX", "THAN_NDX"),
    action="append",
    default=[],
    dest="closer_to",
    help="atoms that are fewer bonds from THIS_NDX than THAN_NDX",
)

find_parser.add_argument(
    "-c", "--chiral-center",
    action="store_true",
    default=False,
    dest="chiral",
    help="find chiral centers",
)

find_parser.add_argument(
    "-f", "--fragment",
    type=str,
    default=None,
    dest="fragments",
    help="fragments containing the specified atoms"
)

finder_combination = find_parser.add_argument_group(
    "match method (Default is atoms matching all)"
)
finder_combination.add_argument(
    "-or", "--match-any",
    action="store_true",
    default=False,
    dest="match_any",
    help="find atoms matching any of the given descriptions",
)

finder_combination.add_argument(
    "-i", "--invert",
    action="store_true",
    default=False,
    dest="invert",
    help="invert match results",
)


args = find_parser.parse_args()

# create a list of geometry-independent finders
finders = []
for ndx in args.ndx:
    finders.append(ndx)

for ele in args.elements:
    finders.append(ele)

for pd in args.point_dist:
    r = pd.pop(-1)
    p = np.array(pd)
    finders.append(WithinRadiusFromPoint(pd, r))

if args.tmetal:
    finders.append(AnyTransitionMetal())

if args.main_group:
    finders.append(AnyNonTransitionMetal())

for vsepr in args.vsepr:
    finders.append(VSEPR(vsepr.replace("_", " ")))

for nb in args.num_neighbors:
    finders.append(NumberOfBonds(nb))

if args.chiral:
    finders.append(ChiralCenters())

if args.delim == "comma":
    delim = ","
elif args.delim == "space":
    delim = " "
elif args.delim == "tab":
    delim = "\t"
elif args.delim == "semicolon":
    delim = ";"

s = ""

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

    geom_finders = [x for x in finders]
    # some finders require and atom upon instantiation
    # add those once we have the geometry
    for ad in args.atom_dist:
        r = float(ad.pop(-1))
        atom_ndx = ad.pop(0)
        for atom in geom.find(atom_ndx):
            geom_finders.append(WithinRadiusFromAtom(atom, r))

    for wb in args.within_bonds:
        n_bonds = int(wb.pop(0))
        atom_ndx = wb.pop(0)
        for atom in geom.find(atom_ndx):
            geom_finders.append(WithinBondsOf(atom, n_bonds))

    for bt in args.bonded_to:
        for atom in geom.find(bt):
            geom_finders.append(BondedTo(atom))
    
    for ct in args.closer_to:
        atom1, atom2 = geom.find(ct)
        geom_finders.append(CloserTo(atom1, atom2))
    
    if args.fragments:
        for atom in geom.find(args.fragments):
            frag_atoms = geom.get_all_connected(atom)
            geom_finders.append(frag_atoms)

    if len(args.infile) > 1:
        s += "%s\n" % str(s)

    try:
        if args.match_any:
            results = geom.find(geom_finders)
        else:
            results = geom.find(*geom_finders)

        if args.invert:
            results = geom.find(NotAny(results))

        s += delim.join(atom.name for atom in results)
        s += "\n"
    except LookupError as e:
        if args.invert:
            s += delim.join(atom.name for atom in geom.atoms)
            s += "\n"
        else:
            s += "%s\n" % str(e)


if not args.outfile:
    print(s.strip())
else:
    with open(args.outfile, "a") as f:
        f.write(s.strip())
