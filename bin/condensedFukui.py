#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

fukui_parser = argparse.ArgumentParser(
    description="print a cube file for a molecular orbital",
    formatter_class=argparse.RawTextHelpFormatter
)

fukui_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="an FCHK file or ORCA output with MO's"
)

fukui_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

info = fukui_parser.add_mutually_exclusive_group(required=True)
info.add_argument(
    "-fd", "--fukui-donor",
    dest="fukui_donor",
    default=False,
    action="store_true",
    help="print Fukui donor values\n"
    "see DOI 10.1002/jcc.24699 for weighting method\n"
    "for details on how the function is condensed, see the GitHub wiki:\n"
    "https://github.com/QChASM/AaronTools.py/wiki/Structure-Analysis-and-Descriptor-Implementation#condensed-fukui"
)

info.add_argument(
    "-fa", "--fukui-acceptor",
    dest="fukui_acceptor",
    default=False,
    action="store_true",
    help="print Fukui acceptor values\n"
    "see DOI 10.1021/acs.jpca.9b07516 for weighting method\n"
    "for details on how the function is condensed, see the GitHub wiki:\n"
    "https://github.com/QChASM/AaronTools.py/wiki/Structure-Analysis-and-Descriptor-Implementation#condensed-fukui"
)

info.add_argument(
    "-f2", "--fukui-dual",
    dest="fukui_dual",
    default=False,
    action="store_true",
    help="print Fukui dual values\n"
    "see DOI 10.1021/acs.jpca.9b07516 for weighting method\n"
    "for details on how the function is condensed, see the GitHub wiki:\n"
    "https://github.com/QChASM/AaronTools.py/wiki/Structure-Analysis-and-Descriptor-Implementation#condensed-fukui"
)

fukui_parser.add_argument(
    "-d", "--delta",
    type=float,
    dest="delta",
    default=0.1,
    help="delta parameter for weighting orbitals in Fukui functions\n"
    "Default: 0.1 Hartree",
)

fukui_parser.add_argument(
    "-nt", "--number-of-threads",
    type=int,
    default=1,
    dest="n_jobs",
    help="number of threads to use when evaluating basis functions\n"
    "this is on top of NumPy's multithreading,\n"
    "so if NumPy uses 8 threads and n_jobs=2, you can\n"
    "expect to see 16 threads in use\n"
    "Default: 1"
)

fukui_parser.add_argument(
    "-m", "--max-array",
    type=int,
    default=10000000,
    dest="max_length",
    help="max. array size to read from FCHK files\n"
    "a reasonable size for setting parsing orbital data\n"
    "can improve performance when reading large FCHK files\n"
    "too small of a value will prevent orbital data from\n"
    "being parsed\n"
    "Default: 10000000",
)

fukui_parser.add_argument(
    "-v", "--vdw-radii",
    default="umn",
    choices=["umn", "bondi"],
    dest="radii",
    help="VDW radii to use in calculation\n" + 
    "umn: main group vdw radii from J. Phys. Chem. A 2009, 113, 19, 5806–5812\n" +
    "    (DOI: 10.1021/jp8111556)\n" + 
    "    transition metals are crystal radii from Batsanov, S.S. Van der Waals\n" +
    "    Radii of Elements. Inorganic Materials 37, 871–885 (2001).\n" +
    "    (DOI: 10.1023/A:1011625728803)\n" + 
    "bondi: radii from J. Phys. Chem. 1964, 68, 3, 441–451 (DOI: 10.1021/j100785a001)\n" +
    "Default: umn",
)

grid_options = fukui_parser.add_argument_group("Lebedev integration options")
grid_options.add_argument(
    "-rp", "--radial-points",
    type=int,
    default=32,
    choices=[20, 32, 64, 75, 99, 127],
    dest="rpoints",
    help="number of radial shells for Gauss-Legendre integration\n" +
    "of the radial component\n" +
    "lower values are faster, but at the cost of accuracy\nDefault: 32"
)

grid_options.add_argument(
    "-ap", "--angular-points",
    type=int,
    default=1454,
    choices=[110, 194, 302, 590, 974, 1454, 2030, 2702, 5810],
    dest="apoints",
    help="number of angular points for Lebedev integration\n" +
    "lower values are faster, but at the cost of accuracy\nDefault: 1454"
)

args = fukui_parser.parse_args()


infiles = glob_files(args.infile, parser=fukui_parser)
for n, f in enumerate(infiles):
    if isinstance(f, str):
        infile = FileReader(
            f, just_geom=False, max_length=args.max_length
        )
    elif len(sys.argv) >= 1:
        infile = FileReader(
            ("from stdin", "fchk", f),
            just_geom=False,
            max_length=args.max_length,
        )

    try:
        orbits = infile.other["orbitals"]
    except KeyError:
        raise RuntimeError("orbital info was not parsed from %s" % f)
    geom = Geometry(infile, refresh_connected=False, refresh_ranks=False)
    
    if args.fukui_donor:
        func = orbits.condensed_fukui_donor_values
    elif args.fukui_acceptor:
        func = orbits.condensed_fukui_acceptor_values
    elif args.fukui_dual:
        func = orbits.condensed_fukui_dual_values

    vals = func(
        geom,
        n_jobs=args.n_jobs,
        rpoints=args.rpoints,
        apoints=args.apoints,
        radii=args.radii,
    )
    
    s = ""
    if len(infiles) > 1:
        s += "%s:\n" % f
    for i, atom in enumerate(geom.atoms):
        s += "%s%3s\t%6.3f\n" % ("\t" if len(infiles) > 1 else "", atom.name, vals[i])

    if not args.outfile:
        print(s)
    else:
        outfile = args.outfile
        mode = "w"
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        elif n > 0:
            mode = "a"
        with open(outfile, mode) as f:
            f.write(s)

    
