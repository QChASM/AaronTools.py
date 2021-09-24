#!/usr/bin/env python3

import sys
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

cube_parser = argparse.ArgumentParser(
    description="print a cube file for a molecular orbital",
    formatter_class=argparse.RawTextHelpFormatter
)

cube_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="an FCHK file, ORCA output with MO's, or NBO files"
)

cube_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

info = cube_parser.add_mutually_exclusive_group(required=False)
info.add_argument(
    "-mo", "--molecular-orbital",
    dest="mo_ndx",
    default="homo",
    help="index of molecular orbital to print (0-indexed)\n"
    "can also give 'homo' or 'lumo' for highest occupied or\n"
    "lowest unoccupied molecular orbital\n"
    "Default: highest occupied MO in the ground state"
)
info.add_argument(
    "-ao", "--atomic-orbital",
    dest="ao_ndx",
    default=None,
    help="index of atomic orbital to print (0-indexed)"
)

info.add_argument(
    "-ed", "--electron-density",
    dest="density",
    default=False,
    action="store_true",
    help="print electron density"
)

info.add_argument(
    "-fd", "--fukui-donor",
    dest="fukui_donor",
    default=False,
    action="store_true",
    help="print Fukui donor values\n"
    "see DOI 10.1002/jcc.24699 for weighting method"
)

info.add_argument(
    "-fa", "--fukui-acceptor",
    dest="fukui_acceptor",
    default=False,
    action="store_true",
    help="print Fukui acceptor values\n"
    "see DOI 10.1021/acs.jpca.9b07516 for weighting method"
)

info.add_argument(
    "-f2", "--fukui-dual",
    dest="fukui_dual",
    default=False,
    action="store_true",
    help="print Fukui dual values\n"
    "see DOI 10.1021/acs.jpca.9b07516 for weighting method"
)

cube_parser.add_argument(
    "-d", "--delta",
    type=float,
    dest="delta",
    default=0.1,
    help="delta parameter for weighting orbitals in Fukui functions\n"
    "Default: 0.1 Hartree",
)

cube_parser.add_argument(
    "-s", "--spacing",
    type=float,
    dest="spacing",
    default=0.2,
    help="spacing between points in the cube file\n"
    "Default: 0.2",
)

cube_parser.add_argument(
    "-p", "--padding",
    type=float,
    dest="padding",
    default=4,
    help="extra space around the molecule\n"
    "Default: 4"
)

cube_parser.add_argument(
    "-xyz", "--standard-axes",
    action="store_true",
    dest="xyz",
    default=False,
    help="use x, y, and z axes to define the directions\n"
    "Default: determine directions using SVD"
)

cube_parser.add_argument(
    "-nt", "--number-of-threads",
    type=int,
    default=1,
    dest="n_jobs",
    help="number of threads to use when evaluating basis functions"
    "this is on top of NumPy's multithreading,\n"
    "so if NumPy uses 8 threads and n_jobs=2, you can\n"
    "expect to see 16 threads in use\n"
    "Default: 1"
)

cube_parser.add_argument(
    "-nbo", "--nbo-file",
    type=str,
    default=None,
    dest="nbo_name",
    help="file containing coefficients for NBO's (e.g. *.37 file)"
    "ignored unless input file is a *.47 file"
)

cube_parser.add_argument(
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


args = cube_parser.parse_args()

kind = args.mo_ndx

if args.density:
    kind = "density"
elif args.fukui_donor:
    kind = "fukui donor"
elif args.fukui_acceptor:
    kind = "fukui acceptor"
elif args.fukui_dual:
    kind = "fukui dual"
elif args.ao_ndx:
    kind = "AO %s" % args.ao_ndx
elif args.mo_ndx.isdigit():
    kind = "MO %s" % args.mo_ndx


for f in glob_files(args.infile, parser=cube_parser):
    if isinstance(f, str):
        infile = FileReader(
            f, just_geom=False, nbo_name=args.nbo_name, max_length=args.max_length
        )
    elif len(sys.argv) >= 1:
        infile = FileReader(
            ("from stdin", "fchk", f),
            just_geom=False,
            nbo_name=args.nbo_name,
            max_length=args.max_length,
        )

    geom = Geometry(infile, refresh_connected=False, refresh_ranks=False)

    out = geom.write(
        outfile=False,
        orbitals=infile.other["orbitals"],
        padding=args.padding,
        kind=kind,
        spacing=args.spacing,
        style="cube",
        xyz=args.xyz,
        delta=args.delta,
        n_jobs=args.n_jobs,
    )
    
    if not args.outfile:
        print(out)
    else:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        with open(outfile, "w") as f:
            f.write(out)

