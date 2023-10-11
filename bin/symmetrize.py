#!/usr/bin/env python3

import sys
from os.path import dirname
import argparse

import numpy as np

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.symmetry import (
    PointGroup,
    InversionCenter, 
    ProperRotation,
    ImproperRotation,
    MirrorPlane,
)
from AaronTools.utils.utils import (
    perp_vector,
    glob_files,
    get_filename,
    get_outfile,
)

pg_parser = argparse.ArgumentParser(
    description="print point group",
    formatter_class=argparse.RawTextHelpFormatter
)

pg_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

pg_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination \nDefault: stdout"
)

pg_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

pg_parser.add_argument(
    "-t", "--tolerance",
    default=0.1,
    type=float,
    dest="tolerance",
    help="tolerance for determining if a symmetry element is valid\n"
    "for the input structure(s)\nDefault: 0.1"
)

pg_parser.add_argument(
    "-a", "--axis-tolerance",
    default=0.01,
    type=float,
    dest="rotation_tolerance",
    help="tolerance for determining if two axes are coincident or orthogonal"
    "\nDefault: 0.01"
)

pg_parser.add_argument(
    "-n", "--max-n",
    default=6,
    type=int,
    dest="max_n",
    help="max. order for proper rotation axes (improper rotations can be 2x this)"
    "\nDefault: 6"
)

pg_parser.add_argument(
    "-e", "--report-error",
    action="store_true",
    default=False,
    dest="report_error",
    help="print all symmetry elements",
)


args = pg_parser.parse_args()

s = ""
for f in glob_files(args.infile, parser=pg_parser):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format, None), just_geom=True)
        else:
            infile = FileReader(f, just_geom=True)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f), just_geom=True)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=True)

    geom = Geometry(infile)
    
    out = geom.write(outfile=False)

    pg = PointGroup(
        geom,
        tolerance=args.tolerance,
        rotation_tolerance=args.rotation_tolerance,
        max_rotation=args.max_n,
    )
    if args.report_error:
        print("%s is %s" % (geom.name, pg.name))
        tot_error, max_error, max_ele = pg.total_error(return_max=True)
        print("total error before symmetrizing: %.4f" % tot_error)
        print("max. error before symmetrizing: %.4f (%s)" % (max_error, max_ele))
    pg.idealize_geometry()
    if args.report_error:
        tot_error, max_error, max_ele = pg.total_error(return_max=True)
        print("total error after symmetrizing: %.4f" % tot_error)
        print("max. error after symmetrizing: %.4f (%s)" % (max_error, max_ele))

    out = geom.write(outfile=False)
    
    if not args.outfile:
        print(out)
    else:
        outfile = get_outfile(
            args.outfile,
            INFILE=get_filename(f, include_parent_dir="$INDIR" not in args.outfile),
            INDIR=dirname(f),
        )
        with open(outfile, "w") as f:
            f.write(out)


