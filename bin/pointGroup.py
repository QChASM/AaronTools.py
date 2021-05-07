import sys
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
from AaronTools.utils.utils import perp_vector, glob_files

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
    "-v", "--verbose",
    action="store_true",
    default=False,
    dest="print_eles",
    help="print all symmetry elements",
)

pg_parser.add_argument(
    "-b", "--bild",
    action="store_true",
    default=False,
    dest="print_bild",
    help="print Chimera(X) bild file to display various symmetry elements",
)

args = pg_parser.parse_args()

s = ""
for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=False)

    geom = Geometry(infile)
    pg = PointGroup(
        geom,
        tolerance=args.tolerance,
        rotation_tolerance=args.rotation_tolerance,
        max_rotation=args.max_n,
    )
    if args.print_bild:
        inv = ""
        mirror = ""
        prots = ""
        irots = ""
        for ele in sorted(pg.elements, reverse=True):
            if isinstance(ele, InversionCenter):
                inv += ".note %s\n" % repr(ele)
                inv += ".sphere   %.5f  %.5f  %.5f  0.1\n" % tuple(pg.center)

            elif isinstance(ele, ProperRotation):
                prots += ".note %s\n" % repr(ele)
                prots += ".color red\n"
                prots += ".arrow   %.5f  %.5f  %.5f  " % tuple(pg.center)
                end = pg.center + ele.n * np.sqrt(ele.exp) * ele.axis
                prots += "%.5f  %.5f  %.5f  0.05\n"  % tuple(end)

            elif isinstance(ele, ImproperRotation):
                irots += ".note %s\n" % repr(ele)
                irots += ".color blue\n"
                irots += ".arrow   %.5f  %.5f  %.5f  " % tuple(pg.center)
                end = pg.center + np.sqrt(ele.n) * np.sqrt(ele.exp) * ele.axis
                irots += "%.5f  %.5f  %.5f  0.05\n"  % tuple(end)
                irots += ".transparency 25\n"
                z = ele.axis
                x = perp_vector(z)
                y = np.cross(x, z)
                for angle in np.linspace(0, 2 * np.pi, num=25):
                    pt2 = ele.n ** 0.9 * x * np.cos(angle)
                    pt2 += ele.n ** 0.9 * y * np.sin(angle)
                    pt2 += pg.center
                    if angle > 0:
                        irots += ".polygon  %6.3f  %6.3f  %6.3f" % tuple(pt1)
                        irots += "     %6.3f  %6.3f  %6.3f" % tuple(pg.center)
                        irots += "     %6.3f  %6.3f  %6.3f" % tuple(pt2)
                        irots += "\n"
                    pt1 = pt2

            elif isinstance(ele, MirrorPlane):
                mirror += ".note %s\n" % repr(ele)
                mirror += ".color purple\n"
                mirror += ".transparency 25\n"
                z = ele.axis
                x = perp_vector(z)
                y = np.cross(x, z)
                for angle in np.linspace(0, 2 * np.pi, num=25):
                    pt2 = 5 * x * np.cos(angle)
                    pt2 += 5 * y * np.sin(angle)
                    pt2 += pg.center
                    if angle > 0:
                        mirror += ".polygon  %6.3f  %6.3f  %6.3f" % tuple(pt1)
                        mirror += "     %6.3f  %6.3f  %6.3f" % tuple(pg.center)
                        mirror += "     %6.3f  %6.3f  %6.3f" % tuple(pt2)
                        mirror += "\n"
                    pt1 = pt2

        if args.outfile:
            with open(args.outfile, "w") as f:
                f.write("\n".join([inv, prots, irots, mirror]))
        else:
            if inv:
                print(inv)
            if prots:
                print(prots)
            if irots:
                print(irots)
            if mirror:
                print(mirror)

    else:
        s += "%s: %s\n" % (f, pg.name)
        if args.print_eles:
            for ele in sorted(pg.elements, reverse=True):
                s += "\t%s\n" % repr(ele)

if not args.print_bild:
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(s)
    else:
        print(s)
