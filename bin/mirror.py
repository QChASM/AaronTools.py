#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import get_filename, glob_files

mirror_parser = argparse.ArgumentParser(
    description="mirror a molecular structure",
    formatter_class=argparse.RawTextHelpFormatter
)

mirror_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file"
)

mirror_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    choices=read_types,
    dest="input_format",
    help="file format of input - xyz is assumed if input is stdin"
)

mirror_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    metavar="output destination",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

plane_options = mirror_parser.add_argument_group("plane")
plane_options.add_argument(
    "-yz", "--yz-plane",
    action="store_true",
    default=False,
    dest="yz_plane",
    help="mirror across the yz plane (default)"
)

plane_options.add_argument(
    "-xz", "--xz-plane",
    action="store_true",
    default=False,
    dest="xz_plane",
    help="mirror across the xz plane"
)

plane_options.add_argument(
    "-xy", "--xy-plane",
    action="store_true",
    default=False,
    dest="xy_plane",
    help="mirror across the xy plane"
)


args = mirror_parser.parse_args()

eye = np.identity(3)

if args.yz_plane:
    eye[0, 0] *= -1

if args.xz_plane:
    eye[1, 1] *= -1

if args.xy_plane:
    eye[2, 2] *= -1

if np.sum(eye) == 3:
    eye[0, 0] *= -1

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
            infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)

    geom.update_geometry(np.dot(geom.coords, eye))

    if args.outfile:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", outfile)
        geom.write(append=True, outfile=outfile)
    else:
        print(geom.write(outfile=False))
