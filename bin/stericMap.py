#!/usr/bin/env python3

import sys
import argparse
import copy

import numpy as np
import matplotlib.pyplot as plt

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.utils.utils import rotation_matrix, get_filename, glob_files


steric_parser = argparse.ArgumentParser(
    description="create a steric map for a ligand",
    formatter_class=argparse.RawTextHelpFormatter
)

steric_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

steric_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: show plot",
)

steric_parser.add_argument(
    "-if",
    "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin"
)

steric_parser.add_argument(
    "-k",
    "--key-atoms",
    default=None,
    required=False,
    dest="key",
    help="atoms coordinated to the center\n" +
    "these atoms\" ligands will be shown on the map",
)

steric_parser.add_argument(
    "-c",
    "--center",
    action="append",
    default=None,
    required=False,
    dest="center",
    help="atom the sphere is centered on\n" +
    "Default: detect metal center (centroid of all metals if multiple are present)",
)

steric_parser.add_argument(
    "-v",
    "--vdw-radii",
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

steric_parser.add_argument(
    "-r",
    "--radius",
    default=3.5,
    type=float,
    dest="radius",
    help="radius around center\nDefault: 3.5 Ångström"
)

steric_parser.add_argument(
    "-oop",
    "--out-of-plane",
    default=None,
    nargs=3,
    type=float,
    metavar=("x_oop", "y_oop", "z_oop"),
    dest="oop_vector",
    help="list of three numbers defining a vector perpendicular to\n" +
    "the desired steric map",
)

steric_parser.add_argument(
    "-ip",
    "--in-plane",
    default=None,
    nargs=3,
    type=float,
    metavar=("x_ip", "y_ip", "z_ip"),
    dest="ip_vector",
    help="list of three numbers defining the \"y axis\" of the steric map",
)

steric_parser.add_argument(
    "-n", "--number-of-points",
    default=100,
    type=int,
    dest="num_pts",
    help="number of points along x and y axes\nDefault: 100",
)

steric_parser.add_argument(
    "-amin", "--altitude-minimum",
    default=None,
    type=float,
    dest="min",
    help="manually set the lower cutoff of the altitude map",
)

steric_parser.add_argument(
    "-amax",
    "--altitude-maximum",
    default=None,
    type=float,
    dest="max",
    help="manually set the upper cutoff of the altitude map",
)

steric_parser.add_argument(
    "-p", "--projection-shape",
    choices=("circle", "square"),
    default="circle",
    dest="shape",
    help="shape of steric map\n" +
    "note that buried volume values with the square shape are still based\n" +
    "on a sphere around the center\n" +
    "Default: circle",
)

vbur_options = steric_parser.add_argument_group("Buried volume options")
vbur_options.add_argument(
    "-vbur", "--buried-volume",
    nargs="?",
    default=False,
    choices=("Lebedev", "MC"),
    # this allows these choices to be case-insensitive, but we can still
    # show normal upper- and lowercase in the help page
    type=lambda x: x.capitalize() if x.lower() == "lebedev" else x.upper(),
    dest="vbur",
    help="show buried volume in each quadrant using the specified method\n" +
    "Default: do not include %%Vbur",
)

vbur_options.add_argument(
    "-rp", "--radial-points",
    type=int,
    default=20,
    choices=[20, 32, 64, 75, 99, 127],
    dest="rpoints",
    help="number of radial shells for Lebedev integration\n" +
    "lower values are faster, but at the cost of accuracy\n" +
    "Default: 20"
)

vbur_options.add_argument(
    "-ap", "--angular-points",
    type=int,
    default=1454,
    choices=[110, 194, 302, 590, 974, 1454, 2030, 2702, 5810],
    dest="apoints",
    help="number of angular points for Lebedev integration\n" +
    "lower values are faster, but at the cost of accuracy\n" +
    "Default: 1454"
)

vbur_options.add_argument(
    "-i",
    "--minimum-iterations",
    type=int,
    default=25,
    metavar="ITERATIONS",
    dest="min_iter",
    help="minimum iterations - each is a batch of 3000 points\n" +
    "MC will continue after this until convergence criteria are met\n" +
    "Default: 25",
)

vbur_options.add_argument(
    "-s",
    "--scale",
    type=float,
    dest="scale",
    default=1.17,
    help="scale VDW radii by this amount\nDefault: 1.17"
)


args = steric_parser.parse_args()

oop_vector = args.oop_vector
if args.oop_vector is not None:
    oop_vector = np.array(args.oop_vector)

ip_vector = args.ip_vector
if args.ip_vector is not None:
    ip_vector = np.array(args.ip_vector)

if args.vbur is None:
    args.vbur = "Lebedev"

for f in glob_files(args.infile):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None))
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f))
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f))

    geom = Geometry(infile)

    x, y, z, min_alt, max_alt, basis, targets = geom.steric_map(
        center=args.center,
        key_atoms=args.key,
        radii=args.radii,
        return_basis=True,
        num_pts=args.num_pts,
        oop_vector=oop_vector,
        ip_vector=ip_vector,
        shape=args.shape,
    )

    if args.ip_vector is None or args.oop_vector is None:
        print(f)

    if args.oop_vector is None:
        z_vec = np.squeeze(basis[:, 2])
        print("out-of-plane vector: %s" % " ".join(["%6.3f" % yi for yi in z_vec]))

    if args.ip_vector is None:
        z_vec = np.squeeze(basis[:, 2])
        y_vec = np.squeeze(basis[:, 1])
        r15 = rotation_matrix(np.deg2rad(15), z_vec)
        yr = y_vec
        for i in range(1, 24):
            yr = np.dot(r15, yr)
            print("in-plane vector rotated by %5.1f degrees: %s" % (
                (15 * i), " ".join(["%6.3f" % yi for yi in yr])
            ))

    if args.min is not None:
        min_alt = args.min

    if args.max is not None:
        max_alt = args.max

    fig, ax = plt.subplots()
    steric_map = ax.contourf(
        x,
        y,
        z,
        extend="min",
        cmap=copy.copy(plt.cm.get_cmap("jet")),
        levels=np.linspace(min_alt, max_alt, num=20),
    )
    steric_map.cmap.set_under("w")
    steric_lines = ax.contour(
        x,
        y,
        z,
        extend="min",
        colors="k",
        levels=np.linspace(min_alt, max_alt, num=20),
    )
    bar = fig.colorbar(steric_map, format="%.1f")
    bar.set_label("altitude (Å)")
    ax.set_aspect("equal")

    if args.vbur:
        vbur = geom.percent_buried_volume(
            center=args.center,
            targets=targets,
            radius=args.radius,
            radii=args.radii,
            scale=args.scale,
            method=args.vbur,
            rpoints=args.rpoints,
            apoints=args.apoints,
            basis=basis,
            min_iter=args.min_iter,
        )

        ax.hlines(0, -args.radius, args.radius, color="k")
        ax.vlines(0, -args.radius, args.radius, color="k")

        ax.text(+0.7 * args.radius, +0.9 * args.radius, "%.1f%%" % vbur[0])
        ax.text(-0.9 * args.radius, +0.9 * args.radius, "%.1f%%" % vbur[1])
        ax.text(-0.9 * args.radius, -0.9 * args.radius, "%.1f%%" % vbur[2])
        ax.text(+0.7 * args.radius, -0.9 * args.radius, "%.1f%%" % vbur[3])

    if not args.outfile:
        plt.show()
    else:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        plt.savefig(outfile, dpi=500)
