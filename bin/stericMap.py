#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.finders import NotAny

from warnings import warn

import matplotlib.pyplot as plt

import copy

steric_parser = argparse.ArgumentParser(description='create a steric map for a ligand', \
    formatter_class=argparse.RawTextHelpFormatter)
steric_parser.add_argument('infile', metavar='input file', 
                         type=str, 
                         nargs='*', 
                         default=[sys.stdin], 
                         help='a coordinate file'
)

steric_parser.add_argument('-o', '--output', 
                           type=str, 
                           default=False, 
                           required=False, 
                           dest='outfile', 
                           help='output destination \nDefault: show plot'
)

steric_parser.add_argument('-if', '--input-format', 
                           type=str, 
                           default=None, 
                           dest='input_format', 
                           choices=read_types, 
                           help="file format of input - xyz is assumed if input is stdin"
)

steric_parser.add_argument('-k', '--key-atoms', 
                           default=None, 
                           required=False, 
                           dest='key', 
                           help='atoms coordinated to the center\n' + 
                                'these atoms\' ligands will be shown on the map',
)

steric_parser.add_argument('-c', '--center', 
                           action="append", 
                           default=None, 
                           required=False, 
                           dest='center', 
                           help='atom the sphere is centered on\nDefault: detect metal center (centroid of all metals if multiple are present)',
)

steric_parser.add_argument('-v', '--vdw-radii', 
                           default="umn",
                           choices=["umn", "bondi"],
                           dest="radii",
                           help="VDW radii to use in calculation\nDefault: umn",
)

steric_parser.add_argument('-r', '--radius',
                           default=3.5,
                           type=float,
                           dest="radius",
                           help="radius around center\nDefault: 3.5 Ångström"
)

steric_parser.add_argument('-oop', '--out-of-plane',
                           default=None,
                           nargs=3,
                           type=float,
                           dest="oop_vector",
                           help="list of three numbers defining a vector perpendicular to\n" +
                                "the desired steric map",
)

steric_parser.add_argument('-ip', '--in-plane',
                           default=None,
                           nargs=3,
                           type=float,
                           dest="ip_vector",
                           help="list of three numbers defining the \"y axis\" of the steric map",
)

steric_parser.add_argument('-n', '--number-of-points',
                           default=100,
                           type=int,
                           dest="num_pts",
                           help="number of points along x and y axes\nDefault: 100",
)

vbur_options = steric_parser.add_argument_group('Buried volume options')
vbur_options.add_argument('-vbur', '--buried-volume',
                          action='store_true',
                          default=False,
                          dest="vbur",
                          help="show buried volume in each quadrant",
)

vbur_options.add_argument('-rp', '--radial-points',
                          type=int,
                          default=20,
                          choices=[20, 32, 64, 75, 99, 127],
                          dest="rpoints",
                          help="number of radial shells for Lebedev integration\nlower values are faster, but at the cost of accuracy\nDefault: 20"
)

vbur_options.add_argument('-ap', '--angular-points',
                          type=int,
                          default=1454,
                          choices=[110, 194, 302, 590, 974, 1454, 2030, 2702, 5810],
                          dest="apoints",
                          help="number of angular points for Lebedev integration\nlower values are faster, but at the cost of accuracy\nDefault: 1454"
)

vbur_options.add_argument('-s', '--scale',
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


for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None))
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(('from stdin', args.input_format[0], f))
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(('from stdin', 'xyz', f))

    geom = Geometry(infile)

    x, y, z, min_alt, max_alt, basis, targets = geom.steric_map(
        center=args.center,
        key_atoms=args.key,
        radii=args.radii,
        return_basis=True,
        num_pts=args.num_pts,
        oop_vector=oop_vector,
        ip_vector=ip_vector,
    )

    fig, ax = plt.subplots()
    steric_map = ax.contourf(x, y, z, extend="min", cmap=copy.copy(plt.cm.get_cmap("jet")), levels=np.linspace(min_alt, max_alt, num=20))
    steric_map.cmap.set_under('w')
    steric_lines = ax.contour(x, y, z, extend="min", colors='k', levels=np.linspace(min_alt, max_alt, num=20))
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
            method="lebedev", 
            rpoints=args.rpoints, 
            apoints=args.apoints,
            basis=basis,
        )
        
        ax.hlines(0, -args.radius, args.radius, color='k')
        ax.vlines(0, -args.radius, args.radius, color='k')
        
        ax.text( 0.7 * args.radius,  0.9 * args.radius, "%.1f%%" % vbur[0])
        ax.text(-0.9 * args.radius,  0.9 * args.radius, "%.1f%%" % vbur[1])
        ax.text(-0.9 * args.radius, -0.9 * args.radius, "%.1f%%" % vbur[2])
        ax.text( 0.7 * args.radius, -0.9 * args.radius, "%.1f%%" % vbur[3])

    if args.outfile is None:
        plt.show()
    else:
        plt.savefig(args.outfile, dpi=500)