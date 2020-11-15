#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types

vbur_parser = argparse.ArgumentParser(description='calculated % volume buried in a sphere around a center atom', \
    formatter_class=argparse.RawTextHelpFormatter)
vbur_parser.add_argument('infile', metavar='input file', 
                         type=str, 
                         nargs='*', 
                         default=[sys.stdin], 
                         help='a coordinate file'
)

vbur_parser.add_argument('-o', '--output', 
                        type=str, 
                        default=False, 
                        required=False, 
                        dest='outfile', 
                        help='output destination \nDefault: stdout'
)

vbur_parser.add_argument('-if', '--input-format', 
                        type=str, 
                        default=None, 
                        dest='input_format', 
                        choices=read_types, 
                        help="file format of input - xyz is assumed if input is stdin"
)

vbur_parser.add_argument('-k', '--key-atoms', 
                         default=None, 
                         required=False, 
                         dest='ligand_atoms', 
                         help='key atoms to define ligand to consider in calculation\nDefault: use all ligands',
)

vbur_parser.add_argument('-e', '--exclude-atoms', 
                         default=None, 
                         required=False, 
                         dest='exclude_atoms', 
                         help='atoms to exclude from the calculation\nDefault: exclude no ligand atoms',
)

vbur_parser.add_argument('-c', '--center', 
                         action="append", 
                         default=None, 
                         required=False, 
                         dest='center', 
                         help='atom the sphere is centered on\nDefault: detect metal center (centroid of all metals if multiple are present)',
)

vbur_parser.add_argument('-v', '--vdw-radii', 
                         default="umn",
                         choices=["umn", "bondi"],
                         dest="radii",
                         help="VDW radii to use in calculation\nDefault: umn",
)

vbur_parser.add_argument('-s', '--scale',
                         default=1.17,
                         type=float,
                         dest="scale",
                         help="scale VDW radii by this amount\nDefault: 1.17",
)

vbur_parser.add_argument('-r', '--radius',
                         default=3.5,
                         type=float,
                         dest="radius",
                         help="radius around center\nDefault: 3.5 Ångström"
)


vbur_parser.add_argument('-m', '--method',
                         type=str,
                         default="Lebedev",
                         choices=["MC", "Lebedev"],
                         dest="method",
                         help="integration method\nDefault: Lebedev"
)

grid_options = vbur_parser.add_argument_group('Lebedev integration options')
grid_options.add_argument('-rp', '--radial-points',
                         type=int,
                         default=20,
                         choices=[20, 32, 64, 75, 99, 127],
                         dest="rpoints",
                         help="number of radial shells for Lebedev integration\nlower values are faster, but at the cost of accuracy\nDefault: 20"
)

grid_options.add_argument('-ap', '--angular-points',
                         type=int,
                         default=1454,
                         choices=[110, 194, 302, 590, 974, 1454, 2030, 2702, 5810],
                         dest="apoints",
                         help="number of angular points for Lebedev integration\nlower values are faster, but at the cost of accuracy\nDefault: 1454"
)

args = vbur_parser.parse_args()

s = ""

np.set_printoptions(precision=5)

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

    ligands = args.ligand_atoms
    if args.ligand_atoms is not None:
        ligands = []
        if not geom.components:
            geom.detect_components()
        key = geom.find(args.ligand_atoms)
        for comp in geom.components:
            if any(k in comp.atoms for k in key):
                ligands.append(comp)
   
    exclude = args.exclude_atoms
    if args.exclude_atoms is not None:
        exlude = geom.find(exclude)
    
    try:
        vbur = geom.percent_buried_volume(ligands=ligands, 
                                          center=args.center, 
                                          radius=args.radius, 
                                          radii=args.radii, 
                                          scale=args.scale, 
                                          exclude=exclude, 
                                          method=args.method, 
                                          rpoints=args.rpoints, 
                                          apoints=args.apoints
        ) 

        if len(args.infile) > 1:
            s += "%20s:\t" % f

        s += "%4.1f\n" % vbur
    except Exception as e:
        raise RuntimeError("calulation failed for %s: %s" % (f, e))

if not args.outfile:
    print(s)
else:
    with open(args.outfile, "w") as f:
        f.write(s)
