#!/usr/bin/env python3

import argparse
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader

def parse_mode_str(s, t):
    """split mode string into modes and mode combos
    e.g.
    t=int, 1,2+3,4 -> [[0], [1,2], [3]]
    t=float 0.1,0.05+0.03,0.07 -> [[0.1], [0.05, 0.03], [0.07]]"""

    #the way this is being used is if t is int, we are changing 1-indexed things to 0-index
    #if t is float, were going to use the result to scale a normal mode (don"t subtract 1)

    if t is not int and t is not float:
        raise TypeError("can only parse mode string into ints or floats, not %s" % repr(t))

    modes = s.split(",")
    out_modes = []
    for mode in modes:
        out_modes.append([])
        for combo in mode.split("+"):
            if t is int:
                out_modes[-1].append(int(combo)-1)
            elif t is float:
                out_modes[-1].append(float(combo))

    return out_modes

freqbild_parser = argparse.ArgumentParser(
    description="print Chimera bild file with vectors for the specified normal modes to std out",
    formatter_class=argparse.RawTextHelpFormatter
)
freqbild_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    default=None,
    help="a frequency job output file"
)

freqbild_parser.add_argument(
    "-m", "--mode",
    type=str,
    nargs=1,
    default=None,
    required=False,
    metavar="1,2+3,4",
    dest="mode_str",
    help="mode(s) to print (1-indexed)\n" +
    "Default is to print all imaginary modes separately\n" +
    "- comma (,) delimited modes will be printed separately\n" +
    "- plus (+) delimited modes will be combined"
)

freqbild_parser.add_argument(
    "-s", "--scale",
    type=str,
    nargs=1,
    default=None,
    required=False,
    dest="scale",
    metavar="max displacement",
    help="scale the longest vector to be this many Angstroms long\n" +
    "default is 1.5\nmay be delimited in accordance with the --mode option"
)

freqbild_parser.add_argument(
    "-r", "--remove-mass",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="mass_weight",
    help="remove mass-weighting from normal modes"
)

freqbild_parser.add_argument(
    "-c", "--color",
    type=str,
    nargs="+",
    default=["green"],
    required=False,
    dest="color",
    metavar=("BILD 1 color", "BILD 2 color"),
    help="color of vectors"
)

args = freqbild_parser.parse_args()

fr = FileReader(args.infile, just_geom=False)
geom = Geometry(fr)

if args.mode_str is None:
    #if not modes were requested, print all the imaginary ones
    modes = [
        [i] for i, freq in enumerate(fr.other["frequency"].data) if freq.frequency < 0
    ]
else:
    #otherwise, split the modes on delimiters
    modes = parse_mode_str(args.mode_str[0], int)

if args.scale is None:
    scale = [[1.5]*len(mode) for mode in modes]
else:
    scale = parse_mode_str(args.scale[0], float)

color = args.color
colors = len(color)
while colors < len(modes):
    color.extend(args.color)
    colors = len(color)

#output is the string of everything we"ll print
output = ""

for i, mode in enumerate(modes):
    output += ".color %s\n.comment " % args.color[i]
    dX = np.zeros((len(geom.atoms), 3))
    #figure out how much we"ll have to scale each mode
    for j, combo in enumerate(mode):
        output += "%f cm^-1" % fr.other["frequency"].data[combo].frequency
        max_norm = 0
        for k, v in enumerate(fr.other["frequency"].data[combo].vector):
            if args.mass_weight:
                n = np.linalg.norm(v) * geom.atoms[k].mass()
            else:
                n = np.linalg.norm(v)
            if n > max_norm:
                max_norm = n

        # scale this mode by 1.5 (or whatever the user asked for)/max_norm
        x_factor = scale[i][j]/max_norm
        dX += x_factor * fr.other["frequency"].data[combo].vector

        output += " x %.2f " % x_factor

    output += "\n"

    for n in range(0, len(geom.atoms)):
        # scale the vector for each atom and add it to output
        if args.mass_weight:
            dX[n] *= geom.atoms[n].mass()

        v_len = np.linalg.norm(dX[n])

        # we also scale the cone part of the arrow
        start = [x for x in geom.atoms[n].coords]
        stop = [x for x in geom.atoms[n].coords + dX[n]]
        head_len = [v_len / (v_len + 0.75)]

        if v_len > 0.1:
            output += (
                ".arrow %10.6f %10.6f %10.6f   " % tuple(start) +
                "%10.6f %10.6f %10.6f   " % tuple(stop) +
                "0.02 0.05 %5.3f\n" % tuple(head_len)
            )

print(output.rstrip())
