#!/usr/bin/env python3

import argparse
from warnings import warn

import numpy as np
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.pathway import Pathway
from AaronTools.utils.utils import get_filename


def width(n):
    """use to determine 0-padding based on number of files we're printing"""
    return np.ceil(np.log10(n))


def parse_mode_str(s, t):
    """split mode string into modes and mode combos
    e.g.
    t=int, 1,2+3,4 -> [[0], [1,2], [3]]
    t=float 0.1,0.05+0.03,0.07 -> [[0.1], [0.05, 0.03], [0.07]]"""

    # the way this is being used is if t is int, we are changing 1-indexed things to 0-index
    # if t is float, were going to use the result to scale a normal mode (don't subtract 1)

    if t is not int and t is not float:
        raise TypeError(
            "can only parse mode string into ints or floats, not %s" % repr(t)
        )

    modes = s.split(",")
    out_modes = []
    for mode in modes:
        out_modes.append([])
        for combo in mode.split("+"):
            if t is int:
                out_modes[-1].append(int(combo) - 1)
            elif t is float:
                out_modes[-1].append(float(combo))

    return out_modes


follow_parser = argparse.ArgumentParser(
    description="move the structure along a normal mode",
    formatter_class=argparse.RawTextHelpFormatter,
)
follow_parser.add_argument(
    "input_file",
    metavar="input file",
    type=str,
    help='input frequency file (i.e. Gaussian output where "freq" was specified)',
)

follow_parser.add_argument(
    "-m",
    "--mode",
    type=str,
    nargs="+",
    required=False,
    default=None,
    dest="mode",
    metavar=("mode 1", "mode 2"),
    help="comma-separated list of modes to follow (1-indexed)",
)

follow_parser.add_argument(
    "-r",
    "--reverse",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="reverse",
    help="follow the normal mode in the opposite direction",
)

follow_parser.add_argument(
    "-a",
    "--animate",
    type=int,
    nargs=1,
    default=None,
    required=False,
    dest="animate",
    metavar="frames",
    help="print specified number of structures to make an animation",
)

follow_parser.add_argument(
    "-rt",
    "--roundtrip",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="roundtrip",
    help="make animation roundtrip",
)

follow_parser.add_argument(
    "-s",
    "--scale",
    type=str,
    nargs=1,
    default=None,
    required=False,
    dest="scale",
    metavar="max displacement",
    help="scale the normal mode so that this is the maximum amount an \natom is displaced",
)

follow_parser.add_argument(
    "-o",
    "--output-destination",
    type=str,
    nargs="+",
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$i in file name will be replaced with zero-padded numbers if --animate is used\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout",
)

args = follow_parser.parse_args()

fr = FileReader(args.input_file, just_geom=False)
geom = Geometry(fr)

if args.mode is None:
    modes = [
        [i]
        for i, freq in enumerate(fr.other["frequency"].data)
        if freq.frequency < 0
    ]
else:
    modes = parse_mode_str(args.mode[0], int)

# copy the list of output files or set all output files to False (print all to stdout)
if args.outfile is not False:
    outfiles = [f for f in args.outfile]
else:
    outfiles = [False for m in modes]

if len(outfiles) != len(modes):
    warn(
        "number of output files does not match number of modes: %i files, %i modes"
        % (len(outfiles), len(modes))
    )

if args.scale is None:
    scale = [[0.35] * len(mode) for mode in modes]
else:
    scale = parse_mode_str(args.scale[0], float)

for i, mode in enumerate(modes):
    if outfiles[i]:
        if "$i" not in outfiles[i]:
            append = True
    else:
        append = False
    dX = np.zeros((len(geom.atoms), 3))
    # figure out how much we'll have to scale each mode
    for j, combo in enumerate(mode):
        max_norm = 0
        for k, v in enumerate(
                fr.other["frequency"].data[combo].vector
        ):
            n = np.linalg.norm(v)
            if n > max_norm:
                max_norm = n

        # scale this mode by 0.35 (or whatever the user asked for)/max_norm
        x_factor = scale[i][j] / max_norm

        if args.reverse:
            x_factor *= -1

        dX += x_factor * fr.other["frequency"].data[combo].vector

    if args.animate is not None:
        # animate by setting up 3 geometries: -, 0, and +
        # then create a Pathway to interpolate between these
        # if roundtrip, - -> 0 -> + -> 0 -> -
        # if not roundtrip, - -> 0 -> +
        w = width(args.animate[0])
        fmt = "%0" + "%i" % w + "i"

        Xf = geom.coords + dX
        X = geom.coords
        Xr = geom.coords - dX

        # make a scales(t) function so we can see the animation progress in the XYZ file comment
        if args.roundtrip:
            other_vars = {}
            for i, mode_scale in enumerate(scale[i]):
                other_vars["scale %i"] = [
                    mode_scale,
                    0,
                    -mode_scale,
                    0,
                    mode_scale,
                ]
            pathway = Pathway(geom, np.array([Xf, X, Xr, X, Xf]), other_vars=other_vars)

        else:
            other_vars = {}
            for i, mode_scale in enumerate(scale[i]):
                other_vars["scale %i"] = [mode_scale, 0, -mode_scale]
            pathway = Pathway(geom, np.array([Xf, X, Xr]), other_vars=other_vars)

        # print animation frames
        for k, t in enumerate(np.linspace(0, 1, num=args.animate[0])):
            if outfiles[i] is not False:
                outfile = outfiles[i].replace("$i", fmt % k)
            else:
                outfile = outfiles[i]

            followed_geom = pathway.geom_func(t)
            followed_geom.comment = (
                "animating mode %s scaled to displace at most [%s]"
                % (
                    repr(mode),
                    ", ".join(str(pathway.var_func[key](t)) for key in other_vars),
                )
            )
            if args.outfile:
                followed_geom.write(
                    append=False,
                    outfile=outfile.replace("$INFILE", get_filename(args.input_file))
                )
            else:
                print(followed_geom.write(outfile=False))

    else:
        w = width(len(modes))
        fmt = "%0" + "%i" % w + "i"

        followed_geom = geom.copy()
        followed_geom.update_geometry(geom.coords + dX)
        followed_geom.comment = "following mode %s scaled to displace at most %s" % (
            repr(mode),
            repr(scale[i]),
        )

        outfile = outfiles[i]
        if args.outfile:
            if "$INFILE" in outfile:
                outfile = outfile.replace("$INFILE", get_filename(args.input_file))
            followed_geom.write(append=True, outfile=outfile)
        else:
            print(followed_geom.write(outfile=False))
