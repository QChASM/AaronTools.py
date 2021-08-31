#!/usr/bin/env python3

import argparse
import sys

from AaronTools.fileIO import FileReader
from AaronTools.utils.utils import get_filename, glob_files

from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

rcParams["savefig.dpi"] = 300


peak_types = ["pseudo-voigt", "gaussian", "lorentzian", "delta"]
plot_types = ["transmittance", "transmittance-velocity", "uv-vis", "uv-vis-velocity", "ecd", "ecd-velocity"]


def peak_type(x):
    out = [y for y in peak_types if y.startswith(x)]
    if out:
        return out[0]
    raise TypeError(
        "peak type must be one of: %s" % ", ".join(
            peak_types
        )
    )

def plot_type(x):
    out = [y for y in plot_types if y.startswith(x)]
    if out:
        return out[0]
    raise TypeError(
        "plot type must be one of: %s" % ", ".join(
            plot_types
        )
    )


uvvis_parser = argparse.ArgumentParser(
    description="plot UV/vis spectrum",
    formatter_class=argparse.RawTextHelpFormatter
)

uvvis_parser.add_argument(
    "infiles", metavar="files",
    type=str,
    nargs="+",
    help="TD-DFT or EOM job output file(s)"
)

uvvis_parser.add_argument(
    "-o", "--output",
    type=str,
    default=None,
    dest="outfile",
    help="output destination\nDefault: show plot",
)

uvvis_parser.add_argument(
    "-t", "--plot-type",
    type=plot_type,
    choices=plot_types,
    default="uv-vis-velocity",
    dest="plot_type",
    help="type of plot\nDefault: uv-vis-velocity (absorbance)",
)

uvvis_parser.add_argument(
    "-ev", "--electron-volt",
    action="store_true",
    default=False,
    dest="ev_unit",
    help="use eV on x axis instead of nm",
)


peak_options = uvvis_parser.add_argument_group("peak options")
peak_options.add_argument(
    "-p", "--peak-type",
    type=peak_type,
    choices=peak_types,
    default="gaussian",
    dest="peak_type",
    help="function for peaks\nDefault: gaussian",
)

peak_options.add_argument(
    "-m", "--voigt-mixing",
    type=float,
    default=0.5,
    dest="voigt_mixing",
    help="fraction of pseudo-Voigt that is Gaussian\nDefault: 0.5",
)

peak_options.add_argument(
    "-fwhm", "--full-width-half-max",
    type=float,
    default=0.5,
    dest="fwhm",
    help="full width at half max. of peaks\nDefault: 0.5 eV",
)

uvvis_parser.add_argument(
    "-s", "--point-spacing",
    default=None,
    type=float,
    dest="point_spacing",
    help="spacing between each x value\n"
    "Default: a non-uniform spacing that is more dense near peaks",
)


scale_options = uvvis_parser.add_argument_group("scale energies (in eV)")
scale_options.add_argument(
    "-ss", "--scalar-shift",
    type=float,
    default=0.0,
    dest="scalar_scale",
    help="subtract scalar shift from each excitation\n"
    "Default: 0 (no shift)",
)

scale_options.add_argument(
    "-l", "--linear-scale",
    type=float,
    default=0.0,
    dest="linear_scale",
    help="subtract linear_scale * energy from each excitation\n"
    "Default: 0 (no scaling)",
)

scale_options.add_argument(
    "-q", "--quadratic-scale",
    type=float,
    default=0.0,
    dest="quadratic_scale",
    help="subtract quadratic_scale * energy^2 from each excitation\n"
    "Default: 0 (no scaling)",
)

center_centric = uvvis_parser.add_argument_group("x-centered interruptions")
center_centric.add_argument(
    "-sc", "--section-centers",
    type=lambda x: [float(v) for v in x.split(",")],
    dest="centers",
    default=None,
    help="split plot into sections with a section centered on each of the specified values\n"
    "values should be separated by commas"
)

center_centric.add_argument(
    "-sw", "--section-widths",
    type=lambda x: [float(v) for v in x.split(",")],
    dest="widths",
    default=None,
    help="width of each section specified by -c/--centers\n"
    "should be separated by commas, with one for each section"
)

minmax_centric = uvvis_parser.add_argument_group("x-range interruptions")
minmax_centric.add_argument(
    "-r", "--ranges",
    type=lambda x: [[float(v) for v in r.split("-")] for r in x.split(",")],
    dest="ranges",
    default=None,
    help="split plot into sections (e.g. 200-350,400-650)"
)

uvvis_parser.add_argument(
    "-fw", "--figure-width",
    type=float,
    dest="fig_width",
    help="width of figure in inches"
)

uvvis_parser.add_argument(
    "-fh", "--figure-height",
    type=float,
    dest="fig_height",
    help="height of figure in inches"
)

uvvis_parser.add_argument(
    "-csv", "--experimental-csv",
    type=str,
    nargs="+",
    dest="exp_data",
    help="CSV file containing observed spectrum data, which will be plotted on top\n"
    "frequency job files should not come directly after this flag"
)

uvvis_parser.add_argument(
    "-rx", "--rotate-x-ticks",
    action="store_true",
    dest="rotate_x_ticks",
    default=False,
    help="rotate x-axis tick labels by 45 degrees"
)

args = uvvis_parser.parse_args()


if bool(args.centers) != bool(args.widths):
    sys.stderr.write(
        "both -sw/--section-widths and -sc/--section-centers must be specified"
    )
    sys.exit(2)

if args.ranges and bool(args.ranges) == bool(args.widths):
    sys.stderr.write(
        "cannot use -r/--ranges with -sw/--section-widths"
    )
    sys.exit(2)
   
centers = args.centers
widths = args.widths
if args.ranges:
    centers = []
    widths = []
    for (xmin, xmax) in args.ranges:
        centers.append((xmin + xmax) / 2)
        widths.append(abs(xmax - xmin))

units = "nm"
if args.ev_unit:
    units = "eV"

exp_data = None
if args.exp_data:
    exp_data = []
    for f in args.exp_data:
        data = np.loadtxt(f, delimiter=",")
        
        for i in range(1, data.shape[1]):
            exp_data.append((data[:,0], data[:,i], None))

for f in glob_files(args.infiles, parser=uvvis_parser):
    fr = FileReader(f, just_geom=False)

    uv_vis = fr.other["uv_vis"]

    fig = plt.gcf()
    fig.clear()

    uv_vis.plot_uv_vis(
        fig,
        centers=centers,
        widths=widths,
        plot_type=args.plot_type,
        peak_type=args.peak_type,
        fwhm=args.fwhm,
        point_spacing=args.point_spacing,
        voigt_mixing=args.voigt_mixing,
        scalar_scale=args.scalar_scale,
        linear_scale=args.linear_scale,
        quadratic_scale=args.quadratic_scale,
        exp_data=exp_data,
        units=units,
        rotate_x_ticks=args.rotate_x_ticks,
    )

    if args.fig_width:
        fig.set_figwidth(args.fig_width)

    if args.fig_height:
        fig.set_figheight(args.fig_height)

    if args.outfile:
        outfile_name = args.outfile.replace("$INFILE", get_filename(f))
        plt.savefig(outfile_name, dpi=300)
    else:
        plt.show()
