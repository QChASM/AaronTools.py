#!/usr/bin/env python3

import argparse
import sys

from AaronTools.fileIO import FileReader
from AaronTools.utils.utils import get_filename, glob_files

from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

rcParams["savefig.dpi"] = 300


ir_parser = argparse.ArgumentParser(
    description="plot IR spectrum",
    formatter_class=argparse.RawTextHelpFormatter
)

ir_parser.add_argument(
    "infiles", metavar="files",
    type=str,
    nargs="+",
    help="frequency job output file(s)"
)

ir_parser.add_argument(
    "-o", "--output",
    type=str,
    default=None,
    dest="outfile",
    help="output destination\nDefault: show plot",
)

ir_parser.add_argument(
    "-t", "--plot-type",
    type=str,
    choices=["transmittance", "absorbance"],
    default="transmittance",
    dest="plot_type",
    help="type of plot\nDefault: transmittance",
)


# TODO: figure out more anharmonic options
# anharmonic_options = ir_parser.add_argument_group("anharmonic options")
ir_parser.add_argument(
    "-na", "--harmonic",
    action="store_false",
    default=True,
    dest="anharmonic",
    help="force to use harmonic frequencies when anharmonic data is in the file",
)


peak_options = ir_parser.add_argument_group("peak options")
peak_options.add_argument(
    "-p", "--peak-type",
    type=str,
    choices=["pseudo-voigt", "gaussian", "lorentzian", "delta"],
    default="pseudo-voigt",
    dest="peak_type",
    help="function for peaks\nDefault: pseudo-voigt",
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
    default=15.0,
    dest="fwhm",
    help="full width at half max. of peaks\nDefault: 15 cm^1",
)

ir_parser.add_argument(
    "-s", "--point-spacing",
    default=None,
    type=float,
    dest="point_spacing",
    help="spacing between each x value\n"
    "Default: a non-uniform spacing that is more dense near peaks",
)


scale_options = ir_parser.add_argument_group("scale frequencies")
scale_options.add_argument(
    "-l", "--linear-scale",
    type=float,
    default=0.0,
    dest="linear_scale",
    help="subtract linear_scale * frequency from each mode (i.e. this is 1 - λ)\n"
    "Default: 0 (no scaling)",
)

scale_options.add_argument(
    "-q", "--quadratic-scale",
    type=float,
    default=0.0,
    dest="quadratic_scale",
    help="subtract quadratic_scale * frequency^2 from each mode\n"
    "Default: 0 (no scaling)",
)

ir_parser.add_argument(
    "-nr", "--no-reverse",
    action="store_false",
    default=True,
    dest="reverse_x",
    help="do not reverse x-axis",
)


section_options = ir_parser.add_argument_group("x-axis interruptions")
section_options.add_argument(
    "-sc", "--section-centers",
    type=lambda x: [float(v) for v in x.split(",")],
    dest="centers",
    default=None,
    help="split plot into sections with a section centered on each of the specified values\n"
    "values should be separated by commas"
)

section_options.add_argument(
    "-sw", "--section-widths",
    type=lambda x: [float(v) for v in x.split(",")],
    dest="widths",
    default=None,
    help="width of each section specified by -c/--centers\n"
    "should be separated by commas, with one for each section"
)

ir_parser.add_argument(
    "-fw", "--figure-width",
    type=float,
    dest="fig_width",
    help="width of figure in inches"
)

ir_parser.add_argument(
    "-fh", "--figure-height",
    type=float,
    dest="fig_height",
    help="height of figure in inches"
)

ir_parser.add_argument(
    "-csv", "--experimental-csv",
    type=str,
    nargs="+",
    dest="exp_data",
    help="CSV file containing observed spectrum data, which will be plotted on top\n"
    "frequency job files should not come directly after this flag"
)

args = ir_parser.parse_args()

exp_data = None
if args.exp_data:
    exp_data = []
    for f in args.exp_data:
        data = np.loadtxt(f, delimiter=",")
        
        for i in range(1, data.shape[1]):
            exp_data.append((data[:,0], data[:,i], None))

for f in glob_files(args.infiles):
    fr = FileReader(f, just_geom=False)

    freq = fr.other["frequency"]

    fig = plt.gcf()
    fig.clear()

    freq.plot_ir(
        fig,
        centers=args.centers,
        widths=args.widths,
        plot_type=args.plot_type,
        peak_type=args.peak_type,
        reverse_x=args.reverse_x,
        fwhm=args.fwhm,
        point_spacing=args.point_spacing,
        voigt_mixing=args.voigt_mixing,
        linear_scale=args.linear_scale,
        quadratic_scale=args.quadratic_scale,
        exp_data=exp_data,
        anharmonic=freq.anharm_data and args.anharmonic,
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
