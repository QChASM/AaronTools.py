#!/usr/bin/env python3

import argparse
import sys

from AaronTools.comp_output import CompOutput
from AaronTools.fileIO import Frequency, FileReader
from AaronTools.utils.utils import get_filename, glob_files

from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

rcParams["savefig.dpi"] = 300


peak_types = ["pseudo-voigt", "gaussian", "lorentzian", "delta"]
plot_types = ["transmittance", "absorbance", "vcd"]
weight_types = ["electronic", "zero-point", "enthalpy", "free", "quasi-rrho", "quasi-harmonic"]

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

def weight_type(x):
    out = [y for y in weight_types if y.startswith(x)]
    if len(out) == 1:
        return out[0]
    raise TypeError(
        "weight type must be one of: %s" % ", ".join(
            weight_types
        )
    )


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
    type=plot_type,
    choices=["transmittance", "absorbance", "vcd"],
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
    type=peak_type,
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

energy_options = ir_parser.add_argument_group("energy interruptions")
energy_options.add_argument(
    "-w", "--weighting-energy",
    type=weight_type,
    dest="weighting",
    default="quasi-rrho",
    choices=weight_types,
    help="type of energy to use for Boltzmann weighting\n"
    "Default: quasi-rrho",
)

energy_options.add_argument(
    "-temp", "--temperature",
    type=float,
    dest="temperature",
    default=298.15,
    help="temperature (K) to use for weighting\n"
    "Default: 298.15",
)

energy_options.add_argument(
    "-w0", "--frequency-cutoff",
    type=float,
    dest="w0",
    default=100,
    help="cutoff frequency for quasi free energy corrections (1/cm)\n" +
    "Default: 100 cm^-1",
)

args = ir_parser.parse_args()

exp_data = None
if args.exp_data:
    exp_data = []
    for f in args.exp_data:
        data = np.loadtxt(f, delimiter=",")
        
        for i in range(1, data.shape[1]):
            exp_data.append((data[:,0], data[:,i], None))

compouts = []
for f in glob_files(args.infiles, parser=ir_parser):
    fr = FileReader(f, just_geom=False)
    co = CompOutput(fr)
    compouts.append(co)

if args.weighting == "electronic":
    weighting = CompOutput.ELECTRONIC_ENERGY
elif args.weighting == "zero-point":
    weighting = CompOutput.ZEROPOINT_ENERGY
elif args.weighting == "enthalpy":
    weighting = CompOutput.RRHO_ENTHALPY
elif args.weighting == "free":
    weighting = CompOutput.RRHO
elif args.weighting == "quasi-rrho":
    weighting = CompOutput.QUASI_RRHO
elif args.weighting == "quasi-harmonic":
    weighting = CompOutput.QUASI_HARMONIC

funcs, freqs, intens = CompOutput.mix_spectra(
    compouts,
    peak_type=args.peak_type,
    plot_type=args.plot_type,
    fwhm=args.fwhm,
    voigt_mixing=args.voigt_mixing,
    linear_scale=args.linear_scale,
    quadratic_scale=args.quadratic_scale,
    # anharmonic=all(bool(co.frequency.anharm_data) for co in compouts) and args.anharmonic,
    anharmonic=False,
    weighting=weighting,
    temperature=args.temperature,
    v0=args.w0,
)

x_values, y_values = Frequency.get_ir_data(
    funcs, freqs, intens,
    point_spacing=args.point_spacing,
    plot_type=args.plot_type,
    peak_type=args.peak_type,
    normalize=True,
    fwhm=args.fwhm,
)

if args.plot_type == "transmittance":
    y_label = "Transmittance (%)"
elif args.plot_type == "absorbance":
    y_label = "Absorbance (arb.)"
elif args.plot_type == "vcd":
    y_label = "ΔAbsorbance (arb.)"


fig = plt.gcf()
Frequency.plot_spectrum(
    fig,
    x_values,
    y_values,
    centers=args.centers,
    widths=args.widths,
    exp_data=exp_data,
    reverse_x=args.reverse_x,
    peak_type=args.peak_type,
    plot_type=args.plot_type,
    y_label=y_label,
)
    
if args.fig_width:
    fig.set_figwidth(args.fig_width)

if args.fig_height:
    fig.set_figheight(args.fig_height)

if args.outfile:
    if args.outfile.lower().endswith("csv"):
        with open(args.outfile, "w") as f:
            y_label = y_label.replace("Δ", "delta ")
            s = ",".join(["frequency (cm^-1)", y_label])
            s += "\n"
            for x, y in zip(x_values, y_values):
                s += ",".join(["%.4f" % z for z in [x, y]])
                s += "\n"
            f.write(s)
    else:
        plt.savefig(args.outfile, dpi=300)
else:
    plt.show()
