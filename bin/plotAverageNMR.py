#!/usr/bin/env python3

import argparse
import sys

from AaronTools.comp_output import CompOutput
from AaronTools.const import COMMONLY_ODD_ISOTOPES
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.spectra import NMR
from AaronTools.utils.utils import get_filename, glob_files

from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

rcParams["savefig.dpi"] = 300


peak_types = ["pseudo-voigt", "gaussian", "lorentzian", "delta"]
weight_types = ["electronic", "zero-point", "enthalpy", "free", "quasi-rrho", "quasi-harmonic"]


def weight_type(x):
    out = [y for y in weight_types if y.startswith(x)]
    if len(out) == 1:
        return out[0]
    raise TypeError(
        "weight type must be one of: %s" % ", ".join(
            weight_types
        )
    )

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


nmr_parser = argparse.ArgumentParser(
    description="plot NMR spectrum",
    formatter_class=argparse.RawTextHelpFormatter
)

nmr_parser.add_argument(
    "infiles", metavar="files",
    type=str,
    nargs="+",
    help="NMR job output file(s)"
)

nmr_parser.add_argument(
    "-o", "--output",
    type=str,
    default=None,
    dest="outfile",
    help="output destination\nDefault: show plot",
)

nmr_parser.add_argument(
    "-pf", "--pulse-frequency",
    default=60.0,
    type=float,
    dest="pulse_frequency",
    help="pulse frequency\nDefault: 60 MHz"
)

nmr_parser.add_argument(
    "-e", "--element",
    default="H",
    dest="element",
    help="plot shifts for specified element\nDefault: H"
)

nmr_parser.add_argument(
    "-c", "--couple-with",
    default=None,
    dest="couple_with",
    help="split only with the specified elements\nDefault: some common odd isotopes"
)

nmr_parser.add_argument(
    "-j", "--coupling-threshold",
    default=0.0,
    type=float,
    dest="coupling_threshold",
    help="coupling threshold when applying splits\nDefault: 0 Hz"
)

peak_options = nmr_parser.add_argument_group("peak options")
peak_options.add_argument(
    "-p", "--peak-type",
    type=peak_type,
    choices=peak_types,
    default="lorentzian",
    dest="peak_type",
    help="function for peaks\nDefault: lorentzian",
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
    default=2.5,
    dest="fwhm",
    help="full width at half max. of peaks\nDefault: 2.5 Hz",
)

nmr_parser.add_argument(
    "-s", "--point-spacing",
    default=None,
    type=float,
    dest="point_spacing",
    help="spacing between each x value\n"
    "Default: a non-uniform spacing that is more dense near peaks",
)

scale_options = nmr_parser.add_argument_group("scale shifts")
scale_options.add_argument(
    "-ss", "--scalar-shift",
    type=float,
    default=0.0,
    dest="scalar_scale",
    help="shift centers\n"
    "Default: 0 (no shift)",
)

scale_options.add_argument(
    "-l", "--linear-scale",
    type=float,
    default=-1.0,
    dest="linear_scale",
    help="multiply the center of each shift by linear_scale\n"
    "Default: -1 (no scaling)",
)

nmr_parser.add_argument(
    "-nr", "--no-reverse",
    action="store_false",
    default=True,
    dest="reverse_x",
    help="do not reverse x-axis",
)

center_centric = nmr_parser.add_argument_group("x-centered interruptions")
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

minmax_centric = nmr_parser.add_argument_group("x-range interruptions")
minmax_centric.add_argument(
    "-r", "--ranges",
    type=lambda x: [[float(v) for v in r.split("-")] for r in x.split(",")],
    dest="ranges",
    default=None,
    help="split plot into sections (e.g. 0-4,7-10)"
)

nmr_parser.add_argument(
    "-fw", "--figure-width",
    type=float,
    dest="fig_width",
    help="width of figure in inches"
)

nmr_parser.add_argument(
    "-fh", "--figure-height",
    type=float,
    dest="fig_height",
    help="height of figure in inches"
)

nmr_parser.add_argument(
    "-csv", "--experimental-csv",
    type=str,
    nargs="+",
    dest="exp_data",
    help="CSV file containing observed spectrum data, which will be plotted on top\n"
    "frequency job files should not come directly after this flag"
)

nmr_parser.add_argument(
    "-rx", "--rotate-x-ticks",
    action="store_true",
    dest="rotate_x_ticks",
    default=False,
    help="rotate x-axis tick labels by 45 degrees"
)


energy_options = nmr_parser.add_argument_group("energy weighting")
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
    "-freq", "--frequency-files",
    type=str,
    nargs="+",
    default=None,
    dest="freq_files",
    help="frequency jobs to use for thermochem"
)

energy_options.add_argument(
    "-sp", "--single-point-files",
    type=str,
    nargs="+",
    default=None,
    required=False,
    dest="sp_files",
    help="single point energies to use for thermochem\n"
    "Default: TD-DFT energies from INFILES"
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
    help="frequency parameter for quasi free energy corrections (1/cm)\n" +
    "Default: 100 cm^-1",
)


args = nmr_parser.parse_args()

couple_with = args.couple_with
if not args.couple_with:
    args.couple_with = COMMONLY_ODD_ISOTOPES
else:
    couple_with = couple_with.split(",")

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

exp_data = None
if args.exp_data:
    exp_data = []
    for f in args.exp_data:
        data = np.loadtxt(f, delimiter=",")
        
        for i in range(1, data.shape[1]):
            exp_data.append((data[:,0], data[:,i], None))

filereaders = []
for f in glob_files(args.infiles, parser=nmr_parser):
    fr = FileReader(f, just_geom=False)
    filereaders.append(fr)

sp_cos = []
if args.sp_files is None:
    sp_cos = [CompOutput(f) for f in filereaders]
else:
    for f in glob_files(args.sp_files, parser=nmr_parser):
        co = CompOutput(f)
        sp_cos.append(co)

compouts = []
if args.freq_files:
    for f in glob_files(args.freq_files, parser=nmr_parser):
        co = CompOutput(f)
        compouts.append(co)

if (args.weighting == "electronic" or "frequency" in fr.other) and not compouts:
    compouts = [CompOutput(fr) for fr in filereaders]

for i, (fr, sp, freq) in enumerate(zip(filereaders, sp_cos, compouts)):
    geom = Geometry(fr)
    rmsd = geom.RMSD(sp.geometry, sort=True)
    if rmsd > 1e-2:
        print(
            "NMR structure might not match SP energy file:\n"
            "%s %s RMSD = %.2f" % (fr.name,  sp.geometry.name, rmsd)
        )
    rmsd = geom.RMSD(freq.geometry, sort=True)
    if rmsd > 1e-2:
        print(
            "NMR structure might not match frequency file:\n"
            "%s %s RMSD = %.2f" % (fr.name,  freq.geometry.name, rmsd)
        )
    for freq2 in compouts[:i]:
        rmsd = freq.geometry.RMSD(freq2.geometry, sort=True)
        if rmsd < 1e-2:
            print(
                "two frequency files appear to be identical:\n"
                "%s %s RMSD = %.2f" % (freq2.geometry.name,  freq.geometry.name, rmsd)
            )

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

weights = CompOutput.boltzmann_weights(
    compouts,
    nrg_cos=sp_cos,
    temperature=args.temperature,
    weighting=weighting,
    v0=args.w0,
)


mixed_nmr = NMR.get_mixed_signals(
    [fr["nmr"] for fr in filereaders],
    weights=weights,
)

if not args.outfile or not args.outfile.lower().endswith("csv"):
    fig = plt.gcf()
    fig.clear()
    
    mixed_nmr.plot_nmr(
        fig,
        centers=centers,
        widths=widths,
        peak_type=args.peak_type,
        fwhm=args.fwhm,
        point_spacing=args.point_spacing,
        voigt_mixing=args.voigt_mixing,
        scalar_scale=args.scalar_scale,
        linear_scale=args.linear_scale,
        exp_data=exp_data,
        rotate_x_ticks=args.rotate_x_ticks,
        geometry=Geometry(filereaders[0]["atoms"]),
        element=args.element,
    )
    
    if args.fig_width:
        fig.set_figwidth(args.fig_width)
    
    if args.fig_height:
        fig.set_figheight(args.fig_height)
    
    if args.outfile:
        plt.savefig(args.outfile, dpi=300)

    else:
        plt.show()

else:

    x_label = "shift (ppm)"

    funcs, x_positions, intensities = mixed_nmr.get_spectrum_functions(
        fwhm=args.fwhm,
        peak_type=args.peak_type,
        voigt_mixing=args.voigt_mixing,
        scalar_scale=args.scalar_scale,
        linear_scale=args.linear_scale,
    )

    x_values, y_values, _ = mixed_nmr.get_plot_data(
        funcs,
        x_positions,
        point_spacing=args.point_spacing,
        peak_type=args.peak_type,
        fwhm=args.fwhm,
    )

    y_label = "Intensity (arb.)"

    with open(args.outfile, "w") as f:
        s = ",".join([x_label, y_label])
        s += "\n"
        for x, y in zip(x_values, y_values):
            s += ",".join(["%.4f" % z for z in [x, y]])
            s += "\n"
        f.write(s)

