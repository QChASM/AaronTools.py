#!/usr/bin/env python

import argparse
import sys
from warnings import warn

from AaronTools.comp_output import CompOutput
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.spectra import ValenceExcitations
from AaronTools.utils.utils import get_filename, glob_files

from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

rcParams["savefig.dpi"] = 300


peak_types = ["pseudo-voigt", "gaussian", "lorentzian", "delta"]
plot_types = [
    "transmittance", "transmittance-velocity",
    "uv-vis", "uv-vis-velocity",
    "ecd", "ecd-velocity"
]
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


uvvis_parser = argparse.ArgumentParser(
    description="plot Boltzmann-averaged UV/vis spectrum",
    formatter_class=argparse.RawTextHelpFormatter
)

uvvis_parser.add_argument(
    "infiles", metavar="files",
    type=str,
    nargs="+",
    help="TD-DFT or EOM job output files"
)

uvvis_parser.add_argument(
    "-o", "--output",
    type=str,
    default=None,
    dest="outfile",
    help="output destination\n"
    "if the file extension is .csv, a CSV file will be written\n"
    "Default: show plot"
)

uvvis_parser.add_argument(
    "-t", "--plot-type",
    type=plot_type,
    choices=plot_types,
    default="uv-vis-velocity",
    dest="plot_type",
    help="type of plot\nDefault: uv-vis-velocity",
)

uvvis_parser.add_argument(
    "-u", "--transient",
    action="store_true",
    dest="transient",
    help="use transient excitation data",
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
    help="width of each section specified by -sc/--section-centers\n"
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

energy_options = uvvis_parser.add_argument_group("energy weighting")
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
    help="cutoff frequency for quasi free energy corrections (1/cm)\n" +
    "Default: 100 cm^-1",
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

filereaders = []
for f in glob_files(args.infiles, parser=uvvis_parser):
    fr = FileReader(f, just_geom=False)
    filereaders.append(fr)

sp_cos = []
if args.sp_files is None:
    sp_cos = [CompOutput(f) for f in filereaders]
else:
    for f in glob_files(args.sp_files, parser=uvvis_parser):
        co = CompOutput(f)
        sp_cos.append(co)

compouts = []
if args.freq_files:
    for f in glob_files(args.freq_files, parser=uvvis_parser):
        co = CompOutput(f)
        compouts.append(co)

if (args.weighting == "electronic" or "frequency" in fr.other) and not compouts:
    compouts = [CompOutput(fr) for fr in filereaders]

for i, (fr, sp, freq) in enumerate(zip(filereaders, sp_cos, compouts)):
    geom = Geometry(fr)
    rmsd = geom.RMSD(sp.geometry, sort=True)
    if rmsd > 1e-2:
        print(
            "TD-DFT structure might not match SP energy file:\n"
            "%s %s RMSD = %.2f" % (fr.name,  sp.geometry.name, rmsd)
        )
    rmsd = geom.RMSD(freq.geometry, sort=True)
    if rmsd > 1e-2:
        print(
            "TD-DFT structure might not match frequency file:\n"
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


data_attr = "data"
if all(fr.other["uv_vis"].transient_data for fr in filereaders) and args.transient:
    data_attr = "transient_data"

mixed_uvvis = ValenceExcitations.get_mixed_signals(
    [fr.other["uv_vis"] for fr in filereaders],
    weights=weights,
    data_attr=data_attr,
)

if not args.outfile or not args.outfile.lower().endswith("csv"):
    fig = plt.gcf()
    fig.clear()
    
    mixed_uvvis.plot_uv_vis(
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
        plt.savefig(args.outfile, dpi=300)

    else:
        plt.show()

else:
    intensity_attr = "dipole_str"
    if args.plot_type.lower() == "uv-vis-veloctiy":
        intensity_attr = "dipole_vel"
    if args.plot_type.lower() == "ecd":
        intensity_attr = "rotatory_str_len"
    if args.plot_type.lower() == "ecd-velocity":
        intensity_attr = "rotatory_str_vel"

    change_x_unit_func = ValenceExcitations.ev_to_nm
    x_label = "wavelength (nm)"
    if units == "eV":
        change_x_unit_func = None
        x_label = r"$h\nu$ (eV)"


    funcs, x_positions, intensities = mixed_uvvis.get_spectrum_functions(
        fwhm=args.fwhm,
        peak_type=args.peak_type,
        voigt_mixing=args.voigt_mixing,
        scalar_scale=args.scalar_scale,
        linear_scale=args.linear_scale,
        quadratic_scale=args.quadratic_scale,
        intensity_attr=intensity_attr,
    )

    x_values, y_values, _ = mixed_uvvis.get_plot_data(
        funcs,
        x_positions,
        point_spacing=args.point_spacing,
        transmittance=args.plot_type == "transmittance",
        peak_type=args.peak_type,
        change_x_unit_func=change_x_unit_func,
        fwhm=args.fwhm,
    )

    if "transmittance" in args.plot_type.lower():
        y_label = "Transmittance (%)"
    elif args.plot_type.lower() == "uv-vis":
        y_label = "Absorbance (arb.)"
    elif args.plot_type.lower() == "uv-vis-velocity":
        y_label = "Absorbance (arb.)"
    elif args.plot_type.lower() == "ecd":
        y_label = "delta_Absorbance (arb.)"
    elif args.plot_type.lower() == "ecd-velocity":
        y_label = "delta_Absorbance (arb.)"
    else:
        y_label = "Absorbance (arb.)"

    with open(args.outfile, "w") as f:
        s = ",".join([x_label, y_label])
        s += "\n"
        for x, y in zip(x_values, y_values):
            s += ",".join(["%.4f" % z for z in [x, y]])
            s += "\n"
        f.write(s)

