#!/usr/bin/env python3

import sys
import argparse
from warnings import warn
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader
from AaronTools.pathway import Pathway


def width(n):
    """use to determine 0-padding based on number of files we"re printing"""
    return np.ceil(np.log10(n))

instruction_args = ["-n", "-e", "-min", "-max", "-t"]

interpolate_parser = argparse.ArgumentParser(
    description="interpolate between input structures",
    formatter_class=argparse.RawTextHelpFormatter
)

interpolate_parser.add_argument(
    "infiles", metavar="infile",
    type=str,
    nargs="*",
    default=[],
    help="one or more input coordinate files"
)

interpolate_parser.add_argument(
    "-n", "--print-nstruc",
    type=int,
    nargs=1,
    default=0,
    required=False,
    dest="n_struc",
    metavar="number of structures",
    help="number of interpolated structures to print"
)

interpolate_parser.add_argument(
    "-max", "--print-maxima",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="print_max",
    help="print coordinates for potential energy maxima"
)

interpolate_parser.add_argument(
    "-min", "--print-minima",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="print_min",
    help="print coordinates for potential energy minima"
)

interpolate_parser.add_argument(
    "-e", "--print-energy",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="print_E",
    help="print energy and energy derivative instead of structures"
)

interpolate_parser.add_argument(
    "-o", "--output-destination",
    type=str,
    nargs=1,
    default=[None],
    required=False,
    metavar="output destination",
    dest="outfile",
    help="output destination\n" +
    "$i will be replaced with zero-padded numbers\n" +
    "Default: traj-$i.xyz for structures, stdout for energies"
)

interpolate_parser.add_argument(
    "-u", "--use-unfinished",
    action="store_const",
    const=True,
    default=False,
    required=False,
    dest="use_incomplete",
    help="use unfinished geometries (e.g. optimization still running)"
)

interpolate_parser.add_argument(
    "-t", "--print-ts",
    type=float,
    nargs="+",
    default=False,
    required=False,
    metavar=("t1", "t2"),
    dest="specific_ts",
    help="space-separated list of t values at which to print structures \n{t| 0 <= t <= 1}"
)

args = interpolate_parser.parse_args()

if not any([args.specific_ts, args.print_E, args.print_min, args.print_max, args.n_struc]):
    interpolate_parser.print_help(sys.stderr)
    raise RuntimeError(
        "one of the following flags should be used: %s" % ", ".join(instruction_args)
    )

if args.use_incomplete and args.print_E:
    warn(
        "Any unfinished geometry optimizations will be used to interpolate energies.\n" +
        "Results may be nonsense"
    )

#take ".xyz" off the outfile b/c aarontools adds it back
outfile = args.outfile[0]

#list of usable input geometries
geom_list = []
nrg_list = []

#read each input file and see if it has normal modes
for f in args.infiles:
    fr = FileReader(f, just_geom=False)
    if "finished" not in fr.other:
        #finished was never set (e.g. input was an xyz file), but we"ll use it anyways
        if args.use_incomplete:
            geom = Geometry(fr)
            geom_list.append(geom)
        else:
            warn("not using %s because it is not marked as finished" % f)
    else:
        if fr.other["finished"] or args.use_incomplete:
            #finished is False, but we"ll use it anyways
            geom = Geometry(fr)
            nrg_list.append(fr.other["energy"])
            geom_list.append(geom)
        else:
            warn("not using %s because it is not marked as finished" % f)

if len(geom_list) <= 1:
    warn(
        "nothing to interpolate: %i usable input structure%s" % (
            len(geom_list), (1 - len(geom_list)) * "s"
        )
    )
    warn("use the -u option to include structures without an associated energy")
    sys.exit(0)

ref_geom = geom_list[0]

if len(nrg_list) < len(geom_list):
    nrg_list = np.zeros(len(geom_list))

#align all input geometries to the reference
for geom, nrg in zip(geom_list, nrg_list):
    centroid = geom.COM(mass_weight=True)
    geom.coord_shift(vector=-1 * centroid)
    geom.RMSD(ref=ref_geom, align=True)

#interpolate between the structures
pathway = Pathway(
    ref_geom,
    np.array([geom.coords for geom in geom_list]),
    other_vars={"energy": nrg_list}
)
s_max, r_max = Pathway.t_to_s(1, pathway.region_length)

#header for writing energies
nrg_out = "t\tE\tdE/dt\n"
#list of geometries to print
write_geoms = []

if args.print_max or args.print_min:
    #to find minima and maxima, find where derivative crosses 0 and
    #sign of derivative at neighboring points
    max_n_min_ts = []
    ts = np.linspace(0, 1, num=10001)
    dt = ts[1] - ts[0]
    for t in ts:
        dnrg_dt = pathway.dvar_func_dt["energy"](t) * pathway.dvar_func_dt["energy"](t + dt)
        if dnrg_dt <= 0 and pathway.dvar_func_dt["energy"](t) > 0 and args.print_max:
            max_n_min_ts.append(t)
        elif dnrg_dt <= 0 and pathway.dvar_func_dt["energy"](t) < 0 and args.print_min:
            max_n_min_ts.append(t)

    for i, t in enumerate(max_n_min_ts):
        nrg = pathway.var_func["energy"](t)
        if args.print_E:
            d_nrg = pathway.dvar_func_dt["energy"](t)
            nrg_out += "%f\t%f\t%f\n" % (t, nrg, d_nrg)
        else:
            geom = pathway.geom_func(t)
            comment = "E(%f) = %f" % (t, nrg)
            geom.comment = comment
            write_geoms.append(geom)

if args.specific_ts:
    #print structures for specified values of t
    for i, t in enumerate(args.specific_ts):
        if args.print_E:
            nrg = pathway.var_func["energy"](t)
            d_nrg = pathway.dvar_func_dt["energy"](t)
            nrg_out += "%f\t%f\t%f\n" % (t, nrg, d_nrg)
        else:
            geom = pathway.geom_func(t)
            nrg = pathway.var_func["energy"](t)
            comment = "E(%f) = %f" % (t, nrg)
            geom.comment = comment
            write_geoms.append(geom)

if args.print_E:
    if args.n_struc:
        ss = np.linspace(0, s_max, num=args.n_struc[0])
        for s in ss:
            t = Pathway.s_to_t(s, pathway.region_length)
            nrg = pathway.var_func["energy"](t)
            d_nrg = pathway.dvar_func_dt["energy"](t)
            nrg_out += "%f\t%f\t%f\n" % (t, nrg, d_nrg)

    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(nrg_out.rstrip())
    else:
        print(nrg_out.rstrip())

else:
    if args.n_struc:
        w = width(args.n_struc[0])
        fmt = "%0" + "%i" % w + "i"

        ts = np.linspace(0, 1, num=args.n_struc[0])
        for i, t in enumerate(ts):
            geom = pathway.geom_func(t)
            nrg = pathway.var_func["energy"](t)
            comment = "E(%f) = %f" % (t, nrg)
            geom.comment = comment
            write_geoms.append(geom)

if len(write_geoms) > 0:
    w = width(len(write_geoms))
    fmt = "%0" + "%i" % w + "i"

if outfile is None:
    outfile = "traj-$i.xyz"

for i, geom in enumerate(write_geoms):
    my_outfile = outfile.replace("$i", fmt % i)
    if my_outfile == outfile:
        #if there"s no $i, we are writing all the structures to the same file
        geom.write(append=True, outfile=my_outfile)
    else:
        geom.write(append=False, outfile=my_outfile)
