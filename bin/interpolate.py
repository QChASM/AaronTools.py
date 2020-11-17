#!/usr/bin/env python3

import sys
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader
from AaronTools.pathway import Pathway

from warnings import warn

def width(n):
    """use to determine 0-padding based on number of files we're printing"""
    return np.ceil(np.log10(n))

instruction_args = ['-n', '-e', '-min', '-max', '-t']

interpolate_parser = argparse.ArgumentParser(description='interpolate between input structures', \
    formatter_class=argparse.RawTextHelpFormatter)
interpolate_parser.add_argument('infiles', metavar='infile', \
                            type=str, \
                            nargs='*', \
                            default=[], \
                            help='one or more input coordinate files')

interpolate_parser.add_argument('-n', '--print-nstruc', \
                            type=int, \
                            nargs=1, \
                            default=0, \
                            required=False, \
                            dest='n_struc', \
                            metavar='number of structures', \
                            help='number of interpolated structures to print')

interpolate_parser.add_argument('-max', '--print-maxima', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='print_max', \
                            help='print coordinates for potential energy maxima')

interpolate_parser.add_argument('-min', '--print-minima', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='print_min', \
                            help='print coordinates for potential energy minima')

interpolate_parser.add_argument('-e', '--print-energy', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='print_E', \
                            help='print energy and energy derivative instead of structures')

interpolate_parser.add_argument('-o', '--output-destination', \
                            type=str, \
                            nargs=1, \
                            default=[None], \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\n' + \
                            '"$i" will be replaced with zero-padded numbers\n' +
                            'Default: traj-$i.xyz for structures, stdout for energies')

interpolate_parser.add_argument('-u', '--use-unfinished', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='use_incomplete', \
                            help='use unfinished geometries (e.g. optimization still running)')

interpolate_parser.add_argument('-t', '--print-ts', \
                            type=float, \
                            nargs='+', \
                            default=False, \
                            required=False, \
                            metavar=('t1', 't2'), \
                            dest='specific_ts', \
                            help='space-separated list of t values at which to print structures \n{t| 0 <= t <= 1}')
"""
#TODO: add this back in
interpolate_parser.add_argument('-q', '--use-normal-modes', \
                            type=str, \
                            nargs=1, \
                            default=False, \
                            required=False, \
                            metavar='frequency_job.log', \
                            dest='freq_file', \
                            help='use normal mode displacements from specified file for \ninterpolation coordinates')
"""

args = interpolate_parser.parse_args()

if not any([args.specific_ts, args.print_E, args.print_min, args.print_max, args.n_struc]):
    interpolate_parser.print_help(sys.stderr)
    raise RuntimeError("one of the following flags should be used: %s" % ', '.join(instruction_args))

if args.use_incomplete and args.print_E:
    warn("Any unfinished geometry optimizations will be used to interpolate energies. Results may be nonsense")

#take '.xyz' off the outfile b/c aarontools adds it back
outfile = args.outfile[0]

#list of usable input geometries
Gs = []

#read each input file and see if it has normal modes
for f in args.infiles:
    G_file = FileReader(f, just_geom=False)
    if 'finished' not in G_file.other:
        #finished was never set (e.g. input was an xyz file), but we'll use it anyways
        if args.use_incomplete:
            G = Geometry(G_file)
            Gs.append(G)
        else:
            warn("not using %s because it is not marked as finished" % f)
    else:
        if G_file.other['finished'] or args.use_incomplete:
            #finished is False, but we'll use it anyways
            G = Geometry(G_file)
            G.other['energy'] = G_file.other['energy']
            Gs.append(G)
        else:
            warn("not using %s because it is not marked as finished" % f)

if len(Gs) <= 1:
    warn("nothing to interpolate: %i usable input structure%s" % (len(Gs), (1-len(Gs))*'s'))
    warn('use the -u option to include structures without an associated energy')
    sys.exit(0)

"""
#if a frequency job is supplied, the normal mode displacements will be used as 
#the interpolation coordinates
if args.freq_file:
    freq_file = FileReader(args.freq_file[0], just_geom=False) 
    #normal modes only look decent close to the original structure
    #we'll align everything to this geometry
    ref_G = Geometry(freq_file)
    freq_centroid = ref_G.COM(mass_weight=True)
    ref_G.coord_shift(vector = -1*freq_centroid)
    Q = [data.vector for data in freq_file.other['frequency'].data]
else:
    ref_G = Gs[0]
    Q = None
"""

ref_G = Gs[0]
Q = None

#align all input geometries to the reference
nrg = []
for G in Gs:
    centroid = G.COM(mass_weight=True)
    G.coord_shift(vector = -1*centroid)
    G.RMSD(ref=ref_G, align=True)
    if 'energy' in G.other:
        nrg.append(G.other['energy'])
    else:
        nrg.append(0)

#interpolate between the structures
S = Pathway(Gs[0], np.array([G.coords for G in Gs]), basis=Q, other_vars={'energy':nrg})
s_max, r_max = Pathway.t_to_s(1, S.region_length)

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
        dEdt = S.dvar_func_dt['energy'](t) * S.dvar_func_dt['energy'](t+dt)
        if dEdt <= 0 and S.dvar_func_dt['energy'](t) > 0 and args.print_max:
            max_n_min_ts.append(t)
        if dEdt <= 0 and S.dvar_func_dt['energy'](t) < 0 and args.print_min:
            max_n_min_ts.append(t)

    for i, t in enumerate(max_n_min_ts):
        E = S.var_func['energy'](t)
        if args.print_E:
            dE = S.dvar_func_dt['energy'](t)
            nrg_out += "%f\t%f\t%f\n" % (t,E,dE)
        else:
            G = S.Geom_func(t)
            comment = "E(%f) = %f" % (t, E)
            G.comment = comment
            write_geoms.append(G.copy())

if args.specific_ts:
    #print structures for specified values of t
    for i, t in enumerate(args.specific_ts):
        if args.print_E:
            E = S.var_func['energy'](t)
            dE = S.dvar_func_dt['energy'](t)
            nrg_out += "%f\t%f\t%f\n" % (t,E,dE)
        else:
            G = S.Geom_func(t)
            E = S.var_func['energy'](t)
            comment = "E(%f) = %f" % (t, E)
            G.comment = comment
            write_geoms.append(G.copy())

if args.print_E:
    if args.n_struc:
        ss = np.linspace(0, s_max, num=args.n_struc[0])
        for s in ss:
            t = Pathway.s_to_t(s, S.region_length)
            E = S.var_func['energy'](t)
            dE = S.dvar_func_dt['energy'](t)
            nrg_out += "%f\t%f\t%f\n" % (t,E,dE)
            
    if outfile is not None:
        with open(outfile, 'w') as f:
            f.write(nrg_out.rstrip())
    else:
        print(nrg_out.rstrip())

else:
    if args.n_struc:
        w = width(args.n_struc[0])
        fmt = "%0" + "%i" % w + "i"
    
        ts = np.linspace(0, 1, num=args.n_struc[0])
        for i, t in enumerate(ts):
            G = S.Geom_func(t)
            E = S.var_func['energy'](t)
            comment = "E(%f) = %f" % (t, E)
            G.comment = comment
            write_geoms.append(G.copy())

if len(write_geoms) > 0:
    w = width(len(write_geoms))
    fmt = "%0" + "%i" % w + "i"

if outfile is None:
    outfile = "traj-$i.xyz"

for i, G in enumerate(write_geoms):
    my_outfile = outfile.replace("$i", fmt % i)
    if my_outfile == outfile:
        #if there's no $i, we are writing all the structures to the same file
        G.write(append=True, outfile=my_outfile)
    else:
        G.write(append=False, outfile=my_outfile)
