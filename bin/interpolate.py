#!/usr/bin/env python3

import sys
import os
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader
from AaronTools.trajectory import Pathway

from warnings import warn

def width(n):
    """use to determine 0-padding based on number of files we're printing"""
    return np.ceil(np.log10(n))

instruction_args = ['-n', '-e', '-min', '-max', '-t']

interpolate_parser = argparse.ArgumentParser(description='interpolate between input structures')
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
                            help='number of interpolated structures to print')

interpolate_parser.add_argument('-max', '--print-maxima', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='print_max', \
                            help='print coordinates for potential energy maxima instead of a fixed number of structures')

interpolate_parser.add_argument('-min', '--print-minima', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='print_min', \
                            help='print coordinates for potential energy minima instead of a fixed number of structures')

interpolate_parser.add_argument('-e', '--print-energy', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='print_E', \
                            help='print energy and energy derivative instead of structures (use with --print-nstruc)')

interpolate_parser.add_argument('-o', '--output-destination', \
                            type=str, \
                            nargs=1, \
                            default=["traj-&i.xyz"], \
                            required=False, \
                            dest='outfile', \
                            help='output destination (default: traj-&i.xyz for structures) - "&i" will be replaced with zero-padded numbers')

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
                            dest='specific_ts', \
                            help='space-separated list of t values, at which to print structures {t| 0 <= t <= 1}')

interpolate_parser.add_argument('-q', '--use-normal-modes', \
                            type=str, \
                            nargs=1, \
                            default=False, \
                            required=False, \
                            dest='freq_file', \
                            help='use normal mode displacements from specified file for interpolation coordinates')

args = interpolate_parser.parse_args()

if not any([args.specific_ts, args.print_E, args.print_min, args.print_max, args.n_struc]):
    warn("one of the following flags should be used: %s" % ', '.join(instruction_args))
    interpolate_parser.print_help(sys.stderr)
    sys.exit(1)

#take '.xyz' off the outfile b/c aarontools adds it back
args.outfile = args.outfile[0].rstrip('.xyz')

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

#align all input geometries to the reference
for G in Gs:
    centroid = G.COM(mass_weight=True)
    G.coord_shift(vector = -1*centroid)
    G.RMSD(ref=ref_G, align=True)

#interpolate between the structures
S = Pathway(Gs, Q)
s_max, r_max = Pathway.t_to_s(1, S.region_length)

if args.print_max or args.print_min:
    #to find minima and maxima, find where derivative crosses 0 and 
    #sign of derivative at neighboring points
    max_n_min_ts = []
    ts = np.linspace(0, 1, num=10001)
    dt = ts[1] - ts[0]
    for t in ts:
        dEdt = S.dE_func(t)*S.dE_func(t+dt)
        if dEdt <= 0 and S.dE_func(t) > 0 and args.print_max:
            max_n_min_ts.append(t)
        if dEdt <= 0 and S.dE_func(t) < 0 and args.print_min:
            max_n_min_ts.append(t)

    w = width(len(max_n_min_ts))
    fmt = "%0" + "%i" % w + "i"
    for i, t in enumerate(max_n_min_ts):
        G = S.Geom_func(t)
        E = S.E_func(t)
        outfile = args.outfile.replace("&i", fmt % i) 
        comment = "E(%f) = %f" % (t, E)
        G.comment = comment
        G.write(name=outfile)

elif args.specific_ts:
    #print structures for specified values of t
    w = width(len(args.specific_ts))
    fmt = "%0" + "%i" % w + "i"
    
    for i, t in enumerate(args.specific_ts):
        G = S.Geom_func(t)
        outfile = args.outfile.replace("&i", fmt % i)
        G.write(name = outfile)

else:
    #print a fixed number of structures
    if args.print_E:
        if args.n_struc:
            ss = np.linspace(0, s_max, num=args.n_struc[0])
            out = ''
            for s in ss:
                t = Pathway.s_to_t(s, S.region_length)
                E = S.E_func(t)
                dE = S.dE_func(t)
                out += "%f\t%f\t%f\n" % (t,E,dE)
            
            if args.outfile != "traj-&i":
                with open(args.outfile, 'w') as f:
                    f.write(out.rstrip())
            else:
                print(out.rstrip())
        else:
            warn("energy print requested, but number of data points not specified")
    else:
        if args.n_struc:
            w = width(args.n_struc[0])
            fmt = "%0" + "%i" % w + "i"
    
            ts = np.linspace(0, 1, num=args.n_struc[0])
            for i, t in enumerate(ts):
                G = S.Geom_func(t)
                E = S.E_func(t)
                comment = "E(%f) = %f" % (t, E)
                G.comment = comment
                outfile = args.outfile.replace("&i", fmt % i)
                G.write(name=outfile)
        
        else:
            warn("no interpolation requested")

