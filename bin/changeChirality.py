#!/usr/bin/env python3

import argparse
import os
import re

from numpy import prod, pi
from sys import stdin, argv, exit
from AaronTools.fileIO import FileReader, read_types
from AaronTools.finders import BondedTo, ChiralCenters
from AaronTools.geometry import Geometry
from warnings import warn

changechiral_parser = argparse.ArgumentParser(description='change handedness of chiral centers', \
    formatter_class=argparse.RawTextHelpFormatter)
changechiral_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

changechiral_parser.add_argument('-i', '--info', \
                                action='store_const', \
                                const=True, \
                                default=False, \
                                required=False, \
                                dest='list_info', \
                                help='list information on detected chiral centers')

changechiral_parser.add_argument('-if', '--input-format', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                choices=read_types, \
                                dest='input_format', \
                                help="file format of input - xyz is assumed if input is stdin")

changechiral_parser.add_argument('-t', '--targets', \
                            type=str, \
                            default=None, \
                            action='append', \
                            required=False, \
                            dest='targets', \
                            help="comma- or hyphen-seperated list of chiral centers to invert (1-indexed)\n" + \
                            "Chiral centers must have at least two fragments not in a ring\n" + \
                            "Detected chiral centers are atoms that:\n" + \
                            "    - have > 2 bonds\n" + \
                            "    - have a non-planar VSEPR shape\n" + \
                            "    - each connected fragment is distinct\n" + \
                            "Default: change chirality of any detected chiral centers")

changechiral_parser.add_argument('-d', '--diastereomers', \
                                action='store_const', \
                                const=True, \
                                default=False, \
                                required=False, \
                                dest='combos', \
                                help='print all diastereomers for selected chiral centers')

changechiral_parser.add_argument('-o', '--output-destination', \
                            type=str, \
                            default=None, \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\n' +
                                 '$i in the filename will be replaced with a number\n' + \
                                 'if a directory is given, default is \'diastereomer-$i.xyz\' in that directory\n' + \
                                 'Default: stdout')

args = changechiral_parser.parse_args()

s = ""

for infile in args.infile:
    if isinstance(infile, str):
        if args.input_format is not None:
            f = FileReader((infile, args.input_format[0], infile))
        else:
            f = FileReader(infile)
    else:
        if args.input_format is not None:
            f = FileReader(('from stdin', args.input_format[0], stdin))
        else:
            f = FileReader(('from stdin', 'xyz', stdin))

    geom = Geometry(f)

    target_list = []
    if args.targets is None:
        try:
            target_list = geom.find(ChiralCenters())
        except LookupError as e:
            warn(str(e))
    else:
        for targ in args.targets:
            target_list.extend(geom.find(targ))

    if args.list_info:
        if len(args.infile) > 1:
            s += "%s\n" % infile
        s += "Target\tElement\n"
        for targ in target_list:
            s += "%-2s\t%-2s\n" % (targ.name, targ.element)
        if infile is not args.infile[-1]:
            s += '\n'
        continue

    if args.combos:
        #this stuff is copy-pasted from makeConf, so it's a bit overkill
        #for getting all diastereomers, as each chiral center can only
        #have 2 options instead of the random number of rotamers
        #a substituent can have
        diastereomers = []
        for targ in target_list:
            diastereomers.append(2)
    
        mod_array = []
        for i in range(0, len(diastereomers)):
            mod_array.append(1)
            for j in range(i + 1, len(diastereomers)):
                mod_array[i] *= diastereomers[j]

        prev_diastereomer = 0
        for diastereomer in range(0, int(prod(diastereomers))):
            for i, targ in enumerate(target_list):
                rot = int( diastereomer / mod_array[i]) % diastereomers[i]
                rot -= int( prev_diastereomer / mod_array[i]) % diastereomers[i]
                angle = rot * pi
                if angle != 0:
                    qualifying_fragments = []
                    for atom in targ.connected:
                        frag = geom.get_fragment(atom, targ)
                        if sum([int(targ in frag_atom.connected) for frag_atom in frag]) == 1:
                            qualifying_fragments.append(frag)

                    if len(qualifying_fragments) < 2:
                        warn("cannot change chirality of atom %s; must have at least two groups not in a ring" % targ.name)
                    else:
                        while len(qualifying_fragments) > 2:
                            for frag in qualifying_fragments:
                                if len(frag) == max([len(f) for f in qualifying_fragments]):
                                    qualifying_fragments.remove(frag)
                                    break
            
                        frag1, frag2 = qualifying_fragments
                        atom1 = geom.find(frag1, BondedTo(targ))[0]
                        atom2 = geom.find(frag2, BondedTo(targ))[0]
                        v1 = targ.bond(atom1)
                        v2 = targ.bond(atom2)
                        rv = v1 + v2

                        geom.rotate(rv, angle=pi, targets=frag1+frag2, center=targ)

            prev_diastereomer = diastereomer

            if args.outfile is None:
                s += geom.write(outfile=False)
                s += '\n'
            else:
                if os.path.isdir(os.path.expanduser(args.outfile)):
                    outfile = os.path.join(os.path.expanduser(args.outfile), "diastereomer-%i.xyz" % (diastereomer + 1))
            
                else:
                    outfile = args.outfile.replace('$i', str(diastereomer + 1))
            
                geom.write(outfile=outfile, append='$i' not in args.outfile)

    else:
        for targ in target_list:
            qualifying_fragments = []
            for atom in targ.connected:
                frag = geom.get_fragment(atom, targ)
                if sum([int(targ in frag_atom.connected) for frag_atom in frag]) == 1:
                    qualifying_fragments.append(frag)

            if len(qualifying_fragments) < 2:
                warn("cannot change chirality of atom %s; must have at least two groups not in a ring" % targ.name)
            else:
                while len(qualifying_fragments) > 2:
                    for frag in qualifying_fragments:
                        if len(frag) == max([len(f) for f in qualifying_fragments]):
                            qualifying_fragments.remove(frag)
                            break
            
                frag1, frag2 = qualifying_fragments
                atom1 = geom.find(frag1, BondedTo(targ))[0]
                atom2 = geom.find(frag2, BondedTo(targ))[0]
                v1 = targ.bond(atom1)
                v2 = targ.bond(atom2)
                rv = v1 + v2

                geom.rotate(rv, angle=pi, targets=frag1+frag2, center=targ)

        if args.outfile is None:
            s += geom.write(outfile=False)
            s += '\n'
        else:
            geom.write(outfile=args.outfile)

    
if args.outfile is None or args.list_info:
    print(s[:-1])
