#!/usr/bin/env python3

import argparse
import os
import re

from numpy import prod
from sys import stdin, argv, exit
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader, read_types
from AaronTools.finders import BondedTo
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

makeconf_parser = argparse.ArgumentParser(description='replace an atom or substituent with another', \
    formatter_class=argparse.RawTextHelpFormatter)
makeconf_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

makeconf_parser.add_argument('-ls', '--list', \
                                action='store_const', \
                                const=True, \
                                default=False, \
                                required=False, \
                                dest='list_avail', \
                                help='list available substituents')

makeconf_parser.add_argument('-i', '--info', \
                                action='store_const', \
                                const=True, \
                                default=False, \
                                required=False, \
                                dest='list_info', \
                                help='list information on substituents to be generated')

makeconf_parser.add_argument('-if', '--input-format', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                choices=read_types, \
                                dest='input_format', \
                                help="file format of input - xyz is assumed if input is stdin")

makeconf_parser.add_argument('-s', '--substituent', metavar='(n=substituent|substituent name)', \
                            type=str, \
                            nargs='*', \
                            default=[], \
                            required=False, \
                            dest='substituents', \
                            help="substituents to rotate \n" + \
                            "n is the 1-indexed position of the starting position of the\n" + \
                            "substituent you are rotating\n" +
                            "if only a substituent name is specified, all substituents with\n" +\
                            "that name will be rotated\n" + \
                            "Default: rotate any detected substituents")

makeconf_parser.add_argument('-o', '--output-destination', \
                            type=str, \
                            default=None, \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\n' +
                                 '$i in the filename will be replaced with conformer number\n' + \
                                 'if a directory is given, default is \'conformer-$i.xyz\' in that directory\n' + \
                                 'Default: stdout')

args = makeconf_parser.parse_args()

#Tony: Let's only print rotatable subs (ie, ones with more than one conformer)
if args.list_avail:
    s = ""
    for i, name in enumerate(sorted(Substituent.list())):
        s += "%-20s" % name
        #if (i + 1) % 3 == 0:
        if (i + 1) % 1 == 0:
            s += '\n'

    print(s.strip())
    exit(0)

detect_subs = False

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
    explicit_subnames = []
    subnames = []
    if len(args.substituents) == 0:
        detect_subs = True
    else:
        for sub in args.substituents:
            if re.search("^\d+=", sub):
                ndx_targets = sub.split('=')[0]
                target_list.append(geom.find(ndx_targets))
                explicit_subnames.append('='.join(sub.split('=')[1:]))
            else:
                detect_subs = True
                subnames.append(sub)

    substituents = []
    if detect_subs:
        geom.detect_substituents()
        if len(subnames) == 0:
            substituents.extend([sub for sub in geom.substituents if sub.name in Substituent.list()])
        else:
            substituents.extend([sub for sub in geom.substituents if sub.name in subnames])

    for a, subname in zip(target_list, explicit_subnames):
        for atom in a:
            for bonded_atom in atom.connected:
                frag = geom.get_fragment(bonded_atom, atom)
                try:
                    sub = Substituent(frag, end=atom)
                    if sub.name == subname:
                        substituents.append(sub)

                except LookupError:
                    pass
    
    if args.list_info:
        total_conf = 1
        if len(args.infile) > 1:
            s += "%s\n" % infile
        else:
            s += ''
        s += 'Substituent \tRotamers\n'
        for sub in substituents:
            if sub.conf_num > 1:
                s += "%2s=%-10s\t%s\n" % (sub.end.name, sub.name, sub.conf_num)
                total_conf *= sub.conf_num
        s += "Total Number of Conformers = %i\n" % total_conf
        if infile is not args.infile[-1]:
            s += '\n'
        continue

    conformers = []
    rotations = []
    for sub in substituents:
        conformers.append(sub.conf_num)
        rotations.append(sub.conf_angle)
    
    mod_array = []
    for i in range(0, len(rotations)):
        mod_array.append(1)
        for j in range(i + 1, len(rotations)):
            mod_array[i] *= conformers[j]

    prev_conf = 0
    for conf in range(0, int(prod(conformers))):
        for i, sub in enumerate(substituents):
            rot = int( (conf - 1) / mod_array[i]) % conformers[i]
            rot -= int( (prev_conf - 1) / mod_array[i]) % conformers[i]
            angle = rotations[i] * rot
            if angle != 0:
                sub_atom = sub.find_exact(BondedTo(sub.end))[0]
                axis = sub_atom.bond(sub.end)
                center = sub.end.coords
                geom.rotate(axis, angle=angle, targets=sub.atoms, center=center)

        prev_conf = conf

        if args.outfile is None:
            s += geom.write(outfile=False)
            s += '\n'
        else:
            if os.path.isdir(os.path.expanduser(args.outfile)):
                outfile = os.path.join(os.path.expanduser(args.outfile), "conformer-%i.xyz" % (conf + 1))
            
            else:
                outfile = args.outfile.replace('$i', str(conf + 1))
           
            geom.write(outfile=outfile, append='$i' not in args.outfile)

    
if args.outfile is None or args.list_info:
    print(s[:-1])
