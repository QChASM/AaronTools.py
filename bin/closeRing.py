#!/usr/bin/env python3

import argparse
from sys import stdin, argv, exit
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader, FileWriter, read_types
from AaronTools.geometry import Geometry
from AaronTools.ringfragment import RingFragment

ring_parser = argparse.ArgumentParser(description='close rings on a geometry', \
    formatter_class=argparse.RawTextHelpFormatter)
ring_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

ring_parser.add_argument('-if', '--input-format', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                choices=read_types, \
                                dest='input_format', \
                                help="file format of input, required if input is stdin")

ring_parser.add_argument('-r', '--ring', metavar='atom1 atom2 ring', \
                            type=str, \
                            nargs=3, \
                            action='append', \
                            default=None, \
                            required=True, \
                            dest='substitutions', \
                            help="substitution instructions \n" + \
                            "atom1 and atom2 specify the position to add the new ring")

ring_parser.add_argument('-f', '--format', \
                            type=str, \
                            nargs=1, \
                            choices=['from_library', 'iupac', 'smiles', 'element'], \
                            default=['from_library'], \
                            required=False, \
                            dest='form', \
                            help='how to get substituents given their names \nDefault: from_library')

ring_parser.add_argument('-o', '--output', \
                            nargs=1, \
                            type=str, \
                            default=False, \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\nDefault: stdout')

args = ring_parser.parse_args()

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
            ring_parser.print_help()
            raise TypeError("when no input file is given, stdin is read and a format must be specified")

    geom = Geometry(f)
   
    targets = {}

    for sub_info in args.substitutions:
        atom1 = geom.atoms[int(sub_info[0])-1]
        atom2 = geom.atoms[int(sub_info[1])-1]
        ring = sub_info[2]

        if args.form[0] == 'from_library':
            ring_geom = RingFragment(ring)
        else:
            path_length = len(geom.short_walk(atom1, atom2))-2
            ring_geom = RingFragment.from_string(ring, end=path_length)

        key = (atom1, atom2)
        if key in targets:
            targets[key].append(ring_geom)
        else:
            targets[key] = [ring_geom]

    for key in targets:
        for ring_geom in targets[key]:
            geom.ring_substitute(list(key), ring_geom)     

    if args.outfile:
        FileWriter.write_xyz(geom, append=False, outfile=args.outfile[0])
    else:
        s = FileWriter.write_xyz(geom, append=False, outfile=False) 
        print(s)

