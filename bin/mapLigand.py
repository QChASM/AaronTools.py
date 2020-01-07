#!/usr/bin/env python3

import argparse
from sys import stdin, argv, exit
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader, FileWriter, read_types
from AaronTools.catalyst import Catalyst
from AaronTools.component import Component

maplig_parser = argparse.ArgumentParser(description='replace a ligand on an organometallic system', \
    formatter_class=argparse.RawTextHelpFormatter)
maplig_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

maplig_parser.add_argument('-if', '--input-format', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                choices=read_types, \
                                dest='input_format', \
                                help="file format of input, required if input is stdin")

maplig_parser.add_argument('-l', '--ligand', metavar='ligand', \
                            type=str, \
                            nargs=1, \
                            default=None, \
                            required=False, \
                            dest='ligand', \
                            help="ligand used to replace the current one")

maplig_parser.add_argument('-k', '--key-atoms', metavar='index', \
                            type=str, \
                            nargs='+', \
                            default=None, \
                            required=False, \
                            dest='key', \
                            help="indices of key atoms on the ligand being replaced (1-indexed)")

maplig_parser.add_argument('-o', '--output', \
                            nargs=1, \
                            type=str, \
                            default=False, \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\nDefault: stdout')

args = maplig_parser.parse_args()

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
            maplig_parser.print_help()
            raise TypeError("when no input file is given, stdin is read and a format must be specified")
    
    cat = Catalyst(f)
    lig = args.ligand[0]
    if args.key is not None:
        cat.map_ligand(lig, args.key)
    else:
        cat.map_ligand(lig, cat.components['ligand'][0].key_atoms)

    if args.outfile:
        FileWriter.write_xyz(cat, append=False, outfile=args.outfile[0])
    else:
        s = FileWriter.write_xyz(cat, append=False, outfile=False) 
        print(s)

