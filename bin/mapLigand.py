#!/usr/bin/env python3

import argparse
from sys import stdin, argv, exit
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader, read_types
from AaronTools.catalyst import Catalyst
from AaronTools.component import Component

maplig_parser = argparse.ArgumentParser(description='replace a ligand on an organometallic system', \
    formatter_class=argparse.RawTextHelpFormatter)
maplig_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

maplig_parser.add_argument('-ls', '--list', \
                                action="store_const", \
                                const=True, \
                                default=False, \
                                required=False, \
                                dest='list_avail', \
                                help='list available ligands')

maplig_parser.add_argument('-if', '--input-format', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                choices=read_types, \
                                dest='input_format', \
                                help="file format of input - xyz is assumed if input is stdin")

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
                            default=[False], \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\nDefault: stdout')

args = maplig_parser.parse_args()

if args.list_avail:
    s = ""
    for i, name in enumerate(sorted(Component.list())):
        s += "%-35s" % name
        if (i + 1) % 3 == 0:
        #if (i + 1) % 1 == 0:
            s += '\n'

    print(s.strip())
    exit(0)

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
    
    cat = Catalyst(f)
    lig = args.ligand[0]
    if args.key is not None:
        cat.map_ligand(lig, args.key)
    else:
        ligand = Component(lig)
        old_key_atoms = []
        j = 0
        while len(old_key_atoms) < len(ligand.key_atoms):
            if j >= len(cat.components['ligand']):
                raise RuntimeError("new ligand appears to have a higher denticity than old ligands combined")
            else:
                old_key_atoms.extend(cat.components['ligand'][j].key_atoms)
                j += 1

        if len(old_key_atoms) % len(ligand.key_atoms) == 0:
            k = 0
            ligands = []
            while k != len(old_key_atoms):
                k += len(ligand.key_atoms)
                ligands.append(ligand.copy())
        
            cat.map_ligand(ligands, old_key_atoms)

    s = cat.write(append=False, outfile=args.outfile[0])
    if not args.outfile[0]:
        print(s)
