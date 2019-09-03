#!/usr/bin/env python3

import argparse
from sys import stdin, argv, exit
from AaronTools.atoms import Atom
from AaronTools.component import Component
from AaronTools.fileIO import FileReader, FileWriter, read_types
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

substitute_parser = argparse.ArgumentParser(description='replace an atom or substituent with another', \
    formatter_class=argparse.RawTextHelpFormatter)
substitute_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

substitute_parser.add_argument('-if', '--input-format', \
                                type=str, \
                                nargs=1, \
                                default=None, \
                                choices=read_types, \
                                dest='input_format', \
                                help="file format of input, required if input is stdin")

substitute_parser.add_argument('-s', '--substitute', metavar='n=new substituent', \
                            type=str, \
                            nargs='*', \
                            default=None, \
                            required=False, \
                            dest='substitutions', \
                            help="substitution instructions \n" + \
                            "n is the 1-indexed position of the starting position of the\n" + \
                            "substituent you are replacing")

substitute_parser.add_argument('-f', '--format', \
                            type=str, \
                            nargs=1, \
                            choices=['from_library', 'iupac', 'smiles', 'element'], \
                            default=['from_library'], \
                            required=False, \
                            dest='form', \
                            help='how to get substituents given their names \nDefault: from_library')

substitute_parser.add_argument('-m', '--minimize', \
                            action='store_const', \
                            const=True, \
                            default=False, \
                            required=False, \
                            dest='mini', \
                            help='minimize LJ potential for added substituents')

substitute_parser.add_argument('-o', '--output', \
                            nargs=1, \
                            type=str, \
                            default=False, \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\nDefault: stdout')

args = substitute_parser.parse_args()

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
            substitute_parser.print_help()
            raise TypeError("when no input file is given, stdin is read and a format must be specified")

    geom = Geometry(f)
    comp = Component(geom)
    
    for sub in args.substitutions:
        ndx_target = int(sub.split('=')[0]) - 1
        sub_name = '='.join(sub.split('=')[1:])
    
        target = geom.atoms[ndx_target] 
        
        if args.form[0] == 'from_library':
            sub = Substituent(sub_name)
        elif args.form[0] in ['iupac', 'smiles']:
            sub = Substituent.from_string(sub_name, args.form[0])
        elif args.form[0] in ['element']:
            sub = None
            new_atom = Atom(element = sub_name, coords=target.coords)

        if sub is not None:
            #replace old substituent with new substituent
            comp.substitute(sub, target)
        else:
            #replace old atom with new atom
            comp -= target
            comp += new_atom
            comp.refresh_connected()
            #try to remove H's from new atom to keep the molecule's change the same
            if hasattr(new_atom, '_connectivity') and \
               hasattr(target, '_connectivity'):
                if len(new_atom.connected)-new_atom._connectivity > \
                   len(target.connected)-target._connectivity:
                    for atom in new_atom.connected:
                        if atom.element == 'H':
                            comp -= atom

            comp.refresh_connected()
            comp.detect_backbone()

    if args.mini:
        comp.minimize_sub_torsion()
    
    if args.outfile:
        FileWriter.write_xyz(comp, append=False, outfile=args.outfile[0])
    else:
        s = FileWriter.write_xyz(comp, append=False, outfile=False) 
        print(s)

