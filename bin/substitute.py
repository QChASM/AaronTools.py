#!/usr/bin/env python3

import argparse
from sys import stdin, argv, exit
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent

substitute_parser = argparse.ArgumentParser(description='replace an atom or substituent with another', \
    formatter_class=argparse.RawTextHelpFormatter)
substitute_parser.add_argument('infile', metavar='input file', \
                            type=str, \
                            nargs='*', \
                            default=[stdin], \
                            help='a coordinate file')

substitute_parser.add_argument('-ls', '--list', \
                                action='store_const', \
                                const=True, \
                                default=False, \
                                required=False, \
                                dest='list_avail', \
                                help='list available substituents')

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

substitute_parser.add_argument('-o', '--output', \
                            nargs=1, \
                            type=str, \
                            default=[False], \
                            required=False, \
                            metavar='output destination', \
                            dest='outfile', \
                            help='output destination\nDefault: stdout')

args = substitute_parser.parse_args()

if args.list_avail:
    s = ""
    for i, name in enumerate(sorted(Substituent.list())):
        s += "%-20s" % name
        #if (i + 1) % 3 == 0:
        if (i + 1) % 1 == 0:
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
            substitute_parser.print_help()
            raise TypeError("when no input file is given, stdin is read and a format must be specified")

    geom = Geometry(f)
   
    target_list = []
    for sub in args.substitutions:
        ndx_targets = sub.split('=')[0]
        target_list.append(geom.find(ndx_targets))

    for i, sub in enumerate(args.substitutions):
        ndx_target = target_list[i]
        sub_name = '='.join(sub.split('=')[1:])
   
        for target in ndx_target:
            sub = Substituent(sub_name)

            #replace old substituent with new substituent
            geom.substitute(sub, target)

            geom.refresh_connected()

    s = geom.write(append=False, outfile=args.outfile[0]) 
    if not args.outfile[0]:
        print(s)

