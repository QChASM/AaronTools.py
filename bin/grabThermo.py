#!/usr/bin/env python3

import sys
from os.path import split as path_split
import numpy as np
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.comp_output import CompOutput

thermo_parser = argparse.ArgumentParser(description='print thermal corrections and free energy', \
    formatter_class=argparse.RawTextHelpFormatter)
thermo_parser.add_argument('infile', metavar='frequency output file', \
                         type=str, \
                         nargs='*', \
                         default=[sys.stdin], \
                         help='completed QM output file with frequency info')

thermo_parser.add_argument('-o', '--output', \
                        type=str, \
                        default=None, \
                        required=False, \
                        dest='outfile', \
                        help='output destination \nDefault: stdout')

thermo_parser.add_argument('-if', '--input-format', \
                        type=str, \
                        nargs=1, \
                        default=None, \
                        dest='input_format', \
                        choices=['.log'], \
                        help="file format of input - required if input is stdin")

thermo_parser.add_argument('-sp', '--single-point', \
                        type=str, \
                        default=None, \
                        required=False, \
                        dest='sp_file', \
                        help='file containing single-point energy')

thermo_parser.add_argument('-t', '--temperature', \
                            type=float, \
                            default=None, \
                            required=False, \
                            dest='temp', \
                            help='compute thermal corrections using the specified temperature (K)')

thermo_parser.add_argument('-w0' , '--frequency-cutoff', \
                            type=float, \
                            default=100.0, \
                            required=False, \
                            dest='w0', \
                            help='cutoff frequency for quasi free energy corrections (1/cm)\nDefault: 100 cm^-1')

args = thermo_parser.parse_args()

output = ""

if args.sp_file is not None:
    sp_file = FileReader(args.sp_file, just_geom=False)
    sp_energy = sp_file.other['energy']
else:
    sp_energy = None

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(('from stdin', args.input_format[0], f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                thermo_parser.print_help()
                raise RuntimeError("when no input file is given, stdin is read and a format must be specified")

    co = CompOutput(infile)
    
    if sp_energy is None:
        nrg = co.energy
    else:
        nrg = sp_energy

    dE, dH, s = co.therm_corr(temperature=args.temp)
    rrho_dG = co.calc_G_corr(v0=0, temperature=args.temp)
    qrrho_dG = co.calc_G_corr(v0=args.w0, temperature=args.temp, quasi_harmonic=False)
    qharm_dG = co.calc_G_corr(v0=args.w0, temperature=args.temp, quasi_harmonic=True)

    if args.temp is None:
        t = co.temperature
    else:
        t = args.temp

    output += "electronic energy of %s = %.8f Eh\n" % (path_split(args.sp_file)[-1] \
            if args.sp_file is not None else path_split(f)[-1], \
            nrg)
    output += "thermochemistry from %s at %.2f K:\n" % (path_split(f)[-1], t)
    output += "    H(RRHO)           = %.8f Eh  (dH = %.8f)\n" % (nrg + dH, dH)
    output += "    G(RRHO)           = %.8f Eh  (dG = %.8f)\n" % (nrg + rrho_dG, rrho_dG)
    output += "  quasi treatments for entropy (w0=%.1f cm^-1):\n" % args.w0
    output += "    G(Quasi-RRHO)     = %.8f Eh  (dG = %.8f)\n" % (nrg + qrrho_dG, qrrho_dG)
    output += "    G(Quasi-harmonic) = %.8f Eh  (dG = %.8f)\n" % (nrg + qharm_dG, qharm_dG)

    output += '\n'

output = output.strip()

if args.outfile is not None:
    with open(args.outfile, 'w') as f:
        f.write(output)
else:
    print(output) 
