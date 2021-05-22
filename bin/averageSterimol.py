#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from AaronTools.comp_output import CompOutput
from AaronTools.const import UNIT
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import boltzmann_coefficients, glob_files

def main(argv):
    sterimol_parser = argparse.ArgumentParser(
        description="calculate Boltzmann-weighted Sterimol parameters - see doi 10.1021/acscatal.8b04043",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    sterimol_parser.add_argument(
        "infiles", metavar="input files",
        type=str,
        nargs="+",
        help="file containing coordinates and energy"
    )
    
    sterimol_parser.add_argument(
        "-if", "--input-format",
        type=str,
        default=None,
        choices=["log", "out", "dat"],
        dest="input_format",
        help="file format of input"
    )
    
    sterimol_parser.add_argument(
        "-s", "--substituent-atom",
        type=str,
        required=True,
        dest="targets",
        help="substituent atom\n" +
        "1-indexed position of the starting position of the\n" +
        "substituent of which you are calculating sterimol\nparameters"
    )
    
    sterimol_parser.add_argument(
        "-a", "--attached-to",
        type=str,
        required=True,
        dest="avoid",
        help="non-substituent atom\n" +
        "1-indexed position of the starting position of the atom\n" +
        "connected to the substituent of which you are calculating\n" +
        "sterimol parameters"
    )
    
    sterimol_parser.add_argument(
        "-r", "--radii",
        type=str,
        default="bondi",
        choices=["bondi", "umn"],
        dest="radii",
        help="VDW radii to use in calculation\n" + 
        "umn: main group vdw radii from J. Phys. Chem. A 2009, 113, 19, 5806–5812\n" +
        "    (DOI: 10.1021/jp8111556)\n" + 
        "    transition metals are crystal radii from Batsanov, S.S. Van der Waals\n" +
        "    Radii of Elements. Inorganic Materials 37, 871–885 (2001).\n" +
        "    (DOI: 10.1023/A:1011625728803)\n" + 
        "bondi: radii from J. Phys. Chem. 1964, 68, 3, 441–451 (DOI: 10.1021/j100785a001)\n" +
        "Default: bondi"
    )
    
    sterimol_parser.add_argument(
        "-l", "--old-l",
        action="store_true",
        required=False,
        dest="old_L",
        help="approximate FORTRAN Sterimol method for determining L\n"
        "This is 0.4 + the ideal bond length for a target-H bond\n"
        "to outer VDW radii of atoms projected onto L-axis\n"
        "Default: L value is from VDW radii of target atom to outer\n"
        "VDW radii of atoms projected onto L-axis"
    )
    
    sterimol_parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=298.15,
        required=False,
        dest="temperature",
        help="temperature in K\nDefault: 298.15"
    )
    
    sterimol_parser.add_argument(
        "-f", "--frequency",
        action="store_true",
        default=False,
        required=False,
        dest="frequency",
        help="input files are frequency job output files\n"
        "additional average values will be calculated for ZPE, H, G, etc."
    )
    
    sterimol_parser.add_argument(
        "-w0",
        "--frequency-cutoff",
        type=float,
        default=100.0,
        required=False,
        dest="w0",
        help="cutoff frequency for quasi free energy corrections (1/cm)\n" +
        "Default: 100 cm^-1"
    )
    
    sterimol_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        required=False,
        dest="verbose",
        help="also print population"
    )
    
    sterimol_parser.add_argument(
        "-o", "--output",
        type=str,
        default=False,
        required=False,
        metavar="output destination",
        dest="outfile",
        help="output destination\n" +
        "Default: stdout"
    )
    
    
    args = sterimol_parser.parse_args(args=argv)
    
    subs = []
    energies = {"E":[]}
    if args.frequency:
        energies["E+ZPE"] = []
        energies["H(RRHO)"] = []
        energies["G(RRHO)"] = []
        energies["G(Quasi-RRHO)"] = []
        energies["G(Quasi-Harmonic)"] = []
    
    for infile in glob_files(args.infiles):
        if args.input_format is not None:
            fr = FileReader((infile, args.input_format, infile), just_geom=False)
        else:
            fr = FileReader(infile, just_geom=False)
    
        geom = Geometry(fr)
        target = args.targets
        avoid = args.avoid
        end = geom.find(avoid)[0]
        frag = geom.get_fragment(target, stop=end)
        sub = Substituent(frag, end=end, detect=False)
        subs.append(sub)
        
        nrg = fr.other["energy"]
        energies["E"].append(nrg)
        if args.frequency:
            co = CompOutput(fr)
            dE, dH, entropy = co.therm_corr(temperature=args.temperature)
            rrho_dG = co.calc_G_corr(v0=0, temperature=args.temperature, method="RRHO")
            qrrho_dG = co.calc_G_corr(v0=args.w0, temperature=args.temperature, method="QRRHO")
            qharm_dG = co.calc_G_corr(v0=args.w0, temperature=args.temperature, method="QHARM")
            energies["E+ZPE"].append(nrg + co.ZPVE)
            energies["H(RRHO)"].append(nrg + dH)
            energies["G(RRHO)"].append(nrg + rrho_dG)
            energies["G(Quasi-RRHO)"].append(nrg + qrrho_dG)
            energies["G(Quasi-Harmonic)"].append(nrg + qharm_dG)
    
    s = ""
    for nrg_type in energies:
        energies_arr = np.array(energies[nrg_type])
        energies_arr *= UNIT.HART_TO_KCAL

        if args.verbose and nrg_type == "E":
            s += "\t".join(["B1", "B2", "B3", "B4", "B5", "L", "file"])
            s += "\n"
            for f, sub in zip(args.infiles, subs):
                data = sub.sterimol(
                    radii=args.radii,
                    old_L=args.old_L,
                )
                s += "\t".join(
                    ["%.2f" % data[x] for x in ["B1", "B2", "B3", "B4", "B5", "L"]]
                )
                s += "\t%s\n" % f
        s += "weighted using %s:\n" % nrg_type
        data = Substituent.weighted_sterimol(
            subs,
            energies_arr,
            args.temperature,
            radii=args.radii,
            old_L=args.old_L,
        )
        if args.verbose:
            coeff = boltzmann_coefficients(energies_arr, args.temperature)
            coeff /= sum(coeff)
            coeff *= 100
            for f, c, e in zip(args.infiles, coeff, energies_arr):
                s += "%s  %.1f%% (%.1f kcal/mol)\n" % (f, c, e - min(energies_arr))
                
        s += "\t".join(["B1", "B2", "B3", "B4", "B5", "L"])
        s += "\n"
        s += "\t".join(["%.2f" % data[x] for x in ["B1", "B2", "B3", "B4", "B5", "L"]])
        s += "\n"
        s += "\n"
    
    if not args.outfile:
        print(s)
    else:
        with open(args.outfile, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main(None)
