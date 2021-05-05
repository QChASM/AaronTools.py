#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from AaronTools.comp_output import CompOutput
from AaronTools.const import UNIT
from AaronTools.fileIO import FileReader
from AaronTools.finders import NotAny, AnyTransitionMetal, AnyNonTransitionMetal
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import boltzmann_coefficients, glob_files

def main(argv):
    vbur_parser = argparse.ArgumentParser(
        description="calculate Boltzmann-weighted percent buried volume parameters",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    vbur_parser.add_argument(
        "infiles", metavar="input files",
        type=str,
        nargs="+",
        help="file containing coordinates and energy"
    )
    
    vbur_parser.add_argument(
        "-if", "--input-format",
        type=str,
        default=None,
        choices=["log", "out", "dat"],
        dest="input_format",
        help="file format of input"
    )
    
    vbur_parser.add_argument(
        "-o", "--output",
        type=str,
        default=False,
        required=False,
        metavar="output destination",
        dest="outfile",
        help="output destination\n" +
        "Default: stdout"
    )

    vbur_parser.add_argument(
        "-l", "--ligand-atoms",
        default=None,
        required=False,
        dest="targets",
        help="atoms to consider in calculation\nDefault: use all atoms except the center",
    )
    
    vbur_parser.add_argument(
        "-e", "--exclude-atoms",
        default=None,
        required=False,
        dest="exclude_atoms",
        help="atoms to exclude from the calculation\nDefault: exclude no ligand atoms",
    )
    
    vbur_parser.add_argument(
        "-c", "--center",
        action="append",
        default=None,
        required=False,
        dest="center",
        help="atom the sphere is centered on\n" +
        "Default: detect metal center (centroid of all metals if multiple are present)",
    )

    vbur_parser.add_argument(
        "-r", "--radius",
        default=3.5,
        type=float,
        dest="radius",
        help="radius around center\nDefault: 3.5 Ångström"
    )

    vbur_parser.add_argument(
        "-vdw", "--vdw-radii",
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

    vbur_parser.add_argument(
        "-s", "--scale",
        default=1.17,
        type=float,
        dest="scale",
        help="scale VDW radii by this amount\nDefault: 1.17",
    )
    
    vbur_parser.add_argument(
        "-t", "--temperature",
        type=str,
        default=298.15,
        required=False,
        dest="temperature",
        help="temperature in K\nDefault: 298.15"
    )
    
    vbur_parser.add_argument(
        "-f", "--frequency",
        action="store_true",
        default=False,
        required=False,
        dest="frequency",
        help="input files are frequency job output files\n"
        "additional average values will be calculated for ZPE, H, G, etc."
    )
    
    vbur_parser.add_argument(
        "-w0",
        "--frequency-cutoff",
        type=float,
        default=100.0,
        required=False,
        dest="w0",
        help="cutoff frequency for quasi free energy corrections (1/cm)\n" +
        "Default: 100 cm^-1"
    )
    
    vbur_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        required=False,
        dest="verbose",
        help="also print population"
    )

    vbur_parser.add_argument(
        "-m", "--method",
        default="Lebedev",
        type=lambda x: x.capitalize() if x.lower() == "lebedev" else x.upper(),
        choices=["MC", "Lebedev"],
        dest="method",
        help="integration method - Monte-Carlo (MC) or Lebedev quadrature (Lebedev)\nDefault: Lebedev"
    )
    
    grid_options = vbur_parser.add_argument_group("Lebedev integration options")
    grid_options.add_argument(
        "-rp", "--radial-points",
        type=int,
        default=20,
        choices=[20, 32, 64, 75, 99, 127],
        dest="rpoints",
        help="number of radial shells for Gauss-Legendre integration\n" +
        "of the radial component\n" +
        "lower values are faster, but at the cost of accuracy\nDefault: 20"
    )
    
    grid_options.add_argument(
        "-ap", "--angular-points",
        type=int,
        default=1454,
        choices=[110, 194, 302, 590, 974, 1454, 2030, 2702, 5810],
        dest="apoints",
        help="number of angular points for Lebedev integration\n" +
        "lower values are faster, but at the cost of accuracy\nDefault: 1454"
    )
    
    mc_options = vbur_parser.add_argument_group("Monte-Carlo integration options")
    mc_options.add_argument(
        "-i", "--minimum-iterations",
        type=int,
        default=25,
        metavar="ITERATIONS",
        dest="min_iter",
        help="minimum iterations - each is a batch of 3000 points\n" +
        "MC will continue after this until convergence criteria are met\n" +
        "Default: 25",
    )

    args = vbur_parser.parse_args(args=argv)
    
    targets = None
    if args.exclude_atoms and not args.targets:
        targets = (NotAny(args.exclude_atoms))
    elif args.exclude_atoms and args.targets:
        targets = (NotAny(args.exclude_atoms), args.targets)
    else:
        targets = NotAny(args.center)
    
    geoms = []
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
        geoms.append(geom)
        
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
            s += "\t".join(["%Vbur", "file"])
            s += "\n"
            for f, geom in zip(args.infiles, geoms):
                data = geom.percent_buried_volume(
                    targets=targets,
                    center=args.center,
                    radius=args.radius,
                    radii=args.radii,
                    scale=args.scale,
                    method=args.method,
                    rpoints=args.rpoints,
                    apoints=args.apoints,
                    min_iter=args.min_iter,
                )
                s += "%.1f%%\t%s\n" % (data, f)
            s += "\n"

        s += "weighted using %s:\n" % nrg_type
        data = Geometry.weighted_percent_buried_volume(
            geoms,
            energies_arr,
            args.temperature,
            targets=targets,
            center=args.center,
            radius=args.radius,
            radii=args.radii,
            scale=args.scale,
            method=args.method,
            rpoints=args.rpoints,
            apoints=args.apoints,
            min_iter=args.min_iter,
        )

        if args.verbose:
            coeff = boltzmann_coefficients(energies_arr, args.temperature)
            coeff /= sum(coeff)
            coeff *= 100
            for f, c, e in zip(args.infiles, coeff, energies_arr):
                s += "%s  %.1f%% (%.1f kcal/mol)\n" % (f, c, e - min(energies_arr))

        s += "%%Vbur: %.2f\n\n" % data
    
    if not args.outfile:
        print(s)
    else:
        with open(args.outfile, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main(None)
