#!/usr/bin/env python3

import sys
import argparse
import os
from glob import glob
from warnings import warn

from numpy import isclose

from AaronTools.comp_output import CompOutput
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.utils.utils import glob_files


thermo_parser = argparse.ArgumentParser(
    description="print gas-phase thermal corrections and free energy",
    formatter_class=argparse.RawTextHelpFormatter
)

thermo_parser.add_argument(
    "infile",
    metavar="frequency output file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="completed QM output file with frequency info"
)

thermo_parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    required=False,
    dest="outfile",
    help="output destination \nDefault: stdout"
)

thermo_parser.add_argument(
    "-if",
    "--input-format",
    type=str,
    nargs=1,
    default=None,
    dest="input_format",
    choices=["log", "out", "dat"],
    help="file format of input - required if input is stdin"
)

thermo_parser.add_argument(
    "-sp",
    "--single-point",
    type=str,
    nargs="*",
    default=[None],
    required=False,
    dest="sp_file",
    help="file containing single-point energy"
)

thermo_parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=None,
    required=False,
    dest="temp",
    help="compute thermal corrections using the specified temperature (K)\n" +
    "Default: value found in file or 298.15"
)

thermo_parser.add_argument(
    "-w0",
    "--frequency-cutoff",
    type=float,
    default=100.0,
    required=False,
    dest="w0",
    help="cutoff frequency for quasi free energy corrections (1/cm)\n" +
    "Default: 100 cm^-1"
)

thermo_parser.add_argument(
    "-csv",
    "--csv-format",
    nargs="?",
    required=False,
    dest="csv",
    default=False,
    choices=["comma", "semicolon", "tab", "space"],
    help="print output in CSV format with the specified delimiter"
)

thermo_parser.add_argument(
    "-r",
    "--recursive",
    metavar="PATTERN",
    type=str,
    nargs="*",
    default=None,
    required=False,
    dest="pattern",
    help="search subdirectories of current directory for files matching PATTERN"
)

thermo_parser.add_argument(
    "-pg",
    "--determine-point-group",
    action="store_true",
    default=False,
    required=False,
    dest="pg",
    help="determine the point group and rotational symmetry number using AaronTools instead of what's in the file"
)


args = thermo_parser.parse_args()

if args.csv is None:
    args.csv = "comma"

if args.csv:
    if args.csv == "comma":
        delim = ","
    elif args.csv == "semicolon":
        delim = ";"
    elif args.csv == "tab":
        delim = "\t"
    elif args.csv == "space":
        delim = " "

    anharm_header = delim.join([
        "E", "E+ZPE", "E+ZPE(anh)", "H(RRHO)", "H(Quasi-RRHO)", "G(RRHO)",
        "G(Quasi-RRHO)", "G(full_Quasi-RRHO)", "G(Quasi-harmonic)","ZPE",
        "ZPE(anh)", "dH(RRHO)", "dG(RRHO)", "dG(Quasi-RRHO)", "dG(Quasi-harmonic)",
        "SP_File", "Thermo_File"
    ])
    harm_header = delim.join([
        "E", "E+ZPE", "H(RRHO)", "H(Quasi-RRHO)", "G(RRHO)", "G(Quasi-RRHO)",
        "G(full_Quasi-RRHO)", "G(Quasi-harmonic)", "ZPE", "dH(RRHO)",
        "dH(Quasi-RRHO)", "dG(RRHO)", "dG(Quasi-RRHO)", "dG(full_Quasi-RRHO)",
        "dG(Quasi-harmonic)", "SP_File", "Thermo_File"
    ])

header = None
output = ""

if args.pattern is None:
    infiles = glob_files(args.infile, parser=thermo_parser)
else:
    infiles = []
    if args.infile == [sys.stdin]:
        directories = [os.getcwd()]
    else:
        directories = []
        for directory in args.infile:
            directories.extend(glob(directory))

    for directory in directories:
        for root, dirs, files in os.walk(directory, topdown=True):
            for pattern in args.pattern:
                full_glob = os.path.join(root, pattern)
                infiles.extend(glob(full_glob))

    infiles.sort()

if args.sp_file != [None]:
    if args.pattern is None:
        sp_filenames = glob_files([f for f in args.sp_file])

    else:
        sp_filenames = []
        if args.infile == [sys.stdin]:
            directories = [os.getcwd()]
        else:
            directories = []
            for directory in args.infile:
                directories.extend(glob(directory))

        for directory in directories:
            for root, dirs, files in os.walk(directory, topdown=True):
                for pattern in args.sp_file:
                    full_glob = os.path.join(root, pattern)
                    sp_filenames.extend(glob(full_glob))

    sp_filenames.sort()

    sp_files = [FileReader(f, just_geom=False) for f in sp_filenames]
    sp_energies = [sp_file.other["energy"] for sp_file in sp_files]

else:
    sp_energies = [None for f in infiles]
    sp_filenames = [None for f in infiles]

while len(sp_energies) < len(infiles):
    sp_energies.extend([sp_file.other["energy"] for sp_file in sp_files])
    sp_filenames.extend(args.sp_file)

while len(infiles) < len(sp_filenames):
    infiles.extend(args.infile)


for sp_nrg, sp_file, f in zip(sp_energies, sp_filenames, infiles):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False)
        else:
            infile = FileReader(f, just_geom=False)
    else:
        if args.input_format is not None:
            infile = FileReader(("from stdin", args.input_format[0], f), just_geom=False)
        else:
            if len(sys.argv) >= 1:
                thermo_parser.print_help()
                raise RuntimeError(
                    "when no input file is given, stdin is read and a format must be specified"
                )

    if "frequency" not in infile.other:
        warn("no frequencies in %s - skipping" % f)
        continue
    freq = infile.other["frequency"]

    co = CompOutput(
        infile,
        determine_pg=args.pg,
    )

    if sp_nrg is None:
        nrg = co.energy
    else:
        nrg = sp_nrg
        sp_geom = Geometry(sp_file)
        freq_geom = Geometry(infile)
        rmsd = sp_geom.RMSD(freq_geom)
        if not isclose(rmsd, 0, atol=1e-5):
            warn(
                "\ngeometries in supposed single-point/thermochemistry pair appear\n" +
                "to be different (rmsd = %.5f)\n" % rmsd +
                "%s\n%s" % (sp_geom.name, freq_geom.name)
            )


    if args.temp is None:
        T = co.temperature
    else:
        T = args.temp

    dE, dH, s = co.therm_corr(temperature=args.temp)
    if freq.anharm_data:
        ZPVE_anh = co.calc_zpe(anharmonic=True)
    rrho_dG = co.calc_G_corr(v0=0, temperature=args.temp, method="RRHO")
    _, qrrho_dH, qrrho_S = co.therm_corr(v0=args.w0, temperature=args.temp, method="QRRHO", enthalpy_method="QRRHO")
    qrrho_dG = dH - T * qrrho_S
    full_qrrho_dG = qrrho_dH - T * qrrho_S
    qharm_dG = co.calc_G_corr(v0=args.w0, temperature=args.temp, method="QHARM")

    if args.csv:
        nrg_str = "%.6f" % nrg
        corrections = [co.ZPVE]
        if freq.anharm_data:
            if header != anharm_header:
                output += "%s\n" % anharm_header
                header = anharm_header
            corrections.append(ZPVE_anh)
        elif header != harm_header:
            output += "%s\n" % harm_header
            header = harm_header
        corrections.extend([dH, qrrho_dH, rrho_dG, qrrho_dG, full_qrrho_dG, qharm_dG])
        therm = [nrg + correction for correction in corrections]
        output += delim.join(
            [nrg_str] +
            ["%.6f" % x for x in therm] + \
            ["%.6f" % x for x in corrections] + \
            [sp_file if sp_file is not None else f, f]
        )
        output += "\n"
    else:
        output += "electronic energy of %s = %.6f Eh\n" % (
            sp_file if sp_file is not None else f, 
            nrg
        )
        output += "    E+ZPE             = %.6f Eh  (ZPE = %.6f)\n" % (nrg + co.ZPVE, co.ZPVE)
        if freq.anharm_data:
            output += "    E+ZPE(anh)        = %.6f Eh  (ZPE(anh) = %.6f)\n" % (
                nrg + ZPVE_anh, ZPVE_anh
            )
        output += "rotational symmetry number: %i\n" % co.rotational_symmetry_number
        output += "thermochemistry from %s at %.2f K:\n" % (f, T)
        output += "    H(RRHO)            = %.6f Eh  (dH = %.6f)\n" % (nrg + dH, dH)
        output += "    H(Quasi-RRHO)      = %.6f Eh  (dH = %.6f)\n" % (nrg + qrrho_dH, qrrho_dH)
        output += "    G(RRHO)            = %.6f Eh  (dG = %.6f)\n" % (nrg + rrho_dG, rrho_dG)
        output += "  quasi treatments for entropy (w0=%.1f cm^-1):\n" % args.w0
        output += "    G(Quasi-RRHO)      = %.6f Eh  (dG = %.6f)\n" % (nrg + qrrho_dG, qrrho_dG)
        output += "    G(full_Quasi-RRHO) = %.6f Eh  (dG = %.6f)\n" % (nrg + full_qrrho_dG, full_qrrho_dG)
        output += "    G(Quasi-harmonic)  = %.6f Eh  (dG = %.6f)\n" % (nrg + qharm_dG, qharm_dG)

        output += "\n"

output = output.strip()

if not args.outfile:
    print(output.strip())
else:
    with open(
            args.outfile,
            "a"
    ) as f:
        f.write(output.strip())
