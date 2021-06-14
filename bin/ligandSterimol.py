#!/usr/bin/env python3

import argparse
import sys

from AaronTools.fileIO import FileReader, read_types
from AaronTools.geometry import Geometry
from AaronTools.component import Component
from AaronTools.substituent import Substituent
from AaronTools.utils.utils import glob_files

def main(argv):
    sterimol_parser = argparse.ArgumentParser(
        description="calculate B1-B5, and L sterimol parameters for ligands - see Verloop, A. and Tipker, J. (1976), Use of linear free energy related and other parameters in the study of fungicidal selectivity. Pestic. Sci., 7: 379-390.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    sterimol_parser.add_argument(
        "infile", metavar="input file",
        type=str,
        nargs="*",
        default=[sys.stdin],
        help="a coordinate file"
    )
    
    sterimol_parser.add_argument(
        "-if", "--input-format",
        type=str,
        default=None,
        choices=read_types,
        dest="input_format",
        help="file format of input\nxyz is assumed if input is stdin"
    )
    
    sterimol_parser.add_argument(
        "-k", "--key-atoms",
        type=str,
        required=True,
        dest="key",
        help="1-indexed position of the ligand's coordinating atoms"
    )
    
    sterimol_parser.add_argument(
        "-c", "--center-atom",
        type=str,
        required=True,
        dest="center",
        help="atom the ligand is coordinated to"
    )

    sterimol_parser.add_argument(
        "-r", "--radii",
        type=str,
        default="bondi",
        choices=["bondi", "umn"],
        dest="radii",
        help="VDW radii to use in calculation\n"
        "umn: main group vdw radii from J. Phys. Chem. A 2009, 113, 19, 5806–5812\n"
        "    (DOI: 10.1021/jp8111556)\n" 
        "    transition metals are crystal radii from Batsanov, S.S. Van der Waals\n"
        "    Radii of Elements. Inorganic Materials 37, 871–885 (2001).\n"
        "    (DOI: 10.1023/A:1011625728803)\n" 
        "bondi: radii from J. Phys. Chem. 1964, 68, 3, 441–451\n(DOI: 10.1021/j100785a001)\n"
        "Default: bondi"
    )

    sterimol_parser.add_argument(
        "-bl", "--bisect-L",
        action="store_true",
        required=False,
        dest="bisect_L",
        help="L axis will bisect (or analogous for higher denticity\n"
        "ligands) the L-M-L angle\n"
        "Default: center to centroid of key atoms"
    )

    sterimol_parser.add_argument(
        "-al", "--at-L",
        default=[None],
        dest="L_value",
        type=lambda x: [float(v) for v in x.split(",")],
        help="get widths at specific L values (comma-separated)\n"
        "Default: use the entire ligand",
    )

    sterimol_parser.add_argument(
        "-v", "--vector",
        action="store_true",
        required=False,
        dest="vector",
        help="print Chimera/ChimeraX bild file for vectors instead of parameter values"
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

    s = ""
    if not args.vector:
        s += "B1\tB2\tB3\tB4\tB5\tL\tfile\n"
    
    for infile in glob_files(args.infile):
        if isinstance(infile, str):
            if args.input_format is not None:
                f = FileReader((infile, args.input_format, infile))
            else:
                f = FileReader(infile)
        else:
            if args.input_format is not None:
                f = FileReader(("from stdin", args.input_format, infile))
            else:
                f = FileReader(("from stdin", "xyz", infile))

        geom = Geometry(f, refresh_ranks=False)
        comp = Component(
            geom.get_fragment(args.key, stop=args.center),
            to_center=geom.find(args.center),
            key_atoms=args.key,
            detect_backbone=False,
        )
        for val in args.L_value:
            data = comp.sterimol(
                to_center=geom.find(args.center),
                return_vector=args.vector,
                radii=args.radii,
                bisect_L=args.bisect_L,
                at_L=val,
            )
    
            if args.vector:
                for key, color in zip(
                        ["B1", "B2", "B3", "B4", "B5", "L"],
                        ["black", "green", "purple", "orange", "red", "blue"]
                ):
                    start, end = data[key]
                    s += ".color %s\n" % color
                    s += ".note Sterimol %s\n" % key
                    s += ".arrow %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f\n" % (*start, *end)
            else:
                s += "%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n" % (
                    data["B1"],
                    data["B2"],
                    data["B3"],
                    data["B4"],
                    data["B5"],
                    data["L"],
                    infile,
                )
    
    if not args.outfile:
        print(s)
    else:
        with open(args.outfile, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main(None)
