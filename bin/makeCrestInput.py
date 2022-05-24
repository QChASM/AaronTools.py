#!/usr/bin/env python3

import sys
from os.path import splitext
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.theory import *
from AaronTools.utils.utils import combine_dicts, get_filename, glob_files

theory_parser = argparse.ArgumentParser(
    description="print CREST input files",
    formatter_class=argparse.RawTextHelpFormatter
)

theory_parser.add_argument(
    "infile", metavar="input file",
    type=str,
    nargs="*",
    default=[sys.stdin],
    help="a coordinate file",
)

theory_parser.add_argument(
    "-o", "--output",
    type=str,
    default=False,
    required=False,
    dest="outfile",
    help="output destination\n" +
    "$INFILE will be replaced with the name of the input file\n" +
    "Default: stdout"
)

theory_parser.add_argument(
    "-if", "--input-format",
    type=str,
    default=None,
    dest="input_format",
    choices=read_types,
    help="file format of input - xyz is assumed if input is stdin",
)

theory_parser.add_argument(
    "-q", "--charge",
    type=int,
    dest="charge",
    default=None,
    help="net charge\nDefault: 0 or what is found in the input file",
)

theory_parser.add_argument(
    "-mult", "--multiplicity",
    type=int,
    dest="multiplicity",
    default=None,
    help="electronic multiplicity\nDefault: 1 or what is found in the input file",
)

theory_parser.add_argument(
    "-p", "--cores",
    type=int,
    dest="processors",
    default=None,
    required=False,
    help="number of cpu cores to use",
)

theory_options = theory_parser.add_argument_group("Theory options")
theory_options.add_argument(
    "-m", "--method",
    type=str,
    dest="method",
    required=False,
    default="GFN2-xTB",
    help="method (e.g. GFN-FF)\nDefault: GFN2-xTB",
)

theory_options.add_argument(
    "-sv", "--solvent",
    required=False,
    default=None,
    dest="solvent",
    help="solvent",
)

theory_options.add_argument(
    "-sm", "--solvent-model",
    required=False,
    default=None,
    dest="solvent_model",
    help="implicit solvent model",
)


search_options = theory_parser.add_argument_group("Search options")
search_options.add_argument(
    "-ca", "--constrained-atoms",
    nargs=1,
    type=str,
    action="append",
    default=None,
    dest="atoms",
    help="comma- or hyphen-separated list of atoms (1-indexed) to constrain during optimization",
)

search_options.add_argument(
    "-cb", "--constrain-bond",
    nargs=1,
    action="append",
    default=None,
    dest="bonds",
    help="list of comma-separated atom pairs\n" +
    "the distance between the atoms in each pair will be constrained during optimization",
)

search_options.add_argument(
    "-cang", "--constrain-angle",
    type=str,
    nargs=1,
    action="append",
    default=None,
    dest="angles",
    help="list of comma-separated atom trios\n" +
    "the angle defined by each trio will be constrained during optimization",
)

search_options.add_argument(
    "-ct", "--constrain-torsion",
    type=str,
    nargs=1,
    action="append",
    default=None,
    dest="torsions",
    help="list of comma-separated atom quartets\n" +
    "the torsional angle defined by each quartet will be constrained during optimization",
)


gaussian_options = theory_parser.add_argument_group("CREST options")
gaussian_options.add_argument(
    "-xc", "--xcontrol",
    action="append",
    nargs="+",
    default=[],
    dest=XTB_CONTROL_BLOCKS,
    metavar=("KEYWORD", "OPTION"),
    help="xcontrol options\nexample: --metadyn coord original.xyz\n" +
    "input file(s) should not be right after --xcontrol",
)

gaussian_options.add_argument(
    "-cmd", "--command",
    action="append",
    nargs="+",
    default=[],
    dest=XTB_COMMAND_LINE,
    metavar=("COMMAND", "VALUE"),
    help="command line options (without --)\n" +
    "example: --command tautomerize\ninput file(s) should not be right after --link0",
)


args = theory_parser.parse_args()

kwargs = {}

for pos in [
        XTB_COMMAND_LINE, XTB_CONTROL_BLOCKS
    ]:
    opts = getattr(args, pos)
    if opts:
        if pos not in kwargs:
            kwargs[pos] = {}

        for opt in opts:
            setting = opt.pop(0)
            if setting not in kwargs[pos]:
                kwargs[pos][setting] = []

            kwargs[pos][setting].extend(opt)

# Theory() is made for each file because we might be using things from the input file
for f in glob_files(args.infile, parser=theory_parser):
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False, get_all=True)
        else:
            infile = FileReader(f, just_geom=False, get_all=True)
    else:
        if args.input_format is not None:
            infile = FileReader(
                ("from stdin", args.input_format[0], f), just_geom=False, get_all=True
            )
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(("from stdin", "xyz", f), just_geom=False, get_all=True)

    geom = Geometry(infile, refresh_ranks=False)


    if args.solvent is not None or args.solvent_model is not None:
        if args.solvent_model is None or args.solvent is None:
            raise RuntimeError("--solvent and --solvent-model must both be specified")

        solvent = ImplicitSolvent(args.solvent_model, args.solvent)

    else:
        solvent = None


        constraints = {}
        if args.atoms is not None:
            constraints["atoms"] = []
            for constraint in args.atoms:
                constraints["atoms"].extend(geom.find(constraint))

        if args.bonds is not None:
            constraints["bonds"] = []
            for bond in args.bonds:
                bonded_atoms = geom.find(bond)
                if len(bonded_atoms) != 2:
                    raise RuntimeError(
                        "not exactly 2 atoms specified in a bond constraint\n" +
                        "use the format --constrain-bond 1,2"
                    )
                constraints["bonds"].append(bonded_atoms)

        if args.angles is not None:
            constraints["angles"] = []
            for angle in args.angle:
                angle_atoms = geom.find(angle)
                if len(angle_atoms) != 3:
                    raise RuntimeError(
                        "not exactly 3 atoms specified in a angle constraint\n" +
                        "use the format --constrain-angle 1,2,3"
                    )
                constraints["angles"].append(angle_atoms)

        if args.torsions is not None:
            constraints["torsions"] = []
            for torsion in args.torsions:
                torsion_atoms = geom.find(torsion)
                if len(torsion_atoms) != 4:
                    raise RuntimeError(
                        "not exactly 4 atoms specified in a torsion constraint\n" +
                        "use the format --constrain-torsion 1,2,3,4"
                    )
                constraints["torsions"].append(torsion_atoms)

        if not constraints.keys():
            constraints = None

        job_type = ConformerSearchJob(constraints=constraints)

    if args.charge is None:
        if "charge" in infile.other:
            charge = infile.other["charge"]
        else:
            charge = 0
    else:
        charge = args.charge


    if args.multiplicity is None:
        if "multiplicity" in infile.other:
            multiplicity = infile.other["multiplicity"]
        else:
            multiplicity = 1
    else:
        multiplicity = args.multiplicity

    other_kwargs = {}

    theory = Theory(
        geometry=geom,
        method=args.method,
        solvent=solvent,
        job_type=job_type,
        charge=charge,
        multiplicity=multiplicity,
        processors=args.processors,
        **kwargs,
    )

    if args.outfile:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        warnings = geom.write(
            append=True,
            outfile=outfile,
            style="crest",
            theory=theory,
            return_warnings=True,
            **other_kwargs
        )
    else:
        out, warnings = geom.write(
            append=True,
            outfile=False,
            style="crest",
            theory=theory,
            return_warnings=True,
            **other_kwargs
        )
        for key, item in out.items():
            print("<<--%s-->>" % key)
            print(item)
    
    if warnings:
        for warning in warnings:
            Geometry.LOG.warning(warning)
