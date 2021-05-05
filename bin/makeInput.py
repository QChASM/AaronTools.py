#!/usr/bin/env python3

import sys
from os.path import splitext
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.theory import *
from AaronTools.utils.utils import combine_dicts, get_filename, glob_files

theory_parser = argparse.ArgumentParser(
    description="print Gaussian, ORCA, Psi4, or SQM input file",
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
    "-of", "--output-format",
    type=str,
    default=None,
    dest="out_format",
    choices=["gaussian", "orca", "psi4", "sqm"],
    help="file format of output",
)

theory_parser.add_argument(
    "-c", "--comment",
    action="append",
    nargs="+",
    default=[],
    dest="comments",
    help="comment to put in the output file\ninput file(s) should not be right after comments",
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

theory_parser.add_argument(
    "-mem", "--memory",
    type=int,
    dest="memory",
    default=None,
    required=False,
    help="total memory in GB\n" +
    "Note: ORCA and Gaussian only use this to limit the storage-intensive\n" +
    "      portions of the calculation (e.g. integrals, wavefunction info)",
)

theory_parser.add_argument(
    "-up", "--use-previous",
    action="store_true",
    default=False,
    required=False,
    dest="use_prev",
    help="use settings that can be parsed from the input file",
)

theory_options = theory_parser.add_argument_group("Theory options")
theory_options.add_argument(
    "-m", "--method",
    type=str,
    dest="method",
    required=False,
    help="method (e.g. B3LYP or MP2)",
)

theory_options.add_argument(
    "-b", "--basis",
    nargs="+",
    type=str,
    action="append",
    required=False,
    default=None,
    dest="basis",
    help="basis set or list of elements and basis set (e.g. C O N aug-cc-pvtz)\n" +
    "elements can be prefixed with ! to exclude them from the basis\n" +
    "tm is a synonym for d-block elements\n" +
    "auxilliary basis sets can be specified by putting aux X before the basis\n" +
    "set name, where X is the auxilliary type (e.g. aux JK cc-pVDZ for cc-pVDZ/JK)\n" +
    "a path to a file containing a basis set definition (like one\n" +
    "downloaded from basissetexchange.org) can be placed after the\n" +
    "basis set name\n" +
    "the file's contents should be appropriate for the software package you are using"
)

theory_options.add_argument(
    "-ecp", "--pseudopotential",
    nargs="+",
    type=str,
    action="append",
    required=False,
    default=None,
    dest="ecp",
    help="ECP or list of elements and ECP (e.g. Pt LANL2DZ)\n" +
    "elements can be prefixed with ! to exclude them from the ECP\n" +
    "tm is a synonym for d-block elements\n" +
    "a path to a file containing a basis set definition (like one\n" +
    "downloaded from basissetexchange.org) can be placed after the\n" +
    "basis set name\n" +
    "the file's contents should be appropriate for the software package you are using"
)

theory_options.add_argument(
    "-ed", "--empirical-dispersion",
    required=False,
    default=None,
    dest="empirical_dispersion",
    help="empirical dispersion keyword",
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

theory_options.add_argument(
    "-g", "--grid",
    required=False,
    default=None,
    dest="grid",
    help="integration grid",
)

job_options = theory_parser.add_argument_group("Job options")
job_options.add_argument(
    "-opt", "--optimize",
    action="store_true",
    required=False,
    dest="optimize",
    help="request geometry optimization job",
)

job_options.add_argument(
    "-freq", "--frequencies",
    action="store_true",
    required=False,
    default=False,
    dest="freq",
    help="request vibrational frequencies job",
)

job_options.add_argument(
    "-e", "--energy",
    action="store_true",
    required=False,
    default=False,
    dest="energy",
    help="request single point energy job",
)

opt_type = theory_parser.add_argument_group("Optimization options")
opt_type.add_argument(
    "-ts", "--transition-state",
    action="store_true",
    default=False,
    dest="ts",
    help="request transition state optimization",
)

opt_type.add_argument(
    "-ca", "--constrained-atoms",
    nargs=1,
    type=str,
    action="append",
    default=None,
    dest="atoms",
    help="comma- or hyphen-separated list of atoms (1-indexed) to constrain during optimization",
)

opt_type.add_argument(
    "-cb", "--constrain-bond",
    nargs=1,
    action="append",
    default=None,
    dest="bonds",
    help="list of comma-separated atom pairs\n" +
    "the distance between the atoms in each pair will be constrained during optimization",
)

opt_type.add_argument(
    "-cang", "--constrain-angle",
    type=str,
    nargs=1,
    action="append",
    default=None,
    dest="angles",
    help="list of comma-separated atom trios\n" +
    "the angle defined by each trio will be constrained during optimization",
)

opt_type.add_argument(
    "-ct", "--constrain-torsion",
    type=str,
    nargs=1,
    action="append",
    default=None,
    dest="torsions",
    help="list of comma-separated atom quartets\n" +
    "the torsional angle defined by each quartet will be constrained during optimization",
)

opt_type.add_argument(
    "-cx", "--constrained-x",
    nargs=1,
    type=str,
    action="append",
    default=None,
    dest="x",
    help="comma- or hyphen-separated list of atoms (1-indexed) to constrain the x coordinate of\n" +
    "available for Gaussian and Psi4",
)

opt_type.add_argument(
    "-cy", "--constrained-y",
    nargs=1,
    type=str,
    action="append",
    default=None,
    dest="y",
    help="comma- or hyphen-separated list of atoms (1-indexed) to constrain the y coordinate of\n" +
    "available for Gaussian and Psi4",
)

opt_type.add_argument(
    "-cz", "--constrained-z",
    nargs=1,
    type=str,
    action="append",
    default=None,
    dest="z",
    help="comma- or hyphen-separated list of atoms (1-indexed) to constrain the z coordinate of\n" +
    "available for Gaussian and Psi4",
)

opt_type.add_argument(
    "-gx", "--grouped-x",
    nargs=2,
    type=str,
    action="append",
    default=None,
    dest="xgroup",
    metavar=("atoms", "value"),
    help="comma- or hyphen-separated list of atoms (1-indexed) to keep in the same yz plane\n" +
    "available for Gaussian and Psi4",
)

opt_type.add_argument(
    "-gy", "--grouped-y",
    nargs=2,
    type=str,
    action="append",
    default=None,
    dest="ygroup",
    metavar=("atoms", "value"),
    help="comma- or hyphen-separated list of atoms (1-indexed) to keep in the same xz plane\n" +
    "available for Gaussian and Psi4",
)

opt_type.add_argument(
    "-gz", "--grouped-z",
    nargs=2,
    type=str,
    action="append",
    default=None,
    dest="zgroup",
    metavar=("atoms", "value"),
    help="comma- or hyphen-separated list of atoms (1-indexed) to keep in the same xy plane\n" +
    "available for Gaussian and Psi4",
)

freq_type = theory_parser.add_argument_group("Frequency options")
freq_type.add_argument(
    "-n", "--numerical",
    action="store_true",
    default=False,
    dest="numerical",
    help="request numerical frequencies",
)

freq_type.add_argument(
    "-t", "--temperature",
    type=float,
    default=298.15,
    dest="temperature",
    help="temperature for calculated thermochemical corrections\nDefault: 298.15",
)

orca_options = theory_parser.add_argument_group("ORCA-specific options")
orca_options.add_argument(
    "--simple",
    action="append",
    default=[],
    dest=ORCA_ROUTE,
    help="keywords for simple input",
)

orca_options.add_argument(
    "--block",
    nargs=3,
    action="append",
    default=[],
    dest=ORCA_BLOCKS,
    metavar=("BLOCK", "OPTION", "VALUE"),
    help="blocks and block options\nexample: --block scf maxiter 500",
)

psi4_options = theory_parser.add_argument_group("Psi4-specific options")
psi4_options.add_argument(
    "--before-molecule",
    action="append",
    default=[],
    dest=PSI4_BEFORE_GEOM,
    metavar="BEFORE MOL",
    help="line to add before the molecule specification",
)

psi4_options.add_argument(
    "--before-job",
    action="append",
    nargs="+",
    default=[],
    dest=PSI4_BEFORE_JOB,
    metavar="BEFORE JOB",
    help="line to add before the job\ninput file(s) should not be right after --before-job",
)

psi4_options.add_argument(
    "--after-job",
    action="append",
    nargs="+",
    default=[],
    dest=PSI4_AFTER_JOB,
    metavar="AFTER JOB",
    help="line to add after the job\ninput file(s) should not be right after --after-job",
)


psi4_options.add_argument(
    "--job",
    action="append",
    nargs="+",
    default=[],
    dest=PSI4_JOB,
    metavar="JOB",
    help="other jobs to add\nexample: --job hessian\ninput file(s) should not be right after --job",
)

psi4_options.add_argument(
    "--setting",
    action="append",
    nargs=2,
    default=[],
    dest=PSI4_SETTINGS,
    metavar=("SETTING", "VALUE"),
    help="settings\nexample: --setting reference uhf",
)

psi4_options.add_argument(
    "--optking",
    action="append",
    nargs=2,
    default=[],
    dest=PSI4_OPTKING,
    metavar=("SETTING", "VALUE"),
    help="optking settings",
)

psi4_options.add_argument(
    "--molecule",
    action="append",
    nargs="+",
    default=[],
    dest=PSI4_MOLECULE,
    metavar=("SETTING", "VALUE"),
    help="options to add to the molecule section\n" +
    "example: --molecule units bohr\ninput file(s) should not be right after --molecule",
)

gaussian_options = theory_parser.add_argument_group("Gaussian-specific options")
gaussian_options.add_argument(
    "--route",
    action="append",
    nargs="+",
    default=[],
    dest=GAUSSIAN_ROUTE,
    metavar=("KEYWORD", "OPTION"),
    help="route options\nexample: --route freq hpmodes\n" +
    "input file(s) should not be right after --route",
)

gaussian_options.add_argument(
    "--link0",
    action="append",
    nargs="+",
    default=[],
    dest=GAUSSIAN_PRE_ROUTE,
    metavar=("COMMAND", "VALUE"),
    help="Link 0 commands (without %%)\n" +
    "example: --link0 chk asdf.chk\ninput file(s) should not be right after --link0",
)

gaussian_options.add_argument(
    "--end-of-file",
    action="append",
    default=[],
    dest=GAUSSIAN_POST,
    metavar="input",
    help="line to add to the end of the file (e.g. for NBORead)",
)

args = theory_parser.parse_args()

if not args.method and not args.use_prev:
    sys.stderr.write("no method specified; -m/--method or -u/--use-previous is required")
    theory_parser.print_help()
    sys.exit(1)

kwargs = {}

blocks = getattr(args, ORCA_BLOCKS)
if blocks:
    kwargs[ORCA_BLOCKS] = {}
    for block in blocks:
        block_name = block[0]
        if block_name not in kwargs[ORCA_BLOCKS]:
            kwargs[ORCA_BLOCKS][block_name] = []
        kwargs[ORCA_BLOCKS][block_name].append("\t".join(block[1:]))

for pos in [
        PSI4_SETTINGS, PSI4_MOLECULE, PSI4_JOB, PSI4_OPTKING,
        GAUSSIAN_ROUTE, GAUSSIAN_PRE_ROUTE
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


for pos in [ORCA_ROUTE]:
    opt = getattr(args, pos)
    if opt:
        if pos not in kwargs:
            kwargs[pos] = []

        kwargs[pos].extend(opt)

for pos in [PSI4_BEFORE_GEOM, PSI4_AFTER_JOB, PSI4_BEFORE_JOB, GAUSSIAN_POST]:
    opt = getattr(args, pos)
    if opt:
        if pos not in kwargs:
            kwargs[pos] = []

        kwargs[pos].extend([" ".join(word) for word in opt])

if args.comments:
    kwargs["comments"] = [" ".join(comment) for comment in args.comments]


# Theory() is made for each file because we might be using things from the input file
for f in glob_files(args.infile):
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

    geom = Geometry(infile)


    if args.method is None and args.use_prev:
        if "method" in infile.other:
            method = infile.other["method"].split("/")[0]
        elif "theory" in infile.other:
            method = infile.other["theory"].method
    elif args.method is not None:
        method = args.method
    else:
        raise RuntimeError(
            "method was not determined from %s and was not specified with --method" % f
        )


    if args.basis is not None:
        basis_sets = []
        for basis in args.basis:
            basis_sets.append(
                BasisSet.parse_basis_str(" ".join(basis))[0]
            )

    elif args.use_prev:
        if "method" in infile.other:
            basis_sets = method.split("/")[-1]
        elif "theory" in infile.other:
            basis_sets = infile.other["theory"].basis.basis
        else:
            basis_sets = None

    else:
        basis_sets = None


    if args.ecp is not None:
        ecps = []
        for ecp in args.ecp:
            ecps.append(
                BasisSet.parse_basis_str(" ".join(ecp), cls=ECP)[0]
            )

    elif args.use_prev:
        if "theory" in infile.other:
            ecps = infile.other["theory"].basis.ecp
        else:
            ecps = None

    else:
        ecps = None


    if ecps is None and basis_sets is None:
        basis_set = None
    else:
        basis_set = BasisSet(basis_sets, ecps)


    if args.solvent is not None or args.solvent_model is not None:
        if args.solvent_model is None or args.solvent is None:
            raise RuntimeError("--solvent and --solvent-model must both be specified")

        solvent = ImplicitSolvent(args.solvent_model, args.solvent)

    else:
        solvent = None


    job_types = []
    if not args.use_prev or (args.optimize or args.freq or args.energy):
        if args.optimize:
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

            if args.x is not None:
                constraints["x"] = []
                for constraint in args.x:
                    constraints["x"].extend(geom.find(constraint))

            if args.y is not None:
                constraints["y"] = []
                for constraint in args.y:
                    constraints["y"].extend(geom.find(constraint))

            if args.z is not None:
                constraints["z"] = []
                for constraint in args.z:
                    constraints["z"].extend(geom.find(constraint))

            if args.xgroup is not None:
                constraints["xgroup"] = []
                for constraint, val in args.xgroup:
                    constraints["xgroup"].append((geom.find(constraint), float(val)))

            if args.ygroup is not None:
                constraints["ygroup"] = []
                for constraint, val in args.ygroup:
                    constraints["ygroup"].append((geom.find(constraint), float(val)))

            if args.zgroup is not None:
                constraints["zgroup"] = []
                for constraint, val in args.zgroup:
                    constraints["zgroup"].append((geom.find(constraint), float(val)))

            if not constraints.keys():
                constraints = None

            job_types.append(OptimizationJob(transition_state=args.ts, constraints=constraints))


        if args.freq:
            job_types.append(FrequencyJob(numerical=args.numerical, temperature=args.temperature))

        if args.energy:
            job_types.append(SinglePointJob())

    elif args.use_prev and "theory" in infile.other:
        job_types = infile.other["theory"].job_type

    grid = args.grid
    if args.use_prev and "theory" in infile.other and not grid:
        grid = infile.other["theory"].grid

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
        method=method,
        basis=basis_set,
        grid=grid,
        solvent=solvent,
        job_type=job_types,
        empirical_dispersion=args.empirical_dispersion,
        charge=charge,
        multiplicity=multiplicity,
        processors=args.processors,
        memory=args.memory,
    )


    if args.out_format:
        style = args.out_format
    else:
        if args.outfile:
            style = splitext(args.outfile)[-1].lstrip(".")
        else:
            raise RuntimeError("file format must be specified if no output file is specified")

    if args.use_prev and "other_kwargs" in infile.other:
        other_kwargs = combine_dicts(other_kwargs, infile.other["other_kwargs"])

    other_kwargs = combine_dicts(kwargs, other_kwargs)

    if args.outfile:
        outfile = args.outfile
        if "$INFILE" in outfile:
            outfile = outfile.replace("$INFILE", get_filename(f))
        warnings = geom.write(
            append=True,
            outfile=outfile,
            style=style,
            theory=theory,
            return_warnings=True,
            **other_kwargs
        )
    else:
        out, warnings = geom.write(
            append=True,
            outfile=False,
            style=style,
            theory=theory,
            return_warnings=True,
            **other_kwargs
        )
        print(out)
    
    for warning in warnings:
        Geometry.LOG.warning(warning)
