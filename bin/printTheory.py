#!/usr/bin/env python3

import sys
from os.path import splitext
import argparse

from AaronTools.geometry import Geometry
from AaronTools.fileIO import FileReader, read_types
from AaronTools.const import ELEMENTS
from AaronTools.theory import *

theory_parser = argparse.ArgumentParser(description='print structure in xyz format', \
    formatter_class=argparse.RawTextHelpFormatter)
theory_parser.add_argument('infile', metavar='input file', \
                           type=str, \
                           nargs='*', \
                           default=[sys.stdin], \
                           help='a coordinate file')

theory_parser.add_argument('-o', '--output', \
                           type=str, \
                           default=False, \
                           required=False, \
                           dest='outfile', \
                           help='output destination \nDefault: stdout')

theory_parser.add_argument('-if', '--input-format', \
                           type=str, \
                           default=None, \
                           dest='input_format', \
                           choices=read_types, \
                           help="file format of input - xyz is assumed if input is stdin")

theory_parser.add_argument('-of', '--output-format', \
                           type=str, \
                           default=None, \
                           dest='out_format', \
                           choices=['gaussian','orca','psi4'], \
                           help="file format of output")

theory_parser.add_argument('-q', '--charge', \
                           type=int, \
                           dest='charge', \
                           default=None, \
                           help='net charge\nDefault: 0 or what is found in the input file')

theory_parser.add_argument('-mult', '--multiplicity', \
                           type=int, \
                           dest='multiplicity', \
                           default=None, \
                           help='electronic multiplicity\nDefault: 1 or what is found in the input file')

theory_parser.add_argument('-p', '--cores', \
                           type=int, \
                           dest='processors', \
                           default=None, \
                           required=False, \
                           help='number of cpu cores to use')

theory_parser.add_argument('-mem', '--memory', \
                           type=int, \
                           dest='memory', \
                           default=None, \
                           required=False, \
                           help='total memory in GB' + \
                                'Note: ORCA and Gaussian only use this to limit the storage-intensive\n' + \
                                '      portions of the calculation (e.g. integrals, wavefunction info)')

theory_options = theory_parser.add_argument_group('Theory options')
theory_options.add_argument('-m', '--method', \
                            type=str, \
                            dest='method', \
                            required=False, \
                            help='method (e.g. B3LYP or MP2)')
 
theory_options.add_argument('-b', '--basis', \
                            nargs='*', \
                            type=str, \
                            action='append', \
                            required=False, \
                            default=None, \
                            dest='basis', \
                            help='basis set or list of elements and basis set (e.g. C O N aug-cc-pvtz)')

theory_options.add_argument('-ecp', '--pseudopotential', \
                            nargs='*', \
                            type=str, \
                            action='append', \
                            required=False, \
                            default=None, \
                            dest='ecp', \
                            help='ECP or list of elements and ECP (e.g. Pt LANL2DZ)')

theory_options.add_argument('-ed', '--empirical-dispersion', \
                            required=False, \
                            default=None, \
                            dest='empirical_dispersion', \
                            help='empirical dispersion keyword')

theory_options.add_argument('-s', '--solvent', \
                            required=False, \
                            default=None, \
                            dest='solvent', \
                            help='solvent')

theory_options.add_argument('-sm', '--solvent-model', \
                            required=False, \
                            default=None, \
                            dest='solvent_model', \
                            help='implicit solvent model')

theory_options.add_argument('-g', '--grid', \
                            required=False, \
                            default=None, \
                            dest='grid', \
                            help='integration grid')

job_options = theory_parser.add_argument_group('Job options')
job_options.add_argument('-opt', '--optimize', \
                         action='store_true', \
                         required=False, \
                         dest='optimize', \
                         help='request geometry optimization job')

job_options.add_argument('-freq', '--frequencies', \
                         action='store_true', \
                         required=False, \
                         default=False, \
                         dest='freq', \
                         help='request vibrational frequencies job')

job_options.add_argument('-e', '--energy', \
                         action='store_true', \
                         required=False, \
                         default=False, \
                         dest='energy', \
                         help='request single point energy job')

opt_type = theory_parser.add_argument_group("Optimization options")
opt_type.add_argument('-ts', '--transition-state', \
                      action='store_true', \
                      default=False, \
                      dest='ts', \
                      help='request transition state optimization')

opt_type.add_argument('-ca', '--constrained-atoms', \
                      type=str, \
                      default=None, \
                      dest='atoms', \
                      help='comma- or hyphen-separated list of atoms (1-indexed) to constrain during optimization')

opt_type.add_argument('-cb', '--constrained-bonds', \
                      type=str, \
                      action='append', \
                      default=None, \
                      dest='bonds', \
                      help='list of comma-separated atom pairs\n' + \
                           'the distance between the atoms in each pair will be constrained during optimization')

opt_type.add_argument('-cang', '--constrained-angles', \
                      type=str, \
                      action='append', \
                      default=None, \
                      dest='angles', \
                      help='list of comma-separated atom trios\n' + \
                           'the angle defined by each trio will be constrained during optimization')

opt_type.add_argument('-ct', '--constrain-torsions', \
                      type=str, \
                      action='append', \
                      default=None, \
                      dest='torsions', \
                      help='list of comma-separated atom quartets\n' + \
                           'the torsional angle defined by each quartet will be constrained during optimization')

freq_type = theory_parser.add_argument_group("Frequency options")
freq_type.add_argument('-n', '--numerical', \
                       action='store_true', \
                       default=False, \
                       dest='numerical', \
                       help='request numerical frequencies')

freq_type.add_argument('-t', '--temperature', \
                       type=float, \
                       default=298.15, \
                       dest='temperature', \
                       help='temperature for calculated thermochemical corrections\nDefault: 298.15')

#TODO: add program-specific options 
#      or not - it might be easier to just add them with a text editor
#TODO: grab presets from the aaron rc file
gaussian_options = theory_parser.add_argument_group('Gaussian options')


orca_options = theory_parser.add_argument_group('ORCA options')


psi4_options = theory_parser.add_argument_group('Psi4 options')


args = theory_parser.parse_args()

for f in args.infile:
    if isinstance(f, str):
        if args.input_format is not None:
            infile = FileReader((f, args.input_format[0], None), just_geom=False, get_all=True)
        else:
            infile = FileReader(f, just_geom=False, get_all=True)
    else:
        if args.input_format is not None:
            infile = FileReader(('from stdin', args.input_format[0], f), just_geom=False, get_all=True)
        else:
            if len(sys.argv) >= 1:
                infile = FileReader(('from stdin', 'xyz', f), just_geom=False, get_all=True)

    geom = Geometry(infile)



    if args.method is None:
        if 'method' in infile.other:
            method = infile.other['method'].split('/')[0]
        else:
            raise RuntimeError("method was not determined from %s and was not specified with --method" % f)
    else:
        method = args.method



    if args.basis is not None:
        basis_sets = []
        for basis in args.basis:
            aux_type = None
            elements = None
            i = 0
            while i < len(basis):
                if any(ele == basis[i] for ele in ELEMENTS) or any(x == basis[i].lower() for x in ['tm', '!tm', 'all']):
                    if elements is None:
                        elements = [basis[i]]
                    else:
                        elements.append(basis[i])
                elif 'aux' in basis[i].lower():
                    i += 1
                    aux_type = basis[i]
                else:
                    basis_sets.append(Basis(basis[i], elements, aux_type))
                    break

                i += 1

    else:
        if 'method' in infile.other:
            basis_sets = method.split('/')[-1]
        else:
            basis_sets = None



    if args.ecp is not None:
        ecps = []
        for ecp in args.ecp:
            elements = None
            i = 0
            while i < len(ecp):
                if any(ele == ecp[i] for ele in ELEMENTS) or any(x == ecp[i].lower() for x in ['tm', '!tm', 'all']):
                    if elements is None:
                        elements = [ecp[i]]
                    else:
                        elements.append(ecp[i])
                else:
                    ecps.append(ECP(ecp[i], elements))
            
                i += 1

    else:
        ecps = None



    if args.ecp is None and args.basis is None:
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
    if args.optimize:
        constraints = {}
        if args.atoms is not None:
            constraints['atoms'] = geom.find(args.atoms)

        if args.bonds is not None:
            constraints['bonds'] = []
            for bond in args.bonds:
                bonded_atoms = geom.find(bond)
                if len(bonded_atoms) != 2:
                    raise RuntimeError("not exactly 2 atoms specified in a bond constraint\nuse the format --constrained-bonds 1,2 3,4")
                constraints['bonds'].append(bonded_atoms)

        if args.angles is not None:
            constraints['angles'] = []
            for angle in args.angle:
                angle_atoms = geom.find(angle)
                if len(angle_atoms) != 3:
                    raise RuntimeError("not exactly 3 atoms specified in a angle constraint\nuse the format --constrained-bonds 1,2,3 4,5,6")
                constraints['angles'].append(angle_atoms)

        if args.torsions is not None:
            constraints['torsions'] = []
            for torsion in args.torsions:
                torsion_atoms = geom.find(torsion)
                if len(torsion_atoms) != 4:
                    raise RuntimeError("not exactly 4 atoms specified in a torsion constraint\nuse the format --constrained-torsions 1,2,3,4 5,6,7,8")
                constraints['torsions'].append(torsion_atoms)

        if len(constraints.keys()) == 0:
            constraints = None

        job_types.append(OptimizationJob(transition_state=args.ts, constraints=constraints))


    if args.freq:
        job_types.append(FrequencyJob(numerical=args.numerical, temperature=args.temperature))

    if args.energy:
        job_types.append(SinglePointJob())

    
    if args.charge is None:
        if 'charge' in infile.other:
            charge = infile.other['charge']
        else:
            charge = 0
    else:
        charge = args.charge


    if args.multiplicity is None:
        if 'multiplicity' in infile.other:
            multiplicity = infile.other['multiplicity']
        else:
            multiplicity = 1
    else:
        multiplicity = args.multiplicity


    other_kwargs = {}

    theory = Theory(method=method, 
                    basis=basis_set, 
                    grid=args.grid, 
                    solvent=None, 
                    job_type=job_types, 
                    empirical_dispersion=args.empirical_dispersion, 
                    charge=charge, 
                    multiplicity=multiplicity,
                    processors=args.processors,
                    memory=args.memory,
             )


    if args.out_format == 'gaussian':
        style = 'com'
        if args.ts and args.optimize:
            other_kwargs[GAUSSIAN_ROUTE] = {'Opt': ['CalcFC']}
    elif args.out_format == 'orca':
        style = 'inp'
    elif args.out_format == 'psi4':
        style = 'in'
    else:
        if args.outfile:
            style = splitext(args.outfile)[-1]
        else:
            raise RuntimeError("file format must be specified if no output file is specified")

    s = geom.write(append=True, outfile=args.outfile, style=style, theory=theory, **other_kwargs)
    if not args.outfile:
        print(s)
