"""For parsing input/output files"""
import os
import re
import sys
from copy import deepcopy
from io import IOBase, StringIO
from math import ceil

import numpy as np

from AaronTools import addlogger
from AaronTools.atoms import Atom
from AaronTools.oniomatoms import OniomAtom
from AaronTools.const import ELEMENTS, PHYSICAL, UNIT
from AaronTools.orbitals import Orbitals
from AaronTools.spectra import (
    Frequency,
    ValenceExcitations,
    NMR,
)
from AaronTools.theory import *
from AaronTools.utils.utils import (
    is_alpha,
    is_int,
    is_num,
    float_num,
    perp_vector,
    rotation_matrix,
    angle_between_vectors,
)

read_types = [
    "xyz",
    "log",
    "com",
    "gjf",
    "sd",
    "sdf",
    "mol",
    "mol2",
    "out",
    "dat",
    "fchk",
    "pdb",
    "pdbqt",
    "cif",
    "mmcif",
    "crest",
    "xtb",
    "sqmout",
    "47",
    "31",
    "qout",
]
write_types = ["xyz", "com", "inp", "inq", "in", "sqmin", "cube", "xtb", "crest", "mol"]
file_type_err = "File type not yet implemented: {}"
#LAH_bonded_to = re.compile("(LAH) bonded to ([0-9]+)")
#LA_atom_type = re.compile("(?<=')[A-Z][A-Z](?=')")
#LA_charge = re.compile("[-+]?[0-9]*\.[0-9]+")
#LA_bonded_to = re.compile("(?<=')([0-9][0-9]?)(?![0-9 A-Z\.])(?=')")
#Svalue = re.compile("(?<=diff= +)-?[0-9]+\.[0-9]+")
NORM_FINISH = "Normal termination"
ORCA_NORM_FINISH = "****ORCA TERMINATED NORMALLY****"
PSI4_NORM_FINISH = "*** Psi4 exiting successfully. Buy a developer a beer!"
ERROR = {
    "Fatal Problem: The smallest alpha delta epsilon is": "OMO_UMO_GAP",
    "SCF has not converged.  Gradients and post-SCF results would be GARBAGE!!": "SCF_CONV",
    "Convergence failure -- run terminated.": "SCF_CONV",
    "Inaccurate quadrature in CalDSu": "CONV_CDS",
    "Error termination request processed by link 9999": "CONV_LINK",
    "FormBX had a problem": "FBX",
    "NtrErr Called from FileIO": "CHK",
    "Wrong number of Negative eigenvalues": "EIGEN",
    "Erroneous write": "QUOTA",
    "Atoms too close": "CLASH",
    "Small interatomic distances encountered:": "CLASH",
    "The combination of multiplicity": "CHARGEMULT",
    "Bend failed for angle": "REDUND",
    "Linear angle in Bend": "REDUND",
    "Error in internal coordinate system": "COORD",
    "galloc: could not allocate memory": "GALLOC",
    "Error imposing constraints": "CONSTR",
    "End of file reading basis center.": "BASIS_READ",
    re.compile("Atomic number out of range for .* basis set."): "BASIS",
    "Unrecognized atomic symbol": "ATOM",
    "malloc failed.": "MEM",
    "A syntax error was detected in the input line": "SYNTAX",
    "Unknown message": "UNKNOWN",
    "Atoms in 1 layers were given but there should be 2": "LAYER",
    "MM function not complete": "MM_PARAM",
    "PCMIOp: Cannot load options.": "PCM",
}

ERROR_ORCA = {
    "ORCA finished by error termination in SCF": "SCF_CONV",
    "SCF NOT CONVERGED AFTER": "SCF_CONV",
    # ORCA doesn't actually exit if the SCF doesn't converge...
    # "CONV_CDS": "",
    "The optimization did not converge but reached the maximum number": "OPT_CONV",
    # ORCA still prints the normal finish line if opt doesn't converge...
    # "FBX": "",
    # "CHK": "",
    # "EIGEN": "", <- ORCA doesn't seem to have this
    # "QUOTA": "",
    "Zero distance between atoms": "CLASH",  # <- only get an error if atoms are literally on top of each other
    "Error : multiplicity": "CHARGEMULT",
    # "REDUND": "",
    # "REDUND": "",
    # "GALLOC": "",
    # "CONSTR": "",
    "The basis set was either not assigned or not available for this element": "BASIS",
    "Element name/number, dummy atom or point charge expected": "ATOM",
    "Error  (ORCA_SCF): Not enough memory available!": "MEM",
    "WARNING: Analytical MP2 frequency calculations": "NUMFREQ",
    "WARNING: Analytical Hessians are not yet implemented for meta-GGA functionals": "NUMFREQ",
    "ORCA finished with error return": "UNKNOWN",
    "UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE": "TYPO",
}

# some exceptions are listed in https://psicode.org/psi4manual/master/_modules/psi4/driver/p4util/exceptions.html
ERROR_PSI4 = {
    "PsiException: Could not converge SCF iterations": "SCF_CONV",
    "psi4.driver.p4util.exceptions.SCFConvergenceError: Could not converge SCF iterations": "SCF_CONV",
    "OptimizationConvergenceError": "OPT_CONV",
    "TDSCFConvergenceError": "TDCF_CONV",
    "The INTCO_EXCEPTion handler": "INT_COORD",
    # ^ this is basically psi4's FBX
    # "CONV_CDS": "",
    # "CONV_LINK": "",
    # "FBX": "",
    # "CHK": "",
    # "EIGEN": "", <- psi4 doesn't seem to have this
    # "QUOTA": "",
    # "ValidationError:": "INPUT", <- generic input error, CHARGEMULT and CLASH would also get caught by this
    "qcelemental.exceptions.ValidationError: Following atoms are too close:": "CLASH",
    "qcelemental.exceptions.ValidationError: Inconsistent or unspecified chg/mult": "CHARGEMULT",
    "MissingMethodError": "INVALID_METHOD",
    # "REDUND": "",
    # "REDUND": "",
    # "GALLOC": "",
    # "CONSTR": "",
    "psi4.driver.qcdb.exceptions.BasisSetNotFound: BasisSet::construct: Unable to find a basis set for": "BASIS",
    "qcelemental.exceptions.NotAnElementError": "ATOM",
    "psi4.driver.p4util.exceptions.ValidationError: set_memory()": "MEM",
    # ERROR_PSI4[""] = "UNKNOWN",
    "Could not converge backtransformation.": "ICOORDS",
}


def step2str(step):
    if int(step) == step:
        return str(int(step))
    else:
        return str(step).replace(".", "-")

def str2step(step_str):
    if "-" in step_str:
        return float(step_str.replace("-", "."))
    else:
        return float(step_str)

def expected_inp_ext(exec_type):
    """
    extension expected for an input file for exec_type

    * Gaussian - .com (.gjf on windows)
    * ORCA - .inp
    * Psi4 - .in
    * SQM - .mdin
    * qchem - .inp

    """
    if exec_type.lower() == "gaussian":
        if sys.platform.startswith("win"):
            return ".gjf"
        return ".com"
    if exec_type.lower() == "orca":
        return ".inp"
    if exec_type.lower() == "psi4":
        return ".in"
    if exec_type.lower() == "sqm":
        return ".mdin"
    if exec_type.lower() == "qchem":
        return ".inp"

def expected_out_ext(exec_type):
    """
    extension expected for an input file for exec_type

    * Gaussian - .log
    * ORCA - .out
    * Psi4 - .out
    * SQM - .mdout
    * qchem - .out

    """
    if exec_type.lower() == "gaussian":
        return ".log"
    if exec_type.lower() == "orca":
        return ".out"
    if exec_type.lower() == "psi4":
        return ".out"
    if exec_type.lower() == "sqm":
        return ".mdout"
    if exec_type.lower() == "qchem":
        return ".out"


class FileWriter:
    """
    class for handling file writing
    """

    @classmethod
    def write_file(
        cls, geom, style=None, append=False, outfile=None, *args, **kwargs
    ):
        """
        Writes file from geometry in the specified style

        :param Geometry geom: the Geometry to use
        :param str style: the file type style to generate
            Currently supported options: "xyz" (default), "com",
            "inp", "inq", "in", "sqmin", "cube", "xtb", "crest", "mol"

            if outfile has one of these extensions, default is that style
        :param bool append: for *.xyz, append geometry to the same file
        :param str|None|False outfile: output destination - default is
            [geometry name] + [extension] or [geometry name] + [step] + [extension]

            if outfile is False, no output file will be written, but the contents will be returned
        :param Theory theory: for com, inp, and in files, an object with a get_header and get_footer method
        """
        if isinstance(outfile, str) and style is None:
            name, ext = os.path.splitext(outfile)
            style = ext.strip(".")

        elif style is None:
            style = "xyz"

        if style.lower() not in write_types:
            if style.lower() == "gaussian":
                style = "com"
            elif style.lower() == "orca":
                style = "inp"
            elif style.lower() == "psi4":
                style = "in"
            elif style.lower() == "sqm":
                style = "sqmin"
            elif style.lower() == "qchem":
                style = "inq"
            elif style.lower() == "pdb":
                style = "pdb"
            else:
                raise NotImplementedError(file_type_err.format(style))

        if (
            outfile is None and
            os.path.dirname(geom.name) and
            not os.access(os.path.dirname(geom.name), os.W_OK)
        ):
            os.makedirs(os.path.dirname(geom.name))
        elif (
            isinstance(outfile, str) and
            os.path.dirname(outfile) and
            not os.access(os.path.dirname(outfile), os.W_OK)
        ):
            os.makedirs(os.path.dirname(outfile))
        if style.lower() == "xyz":
            if "oniom" in kwargs and "models" not in kwargs:
                out = cls.write_oniom_xyz(geom, append, outfile, **kwargs)
            elif "oniom" in kwargs and "models" in kwargs:
                out = cls.write_multi_xyz(geom, append, outfile, **kwargs)
            else:
                out = cls.write_xyz(geom, append, outfile)

        elif style.lower() == "mol":
            out = cls.write_mol(geom, outfile=outfile)
        elif style.lower() == "com":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
            else:
                raise TypeError(
                    "when writing 'com/gjf' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
            out = cls.write_com(geom, theory, outfile, **kwargs)
        elif style.lower() == "inp":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_inp(geom, theory, outfile=outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'inp' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "in":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_in(geom, theory, outfile=outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'in' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "sqmin":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_sqm(geom, theory, outfile=outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'sqmin' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "inq":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_inq(geom, theory, outfile=outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'inq' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "xtb":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_xtb(geom, theory, outfile=outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'xtb' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )
        elif style.lower() == "crest":
            if "theory" in kwargs:
                theory = kwargs["theory"]
                del kwargs["theory"]
                out = cls.write_crest(geom, theory, outfile=outfile, **kwargs)
            else:
                raise TypeError(
                    "when writing 'crest' files, **kwargs must include: theory=Aaron.Theory() (or AaronTools.Theory())"
                )

        elif style.lower() == "cube":
            out = cls.write_cube(geom, outfile=outfile, **kwargs)

        elif style.lower() == "pdb":
            out = cls.write_pdb(geom, append, outfile=outfile, **kwargs)

        return out

    @classmethod
    def write_xyz(cls, geom, append, outfile=None):
        """
        write xyz file
        """
        mode = "a" if append else "w"
        fmt = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f}\n"
        s = "%i\n" % len(geom.atoms)
        s += "%s\n" % geom.comment
        for atom in geom.atoms:
            s += fmt.format(atom.element, *atom.coords)

        if outfile is None:
            # if no output file is specified, use the name of the geometry
            with open(geom.name + ".xyz", mode) as f:
                f.write(s)
        elif outfile is False:
            # if no output file is desired, just return the file contents
            return s.strip()
        else:
            # write output to the requested destination
            with open(outfile, mode) as f:
                f.write(s)

        return

    @classmethod
    def write_multi_xyz(cls, geom, append, outfile=None, **kwargs):
        """
        write multiple oniom xyz files from geometry with multiple poses such as a pdb derived geometry

        kwargs["models"] can be string "all", string of model number e.g. "2", string of model range e.g. "1-5",
            or list of model numbers including ranges e.g. ["1", "3-5", "10"]
        kwargs["oniom"] can be string "all" or string "frag" which requires a specification of the fragment in another kwarg
        kwargs["layer"] can be defined if kwargs["oniom"] == "frag", can be "H", "M", or "L"
        """
        models = None
        geom_list = [geom]
        if "models" in kwargs.keys():
            models = kwargs["models"]
        if models is not None:
            if isinstance(models, str):
                if models != "all":
                    try:
                        models = int(models)
                        models = ["model_%s" % str(models)]
                    except ValueError:
                        if "-" in models:
                            models = models.split("-")
                            model_list = []
                            for i in range(int(models[0]), int(models[1])+1):
                                model_list.append("models_%s" % str(i))
                            models = model_list
                        else: raise ValueError("improper specification of included models")
            elif isinstance(models, list):
                model_list = []
                for model in models:
                    if "-" in model:
                        model = model.split("-")
                        for i in range(int(model[0]), int(model[1])+1):
                            model_list.append("model_%s" % str(i))
                    else:
                        model_list.append("model_%s" % str(model))
                models = model_list

            for key in geom.other.keys():
                if key.startswith("model"):
                    if models == "all":
                        geom_list.append(Geometry(structure=geom.other[key], name=geom.name + "_" + key, refresh_connected=False, refresh_ranks = False))
                    elif isinstance(models, list):
                        if key in models:
                            geom_list.append(Geometry(structure=geom.other[key], name=geom.name + "_" + key, refresh_connected=False, refresh_ranks = False))

        counter = 0
        for geom in geom_list:
            if outfile == False:
                FileWriter.write_oniom_xyz(geom, append, outfile = False, **kwargs)
            elif outfile==None:
                FileWriter.write_oniom_xyz(geom, append, outfile = geom.name, **kwargs)
            else:
                counter += 1
                outfile_name = outfile.split(".")[0] + "_" + str(counter) + "." + outfile.split(".")[1]
                FileWriter.write_oniom_xyz(geom, append, outfile = outfile_name, **kwargs)
        return

    @classmethod
    def write_oniom_xyz(cls, geom, append, outfile=None, **kwargs):
        """
        write xyz files with additional columns for atomtype, charge, and link atom info

        kwargs["oniom"] can be string "all" or string "frag" which requires a specification of the fragment in another kwarg
        kwargs["layer"] can be defined if kwargs["oniom"] == "frag", can be "H", "M", or "L"
        """
        frag = kwargs["oniom"]
        if frag == 'all':
            geom.sub_links()
        elif frag == 'layer':
            geom=geom.oniom_frag(layer=kwargs["layer"], as_object=True)

        mode = "a" if append else "w"
        fmt1a = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {:3s} {: 8.6f} {:2s} {:2s} {: 8.6f} {:2d}\n"
        fmt1b = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {:3s} {:2s} {:2s} {:2d}\n"
        fmt1c = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {: 8.6f} {:2s} {: 8.6f} {:2d}\n"
        fmt1d = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {:2s} {:2d}\n"
        fmt2a = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {:3s} {: 8.6f}\n"
        fmt2b = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {:3s}\n"
        fmt2c = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s} {: 8.6f}\n"
        fmt2d = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} {:2s}\n"
        fmt3 = "{:3s} {: 10.5f} {: 10.5f} {: 10.5f} \n"

        s = "%i\n" % len(geom.atoms)
        s += "%s\n" % geom.comment
        for atom in geom.atoms:
            if atom.link_info:
                if "atomtype" not in atom.link_info.keys():
                    connected_elements = []
                    for connected in atom.connected:
                        connected_elements.append(connected.element)
                    if "C" in connected_elements:
                        atom.link_info["atomtype"] = "hc"
                    elif "C" not in connected_elements and "N" in connected_elements:
                        atom.link_info["atomtype"] = "hn"
                    elif "C" not in connected_elements and "O" in connected_elements:
                        atom.link_info["atomtype"] = "ho"
                    elif "C" not in connected_elements and "S" in connected_elements:
                        atom.link_info["atomtype"] = "hs"
                    elif "C" not in connected_elements and "P" in connected_elements:
                        atom.link_info["atomtype"] = "hp"
                if "charge" not in atom.link_info.keys():
                    atom.link_info["charge"] = atom.charge
                if "element" not in atom.link_info.keys():
                    atom.link_info["element"] = "H"
                if "connected" not in atom.link_info.keys():
                    print("Determining link atom connection from connectivity")
                    for connected in atom.connected:
                        if connected.layer == "":
                            raise ValueError("cannot determine link atom connection without defined layers")
                        elif connected.layer != atom.layer:
                            for i, a in enumerate(geom.atoms):
                                if a == connected:
                                    atom.link_info["connected"] = i+1
                                    break
                    if "connected" not in atom.link_info.keys():
                        raise ValueError("Cannot determine link atom connection based on layers")
            try:
                if atom.atomtype != "" and atom.charge != "" and atom.link_info:
                    s += fmt1a.format(atom.element, *atom.coords, atom.layer, atom.atomtype, atom.charge, atom.link_info["element"], atom.link_info["atomtype"], float(atom.link_info["charge"]), int(atom.link_info["connected"]))
                elif atom.atomtype != "" and atom.charge == "" and atom.link_info:
                    s += fmt1b.format(atom.element, *atom.coords, atom.layer, atom.atomtype, atom.link_info["element"], atom.link_info["atomtype"], int(atom.link_info["connected"]))
                elif atom.atomtype == "" and atom.charge != "" and atom.link_info:
                    s += fmt1c.format(atom.element, *atom.coords, atom.layer, atom.charge, atom.link_info["element"], float(atom.link_info["charge"]), int(atom.link_info["connected"]))
                elif atom.atomtype == "" and atom.charge == "" and atom.link_info:
                    s += fmt1d.format(atom.element, *atom.coords, atom.layer, atom.link_info["element"], int(atom.link_info["connected"]))
                elif atom.atomtype != "" and atom.charge != "" and not atom.link_info:
                    s += fmt2a.format(atom.element, *atom.coords, atom.layer, atom.atomtype, atom.charge)
                elif atom.atomtype != "" and atom.charge == "" and not atom.link_info:
                    s += fmt2b.format(atom.element, *atom.coords, atom.layer, atom.atomtype)
                elif atom.atomtype == "" and atom.charge != "" and not atom.link_info:
                    s += fmt2c.format(atom.element, *atom.coords, atom.layer, atom.charge)
                elif atom.atomtype == "" and atom.charge == "" and not atom.link_info:
                    s += fmt2d.format(atom.element, *atom.coords, atom.layer)
            except ValueError:
                self.LOG.warning("no layers designated for OniomAtom object(s)")
                s += fmt3.format(atom.element, *atom.coords)

        s = s.rstrip()

        if outfile is None:
            #if no output file is specified, use the name of the geometry
            with open(geom.name + ".xyz", mode) as f:
                f.write(s)
        elif outfile is False:
            #if no output file is desired, just return the file contents
            return s
        else:
            #write output to the requested destination
            with open(outfile, mode) as f:
                f.write(s)

        return

    @classmethod
    def write_mol(
        cls, geom, outfile=None, **kwargs
    ):
        """write V2000 mol file"""
        from AaronTools.finders import ChiralCenters
        from AaronTools.const import ELECTRONEGATIVITY

        elements = geom.element_counts()
        s = ""
        for ele, n in sorted(elements.items(), key=lambda ele: -1 if ele[0] == "C" else ELEMENTS.index(ele[0])):
            s += "%s%i" % (ele, n)
        s += "\nAaronTools\n%s\n" % geom.comment

        def bond_order_to_code(x):
            if x == 1.5:
                return 4
            return int(x)

        atom_block = ""
        bond_block = ""
        n_bonds = 0
        for i, atom in enumerate(geom.atoms):
            atom_block += "%10.4f%10.4f%10.4f %3s 0%3i  0  0  0  0  0  0  0  0\n" % (
                *atom.coords,
                atom.element,
                0 # if not hasattr(atom, "_saturation") else len(atom.connected) - atom._saturation,
            )
            n_bonds += len(atom.connected)

        try:
            geom.find(ChiralCenters())
            chiral = True
        except LookupError:
            chiral = False

        s += "%3i%3i  0  0%3i  0  0  0  0  0  0 V2000\n" % (
            len(geom.atoms),
            n_bonds // 2,
            1 if chiral else 0,
        )
        s += atom_block

        # determine bond info
        # need to be extra careful with aromatic bonds b/c
        # sometimes conjugated bonds look like aromatic to AaronTools
        # only atoms in a ring should have aromatic bonds, and those
        # bonds should only be to atoms in the same ring
        rings = []
        bonds = dict()
        ndx = {atom: i for i, atom in enumerate(geom.atoms)}
        for atom in geom.atoms:
            for i, atom2 in enumerate(atom.connected):
                bond_order = atom.bond_order(atom2)
                if ndx[atom] < ndx[atom2]:
                    bonds[(atom, atom2)] = bond_order
                for atom3 in list(atom.connected)[:i]:
                    try:
                        path = geom.shortest_path(atom2, atom3, avoid=atom)
                        rings.append(set([*path, atom]))
                    except LookupError:
                        pass

        for bond, order in bonds.items():
            if order == 1.5:
                for ring in rings:
                    if len(ring.intersection(bond)) == 2:
                        break
                    elif len(ring.intersection(bond)) == 1:
                        bonds[bond] = 1
                        break
                else:
                    # flip the sign to show that this is something AaronTools
                    # says is aromatic, but isn't in a ring
                    if bonds[bond] == 1.5:
                        bonds[bond] *= -1

        # group delocalized bonds together
        contiguous_aro_bonds = []
        for bond, order in bonds.items():
            if order < 0:
                for group in contiguous_aro_bonds:
                    if group.intersection(bond):
                        group.update(bond)
                        break
                else:
                    contiguous_aro_bonds.append(set(bond))

        for i, group in enumerate(contiguous_aro_bonds):
            pass
            # print(i)
            # for atom in group:
            #     print(atom)
            # print("\n\n")

        # combine groups of delocalized bonds if they overlap
        overlapping_groups = False
        for i, group1 in enumerate(contiguous_aro_bonds):
            for group2 in contiguous_aro_bonds[:i]:
                if group1.intersection(group2):
                    overlapping_groups = True
        while overlapping_groups:
            overlapping_groups = False
            for i, group1 in enumerate(contiguous_aro_bonds):
                found_overlap = False
                for j, group2 in enumerate(contiguous_aro_bonds[:i]):
                    if group1.intersection(group2):
                        group1.update(group2)
                        found_overlap = True
                        contiguous_aro_bonds.pop(j)
                        break
                if found_overlap:
                    break

            for i, group1 in enumerate(contiguous_aro_bonds):
                for group2 in contiguous_aro_bonds[:i]:
                    if group1.intersection(group2):
                        overlapping_groups = True


        # finding the longest path from one atom to another in a group
        # will give us the chain in order
        for group in contiguous_aro_bonds:
            longest_path = []
            for i, atom1 in enumerate(group):
                for atom2 in list(group)[:i]:
                    avoid = [atom for atom in atom1.connected if atom not in group]
                    avoid.extend([atom for atom in atom2.connected if atom not in group])
                    path = geom.shortest_path(
                        atom1, atom2, avoid,
                    )
                    if len(path) > len(longest_path):
                        longest_path = path

            # there might be branches coming off of the main chain
            branches = [longest_path]
            excluded = group - set(longest_path)
            included = group.intersection(longest_path)
            while excluded:
                for branch in branches:
                    for atom in branch[1:-1]:
                        branch_added = False
                        for atom2 in atom.connected.intersection(excluded):
                            longest_path = []
                            for atom3 in excluded - set([atom2]):
                                try:
                                    path = geom.shortest_path(
                                        atom2, atom3, avoid=included,
                                    )
                                    if len(path) > len(longest_path):
                                        longest_path = path
                                except LookupError:
                                    pass
                            if longest_path:
                                branches.append(longest_path)
                                branch_added = True
                                break
                        if branch_added:
                            excluded = excluded - set(branches[-1])
                            included = included.union(set(branches[-1]))

            for branch in branches:
                # if a branch one has two atoms (one bond), look
                # at the neighbors of this to determine a better
                # bond order
                if len(branch) == 2:
                    branch_bond = (branch[0], branch[1])
                    if ndx[branch[0]] > ndx[branch[1]]:
                        branch_bond = (branch[1], branch[0])
                    total_diff = 0
                    for atom in branch:
                        for neighbor in atom.connected - set(branch):
                            total_diff += neighbor._saturation
                            for neighbor2 in neighbor.connected:
                                bond = (neighbor, neighbor2)
                                if ndx[neighbor] > ndx[neighbor2]:
                                    bond = (neighbor2, neighbor)
                                if bonds[bond] > 0:
                                    total_diff -= bonds[bond]
                                else:
                                    total_diff -= 1

                    if total_diff <= 1:
                        bonds[branch_bond] = 1
                    continue

                # for longer chains, just alternate double and single bonds
                # favor double bonds at the more electronegative side of the chain?
                # maybe there's a better way
                start_e_nrg = sum(ELECTRONEGATIVITY[atom.element] for atom in branch[:2])
                end_e_nrg = sum(ELECTRONEGATIVITY[atom.element] for atom in branch[-2:])
                if end_e_nrg > start_e_nrg:
                    branch.reverse()
                for i, atom in enumerate(branch[:-1]):
                    atom2 = branch[i + 1]
                    bond = (atom, atom2)
                    if ndx[atom] > ndx[atom2]:
                        bond = (atom2, atom)

                    if i % 2 == 0:
                        bonds[bond] = 2
                    else:
                        bonds[bond] = 1

        for bond in bonds:
            # print(bond, bonds[bond])
            bond_block += "%3i%3i%3i  0  0  0  0\n" % (
                ndx[bond[0]] + 1, ndx[bond[1]] + 1, bond_order_to_code(bonds[bond])
            )


        s += bond_block
        s += "M  END\n"

        if outfile is None:
            # if no output file is specified, use the name of the geometry
            with open(geom.name + ".mol", "w") as f:
                f.write(s)
        elif outfile is False:
            # if no output file is desired, just return the file contents
            return s.strip()
        else:
            # write output to the requested destination
            with open(outfile, "w") as f:
                f.write(s)

    @classmethod
    def write_com(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write Gaussian input file for given Theory and Geometry

        :param Geometry geom: structure
        :param Theory theory: input file parameters
        :param None|False|str outfile:
            output file option

            * None - geom.name + ".com" is used as output destination
            * False - return contents of the input file as a str
            * str - output destination

        :param bool return_warnings: True to return a list of warnings (e.g. basis
            set might be misspelled

        :param kwargs: passed to Theory methods (make_header, make_molecule, etc.)
        """
        # get file content string
        header, header_warnings = theory.make_header(
            geom, return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, return_warnings=True, **kwargs
        )
        footer, footer_warnings = theory.make_footer(
            geom, return_warnings=True, **kwargs
        )

        s = header + mol + footer
        warnings = header_warnings + mol_warnings + footer_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                outfile = "{}.{}.com".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.com".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{{ name }}", name)
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings
        return

    @classmethod
    def write_inp(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write ORCA input file for the given Theory() and Geometry()

        :param Geometry geom: structure
        :param Theory theory: input file parameters
        :param None|False|str outfile:

            * None - geom.name + ".inp" is used as output destination
            * False - return contents of the input file as a str
            * str - output destination

        :param bool return_warnings: True to return a list of warnings (e.g. basis
            set might be misspelled
        :param kwargs: passed to Theory methods (make_header, make_molecule, etc.)
        """
        fmt = "{:<3s} {: 9.5f} {: 9.5f} {: 9.5f}\n"
        header, warnings = theory.make_header(
            geom, style="orca", return_warnings=True, **kwargs
        )
        footer = theory.make_footer(
            geom, style="orca", return_warnings=False, **kwargs
        )
        s = header
        for atom in geom.atoms:
            if atom.is_dummy:
                s += fmt.format("DA", *atom.coords)
                continue
            s += fmt.format(atom.element, *atom.coords)

        s += "*\n"

        s += footer

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                outfile = "{}.{}.inp".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.inp".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{{ name }}", name)
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings

    @classmethod
    def write_inq(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write QChem input file for the given Theory() and Geometry()

        :param Geometry geom: structure
        :param Theory theory: input file parameters
        :param None|False|str outfile:

            * None - geom.name + ".inq" is used as output destination
            * False - return contents of the input file as a str
            * str - output destination

        :param bool return_warnings: True to return a list of warnings (e.g. basis
            set might be misspelled
        :param kwargs: passed to Theory methods (make_header, make_molecule, etc.)
        """
        fmt = "{:<3s} {: 9.5f} {: 9.5f} {: 9.5f}\n"
        header, header_warnings = theory.make_header(
            geom, style="qchem", return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, style="qchem", return_warnings=True, **kwargs
        )

        out = header + mol
        warnings = header_warnings + mol_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                outfile = "{}.{}.inq".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.inq".format(geom.name)
        if outfile is False:
            if return_warnings:
                return out, warnings
            return out
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            out = out.replace("{{ name }}", name)
            with open(outfile, "w") as f:
                f.write(out)

        if return_warnings:
            return warnings

    @classmethod
    def write_in(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write Psi4 input file for the given Theory() and Geometry()

        :param Geometry geom: structure
        :param Theory theory: input file parameters
        :param None|False|str outfile:

            * None - geom.name + ".in" is used as output destination
            * False - return contents of the input file as a str
            * str - output destination

        :param bool return_warnings: True to return a list of warnings (e.g. basis
            set might be misspelled
        :param kwargs: passed to Theory methods (make_header, make_molecule, etc.)
        """
        header, header_warnings = theory.make_header(
            geom, style="psi4", return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, style="psi4", return_warnings=True, **kwargs
        )
        footer, footer_warnings = theory.make_footer(
            geom, style="psi4", return_warnings=True, **kwargs
        )

        s = header + mol + footer
        warnings = header_warnings + mol_warnings + footer_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                outfile = "{}.{}.in".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = "{}.in".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{{ name }}", name)
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings

    @classmethod
    def write_sqm(
        cls, geom, theory, outfile=None, return_warnings=False, **kwargs
    ):
        """
        write SQM input file for the given Theory() and Geometry()

        :param Geometry geom: structure
        :param Theory theory: input file parameters
        :param None|False|str outfile:

            * None - geom.name + ".sqmin" is used as output destination
            * False - return contents of the input file as a str
            * str - output destination

        :param bool return_warnings: True to return a list of warnings (e.g. basis
            set might be misspelled
        :param kwargs: passed to Theory methods (make_header, make_molecule, etc.)
        """
        header, header_warnings = theory.make_header(
            geom, style="sqm", return_warnings=True, **kwargs
        )
        mol, mol_warnings = theory.make_molecule(
            geom, style="sqm", return_warnings=True, **kwargs
        )

        s = header + mol
        warnings = header_warnings + mol_warnings

        if outfile is None:
            # if outfile is not specified, name file in Aaron format
            if "step" in kwargs:
                outfile = "{}.{}.sqmin".format(
                    geom.name, step2str(kwargs["step"])
                )
            else:
                outfile = "{}.sqmin".format(geom.name)
        if outfile is False:
            if return_warnings:
                return s, warnings
            return s
        else:
            fname = os.path.basename(outfile)
            name, ext = os.path.splitext(fname)
            # could use jinja, but it's one thing...
            s = s.replace("{{ name }}", name)
            with open(outfile, "w") as f:
                f.write(s)

        if return_warnings:
            return warnings

    @classmethod
    def write_cube(
        cls,
        geom,
        orbitals=None,
        outfile=None,
        kind="homo",
        padding=4.0,
        spacing=0.2,
        alpha=True,
        xyz=False,
        n_jobs=1,
        delta=0.1,
        **kwargs,
    ):
        """
        write a cube file for a molecular orbital

        :param Geometry geom: structure
        :param Orbitals orbitals: orbital data
        :param str outfile: output destination
        :param str|int mo: index of molecular orbital or "homo" for ground state
            highest occupied molecular orbital or "lumo" for first
            ground state unoccupied MO
            can also be an array of MO coefficients

        :param str|int ao: index of atomic orbital to print
        :param float padding: padding around geom's coordinates
        :param float spacing: targeted spacing between points
        :param int n_jobs: number of parallel threads to use
            this is on top of NumPy's multithreading, so
            if NumPy uses 8 threads and n_jobs=2, you can
            expect to see 16 threads in use

        :param float delta: see Orbitals.fukui_donor_value or fukui_acceptor_value
        """
        if orbitals is None:
            raise RuntimeError(
                "no Orbitals() instance given to FileWriter.write_cube"
            )

        n_pts1, n_pts2, n_pts3, v1, v2, v3, com, u = orbitals.get_cube_array(
            geom,
            standard_axes=xyz,
            spacing=spacing,
            padding=padding,
        )

        mo = None
        if kind.lower() == "homo":
            mo = max(orbitals.n_alpha, orbitals.n_beta) - 1
        elif kind.lower() == "lumo":
            mo = max(orbitals.n_alpha, orbitals.n_beta)
        elif kind.lower().startswith("mo"):
            mo = int(kind.split()[-1])
        elif kind.lower().startswith("ao"):
            mo = np.zeros(orbitals.n_mos)
            mo[int(kind.split()[-1])] = 1

        s = ""
        s += " %s\n" % geom.comment
        s += " %s\n" % kind

        # the '-' in front of the number of atoms indicates that this is
        # MO info so there's an extra data entry between the molecule
        # and the function values
        bohr_com = com / UNIT.A0_TO_BOHR
        if isinstance(mo, int):
            s += " -"
        else:
            s += "  "
        s += "%i %13.5f %13.5f %13.5f 1\n" % (
            len(geom.atoms), *bohr_com,
        )

        # the basis vectors of cube files are ordered based on the
        # spacing between points along that axis
        # or maybe it's the number of points?
        # we use the first one
        for n, v in sorted(
            zip([n_pts1, n_pts2, n_pts3], [v1, v2, v3]),
            key=lambda p: np.linalg.norm(p[1]),
        ):
            bohr_v = v / UNIT.A0_TO_BOHR
            s += " %5i %13.5f %13.5f %13.5f\n" % (
                n, *bohr_v
            )
        # contruct an array of points for the grid
        coords, n_list = orbitals.get_cube_points(
            n_pts1, n_pts2, n_pts3, v1, v2, v3, com
        )

        # write the structure in bohr
        for atom in geom.atoms:
            s += " %5i %13.5f %13.5f %13.5f %13.5f\n" % (
                ELEMENTS.index(atom.element),
                ELEMENTS.index(atom.element),
                atom.coords[0] / UNIT.A0_TO_BOHR,
                atom.coords[1] / UNIT.A0_TO_BOHR,
                atom.coords[2] / UNIT.A0_TO_BOHR,
            )

        # extra section - only for MO data
        if isinstance(mo, int):
            s += " %5i %5i\n" % (1, mo + 1)

        # get values for this MO
        if kind.lower() == "density":
            val = orbitals.density_value(coords, n_jobs=n_jobs)
        elif kind.lower() == "fukui donor":
            val = orbitals.fukui_donor_value(
                coords, n_jobs=n_jobs, delta=delta
            )
        elif kind.lower() == "fukui acceptor":
            val = orbitals.fukui_acceptor_value(
                coords, n_jobs=n_jobs, delta=delta
            )
        elif kind.lower() == "fukui dual":
            val = orbitals.fukui_dual_value(
                coords, n_jobs=n_jobs, delta=delta
            )
        else:
            val = orbitals.mo_value(mo, coords, n_jobs=n_jobs)

        # write to a file
        for n1 in range(0, n_list[0]):
            for n2 in range(0, n_list[1]):
                val_ndx = n1 * n_list[2] * n_list[1] + n2 * n_list[2]
                val_subset = val[val_ndx : val_ndx + n_list[2]]
                for i, v in enumerate(val_subset):
                    if abs(v) < 1e-30:
                        v = 0
                    s += "%13.5e" % v
                    if (i + 1) % 6 == 0:
                        s += "\n"
                if (i + 1) % 6 != 0:
                    s += "\n"

        if outfile is None:
            # if no output file is specified, use the name of the geometry
            with open(geom.name + ".cube", "w") as f:
                f.write(s)
        elif outfile is False:
            # if no output file is desired, just return the file contents
            return s
        else:
            # write output to the requested destination
            with open(outfile, "w") as f:
                f.write(s)
        return

    @classmethod
    def write_pdb(cls, geom, append, outfile=None, qt=False):
        if not isinstance(geom.atoms[0], OniomAtom):
            geom = geom.make_oniom()
        mode = "a" if append else "w"
        if "model_2" in geom.other.keys():
            models = True
        else:
            models = False
        s = ""
        def spaced(spac, val, align="right"):
            if not isinstance(val, str):
                val = str(val).strip()
            val_predecimal = len(val.split(".")[0])
            writ_space = spac
            if len(val) > writ_space:
                val=str(round(float(val),writ_space-val_predecimal-1))
            n = writ_space-len(val)
            sp = " "
            spaces = n*sp
            if align=="right":
                rv = spaces+val
            elif align=="left":
                rv = val+spaces
            elif align=="atomtype":
                if len(val) == 1:
                    rv = sp + val + (n-1)*sp
                elif len(val) in [2,3]:
                    if val[0].isalpha() and val[1].isalpha():
                        rv = val+spaces
                    if val[0].isalpha() and val[1].isdigit():
                        if len(val)==3:
                            rv = spaces+val
                        else:
                            rv = sp + val + sp
                else:
                    rv = val+spaces
            return rv

        connectivity = []
        con_spac = 5

        def write_atoms(atoms, s, get_connect=False):
            for i, atom in enumerate(atoms):
                atom.index = i
                serial_spac = 5
                atom_spac = 4
                res_spac = 3
                coord_spac = 8
                ele_spac = 2
                if get_connect:
                    connectivity.append([])
                    connectivity[-1].append(atom)
                    for connected in atom.connected:
                        connectivity[-1].append(connected)
                if qt == False:
                    charge_spac = 2
                else:
                    charge_spac = 10
                if atom.res:
                    s += "ATOM  "
                else:
                    s += "HETATM"
                s += spaced(serial_spac, str(i+1))
                s += " "
                if qt==True:
                    s += spaced(atom_spac, atom.element)
                else:
                    s += spaced(atom_spac, atom.atomtype, align="atomtype")
                s += " "
                s += spaced(res_spac, atom.res)
                s += 10 * " "
                for coord in atom.coords:
                    s += spaced(coord_spac, coord)
                if qt == False:
                    s += 22*" "
                    s += spaced(ele_spac, atom.element)
                    if hasattr(atom, "charge") and atom.charge != None and atom.charge != "":
                        s += "{: 4.2f}".format(atom.charge)
                else:
                    s += 12 * " "
                    s += spaced(charge_spac, atom.charge, align="left")
                    s += spaced(ele_spac, atom.element)
                s += "\n"
            return s

        if hasattr(geom, "name"):
            s += "HEADER"
            s += " " * 52
            s += geom.name
            s += "\n"

        if hasattr(geom, "other"):
            if isinstance(geom.other, dict) and "source" in geom.other.keys():
                s += "EXPDATA"
                s += " " *3
                s += geom.other["source"]
                s += "\n"

        if models == True:
            num_models = 1
            s += "MODEL 1\n"
            s = write_atoms(geom.atoms, s, get_connect=True)
            s += "ENDMDL\n"
            for key in geom.other.keys():
                if key.startswith("model"):
                    num_models += 1
                    s += "MODEL %s\n" % str(num_models)
                    s = write_atoms(geom.other[key], s)
                    s += "ENDMDL\n"

        elif models == False:
            s = write_atoms(geom.atoms, s, get_connect=True)

        for connection in connectivity:
            s += "CONECT"
            for connect in connection:
                s += spaced(con_spac, connect.index+1)
            s += "\n"

        if outfile is None:
            #if no output file is specified, use the name of the geometry
            with open(geom.name + ".xyz", mode) as f:
                f.write(s)
        elif outfile is False:
            #if no output file is desired, just return the file contents
            return s
        else:
            #write output to the requested destination
            with open(outfile, mode) as f:
                f.write(s)

        return

    @classmethod
    def write_xtb(
        cls,
        geom,
        theory,
        outfile=None,
        return_warnings=False,
        **kwargs,
    ):
        """write input files for xtb"""
        if theory.job_type:
            for job in theory.job_type:
                if hasattr(job, "geometry"):
                    job.geometry = geom

        contents = dict()
        cli, cli_warnings = theory.get_xtb_cmd(
            return_warnings=True, **kwargs
        )
        contents["cmd"] = cli
        xcontrol, xc_warnings, write_ref = theory.get_xtb_control(
            return_warnings=True, **kwargs
        )
        contents["xc"] = xcontrol

        contents["xyz"] = cls.write_xyz(geom, append=False, outfile=False)

        warnings = cli_warnings + xc_warnings

        if write_ref:
            contents[write_ref] = contents["xyz"]

        if outfile is False:
            if return_warnings:
                return contents, warnings
            return contents

        if outfile is None:
            if "step" in kwargs:
                outfile = "{}.{}".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = geom.name

        dirname, basename = os.path.split(outfile)
        name, ext = os.path.splitext(basename)

        cls.write_dict_files(contents, dirname, name)

        if return_warnings:
            return warnings

    @classmethod
    def write_crest(
        cls,
        geom,
        theory,
        outfile=None,
        return_warnings=False,
        **kwargs,
    ):
        """write crest input files"""
        if theory.job_type:
            for job in theory.job_type:
                if hasattr(job, "geometry"):
                    job.geometry = geom

        contents = dict()
        cli, cli_warnings = theory.get_crest_cmd(
            return_warnings=True, **kwargs
        )
        contents["cmd"] = cli
        xcontrol, xc_warnings, write_ref = theory.get_xtb_control(
            return_warnings=True, crest=True, **kwargs
        )
        contents["xc"] = xcontrol
        contents["xyz"] = cls.write_xyz(geom, append=False, outfile=False)
        if write_ref:
            contents[write_ref] = contents["xyz"]

        warnings = cli_warnings + xc_warnings

        if outfile is False:
            if return_warnings:
                return contents, warnings
            return contents

        if outfile is None:
            if "step" in kwargs:
                outfile = "{}.{}".format(geom.name, step2str(kwargs["step"]))
            else:
                outfile = geom.name

        dirname, basename = os.path.split(outfile)
        name, ext = os.path.splitext(basename)

        cls.write_dict_files(contents, dirname, name)

        if return_warnings:
            return warnings

    @staticmethod
    def write_dict_files(contents, dirname, name):
        """
        write data to different files

        :param dict contents: keys are either a file name (includes a ".") or
            a file extension (no ".")
        
        :param str dirname: where to write files
            e.g. calling with contents as

                {"run.sh": "cat {{ name }}.txt", "txt": "hi"}

            and name as "test"
            will write run.sh and test.txt to dirname
        """
        for key, data in contents.items():
            if "." in key:
                output_path = os.path.join(dirname, key)
            else:
                output_path = os.path.join(dirname, "%s.%s" % (name, key))

            with open(output_path, "w") as f:
                f.write(data.replace("{{ name }}", name))


@addlogger
class FileReader:
    """
    class for reading files

    Attributes

    * name
    * file_type
    * comment
    * atoms list(Atom) or list(OniomAtom)
    * other dict

    """

    LOG = None
    LOGLEVEL = "DEBUG"

    def __init__(
        self,
        fname,
        get_all=False,
        just_geom=True,
        oniom=False,
        scan_read_all=False,
        freq_name=None,
        conf_name=None,
        nbo_name=None,
        max_length=10000000,
    ):
        """
        :param str|tuple fname: either a string specifying the file name of the file to read
            or a tuple of (str(name), str(file_type), str(content))
        :param bool get_all: if true, optimization steps are also saved in
            self.all_geom; otherwise only saves last geometry
        :param bool just_geom: if true, does not store other information, such as
            frequencies, only what is needed to construct a Geometry() obj
        :param str freq_name: Name of the file containing the frequency output. Only use
            if this information is in a different file than `fname` (eg: xtb runs
            using the --hess runtype option)
        :param str nbo_name: Name of the file containing the NBO orbital coefficients
            in the AO basis. Only used when reading .47 files.
        :param int max_length: maximum array size to store from FCHK files
            
            any array that would be larger than this will be the
            size the array would be ignored
        """
        # Initialization
        self.name = ""
        self.file_type = ""
        self.comment = ""
        self.atoms = []
        self.other = {}
        self.content = None
        self.all_geom = None

        # get file name and extention
        if isinstance(fname, str):
            self.name, self.file_type = os.path.splitext(fname)
            self.file_type = self.file_type.lower()[1:]
        elif isinstance(fname, (tuple, list)):
            self.name = fname[0]
            self.file_type = fname[1]
            self.content = fname[2]
        if self.file_type not in read_types:
            raise NotImplementedError(file_type_err.format(self.file_type))

        # Fill in attributes with geometry information
        if self.content is None:
            self.read_file(
                get_all, just_geom, scan_read_all,
                freq_name=freq_name,
                conf_name=conf_name,
                nbo_name=nbo_name,
                max_length=max_length,
                oniom=oniom
            )
        elif isinstance(self.content, str):
            if os.path.isfile(self.name):
                f = open(self.name, "r", encoding="utf8")
            elif len(self.content.splitlines()):
                f = StringIO(self.content)
            else:
                fname = ".".join([self.name, self.file_type])
                fname = os.path.expanduser(fname)
                if os.path.isfile(fname):
                    f = open(fname, "r", encoding="utf8")
                else:
                    raise FileNotFoundError(
                        "Error while looking for %s: could not find %s or %s in %s"
                        % (self.name, fname, self.name, os.getcwd())
                    )

        elif isinstance(self.content, IOBase):
            f = self.content

        if self.content is not None:
            if self.file_type == "log":
                self.read_log(f, get_all, just_geom, scan_read_all)
            elif any(self.file_type == ext for ext in ["sd", "sdf", "mol"]):
                self.read_sd(f)
            elif self.file_type == "xyz":
                self.read_xyz(f, get_all)
            elif self.file_type == "mol2":
                self.read_mol2(f, get_all)
            elif any(self.file_type == ext for ext in ["com", "gjf"]):
                self.read_com(f)
            elif self.file_type == "out":
                self.read_orca_out(f, get_all, just_geom)
            elif self.file_type == "dat":
                self.read_psi4_out(f, get_all, just_geom)
            elif self.file_type == "fchk":
                self.read_fchk(f, just_geom, max_length=max_length)
            elif self.file_type == "pdb":
                self.read_pdb(f, qt=False)
            elif self.file_type == "pdbqt":
                self.read_pdb(f, qt=True)
            elif self.file_type in ("mmcif", "cif"):
                self.read_mmcif(f)
            elif self.file_type == "crest":
                self.read_crest(f, conf_name=conf_name)
            elif self.file_type == "xtb":
                self.read_xtb(f, freq_name=freq_name)
            elif self.file_type == "sqmout":
                self.read_sqm(f)
            elif self.file_type == "47":
                self.read_nbo_47(f, nbo_name=nbo_name)
            elif self.file_type == "31":
                self.read_nbo_31(f, nbo_name=nbo_name)
            elif self.file_type == "qout":
                self.read_qchem_out(f, get_all, just_geom)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.other[key]

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        self.other[key] = value

    def __contains__(self, key):
        if hasattr(self, key):
            return True
        return key in self.other

    def __delitem__(self, key):
        if hasattr(self, key):
            delattr(self, key)
        if key in self.other:
            del self.other["key"]

    def keys(self):
        attr_keys = set(self.__dict__.keys())
        other_keys = set(self.other.keys())
        keys = attr_keys.union(other_keys)
        keys -= set([
            "content",
            "other",
        ])
        return tuple(keys)

    def values(self):
        keys = self.keys()
        return tuple(self[key] for key in keys)

    def items(self):
        keys = self.keys()
        return tuple((key, self[key]) for key in keys)

    def read_file(
        self, get_all=False, just_geom=True, scan_read_all=False,
        freq_name=None, conf_name=None, nbo_name=None, oniom=False,
        max_length=10000000,
    ):
        """
        Reads geometry information from fname.

        :param bool get_all: If false (default), only keep the last geom
            If true, self is last geom, but return list
            of all others encountered
        :param str nbo_name: nbo output file containing coefficients to
            map AO's to orbitals
        :param int max_length: max. array size for arrays to store in FCHK
            files - anything larger will be the size
            the array would not be stored
        """
        if os.path.isfile(self.name):
            f = open(self.name, "r")
        else:
            fname = ".".join([self.name, self.file_type])
            fname = os.path.expanduser(fname)
            if os.path.isfile(fname):
                f = open(fname, "r")
            else:
                raise FileNotFoundError(
                    "Error while looking for %s: could not find %s or %s in %s"
                    % (self.name, fname, self.name, os.getcwd())
                )

        if self.file_type == "xyz":
            self.read_xyz(f, get_all, oniom)
        elif self.file_type == "log":
            self.read_log(f, get_all, just_geom, scan_read_all)
        elif any(self.file_type == ext for ext in ["com", "gjf"]):
            self.read_com(f)
        elif any(self.file_type == ext for ext in ["sd", "sdf", "mol"]):
            self.read_sd(f)
        elif self.file_type == "mol2":
            self.read_mol2(f)
        elif self.file_type == "out":
            self.read_orca_out(f, get_all, just_geom)
        elif self.file_type == "dat":
            self.read_psi4_out(f, get_all, just_geom)
        elif self.file_type == "fchk":
            self.read_fchk(f, just_geom, max_length=max_length)
        elif self.file_type == "pdb":
            self.read_pdb(f, qt=False)
        elif self.file_type == "pdbqt":
            self.read_pdb(f, qt=True)
        elif self.file_type in ("mmcif", "cif"):
            self.read_mmcif(f)
        elif self.file_type == "crest":
            self.read_crest(f, conf_name=conf_name)
        elif self.file_type == "xtb":
            self.read_xtb(f, freq_name=freq_name)
        elif self.file_type == "sqmout":
            self.read_sqm(f)
        elif self.file_type == "47":
            self.read_nbo_47(f, nbo_name=nbo_name)
        elif self.file_type == "31":
            self.read_nbo_31(f, nbo_name=nbo_name)
        elif self.file_type == "qout":
            self.read_qchem_out(f, get_all, just_geom)

        f.close()
        return

    def skip_lines(self, f, n):
        for i in range(n):
            f.readline()
        return

    def read_xyz(self, f, get_all=False, oniom=False):
        """read xyz files"""
        self.all_geom = []
        # number of atoms
        f.readline()
        # comment
        self.comment = f.readline().strip()
        # atom info
        atom_count = 0
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                int(line)
                if get_all:
                    self.all_geom += [
                        {
                            "comment": deepcopy(self.comment),
                            "atoms": deepcopy(self.atoms),
                        }
                    ]
                self.comment = f.readline().strip()
                self.atoms = []
                atom_count = 0
            except ValueError:
                line = line.split()
                element = line[0]
                coords = line[1:4]
                layer = ""
                atomtype = ""
                charge = ""
                link_info = {}
                #tags = ""
                if len(line) > 4:
                    layer = line[4]
                    oniom = True
                    if len(line) == 11:
                        atomtype = line[5]
                        charge = line[6]
                        #tags=line[7:]
                        link_info["element"] = line[7]
                        link_info["atomtype"] = line[8]
                        link_info["charge"] = line[9]
                        link_info["connected"] = line[10]
                    if len(line) == 9:
                        #tags = line[6:]
                        link_info["element"] = line[6]
                        link_info["connected"] = line[8]
                        if is_alpha(line[5][0]):
                            atomtype = line[5]
                            link_info["atomtype"] = line[7]
                        else:
                            charge = line[5]
                            link_info["charge"] = line[7]
                    if len(line) == 7:
                        if line[6].isdigit():
                            #tags = line[5:]
                            link_info["element"] = line[5]
                            link_info["connected"] = line[6]
                        else:
                            atomtype = line[5]
                            charge = line[6]
                    if len(line) == 6:
                        if is_alpha(line[5][0]):
                            atomtype = line[5]
                        else:
                            charge = line[5]
                if oniom == True:
                    atom_count += 1
                    self.atoms += [OniomAtom(element=element, coords=coords, layer=layer, atomtype=atomtype, charge=charge, link_info=link_info, name=str(atom_count))]
                else:
                    atom_count += 1
                    self.atoms += [Atom(element=line[0], coords=line[1:4], name=str(atom_count))]

        if get_all:
            self.all_geom += [
                {
                    "comment": self.comment,
                    "atoms": self.atoms,
                }
            ]

    def read_sd(self, f, get_all=False):
        """read sdf file"""
        self.all_geom = []
        lines = f.readlines()
        progress = 0
        for i, line in enumerate(lines):
            progress += 1
            if "$$$$" in line:
                progress = 0
                if get_all:
                    self.all_geom.append({
                        "comment": deepcopy(self.comment),
                        "atoms": deepcopy(self.atoms),
                    })

                continue

            if progress == 3:
                self.comment = line.strip()

            if progress == 4:
                natoms = int(line[0:3])
                nbonds = int(line[3:6])

            if progress == 5:
                self.atoms = []
                for line in lines[i : i + natoms]:
                    atom_info = line.split()
                    self.atoms += [
                        Atom(element=atom_info[3], coords=atom_info[0:3])
                    ]

                for line in lines[i + natoms : i + natoms + nbonds]:
                    a1, a2 = [int(line[3 * j: 3 * (j + 1)]) - 1 for j in [0, 1]]
                    self.atoms[a1].connected.add(self.atoms[a2])
                    self.atoms[a2].connected.add(self.atoms[a1])

                for j, a in enumerate(self.atoms):
                    a.name = str(j + 1)

                self.other["charge"] = 0
                for line in lines[i + natoms + nbonds:]:
                    if "CHG" in line:
                        self.other["charge"] += int(line.split()[-1])
                    if "$$$$" in line:
                        break

        if get_all:
            self.all_geom.append({
                "comment": self.comment,
                "atoms": self.atoms,
            })

    def read_mol2(self, f, get_all=False):
        """
        read TRIPOS mol2
        """
        atoms = []

        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("@<TRIPOS>MOLECULE"):
                self.comment = lines[i + 1]
                info = lines[i + 2].split()
                n_atoms = int(info[0])
                n_bonds = int(info[1])
                i += 3

            elif lines[i].startswith("@<TRIPOS>ATOM"):
                for j in range(0, n_atoms):
                    i += 1
                    info = lines[i].split()
                    # name = info[1]
                    coords = np.array([float(x) for x in info[2:5]])
                    element = re.match("([A-Z][a-z]?)", info[1]).group(1)
                    atoms.append(
                        Atom(element=element, coords=coords, name=str(j + 1))
                    )

                self.atoms = atoms

            elif lines[i].startswith("@<TRIPOS>BOND"):
                for j in range(0, n_bonds):
                    i += 1
                    info = lines[i].split()
                    a1, a2 = [int(ndx) - 1 for ndx in info[1:3]]
                    self.atoms[a1].connected.add(self.atoms[a2])
                    self.atoms[a2].connected.add(self.atoms[a1])

            i += 1

        if get_all:
            self.all_geom.append({
                "comment": self.comment,
                "atoms": self.atoms,
            })

    def read_psi4_out(self, f, get_all=False, just_geom=True):
        """read psi4 output file"""
        uv_vis = ""
        coord_unit_bohr = False
        def get_atoms(f, n, bohr):
            rv = []
            self.skip_lines(f, 1)
            n += 2
            line = f.readline()
            i = 0
            mass = 0
            while line.strip():
                i += 1
                line = line.strip()
                atom_info = line.split()
                element = atom_info[0]
                # might be a ghost atom - like for sapt
                if "Gh" in element:
                    element = element.strip("Gh(").strip(")")
                coords = np.array([float(x) for x in atom_info[1:-1]])
                if bohr:
                    coords *= UNIT.A0_TO_BOHR
                atom_mass = float(atom_info[-1])
                rv += [Atom(element=element, coords=coords, mass=atom_mass, name=str(i))]
                mass += atom_mass

                line = f.readline()
                n += 1

            return rv, mass, n

        line = f.readline()
        n = 1
        read_geom = False
        while line != "":
            if "* O   R   C   A *" in line:
                self.file_type = "out"
                return self.read_orca_out(
                    f, get_all=get_all, just_geom=just_geom
                )

            if "A Quantum Leap Into The Future Of Chemistry" in line:
                self.file_type = "qout"
                return self.read_qchem_out(
                    f, get_all=get_all, just_geom=just_geom
                )

            if line.startswith("    Geometry (in Angstrom), charge"):
                coord_unit_bohr = False
                if not just_geom:
                    self.other["charge"] = int(line.split()[5].strip(","))
                    self.other["multiplicity"] = int(
                        line.split()[8].strip(":")
                    )

            elif line.strip() == "SCF":
                read_geom = True

            elif line.startswith("    Geometry (in Bohr), charge"):
                coord_unit_bohr = True

            elif line.strip().startswith("Center") and read_geom:
                read_geom = False
                if get_all and len(self.atoms) > 0:
                    if self.all_geom is None:
                        self.all_geom = []

                    self.all_geom += [{
                        "atoms": deepcopy(self.atoms),
                        "data": deepcopy(self.other),
                    }]

                self.atoms, mass, n = get_atoms(f, n, coord_unit_bohr)
                if not just_geom:
                    self.other["mass"] = mass
                    self.other["mass"] *= UNIT.AMU_TO_KG

            if just_geom:
                line = f.readline()
                n += 1
                continue
            else:
                if line.strip().startswith("Total Energy ="):
                    self.other["energy"] = float(line.split()[-1])

                elif line.strip().startswith("Total E0"):
                    self.other["energy"] = float(line.split()[-2])

                elif line.strip().startswith("Correction ZPE"):
                    self.other["ZPVE"] = float(line.split()[-4])

                elif line.strip().startswith("Total ZPE"):
                    self.other["E_ZPVE"] = float(line.split()[-2])

                elif line.strip().startswith("Total H, Enthalpy"):
                    self.other["enthalpy"] = float(line.split()[-2])

                elif line.strip().startswith("Total G, Free"):
                    self.other["free_energy"] = float(line.split()[-2])
                    self.other["temperature"] = float(line.split()[-4])

                elif "symmetry no. =" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1].strip(")")
                    )

                elif (
                    line.strip().startswith("Rotational constants:")
                    and line.strip().endswith("[cm^-1]")
                    and "rotational_temperature" not in self.other
                ):
                    self.other["rotational_temperature"] = [
                        float(x) if is_num(x) else 0
                        for x in line.split()[-8:-1:3]
                    ]
                    self.other["rotational_temperature"] = [
                        x
                        * PHYSICAL.SPEED_OF_LIGHT
                        * PHYSICAL.PLANCK
                        / PHYSICAL.KB
                        for x in self.other["rotational_temperature"]
                    ]

                elif line.startswith("  Vibration "):
                    freq_str = ""
                    while not line.strip().startswith("=="):
                        freq_str += line
                        line = f.readline()
                        n += 1

                    self.other["frequency"] = Frequency(
                        freq_str, hpmodes=False, style="psi4", atoms=self.atoms,
                    )

                elif PSI4_NORM_FINISH in line:
                    self.other["finished"] = True

                elif line.startswith("    Convergence Criteria"):
                    # for tolerances:
                    # psi4 puts '*' next to converged values and 'o' in place of things that aren't monitored
                    grad = {}

                    dE_tol = line[24:38]
                    if "o" in dE_tol:
                        dE_tol = None
                    else:
                        dE_tol = dE_tol.split()[0]

                    max_f_tol = line[38:52]
                    if "o" in max_f_tol:
                        max_f_tol = None
                    else:
                        max_f_tol = max_f_tol.split()[0]

                    rms_f_tol = line[52:66]
                    if "o" in rms_f_tol:
                        rms_f_tol = None
                    else:
                        rms_f_tol = rms_f_tol.split()[0]

                    max_d_tol = line[66:80]
                    if "o" in max_d_tol:
                        max_d_tol = None
                    else:
                        max_d_tol = max_d_tol.split()[0]

                    rms_d_tol = line[80:94]
                    if "o" in rms_d_tol:
                        rms_d_tol = None
                    else:
                        rms_d_tol = rms_d_tol.split()[0]

                    line = f.readline()
                    line = f.readline()
                    n += 2

                    # for convergence:
                    # psi4 puts '*' next to converged values and 'o' next to things that aren't monitored
                    if dE_tol is not None:
                        dE_conv = line[24:38]
                        dE = float(dE_conv.split()[0])
                        grad["Delta E"] = {}
                        grad["Delta E"]["value"] = dE
                        grad["Delta E"]["converged"] = "*" in dE_conv

                    if max_f_tol is not None:
                        max_f_conv = line[38:52]
                        max_f = float(max_f_conv.split()[0])
                        grad["Max Force"] = {}
                        grad["Max Force"]["value"] = max_f
                        grad["Max Force"]["converged"] = "*" in max_f_conv

                    if rms_f_tol is not None:
                        rms_f_conv = line[52:66]
                        rms_f = float(rms_f_conv.split()[0])
                        grad["RMS Force"] = {}
                        grad["RMS Force"]["value"] = rms_f
                        grad["RMS Force"]["converged"] = "*" in rms_f_conv

                    if max_d_tol is not None:
                        max_d_conv = line[66:80]
                        max_d = float(max_d_conv.split()[0])
                        grad["Max Disp"] = {}
                        grad["Max Disp"]["value"] = max_d
                        grad["Max Disp"]["converged"] = "*" in max_d_conv

                    if rms_d_tol is not None:
                        rms_d_conv = line[80:94]
                        rms_d = float(rms_d_conv.split()[0])
                        grad["RMS Disp"] = {}
                        grad["RMS Disp"]["value"] = rms_d
                        grad["RMS Disp"]["converged"] = "*" in max_d_conv

                    self.other["gradient"] = grad

                elif "Total Gradient" in line:
                    gradient = np.zeros((len(self.atoms), 3))
                    self.skip_lines(f, 2)
                    n += 2
                    for i in range(0, len(self.atoms)):
                        n += 1
                        line = f.readline()
                        info = line.split()
                        gradient[i] = np.array([float(x) for x in info[1:]])

                    self.other["forces"] = -gradient

                elif "SAPT Results" in line:
                    self.skip_lines(f, 1)
                    n += 1
                    while "Total sSAPT" not in line:
                        n += 1
                        line = f.readline()
                        if "---" in line:
                            break
                        if len(line.strip()) > 0:
                            if "Special recipe" in line:
                                continue
                            item = line[:26].strip()
                            val = 1e-3 * float(line[34:47])
                            self.other[item] = val

                elif "SCF energy" in line:
                    self.other["scf_energy"] = float(line.split()[-1])

                elif "correlation energy" in line and "=" in line:
                    item = line.split("=")[0].strip()
                    self.other[item] = float(line.split()[-1])

                elif "Full point group" in line:
                    self.other["full_point_group"] = line.split()[-1]

                elif "Molecular point group" in line:
                    self.other["molecular_point_group"] = line.split()[-1]

                elif (
                    "total energy" in line
                    and "=" in line
                    or re.search("\(.\) energy", line)
                ):
                    item = line.split("=")[0].strip().strip("*").strip()
                    self.other[item] = float(line.split()[-1])
                    # hopefully the highest level energy gets printed last
                    self.other["energy"] = self.other[item]

                elif "Total Energy" in line and "=" in line:
                    item = line.split("=")[0].strip().strip("*").strip()
                    self.other[item] = float(line.split()[-2])
                    # hopefully the highest level energy gets printed last
                    self.other["energy"] = self.other[item]

                elif "Correlation Energy" in line and "=" in line:
                    item = line.split("=")[0].strip().strip("*").strip()
                    if "DFT Exchange-Correlation" in item:
                        self.other[item] = float(line.split()[-1])
                    else:
                        self.other[item] = float(line.split()[-2])

                elif "Ground State -> Excited State Transitions" in line:
                    self.skip_lines(f, 3)
                    n += 3
                    line = f.readline()
                    s = ""
                    while line.strip():
                        s += line
                        n += 1
                        line = f.readline()

                    self.other["uv_vis"] = ValenceExcitations(s, style="psi4")

                elif "Excitation Energy" in line and "Rotatory" in line:
                    self.skip_lines(f, 2)
                    n += 2
                    line = f.readline()
                    s = ""
                    while line.strip():
                        s += line
                        n += 1
                        line = f.readline()

                    self.other["uv_vis"] = ValenceExcitations(s, style="psi4")

                elif re.search("\| State\s*\d+", line):
                    # read energies from property calculation
                    uv_vis += line

                elif "Excited state properties:" in line:
                    # read osc str or rotation from property calculation
                    while line.strip():
                        uv_vis += line
                        n += 1
                        line = f.readline()

                    if "Oscillator" in uv_vis or "Rotation" in uv_vis:
                        self.other["uv_vis"] = ValenceExcitations(uv_vis, style="psi4")

                if "error" not in self.other:
                    for err in ERROR_PSI4:
                        if err in line:
                            self.other["error"] = ERROR_PSI4[err]
                            self.other["error_msg"] = line.strip()

                line = f.readline()
                n += 1

        if get_all:
            if not self.all_geom:
                self.all_geom = []
            self.all_geom += [{
                "atoms": self.atoms,
                "data": self.other,
            }]

        if "error" not in self.other:
            self.other["error"] = None

    def read_orca_out(self, f, get_all=False, just_geom=True, scan_read_all=False):
        """read orca output file"""

        self.all_geom = []
        nrg_regex = re.compile("(?:[A-Za-z]+\s+)?E\((.*)\)\s*\.\.\.\s*(.*)$")
        opt_cycle = re.compile("GEOMETRY OPTIMIZATION CYCLE\s*(\d+)")

        is_scan_job = False
        step_converged = False

        def add_grad(grad, name, line):
            grad[name] = {}
            grad[name]["value"] = line.split()[-3]
            grad[name]["converged"] = line.split()[-1] == "YES"

        def get_atoms(f, n):
            """parse atom info"""
            rv = []
            self.skip_lines(f, 1)
            n += 2
            line = f.readline()
            i = 0
            while line.strip():
                i += 1
                line = line.strip()
                atom_info = line.split()
                element = atom_info[0]
                if element == "-":
                    element = "X"
                coords = np.array([float(x) for x in atom_info[1:]])
                rv += [Atom(element=element, coords=coords, name=str(i))]

                line = f.readline()
                n += 1

            return rv, n

        line = f.readline()
        n = 1
        while line != "":
            if (
                "Psi4: An Open-Source Ab Initio Electronic Structure Package"
                in line
            ):
                self.file_type = "dat"
                return self.read_psi4_out(
                    f, get_all=get_all, just_geom=just_geom
                )

            if (
                "A Quantum Leap Into The Future Of Chemistry"
                in line
            ):
                self.file_type = "qout"
                return self.read_qchem_out(
                    f, get_all=get_all, just_geom=just_geom
                )

            if (
                "Entering Gaussian System"
                in line
            ):
                self.file_type = "log"
                return self.read_log(
                    f, get_all=get_all, just_geom=just_geom, scan_read_all=scan_read_all
                )

            if line.startswith("CARTESIAN COORDINATES (ANGSTROEM)"):
                if is_scan_job and not scan_read_all and step_converged and get_all and len(self.atoms) > 0:
                    if self.all_geom is None:
                        self.all_geom = []
                    self.all_geom += [{
                        "atoms": deepcopy(self.atoms),
                        "data": deepcopy(self.other),
                    }]

                elif (not is_scan_job or scan_read_all) and get_all and len(self.atoms) > 0:
                    if self.all_geom is None:
                        self.all_geom = []
                    self.all_geom += [{
                        "atoms": deepcopy(self.atoms),
                        "data": deepcopy(self.other),
                    }]

                self.atoms, n = get_atoms(f, n)
                step_converged = False

            if just_geom:
                line = f.readline()
                n += 1
                continue
            else:

                nrg = nrg_regex.match(line)
                if nrg is not None:
                    nrg_type = nrg.group(1)
                    # for some reason, ORCA prints MP2 correlation energy
                    # as E(MP2) for CC jobs
                    if nrg_type == "MP2":
                        nrg_type = "MP2 CORR"
                        self.other["E(%s)" % nrg_type] = float(nrg.group(2))

                if line.startswith("FINAL SINGLE POINT ENERGY"):
                    # if the wavefunction doesn't converge, ORCA prints a message next
                    # to the energy so we can't use line.split()[-1]
                    self.other["energy"] = float(line.split()[4])

                if line.startswith("TOTAL SCF ENERGY"):
                    self.skip_lines(f, 2)
                    line = f.readline()
                    n += 3
                    self.other["scf_energy"] = float(line.split()[3])

                elif "E(SOC CIS)" in line:
                    self.other["SOC CIS/TD root energy"] = float(line.split()[3])

                elif "DE(CIS)" in line:
                    self.other["CIS/TD root energy"] = float(line.split()[2])

                elif "Dispersion correction" in line:
                    try:
                        self.other["dispersion correction energy"] = float(line.split()[2])
                    except ValueError:
                        pass

                elif "TOTAL ENERGY:" in line:
                    item = line.split()[-5] + " energy"
                    self.other[item] = float(line.split()[-2])

                elif "CORRELATION ENERGY" in line and "Eh" in line:
                    energy_type = re.search("\s*([\S\s]+) CORRELATION ENERGY", line).group(1)
                    item = energy_type + " correlation energy"
                    self.other[item] = float(line.split()[-2])

                elif re.match("E\(\S+\)\s+...\s+-?\d+\.\d+$", line):
                    nrg = re.match("(E\(\S+\))\s+...\s+(-?\d+\.\d+)$", line)
                    self.other["energy"] = float(nrg.group(2))
                    self.other[nrg.group(1)] = float(nrg.group(2))

                elif "*    Relaxed Surface Scan    *" in line:
                    is_scan_job = True

                elif "THE OPTIMIZATION HAS CONVERGED" in line:
                    step_converged = True

                elif line.startswith("CARTESIAN GRADIENT"):
                    gradient = np.zeros((len(self.atoms), 3))
                    if "NUMERICAL" in line:
                        self.skip_lines(f, 1)
                        n += 1
                    else:
                        self.skip_lines(f, 2)
                        n += 2
                    for i in range(0, len(self.atoms)):
                        n += 1
                        line = f.readline()
                        # orca prints a warning before gradient if some
                        # coordinates are constrained
                        if line.startswith("WARNING:"):
                            continue
                        info = line.split()
                        gradient[i] = np.array([float(x) for x in info[3:]])

                    self.other["forces"] = -gradient

                elif line.startswith("VIBRATIONAL FREQUENCIES"):
                    stage = "frequencies"
                    freq_str = "VIBRATIONAL FREQUENCIES\n"
                    self.skip_lines(f, 2)
                    n += 3
                    line = f.readline()
                    hit = {
                        "modes": False,
                        "spectrum": False,
                    }
                    while not (stage == "THERMO" and line == "\n") and line:
                        if "--" not in line and line != "\n":
                            freq_str += line

                        if "NORMAL MODES" in line:
                            stage = "modes"
                            hit["modes"] = True
                            self.skip_lines(f, 6)
                            n += 6

                        if "RAMAN SPECTRUM" in line:
                            stage = "RAMAN"
                            self.skip_lines(f, 2)
                            n += 2

                        if "IR SPECTRUM" in line:
                            if not hit["modes"]:
                                break
                            hit["spectrum"] = True
                            stage = "IR"
                            self.skip_lines(f, 2)
                            n += 2

                        if "THERMOCHEMISTRY" in line:
                            stage = "THERMO"

                        n += 1
                        line = f.readline()
                    
                    if all(hit.values()):
                        self.other["frequency"] = Frequency(
                            freq_str, hpmodes=False, style="orca", atoms=self.atoms,
                        )

                elif line.startswith("CHEMICAL SHIFTS"):
                    nmr_data = []
                    while line:
                        nmr_data.append(line)
                        n += 1
                        line = f.readline()
                        if "Maximum memory used throughout the entire EPRNMR-calculation:" in line:
                            break
                    self.other["nmr"] = NMR("".join(nmr_data), style="orca")

                elif line.startswith("Temperature"):
                    self.other["temperature"] = float(line.split()[2])

                elif line.startswith("Total Mass"):
                    # this may only get printed for freq jobs
                    self.other["mass"] = float(line.split()[3])
                    self.other["mass"] *= UNIT.AMU_TO_KG

                elif line.startswith(" Total Charge"):
                    self.other["charge"] = int(line.split()[-1])

                elif line.startswith(" Multiplicity"):
                    self.other["multiplicity"] = int(line.split()[-1])

                elif "rotational symmetry number" in line:
                    # TODO: make this cleaner
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-2]
                    )

                elif "Symmetry Number:" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1]
                    )

                elif line.startswith("Zero point energy"):
                    self.other["ZPVE"] = float(line.split()[4])

                elif line.startswith("Total Enthalpy"):
                    self.other["enthalpy"] = float(line.split()[3])

                elif line.startswith("Final Gibbs"):
                    # NOTE - Orca seems to only print Grimme's Quasi-RRHO free energy
                    # RRHO can be computed in AaronTool's CompOutput by setting the w0 to 0
                    self.other["free_energy"] = float(line.split()[5])

                elif line.startswith("Rotational constants in cm-1:"):
                    # orca doesn't seem to print rotational constants in older versions
                    self.other["rotational_temperature"] = [
                        float(x) for x in line.split()[-3:]
                    ]
                    self.other["rotational_temperature"] = [
                        x
                        * PHYSICAL.SPEED_OF_LIGHT
                        * PHYSICAL.PLANCK
                        / PHYSICAL.KB
                        for x in self.other["rotational_temperature"]
                    ]

                elif "Point Group:" in line:
                    self.other["full_point_group"] = line.split()[2][:-1]

                elif "Symmetry Number" in line:
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-1]
                    )

                elif "sn is the rotational symmetry number" in line:
                    # older versions of orca print this differently
                    self.other["rotational_symmetry_number"] = int(
                        line.split()[-2]
                    )

                elif "Geometry convergence" in line:
                    grad = {}
                    self.skip_lines(f, 2)
                    n += 3
                    line = f.readline()
                    while line and re.search("\w", line):
                        if re.search("Energy\schange", line):
                            add_grad(grad, "Delta E", line)
                        elif re.search("RMS\sgradient", line):
                            add_grad(grad, "RMS Force", line)
                        elif re.search("MAX\sgradient", line):
                            add_grad(grad, "Max Force", line)
                        elif re.search("RMS\sstep", line):
                            add_grad(grad, "RMS Disp", line)
                        elif re.search("MAX\sstep", line):
                            add_grad(grad, "Max Disp", line)

                        line = f.readline()
                        n += 1

                    self.other["gradient"] = grad

                elif "MAYER POPULATION ANALYSIS" in line:
                    self.skip_lines(f, 2)
                    n += 2
                    line = f.readline()
                    data = dict()
                    headers = []
                    while line.strip():
                        info = line.split()
                        header = info[0]
                        name = " ".join(info[2:])
                        headers.append(header)
                        data[header] = (name, [])
                        line = f.readline()
                    self.skip_lines(f, 1)
                    n += 1
                    for i in range(0, len(self.atoms)):
                        line = f.readline()
                        info = line.split()[2:]
                        for header, val in zip(headers, info):
                            data[header][1].append(float(val))

                    for header in headers:
                        self.other[data[header][0]] = np.array(data[header][1])

                elif line.startswith("LOEWDIN ATOMIC CHARGES"):
                    self.skip_lines(f, 1)
                    n += 1
                    charges = np.zeros(len(self.atoms))
                    for i in range(0, len(self.atoms)):
                        line = f.readline()
                        n += 1
                        charges[i] = float(line.split()[-1])
                    self.other["Löwdin Charges"] = charges

                elif line.startswith("MULLIKEN ATOMIC CHARGES"):
                    self.skip_lines(f, 1)
                    n += 1
                    charges = np.zeros(len(self.atoms))
                    for i in range(0, len(self.atoms)):
                        line = f.readline()
                        n += 1
                        charges[i] = float(line.split()[-1])
                    self.other["Mulliken Charges"] = charges

                elif line.startswith("BASIS SET IN INPUT FORMAT"):
                    # read basis set primitive info
                    self.skip_lines(f, 3)
                    n += 3
                    line = f.readline()
                    n += 1
                    self.other["basis_set_by_ele"] = dict()
                    while "--" not in line and line != "":
                        members = re.search("Members:([\S\s]+)", line)
                        new_gto = re.search("NewGTO\s+(\S+)", line)
                        ele = None
                        names = None
                        if new_gto:
                            ele = new_gto.group(1)
                        elif members:
                            names = map(int, members.group(1).split()[1::2])
                            line = f.readline()
                            n += 1

                        if new_gto or members:
                            line = f.readline()
                            n += 1
                            primitives = []
                            while "end" not in line and line.strip():
                                shell_type, n_prim = line.split()
                                n_prim = int(n_prim)
                                exponents = []
                                con_coeffs = []
                                for i in range(0, n_prim):
                                    line = f.readline()
                                    n += 1
                                    info = line.split()
                                    exponent = float(info[1])
                                    con_coeff = [float(x) for x in info[2:]]
                                    exponents.append(exponent)
                                    con_coeffs.extend(con_coeff)
                                primitives.append(
                                    (
                                        shell_type,
                                        n_prim,
                                        exponents,
                                        con_coeffs,
                                    )
                                )
                                line = f.readline()
                                n += 1
                            if ele:
                                self.other["basis_set_by_ele"][ele] = primitives
                            else:
                                for name in names:
                                    self.other["basis_set_by_ele"][name] = primitives
                        line = f.readline()
                        n += 1

                elif "Basis Dimension" in line:
                    self.other["n_basis"] = int(line.split()[-1])

                elif "EXCITED STATES" in line or re.search("STEOM.* RESULTS", line) or line.startswith("APPROXIMATE EOM LHS"):
                    s = ""
                    done = False
                    while not done:
                        s += line
                        n += 1
                        line = f.readline()
                        if (
                            "ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR" in line or
                            re.search("TDM done", line) or
                            "TIMINGS" in line or
                            line == ""
                        ):
                            done = True
                        if "SOC stabilization of the ground state" in line:
                            self.other["SOC GS stabilization energy"] = float(line[39:48]) / UNIT.HART_TO_EV
                        if "CALCULATED SOCME BETWEEN" in line:
                            # SOC in cm^-1
                            self.skip_lines(f, 4)
                            n += 5
                            line = f.readline()
                            soc_x = []
                            soc_y = []
                            soc_z = []
                            while line.strip():
                                re_z = float(line[23:30])
                                im_z = float(line[32:40])
                                soc_z.append(complex(re_z, im_z))
                                re_x = float(line[47:54])
                                im_x = float(line[56:64])
                                soc_x.append(complex(re_x, im_x))
                                re_y = float(line[71:78])
                                im_y = float(line[80:88])
                                soc_y.append(complex(re_y, im_y))
                                line = f.readline()
                                n += 1
                            # determine number of roots
                            roots = int(np.sqrt(1 + 4 * len(soc_x)) - 1)
                            # +1 for ground state
                            n_gs = int(roots / 2 + 1)
                            n_flipped = int(roots / 2)
                            socme_dim = n_gs + n_flipped
                            self.other["soc x"] = np.zeros((socme_dim, socme_dim), dtype=complex)
                            self.other["soc y"] = np.zeros((socme_dim, socme_dim), dtype=complex)
                            self.other["soc z"] = np.zeros((socme_dim, socme_dim), dtype=complex)
                            ndx = 0
                            for i in range(0, n_flipped):
                                for j in range(0, n_gs):
                                    self.other["soc x"][n_gs + i, j] = soc_x[ndx]
                                    self.other["soc x"][j, n_gs + i] = soc_x[ndx]
                                    self.other["soc y"][n_gs + i, j] = soc_y[ndx]
                                    self.other["soc y"][j, n_gs + i] = soc_y[ndx]
                                    self.other["soc z"][n_gs + i, j] = soc_z[ndx]
                                    self.other["soc z"][j, n_gs + i] = soc_z[ndx]
                                    ndx += 1

                            self.other["soc (cm^-1)"] = np.sqrt(
                                self.other["soc x"].real ** 2 + self.other["soc x"].imag ** 2 +
                                self.other["soc y"].real ** 2 + self.other["soc y"].imag ** 2 +
                                self.other["soc z"].real ** 2 + self.other["soc z"].imag ** 2
                            )

                    self.other["uv_vis"] = ValenceExcitations(s, style="orca")

                if "INPUT FILE" in line:
                    try:
                        coord_match = re.compile("\*\s*(xyz|int|gzmat)\s+-?\d+\s+(\d+)", re.IGNORECASE)
                        while "****END OF INPUT****" not in line:
                            line = f.readline()
                            n += 1
                            if "mult" in line.lower():
                                self.other["multiplicity"] = int(float(line.split()[-1]))
                            if coord_match.search(line):
                                self.other["multiplicity"] = int(coord_match.search(line).group(2))

                    except ValueError:
                        pass

                elif line.startswith("MOLECULAR ORBITALS"):
                    # read molecular orbitals
                    self.skip_lines(f, 1)
                    n += 1
                    line = f.readline()
                    at_info = re.compile(
                        "\s*(\d+)\S+\s+\d+(?:s|p[xyz]|d(?:z2|xz|yz|x2y2|xy)|[fghi][\+\-]?\d+)"
                    )
                    args = [
                        ("alpha_coefficients", "beta_coefficients"),
                        ("alpha_nrgs", "beta_nrgs"),
                        ("alpha_occupancies", "beta_occupancies"),
                    ]

                    for coeff_name, nrg_name, occ_name in zip(*args):
                        if not line.strip():
                            break
                        self.other[coeff_name] = []
                        self.other[nrg_name] = []
                        self.other[occ_name] = []
                        self.other["shell_to_atom"] = []
                        mo_coefficients = []
                        orbit_nrgs = []
                        occupancy = []
                        while line.strip() != "":
                            at_match = at_info.match(line)
                            if at_match:
                                ndx = int(at_match.group(1))
                                self.other["shell_to_atom"].append(ndx)
                                coeffs = []
                                # there might not always be a space between the coefficients
                                # so we can't just split(), but they are formatted(-ish)
                                for coeff in re.findall("-?\d+\.\d\d\d\d\d\d", line[16:]):
                                    coeffs.append(float(coeff))
                                for coeff, mo in zip(coeffs, mo_coefficients):
                                    mo.append(coeff)
                            elif "--" not in line:
                                orbit_nrgs = occupancy
                                occupancy = [float(x) for x in line.split()]
                            elif "--" in line:
                                self.other[nrg_name].extend(orbit_nrgs)
                                self.other[occ_name].extend(occupancy)
                                if mo_coefficients:
                                    self.other[coeff_name].extend(
                                        mo_coefficients
                                    )
                                    if not all(
                                        len(coeff) == len(mo_coefficients[0])
                                        for coeff in mo_coefficients
                                    ):
                                        self.LOG.warning(
                                            "orbital coefficients may not "
                                            "have been parsed correctly"
                                        )
                                mo_coefficients = [[] for x in orbit_nrgs]
                                orbit_nrgs = []
                            line = f.readline()
                            n += 1
                        self.other[coeff_name].extend(mo_coefficients)
                        line = f.readline()
                        # line = f.readline()

                elif line.startswith("N(Alpha)  "):
                    self.other["n_alpha"] = int(
                        np.rint(float(line.split()[2]))
                    )

                elif opt_cycle.search(line):
                    self.other["opt_steps"] = int(opt_cycle.search(line).group(1))

                elif line.startswith("N(Beta)  "):
                    self.other["n_beta"] = int(np.rint(float(line.split()[2])))

                elif line.startswith("OVERLAP MATRIX"):
                    self.skip_lines(f, 1)
                    n += 1
                    n_blocks = int(np.ceil(self.other["n_basis"] / 6))
                    ao_overlap = np.zeros((self.other["n_basis"], self.other["n_basis"]))

                    for i in range(0, n_blocks):
                        line = f.readline()
                        n += 1
                        start = 6 * i
                        stop = min(6 * (i + 1), self.other["n_basis"])
                        for j in range(0, self.other["n_basis"]):
                            line = f.readline()
                            n += 1
                            data = [float(x) for x in line.split()[1:]]
                            ao_overlap[j, start:stop] = data

                    self.other["ao_overlap"] = ao_overlap

                elif ORCA_NORM_FINISH in line:
                    self.other["finished"] = True

                # TODO E_ZPVE
                if "error" not in self.other or not self["error"]:
                    for err in ERROR_ORCA:
                        if err in line:
                            self.other["error"] = ERROR_ORCA[err]
                            self.other["error_msg"] = line.strip()
                            break
                    else:
                        self.other["error"] = False
                        if "!!!!!!!!" in line:
                            line = f.readline()
                            n += 1
                            self.other["error"] = line.strip()
                            line = f.readline()
                            n += 1
                            self.other["error_msg"] = ""
                            found_error = False
                            while "!!!!!!!" not in line:
                                for err in ERROR_ORCA:
                                    if err in line:
                                        self.other["error"] = ERROR_ORCA[err]
                                self.other["error_msg"] += line.strip() + "\n"
                                line = f.readline()
                                n += 1
                            if "REBUILDING A NEW SET OF INTERNALS" in self.other["error_msg"]:
                                del self.other["error"]
                                del self.other["error_msg"]

                line = f.readline()
                n += 1

        if not just_geom:
            if "finished" not in self.other:
                self.other["finished"] = False

            if (
                "alpha_coefficients" in self.other
                and "basis_set_by_ele" in self.other
            ):
                self.other["orbitals"] = Orbitals(self)

        if get_all:
            self.all_geom += [{
                "atoms": self.atoms,
                "data": self.other,
            }]

        if "error" not in self.other:
            self.other["error"] = None

    def read_qchem_out(self, f, get_all=False, just_geom=True):
        """read qchem output file"""
        def get_atoms(f, n):
            """parse atom info"""
            rv = []
            self.skip_lines(f, 2)
            n += 1
            line = f.readline()
            i = 0
            while "--" not in line:
                i += 1
                line = line.strip()
                atom_info = line.split()
                element = atom_info[1]
                coords = np.array([float(x) for x in atom_info[2:5]])
                rv += [Atom(element=element, coords=coords, name=str(i))]

                line = f.readline()
                n += 1

            return rv, n

        def add_grad(grad, name, line):
            grad[name] = {}
            grad[name]["value"] = line.split()[-3]
            grad[name]["converged"] = line.split()[-1] == "YES"

        line = f.readline()
        n = 1
        while line != "":
            if (
                "Psi4: An Open-Source Ab Initio Electronic Structure Package"
                in line
            ):
                self.file_type = "dat"
                return self.read_psi4_out(
                    f, get_all=get_all, just_geom=just_geom
                )

            if "* O   R   C   A *" in line:
                self.file_type = "out"
                return self.read_orca_out(
                    f, get_all=get_all, just_geom=just_geom
                )


            if (
                "A Quantum Leap Into The Future Of Chemistry"
                in line
            ):
                self.file_type = "qout"
                return self.read_qchem_out(
                    f, get_all=get_all, just_geom=just_geom
                )

            if "Standard Nuclear Orientation (Angstroms)" in line:
                if get_all and len(self.atoms) > 0:
                    if self.all_geom is None:
                        self.all_geom = []
                    self.all_geom += [{
                        "atoms": deepcopy(self.atoms),
                        "data": deepcopy(self.other),
                    }]

                self.atoms, n = get_atoms(f, n)

            if just_geom:
                line = f.readline()
                n += 1
                continue
            else:
                if "energy in the final basis set" in line:
                    self.other["energy"] = float(line.split()[-1])
                    if "SCF" in line:
                        self.other["scf_energy"] = self.other["energy"]

                if re.search(r"energy\s+=\s+-?\d+\.\d+", line):
                    info = re.search(r"\s*([\S\s]+)\s+energy\s+=\s+(-?\d+\.\d+)", line)
                    kind = info.group(1)
                    if len(kind.split()) <= 2:
                        val = float(info.group(2))
                        if "correlation" not in kind and len(kind.split()) <= 2:
                            self.other["E(%s)" % kind.split()[0]] = val
                            self.other["energy"] = val
                        else:
                            self.other["E(corr)(%s)" % kind.split()[0]] = val

                if "Total energy:" in line:
                    self.other["energy"] = float(line.split()[-2])

                #MPn energy is printed as EMPn(SDQ)
                if re.search("EMP\d(?:[A-Z]+)?\s+=\s*-?\d+.\d+$", line):
                    self.other["energy"] = float(line.split()[-1])
                    self.other["E(%s)" % line.split()[0][1:]] = self.other["energy"]

                if "Molecular Point Group" in line:
                    self.other["full_point_group"] = line.split()[3]

                if "Largest Abelian Subgroup" in line:
                    self.other["abelian_subgroup"] = line.split()[3]

                if "Ground-State Mulliken Net Atomic Charges" in line:
                    charges = []
                    self.skip_lines(f, 3)
                    n += 2
                    line = f.readline()
                    while "--" not in line:
                        charge = float(line.split()[-1])
                        charges.append(charge)
                        line = f.readline()
                        n += 1

                    self.other["Mulliken Charges"] = charges

                if "Cnvgd?" in line:
                    grad = {}
                    line = f.readline()
                    while line and re.search("\w", line):
                        if re.search("Energy\schange", line):
                            add_grad(grad, "Delta E", line)
                        elif re.search("Displacement", line):
                            add_grad(grad, "Disp", line)
                        elif re.search("Gradient", line):
                            add_grad(grad, "Max Disp", line)

                        line = f.readline()
                        n += 1

                    self.other["gradient"] = grad

                if "VIBRATIONAL ANALYSIS" in line:
                    freq_str = ""
                    self.skip_lines(f, 10)
                    n += 9
                    line = f.readline()
                    while "STANDARD THERMODYNAMIC QUANTITIES" not in line:
                        n += 1
                        freq_str += line
                        line = f.readline()
                    self.other["frequency"] = Frequency(
                        freq_str, style="qchem", atoms=self.atoms,
                    )
                    self.other["temperature"] = float(line.split()[4])

                if "Rotational Symmetry Number is" in line:
                    self.other["rotational_symmetry_number"] = int(line.split()[-1])

                if "Molecular Mass:" in line:
                    self.other["mass"] = float(line.split()[-2]) * UNIT.AMU_TO_KG

                if "$molecule" in line.lower():
                    line = f.readline()
                    while "$end" not in line.lower() and line:
                        if re.search("\d+\s+\d+", line):
                            match = re.search("^\s*(\d+)\s+(\d+)\s*$", line)
                            self.other["charge"] = int(match.group(1))
                            self.other["multiplicity"] = int(match.group(2))
                            break
                        line = f.readline()

                if "Principal axes and moments of inertia" in line:
                    self.skip_lines(f, 1)
                    line = f.readline()
                    rot_consts = np.array([
                        float(x) for x in line.split()[2:]
                    ])
                    rot_consts *= UNIT.AMU_TO_KG
                    rot_consts *= UNIT.A0_TO_BOHR ** 2
                    rot_consts *= 1e-20
                    rot_consts = PHYSICAL.PLANCK ** 2 / (8 * np.pi ** 2 * rot_consts * PHYSICAL.KB)

                    self.other["rotational_temperature"] = rot_consts

                if line.startswith("Mult"):
                    self.other["multiplicity"] = int(line.split()[1])

                # TD-DFT excitations
                if re.search("TDDFT.* Excitation Energies", line):
                    excite_s = ""
                    self.skip_lines(f, 2)
                    line = f.readline()
                    n += 3
                    while "---" not in line and line:
                        excite_s += line
                        line = f.readline()
                        n += 1

                    self.other["uv_vis"] = ValenceExcitations(
                        excite_s, style="qchem",
                    )

                # ADC excitations
                if re.search("Excited State Summary", line):
                    excite_s = ""
                    self.skip_lines(f, 2)
                    line = f.readline()
                    n += 3
                    while "===" not in line and line:
                        excite_s += line
                        line = f.readline()
                        n += 1

                    self.other["uv_vis"] = ValenceExcitations(
                        excite_s, style="qchem",
                    )

                # EOM excitations
                if re.search("Start computing the transition properties", line):
                    excite_s = ""
                    line = f.readline()
                    n += 1
                    while "All requested transition properties have been computed" not in line and line:
                        excite_s += line
                        line = f.readline()
                        n += 1

                    self.other["uv_vis"] = ValenceExcitations(
                        excite_s, style="qchem",
                    )

                if line.startswith(" Gradient of SCF Energy"):
                    # why on earth do they print the gradient like this
                    gradient = np.zeros((len(self.atoms), 3))
                    n_blocks = int(np.ceil(len(self.atoms) / 6))
                    # printed in groups of up to six atoms
                    # there are 3 rows in each block, one for each of X, Y, and Z
                    # component of the gradient
                    # each column is for one of the (up to) six atoms
                    for i in range(0, n_blocks):
                        self.skip_lines(f, 1)
                        n += 1
                        # start and stop are atom index range for this block
                        start = 6 * i
                        stop = min(6 * (i + 1) , len(self.atoms))
                        for j in range(0, 3):
                            line = f.readline()
                            n += 1
                            dx = [float(x) for x in line.split()[1:]]
                            for k, l in enumerate(range(start, stop)):
                                gradient[l, j] = dx[k]

                    self.other["forces"] = -gradient

                if "Thank you very much for using Q-Chem" in line:
                    self.other["finished"] = True

                line = f.readline()
                n += 1

        if not just_geom and "finished" not in self.other:
            self.other["finished"] = False

        if get_all:
            self.all_geom += [{
                "atoms": deepcopy(self.atoms),
                "data": deepcopy(self.other),
            }]

        if "error" not in self.other:
            self.other["error"] = None

    def read_log(self, f, get_all=False, just_geom=True, scan_read_all=False):
        """read gaussian output file"""
        isotope = re.compile(r" Atom\s+(\d+) has atomic number")
        # orientation = re.compile(r"(Standard|Input) orientation:\s*$")
        # zmat = re.compile(r"Symbolic Z-matrix")
        # modred = re.compile(r"The following ModRedundant input section has been read:")
        # optimized_parameters = re.compile(r"!   Optimized Parameters   !")
        most_energies = re.compile(r"\s+(E\(\S+\))\s*=\s*(\S+)")
        mp_energies = re.compile(r"([RU]MP\d+(?:\(\S+\))?)\s*=\s*(\S+)")
        # temperature = re.compile(r"^ Temperature\s*\d+\.\d+")
        
        def get_atoms(f, n):
            rv = self.atoms
            self.skip_lines(f, 4)
            line = f.readline()
            n += 5
            atnum = 0
            while "--" not in line:
                line = line.strip()
                line = line.split()
                for l in line:
                    try:
                        float(l)
                    except ValueError:
                        msg = "Error detected with log file on line {}"
                        raise IOError(msg.format(n))
                try:
                    rv[atnum].coords = np.array(line[3:], dtype=float)
                except IndexError:
                    pass
                    rv.append(Atom(
                        element=ELEMENTS[int(line[1])],
                        name=str(atnum + 1),
                    ))
                    rv[atnum].coords = np.array(line[3:], dtype=float)

                atnum += 1
                line = f.readline()
                n += 1
            return rv, n

        def set_angle(atom1, atom2, atom3, target_angle):
            b1 = atom1.coords - atom2.coords
            b2 = atom3.coords - atom2.coords
            current_angle = angle_between_vectors(b1, b2)
            d_theta = target_angle - current_angle
            coords = np.array([
                b1,
                [0, 0, 0],
                b2,
            ])
            v = perp_vector(coords)
            R = rotation_matrix(d_theta, v)
            atom3.coords -= atom2.coords
            atom3.coords = np.dot(R, atom3.coords)
            atom3.coords += atom2.coords
            b1 = atom1.coords - atom2.coords
            b2 = atom3.coords - atom2.coords
            current_angle = angle_between_vectors(b1, b2)
            return abs(current_angle - target_angle) < 1e-5

        def set_torsion(atom1, atom2, atom3, atom4, target_angle):
            b1 = atom4.coords - atom3.coords
            b2 = atom3.coords - atom2.coords
            b3 = atom2.coords - atom1.coords
            v1 = np.cross(b1, b2)
            v2 = np.cross(b2, b3)
            current_angle = -np.arctan2(
                np.dot(b2, np.cross(v1, v2)),
                np.linalg.norm(b2) * np.dot(v1, v2)
            )
            da = target_angle - current_angle
            R = rotation_matrix(-da, atom2.coords - atom3.coords)
            atom4.coords -= atom3.coords
            atom4.coords = np.dot(R, atom4.coords)
            atom4.coords += atom3.coords
            b1 = atom4.coords - atom3.coords
            b2 = atom3.coords - atom2.coords
            b3 = atom2.coords - atom1.coords
            v1 = np.cross(b1, b2)
            v2 = np.cross(b2, b3)
            current_angle = -np.arctan2(
                np.dot(b2, np.cross(v1, v2)),
                np.linalg.norm(b2) * np.dot(v1, v2)
            )
            return abs(current_angle - target_angle) < 1e-5

        def get_input(f, n):
            rv = []
            line = f.readline()
            n += 1
            match = re.search(
                "Charge\s*=\s*(-?\d+)\s*Multiplicity\s*=\s*(\d+)", line
            )
            if match is not None:
                self.other["charge"] = int(match.group(1))
                self.other["multiplicity"] = int(match.group(2))
            line = f.readline()
            n += 1
            a = 0
            info = line.split()
            if len(line.split()) == 1 or any(info[1] == x for x in ["0", "-1"]):
                # parse z matrix input
                # z matrices can have variables, we'll need to interpolate those
                variables = dict()
                reading_molecule = True
                molecule_data = ""
                while line.split():
                    if "Variables:" in line or "Constants:" in line:
                        # switch from reading molecule to reading variables
                        # variable list comes after atoms
                        reading_molecule = False

                    if reading_molecule:
                        molecule_data += line
                    else:
                        try:
                            var, value = line.split()
                            variables[var] = value
                        except ValueError:
                            pass

                    line = f.readline()

                for i, line in enumerate(molecule_data.splitlines()):
                    # get coordinates for molecule
                    a += 1
                    coords = np.zeros(3)
                    info = line.split()
                    element = info[0].split("(")[0].split("-")[0]
                    if not element.isdigit():
                        element = element.rstrip("1234567890")
                    else:
                        element = ELEMENTS[int(element)]
                    rv += [
                        Atom(
                            element=element,
                            name=str(a),
                            coords=coords,
                        )
                    ]


                    # first atom, leave it at the origin
                    if len(info) == 1:
                        continue

                    bond_ndx = int(info[1])
                    # if the first thing is > 0, it defines a distance
                    # between this atom and the bond_ndx atom
                    # otherwise, it is cartesian coordinates
                    if bond_ndx <= 0:
                        # if bond_ndx < 0:
                        #     rv[-1].flag = True
                        # if the number after the element is 0, then
                        # the input is cartesian coordinates
                        coords = info[2:]
                        for j, coord in enumerate(coords):
                            try:
                                coords[j] = variables[coord]
                            except KeyError:
                                pass
                        coords = [float(coord) for coord in coords]
                        rv[-1].coords = np.array(coords)
                        continue

                    bond_ndx -= 1
                    rv[-1].coords += rv[bond_ndx].coords
                    bond_length = info[2]
                    sign = 1
                    # interpolate variable
                    try:
                        bond_length = variables[bond_length]
                    except KeyError:
                        try:
                            # these might have a minus sign
                            bond_length = variables[bond_length.lstrip("-")]
                            sign = -1
                        except KeyError:
                            pass
                    # convert text to float and fix sign
                    bond_length = sign * float(bond_length)
                    rv[-1].coords[2] += bond_length
                    # only bond length was defined (i.e. 2nd atom in z matrix input)
                    # go on to next atom
                    if len(info) < 4:
                        continue

                    # similar process for angles
                    angle_ndx = int(info[3]) - 1
                    target_angle = info[4]
                    sign = 1
                    try:
                        target_angle = variables[target_angle]
                    except KeyError:
                        try:
                            target_angle = variables[target_angle.lstrip("-")]
                            sign = -1
                        except KeyError:
                            pass
                    target_angle = sign * float(target_angle)
                    target_angle = np.deg2rad(target_angle)
                    angle_set = False
                    # for whatever reason, this doesn't work on the first try a lot
                    j = 0
                    while not angle_set and j < 10:
                        angle_set = set_angle(rv[angle_ndx], rv[bond_ndx], rv[-1], target_angle)
                        j += 1

                    # if there's no torsion info, go to next atom
                    if len(info) < 6:
                        continue

                    coord_code = 0
                    if len(info) >= 8:
                        coord_code = int(info[7])

                    # next entry can either be a 1-2-3 angle or a 1-2-3-4 angle
                    if coord_code == 1:
                        # coordinates are a bond length and two 1-2-3 angles
                        angle_ndx = int(info[5]) - 1
                        v = perp_vector([rv[-1].coords, rv[bond_ndx].coords, rv[angle_ndx].coords])
                        current_angle = rv[bond_ndx].angle(rv[-1], rv[angle_ndx])
                        target_angle = info[6]
                        try:
                            target_angle = variables[target_angle]
                        except KeyError:
                            try:
                                target_angle = variables[target_angle.lstrip("-")]
                                sign = -1
                            except KeyError:
                                pass
                        target_angle = sign * float(target_angle)
                        target_angle = np.deg2rad(target_angle)
                        angle_set = False
                        # for whatever reason, this doesn't work on the first try a lot
                        j = 0
                        while not angle_set and j < 10:
                            angle_set = set_angle(rv[angle_ndx], rv[bond_ndx], rv[-1], target_angle)
                            j += 1
                        continue

                    # 1-2-3-4 angle
                    torsion_ndx = int(info[5]) - 1

                    target_angle = info[6]
                    try:
                        target_angle = variables[target_angle]
                    except KeyError:
                        try:
                            target_angle = variables[target_angle.lstrip("-")]
                            sign = -1
                        except KeyError:
                            pass
                    target_angle = sign * float(target_angle)
                    target_angle = np.deg2rad(target_angle)

                    angle_set = False
                    # this seems to always work on the first try
                    j = 0
                    while not angle_set and j < 10:
                        angle_set = set_torsion(
                            rv[torsion_ndx],
                            rv[angle_ndx],
                            rv[bond_ndx],
                            rv[-1],
                            target_angle,
                        )
                        j += 1


                # print(len(rv))
                # print("")
                # for a in rv:
                #     print(a.element, *a.coords)

                return rv, n

            while len(line.split()) > 1:
                line  = line.split()
                element = line[0].split("(")[0].split("-")[0]
                if not element.isdigit():
                    element = element.rstrip("1234567890")
                else:
                    element = ELEMENTS[int(element)]
                if len(line) == 5:
                    flag = not bool(line[1])
                    a += 1
                    rv += [Atom(element=element, flag=flag, coords=line[2:], name=str(a))]
                elif len(line) == 4:
                    a += 1
                    rv += [Atom(element=element, coords=line[1:], name=str(a))]
                line = f.readline()
                n += 1
            return rv, n

        def get_oniom_atoms(f, n):
            rv = self.atoms
            self.skip_lines(f, 4)
            line = f.readline()
            n += 5
            atnum = 0
            while "--" not in line:
                line = line.strip()
                line = line.split()
                for l in line:
                    try:
                        float(l)
                    except ValueError:
                        msg = "Error detected with log file on line {}"
                        raise IOError(msg.format(n))
                rv[atnum].coords = np.array(line[3:], dtype=float)
                atnum += 1
                line = f.readline()
                n += 1
            return rv, n

        def get_oniom_info(f, n):
            rv = []
            line = f.readline()
            n += 1
            charge = []
            multiplicity = []
            while "Charge" in line:
                match = re.search(
                    "Charge\s*=\s*(-?\d+)\s*Multiplicity\s*=\s*(\d+)", line
                )
                if match is not None:
                    charge.append(int(match.group(1)))
                    multiplicity.append(int(match.group(2)))
                line = f.readline()
                n += 1
            self.other["charge"] = charge
            self.other["multiplicity"] = multiplicity
            while len(line.split()) > 0:
                nums = float_num.findall(line)
                line = line.split()
                is_oniom = False
                flag = ""
                atomtype = ""
                charge = ""
                #tags = []
                link_info = {}
                has_flag = False
                if len(line[0].split("-")) == 2:
                    if not is_alpha(line[0].split("-")[1][0]):
                        charge = nums[0]
                    elif is_alpha(line[0].split("-")[1][0]):
                        atomtype = line[0].split("-")[1]
                if len(line[0].split("-")) == 4:
                    atomtype = line[0].split("-")[1]
                    charge = str(-1 * float(line[0].split("-")[3]))
                if len(line[0].split("-")) == 3:
                    if not is_alpha(line[0].split("-")[1][0]):
                        charge = nums[0]
                    elif is_alpha(line[0].split("-")[1][0]):
                        atomtype = line[0].split("-")[1]
                        charge = line[0].split("-")[2]
                if len(line)%2 == 0:
                    has_flag = True
                    flag = line[1]
                    coords = line[2:5]
                if not has_flag:
                    coords = line[1:4]
                if len(line) > 6:
                    link_atom = line[len(line)-2:].split()
                    link_info["connected"] = link_atom[1]
                    info = link_atom[0].split("-")
                    link_info["element"] = info[0]
                    if len(info) == 3:
                        link_info["atomtype"] = info[1]
                        link_info["charge"] = info[2]
                    elif len(info) == 2:
                        if is_alpha(info[1][0]):
                            link_info["atomtype"] = info[1]
                        else:
                            link_info["charge"] = info[1]
                    #tags.append(line[len(line)-2:])
                    layer = line[len(line)-3]
                if len(line) < 7:
                    layer = line[len(line)-1]
                a = OniomAtom(element=line[0].split("-")[0],flag=flag,coords=coords,layer=layer,atomtype=atomtype,charge=charge,link_info=link_info)
                rv += [a]
                line = f.readline()
                n += 1
            return rv, n

        def get_params(f, n):
            rv = []
            self.skip_lines(f, 2)
            n += 3
            line = f.readline()
            if "Definition" in line:
                definition = True
            else:
                definition = False
            self.skip_lines(f, 1)
            n += 2
            line = f.readline()
            while "--" not in line:
                line = line.split()
                param = line[1]
                if definition:
                    val = float(line[3])
                else:
                    val = float(line[2])
                rv.append((param, val))
                line = f.readline()
                n += 1
            return rv, n

        def get_modredundant(f, n):
            """read constraints for modredundant section"""
            rv = {}
            line = f.readline()
            n += 1
            while line.strip():
                atom_match = re.search("X\s+(\d+)\s+([FS])(\s+(\d+)\s+(\d+\.\d?))?", line)
                bond_match = re.search("B\s+(\d+)\s+(\d+)\s+([FS])(\s+(\d+)\s+(\d+\.\d?))?", line)
                angle_match = re.search("A\s+(\d+)\s+(\d+)\s+(\d+)\s+([FS])(\s+(\d+)\s+(\d+\.\d?))?", line)
                torsion_match = re.search(
                    "D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([FS])(\s+(\d+)\s+(\d+\.\d?))?", line
                )
                if atom_match:
                    if "atoms" not in rv:
                        rv["atoms"] = ""
                    else:
                        rv["atoms"] += ","
                    rv["atoms"] += atom_match.group(1)
                    if atom_match.group(2) == "S":
                        if "scan" not in rv:
                            rv["scan"] = []
                        rv["scan"].append(",".join([atom_match.group(1), atom_match.group(4), atom_match.group(5)]))
                elif bond_match:
                    if "bonds" not in rv:
                        rv["bonds"] = []
                    rv["bonds"].append(
                        ",".join([bond_match.group(1), bond_match.group(2)])
                    )
                    if bond_match.group(3) == "S":
                        if "scan" not in rv:
                            rv["scan"] = []
                        rv["scan"].append(",".join([bond_match.group(1), bond_match.group(2), bond_match.group(5), bond_match.group(6)]))
                elif angle_match:
                    if "angles" not in rv:
                        rv["angles"] = []
                    rv["angles"].append(
                        ",".join(
                            [
                                angle_match.group(1),
                                angle_match.group(2),
                                angle_match.group(3),
                            ]
                        )
                    )
                    if angle_match.group(4) == "S":
                        if "scan" not in rv:
                            rv["scan"] = []
                        rv["scan"].append(",".join([angle_match.group(1), angle_match.group(2), angle_match.group(3), angle_match.group(6), angle_match.group(7)]))
                elif torsion_match:
                    if "torsions" not in rv:
                        rv["torsions"] = []
                    rv["torsions"].append(
                        ",".join(
                            [
                                torsion_match.group(1),
                                torsion_match.group(2),
                                torsion_match.group(3),
                                torsion_match.group(4),
                            ]
                        )
                    )
                    if torsion_match.group(5) == "S":
                        if "scan" not in rv:
                            rv["scan"] = []
                        rv["scan"].append(",".join([torsion_match.group(1), torsion_match.group(2), torsion_match.group(3), torsion_match.group(4), torsion_match.group(7), torsion_match.group(8)]))

                line = f.readline()
                n += 1

            return rv, n

        self.all_geom = []
        line = f.readline()
        self.other["archive"] = ""
        constraints = {}
        grad = {}
        self.other["opt_steps"] = 0
        found_archive = False
        n = 1
        route = None
        oniom = False
        has_params = False
        while line != "":
            if line.strip().startswith("AtFile"):
                parameters = line.split()[1]
                has_params = True
            # route
            # we need to grab the route b/c sometimes 'hpmodes' can get split onto multiple lines:
            # B3LYP/genecp EmpiricalDispersion=GD3 int=(grid=superfinegrid) freq=(h
            # pmodes,noraman,temperature=313.15)
            if line.strip().startswith("#") and route is None:
                route = ""
                while "------" not in line:
                    route += line[1:].splitlines()[0]
                    n += 1
                    line = f.readline()
                oniom = "oniom" in route.lower()
            # archive entry
            elif line.strip().startswith("1\\1\\"):
                found_archive = True
                line = "@" + line.strip()[4:]
            elif found_archive and line.strip().endswith("@"):
                self.other["archive"] = self.other["archive"][:-2] + "\\\\"
                found_archive = False
            elif found_archive:
                self.other["archive"] += line.strip()

            # input atom specs and charge/mult
            if not oniom and "Symbolic Z-matrix:" in line:
                self.atoms, n = get_input(f, n)

            #Pseudopotential info
            elif "Pseudopotential Parameters" in line:
                self.other["ECP"] = []
                self.skip_lines(f, 4)
                n += 5
                line = f.readline()
                while "=====" not in line:
                    line = line.split()
                    if line[0].isdigit() and line[1].isdigit():
                        ele = line[1]
                        n += 1
                        line = f.readline().split()
                        if line[0] != "No":
                            self.other["ECP"].append(ELEMENTS[int(ele)])
                    n += 1
                    line = f.readline()

            # geometry
            elif not oniom and (
                "Input orientation" in line or
                "Standard orientation" in line
            ):
                record_coords = True
                if "scan" in constraints:
                    if scan_read_all == False:
                        # only want to only record converged geometries for scans
                        record_coords = False
                        if "gradient" in self.other:
                            true_count=0
                            for i in self.other["gradient"]:
                                if self.other["gradient"][i]["converged"] == True:
                                    true_count+=1
                            if true_count == len(self.other["gradient"]):
                                record_coords = True
                    elif scan_read_all == True:
                        record_coords = True
                if get_all and self.atoms and record_coords:
                    self.all_geom += [{
                        "atoms": deepcopy(self.atoms),
                        "data": deepcopy(self.other),
                    }]
                    # delete gradient so we don't double up on standard and input orientation
                    try:
                        del self.other["gradient"]
                    except KeyError:
                        pass
                if record_coords:
                    self.atoms, n = get_atoms(f, n)
                self.other["opt_steps"] += 1

            elif oniom and (
                "Input orientation" in line or
                "Standard orientation" in line
            ):
                 if get_all and len(self.atoms) > 0:
                    self.all_geom += [{
                        "atoms": deepcopy(self.atoms),
                        "data": deepcopy(self.other),
                    }]
                 self.atoms, n = get_oniom_atoms(f, n)
                 self.other["opt_steps"] += 1

            #oniom atom types and input charges
            elif oniom and "Symbolic Z-matrix" in line:
                self.atoms, n = get_oniom_info(f, n)

            elif "The following ModRedundant input section has been read:" in line:
                constraints, n = get_modredundant(f, n)

            elif just_geom:
                line = f.readline()
                n += 1
                continue
                # z-matrix parameters
            # elif "!   Optimized Parameters   !" in line:
            elif "!   Optimized Parameters   !" in line:
                self.other["params"], n = get_params(f, n)

            # status
            elif NORM_FINISH in line:
                self.other["finished"] = True

            # read energies from different methods
            elif "SCF Done" in line:
                tmp = [word.strip() for word in line.split()]
                idx = tmp.index("=")
                self.other["energy"] = float(tmp[idx + 1])
                self.other["scf_energy"] = float(tmp[idx + 1])

            elif line.startswith(" Energy= "):
                self.other["energy"] = float(line.split()[1])

            # CC energy
            elif line.startswith(" CCSD(T)= "):
                self.other["energy"] = float(line.split()[-1].replace("D", "E"))
                self.other["E(CCSD(T))"] = self.other["energy"]

            # basis set details
            elif line.startswith(" NBasis") and "NFC" in line:
                n_basis = int(re.match(" NBasis=\s*(\d+)", line).group(1))
                self.other["n_basis"] = n_basis
                n_frozen = int(re.search(" NFC=\s*(\d+)", line).group(1))
                self.other["n_frozen"] = n_frozen

            elif line.startswith(" NROrb"):
                n_occupied_alpha = int(re.search(" NOA=\s*(\d+)", line).group(1))
                self.other["n_occupied_alpha"] = n_occupied_alpha
                n_occupied_beta = int(re.search(" NOB=\s*(\d+)", line).group(1))
                self.other["n_occupied_beta"] = n_occupied_beta
                n_virtual_alpha = int(re.search(" NVA=\s*(\d+)", line).group(1))
                self.other["n_virtual_alpha"] = n_virtual_alpha
                n_virtual_beta = int(re.search(" NVB=\s*(\d+)", line).group(1))
                self.other["n_virtual_beta"] = n_virtual_beta

            # Frequencies 
            elif "Harmonic frequencies" in line:
                if route is not None and "hpmodes" in route.lower():
                    self.other["hpmodes"] = True
                freq_str = line
                line = f.readline()
                while line.strip():
                    n += 1
                    freq_str += line
                    line = f.readline()
                if "hpmodes" not in self.other:
                    self.other["hpmodes"] = False
                self.other["frequency"] = Frequency(
                    freq_str, hpmodes=self.other["hpmodes"], atoms=self.atoms,
                )

            elif "Anharmonic Infrared Spectroscopy" in line:
                self.skip_lines(f, 5)
                n += 5
                anharm_str = ""
                combinations_read = False
                combinations = False
                line = f.readline()
                while not combinations_read:
                    n += 1
                    anharm_str += line
                    if "Combination Bands" in line:
                        combinations = True
                    line = f.readline()
                    if combinations and line == "\n":
                        combinations_read = True

                self.other["frequency"].parse_gaussian_lines(
                    anharm_str.splitlines(), harmonic=False,
                )

            # X matrix for anharmonic
            elif "Total Anharmonic X Matrix" in line:
                self.skip_lines(f, 1)
                n += 1
                n_freq = len(self.other["frequency"].data)
                n_sections = int(np.ceil(n_freq / 5))
                x_matrix = np.zeros((n_freq, n_freq))
                for section in range(0, n_sections):
                    header = f.readline()
                    n += 1
                    for j in range(5 * section, n_freq):
                        line = f.readline()
                        n += 1
                        ll = 5 * section
                        ul = 5 * section + min(j - ll + 1, 5)
                        x_matrix[j, ll:ul] = [
                            float(x.replace("D", "e"))
                            for x in line.split()[1:]
                        ]
                x_matrix += np.tril(x_matrix, k=-1).T
                self.other["X_matrix"] = x_matrix

            elif "Total X0" in line:
                self.other["X0"] = float(line.split()[5])

            # TD-DFT output
            elif line.strip().startswith("Ground to excited state"):
                uv_vis = ""
                highest_state = 0
                done = False
                read_states = False
                while not done:
                    n += 1
                    uv_vis += line
                    if not read_states and line.strip() and line.split()[0].isdigit():
                        state = int(line.split()[0])
                        if state > highest_state:
                            highest_state = state
                    if line.strip().startswith("Ground to excited state transition velocity"):
                        read_states = True
                    if re.search("Excited State\s*%i:" % highest_state, line):
                        done = True
                    if line.strip().startswith("Total Energy, E"):
                        nrg = re.search(
                            r"Total Energy, E\((\S+)\)\s*=\s*(-?\d+\.\d+)", line
                        )
                        self.other["E(%s)" % nrg.group(1)] = float(nrg.group(2))
                        self.other["energy"] = float(nrg.group(2))

                    line = f.readline()
                self.other["uv_vis"] = ValenceExcitations(
                    uv_vis, style="gaussian"
                )

            elif line.startswith(" S**2 before annihilation"):
                self.other["S^2 before"] = float(line.split()[3].strip(","))
                self.other["S^2 annihilated"] = float(line.split()[-1])

            # Thermo
            # elif temperature.match(line):
            elif line.startswith(" Temperature"):
                self.other["temperature"] = float(
                    float_num.search(line).group(0)
                )
                line = f.readline()
                while line and not line.startswith(" Molecular mass:"):
                    if isotope.match(line):
                        ndx = int(isotope.match(line).group(1)) - 1
                        self.atoms[ndx]._mass = float(line.split()[-1])
                    line = f.readline()
                
                self.other["mass"] = float(float_num.search(line).group(0))
                self.other["mass"] *= UNIT.AMU_TO_KG

            elif "Rotational constants (GHZ):" in line:
                rot = float_num.findall(line)
                rot = [
                    float(r) * PHYSICAL.PLANCK * (10 ** 9) / PHYSICAL.KB
                    for r in rot
                ]
                self.other["rotational_temperature"] = rot

            # rotational constants from anharmonic frequency jobs
            elif "Rotational Constants (in MHz)" in line:
                self.skip_lines(f, 2)
                n += 2
                equilibrium_rotational_temperature = np.zeros(3)
                ground_rotational_temperature = np.zeros(3)
                centr_rotational_temperature = np.zeros(3)
                for i in range(0, 3):
                    line = f.readline()
                    n += 1
                    info = line.split()
                    Be = float(info[1])
                    B00 = float(info[3])
                    B0 = float(info[5])
                    equilibrium_rotational_temperature[i] = Be
                    ground_rotational_temperature[i] = B00
                    centr_rotational_temperature[i] = B0
                equilibrium_rotational_temperature *= (
                    PHYSICAL.PLANCK * 1e6 / PHYSICAL.KB
                )
                ground_rotational_temperature *= (
                    PHYSICAL.PLANCK * 1e6 / PHYSICAL.KB
                )
                centr_rotational_temperature *= (
                    PHYSICAL.PLANCK * 1e6 / PHYSICAL.KB
                )
                self.other[
                    "equilibrium_rotational_temperature"
                ] = equilibrium_rotational_temperature
                self.other[
                    "ground_rotational_temperature"
                ] = ground_rotational_temperature
                self.other[
                    "centr_rotational_temperature"
                ] = centr_rotational_temperature

            elif "Sum of electronic and zero-point Energies=" in line:
                self.other["E_ZPVE"] = float(float_num.search(line).group(0))
            elif "Sum of electronic and thermal Enthalpies=" in line:
                self.other["enthalpy"] = float(float_num.search(line).group(0))
            elif "Sum of electronic and thermal Free Energies=" in line:
                self.other["free_energy"] = float(
                    float_num.search(line).group(0)
                )
            elif "Zero-point correction=" in line:
                self.other["ZPVE"] = float(float_num.search(line).group(0))
            elif "Rotational symmetry number" in line:
                self.other["rotational_symmetry_number"] = int(
                    re.search("\d+", line).group(0)
                )

            # Gradient
            elif "Threshold  Converged?" in line:
                line = f.readline()
                n += 1
                grad = {}

                def add_grad(line, name, grad):
                    line = line.split()
                    grad[name] = {
                        "value": line[2],
                        "threshold": line[3],
                        "converged": True if line[4] == "YES" else False,
                    }
                    return grad

                while line != "":
                    if "Predicted change in Energy" in line:
                        break
                    if re.search("Maximum\s+Force", line) is not None:
                        grad = add_grad(line, "Max Force", grad)
                    if re.search("RMS\s+Force", line) is not None:
                        grad = add_grad(line, "RMS Force", grad)
                    if re.search("Maximum\s+Displacement", line) is not None:
                        grad = add_grad(line, "Max Disp", grad)
                    if re.search("RMS\s+Displacement", line) is not None:
                        grad = add_grad(line, "RMS Disp", grad)
                    line = f.readline()
                    n += 1
                self.other["gradient"] = grad

            # electronic properties
            elif "Electrostatic Properties (Atomic Units)" in line:
                self.skip_lines(f, 5)
                n += 5
                self.other["electric_potential"] = []
                self.other["electric_field"] = []
                line = f.readline()
                while "--" not in line:
                    info = line.split()
                    self.other["electric_potential"].append(float(info[2]))
                    self.other["electric_field"].append([float(x) for x in info[3:]])
                    line = f.readline()
                    n += 1
                self.other["electric_potential"] = np.array(self.other["electric_potential"])
                self.other["electric_field"] = np.array(self.other["electric_field"])

            # optical features
            elif "[Alpha]" in line:
                alpha_match = re.search("\[Alpha\].*\(\s*(.*\s?.*)\)\s*=\s*(-?\d+\.\d+)", line)
                self.other["optical_rotation_(%s)" % alpha_match.group(1)] = \
                float(alpha_match.group(2))

            # symmetry
            elif "Full point group" in line:
                self.other["full_point_group"] = line.split()[-3]

            elif "Largest Abelian subgroup" in line:
                self.other["abelian_subgroup"] = line.split()[-3]

            elif "Largest concise Abelian subgroup" in line:
                self.other["concise_abelian_subgroup"] = line.split()[-3]

            # forces
            elif "Forces (Hartrees/Bohr)" in line:
                gradient = np.zeros((len(self.atoms), 3))
                self.skip_lines(f, 2)
                n += 2
                for i in range(0, len(self.atoms)):
                    n += 1
                    line = f.readline()
                    info = line.split()
                    gradient[i] = np.array([float(x) for x in info[2:]])

                self.other["forces"] = gradient

            # nbo stuff
            elif "N A T U R A L   A T O M I C   O R B I T A L   A N D" in line:
                self.read_nbo(f)

            # atomic charges
            elif any(("Mulliken" in line, "Hirshfeld" in line, "ESP" in line, "APT" in line)) and "hydrogens" not in line:
                charge_match = re.search("(\S+) charges.*:", line)
                if charge_match:
                    self.skip_lines(f, 1)
                    n += 1
                    charges = []
                    for i in range(0, len(self.atoms)):
                        line = f.readline()
                        n += 1
                        charges.append(float(line.split()[2]))
                        self.atoms[i].charge = float(line.split()[2])
                    self.other[charge_match.group(1) + " Charges"] = charges

            elif "Hirshfeld charges, spin densities, dipoles, and CM5 charges" in line:
                self.skip_lines(f, 1)
                n += 1
                cm5_charges = []
                hirshfeld_charges = []
                for i in range(0, len(self.atoms)):
                    line = f.readline()
                    n += 1
                    data = line.split()
                    hirshfeld = float(data[2])
                    cm5 = float(data[7])
                    hirshfeld_charges.append(hirshfeld)
                    cm5_charges.append(cm5)
                self.other["Hirshfeld Charges"] = hirshfeld_charges
                self.other["CM5 Charges"] = cm5_charges

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # BE CAREFUL ABOUT WHAT'S AFTER THIS
            # WE PUT A REGEX FOR FLOATING POINT NUMBERS HERE FOR
            # PERFORMANCE REASONS
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            elif "." in line and (
                "0" in line or
                "1" in line or
                "2" in line or
                "3" in line or
                "4" in line or
                "5" in line or
                "6" in line or
                "7" in line or
                "8" in line or
                "9" in line
            ):
            # elif re.search("\d+\.\d+", line):
                nrg_match = most_energies.search(line)
                mp_match = mp_energies.search(line)
                # will also match the SCF line (hence the else here)
                # the match in the SCF line could be confusing b/c
                # the SCF line could be
                # SCF Done:  E(RB2PLYPD3) =  -76.2887108570     A.U. after   10 cycles
                # and later on, there will be a line...
                #  E2(B2PLYPD3) =    -0.6465105880D-01 E(B2PLYPD3) =    -0.76353361915801D+02
                # this will give:
                # * E(RB2PLYPD3) = -76.2887108570
                # * E(B2PLYPD3) = -76.353361915801
                # very similar names for very different energies...
                # most energies
                if nrg_match:
                    nrg = float(nrg_match.group(2).replace("D", "E"))
                    if nrg_match.group(1) != "E(TD-HF/TD-DFT)":
                        self.other["energy"] = nrg
                    self.other[nrg_match.group(1)] = nrg
                # MP energies
                elif mp_match:
                    self.other["energy"] = float(mp_match.group(2).replace("D", "E"))
                    self.other["E(%s)" % mp_match.group(1)] = self.other["energy"]

            # capture errors
            # only keep first error, want to fix one at a time
            elif "error" not in self.other:
                for err in ERROR:
                    if isinstance(err, str):
                        if err in line:
                            self.other["error"] = ERROR[err]
                            self.other["error_msg"] = line.strip()
                            break

                    elif err.search(line):
                        self.other["error"] = ERROR[err]
                        self.other["error_msg"] = line.strip()
                        break

            line = f.readline()
            n += 1

        if (
            get_all and
            scan_read_all == False and
            not just_geom and not
            all(grad[i]["converged"] for i in grad)
        ):
            self.other["gradient"] = grad

        if get_all:
            self.all_geom += [{
                "atoms": self.atoms,
                "data": self.other,
            }]

        if not just_geom:
            if route is not None:
                other_kwargs = {GAUSSIAN_ROUTE: {}}
                route_spec = re.compile("(\w+)=?\((.*)\)")
                if oniom == False:
                    method_and_basis = re.search(
                        "#(?:[NnPpTt]\s+?)(\S+)|#\s*?(\S+)", route
                    )
                if oniom == True:
                    method_and_basis = re.search(
                        "#(?:[NnPpTt]*\s+?)(?:[OoNnIiMm]*\()(\S+?)(?::)([A-z0-9\/\(\)-]*)(?:=\()?([A-z,]*)?(?:\))?(?::)?([A-z-\(\)0-9]*)?(?:=\()?([A-z,]*)(?:\)*=)?([A-z,]*)?"
                    , route)
                if method_and_basis is not None:
                    if oniom == False:
                        if method_and_basis.group(2):
                            method_info = method_and_basis.group(2).split("/")
                        else:
                            method_info = method_and_basis.group(1).split("/")

                        method = method_info[0]
                        if len(method_info) > 1:
                            basis = method_info[1]
                        else:
                            basis = None
                    if oniom == True:
                        method = None
                        basis = None
                        medium_info = None
                        mm_options = {}
                        oniom_options = []
                        high_info = method_and_basis.group(1).split("/")
                        if method_and_basis.group(4) and len(method_and_basis.group(4)) > 1:
                            medium_info = method_and_basis.group(2).split("/")
                            low_info = method_and_basis.group(4).split("/")
                        if (method_and_basis.group(4) and len(method_and_basis.group(4)) <= 1) or not method_and_basis.group(4):
                            low_info = method_and_basis.group(2).split("/")
                        if method_and_basis.group(6):
                            oniom_options = method_and_basis.group(6).split(",")
                        high_method = high_info[0]
                        try:
                            high_basis = high_info[1]
                        except IndexError:
                            high_basis = None
                        if medium_info is not None:
                            medium_method = medium_info[0]
                            try:
                                medium_basis = medium_info[1]
                            except IndexError:
                                medium_basis = None
                        low_method = low_info[0]
                        try:
                            low_basis = low_info[1]
                        except IndexError:
                            low_basis = None
                        if medium_info is None:
                            medium_method = None
                            medium_basis = None
                        def fix_paren(string):
                            if ")" in string:
                                if "(" not in string:
                                    string = string.split(")")[0]
                                if "(" in string:
                                    left = 0
                                    right = 0
                                    for x in string:
                                        if x == "(":
                                            left += 1
                                        if x == ")":
                                            right += 1
                                    if left < right:
                                        string = string[:-1]
                            else:
                                pass
                            return string
                        if low_basis is None:
                            low_method = fix_paren(low_method)
                        elif low_basis is not None:
                            low_basis = fix_paren(low_basis)
                        if method_and_basis.group(3) and not method_and_basis.group(5):
                            mm_options[low_method] = method_and_basis.group(3).split(",")
                        if method_and_basis.group(5) and not method_and_basis.group(3):
                            mm_options[low_method] = method_and_basis.group(5).split(",")
                        if method_and_basis.group(5) and method_and_basis.group(3):
                            mm_options[low_method] = method_and_basis.group(5).split(",")
                            mm_options[medium_method] = method_and_basis.group(3).split(",")
                        if mm_options != {}:
                            other_kwargs["mm"] = mm_options
                        if oniom_options != []:
                            other_kwargs["oniom"] = oniom_options

                    if has_params:
                        other_kwargs["parameters"] = parameters

                    route_options = route.split()
                    job_type = []
                    grid = None
                    solvent = None
                    for option in route_options:
                        if option.startswith("#"):
                            continue
                        if method is not None and option.startswith(method):
                                continue
                        if option.lower().startswith("oniom"):
                                continue

                        option_lower = option.lower()
                        if option_lower.startswith("opt"):
                            ts = False
                            match = route_spec.search(option)
                            if match:
                                options = match.group(2).split(",")
                            elif option_lower.startswith("opt="):
                                options = ["".join(option.split("=")[1:])]
                            else:
                                if not constraints:
                                    # if we didn't read constraints, try using flagged atoms instead
                                    from AaronTools.finders import FlaggedAtoms

                                    constraints = {"atoms": FlaggedAtoms}
                                    if not any(atom.flag for atom in self.atoms):
                                        constraints = None
                                job_type.append(
                                    OptimizationJob(constraints=constraints)
                                )
                                continue

                            other_kwargs[GAUSSIAN_ROUTE]["opt"] = []

                            for opt in options:
                                if opt.lower() == "ts":
                                    ts = True
                                else:
                                    other_kwargs[GAUSSIAN_ROUTE]["opt"].append(
                                        opt
                                    )

                            job_type.append(
                                OptimizationJob(
                                    transition_state=ts,
                                    constraints=constraints,
                                )
                            )

                        elif option_lower.startswith("freq"):
                            temp = 298.15
                            match = route_spec.search(option)
                            if match:
                                options = match.group(2).split(",")
                            elif option_lower.startswith("freq="):
                                options = "".join(option.split("=")[1:])
                            else:
                                job_type.append(FrequencyJob())
                                continue

                            other_kwargs[GAUSSIAN_ROUTE]["freq"] = []

                            for opt in options:
                                if opt.lower().startswith("temp"):
                                    temp = float(opt.split("=")[1])
                                else:
                                    other_kwargs[GAUSSIAN_ROUTE][
                                        "freq"
                                    ].append(opt)

                            job_type.append(FrequencyJob(temperature=temp))

                        elif option_lower == "sp":
                            job_type.append(SinglePointJob())

                        elif option_lower.startswith("int"):
                            match = route_spec.search(option)
                            if match:
                                options = match.group(2).split(",")
                            elif option_lower.startswith("freq="):
                                options = "".join(option.split("=")[1:])
                            else:
                                job_type.append(FrequencyJob())
                                continue

                            for opt in options:
                                if opt.lower().startswith("grid"):
                                    grid_name = opt.split("=")[1]
                                    grid = IntegrationGrid(grid_name)
                                else:
                                    if (
                                        "Integral"
                                        not in other_kwargs[GAUSSIAN_ROUTE]
                                    ):
                                        other_kwargs[GAUSSIAN_ROUTE][
                                            "Integral"
                                        ] = []
                                    other_kwargs[GAUSSIAN_ROUTE][
                                        "Integral"
                                    ].append(opt)

                        else:
                            # TODO: parse solvent
                            match = route_spec.search(option)
                            if match:
                                keyword = match.group(1)
                                options = match.group(2).split(",")
                                other_kwargs[GAUSSIAN_ROUTE][keyword] = options
                            elif "=" in option:
                                keyword = option.split("=")[0]
                                options = "".join(option.split("=")[1:])
                                other_kwargs[GAUSSIAN_ROUTE][keyword] = [
                                    options
                                ]
                            else:
                                other_kwargs[GAUSSIAN_ROUTE][option] = []
                                continue

                    self.other["other_kwargs"] = other_kwargs
                    if oniom == True:
                        try:
                            theory = Theory(
                                charge=self.other["charge"],
                                multiplicity=self.other["multiplicity"],
                                job_type=job_type,
                                high_method=high_method,
                                medium_method=medium_method,
                                low_method=low_method,
                                high_basis=high_basis,
                                medium_basis=medium_basis,
                                low_basis=low_basis,
                                grid=grid,
                                solvent=solvent,
                            )
                            theory.kwargs = self.other["other_kwargs"]
                            self.other["theory"] = theory
                        except KeyError:
                            print(high_method)
                            print(high_basis)
                            if medium_method is not None: print(medium_method)
                            if medium_basis is not None: print(medium_basis)
                            print(low_method)
                            if low_basis is not None: print(low_basis)
                            print(solvent)
                    elif oniom == False:
                        try:
                            theory = Theory(
                                charge=self.other["charge"],
                                multiplicity=self.other["multiplicity"],
                                job_type=job_type,
                                basis=basis,
                                method=method,
                                grid=grid,
                                solvent=solvent,
                            )
                            theory.kwargs = self.other["other_kwargs"]
                            self.other["theory"] = theory
                        except KeyError:
                            # if there is a serious error, too little info may be available
                            # to properly create the theory object
                            #pass
                            print(method)
                            print(basis)
                            print(solvent)

        for i, a in enumerate(self.atoms):
            a.name = str(i + 1)

        if "finished" not in self.other:
            self.other["finished"] = False
        if "error" not in self.other:
            self.other["error"] = None
        return

    def read_com(self, f):
        """read gaussian input file"""
        found_atoms = False
        found_constraint = False
        atoms = []
        other = {}
        for line in f:
            # header
            if line.startswith("%"):
                continue
            if line.startswith("#"):
                method = re.search("^#([NnPpTt]\s+?)(\S+)|^#\s*?(\S+)", line)
                # route can be #n functional/basis ...
                # or #functional/basis ...
                # or # functional/basis ...
                if method.group(3):
                    other["method"] = method.group(3)
                else:
                    other["method"] = method.group(2)
                if "temperature=" in line:
                    other["temperature"] = float(
                        re.search("temperature=(\d+\.?\d*)", line).group(1)
                    )
                if "solvent=" in line:
                    other["solvent"] = re.search(
                        "solvent=(\S+)\)", line
                    ).group(1)
                if "scrf=" in line:
                    # solvent model should be non-greedy b/c solvent name can have commas
                    other["solvent_model"] = re.search(
                        "scrf=\((\S+?),", line
                    ).group(1)
                if "EmpiricalDispersion=" in line:
                    other["emp_dispersion"] = re.search(
                        "EmpiricalDispersion=(\S+)", line
                    ).group(1)
                if "int=(grid" in line or "integral=(grid" in line.lower():
                    other["grid"] = re.search(
                        "(?:int||Integral)=\(grid[(=](\S+?)\)", line
                    ).group(1)
                # comments can be multiple lines long
                # but there should be a blank line between the route and the comment
                # and another between the comment and the charge+mult
                blank_lines = 0
                while blank_lines < 2:
                    line = f.readline().strip()
                    if len(line) == 0:
                        blank_lines += 1
                    else:
                        if "comment" not in other:
                            other["comment"] = ""
                        other["comment"] += "%s\n" % line
                other["comment"] = (
                    other["comment"].strip() if "comment" in other else ""
                )
                line = f.readline()
                if len(line.split()) > 1:
                    line = line.split()
                else:
                    line = line.split(",")
                other["charge"] = int(line[0])
                other["multiplicity"] = int(line[1])
                found_atoms = True
                continue
            # constraints
            if found_atoms and line.startswith("B") and line.endswith("F"):
                found_constraint = True
                if "constraint" not in other:
                    other["constraint"] = []
                other["constraint"] += [float_num.findall(line)]
                continue
            # footer
            if found_constraint:
                if "footer" not in other:
                    other["footer"] = ""
                other["footer"] += line
                continue
            # atom coords
            nums = float_num.findall(line)
            line = line.split()
            is_oniom = False
            flag = ""
            atomtype = ""
            charge = ""
            tags = []
            has_flag = False
            if "oniom" in other["method"].lower():
                is_oniom = True
            if not is_oniom:
                if len(line) == 5 and is_alpha(line[0]) and len(nums) == 4:
                    if not is_int(line[1]):
                        continue
                    a = Atom(element=line[0], coords=nums[1:], flag=nums[0])
                    atoms += [a]
                elif len(line) == 4 and is_alpha(line[0]) and len(nums) == 3:
                    a = Atom(element=line[0], coords=nums)
                    atoms += [a]
            elif is_oniom:
                link_info = {}
                if len(line) > 0 and len(line[0].split("-")) > 0 and len(nums) > 2:
                    if len(line[0].split("-")) == 2:
                        if not is_alpha(line[0].split("-")[1][0]):
                            charge = nums[0]
                        elif is_alpha(line[0].split("-")[1][0]):
                            atomtype = line[0].split("-")[1]
                    if len(line[0].split("-")) == 4:
                        atomtype = line[0].split("-")[1]
                        charge = str(-1 * float(line[0].split("-")[3]))
                    if len(line[0].split("-")) == 3:
                        if not is_alpha(line[0].split("-")[1][0]):
                            charge = nums[0]
                        elif is_alpha(line[0].split("-")[1][0]):
                            atomtype = line[0].split("-")[1]
                            charge = line[0].split("-")[2]
                    if len(line)%2 == 0:
                        has_flag = True
                        flag = line[1]
                        coords = line[2:5]
                    if not has_flag:
                        coords = line[1:4]
                    if len(line) > 6:
                        #tags.append(line[len(line)-2:])
                        link_atom = line[len(line)-2:]
                        link_info["connected"] = link_atom[1]
                        info = link_atom[0].split("-")
                        link_info["element"] = info[0]
                        if len(info) == 3:
                            link_info["atomtype"] = info[1]
                            link_info["charge"] = info[2]
                        elif len(info) == 2:
                            if is_alpha(info[1][0]):
                                link_info["atomtype"] = info[1]
                            else:
                                link_info["charge"] = info[1]
                        layer = line[len(line)-3]
                    if len(line) < 7:
                        layer = line[len(line)-1]
                    a = OniomAtom(element=line[0].split("-")[0],flag=flag,coords=coords,layer=layer,atomtype=atomtype,charge=charge,link_info=link_info)
                    atoms += [a]
        for i, a in enumerate(atoms):
            a.name = str(i + 1)
        self.atoms = atoms
        self.other = other
        return

    def read_fchk(self, f, just_geom=True, max_length=10000000):
        """read formatted checkpoint file"""
        def parse_to_list(
            i, lines, length, data_type, debug=False, max_length=max_length,
        ):
            """takes a block in an fchk file and turns it into an array
            block headers all end with N=   <int>
            the length of the array will be <int>
            the data type is specified by data_type"""
            i += 1
            line = f.readline()
            # print("first line", line)
            items_per_line = len(line.split())
            # print("items per line", items_per_line)
            total_items = items_per_line
            num_lines = ceil(length / items_per_line)

            # print("lines in block", num_lines)

            block = [line]
            for k in range(0, num_lines - 1):
                line = f.readline()
                if max_length < length:
                    continue
                block.append(line)

            if max_length < length:
                return length, i + num_lines
            block = " ".join(block)

            if debug:
                print("full block")
                print(block)

            return (
                np.fromstring(block, count=length, dtype=data_type, sep=" "),
                i + num_lines,
            )

        self.atoms = []
        atom_numbers = []
        atom_coords = []

        other = {}

        int_info = re.compile("([\S\s]+?)\s*I\s*(N=)?\s*(-?\d+)")
        real_info = re.compile(
            "([\S\s]+?)\s*R\s*(N=)\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)"
        )
        char_info = re.compile(
            "([\S\s]+?)\s*C\s*(N=)?\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)"
        )

        theory = Theory()

        line = f.readline()

        i = 0
        while line != "":
            if i == 0:
                other["comment"] = line.strip()
            elif i == 1:
                i += 1
                line = f.readline()
                job_info = line.split()
                if job_info[0] == "SP":
                    theory.job_type = [SinglePointJob()]
                elif job_info[0] == "FOPT":
                    theory.job_type[OptimizationJob()]
                elif job_info[0] == "FTS":
                    theory.job_type = [OptimizationJob(transition_state=True)]
                elif job_info[0] == "FORCE":
                    theory.job_type = [ForceJob()]
                elif job_info[0] == "FREQ":
                    theory.job_type = [FrequencyJob()]

                theory.method = job_info[1]
                if len(job_info) > 2:
                    theory.basis = job_info[2]

                i += 1
                line = f.readline()
                continue

            int_match = int_info.match(line)
            real_match = real_info.match(line)
            char_match = char_info.match(line)
            if int_match is not None:
                data = int_match.group(1)
                # print("int", data)
                value = int_match.group(3)
                if data == "Charge" and not just_geom:
                    theory.charge = int(value)
                elif data == "Multiplicity" and not just_geom:
                    theory.multiplicity = int(value)
                elif data == "Atomic numbers":
                    atom_numbers, i = parse_to_list(i, f, int(value), int)
                elif not just_geom:
                    if int_match.group(2):
                        other[data], i = parse_to_list(
                            i, f, int(value), int
                        )
                    else:
                        other[data] = int(value)

            elif real_match is not None:
                data = real_match.group(1)
                # print("real", data)
                value = real_match.group(3)
                if data == "Current cartesian coordinates":
                    atom_coords, i = parse_to_list(i, f, int(value), float)
                elif data == "Total Energy":
                    other["energy"] = float(value)
                elif not just_geom:
                    if real_match.group(2):
                        other[data], i = parse_to_list(
                            i, f, int(value), float
                        )
                    else:
                        other[data] = float(value)

            # elif char_match is not None:
            #     data = char_match.group(1)
            #     value = char_match.group(3)
            #     if not just_geom:
            #         other[data] = lines[i + 1]
            #         i += 1

            line = f.readline()
            i += 1

        self.other = other
        self.other["theory"] = theory

        if isinstance(atom_coords, int):
            raise RuntimeError(
                "max. array size is insufficient to parse atom data\n"
                "must be at least %i" % atom_coords
            )
        coords = np.reshape(atom_coords, (len(atom_numbers), 3))
        for n, (atnum, coord) in enumerate(zip(atom_numbers, coords)):
            atom = Atom(
                element=ELEMENTS[atnum],
                coords=UNIT.A0_TO_BOHR * coord,
                name=str(n + 1),
            )
            self.atoms.append(atom)

        try:
            self.other["orbitals"] = Orbitals(self)
        except (NotImplementedError, KeyError):
            pass
        except (TypeError, ValueError) as err:
            self.LOG.warning(
                "could not create Orbitals, try increasing the max.\n"
                "array size to read from FCHK files\n\n"
                "%s" % err
            )
            for key in [
                "Alpha MO coefficients", "Beta MO coefficients",
                "Shell types", "Shell to atom map", "Contraction coefficients",
                "Primitive exponents", "Number of primitives per shell",
                "Coordinates of each shell",
            ]:
                if key in self.other and isinstance(self.other[key], int):
                    self.LOG.warning(
                        "size of %s is > %i: %i" % (key, max_length, self.other[key])
                    )

    def read_pdb(self, f, qt=False):
        """
        read pdb or pdbqt file
        """
        line = f.readline()
        n = 1
        def get_atoms(f, n, line):
            rv = []
            while line.split()[0] in {"ATOM", "TER", "ANISOU", "HETATM", "ROOT", "BRANCH", "ENDROOT", "ENDBRANCH", "ENDMDL"}:
                endmdl = False
                if line.split()[0].upper() in ("ATOM", "HETATM"):
                    if qt==False:
                        if line[76:78].strip() != "":
                            element = line[76:78].strip()
                        else:
                            element = ''.join(i for i in line[12:16].strip() if not i.isdigit())
                        atomtype = line[12:16].strip()
                        charge = line[78:].strip()
                    if qt==True:
                        element = ''.join(i for i in line[12:16].strip() if not i.isdigit())
                        atomtype = line[78:].strip()
                        charge = line[66:76].strip()
                    a = OniomAtom(element=element, coords=[line[30:38], line[38:46], line[46:54]], name=line[6:11].strip(), res=line[17:20].strip(), atomtype=atomtype, charge=charge)
                    rv += [a]
                elif line.startswith("ENDMDL"):
                    endmdl = True
                    break
                line = f.readline()
                n += 1
            return rv, n, endmdl
        num_models = 0
        while line != "":
            if line.startswith("HEADER"):
                self.name = line[62:66]
            if line.startswith("EXPDTA"):
                self.other["source"] = line[10:].strip()
            elif line.startswith("MODEL"):
                model_num = int(line.split()[1])
                line = f.readline()
                n += 1
                if model_num == 1:
                    self.atoms, n, endmdl = get_atoms(f, n, line)
                elif model_num > 1:
                    self.other["model_%s" % str(model_num)], n, endmdl = get_atoms(f, n, line)
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                atoms, n, endmdl = get_atoms(f, n, line)
                if endmdl == True:
                    num_models+=1
                    if num_models == 1:
                        self.atoms = atoms
                    elif num_models > 1:
                        self.other["model_%s" % str(num_models)] = atoms
                elif endmdl == False:
                    self.atoms = atoms
            elif line.startswith("CONECT"):
                while line.startswith("CONECT"):
                    for atom_num in line.split()[3:]:
                        self.atoms[int(line.split()[1])-1].connected.add(self.atoms[int(atom_num)-1])
                    line = f.readline()
                    n += 1
            line = f.readline()
            n += 1
        return

    def read_mmcif(self, f):
        """read mmcif"""
        line = f.readline()
        n = 1
        nloops=0
        current_model="data_UNK"
        self.other[current_model] = {}
        def read_loop(line, n, nloops):
            entries = "(?:[\'\"].*?[\'\"]|\S)+"
            self.other[current_model]["loop_" + str(nloops)]={}
            self.other[current_model]["loop_" + str(nloops)]["titles"]=[]
            self.other[current_model]["loop_" + str(nloops)]["items"]=[]
            n_titles = 0
            name_ndx=None
            ele_ndx=None
            type_ndx=None
            charge_ndx=None
            res_ndx=None
            x_ndx=None
            y_ndx=None
            z_ndx=None
            read_atoms=False
            while line.startswith("_"):
                self.other[current_model]["loop_" + str(nloops)]["titles"].append(line.strip())
                if line.startswith("_atom_site.id"):
                    read_atoms=True
                    atoms=[]
                    name_ndx = n_titles
                if line.startswith("_atom_site.type_symbol"):
                    ele_ndx = n_titles
                if line.startswith("_atom_site.label_atom_id"):
                    type_ndx = n_titles
                if line.startswith("_atom_site.auth_comp_id"):
                    res_ndx = n_titles
                if line.strip()=="_atom_site.Cartn_x":
                    x_ndx = n_titles
                if line.strip()=="_atom_site.Cartn_y":
                    y_ndx = n_titles
                if line.strip()=="_atom_site.Cartn_z":
                    z_ndx = n_titles
                if line.startswith("_atom_site.pdbx_formal_charge"):
                    charge_ndx = n_titles
                line = f.readline()
                n += 1
                n_titles+=1
            while not line.startswith("_") and line != "" and "#" not in line:
                item_list=[]
                items=[]
                while len(item_list) < n_titles:
                    item = ""
                    items=[]
                    if line.startswith(";"):
                        item = line.strip()
                        line = f.readline()
                        while not line.startswith(";"):
                            item = item + line.strip()
                            line = f.readline()
                    else:
                        items=re.findall(entries, line.strip())
                        if read_atoms:
                            atom = OniomAtom()
                            if name_ndx:
                                atom.name=items[name_ndx]
                            if ele_ndx:
                                atom.element=items[ele_ndx]
                            if res_ndx:
                                atom.res=items[res_ndx]
                            if type_ndx:
                                atom.atomtype=items[type_ndx]
                            if charge_ndx:
                                atom.charge=items[charge_ndx]
                            if x_ndx:
                                atom.coords=[float(items[x_ndx]), float(items[y_ndx]), float(items[z_ndx])]
                            atoms.append(atom)
                    item_list = item_list + items
                    if item != "":
                        item_list.append(item.strip(";"))
                    line = f.readline()
                    n+=1

                self.other[current_model]["loop_" + str(nloops)]["items"].append(item_list)
                #print(self.other[current_model]["loop_" + str(nloops)]["items"])
            if read_atoms:
                if self.atoms==[]:
                    self.atoms=atoms
                else:
                    self.other[current_model]["atoms"]=atoms
            nloops +=1
            return


        while line:
            if line.startswith("data"):
                current_model = line.strip()
                self.other[current_model] = {}
            elif line.startswith("loop"):
                line=f.readline()
                n+=1
                read_loop(line, n, nloops)
            elif line.startswith("_") and len(line.split())==2:
                if line.split()[1] != "?":
                    self.other[current_model][line.split()[0]]=line.split()[1]
            line = f.readline()
            n += 1
        return

    def read_nbo(self, f):
        """
        read nbo data
        """
        line = f.readline()
        while line:
            if "natural bond orbitals (summary):" in line.lower():
                break

            if "NATURAL POPULATIONS:" in line:
                self.skip_lines(f, 3)
                ao_types = []
                ao_atom_ndx = []
                nao_types = []
                occ = []
                nrg = []
                blank_lines = 0
                while blank_lines <= 1:
                    match = re.search(
                        "\d+\s+[A-Z][a-z]?\s+(\d+)\s+(\S+)\s+([\S\s]+?)(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                        line
                    )
                    if match:
                        ao_atom_ndx.append(int(match.group(1)) - 1)
                        ao_types.append(match.group(2))
                        nao_types.append(match.group(3))
                        occ.append(float(match.group(4)))
                        nrg.append(float(match.group(5)))
                        blank_lines = 0
                    else:
                        blank_lines += 1
                    line = f.readline()
                self.other["ao_types"] = ao_types
                self.other["ao_atom_ndx"] = ao_atom_ndx
                self.other["nao_type"] = nao_types
                self.other["ao_occ"] = occ
                self.other["ao_nrg"] = nrg

            if "Summary of Natural Population Analysis:" in line:
                self.skip_lines(f, 5)
                core_occ = []
                val_occ = []
                rydberg_occ = []
                nat_q = []
                line = f.readline()
                while "==" not in line:
                    info = line.split()
                    core_occ.append(float(info[3]))
                    val_occ.append(float(info[4]))
                    rydberg_occ.append(float(info[5]))
                    nat_q.append(float(info[2]))
                    line = f.readline()
                self.other["Natural Charges"] = nat_q
                self.other["core_occ"] = core_occ
                self.other["valence_occ"] = val_occ
                self.other["rydberg_occ"] = rydberg_occ

            if "Wiberg bond index matrix in the NAO basis" in line:
                dim = len(self.other["Natural Charges"])
                bond_orders = np.zeros((dim, dim))
                done = False
                j = 0
                for block in range(0, ceil(dim / 9)):
                    offset = 9 * j
                    self.skip_lines(f, 3)
                    for i in range(0, dim):
                        line = f.readline()
                        for k, bo in enumerate(line.split()[2:]):
                            bo = float(bo)
                            bond_orders[i][offset + k] = bo
                    j += 1
                self.other["wiberg_nao"] = bond_orders

            line = f.readline()

    def read_crest(self, f, conf_name=None):
        """
        conf_name = False to skip conformer loading (doesn't get written until crest job is done)
        """
        if conf_name is None:
            conf_name = os.path.join(
                os.path.dirname(self.name), "crest_conformers.xyz"
            )
        line = True
        self.other["finished"] = False
        self.other["error"] = None
        while line:
            line = f.readline()
            if "terminated normally" in line:
                self.other["finished"] = True
            elif "population of lowest" in line:
                self.other["best_pop"] = float(float_num.findall(line)[0])
            elif "ensemble free energy" in line:
                self.other["free_energy"] = (
                    float(float_num.findall(line)[0]) / UNIT.HART_TO_KCAL
                )
            elif "ensemble entropy" in line:
                self.other["entropy"] = (
                    float(float_num.findall(line)[1]) / UNIT.HART_TO_KCAL
                )
            elif "ensemble average energy" in line:
                self.other["avg_energy"] = (
                    float(float_num.findall(line)[0]) / UNIT.HART_TO_KCAL
                )
            elif "E lowest" in line:
                self.other["energy"] = float(float_num.findall(line)[0])
            elif "T /K" in line:
                self.other["temperature"] = float(float_num.findall(line)[0])
            elif (
                line.strip()
                .lower()
                .startswith(("forrtl", "warning", "*warning"))
            ):
                self.other["error"] = "UNKNOWN"
                if "error_msg" not in self.other:
                    self.other["error_msg"] = ""
                self.other["error_msg"] += line
            elif "-chrg" in line:
                self.other["charge"] = int(float_num.findall(line)[0])
            elif "-uhf" in line:
                self.other["multiplicity"] = (
                    int(float_num.findall(line)[0]) + 1
                )

        if self.other["finished"] and conf_name:
            self.other["conformers"] = FileReader(
                conf_name,
                get_all=True,
            ).all_geom
            self.comment, self.atoms = self.other["conformers"][0]
            self.other["conformers"] = self.other["conformers"][1:]

    def read_xtb(self, f, freq_name=None):
        """read xtb output"""
        line = True
        self.other["finished"] = False
        self.other["error"] = None
        self.atoms = []
        self.comment = ""
        while line:
            line = f.readline()
            if "Optimized Geometry" in line:
                line = f.readline()
                n_atoms = int(line.strip())
                line = f.readline()
                self.comment = " ".join(line.strip().split()[2:])
                for i in range(n_atoms):
                    line = f.readline()
                    elem, x, y, z = line.split()
                    self.atoms.append(Atom(element=elem, coords=[x, y, z]))
            if "normal termination" in line:
                self.other["finished"] = True
            if "abnormal termination" in line:
                self.other["error"] = "UNKNOWN"
            if line.strip().startswith("#ERROR"):
                if "error_msg" not in self.other:
                    self.other["error_msg"] = ""
                self.other["error_msg"] += line
            if "charge" in line and ":" in line:
                self.other["charge"] = int(float_num.findall(line)[0])
            if "spin" in line and ":" in line:
                self.other["multiplicity"] = (
                    2 * float(float_num.findall(line)[0]) + 1
                )
            if "total energy" in line:
                self.other["energy"] = (
                    float(float_num.findall(line)[0]) * UNIT.HART_TO_KCAL
                )
            if "zero point energy" in line:
                self.other["ZPVE"] = (
                    float(float_num.findall(line)[0]) * UNIT.HART_TO_KCAL
                )
            if "total free energy" in line:
                self.other["free_energy"] = (
                    float(float_num.findall(line)[0]) * UNIT.HART_TO_KCAL
                )
            if "electronic temp." in line:
                self.other["temperature"] = float(float_num.findall(line)[0])
        if freq_name is not None:
            with open(freq_name) as f_freq:
                self.other["frequency"] = Frequency(f_freq.read())

    def read_sqm(self, f):
        """read sqm output"""
        lines = f.readlines()

        self.other["finished"] = False

        self.atoms = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Atomic Charges for Step" in line:
                elements = []
                for info in lines[i + 2 :]:
                    if not info.strip() or not info.split()[0].isdigit():
                        break
                    ele = info.split()[1]
                    elements.append(ele)
                i += len(elements) + 2

            if "Final Structure" in line:
                k = 0
                for info in lines[i + 4 :]:
                    data = info.split()
                    coords = np.array([x for x in data[4:7]])
                    self.atoms.append(
                        Atom(
                            name=str(k + 1),
                            coords=coords,
                            element=elements[k],
                        )
                    )
                    k += 1
                    if k == len(elements):
                        break
                i += k + 4

            if "Calculation Completed" in line:
                self.other["finished"] = True

            if "Total SCF energy" in line:
                self.other["energy"] = (
                    float(line.split()[4]) / UNIT.HART_TO_KCAL
                )

            i += 1

        if not self.atoms:
            # there's no atoms if there's an error
            # error is probably on the last line
            self.other["error"] = "UNKNOWN"
            self.other["error_msg"] = line

    def read_nbo_47(self, f, nbo_name=None):
        """read nbo .47 file"""
        lines = f.readlines()
        bohr = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(" $"):
                section = line.split()[0]
                if section.startswith("$COORD"):
                    i += 1
                    self.atoms = []
                    line = lines[i]
                    while not line.startswith(" $END"):
                        if re.search("\d+\s+\d+(?:\s+-?\d+\.\d+\s){3}", line):
                            info = line.split()
                            ndx = int(info[0])
                            coords = [float(x) for x in info[2:5]]
                            self.atoms.append(
                                Atom(
                                    element=ELEMENTS[ndx],
                                    name=str(len(self.atoms) + 1),
                                    coords=np.array(coords),
                                )
                            )
                        i += 1
                        line = lines[i]
                elif section.startswith("$BASIS"):
                    reading_centers = False
                    reading_labels = False
                    i += 1
                    line = lines[i]
                    while not line.startswith(" $END"):
                        if "CENTER" in line.upper():
                            self.other["shell_to_atom"] = [
                                int(x) for x in line.split()[2:]
                            ]
                            reading_centers = True
                            reading_labels = False
                        elif "LABEL" in line.upper():
                            self.other["momentum_label"] = [
                                int(x) for x in line.split()[2:]
                            ]
                            reading_labels = True
                            reading_centers = False
                        elif reading_centers:
                            self.other["shell_to_atom"].extend(
                                [int(x) for x in line.split()]
                            )
                        elif reading_labels:
                            self.other["momentum_label"].extend(
                                [int(x) for x in line.split()]
                            )
                        i += 1
                        line = lines[i]
                elif section.startswith("$CONTRACT"):
                    int_sections = {
                        "NCOMP": "funcs_per_shell",
                        "NPRIM": "n_prim_per_shell",
                        "NPTR": "start_ndx",
                    }
                    float_sections = {
                        "EXP": "exponents",
                        "CS": "s_coeff",
                        "CP": "p_coeff",
                        "CD": "d_coeff",
                        "CF": "f_coeff",
                    }
                    i += 1
                    line = lines[i]
                    while not line.startswith(" $END"):
                        if any(line.strip().startswith(section) for section in int_sections):
                            section = line.split()[0]
                            self.other[int_sections[section]] = [
                                int(x) for x in line.split()[2:]
                            ]
                            i += 1
                            line = lines[i]
                            while "=" not in line and "$" not in line:
                                self.other[int_sections[section]].extend([
                                    int(x) for x in line.split()
                                ])
                                i += 1
                                line = lines[i]
                        elif any(line.strip().startswith(section) for section in float_sections):
                            section = line.split()[0]
                            self.other[float_sections[section]] = [
                                float(x) for x in line.split()[2:]
                            ]
                            i += 1
                            line = lines[i]
                            while "=" not in line and "$" not in line:
                                self.other[float_sections[section]].extend([
                                    float(x) for x in line.split()
                                ])
                                i += 1
                                line = lines[i]
                        else:
                            i += 1
                            line = lines[i]

                elif section.startswith("$GENNBO"):
                    if "BOHR" in section.upper():
                        bohr = True
                    nbas = re.search("NBAS=(\d+)", line)
                    n_funcs = int(nbas.group(1))
                    if "CUBICF" in section.upper():
                        self.LOG.warning("cubic F shell will not be handled correctly")
            i += 1

        if nbo_name is not None:
            self._read_nbo_coeffs(nbo_name)

    def _read_nbo_coeffs(self, nbo_name):
        """
        read coefficients in AO basis for NBO's/NLHO's/NAO's/etc.
        called by methods that read NBO input (.47) or output files (.31)
        """
        with open(nbo_name, "r") as f2:
            lines = f2.readlines()
        kind = re.search("P?(\S+)s", lines[1]).group(1)
        desc_file = os.path.splitext(nbo_name)[0] + ".46"
        if os.path.exists(desc_file):
            with open(desc_file, "r") as f3:
                desc_lines = f3.readlines()
                for k, line in enumerate(desc_lines):
                    if kind in line:
                        self.other["orbit_kinds"] = []
                        n_orbits = int(line.split()[1])
                        k += 1
                        while len(self.other["orbit_kinds"]) < n_orbits:
                            self.other["orbit_kinds"].extend([
                                desc_lines[k][i: i + 10]
                                for i in range(1, len(desc_lines[k]) - 1, 10)
                            ])
                            k += 1
        else:
            self.LOG.warning(
                "no .46 file found - orbital descriptions will be unavialable"
            )

        j = 3
        self.other["alpha_coefficients"] = []
        while len(self.other["alpha_coefficients"]) < sum(self.other["funcs_per_shell"]):
            mo_coeff = []
            while len(mo_coeff) < sum(self.other["funcs_per_shell"]):
                mo_coeff.extend([float(x) for x in lines[j].split()])
                j += 1
            self.other["alpha_coefficients"].append(mo_coeff)
        self.other["orbitals"] = Orbitals(self)

    def read_nbo_31(self, f, nbo_name=None):
        """read nbo .31 file"""
        lines = f.readlines()
        comment = lines[0].strip()
        info = lines[3].split()
        n_atoms = int(info[0])
        self.atoms = []
        for i in range(5, 5 + n_atoms):
            atom_info = lines[i].split()
            ele = ELEMENTS[int(atom_info[0])]
            coords = np.array([float(x) for x in atom_info[1:4]])
            self.atoms.append(
                Atom(
                    element=ele,
                    coords=coords,
                    name=str(i-4),
                )
            )

        i = n_atoms + 6
        line = lines[i]
        self.other["shell_to_atom"] = []
        self.other["momentum_label"] = []
        self.other["funcs_per_shell"] = []
        self.other["start_ndx"] = []
        self.other["n_prim_per_shell"] = []
        while "---" not in line:
            info = line.split()
            ndx = int(info[0])
            funcs = int(info[1])
            start_ndx = int(info[2])
            n_prim = int(info[3])
            self.other["shell_to_atom"].extend([ndx for j in range(0, funcs)])
            self.other["funcs_per_shell"].append(funcs)
            self.other["start_ndx"].append(start_ndx)
            self.other["n_prim_per_shell"].append(n_prim)
            i += 1
            line = lines[i]
            momentum_labels = [int(x) for x in line.split()]
            self.other["momentum_label"].extend(momentum_labels)
            i += 1
            line = lines[i]

        i += 1
        self.other["exponents"] = []
        line = lines[i]
        while line.strip() != "":
            exponents = [float(x) for x in line.split()]
            self.other["exponents"].extend(exponents)
            i += 1
            line = lines[i]

        i += 1
        self.other["s_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["s_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["p_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["p_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["d_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["d_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["f_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["f_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        i += 1
        self.other["g_coeff"] = []
        line = lines[i]
        while line.strip() != "":
            coeff = [float(x) for x in line.split()]
            self.other["g_coeff"].extend(coeff)
            i += 1
            line = lines[i]

        if nbo_name is not None:
            self._read_nbo_coeffs(nbo_name)
