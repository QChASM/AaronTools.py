<a href="https://badge.fury.io/py/AaronTools"><img src="https://badge.fury.io/py/AaronTools.svg" alt="PyPI version"></a>
<a href='https://aarontools.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/aarontools/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://pypi.org/project/AaronTools/1.1/"><img src="https://img.shields.io/pypi/dm/aarontools.svg"></a>
<a href="https://doi.org/10.1002/wcms.1510"><img src="https://img.shields.io/badge/DOI-10.1002/wcms.1510-blue"></a>

# AaronTools.py
AaronTools provides a collection of tools for automating routine tasks encountered when running quantum chemistry computations.

These tools can be used either directly within a Python script using AaronTools objects, or via a series of command-line scripts. 

## Documentation
AaronTools documenation, including tutorials on the Python API, can be found on our <a href="https://aarontools.readthedocs.io/en/latest/">Read-the-Docs Page</a>.

## Installation
* with pypi: <code>pip install AaronTools</code>

See the <a href="https://aarontools.readthedocs.io/en/latest/tutorials/install.html">installation guide</a> for more details, including how to manually install from this GitHub repository.

## Citation
If you use AaronTools, please cite:

V. M. Ingman, A. J. Schaefer, L. R. Andreola, and S. E. Wheeler "QChASM: Quantum Chemistry Automation and Structure Manipulation" <a href="http://dx.doi.org/10.1002/wcms.1510" target="_blank"><i>WIREs Comp. Mol. Sci.</i> <b>11</b>, e1510 (2021)</a>

## Features
AaronTools has a wide variety of features that can be accessed through the Python API as well as command line scripts (CLSs). These features include, but are not limited to:

* Calculating steric parameters
  * Sterimol Parameters, including Sterimol2Vec (<a href="https://aarontools.readthedocs.io/en/latest/cls/substituentSterimol.html">substituent CLS</a>, <a href="https://aarontools.readthedocs.io/en/latest/cls/ligandSterimol.html">ligand CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/substituent.py#L504">substituent API</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/component.py#L219">ligand API</a>)
  * Buried Volume (<a href="https://aarontools.readthedocs.io/en/latest/cls/percentVolumeBuried.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/geometry.py#L2408">API</a>)
  * Steric Maps (<a href="https://aarontools.readthedocs.io/en/latest/cls/stericMap.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/geometry.py#L2739">API</a>)
  * Ligand Solid Angle (<a href="https://aarontools.readthedocs.io/en/latest/cls/solidAngle.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/component.py#L901">API</a>)
  * Ligand Cone Angle - Tolman for asymmetric mono- and bidentate ligands, as well as exact cone angles (<a href="https://aarontools.readthedocs.io/en/latest/cls/coneAngle.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/component.py#L456">API</a>)
* Molecular Structure Editing
  * add or modify substituents (<a href="https://aarontools.readthedocs.io/en/latest/cls/substitute.html">single CLS</a>, <a href="https://aarontools.readthedocs.io/en/latest/cls/multiSubstitute.html">batch CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/geometry.py#L3979">single API</a>)
  * swap ligands (<a href="https://aarontools.readthedocs.io/en/latest/cls/mapLigand.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/geometry.py#L4907">API</a>)
* Generating Molecular Structures
  * coordination complexes (<a href="https://aarontools.readthedocs.io/en/latest/cls/getCoordinationComplexes.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/geometry.py#L271">API</a>)
  * organic molecules from SMILES or IUPAC (<a href="https://aarontools.readthedocs.io/en/latest/cls/fetchMolecule.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/geometry.py#L154">API</a>)
* Quantum Computation Setup and Processing
  * making input files for Gaussian, ORCA, Psi4, Q-Chem, xTB, and SQM (<a href="https://aarontools.readthedocs.io/en/latest/cls/makeInput.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/fileIO.py#L190">API</a>)
  * parsing data from output files (<a href="https://aarontools.readthedocs.io/en/latest/cls/printInfo.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/fileIO.py#L1077">API</a>)
  * thermochemistry based on Boltzmann-populated vibrational modes (<a href="https://aarontools.readthedocs.io/en/latest/cls/grabThermo.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/comp_output.py#L36">API</a>)
  * normal vibrational modes (<a href="https://aarontools.readthedocs.io/en/latest/cls/follow.html">displace along mode CLS</a>, <a href="https://aarontools.readthedocs.io/en/latest/cls/printFreq.html">print data CLS</a>, <a href="https://aarontools.readthedocs.io/en/latest/cls/plotIR.html">IR or VCD spectrum CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/spectra.py#L691">API</a>)
  * valence excitations (<a href="https://aarontools.readthedocs.io/en/latest/cls/plotUVVis.html">UV/vis or ECD spectrum CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/ff461166927faff684d6d16e4deb1e4a45375eae/spectra.py#L1327">API</a>)
  * orbitals, electron density, and Fukui functions (<a href="https://aarontools.readthedocs.io/en/latest/cls/printCube.html">CLS</a>, <a href="https://github.com/QChASM/AaronTools.py/blob/e5f218341e47c74e41df3340ab6a31d3cadcaf6a/orbitals.py#L14">API</a>)
* Parse data from output files of several popular quantum chemistry programs (<a href="https://aarontools.readthedocs.io/en/latest/tutorials/coding_with_filereaders.html">Read-the-Docs Tutorial</a>)
  * Gaussian
  * ORCA
  * Psi4
  * Q-Chem
  * NBO
  * xTB

Features are explained in more detail in the <a href="https://aarontools.readthedocs.io/en/latest/">documentation and in docstrings of the Python API. 


## Other Versions

### ChimeraX Plugin
The majority of these features are also available with a graphical interface in the <a href="https://cxtoolshed.rbvi.ucsf.edu/apps/seqcrow">SEQCROW plugin</a> for <a href="https://www.cgl.ucsf.edu/chimerax/">ChimeraX</a>.

### Perl
A Perl implementation of AaronTools is also <a href="https://github.com/QChASM/AaronTools">available here.</a>
However, users are <em>strongly urged</em> to use the Python version since it has far more powerful features and, unlike the Perl version, will continue to be developed and supported.


## Contact
If you have any questions or would like to discuss bugs or additional needed features, feel free to contact us at qchasm@uga.edu
