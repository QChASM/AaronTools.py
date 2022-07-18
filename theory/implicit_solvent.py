"""used to specify implicit solvent info for Theory()"""
from AaronTools import addlogger
from AaronTools.theory import (
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_ROUTE,
    PSI4_SETTINGS,
    PSI4_SOLVENT,
    XTB_COMMAND_LINE,
)


_KNOWN_GAUSSIAN_SOLVENTS = [
    "Water",
    "Acetonitrile",
    "Methanol",
    "Ethanol",
    "IsoQuinoline",
    "Quinoline",
    "Chloroform",
    "DiethylEther",
    "DichloroMethane",
    "DiChloroEthane",
    "CarbonTetraChloride",
    "Benzene",
    "Toluene",
    "ChloroBenzene",
    "NitroMethane",
    "Heptane",
    "CycloHexane",
    "Aniline",
    "Acetone",
    "TetraHydroFuran",
    "DiMethylSulfoxide",
    "Argon",
    "Krypton",
    "Xenon",
    "n-Octanol",
    "1,1,1-TriChloroEthane",
    "1,1,2-TriChloroEthane",
    "1,2,4-TriMethylBenzene",
    "1,2-DiBromoEthane",
    "1,2-EthaneDiol",
    "1,4-Dioxane",
    "1-Bromo-2-MethylPropane",
    "1-BromoOctane",
    "1-BromoPentane",
    "1-BromoPropane",
    "1-Butanol",
    "1-ChloroHexane",
    "1-ChloroPentane",
    "1-ChloroPropane",
    "1-Decanol",
    "1-FluoroOctane",
    "1-Heptanol",
    "1-Hexanol",
    "1-Hexene",
    "1-Hexyne",
    "1-IodoButane",
    "1-IodoHexaDecane",
    "1-IodoPentane",
    "1-IodoPropane",
    "1-NitroPropane",
    "1-Nonanol",
    "1-Pentanol",
    "1-Pentene",
    "1-Propanol",
    "2,2,2-TriFluoroEthanol",
    "2,2,4-TriMethylPentane",
    "2,4-DiMethylPentane",
    "2,4-DiMethylPyridine",
    "2,6-DiMethylPyridine",
    "2-BromoPropane",
    "2-Butanol",
    "2-ChloroButane",
    "2-Heptanone",
    "2-Hexanone",
    "2-MethoxyEthanol",
    "2-Methyl-1-Propanol",
    "2-Methyl-2-Propanol",
    "2-MethylPentane",
    "2-MethylPyridine",
    "2-NitroPropane",
    "2-Octanone",
    "2-Pentanone",
    "2-Propanol",
    "2-Propen-1-ol",
    "3-MethylPyridine",
    "3-Pentanone",
    "4-Heptanone",
    "4-Methyl-2-Pentanone",
    "4-MethylPyridine",
    "5-Nonanone",
    "AceticAcid",
    "AcetoPhenone",
    "a-ChloroToluene",
    "Anisole",
    "Benzaldehyde",
    "BenzoNitrile",
    "BenzylAlcohol",
    "BromoBenzene",
    "BromoEthane",
    "Bromoform",
    "Butanal",
    "ButanoicAcid",
    "Butanone",
    "ButanoNitrile",
    "ButylAmine",
    "ButylEthanoate",
    "CarbonDiSulfide",
    "Cis-1,2-DiMethylCycloHexane",
    "Cis-Decalin",
    "CycloHexanone",
    "CycloPentane",
    "CycloPentanol",
    "CycloPentanone",
    "Decalin-mixture",
    "DiBromoMethane",
    "DiButylEther",
    "DiEthylAmine",
    "DiEthylSulfide",
    "DiIodoMethane",
    "DiIsoPropylEther",
    "DiMethylDiSulfide",
    "DiPhenylEther",
    "DiPropylAmine",
    "E-1,2-DiChloroEthene",
    "E-2-Pentene",
    "EthaneThiol",
    "EthylBenzene",
    "EthylEthanoate",
    "EthylMethanoate",
    "EthylPhenylEther",
    "FluoroBenzene",
    "Formamide",
    "FormicAcid",
    "HexanoicAcid",
    "IodoBenzene",
    "IodoEthane",
    "IodoMethane",
    "IsoPropylBenzene",
    "m-Cresol",
    "Mesitylene",
    "MethylBenzoate",
    "MethylButanoate",
    "MethylCycloHexane",
    "MethylEthanoate",
    "MethylMethanoate",
    "MethylPropanoate",
    "m-Xylene",
    "n-ButylBenzene",
    "n-Decane",
    "n-Dodecane",
    "n-Hexadecane",
    "n-Hexane",
    "NitroBenzene",
    "NitroEthane",
    "n-MethylAniline",
    "n-MethylFormamide-mixture",
    "n,n-DiMethylAcetamide",
    "n,n-DiMethylFormamide",
    "n-Nonane",
    "n-Octane",
    "n-Pentadecane",
    "n-Pentane",
    "n-Undecane",
    "o-ChloroToluene",
    "o-Cresol",
    "o-DiChloroBenzene",
    "o-NitroToluene",
    "o-Xylene",
    "Pentanal",
    "PentanoicAcid",
    "PentylAmine",
    "PentylEthanoate",
    "PerFluoroBenzene",
    "p-IsoPropylToluene",
    "Propanal",
    "PropanoicAcid",
    "PropanoNitrile",
    "PropylAmine",
    "PropylEthanoate",
    "p-Xylene",
    "Pyridine",
    "sec-ButylBenzene",
    "tert-ButylBenzene",
    "TetraChloroEthene",
    "TetraHydroThiophene-s,s-dioxide",
    "Tetralin",
    "Thiophene",
    "Thiophenol",
    "trans-Decalin",
    "TriButylPhosphate",
    "TriChloroEthene",
    "TriEthylAmine",
    "Xylene-mixture",
    "Z-1,2-DiChloroEthene",
]

KNOWN_GAUSSIAN_SOLVENTS = {
    "SMD": _KNOWN_GAUSSIAN_SOLVENTS,
    "CPCM": _KNOWN_GAUSSIAN_SOLVENTS,
    "PCM": _KNOWN_GAUSSIAN_SOLVENTS,
    "DIPOLE": _KNOWN_GAUSSIAN_SOLVENTS,
    "IPCM": _KNOWN_GAUSSIAN_SOLVENTS,
    "ISODENSITY": _KNOWN_GAUSSIAN_SOLVENTS,
    "IEFPCM": _KNOWN_GAUSSIAN_SOLVENTS,
    "SCIPCM": _KNOWN_GAUSSIAN_SOLVENTS,
}

KNOWN_ORCA_SOLVENTS = {
    "SMD": [
        "1,1,1-TRICHLOROETHANE",
        "CYCLOPENTANE",
        "1,1,2-TRICHLOROETHANE",
        "CYCLOPENTANOL",
        "1,2,4-TRIMETHYLBENZENE",
        "CYCLOPENTANONE",
        "1,2-DIBROMOETHANE",
        "DECALIN (CIS/TRANS MIXTURE)",
        "1,2-DICHLOROETHANE",
        "CIS-DECALIN",
        "1,2-ETHANEDIOL",
        "N-DECANE",
        "1,4-DIOXANE",
        "DIBROMOMETHANE",
        "1-BROMO-2-METHYLPROPANE",
        "DIBUTYLETHER",
        "1-BROMOOCTANE",
        "O-DICHLOROBENZENE",
        "1-BROMOPENTANE",
        "E-1,2-DICHLOROETHENE",
        "1-BROMOPROPANE",
        "Z-1,2-DICHLOROETHENE",
        "1-BUTANOL",
        "DICHLOROMETHANE",
        "1-CHLOROHEXANE",
        "DIETHYL ETHER",
        "1-CHLOROPENTANE",
        "DIETHYL SULFIDE",
        "1-CHLOROPROPANE",
        "DIETHYLAMINE",
        "1-DECANOL",
        "DIIODOMETHANE",
        "1-FLUOROOCTANE",
        "DIISOPROPYL ETHER",
        "1-HEPTANOL",
        "CIS-1,2-DIMETHYLCYCLOHEXANE",
        "1-HEXANOL",
        "DIMETHYL DISULFIDE",
        "1-HEXENE",
        "N,N-DIMETHYLACETAMIDE",
        "1-HEXYNE",
        "N,N-DIMETHYLFORMAMIDE",
        "DMF",
        "1-IODOBUTANE",
        "DIMETHYLSULFOXIDE",
        "DMSO",
        "1-IODOHEXADECANE",
        "DIPHENYLETHER",
        "1-IODOPENTANE",
        "DIPROPYLAMINE",
        "1-IODOPROPANE",
        "N-DODECANE",
        "1-NITROPROPANE",
        "ETHANETHIOL",
        "1-NONANOL",
        "ETHANOL",
        "1-OCTANOL",
        "ETHYL ETHANOATE",
        "1-PENTANOL",
        "ETHYL METHANOATE",
        "1-PENTENE",
        "ETHYL PHENYL ETHER",
        "1-PROPANOL",
        "ETHYLBENZENE",
        "2,2,2-TRIFLUOROETHANOL",
        "FLUOROBENZENE",
        "2,2,4-TRIMETHYLPENTANE",
        "FORMAMIDE",
        "2,4-DIMETHYLPENTANE",
        "FORMIC ACID",
        "2,4-DIMETHYLPYRIDINE",
        "N-HEPTANE",
        "2,6-DIMETHYLPYRIDINE",
        "N-HEXADECANE",
        "2-BROMOPROPANE",
        "N-HEXANE",
        "2-BUTANOL",
        "HEXANOIC ACID",
        "2-CHLOROBUTANE",
        "IODOBENZENE",
        "2-HEPTANONE",
        "IODOETHANE",
        "2-HEXANONE",
        "IODOMETHANE",
        "2-METHOXYETHANOL",
        "ISOPROPYLBENZENE",
        "2-METHYL-1-PROPANOL",
        "P-ISOPROPYLTOLUENE",
        "2-METHYL-2-PROPANOL",
        "MESITYLENE",
        "2-METHYLPENTANE",
        "METHANOL",
        "2-METHYLPYRIDINE",
        "METHYL BENZOATE",
        "2-NITROPROPANE",
        "METHYL BUTANOATE",
        "2-OCTANONE",
        "METHYL ETHANOATE",
        "2-PENTANONE",
        "METHYL METHANOATE",
        "2-PROPANOL",
        "METHYL PROPANOATE",
        "2-PROPEN-1-OL",
        "N-METHYLANILINE",
        "E-2-PENTENE",
        "METHYLCYCLOHEXANE",
        "3-METHYLPYRIDINE",
        "N-METHYLFORMAMIDE (E/Z MIXTURE)",
        "3-PENTANONE",
        "NITROBENZENE",
        "PhNO2",
        "4-HEPTANONE",
        "NITROETHANE",
        "4-METHYL-2-PENTANONE",
        "NITROMETHANE",
        "MeNO2",
        "4-METHYLPYRIDINE",
        "O-NITROTOLUENE",
        "5-NONANONE",
        "N-NONANE",
        "ACETIC ACID",
        "N-OCTANE",
        "ACETONE",
        "N-PENTADECANE",
        "ACETONITRILE",
        "MeCN",
        "PENTANAL",
        "ACETOPHENONE",
        "N-PENTANE",
        "ANILINE",
        "PENTANOIC ACID",
        "ANISOLE",
        "PENTYL ETHANOATE",
        "BENZALDEHYDE",
        "PENTYLAMINE",
        "BENZENE",
        "PERFLUOROBENZENE",
        "BENZONITRILE",
        "PROPANAL",
        "BENZYL ALCOHOL",
        "PROPANOIC ACID",
        "BROMOBENZENE",
        "PROPANONITRILE",
        "BROMOETHANE",
        "PROPYL ETHANOATE",
        "BROMOFORM",
        "PROPYLAMINE",
        "BUTANAL",
        "PYRIDINE",
        "BUTANOIC ACID",
        "TETRACHLOROETHENE",
        "BUTANONE",
        "TETRAHYDROFURAN",
        "THF",
        "BUTANONITRILE",
        "TETRAHYDROTHIOPHENE-S,S-DIOXIDE",
        "BUTYL ETHANOATE",
        "TETRALIN",
        "BUTYLAMINE",
        "THIOPHENE",
        "N-BUTYLBENZENE",
        "THIOPHENOL",
        "SEC-BUTYLBENZENE",
        "TOLUENE",
        "TERT-BUTYLBENZENE",
        "TRANS-DECALIN",
        "CARBON DISULFIDE",
        "TRIBUTYLPHOSPHATE",
        "CARBON TETRACHLORIDE",
        "CCl4",
        "TRICHLOROETHENE",
        "CHLOROBENZENE",
        "TRIETHYLAMINE",
        "CHLOROFORM",
        "N-UNDECANE",
        "A-CHLOROTOLUENE",
        "WATER",
        "O-CHLOROTOLUENE",
        "XYLENE (MIXTURE)",
        "M-CRESOL",
        "M-XYLENE",
        "O-CRESOL",
        "O-XYLENE",
        "CYCLOHEXANE",
        "P-XYLENE",
        "CYCLOHEXANONE",
    ],
    "CPCM": [
        "Water",
        "Acetonitrile",
        "Acetone",
        "Ammonia",
        "Ethanol",
        "Methanol",
        "CH2Cl2",
        "CCl4",
        "DMF",
        "DMSO",
        "Pyridine",
        "THF",
        "Chloroform",
        "Hexane",
        "Benzene",
        "CycloHexane",
        "Octanol",
        "Toluene",
    ],
}

KNOWN_PSI4_SOLVENTS = {
    "CPCM": [
        "water",
        "propylene carbonate",
        "dimethylsolfoxide",
        "nitromethane",
        "aceotonitrile",
        "methanol",
        "ethanol",
        "acetone",
        "1,2-dichloroethane",
        "methylenechloride",
        "CH2Cl2",
        "tetrahydrofuran",
        "aniline",
        "chlorobenzene",
        "chloroform",
        "toluene",
        "1,4-dioxane",
        "benzene",
        "carbon tetrachloride",
        "cyclohexane",
        "n-heptane",
    ],
    "IEFPCM": [
        "water",
        "propylene carbonate",
        "dimethylsolfoxide",
        "nitromethane",
        "aceotonitrile",
        "methanol",
        "ethanol",
        "acetone",
        "1,2-dichloroethane",
        "methylenechloride",
        "CH2Cl2",
        "tetrahydrofuran",
        "aniline",
        "chlorobenzene",
        "chloroform",
        "toluene",
        "1,4-dioxane",
        "benzene",
        "carbon tetrachloride",
        "cyclohexane",
        "n-heptane",
    ],
}

KNOWN_XTB_SOLVENTS = {
    "ALPB": [
        "acetone",
        "acetonitrile",
        "aniline",
        "benzaldehyde",
        "benzene",
        "ch2cl2",
        "chcl3",
        "cs2",
        "dioxane",
        "dmf",
        "dmso",
        "ether",
        "ethylacetate",
        "furane",
        "hexandecane",
        "hexane",
        "methanol",
        "nitromethane",
        "octanol",
        "woctanol",
        "phenol",
        "toluene",
        "thf",
        "water",
    ],
    "GBSA": [
        "acetone",
        "acetonitrile",
        "benzene",
        "CH2Cl2",
        "CHCl3",
        "CS2",
        "DMF",
        "DMSO",
        "ether",
        "H2O",
        "methanol",
        "n-hexane",
        "THF",
        "toluene",
    ]
}


class ImplicitSolvent:
    """implicit solvent info"""

    LOG = None

    def __init__(self, solvent_model, solvent):
        self.solvent_model = solvent_model
        self.solvent = solvent

    def __repr__(self):
        return "%s(%s)" % (self.solvent_model.upper(), self.solvent.lower())

    def __eq__(self, other):
        return repr(self) == repr(other)

    def copy(self):
        new_dict = dict()
        for key, value in self.__dict__.items():
            try:
                new_dict[key] = value.copy()
            except AttributeError:
                new_dict[key] = value
                # ignore chimerax objects so seqcrow doesn't print a
                # warning when a geometry is copied
                if "chimerax" in value.__class__.__module__:
                    continue
                if value.__class__.__module__ != "builtins":
                    self.LOG.warning(
                        "No copy method for {}: in-place changes may occur".format(
                            type(value)
                        )
                    )
        
        return self.__class__(**new_dict)
 
    def get_gaussian(self):
        """returns dict() with solvent information for gaussian input files"""
        # need to check if solvent model is available
        warnings = []
        if self.solvent.lower() == "gas":
            # all gas, no solvent
            return (dict(), warnings)

        model = self.solvent_model.upper()

        if not any(
            model == x for x in KNOWN_GAUSSIAN_SOLVENTS.keys()
        ):
            warnings.append(
                "solvent model is not available in Gaussian: %s\nuse one of: %s"
                % (
                    self.solvent_model,
                    " ".join(list(KNOWN_GAUSSIAN_SOLVENTS.keys()))
                )
            )

        # check some orca solvent keywords and switch to gaussian ones
        solvent = self.solvent
        if solvent.lower() == "ch2cl2":
            solvent = "DiChloroMethane"
        elif solvent.lower() == "ccl4":
            solvent = "CarbonTetraChloride"
        elif solvent.lower() == "THF":
            solvent = "TetraHydroFuran"
        else:
            if not any(
                    solvent.lower() == gaussian_sol.lower()
                    for gaussian_sol in KNOWN_GAUSSIAN_SOLVENTS[model]
            ):
                warnings.append(
                    "solvent is unknown to Gaussian: %s\n" % solvent +
                    "see AaronTools.theory.implicit_solvent.KNOWN_GAUSSIAN_SOLVENTS"
                )

        # route option: scrf(model,solvent=solvent solvent_model)
        return (
            {GAUSSIAN_ROUTE: {"scrf": [self.solvent_model, "solvent=%s" % solvent]}},
            warnings,
        )

    def get_orca(self):
        """returns dict() with solvent information for orca input files"""
        warnings = []
        if self.solvent.lower() == "gas":
            return (dict(), warnings)

        model = self.solvent_model.upper()

        if not any(
            model == x for x in KNOWN_ORCA_SOLVENTS.keys()
        ):
            warnings.append(
                "solvent model is not available in ORCA: %s\nuse one of: %s"
                % (
                    self.solvent_model,
                    " ".join(list(KNOWN_ORCA_SOLVENTS.keys()))
                )
            )

        out = {}
        cpcm = True
        # route option: CPCM(solvent name)
        # if using smd, add block %cpcm smd true end
        if model == "SMD":
            cpcm = False
            out[ORCA_BLOCKS] = {"cpcm": ["smd    true"]}

        solvent = self.solvent
        # orca has different solvents for cpcm and smd...
        # check both lists, might be able to switch a gaussian keyword to the correct orca one
        if cpcm:
            if solvent.lower() == "dichloromethane":
                solvent = "CH2Cl2"
            elif solvent.lower() == "carbontetrachloride":
                solvent = "CCl4"
            elif solvent.lower() == "tetrahydrofuran":
                solvent = "THF"
            else:
                if not any(
                        solvent.lower() == orca_sol.lower()
                        for orca_sol in KNOWN_ORCA_SOLVENTS[model]
                ):
                    warnings.append(
                        "solvent is unknown to ORCA: %s\n" % solvent +
                        "see AaronTools.theory.implicit_solvent.KNOWN_ORCA_SOLVENTS"
                    )

        else:
            # TODO: look for gaussian/orca pcm solvent names that need to change
            if not any(
                    solvent.lower() == orca_sol.lower()
                    for orca_sol in KNOWN_ORCA_SOLVENTS[model]
            ):
                warnings.append(
                    "solvent is unknown to ORCA: %s\n" % solvent +
                    "see AaronTools.theory.implicit_solvent.KNOWN_ORCA_SOLVENTS"
                )

        out[ORCA_ROUTE] = ["CPCM(%s)" % solvent]

        return (out, warnings)

    def get_psi4(self):
        """returns dict() with solvent information for psi4 input files"""

        warnings = []

        if self.solvent.lower() == "gas":
            return (dict(), warnings)

        model = self.solvent_model.upper()

        if not any(
            model == x for x in KNOWN_PSI4_SOLVENTS.keys()
        ):
            warnings.append(
                "solvent model is not available in Psi4: %s\nuse one of: %s"
                % (
                    self.solvent_model,
                    " ".join(list(KNOWN_PSI4_SOLVENTS.keys()))
                )
            )

        out = {}
        # route option: CPCM(solvent name)
        # if using smd, add block %cpcm smd true end
        out[PSI4_SETTINGS] = {"pcm": "true"}

        solvent = self.solvent
        # orca has different solvents for cpcm and smd...
        # check both lists, might be able to switch a gaussian keyword to the correct orca one
        if solvent.lower() == "dichloromethane":
            solvent = "CH2Cl2"
        elif solvent.lower() == "carbontetrachloride":
            solvent = "carbon tetrachloride"
        elif solvent.lower() == "thf":
            solvent = "tetrahydrofuran"
        else:
            if not any(
                    solvent.lower() == psi4_sol.lower()
                    for psi4_sol in KNOWN_PSI4_SOLVENTS[model]
            ):
                warnings.append(
                    "solvent may be unknown to Psi4: %s\n" % solvent +
                    "see AaronTools.theory.implicit_solvent.KNOWN_PSI4_SOLVENTS"
                )

        out[PSI4_SOLVENT] = {
            "Units": "Angstrom",
            "Medium":
                [
                    "SolverType = %s" % self.solvent_model,
                    "Solvent = \"%s\"" % solvent,
                ],
            "Mode": "Implicit"
        }

        return (out, warnings)

    def get_qchem(self):
        """returns dict() with solvent information for qchem input files"""

        warnings = []

        if self.solvent.lower() == "gas":
            return (dict(), warnings)

        raise NotImplementedError(
            "cannot create solvent info for Q-Chem input files"
        )

    def get_xtb(self):
        warnings = []
        out = dict()
        if self.solvent.lower() == "gas":
            # all gas, no solvent
            return (dict(), warnings)
        
        model = self.solvent_model.upper()

        if not any(
            model == x for x in KNOWN_XTB_SOLVENTS.keys()
        ):
            warnings.append(
                "solvent model is not available in Psi4: %s\nuse one of: %s"
                % (
                    self.solvent_model,
                    " ".join(list(KNOWN_XTB_SOLVENTS.keys()))
                )
            )

        if not any(
            xtb_sol.lower() == self.solvent.lower() for xtb_sol in KNOWN_XTB_SOLVENTS[model]
        ):
            warnings.append(
                "solvent may be unknown to xTB: %s\n" % self.solvent +
                "see AaronTools.theory.implicit_solvent.KNOWN_XTB_SOLVENTS"
            )
        
        out[XTB_COMMAND_LINE] = {self.solvent_model.lower(): [self.solvent]}
        return out, warnings
        