import json

from inspect import signature

import numpy as np

from AaronTools.atoms import Atom
from AaronTools.comp_output import CompOutput
from AaronTools.component import Component
from AaronTools.spectra import (
    Frequency,
    HarmonicVibration,
    AnharmonicVibration,
    ValenceExcitations,
    ValenceExcitation,
    TransientExcitation,
    SOCExcitation,
    NMR,
    Shift,
)
from AaronTools.finders import (
    Finder,
    AnyNonTransitionMetal,
    AnyTransitionMetal,
    NotAny,
    get_class,
)
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.theory import (
    Theory,
    ImplicitSolvent,
    OptimizationJob,
    FrequencyJob,
    ForceJob,
    SinglePointJob,
    NMRJob,
    BasisSet,
    Basis,
    ECP,
)


class ATEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Calls appropriate encoding method for supported AaronTools types.
        
        If type not supported, calls the default `default` method
        """
        if isinstance(obj, Atom):
            return self._encode_atom(obj)
        elif isinstance(obj, Geometry):
            return self._encode_geometry(obj)
        elif isinstance(obj, CompOutput):
            return self._encode_comp_output(obj)
        elif isinstance(obj, Frequency):
            return self._encode_frequency(obj)
        elif isinstance(obj, ValenceExcitations):
            return self._encode_valence_excitations(obj)
        elif isinstance(obj, HarmonicVibration) or isinstance(obj, AnharmonicVibration):
            return self._encode_vibration(obj)
        elif isinstance(obj, Theory):
            return self._encode_theory(obj)
        elif isinstance(obj, Finder):
            return self._encode_finder(obj)
        elif isinstance(obj, NMR):
            return self._encode_nmr(obj)
        else:
            super().default(obj)

    def _encode_atom(self, obj):
        """
        Encodes the data necessary to re-inialize an equivalent atom.

        Note: constraint information is lost and must be re-initialized by the
        parent geometry through the use of Geometry.parse_comment()
        """
        rv = {"_type": obj.__class__.__name__}
        rv["element"] = obj.element
        rv["coords"] = list(obj.coords)
        rv["flag"] = obj.flag
        rv["name"] = obj.name
        rv["tags"] = list(sorted(obj.tags))
        rv["charge"] = obj.charge
        rv["_rank"] = obj._rank
        return rv

    def _encode_geometry(self, obj):
        """
        Encodes data necessary to re-initialize a geometry object.
        """
        rv = {"_type": obj.__class__.__name__}

        # for Geometry and all child classes
        rv["name"] = obj.name
        rv["atoms"] = obj.atoms
        rv["connectivity"] = []
        for a in obj.atoms:
            rv["connectivity"] += [[obj.atoms.index(b) for b in a.connected]]

        # for Geometry and all child classes but Substituent
        if hasattr(obj, "comment"):
            rv["comment"] = obj.comment

        # for Catalyst child classes
        if isinstance(obj, Geometry) and obj.components:
            # comment
            obj.fix_comment()
            rv["comment"] = obj.comment

        # for Substituent child class
        if hasattr(obj, "conf_num"):
            rv["conf_num"] = obj.conf_num
        if hasattr(obj, "conf_angle"):
            rv["conf_angle"] = obj.conf_angle
        if hasattr(obj, "end"):
            rv["end"] = obj.end

        # for Component child class
        if hasattr(obj, "key_atoms"):
            rv["key_atoms"] = obj.key_atoms
        return rv

    def _encode_comp_output(self, obj):
        rv = {"_type": obj.__class__.__name__}
        rv["geometry"] = obj.geometry
        rv["opts"] = obj.opts
        rv["frequency"] = obj.frequency
        rv["archive"] = obj.archive

        rv["E_ZPVE"] = obj.E_ZPVE
        rv["ZPVE"] = obj.ZPVE
        rv["energy"] = obj.energy
        rv["enthalpy"] = obj.enthalpy
        rv["free_energy"] = obj.free_energy
        rv["grimme_g"] = obj.grimme_g

        rv["mass"] = obj.mass
        rv["charge"] = obj.charge
        rv["multiplicity"] = obj.multiplicity
        rv["temperature"] = obj.temperature

        rv["gradient"] = obj.gradient
        rv["rotational_symmetry_number"] = obj.rotational_symmetry_number
        rv["rotational_temperature"] = obj.rotational_temperature

        rv["error"] = obj.error
        rv["error_msg"] = obj.error_msg
        rv["finished"] = obj.finished

        return rv

    def _encode_frequency(self, obj):
        rv = {"_type": obj.__class__.__name__}
        data = []
        for d in obj.data:
            entry = {}
            for k, v in d.__dict__.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                entry[k] = v
            data += [entry.copy()]
        rv["data"] = data
        if obj.anharm_data:
            anharm_data = []
            for d in obj.anharm_data:
                entry = {}
                for k, v in d.__dict__.items():
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    entry[k] = v
                anharm_data += [entry.copy()]
            rv["anharm_data"] = anharm_data
        return rv

    def _encode_vibration(self, obj):
        rv = {"_type": obj.__class__.__name__}
        for k, v in obj.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            rv[k] = v
        return rv

    def _encode_valence_excitations(self, obj):
        rv = {"_type": obj.__class__.__name__}
        data = []
        for d in obj.data:
            entry = {}
            for k, v in d.__dict__.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                entry[k] = v
            data += [entry.copy()]
        rv["data"] = data
        if obj.transient_data:
            transient_data = []
            for d in obj.transient_data:
                entry = {}
                for k, v in d.__dict__.items():
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    entry[k] = v
                transient_data += [entry.copy()]
            rv["transient_data"] = transient_data
        if obj.spin_orbit_data:
            spin_orbit_data = []
            for d in obj.spin_orbit_data:
                entry = {}
                for k, v in d.__dict__.items():
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    entry[k] = v
                spin_orbit_data += [entry.copy()]
            rv["spin_orbit_data"] = spin_orbit_data
        return rv

    def _encode_nmr(self, obj):
        rv = {"_type": obj.__class__.__name__}
        data = []
        for d in obj.data:
            entry = {}
            for k, v in d.__dict__.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                entry[k] = v
            data += [entry.copy()]
        rv["data"] = data
        rv["coupling"] = obj.coupling
        rv["n_atoms"] = obj.n_atoms
        return rv

    def _encode_theory(self, obj):
        rv = {"_type": obj.__class__.__name__}
        
        if obj.method:
            rv["method"] = obj.method.name
            rv["semi-empirical"] = obj.method.is_semiempirical
        if obj.grid:
            rv["grid"] = obj.grid.name
        if obj.empirical_dispersion:
            rv["disp"] = obj.empirical_dispersion.name
        if obj.solvent:
            rv["solvent model"] = obj.solvent.solvent_model
            rv["solvent"] = obj.solvent.solvent
        if obj.processors:
            rv["nproc"] = obj.processors
        if obj.memory:
            rv["mem"] = obj.memory
        if obj.kwargs:
            rv["other"] = obj.kwargs

        if obj.job_type:
            rv["jobs"] = {}
            for job in obj.job_type:
                job_type = job.__class__.__name__
                rv["jobs"][job_type] = {}
                for arg in signature(job.__init__).parameters:
                    if arg == "self" or arg == "geometry" or "*" in arg:
                        continue
                    try:
                        rv["jobs"][job_type][arg] = getattr(job, arg)
                    except AttributeError:
                        pass
                
        if obj.basis:
            rv["basis"] = {"name": [], "elements":[], "file":[], "auxiliary":[]}
            if obj.basis.basis:
                for basis in obj.basis.basis:
                    rv["basis"]["name"].append(basis.name)
                    rv["basis"]["elements"].append([])
                    for ele in basis.ele_selection:
                        if isinstance(ele, str):
                            rv["basis"]["elements"][-1].append(ele)
                        elif isinstance(ele, AnyTransitionMetal):
                            rv["basis"]["elements"][-1].append("tm")
                        elif isinstance(ele, AnyNonTransitionMetal):
                            rv["basis"]["elements"][-1].append("!tm")
                            
                    if basis.not_anys:
                        for ele in basis.not_anys:
                            if isinstance(ele, str):
                                rv["basis"]["elements"][-1].append("!%s" % ele)
                            elif isinstance(ele, AnyTransitionMetal):
                                rv["basis"]["elements"][-1].append("!tm")
                            elif isinstance(ele, AnyNonTransitionMetal):
                                rv["basis"]["elements"][-1].append("!!tm")
    
    
                    rv["basis"]["file"].append(basis.user_defined)
                    rv["basis"]["auxiliary"].append(basis.aux_type)

            if obj.basis.ecp:
                rv["ecp"] = {"name": [], "elements":[], "file":[]}
                for basis in obj.basis.ecp:
                    rv["ecp"]["name"].append(basis.name)
                    rv["ecp"]["elements"].append([])
                    for ele in basis.ele_selection:
                        if isinstance(ele, str):
                            rv["ecp"]["elements"][-1].append(ele)
                        elif isinstance(ele, AnyTransitionMetal):
                            rv["ecp"]["elements"][-1].append("tm")
                        elif isinstance(ele, AnyNonTransitionMetal):
                            rv["ecp"]["elements"][-1].append("!tm")

                    if basis.not_anys:
                        for ele in basis.not_anys:
                            if isinstance(ele, str):
                                rv["ecp"]["elements"][-1].append("!%s" % ele)
                            elif isinstance(ele, AnyTransitionMetal):
                                rv["ecp"]["elements"][-1].append("!tm")
                            elif isinstance(ele, AnyNonTransitionMetal):
                                rv["ecp"]["elements"][-1].append("!!tm")
    
                    rv["ecp"]["file"].append(basis.user_defined)

            if obj.kwargs:
                rv["other"] = obj.kwargs

        return rv

    def _encode_finder(self, obj):
        try:
            get_class(obj.__class__.__name__)
        except ValueError:
            return None
        rv = {"_type": "Finder"}
        rv["_spec_type"] = obj.__class__.__name__
        rv["kwargs"] = obj.__dict__
        for kw in rv["kwargs"]:
            if isinstance(rv["kwargs"][kw], np.ndarray):
                rv["kwargs"][kw] = rv["kwargs"][kw].tolist()
        return rv


class ATDecoder(json.JSONDecoder):
    with_progress = False

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj
        if obj["_type"] == "Atom":
            return self._decode_atom(obj)
        if obj["_type"] == "Substituent":
            return self._decode_substituent(obj)
        if obj["_type"] in ["Geometry", "Component"]:
            return self._decode_geometry(obj)
        if obj["_type"] == "Frequency":
            return self._decode_frequency(obj)
        if obj["_type"] == "HarmonicVibration":
            return self._decode_vibration(obj, cls=HarmonicVibration)
        if obj["_type"] == "AnharmonicVibration":
            return self._decode_vibration(obj, cls=AnharmonicVibration)
        if obj["_type"] == "ValenceExcitations":
            return self._decode_valence_excitations(obj)
        if obj["_type"] == "CompOutput":
            return self._decode_comp_output(obj)
        if obj["_type"] == "Theory":
            return self._decode_theory(obj)
        if obj["_type"] == "Finder":
            return self._decode_finder(obj)
        if obj["_type"] == "NMR":
            return self._decode_nmr(obj)

    def _decode_atom(self, obj):
        kwargs = {}
        for key in ["element", "coords", "flag", "name", "tags", "charge"]:
            if key not in obj:
                continue
            kwargs[key] = obj[key]
        rv = Atom(**kwargs)
        rv._rank = obj["_rank"]
        return rv

    def _decode_geometry(self, obj):
        if ATDecoder.with_progress:
            print("Loading structure", obj["name"], " " * 50, end="\r")
        kwargs = {"structure": obj["atoms"]}
        for key in ["name", "comment"]:
            kwargs[key] = obj[key]
        geom = Geometry(**kwargs, refresh_connected=False, refresh_ranks=False)
        for i, connected in enumerate(obj["connectivity"]):
            for c in connected:
                geom.atoms[i].connected.add(geom.atoms[c])

        if obj["_type"] == "Component":
            key_atom_names = [a.name for a in obj["key_atoms"]]
            return Component(geom, key_atoms=key_atom_names)
        else:
            return geom

    def _decode_substituent(self, obj):
        kwargs = {}
        for key in ["name", "end", "conf_num", "conf_angle"]:
            kwargs[key] = obj[key]
        ranks = [a._rank for a in obj["atoms"]]
        obj = self._decode_geometry(obj)
        for a, r in zip(obj.atoms, ranks):
            a._rank = r
        return Substituent(obj, **kwargs)

    def _decode_frequency(self, obj):
        data = []
        for d in obj["data"]:
            kw = {k:v for k, v in d.items()}
            freq = kw.pop("frequency")
            kw["vector"] = np.array(kw["vector"])
            data += [
                HarmonicVibration(freq, **kw)
            ]
        freq_obj = Frequency(data)
        if "anharm_data" in obj:
            anharm_data = []
            for d in obj["anharm_data"]:
                kw = {k:v for k, v in d.items()}
                freq = kw.pop("frequency")
                anharm_data += [
                    AnharmonicVibration(freq, **kw)
                ]
            freq_obj.anharm_data = anharm_data
        return freq_obj

    def _decode_vibration(self, obj, cls=None):
        data = {}
        for k, v in obj.items():
            if k == "_type":
                continue
            if k == "frequency":
                continue
            data[k] = v
        if "vector" in data:
            data["vector"] = np.array(data["vector"])
        out = cls(obj["frequency"], **data)
        return out

    def _decode_valence_excitations(self, obj):
        data = []
        for d in obj["data"]:
            kw = {k:v for k, v in d.items()}
            excitation_energy = kw.pop("excitation_energy")
            data += [
                ValenceExcitation(excitation_energy, **kw)
            ]
        excitation_obj = ValenceExcitations(data)
        if "transient_data" in obj and obj["transient_data"]:
            transient_data = []
            for d in obj["transient_data"]:
                kw = {k:v for k, v in d.items()}
                excitation_energy = kw.pop("excitation_energy")
                transient_data += [
                    TransientExcitation(excitation_energy, **kw)
                ]
            excitation_obj.transient_data = transient_data
        if "spin_orbit_data" in obj and obj["spin_orbit_data"]:
            spin_orbit_data = []
            for d in obj["spin_orbit_data"]:
                kw = {k:v for k, v in d.items()}
                excitation_energy = kw.pop("excitation_energy")
                spin_orbit_data += [
                    TransientExcitation(excitation_energy, **kw)
                ]
            excitation_obj.spin_orbit_data = spin_orbit_data
        return excitation_obj

    def _decode_nmr(self, obj):
        data = []
        for d in obj["data"]:
            kw = {k: v for k, v in d.items()}
            shift = kw.pop("shift")
            data += [
                Shift(shift, **kw)
            ]
        nmr_obj = NMR(data, n_atoms=obj["n_atoms"])
        nmr_obj.coupling = {}
        for i in obj["coupling"]:
            nmr_obj.coupling.setdefault(int(i), {})
            for j in obj["coupling"][i]:
                nmr_obj.coupling[int(i)][int(j)] = obj["coupling"][i][j]
        return nmr_obj

    def _decode_comp_output(self, obj):
        keys = [
            "geometry",
            "opts",
            "frequency",
            "archive",
            "energy",
            "enthalpy",
            "free_energy",
            "grimme_g",
            "gradient",
            "E_ZPVE",
            "ZPVE",
            "mass",
            "temperature",
            "multiplicity",
            "charge",
            "rotational_temperature",
            "rotational_symmetry_number",
            "error",
            "error_msg",
            "finished",
        ]
        rv = CompOutput()
        for key in keys:
            rv.__dict__[key] = obj[key]
        return rv

    def _decode_theory(self, obj):
        rv = Theory()
        if "method" in obj:
            rv.method = obj["method"]
            if "semi-empirical" in obj:
                rv.method.is_semiempirical = obj["semi-empirical"]
        
        if "grid" in obj:
            rv.grid = obj["grid"]
        
        if "solvent model" in obj and "solvent" in obj:
            rv.solvent = ImplicitSolvent(obj["solvent model"], obj["solvent"])
        
        if "disp" in obj:
            rv.empirical_dispersion = obj["disp"]
        
        if "nproc" in obj:
            rv.processors = obj["nproc"]
        
        if "mem" in obj:
            rv.memory = obj["mem"]

        if "jobs" in obj:
            jobs = []
            for job in obj["jobs"]:
                if job == "OptimizationJob":
                    jobs.append(OptimizationJob(**obj["jobs"][job]))
                elif job == "FrequencyJob":
                    jobs.append(FrequencyJob(**obj["jobs"][job]))
                elif job == "SinglePointJob":
                    jobs.append(SinglePointJob(**obj["jobs"][job]))
                elif job == "ForceJob":
                    jobs.append(ForceJob(**obj["jobs"][job]))
                elif job == "NMRJob":
                    jobs.append(NMRJob(**obj["jobs"][job]))
            rv.job_type = jobs
        
        if "basis" in obj or "ecp" in obj:
            rv.basis = BasisSet([], [])
        
        if "basis" in obj:
            for name, aux_type, file, elements in zip(
                    obj["basis"]["name"],
                    obj["basis"]["auxiliary"],
                    obj["basis"]["file"],
                    obj["basis"]["elements"],
            ):
                rv.basis.basis.append(
                    Basis(
                        name,
                        elements=elements,
                        aux_type=aux_type,
                        user_defined=file,
                    )
                ) 

        if "ecp" in obj:
            for name, file, elements in zip(
                    obj["ecp"]["name"],
                    obj["ecp"]["file"],
                    obj["ecp"]["elements"],
            ):
                rv.basis.ecp.append(
                    ECP(
                        name,
                        elements=elements,
                        user_defined=file,
                    )
                )

        if "other" in obj:
            rv.kwargs = obj["other"]
        
        return rv

    def _decode_finder(self, obj):
        specific_type = obj["_spec_type"]
        kwargs = obj["kwargs"]
        try:
            cls = get_class(specific_type)
        except ValueError:
            return None
        args = []
        sig = signature(cls.__init__)
        for param in sig.parameters.values():
            if param.name in kwargs and (
                param.kind == param.POSITIONAL_ONLY or
                param.kind == param.POSITIONAL_OR_KEYWORD
            ):
                args.append(kwargs.pop(param.name))
        return cls(*args, **kwargs)
