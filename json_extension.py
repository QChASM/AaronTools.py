import json

from inspect import signature

import numpy as np

from AaronTools.atoms import Atom
from AaronTools.comp_output import CompOutput
from AaronTools.component import Component
from AaronTools.fileIO import Frequency
from AaronTools.finders import AnyNonTransitionMetal, AnyTransitionMetal, NotAny
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.theory import (
    Theory,
    ImplicitSolvent,
    OptimizationJob,
    FrequencyJob,
    ForceJob,
    SinglePointJob,
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
        elif isinstance(obj, Theory):
            return self._encode_theory(obj)
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
            rv["solvent model"] = obj.solvent.name
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
                    for ele in basis.ele_selection:
                        if isinstance(ele, str):
                            rv["ecp"]["elements"].append(ele)
                        elif isinstance(ele, AnyTransitionMetal):
                            rv["ecp"]["elements"].append("tm")
                        elif isinstance(ele, AnyNonTransitionMetal):
                            rv["ecp"]["elements"].append("!tm")
                        else:
                            rv["ecp"]["elements"].append([])
                            
                    if basis.not_anys:
                        for ele in basis.not_anys:
                            if isinstance(ele, str):
                                rv["ecp"]["elements"][-1].append("!%s" % ele)
                            elif isinstance(ele, AnyTransitionMetal):
                                rv["ecp"]["elements"][-1].append("!tm")
                            elif isinstance(ele, AnyNonTransitionMetal):
                                rv["ecp"]["elements"][-1].append("tm")
    
                    rv["ecp"]["file"].append(basis.user_defined)

            if obj.kwargs:
                rv["other"] = obj.kwargs

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
        if obj["_type"] == "CompOutput":
            return self._decode_comp_output(obj)
        if obj["_type"] == "Theory":
            return self._decode_theory(obj)

    def _decode_atom(self, obj):
        kwargs = {}
        for key in ["element", "coords", "flag", "name", "tags"]:
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
            data += [
                Frequency.Data(d["frequency"], d["intensity"], d["vector"])
            ]
        return Frequency(data)

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
            "frequency",
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
