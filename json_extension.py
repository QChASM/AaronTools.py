import json
import numpy as np

from AaronTools.atoms import Atom
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.component import Component
from AaronTools.catalyst import Catalyst
from AaronTools.comp_output import CompOutput
from AaronTools.fileIO import Frequency


class JSONEncoder(json.JSONEncoder):
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
        else:
            super().default(obj)

    def _encode_atom(self, obj):
        """
        Encodes the data necessary to re-inialize an equivalent atom.

        Note, connectivity/constraint information is lost and must be
        re-initialized by the parent geometry through the use of the
        Geometry.refresh_connected and Geometry.parse_comment methods
        """
        rv = {'_type': obj.__class__.__name__}
        rv['element'] = obj.element
        rv['coords'] = list(obj.coords)
        rv['flag'] = obj.flag
        rv['name'] = obj.name
        rv['tags'] = list(sorted(obj.tags))
        return rv

    def _encode_geometry(self, obj):
        """
        Encodes data necessary to re-initialize a geometry object.
        """
        rv = {'_type': obj.__class__.__name__}

        # for Geometry and all child classes
        rv['name'] = obj.name
        rv['atoms'] = obj.atoms

        # for Geometry and all child classes but Substituent
        if hasattr(obj, 'comment'):
            rv['comment'] = obj.comment

        # for Substituent and Catalyst child classes
        if hasattr(obj, 'conf_num'):
            rv['conf_num'] = obj.conf_num

        # for Substituent child class
        if hasattr(obj, 'conf_angle'):
            rv['conf_angle'] = obj.conf_angle
        if hasattr(obj, 'end'):
            rv['end'] = obj.end

        return rv

    def _encode_comp_output(self, obj):
        rv = {'_type': obj.__class__.__name__}
        rv['geometry'] = obj.geometry
        rv['opts'] = obj.opts
        rv['frequency'] = obj.frequency
        rv['archive'] = obj.archive

        rv['E_ZPVE'] = obj.E_ZPVE
        rv['ZPVE'] = obj.ZPVE
        rv['energy'] = obj.energy
        rv['enthalpy'] = obj.enthalpy
        rv['free_energy'] = obj.free_energy
        rv['grimme_g'] = obj.grimme_g

        rv['mass'] = obj.mass
        rv['charge'] = obj.charge
        rv['multiplicity'] = obj.multiplicity
        rv['temperature'] = obj.temperature

        rv['gradient'] = obj.gradient
        rv['rotational_symmetry_number'] = obj.rotational_symmetry_number
        rv['rotational_temperature'] = obj.rotational_temperature

        rv['error'] = obj.error
        rv['error_msg'] = obj.error_msg
        rv['finished'] = obj.finished

        return rv

    def _encode_frequency(self, obj):
        rv = {'_type': obj.__class__.__name__}
        data = []
        for d in obj.data:
            entry = {}
            for k, v in d.__dict__.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                entry[k] = v
            data += [entry.copy()]
        rv['data'] = data
        return rv


class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '_type' not in obj:
            return obj
        if obj['_type'] == 'Atom':
            return self._decode_atom(obj)
        if obj['_type'] == 'Substituent':
            return self._decode_substituent(obj)
        if obj['_type'] in ['Geometry', 'Component', 'Catalyst']:
            return self._decode_geometry(obj)
        if obj['_type'] == 'Frequency':
            return self._decode_frequency(obj)

    def _decode_atom(self, obj):
        kwargs = {}
        for key in ['element', 'coords', 'flag', 'name', 'tags']:
            kwargs[key] = obj[key]
        return Atom(**kwargs)

    def _decode_geometry(self, obj):
        kwargs = {'structure': obj['atoms']}
        for key in ['name', 'comment']:
            kwargs[key] = obj[key]
        geom = Geometry(**kwargs)

        if obj['_type'] == 'Component':
            return Component(geom)
        elif obj['_type'] == 'Catalyst':
            kwargs = {'conf_num': obj['conf_num']}
            return Catalyst(geom, **kwargs)
        else:
            return geom

    def _decode_substituent(self, obj):
        kwargs = {}
        for key in ['name', 'end', 'conf_num', 'conf_angle']:
            kwargs[key] = obj[key]
        return Substituent(obj['atoms'], **kwargs)

    def _decode_frequency(self, obj):
        data = []
        for d in obj['data']:
            data += [Frequency.Data(d['frequency'],
                                    d['intensity'],
                                    d['vector'])]
        return Frequency(data)
