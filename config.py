import configparser
import itertools as it
import os
import re
from getpass import getuser

import AaronTools
from AaronTools import addlogger
from AaronTools.const import AARONLIB, AARONTOOLS
from AaronTools.theory import (
    GAUSSIAN_COMMENT,
    GAUSSIAN_CONSTRAINTS,
    GAUSSIAN_COORDINATES,
    GAUSSIAN_GEN_BASIS,
    GAUSSIAN_GEN_ECP,
    GAUSSIAN_POST,
    GAUSSIAN_PRE_ROUTE,
    GAUSSIAN_ROUTE,
    ORCA_BLOCKS,
    ORCA_COMMENT,
    ORCA_COORDINATES,
    ORCA_ROUTE,
    PSI4_AFTER_JOB,
    PSI4_BEFORE_GEOM,
    PSI4_BEFORE_JOB,
    PSI4_COMMENT,
    PSI4_JOB,
    PSI4_MOLECULE,
    PSI4_OPTKING,
    PSI4_SETTINGS,
    SQM_COMMENT,
    SQM_QMMM,
    Theory,
)
from AaronTools.theory.implicit_solvent import ImplicitSolvent
from AaronTools.theory.job_types import (
    ForceJob,
    FrequencyJob,
    OptimizationJob,
    SinglePointJob,
)

THEORY_OPTIONS = [
    "GAUSSIAN_COMMENT",
    "GAUSSIAN_CONSTRAINTS",
    "GAUSSIAN_COORDINATES",
    "GAUSSIAN_GEN_BASIS",
    "GAUSSIAN_GEN_ECP",
    "GAUSSIAN_POST",
    "GAUSSIAN_PRE_ROUTE",
    "GAUSSIAN_ROUTE",
    "ORCA_BLOCKS",
    "ORCA_COMMENT",
    "ORCA_COORDINATES",
    "ORCA_ROUTE",
    "PSI4_AFTER_JOB",
    "PSI4_BEFORE_GEOM",
    "PSI4_BEFORE_JOB",
    "PSI4_COMMENT",
    "PSI4_MOLECULE",
    "PSI4_JOB",
    "PSI4_OPTKING",
    "PSI4_SETTINGS",
    "SQM_COMMENT",
    "SQM_QMMM",
]


@addlogger
class Config(configparser.ConfigParser):
    """
    Reads configuration information from INI files found at:
        $QCHASM/AaronTools/config.ini
        $AARONLIB/config.ini
        ./config.ini or /path/to/file supplied during initialization
    Access to configuration information available using dictionary notation.
        eg: self[`section_name`][`option_name`] returns `option_value`
    See help(configparser.ConfigParser) for more information
    """

    LOG = None
    SPEC_ATTRS = [
        "_changes",
        "_changed_list",
        "_args",
        "_kwargs",
        "infile",
        "metadata",
    ]

    @classmethod
    def _process_content(cls, filename, quiet=True):
        """
        process file content to handle optional default section header
        """
        contents = filename
        if os.path.isfile(filename):
            try:
                with open(filename) as f:
                    contents = f.read()
            except Exception as e:
                if not quiet:
                    cls.LOG.INFO("failed to read %s: %s", filename, e)
                return ""
        try:
            configparser.ConfigParser().read_string(contents)
        except configparser.MissingSectionHeaderError:
            contents = "[DEFAULT]\n" + contents
        return contents

    def __init__(
        self, infile=None, quiet=False, skip_user_default=False, **kwargs
    ):
        """
        infile: the configuration file to read
        quiet: prints helpful status information
        skip_user_default: change to True to skip importing user's default config files
        **kwargs: passed to initialization of parent class
        """
        configparser.ConfigParser.__init__(
            self, interpolation=None, comment_prefixes=("#"), **kwargs
        )
        self.infile = infile
        if not quiet:
            print("Reading configuration...")
        self._read_config(infile, quiet, skip_user_default)

        # enforce case-sensitivity in certain sections
        for section in self:
            if section in [
                "Substitution",
                "Mapping",
                "Configs",
                "Results",
            ]:
                continue
            for option, value in list(self[section].items()):
                if section == "Geometry" and option.lower().startswith(
                    "structure"
                ):
                    del self[section][option]
                    option = option.split(".")
                    option[0] = option[0].lower()
                    option = ".".join(option)
                    self[section][option] = value
                    continue
                if section == "Geometry" and "structure" in value.lower():
                    re.sub("structure", "structure", value, flags=re.I)
                    self[section][option] = value
                if option.lower() != option:
                    self[section][option.lower()] = value
                    del self[section][option]

        # handle included sections
        self._parse_includes()
        if infile is not None:
            self.read(infile)

        # set additional default values
        if infile:
            if "top_dir" not in self["DEFAULT"]:
                self["DEFAULT"]["top_dir"] = os.path.dirname(
                    os.path.abspath(infile)
                )
            if "name" not in self["DEFAULT"]:
                self["DEFAULT"]["name"] = ".".join(
                    os.path.relpath(
                        infile, start=self["DEFAULT"]["top_dir"]
                    ).split(".")[:-1]
                )
        else:
            if "top_dir" not in self["DEFAULT"]:
                self["DEFAULT"]["top_dir"] = os.path.abspath(os.path.curdir)

        # handle substitutions/mapping
        self._changes = {}
        self._changed_list = []
        self._parse_changes()

        # for passing to Theory(*args, **kwargs)
        self._args = []
        self._kwargs = {}

        # metadata is username and project name
        self.metadata = {
            "user": self.get(
                "DEFAULT",
                "user",
                fallback=self.get("HPC", "user", fallback=getuser()),
            ),
            "project": self.get("DEFAULT", "project", fallback=""),
        }

    def optionxform(self, option):
        return str(option)

    def __str__(self):
        rv = ""
        for section in self:
            if "." in section:
                continue
            rv += "[{}]\n".format(section)
            for option, value in self[section].items():
                rv += "{} = {}\n".format(option, value)
            rv += "\n"
        return rv

    def copy(self):
        config = Config(infile=None, quiet=True)
        for section in config.sections():
            config.remove_section(section)
        for option in list(config["DEFAULT"].keys()):
            config.remove_option("DEFAULT", option)
        for section in ["DEFAULT"] + self.sections():
            try:
                config.add_section(section)
            except (configparser.DuplicateSectionError, ValueError):
                pass
            for key, val in self[section].items():
                config[section][key] = val
        for section in self.SPEC_ATTRS:
            setattr(config, section, getattr(self, section))
        return config

    def for_change(self, change, structure=None):
        this_config = self.copy()
        if structure is not None:
            this_config["Job"]["name"] = structure.name
        if change:
            this_config["Job"]["name"] = os.path.join(
                change, this_config["Job"]["name"]
            )
        this_config._changes = {change: self._changes[change]}
        return this_config

    def _parse_changes(self):
        for section in ["Substitution", "Mapping"]:
            if section not in self:
                continue
            if self[section].getboolean("reopt", fallback=False):
                self._changes[""] = ({}, None)
            for key, val in self[section].items():
                if key in self["DEFAULT"]:
                    continue
                del self[section][key]
                key = "\n".join(["".join(k.split()) for k in key.split("\n")])
                val = "\n".join(["".join(v.split()) for v in val.split("\n")])
                self[section][key] = val
            for key, val in self[section].items():
                if key in self["DEFAULT"] or key == "reopt":
                    continue
                if "=" not in val:
                    val = [v.strip() for v in val.split(",")]
                else:
                    tmp = [v.strip() for v in val.split(";")]
                    val = []
                    for t in tmp:
                        t = t.strip()
                        if not t:
                            continue
                        elif "\n" in t:
                            val += t.split("\n")
                        else:
                            val += [t]
                    tmp = {}
                    for i, v in enumerate(val):
                        if i == 0 and len(v.split("=")) == 1:
                            v = "{}={}".format(key, v)
                            val[i] = v
                            del self[section][key]
                            key = ""
                            self[section]["~PLACEHOLDER~"] = ";".join(val)
                        v = v.split("=")
                        if (
                            not key.startswith("&combinations")
                            and "(" not in v[0]
                        ):
                            v[0] = v[0].split(",")
                        else:
                            v[0] = [v[0]]
                        for k in v[0]:
                            tmp[k] = v[1]
                    val = tmp
                # handle request for combinations
                if key.startswith("&combination"):
                    atoms = []
                    subs = []
                    # val <= { "2, 4": "H, CH3", "7, 9": "OH, NH2", .. }
                    for k, v in val.items():
                        if "(" not in k:
                            # regular substituents
                            atoms.append(k.split(","))
                        else:
                            # ring substitutions
                            atoms.append(re.findall("\(.*?\)", k))
                        subs.append([None] + [i for i in v.strip().split(",")])
                    # atoms <= [ [2, 4],         [7, 9],      .. ]
                    # subs  <= [ [None, H, CH3], [None, OH, NH2], .. ]
                    for combo in it.product(*[range(len(s)) for s in subs]):
                        # combos <= (0, 0,..), (0,.., 1),..(1,.. 0),..(1,.., 1),..
                        if not any(combo):
                            # skip if no substitutions
                            # (already included if reopt=True)
                            continue
                        name = []
                        tmp = {}
                        for i, p in enumerate(combo):
                            # don't add subsitution if sub == None
                            if subs[i][p] is None:
                                continue
                            name.append(subs[i][p])
                            for a in atoms[i]:
                                tmp[a] = subs[i][p]
                        name = "_".join(name)
                        self._changes[name] = (
                            tmp,
                            section,
                        )
                else:
                    if isinstance(val, list):
                        name = "_".join(val)
                        val = {key: ",".join(val)}
                    elif not key:
                        name = "_".join(
                            [
                                "_".join([v] * len(k.split(",")))
                                for k, v in val.items()
                            ]
                        )
                        self[section][name] = self[section]["~PLACEHOLDER~"]
                        del self[section]["~PLACEHOLDER~"]
                    else:
                        name = key
                    self._changes[name] = (val, section)

    def parse_functions(self):
        """
        Evaluates functions supplied in configuration file
        Functions indicated by "%{...}"
        Pulls in values of options indicated by $option_name
        Eg:
            ppn = 4
            memory = %{ $ppn * 2 }GB --> memory = 8GB
        """
        func_patt = re.compile("(%{(.*?)})")
        attr_patt = re.compile("\$([a-zA-Z0-9_:]+)")
        for section in ["DEFAULT"] + self.sections():
            # evaluate functions
            for key, val in self[section].items():
                match_list = func_patt.findall(val)
                while match_list:
                    match = match_list.pop()
                    eval_match = match[1]
                    for attr in attr_patt.findall(match[1]):
                        if ":" in attr:
                            from_section, option = attr.split(":")
                        else:
                            option, from_section = attr, section
                        option = self[from_section][option]
                        eval_match = eval_match.replace("$" + attr, option, 1)
                    try:
                        eval_match = eval(eval_match, {})
                    except TypeError as e:
                        raise TypeError(
                            "{} for\n\t[{}]\n\t{} = {}\nin config file. Could not evaluate {}".format(
                                e.args[0], section, key, val, eval_match
                            )
                        )
                    except (NameError, SyntaxError):
                        if attr_patt.findall(eval_match):
                            eval_match = "%{" + eval_match.strip() + "}"
                        else:
                            eval_match = eval_match.strip()
                    val = val.replace(match[0], str(eval_match))
                    self[section][key] = val
                    match_list = func_patt.findall(val)

    def getlist(self, section, option, *args, delim=",", **kwargs):
        """returns a list of option values by splitting on the delimiter specified by delim"""
        raw = self.get(section, option, *args, **kwargs)
        out = [x.strip() for x in raw.split(delim) if len(x.strip()) > 0]
        return out

    def read(self, filename, quiet=True):
        self.read_string(self._process_content(filename, quiet=quiet))

    def _read_config(self, infile, quiet, skip_user_default):
        """
        Reads configuration information from `infile` after pulling defaults
        """
        filenames = [
            os.path.join(AARONTOOLS, "config.ini"),
        ]
        if not skip_user_default:
            filenames += [os.path.join(AARONLIB, "config.ini")]
        if infile:
            filenames += [infile]
        local_only = False
        job_include = None
        for i, filename in enumerate(filenames):
            if not quiet:
                if os.path.isfile(filename):
                    print("    ✓", end="  ")
                else:
                    print("    ✗", end="  ")
                print(filename)
            content = self._process_content(filename)
            self.read(content, quiet=quiet)
            job_include = self.get("Job", "include", fallback=job_include)
            if filename != infile:
                self.remove_option("Job", "include")
            # local_only can only be overridden at the user level if "False" in the system config file
            if i == 0:
                local_only = self["DEFAULT"].getboolean("local_only")
            elif local_only:
                self["DEFAULT"]["local_only"] = str(local_only)
        if "Job" in self:
            type_spec = [
                re.search("(?<!_)type", option) for option in self["Job"]
            ]
        else:
            type_spec = []
        if not any(type_spec):
            self.set("Job", "include", job_include)

    def get_other_kwargs(self, section="Theory"):
        """
        Returns dict() that can be unpacked and passed to Geometry.write along with a theory
        Example:
        [Theory]
        route = pop NBORead
                opt MaxCycle=1000, NoEigenTest
        end_of_file = $nbo RESONANCE NBOSUM E2PERT=0.0 NLMO BNDIDX $end

        this adds opt(MaxCycle=1000,NoEigenTest) pop=NBORead to the route with any other
        pop or opt options being added by the job type

        'two-layer' options can also be specified as a python dictionary
        the following is equivalent to the above example:
        [Theory]
        route = {"pop":["NBORead"], "opt":["MaxCycle=1000", NoEigenTest"]}
        end_of_file = $nbo RESONANCE NBOSUM E2PERT=0.0 NLMO BNDIDX $end
        """
        # these need to be dicts
        two_layer = [GAUSSIAN_ROUTE, GAUSSIAN_PRE_ROUTE, ORCA_BLOCKS, PSI4_JOB]

        # these need to be dicts, but can only have one value
        two_layer_single_value = [
            PSI4_OPTKING,
            PSI4_SETTINGS,
            PSI4_MOLECULE,
        ]

        # these need to be lists
        one_layer = [
            GAUSSIAN_COMMENT,
            GAUSSIAN_CONSTRAINTS,
            GAUSSIAN_POST,
            ORCA_COMMENT,
            ORCA_ROUTE,
            PSI4_AFTER_JOB,
            PSI4_BEFORE_GEOM,
            PSI4_BEFORE_JOB,
            PSI4_COMMENT,
        ]

        theory_kwargs = [
            "method",
            "charge",
            "multiplicity",
            "type",
            "basis",
            "ecp",
        ]

        # two layer options are separated by newline
        # individual options are split on white space, with the first defining the primary layer
        out = {}
        for option in two_layer:
            value = self[section].get(option, fallback=False)
            value = self._kwargs.get(option, value)
            if value:
                if isinstance(value, dict):
                    out[option] = value
                elif "{" in value:
                    # if it's got brackets, it's probably a python-looking dictionary
                    # eval it instead of parsing
                    out[option] = eval(value, {})
                else:
                    out[option] = {}
                    for v in value.splitlines():
                        key = v.split()[0]
                        if len(v.split()) > 1:
                            info = v.split()[1].split(",")
                        else:
                            info = []

                        out[option][key] = [x.strip() for x in info]

        for option in two_layer_single_value:
            value = self.get(section, option, fallback=False)
            value = self._kwargs.get(option, value)
            if value:
                if "{" in value:
                    out[option] = eval(value, {})
                else:
                    out[option] = {}
                    for v in value.splitlines():
                        key = v.split()[0]
                        if len(v.split()) > 1:
                            info = [v.split()[1]]
                        else:
                            info = []

                        out[option][key] = [x.strip() for x in info]

        for option in one_layer:
            value = self[section].get(option, fallback=False)
            value = self._kwargs.get(option, value)
            if value:
                out[option] = value.splitlines()

        for option in theory_kwargs:
            value = self[section].get(option, fallback=False)
            if value:
                out[option] = value

        return out

    def get_constraints(self, geometry):
        constraints = {}
        try:
            con_list = re.findall("\(.*?\)", self["Geometry"]["constraints"])
        except KeyError:
            try:
                geometry.parse_comment()
                con_list = geometry.other["constraint"]
            except KeyError:
                raise RuntimeError(
                    "Constraints for forming/breaking bonds must be specified for TS search"
                )
        for con in con_list:
            tmp = []
            try:
                for c in eval(con, {}):
                    tmp += geometry.find(str(c))
            except TypeError:
                for c in con:
                    tmp += geometry.find(str(c))
            con = [a.name for a in tmp]
            if len(con) == 1:
                constraints.setdefault("atoms", [])
                constraints["atoms"] += [con]
            elif len(con) == 2:
                constraints.setdefault("bonds", [])
                constraints["bonds"] += [con]
            elif len(con) == 3:
                constraints.setdefault("angles", [])
                constraints["angles"] += [con]
            elif len(con) == 4:
                constraints.setdefault("torsions", [])
                constraints["torsions"] += [con]
        return constraints

    def get_theory(self, geometry, section="Theory"):
        """
        Get the theory object according to configuration information
        """
        if not self.has_section(section):
            self.LOG.warning(
                'config has no "%s" section, switching to "Theory"' % section
            )
            section = "Theory"

        kwargs = self.get_other_kwargs(section=section)
        theory = Theory(*self._args, geometry=geometry, **kwargs)

        # build ImplicitSolvent object
        if self[section].get("solvent", fallback="gas") == "gas":
            theory.solvent = None
        elif self[section]["solvent"]:
            theory.solvent = ImplicitSolvent(
                self[section]["solvent_model"],
                self[section]["solvent"],
            )

        # build JobType list
        job_type = self["Job"].get("type", fallback=False)
        if job_type:
            theory.job_type = []
            job_type = job_type.split(".")
            if "opt" in job_type[0] or "conf" in job_type[0]:
                constraints = None
                ts = False
                if len(job_type) > 1:
                    if "change" in job_type[1]:
                        theory.geometry.freeze()
                        theory.geometry.relax(self._changed_list)
                    elif "constrain" in job_type[1]:
                        constraints = self.get_constraints(theory.geometry)
                    elif "ts" == job_type[1]:
                        ts = True

                if "opt" in job_type[0] or "conf" in job_type[0]:
                    theory.job_type += [
                        OptimizationJob(
                            transition_state=ts,
                            geometry=geometry,
                            constraints=constraints,
                        )
                    ]
            if "freq" in job_type[0]:
                theory.job_type += [
                    FrequencyJob(
                        numerical=self[section].get(
                            "numerical", fallback=False
                        ),
                        temperature=self[section].get(
                            "temperature", fallback=None
                        ),
                    )
                ]
            if "single-point" in job_type or "SP" in job_type:
                theory.job_type += [SinglePointJob()]
            if "force" in job_type or "gradient" in job_type:
                theory.job_type += [
                    ForceJob(
                        numerical=self[section].get(
                            "numerical", fallback=False
                        ),
                    )
                ]
        else:
            # default to opt+freq
            theory.job_type = [
                OptimizationJob(geometry=geometry),
                FrequencyJob(
                    numerical=self[section].get("numerical", fallback=False),
                    temperature=self[section].get(
                        "temperature", fallback=None
                    ),
                ),
            ]
        # return updated theory object
        return theory

    def get_template(self):
        # captures name placeholder and iterator from for-loop initilaizer
        for_patt = re.compile("&for\s+(.+)\s+in\s+(.+)")
        # captures structure_dict-style structure/suffix -> (structure['suffix'], suffix)
        parsed_struct_patt = re.compile("(structure\['(\S+?)'\])\.?")
        # captures config-style structure/suffix -> (structure.suffix, suffix)
        structure_patt = re.compile("(structure\.(\S+?))\.?")

        def get_multiple(filenames, path=None, suffix=None):
            rv = []
            for name in filenames:
                kind = "Minimum"
                if name.startswith("TS"):
                    kind = "TS"
                if path is not None:
                    name = os.path.join(path, name)
                if not os.path.isfile(name):
                    continue
                geom = AaronTools.geometry.Geometry(name)
                if suffix is not None:
                    geom.name += ".{}".format(suffix)
                rv += [(geom, kind)]
            return rv

        def structure_assignment(line):
            # assignments must be done outside of eval()
            # left -> structure.suffix -> structure_dict["suffix"]
            # right -> eval(right)
            # left = right -> structure_dict[suffix] = eval(right)
            left = line.split("=")[0].strip()
            right = line.split("=")[1].strip()
            suffix_match = parsed_struct_patt.search(left)
            if suffix_match is None:
                raise RuntimeError(
                    "Can only assign to Geometry objects with names of the form `structure.suffix`"
                )
            suffix = suffix_match.group(2)
            structure_dict[suffix] = eval(right, eval_dict)
            structure_dict[suffix].name = ".".join(
                structure_dict[suffix].name.split(".")[:-1] + [suffix]
            )
            if structure_dict[suffix].name.startswith("TS"):
                kind_dict[suffix] = "TS"

        def structure_suffix_parse(line, for_loop=None):
            if for_loop is not None:
                for_match, it_val = for_loop
            for structure_match in structure_patt.findall(line):
                # if our suffix is not the iterator, keep it's value for the dict key
                if for_loop is None or structure_match[1] != for_match.group(
                    1
                ):
                    suffix = structure_match[1]
                else:
                    suffix = str(it_val)
                # change to dict-style syntax (structure.suffix -> structure["suffix"])
                line = line.replace(
                    structure_match[0],
                    "structure['{}']".format(suffix),
                )
                if suffix not in structure_dict:
                    structure_dict[suffix] = AaronTools.geometry.Geometry()
                    kind_dict[suffix] = None
            return line

        structure_dict = {}
        kind_dict = {}
        structure_list = []
        # load templates from AARONLIB
        if "Reaction" in self:
            path = None
            if "template" in self["Reaction"]:
                path = os.path.join(
                    AARONLIB,
                    "template_geoms",
                    self["Reaction"]["reaction"],
                    self["Reaction"]["template"],
                )
                for dirpath, dirnames, filenames in os.walk(path):
                    structure_list += get_multiple(filenames, path=dirpath)
            else:
                path = os.path.join(
                    AARONLIB,
                    "template_geoms",
                    self["Reaction"]["reaction"],
                )
                for dirpath, dirnames, filenames in os.walk(path):
                    structure_list += get_multiple(filenames, path=dirpath)
            for structure, kind in structure_list:
                structure.name = os.path.relpath(structure.name, path)
        if "Geometry" not in self:
            return structure_list

        # load templates from config[Geometry]
        # store in structure_dict, keyed by structure option suffix
        # `structure.suffix = geom.xyz` store as {suffix: geom.xyz}
        # `structure = geom.xyz` (no suffix), store as {"": geom.xyz}
        if "structure" in self["Geometry"]:
            structure_dict[""] = self["Geometry"]["structure"]
        else:
            for key in self["Geometry"]:
                if key.startswith("structure."):
                    suffix = ".".join(key.split(".")[1:])
                    structure_dict[suffix] = self["Geometry"][key]
        # create Geometry objects
        for suffix, structure in structure_dict.items():
            if structure is not None and os.path.isdir(structure):
                # if structure is a directory
                for dirpath, dirnames, filenames in os.walk(structure):
                    structure_list += get_multiple(
                        filenames, path=dirpath, suffix=suffix
                    )
            elif structure is not None:
                try:
                    # if structure is a filename
                    structure = AaronTools.geometry.Geometry(structure)
                except FileNotFoundError:
                    # if structure is a filename
                    structure = AaronTools.geometry.Geometry(
                        os.path.join(self["DEFAULT"]["top_dir"], structure)
                    )
                except (IndexError, NotImplementedError):
                    # if structure is a smiles string
                    structure = AaronTools.geometry.Geometry.from_string(
                        structure
                    )
                    self._changes[""] = ({}, None)
                # adjust structure attributes
                if "name" in self["Job"]:
                    structure.name = self["Job"]["name"]
                if "Geometry" in self and "comment" in self["Geometry"]:
                    structure.comment = self["Geometry"]["comment"]
                    structure.parse_comment()
                if suffix:
                    structure.name += ".{}".format(suffix)
                structure_dict[suffix] = structure
                kind_dict[suffix] = None

        # for loop for structure modification/creation
        # structure.suffix = geom.xyz
        # &for name in <iterator>:
        #    structure.name = structure.suffix.copy()
        #    structure.name.method_call(*args, **kwargs)

        if "Geometry" in self:
            for key in self["Geometry"]:
                if not key.startswith("&for"):
                    continue
                for_match = for_patt.search(key)
                if for_match is None:
                    raise SyntaxError(
                        "Malformed &for loop specification in config"
                    )
                lines = self["Geometry"][key].split("\n")
                for it_val in eval(for_match.group(2), {}):
                    eval_dict = {
                        "Geometry": AaronTools.geometry.Geometry,
                        "structure": structure_dict,
                        for_match.group(1): it_val,
                    }
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        line = structure_suffix_parse(
                            line,
                            for_loop=(for_match, it_val),
                        )
                        if "=" in line:
                            structure_assignment(line)
                        else:
                            eval(line, eval_dict)

        # add structure_dict to structure list
        for suffix in structure_dict:
            structure_list += [(structure_dict[suffix], kind_dict[suffix])]

        # apply functions found in [Geometry] section
        if "Geometry" in self and "&call" in self["Geometry"]:
            eval_dict = {
                "Geometry": AaronTools.geometry.Geometry,
                "structure": structure_dict,
            }
            lines = self["Geometry"]["&call"]
            for line in lines.split("\n"):
                line = structure_suffix_parse(line)
                if parsed_struct_patt.search(line.strip()):
                    try:
                        eval(line.strip(), eval_dict)
                    except SyntaxError:
                        structure_assignment(line)
                    for suffix in structure_dict:
                        val = (structure_dict[suffix], kind_dict[suffix])
                        if val not in structure_list:
                            structure_list += [val]
                elif line.strip():
                    for structure, kind in structure_list:
                        eval_dict["structure"] = structure
                        eval(line.strip(), eval_dict)
        return structure_list

    def make_changes(self, structure):
        if not self._changes:
            return structure
        changed = []
        for name, (changes, kind) in self._changes.items():
            for key, val in changes.items():
                if kind == "Substitution" and "(" not in key:
                    # regular substitutions
                    for k in key.split(","):
                        k = k.strip()
                        if val.lower() == "none":
                            structure -= structure.get_fragment(k)
                        else:
                            sub = structure.substitute(val, k)
                            for atom in sub:
                                changed += [atom.name]
                elif kind == "Substitution":
                    # fused ring substitutions
                    target_patt = re.compile("\((.*?)\)")
                    for k in target_patt.findall(key):
                        k = [i.strip() for i in k.split(",")]
                        if val.lower() == "none":
                            structure -= structure.get_fragment(*k)
                        else:
                            sub = structure.ring_substitute(k, val)
                            for atom in sub:
                                changed += [atom.name]
                elif kind == "Mapping":
                    key = [k.strip() for k in key.split(",")]
                    new_ligands = structure.map_ligand(val, key)
                    for ligand in new_ligands:
                        for atom in ligand:
                            changed += [atom.name]
        try:
            con_list = list(
                eval(self["Geometry"].get("constraints", "[]"), {})
            )
        except KeyError:
            structure.parse_comment()
            try:
                con_list = structure.other["constraint"]
            except KeyError:
                con_list = []
        for con in con_list:
            for c in con:
                try:
                    changed.remove(str(c))
                except ValueError:
                    pass
        self._changed_list = changed
        return structure

    def _parse_includes(self):
        """
        Moves option values from subsections into parent section
        Eg:
            [HPC]
            include = Wheeler
            ppn = 12

            [HPC.Wheeler]
            nodes = 1
            queue = wheeler_q

            ^^^evaluates to:
            [Job]
            nodes = 1
            ppn = 12
            queue = wheeler_q
        """
        for section in ["DEFAULT"] + self.sections():
            # add requested subsections to parent section
            if self.has_option(section, "include"):
                include_section = self[section]["include"].split(".")
                if include_section[0] in self.sections():
                    # include specifies full section name, eg:
                    # include = Job.Minimum --> [Job.Minimum]
                    include_section = ".".join(include_section)
                else:
                    # short-form of include, eg:
                    # [Job]
                    # include = Minimum
                    # --> [Job.Minimum]
                    include_section = [section] + include_section
                    include_section = ".".join(include_section)
                for key, val in self[include_section].items():
                    self[section][key] = val
            # handle non-default capitalization of default section
            if section.lower() == "default":
                for key, val in self[section].items():
                    self["DEFAULT"][key] = val

    def as_dict(self, spec=None, skip=None):
        """
        Forms a metadata spec dictionary from configuration info
        :spec: (dict) if given, append key/vals to that dict
        :skip: (list) skip storing stuff according to (section, option) or attrs
               section, option, and attrs are strings that can be regex (full match only)
               eg: skip=[("Job", ".*"), "conformer"] will skip everything in the Job
               section and the Config.conformer attribute
        """
        if spec is None:
            spec = {}
        if skip is None:
            skip = []
        skip_attrs = []
        skip_sections = []
        skip_options = []
        for s in skip:
            if isinstance(s, tuple):
                skip_sections.append(s[0])
                skip_options.append(s[1])
            else:
                skip_attrs.append(s)

        for attr in self.SPEC_ATTRS:
            for s in skip_attrs:
                if re.fullmatch(s, attr):
                    break
            else:
                spec[attr] = getattr(self, attr)

        for section in ["DEFAULT"] + self.sections():
            if "." in section:
                # these are include sections that should already be pulled into
                # the main body of the config file
                continue
            for option in self[section]:
                for i, s in enumerate(skip_sections):
                    o = skip_options[i]
                    if re.fullmatch(s, section) and re.fullmatch(o, option):
                        break
                else:
                    # only include default options once, unless they are overridden
                    if (
                        section != "DEFAULT"
                        and option in self["DEFAULT"]
                        and self["DEFAULT"][option] == self[section][option]
                    ):
                        continue
                    spec["{}/{}".format(section, option)] = self[section][
                        option
                    ]
        return spec

    def read_spec(self, spec):
        """
        Loads configuration metadata from spec dictionaries
        """
        for attr in spec:
            if attr in self.SPEC_ATTRS:
                setattr(self, attr, spec[attr])
            if "/" in attr:
                section, key = attr.split("/")
                if section not in self:
                    self.add_section(section)
                self[section][key] = spec[attr]

    def for_step(self, step=None):
        """
        Generates a config copy with only options for the given step
        """
        config = self.copy()
        # find step-specific options
        for section in ["DEFAULT"] + config.sections():
            for key, val in config[section].items():
                remove_key = key
                key = key.strip().split()
                if len(key) == 1:
                    continue
                key_step = float(key[0])
                key = " ".join(key[1:])
                # screen based on step
                if key_step == float(step):
                    config[section][key] = val
                # clean up metadata
                del config[section][remove_key]
        # other job-specific additions
        if config.has_section("HPC") and "host" in config["HPC"]:
            try:
                config["HPC"]["work_dir"] = config["HPC"].get("remote_dir")
            except TypeError as e:
                raise RuntimeError(
                    "Must specify remote working directory for HPC (remote_dir = /path/to/HPC/work/dir)"
                ) from e
        else:
            if not config.has_section("HPC"):
                config.add_section("HPC")
            config["HPC"]["work_dir"] = config["DEFAULT"].get("top_dir")
        # parse user-supplied functions in config file
        config.parse_functions()
        return config
