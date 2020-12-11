import configparser
import itertools as it
import os
import re
import sys
from copy import deepcopy
from warnings import warn

import AaronTools
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
    PSI4_COORDINATES,
    PSI4_JOB,
    PSI4_OPTKING,
    PSI4_SETTINGS,
    Theory,
)
from AaronTools.theory.implicit_solvent import ImplicitSolvent
from AaronTools.theory.job_types import (
    FrequencyJob,
    OptimizationJob,
    SinglePointJob,
)

SECTIONS = ["DEFAULT", "HPC", "Job", "Substitution", "Mapping", "Reaction"]
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
    "PSI4_COORDINATES",
    "PSI4_JOB",
    "PSI4_OPTKING",
    "PSI4_SETTINGS",
]


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

    def __init__(self, infile="config.ini", quiet=False, **kwargs):
        """
        infile: the configuration file to read
        quiet: prints helpful status information
        **kwargs: passed to initialization of parent class
        """
        configparser.ConfigParser.__init__(
            self, interpolation=None, comment_prefixes=("#"), **kwargs
        )
        if not quiet:
            print("Reading configuration...")
        if infile is not None:
            self.read_config(infile, quiet)
            # enforce selective case sensitivity
            for section in self:
                if section in [
                    "Substitution",
                    "Mapping",
                    "Configs",
                    "Results",
                ]:
                    continue
                for option, value in self[section].items():
                    if option.lower() != option:
                        self[section][option.lower()] = value
                        del self[section][option]
            # handle included sections
            self._parse_includes()
            # set additional default values
            if "top_dir" not in self["DEFAULT"]:
                self["DEFAULT"]["top_dir"] = os.path.dirname(
                    os.path.abspath(infile)
                )
            if "name" not in self["DEFAULT"]:
                self["DEFAULT"]["name"] = ".".join(infile.split(".")[:-1])
        # handle substitutions/mapping
        self._changes = {}
        self._changed_list = []
        self._parse_changes()
        # for passing to Theory(*args, **kwargs)
        self._args = []
        self._kwargs = {}

    def optionxform(self, option):
        return str(option)

    def _parse_changes(self):
        for section in ["Substitution", "Mapping"]:
            if section not in self:
                continue
            if "reopt" in self[section] and self.getboolean(section, "reopt"):
                self._changes[""] = {}, None
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
                    for v in val:
                        v = v.strip().split("=")
                        tmp[v[0].strip()] = v[1].strip()
                    val = tmp
                # handle request for all combinations
                if key.startswith("&combination"):
                    atoms = []
                    subs = []
                    # val <= { "2, 4": "H, CH3", "7, 9": "OH, NH2", .. }
                    for k, v in val.items():
                        atoms.append([i.strip() for i in k.split(",")])
                        subs.append(
                            [None] + [i.strip() for i in v.strip().split(",")]
                        )
                    # atoms = [ [2, 4],         [7, 9],      .. ]
                    # subs  = [ [None, H, CH3], [None, OH, NH2], .. ]
                    for combo in it.product(*[range(len(s)) for s in subs]):
                        # combos = (0, 0,..), (0,.., 1),..(1,.. 0),..(1,.., 1),..
                        if not any(combo):
                            # skip if no substitutions
                            continue
                        name = []
                        combo_key = []
                        combo_val = []
                        for i, p in enumerate(combo):
                            # don't add subsitution if sub == None
                            if subs[i][p] is None:
                                continue
                            name.append(subs[i][p])
                            combo_key.append(",".join(atoms[i]))
                            combo_val.append(subs[i][p])
                        name = "_".join(name)
                        self._changes[name] = (
                            dict(zip(combo_key, combo_val)),
                            section,
                        )
                else:
                    if isinstance(val, list):
                        val = {key: ",".join(val)}
                        key = ""
                    self._changes[key] = (val, section)

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
        for attr in self._sections:
            config.add_section(attr)
            for key, val in self[attr].items():
                config[attr][key] = val
        for attr in [
            "_interpolation",
            "_changes",
            "_changed_list",
            "_args",
            "_kwargs",
        ]:
            config.__dict__[attr] = self.__dict__[attr]
        return config

    def parse_functions(self):
        """
        Evaluates functions supplied in configuration file
        Functions indicated by "%{...}"
        Pulls in values of options indicated by $option_name
        Eg:
            ppn = 4
            memory = %{ $ppn * 2 }GB --> memory = 8GB
        """
        func_patt = re.compile("%{(.*?)}")
        attr_patt = re.compile("\$([a-zA-Z0-9_:]+)")
        for section in self._sections:
            # evaluate functions
            for key, val in self[section].items():
                for match in func_patt.findall(val):
                    eval_match = match
                    for attr in attr_patt.findall(match):
                        if ":" in attr:
                            option, from_section = attr.split(":")
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
                        eval_match = eval_match.strip()
                    val = val.replace("%{" + match + "}", str(eval_match))
                    self[section][key] = val

    def getlist(self, section, option, *args, delim=",", **kwargs):
        """returns a list of option values by splitting on the delimiter specified by delim"""
        raw = self.get(section, option, *args, **kwargs)
        out = [x.strip() for x in raw.split(delim) if len(x.strip()) > 0]
        return out

    def read_config(self, infile, quiet):
        """
        Reads configuration information from `infile` after pulling defaults
        """
        for filename in [
            os.path.join(AARONTOOLS, "config.ini"),
            os.path.join(AARONLIB, "config.ini"),
            infile,
        ]:
            if not quiet:
                if os.path.isfile(filename):
                    print("    ✓", end="  ")
                else:
                    print("    ✗", end="  ")
                print(filename)
            try:
                # if infile is multi-line, it's probably a string and not a file name
                if len(filename.splitlines()) > 1:
                    self.read_string(filename)
                else:
                    success = self.read(filename)
                    if not quiet and len(success) == 0:
                        print("failed to read %s" % filename)
            except configparser.MissingSectionHeaderError:
                # add global options to default section
                with open(filename) as f:
                    contents = "[DEFAULT]\n" + f.read()
                self.read_string(contents)

    def get_other_kwargs(self, section="Job"):
        """
        Returns dict() that can be unpacked and passed to Geometry.write along with a theory
        Example:
        [Job]
        route = pop NBORead
                opt MaxCycle=1000, NoEigenTest
        end_of_file = $nbo RESONANCE NBOSUM E2PERT=0.0 NLMO BNDIDX $end

        this adds opt(MaxCycle=1000,NoEigenTest) pop=NBORead to the route with any other
        pop or opt options being added by the job type

        'two-layer' options can also be specified as a python dictionary
        the following is equivalent to the above example:
        [Job]
        route = {"pop":["NBORead"], "opt":["MaxCycle=1000", NoEigenTest"]}
        end_of_file = $nbo RESONANCE NBOSUM E2PERT=0.0 NLMO BNDIDX $end
        """
        # these need to be dicts
        two_layer = [GAUSSIAN_ROUTE, GAUSSIAN_PRE_ROUTE, ORCA_BLOCKS, PSI4_JOB]

        # these need to be dicts, but can only have one value
        two_layer_single_value = [
            PSI4_OPTKING,
            PSI4_SETTINGS,
            PSI4_COORDINATES,
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

        job_kwargs = [
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
            if value:
                # if it's got brackets, it's probably a python-looking dictionary
                # eval it instead of parsing
                if "{" in value:
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
            if value:
                out[option] = value.splitlines()

        for option in job_kwargs:
            value = self[section].get(option, fallback=False)
            if value:
                out[option] = value

        return out

    def get_theory(self, geometry, section="Job"):
        """
        Get the theory object according to configuration information
        """
        if not self.has_section(section):
            warn('config has no "%s" section, switching to "Job"' % section)
            section = "Job"

        kwargs = self.get_other_kwargs(section=section)
        # kwargs = {}
        # for key, val in self[section].items():
        #     if key.upper() in THEORY_OPTIONS:
        #         kwargs[key.upper()] = eval(val)
        #     else:
        #         kwargs[key] = val

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
        job_type = self[section].get("type", fallback=False)
        if job_type:
            theory.job_type = []
            job_type = job_type.split(".")
            if "opt" in job_type[0]:
                constraints = None
                ts = False
                if len(job_type) > 1:
                    if "change" in job_type[1]:
                        theory.geometry.freeze()
                        theory.geometry.relax(self._changed_list)
                    elif "constrain" in job_type[1]:
                        constraints = {}
                        try:
                            con_list = re.findall(
                                "\(.*?\)", self["Geometry"]["constraints"]
                            )
                        except KeyError:
                            try:
                                theory.geometry.parse_comment()
                                con_list = theory.geometry.other["constraint"]
                            except KeyError:
                                raise RuntimeError(
                                    "Constraints for forming/breaking bonds must be specified for TS search"
                                )
                        for con in con_list:
                            try:
                                con = tuple(
                                    geometry.find(str(c))[0]
                                    for c in eval(con, {})
                                )
                            except TypeError:
                                con = tuple(
                                    geometry.find(str(c))[0] for c in con
                                )
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
                    elif "ts" == job_type[1]:
                        ts = True

                theory.job_type += [
                    OptimizationJob(
                        transition_state=ts,
                        geometry=geometry,
                        constraints=constraints,
                    )
                ]
            if "freq" in job_type[0]:
                if self[section].get("temperature", fallback=False):
                    theory.job_type += [
                        FrequencyJob(temperature=self[section]["temperature"])
                    ]
                else:
                    theory.job_type += [FrequencyJob()]
            if "single-point" in job_type or "SP" in job_type:
                theory.job_type += [SinglePointJob()]
        else:
            theory.job_type = [SinglePointJob()]
        # return updated theory object
        return theory

    def get_template(self):
        if "Reaction" in self:
            structures = []
            if "template" in self["Reaction"]:
                path = os.path.join(
                    AARONLIB,
                    "TS_geoms",
                    self["Reaction"]["reaction"],
                    self["Reaction"]["template"],
                )
                for dirpath, dirnames, filenames in os.walk(path):
                    for name in filenames:
                        if name.startswith("TS"):
                            kind = "TS"
                        elif name.startswith("INT"):
                            kind = "Minimum"
                        name = os.path.join(dirpath, name)
                        structure = AaronTools.geometry.Geometry(name)
                        structure.name = os.path.relpath(name, path)
                        structure.name = ".".join(
                            structure.name.split(".")[:-1]
                        )
                        structures += [(structure, kind)]
            else:
                path = os.path.join(
                    AARONLIB,
                    "TS_geoms",
                    self["Reaction"]["reaction"],
                )
                for name in os.listdir(path):
                    if name.startswith("TS"):
                        kind = "TS"
                    elif name.startswith("INT"):
                        kind = "Minimum"
                    name = os.path.join(path, name)
                    if not os.path.isfile(name):
                        continue
                    structure = AaronTools.geometry.Geometry(name)
                    structure.name = os.path.relpath(name, path)
                    structure.name = ".".join(structure.name.split(".")[:-1])
                    structures += [(structure, kind)]
            return structures
        # get starting structure
        if "Geometry" not in self or "structure" not in self["Geometry"]:
            structure = "{}.xyz".format(self["Job"]["name"])
            self["Job"]["name"] = os.path.basename(self["Job"]["name"])
            s = "No structure indicated in config. Trying to use {}".format(
                structure
            )
            print(s, file=sys.stderr)
        else:
            structure = self["Geometry"]["structure"]
        try:
            structure = AaronTools.geometry.Geometry(structure)
        except IndexError:
            structure = AaronTools.geometry.Geometry.from_string(structure)
            self._changes[""] = ({}, None)
        # adjust structure attributes
        if "name" in self["Job"]:
            structure.name = self["Job"]["name"]
        if "Geometry" in self and "comment" in self["Geometry"]:
            structure.comment = self["Geometry"]["comment"]
            structure.parse_comment()
        # apply functions found in [Geometry] section
        # structure.write()
        # os.system("chimera {}.xyz".format(structure.name))
        if "Geometry" in self:
            if "&call" in self["Geometry"]:
                lines = self["Geometry"]["&call"]
                for line in lines.split("\n"):
                    if line.strip():
                        eval(
                            line.strip(),
                            {
                                "structure": structure,
                                "Geometry": AaronTools.geometry.Geometry,
                            },
                        )
        return structure

    def _parse_includes(self):
        """
        Moves option values from subsections into parent section
        Eg:
            [Job]
            include = Wheeler
            ppn = 12
            queue = batch

            [Job.Wheeler]
            nodes = 1
            queue = wheeler_q

            ^^^evaluates to:
            [Job]
            nodes = 1
            ppn = 12
            queue = wheeler_q
        """
        for section in self._sections:
            # add requested subsections to parent section
            if self.has_option(section, "include"):
                include_section = "{}.{}".format(
                    section, self[section]["include"]
                )
                for key, val in self[include_section].items():
                    self[section][key] = val
            # handle non-default capitalization of default section
            if section.lower() == "default":
                for key, val in self[section].items():
                    self["DEFAULT"][key] = val

    def get_spec(self, spec=None, skip_keys=None):
        if spec is None:
            spec = {}
        if skip_keys is None:
            skip_keys = []
        for attr in ["_changes", "_changed_list", "_args", "_kwargs"]:
            if attr in skip_keys:
                continue
            spec[attr] = self.__dict__[attr]
        for section in self._sections:
            for key, val in self.items(section=section):
                if key in skip_keys:
                    continue
                spec["{}/{}".format(section, key).replace(".", "#")] = val
        return spec

    def for_step(self, step):
        config = self.copy()
        # find step-specific options
        for section in config._sections:
            for key, val in config[section].items():
                remove_key = key
                key = key.strip().split()
                if len(key) == 1:
                    continue
                key_step = key[0]
                key = " ".join(key[1:])
                try:
                    key_step = float(key_step)
                except ValueError:
                    key_step = key_step.strip()
                # screen based on step
                if key_step == "low" and step < 2 and step != 0:
                    config[section][key] = val
                if key_step == "high" and step >= 5:
                    config[section][key] = val
                if isinstance(key_step, float) and key_step == float(step):
                    config[section][key] = val
                # clean up metadata
                del config[section][remove_key]
        # other job-specific additions
        if "host" in config["HPC"]:
            try:
                config["HPC"]["work_dir"] = config["HPC"].get("remote_dir")
            except TypeError as e:
                raise RuntimeError(
                    "Must specify remote working directory for HPC (remote_dir = /path/to/HPC/work/dir)"
                ) from e
        else:
            config["HPC"]["work_dir"] = config["Job"].get("top_dir")
        config["HPC"]["job_name"] = config["Job"]["name"]
        if step:
            config["HPC"]["job_name"] = "{}.{}".format(
                config["HPC"]["job_name"], step
            )
        # parse user-supplied functions in config file
        config.parse_functions()
        return config
