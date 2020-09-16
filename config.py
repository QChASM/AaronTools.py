import configparser
import os
import re
from copy import deepcopy

from AaronTools.const import AARONLIB, QCHASM
from AaronTools.theory import Theory
from AaronTools.theory.implicit_solvent import ImplicitSolvent
from AaronTools.theory.job_types import (
    FrequencyJob,
    OptimizationJob,
    SinglePointJob,
)

SECTIONS = ["DEFAULT", "HPC", "Job", "Substitution", "Mapping", "Reaction"]


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
        configparser.ConfigParser.__init__(self, interpolation=None, **kwargs)
        if not quiet:
            print("Reading configuration...")
        if infile is not None:
            self.read_config(infile, quiet)
            self._parse_includes()
            # set additional default values
            if "top_dir" not in self["DEFAULT"]:
                self["DEFAULT"]["top_dir"] = os.path.dirname(
                    os.path.abspath(infile)
                )

    def copy(self):
        config = Config(infile=None, quiet=True)
        for attr in self.__dict__:
            if attr in SECTIONS:
                continue
            config.__dict__[attr] = deepcopy(self.__dict__[attr])
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
            for key, val in self.items(section):
                for match in func_patt.findall(val):
                    eval_match = match
                    for attr in attr_patt.findall(match):
                        if ":" in attr:
                            option, from_section = attr.split(":")
                        else:
                            option, from_section = attr, section
                        option = self.get(from_section, option)
                        try:
                            option = re.search("\d+\.?\d*", option).group(0)
                        except AttributeError:
                            pass
                        eval_match = eval_match.replace("$" + attr, option, 1)
                    try:
                        eval_match = eval(eval_match)
                    except TypeError as e:
                        raise TypeError(
                            "{} for\n\t[{}]\n\t{} = {}\nin config file. Could not evaluate {}".format(
                                e.args[0], section, key, val, eval_match
                            )
                        )
                    val = val.replace("%{" + match + "}", str(eval_match))
                    self[section][key] = val

    def read_config(self, infile, quiet):
        """
        Reads configuration information from `infile` after pulling defaults
        """
        for filename in [
            os.path.join(QCHASM, "AaronTools/config.ini"),
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
                self.read(filename)
            except configparser.MissingSectionHeaderError:
                # add global options to default section
                with open(filename) as f:
                    contents = "[DEFAULT]\n" + f.read()
                self.read_string(contents)

    def get_theory(self, geometry):
        """
        Get the theory object according to configuration information
        """
        theory = Theory(geometry=geometry, **dict(self["Job"].items()))
        # build ImplicitSolvent object
        if self["Job"].get("solvent") == "gas":
            theory.solvent = None
        elif self["Job"].get("solvent"):
            theory.solvent = ImplicitSolvent(
                self["Job"].get("solvent"), self["Job"].get("solvent_model")
            )
        # build JobType list
        job_type = self["Job"].get("type")
        if job_type:
            theory.job_type = []
            job_type = job_type.split(".")
            if "opt" in job_type[0]:
                constraints = None
                ts = False
                if "change" in job_type[1]:
                    geometry = geometry.copy()
                    geometry.freeze(self)
                if "constrain" in job_type[1]:
                    constraints = {}
                    for con in list(eval(self["Geometry"]["constraints"])):
                        con = tuple(geometry.find(str(c))[0] for c in con)
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
                if "ts" in job_type[1]:
                    ts = True
                theory.job_type += [
                    OptimizationJob(
                        transition_state=ts,
                        geometry=geometry,
                        constraints=constraints,
                    )
                ]
            if "freq" in job_type[0]:
                if "temperature" in self["Job"]:
                    theory.job_type += [
                        FrequencyJob(
                            temperature=self["Job"].get("temperature")
                        )
                    ]
                else:
                    theory.job_type += [FrequencyJob()]
            if "single-point" in job_type:
                theory.job_type += [SinglePointJob()]
        else:
            theory.job_type = [SinglePointJob()]
        # return updated theory object
        return theory

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
                    section, self.get(section, "include")
                )
                for key, val in self.items(include_section):
                    self.set(section, key, val)
            # handle non-default capitalization of default section
            if section.lower() == "default":
                for key, val in self.items(section):
                    self.set("DEFAULT", key, val)
