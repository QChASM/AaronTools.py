import configparser
import inspect
import json
import logging
import os
import re
import tempfile

from AaronTools.const import AARONLIB, AARONTOOLS

config = configparser.ConfigParser(interpolation=None, comment_prefixes=("#"))
for filename in [
    os.path.join(AARONTOOLS, "config.ini"),
    os.path.join(AARONLIB, "config.ini"),
]:
    try:
        config.read(filename)
    except configparser.MissingSectionHeaderError:
        # add global options to default section
        with open(filename) as f:
            contents = "[DEFAULT]\n" + f.read()
        config.read_string(contents)

if "log_level" in config["DEFAULT"]:
    LOGLEVEL = config["DEFAULT"]["log_level"].upper()
else:
    LOGLEVEL = "WARNING"
if "print_citations" in config["DEFAULT"]:
    PRINT_CITATIONS = config["DEFAULT"].getboolean("print_citations")
else:
    PRINT_CITATIONS = False
try:
    SAVE_CITATIONS = config["DEFAULT"].getboolean("save_citations")
except ValueError:
    SAVE_CITATIONS = config["DEFAULT"].get("save_citations")
if SAVE_CITATIONS is False:
    SAVE_CITATIONS = None

logging.logThreads = 0
logging.logProcesses = 0

logging.captureWarnings(True)


class CustomFilter(logging.Filter):
    def __init__(self, name="", level=None, override=None, cite=False):
        super().__init__(name=name)

        self.level = logging.WARNING
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        if level is not None:
            self.level = level
        self.override = {}
        if override is not None:
            self.override = override
        self.cite = cite

    def filter(self, record):
        if record.funcName == "citation":
            found = False
            for frame in reversed(inspect.stack()):
                if found:
                    record.funcName = frame.function
                    break
                if frame.function == "_callTestMethod":
                    found = True
            else:
                record.funcName = inspect.stack()[-2].function
            record.levelname = "CITATION"
            if not self.cite:
                return False
            self.parse_message(record)
            return True
        for level, func_list in self.override.items():
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            if record.funcName not in func_list:
                continue
            if record.levelno < level:
                return False
            self.parse_message(record)
            return True
        if record.levelno < self.level:
            return False
        self.parse_message(record)
        return True

    def parse_message(self, record):
        """
        Formats message to print prettily to console
        """
        if isinstance(record.msg, str):
            record.msg = re.sub(
                "\n(\S)", lambda x: "\n  %s" % x.group(1), record.msg
            )
        msg = ["\n  "]
        for word in re.findall("\S+\s*", record.getMessage()):
            if len("".join(msg).split("\n")[-1]) + len(word) < 80:
                msg.append(word)
            else:
                msg.append("\n  {}".format(word))
        record.getMessage = lambda: "".join(msg)


class CitationHandler(logging.FileHandler):
    def __init__(self, filename, **kwargs):
        filename = os.path.expandvars(filename)
        if not os.path.exists(os.path.dirname(filename)):
            # might be trying to put citations in $AARONLIB, but user
            # didn't bother to set the environment variable and just
            # uses the default
            from AaronTools.const import AARONLIB
            if "$AARONLIB" in filename:
                filename = filename.replace("$AARONLIB", AARONLIB)
            elif "${AARONLIB}" in filename:
                filename = filename.replace("${AARONLIB}", AARONLIB)
            elif "%AARONLIB%" in filename:
                filename = filename.replace("%AARONLIB%", AARONLIB)
        super().__init__(filename, **kwargs)

    def emit(self, record):
        """
        Adds a record to the citation file if it's not already present
        """
        if record.levelname != "CITATION":
            return
        msg = record.msg.replace("\n  ", " ")
        record.getMessage = lambda: "".join(msg)
        # check for duplicates
        dupe = False
        with open(self.baseFilename) as f:
            for line in f.readlines():
                if line.strip() == self.format(record):
                    dupe = True
                    break
        if not dupe:
            super().emit(record)


class ATLogger(logging.Logger):
    def __init__(
        self, name, level=None, override=None, fmt=None, add_hdlrs=None
    ):
        """
        :level: the log level to use
        :override: dict(level=funcName) to override loglevel for certain funcitons
        :fmt: formatting string (optional)
        :add_hdlrs: list(str(handlerName)) or list(Handler())
        """
        super().__init__(name, level=1)
        if level is None:
            level = LOGLEVEL
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.level = level
        if fmt is None:
            fmt = "%(levelname)s %(name)s.%(funcName)s %(message)s"

        formatter = logging.Formatter(fmt=fmt)
        handlers = [(logging.StreamHandler(), PRINT_CITATIONS)]
        if SAVE_CITATIONS is not None and os.access(SAVE_CITATIONS, os.W_OK):
            handlers += [(CitationHandler(SAVE_CITATIONS), True)]
        if add_hdlrs is not None:
            for hdlr in add_hdlrs:
                if isinstance(hdlr, str):
                    hdlr = getattr(logging, hdlr)
                    handlers.append((hdlr(), PRINT_CITATIONS))
                else:
                    handlers.append(hdlr, PRINT_CITATIONS)
        for hdlr, cite in handlers:
            hdlr.setFormatter(formatter)
            hdlr.addFilter(
                CustomFilter(
                    name=name, level=self.level, override=override, cite=cite
                )
            )
            self.addHandler(hdlr)

    def citation(self, msg, *args, **kwargs):
        self.info(msg, *args, **kwargs)


def getlogger(name=None, level=None, override=None, fmt=None):
    """
    Get the logger without using the class decorator
    :level: the log level to apply, defaults to WARNING
    :override: a dictionary of the form {new_level: function_name_list} will apply the
        `new_level` to log records produced from functions with names in
        `function_name_list`, eg:
            override={"DEBUG": ["some_function"]}
        will set the log level to DEBUG for any messages produced during the run of
        some_function()
    """
    if name is None:
        package = None
        for frame in reversed(inspect.stack()):
            res = inspect.getargvalues(frame.frame)
            if "__name__" in res.locals and name is None:
                name = res.locals["__name__"]
            if "__package__" in res.locals and package is None:
                package = res.locals["__package__"]
            if name is not None and package is not None:
                break
        name = "{}{}{}".format(
            name if name is not None else "",
            "." if package is not None else "",
            package if package is not None else "",
        )
    log = ATLogger(name, level=level, override=override, fmt=fmt)
    return log


def addlogger(cls):
    """
    Import this function and use it as a class decorator.
    Log messages using the created LOG class attribute.

    Useful class attributes to set that will be picked up by this decorator:
    :LOG: Will be set to the logger instance during class initialization
    :LOGLEVEL: Set this to use a different log level than what is in your config. Only
        do this for testing purposes, and do not include it when pushing commits to the
        master AaronTools branch.
    :LOGLEVEL_OVERRIDE: Use this dict to override the log level set in the config file
        for records originating in particular functions. Keys are log levels, values
        are lists of strings corresponding to function names (default: {})

    Example:
    ```
    from AaronTools import addlogger

    @addlogger
    class Someclass:
        LOG = None
        LOGLEVEL = "WARNING"
        LOGLEVEL_OVERRIDE = {"DEBUG": ["some_function"]}

        # this won't be printed b/c "INFO" < LOGLEVEL
        LOG.info("loading class")

        def some_function(self):
            # this message will be printed thanks to LOGLEVEL_OVERRIDE
            self.LOG.debug("function called")
    ```
    """
    name = "{}.{}".format(cls.__module__, cls.__name__)
    level = None
    if hasattr(cls, "LOGLEVEL") and cls.LOGLEVEL is not None:
        level = cls.LOGLEVEL
    override = None
    if hasattr(cls, "LOGLEVEL_OVERRIDE") and cls.LOGLEVEL_OVERRIDE is not None:
        override = cls.LOGLEVEL_OVERRIDE

    cls.LOG = ATLogger(name, level=level, override=override)
    return cls
