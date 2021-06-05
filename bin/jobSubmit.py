#!/usr/bin/env python3

import argparse
from warnings import warn

from AaronTools.config import Config
from AaronTools.job_control import SubmitProcess
from AaronTools.utils.utils import glob_files

config = Config(quiet=True)

default_proc = config.getint(
    "Job", "processors", fallback=config.getint("Job", "procs", fallback=4)
)
default_mem = config.getint("Job", "memory", fallback=8)
default_walltime = config.getint("Job", "walltime", fallback=12)
default_template = config.get("Job", "template", fallback=None)


submit_parser = argparse.ArgumentParser(
    description="submit a QM computation to the queue",
    formatter_class=argparse.RawTextHelpFormatter,
)

submit_parser.add_argument(
    "infile",
    metavar="input file",
    type=str,
    nargs="+",
    help="a Psi4, ORCA, or Gaussian input file",
)

submit_parser.add_argument(
    "-d",
    "--config-default",
    type=str,
    default=None,
    dest="section",
    help="use memory, processors, walltime, and template from\n"
    "the specified seciton of the AaronTools config",
)

submit_parser.add_argument(
    "-j",
    "--job-template",
    type=str,
    default=None,
    dest="template",
    help="template job template file (i.e. for `qsub`, `bsub`, or `sbatch`)",
)

submit_parser.add_argument(
    "-p",
    "--processors",
    type=int,
    required=False,
    default=None,
    dest="processors",
    help="number of processors\n" "Default: %i" % default_proc,
)

submit_parser.add_argument(
    "-m",
    "--memory",
    type=int,
    required=False,
    default=None,
    dest="memory",
    help="memory in GB\n" "Default: %i" % default_mem,
)

submit_parser.add_argument(
    "-t",
    "--walltime",
    type=int,
    required=False,
    default=None,
    dest="time",
    help="walltime in hours\n" "Default: %i" % default_walltime,
)

submit_parser.add_argument(
    "-wl",
    "--wait-last",
    action="store_true",
    default=False,
    dest="wait_last",
    help="wait for the last job to finish before exiting",
)

submit_parser.add_argument(
    "-we",
    "--wait-each",
    action="store_true",
    default=False,
    dest="wait_each",
    help="wait for each job to finish before submitting the next",
)

args = submit_parser.parse_args()

for i, f in enumerate(glob_files(args.infile)):
    # TODO: if processors etc. is not specified, read the input file to see if
    #       processors were specified

    processors = args.processors
    memory = args.memory
    walltime = args.time
    template = args.template

    if args.section is not None:
        if args.processors is None:
            processors = config.getint(
                args.section, "processors", fallback=None
            )
        if args.memory is None:
            memory = config.getint(args.section, "memory", fallback=None)
        if args.time is None:
            walltime = config.getint(args.section, "walltime", fallback=None)
        if args.template is None:
            template = config.get(args.section, "template", fallback=None)

    if processors is None:
        processors = default_proc
    if memory is None:
        memory = default_mem
    if walltime is None:
        walltime = default_walltime
    if template is None:
        template = default_template

    submit_process = SubmitProcess(
        f, walltime, processors, memory, template=template
    )

    try:
        if args.wait_each or (i == len(args.infile) - 1 and args.wait_last):
            submit_process.submit(wait=True, quiet=False)
        else:
            submit_process.submit(wait=False, quiet=False)

    except Exception as e:
        warn("failed to submit %s: %s" % (f, str(e)))
