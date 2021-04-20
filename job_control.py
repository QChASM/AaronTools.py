import os
import re
import subprocess
from time import sleep

from jinja2 import Environment, FileSystemLoader

from AaronTools import addlogger
from AaronTools.const import AARONLIB

USER = os.getenv("USER")
QUEUE_TYPE = os.getenv("QUEUE_TYPE", "None").upper()


class JobControl:
    pass


@addlogger
class SubmitProcess:
    """class for submitting jobs to the queue
    attributes:

    name:       name of job and input file minus the extension
    exe:        type of input file (com, in, inp)
    directory:  directory the input file is in
    walltime:   allocated walltime in hours
    processors: allocated processors
    memory:     allocated memory in GB
    template:   template job file"""

    LOG = None

    def __init__(self, fname, walltime, processors, memory, template=None):
        """fname:   str     - path to input file (e.g. /home/CoolUser/CoolStuff/neat.com
        walltime:   int/str - walltime in hours
        processors: int/str - allocated processors
        memory:     int/str - allocated memory in GB
        template:   str     - path to template file; if template is None, will look for
                              psi4.job, orca.job, or gaussian.job (depending on
                              extension on fname)"""
        directory, filename = os.path.split(fname)
        self.name, exe = os.path.splitext(filename)
        self.exe = exe[1:]
        self.directory = os.path.abspath(directory)
        self.walltime = walltime
        self.processors = processors
        self.memory = memory
        self.set_template(template)

    @staticmethod
    def unfinished_jobs_in_dir(directory, retry=True):
        """returns list(jobids (str)) of jobids in directory
        retry: bool - if there's an error while checking the queue, sleep 300s and try again"""
        if QUEUE_TYPE == "LSF":
            args = ["bjobs", "-l", "2"]
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = proc.communicate()
            if len(err) != 0 and retry:
                SubmitProcess.LOG.warning(
                    "error checking queue: %s\nsleeping 300s before trying again"
                    % err.decode("utf-8")
                )
                sleep(300)
                return SubmitProcess.unfinished_jobs_in_dir(directory, retry)

            else:
                out = out.decode("utf-8")
                out = out.replace("\s", "")
                out = out.replace("\r", "")
                out = out.replace("\n", "")

                jobs = re.findall("(Job<\d+>.*RUNLIMIT)", out)

                job_ids = []
                for job in jobs:
                    test = re.match("Job<(\d+)>\S+CWD<.+%s>" % directory, job)
                    if test:
                        job_ids.append(test.group(1))

                return job_ids

        elif QUEUE_TYPE == "PBS":
            args = ["qstat", "-fx"]
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = proc.communicate()
            if len(err) != 0 and retry:
                SubmitProcess.LOG.warning(
                    "error checking queue: %s\nsleeping 300s before trying again"
                    % err.decode("utf-8")
                )
                sleep(300)
                return SubmitProcess.unfinished_jobs_in_dir(directory, retry)

            else:
                out = out.decode("utf-8")
                out = out.replace("\n", "")
                out = out.replace("\r", "")

                jobs = re.findall("<Job>(.+?)<\/Job>", out)

                job_ids = []
                for job in jobs:
                    # Q - queued
                    # R - running
                    # S - suspended
                    test = re.match(
                        "<Job_Id>(\d+).+<job_state>[QRS].+PBS_O_WORKDIR=[^,<>]*%s"
                        % directory,
                        job,
                    )
                    if test:
                        job_ids.append(test.group(1))

                return job_ids

        elif QUEUE_TYPE == "SLURM":
            args = ["squeue", "-o", "%i#%Z#%t", "-u", USER]
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = proc.communicate()
            if len(err) != 0 and retry:
                SubmitProcess.LOG.warning(
                    "error checking queue: %s\nsleeping 300s before trying again"
                    % err.decode("utf-8")
                )
                sleep(300)
                return SubmitProcess.unfinished_jobs_in_dir(directory, retry)

            else:
                out = out.decode("utf-8")
                job_ids = []
                for job in out.splitlines():
                    jobid, job_path, job_status = job.split("#")
                    if directory.endswith(job_path) and job_status in [
                        "R",
                        "PD",
                    ]:
                        job_ids.append(jobid)

                return job_ids

        elif QUEUE_TYPE == "SGE":
            # for SGE, we first grab the job ids for the jobs the user is running
            # we then call qstat again to get the directory those jobs are in
            args = ["qstat", "-s", "pr", "-u", USER]
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = proc.communicate()
            if len(err) != 0 and retry:
                SubmitProcess.LOG.warning(
                    "error checking queue: %s\nsleeping 300s before trying again"
                    % err.decode("utf-8")
                )
                sleep(300)
                return SubmitProcess.unfinished_jobs_in_dir(directory, retry)

            else:
                out = out.decode("utf-8")
                jobs = re.findall("^\s*?(\w+)", out)
                # first line is a header
                jobs.pop(0)

                jlist = ",".join(jobs)

                args = ["qstat", "-j", jlist]
                proc = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                out, err = proc.communicate()
                if len(err) != 0 and retry:
                    SubmitProcess.LOG.warning(
                        "error checking queue: %s\nsleeping 300s before trying again"
                        % err.decode("utf-8")
                    )
                    sleep(300)
                    return SubmitProcess.unfinished_jobs_in_dir(
                        directory, retry
                    )

                else:
                    out = out.decode("utf-8")
                    job_ids = []
                    for line in out.splitlines():
                        job_number = re.search("job_number:\s+(\d+)", line)
                        workdir = re.search(
                            "sge_o_workdir:\s+[\S]+%s$" % directory, line
                        )
                        if job_number:
                            job = job_number.group(1)

                        if workdir:
                            job_ids.append(job)

                    return job_ids

    def submit(self, wait=False, quiet=True):
        """submit job to the queue
        wait: bool/int - do not leave the function until any job in the directory
                         finishes (polled every 5 minutes or 'wait' seconds)"""

        job_file = os.path.join(self.directory, self.name + ".job")

        opts = {
            "name": self.name,
            "walltime": self.walltime,
            "processors": self.processors,
            "memory": self.memory,
        }

        tm = self.template.render(**opts)

        with open(job_file, "w") as f:
            f.write(tm)

        stdin = None
        if QUEUE_TYPE == "LSF":
            args = ["bsub"]
            stdin = open(job_file, "r")
        elif QUEUE_TYPE == "SLURM":
            args = ["sbatch", job_file]
        elif QUEUE_TYPE == "PBS":
            args = ["qsub", job_file]
        elif QUEUE_TYPE == "SGE":
            args = ["qsub", job_file]
        else:
            raise NotImplementedError(
                "%s queues not supported, only LSF, SLURM, PBS, and SGE"
                % QUEUE_TYPE
            )

        proc = subprocess.Popen(
            args,
            cwd=self.directory,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.submit_out, self.submit_err = proc.communicate()

        if len(self.submit_err) != 0:
            raise RuntimeError(
                "error with submitting job %s: %s"
                % (self.name, self.submit_err.decode("utf-8"))
            )

        if not quiet:
            print(self.submit_out.decode("utf-8").strip())

        if wait is not False:
            if wait is True:
                wait_time = 30
            else:
                wait_time = abs(wait)

            while len(self.unfinished_jobs_in_dir(self.directory)) != 0:
                # print(self.unfinished_jobs_in_dir(self.directory))
                sleep(wait_time)

        return

    def set_template(self, filename):
        """
        sets job template to filename
        AARONLIB directories are searched
        """
        environment = Environment(loader=FileSystemLoader(AARONLIB))
        if filename is None:
            if self.exe == "com" or self.exe == "gjf":
                filename = "Gaussian_template.txt"
            elif self.exe == "inp":
                filename = "ORCA_template.txt"
            elif self.exe == "in":
                filename = "Psi4_template.txt"

        self.template = environment.get_template(filename)
