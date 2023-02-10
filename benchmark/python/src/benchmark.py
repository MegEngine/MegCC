#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

import numpy as np

megcc_path = Path(os.path.split(
    os.path.realpath(__file__))[0]).parent.parent.parent.absolute()


class ValidModel:
    path = ""
    name = ""
    extension = ""

    def __init__(self, model_path="", model_name="", extension=""):
        self.path = model_path
        self.name = model_name
        self.extension = extension


class ValidOutputDir:
    local_path = ""
    remote_path = ""
    tag = ""

    def __init__(self, local_path="", remote_path=""):
        self.local_path = local_path
        self.remote_path = remote_path


class BenchMarkRunnerBase:
    model = None
    benchmark_build_dir = ""
    benchmark_arch = ""
    benchmark_framework = ""
    output_dir = None
    log_level = -1
    benchmark_exec_func = ""

    def __init__(self, benchmark_build_dir="", benchmark_arch="x86"):
        if benchmark_build_dir == "":
            benchmark_build_dir = "{}/benchmark/build_{}".format(
                megcc_path, benchmark_arch)
        self.benchmark_build_dir = benchmark_build_dir
        self.benchmark_arch = benchmark_arch

    def build(self, x86_target="fallback", build_options=""):
        # build prepare
        if not os.path.exists(self.benchmark_build_dir) or os.path.isfile(
                self.benchmark_build_dir):
            os.makedirs(self.benchmark_build_dir)
        # build megengine lib and set cmake build options
        cmd = "cd {} && cmake {}/benchmark {} -G Ninja && ninja install/strip".format(
            self.benchmark_build_dir, megcc_path, build_options)
        subprocess.check_call(cmd, shell=True)

    def set_config(
        self,
        profile_kernel=False,
        benchmark_framework="megcc",
        model=None,
        output_dir=None,
    ):
        if profile_kernel:
            self.log_level = 0
        else:
            self.log_level = 3
        self.benchmark_framework = benchmark_framework
        self.output_dir = output_dir
        self.model = model
        self.benchmark_exec_func = "{}/install/bin/benchmarker".format(
            self.benchmark_build_dir)

    def run_local(self):
        if not os.path.exists(self.output_dir.local_path) or os.path.isfile(
                self.output_dir.local_path):
            os.makedirs(self.output_dir.local_path)
        logfile = open(
            "{}/{}-{}-{}-{}-log-local.txt".format(
                self.output_dir.local_path,
                self.benchmark_framework,
                self.benchmark_arch,
                self.model.name,
                self.log_level,
            ),
            "w",
        )
        run_options = ""
        if self.log_level == 0:
            run_options += " --profile"
        if self.benchmark_framework == "mge":
            run_options += " --mge"
        cmd = "{} {} {}".format(self.benchmark_exec_func, self.model.path,
                                run_options)
        subprocess.check_call(cmd,
                              shell=True,
                              stdout=logfile,
                              stderr=subprocess.STDOUT)
