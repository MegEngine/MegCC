#! /usr/bin/env python3
import os

import numpy as np
import yaml
from src.benchmark import BenchMarkRunnerBase, ValidModel, ValidOutputDir
from src.models import *
from common import *

all_models = AllModel()
# available arch_str = ["x86", "arm64", "armv7", "riscv"]
arch_list = ["x86"]
framework_str = ["megcc"]
models_dir = "{}/benchmark/model/generated_models".format(megcc_path)
benchmarker_list = {}
# set as your own ssh device host and workdir(make sure install sshd and rsync on your device)
ssh_device_info = [{"name": "", "host": "", "workdir": ""}]


class BenchmarkRunner(BenchMarkRunnerBase):
    remote_config = None
    remote_config_file = "{}/benchmark/config/cofnig.yaml".format(megcc_path)

    def __init__(self, benchmark_build_dir="", benchmark_arch="x86"):
        super().__init__(benchmark_build_dir, benchmark_arch)

    def run_ssh_device(self, ssh_name, ssh_host, ssh_workdir):
        if not os.path.exists(self.output_dir.local_path) or os.path.isfile(
                self.output_dir.local_path):
            os.makedirs(self.output_dir.local_path)
        logfile = open(
            "{}/{}-{}-{}-{}-log-{}.txt".format(
                self.output_dir.local_path,
                self.benchmark_framework,
                self.benchmark_arch,
                self.model.name,
                self.log_level,
                ssh_name,
            ),
            "w",
        )
        run_options = ""
        if self.log_level == 0:
            run_options += " --profile"
        if self.benchmark_framework == "mge":
            run_options += " --mge"
        config_name = "benchmark-{}-{}-{}".format(self.benchmark_framework,
                                                  self.benchmark_arch,
                                                  self.model.name)
        for file_ in [self.benchmark_exec_func, self.model.path]:
            cmd = "rsync -aP -zz {} {}:{}/".format(file_, ssh_host, ssh_workdir)
            subprocess.check_call(cmd, shell=True)
        cmd = ' ssh -t {} "unset LD_PRELOAD && cd {} && LD_LIBRARY_PATH=./ && chmod +x ./benchmarker && ./benchmarker {}.{} {}" '.format(
            ssh_host, ssh_workdir, self.model.name, self.model.exten,
            run_options)
        subprocess.check_call(cmd,
                              shell=True,
                              stdout=logfile,
                              stderr=subprocess.STDOUT)


# build benchmarker
def gen_benchmarker(arch_str):
    for arch_desc in arch_str:
        benchmark_build_dir = "{}/benchmark/build/{}".format(megcc_path, arch_desc)
        benchmarker = BenchmarkRunner(
            benchmark_build_dir=benchmark_build_dir, benchmark_arch=arch_desc
        )
        benchmarker_list[arch_desc] = benchmarker


# set test config and run
def set_config_and_run(arch_str):
    for arch_desc in arch_str:
        kernel_build_dir = kernel_build_dirs[arch_desc]
        for model in all_models.models:
            for framework in framework_str:
                for log_level in [False, True]:
                    if framework == "megcc":
                        exten = "tiny"
                        model_path = "{}/{}.tiny".format(kernel_build_dir, model.name)
                    model_ = ValidModel(model_path, model.name, exten)
                    output_dir_ = ValidOutputDir(
                        "{}/benchmark/output".format(megcc_path), "output"
                    )
                    benchmarker_list[arch_desc].set_config(
                        profile_kernel=log_level,
                        benchmark_framework=framework,
                        model=model_,
                        output_dir=output_dir_,
                    )
                    if arch_desc == "x86":
                        benchmarker_list[arch_desc].run_local()
                    elif (
                        arch_desc == "riscv"
                        or arch_desc == "armv7"
                        or arch_desc == "arm64"
                    ):
                        # run for different device may avoid the effection of device heat radiation
                        for ssh_device in ssh_device_info:
                            ssh_name = ssh_device["name"]
                            ssh_host = ssh_device["host"]
                            ssh_workdir = ssh_device["workdir"]
                            benchmarker_list[arch_desc].run_ssh_device(
                                ssh_name, ssh_host, ssh_workdir
                            )
                    else:
                        print("unsupported arch type: {} in megcc".format(arch_desc))
                        return


def main():
    build_model_and_megcc_lib(all_models, models_dir, arch_list)
    gen_benchmarker(arch_list)
    build_benchmarker(x86_target="fallback", arch_str=arch_list, benchmarkers=benchmarker_list)
    set_config_and_run(arch_list)

if __name__ == "__main__":
    main()
