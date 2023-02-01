#!/usr/bin/env python3

# file runtime/example/Nonstandard_OS/tee/test_optee.py
# This file is part of MegCC, a deep learning compiler developed by Megvii.
# copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

import argparse
import logging
import os
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--tinynn_lib_install_dir",
        help="tinynn static lib build with runtime and input install dir",
        required=True,
    )
    parser.add_argument(
        "--optee_repo_dir",
        help=
        "optee repo dir, please refs https://optee.readthedocs.io/en/latest/building/gits/build.html to init",
        required=True,
    )
    args = parser.parse_args()

    assert os.path.isdir(
        args.tinynn_lib_install_dir
    ), "invalid --tinynn_lib_install_dir(can not find or not dir): {}".format(
        args.tinynn_lib_install_dir)
    assert os.path.isdir(
        args.optee_repo_dir
    ), "invalid --optee_repo_dir(can not find or not dir): {}".format(
        args.optee_repo_dir)
    optee_repo_sub = ["optee_os", "build", "optee_examples"]
    for s in optee_repo_sub:
        s_dir = os.path.join(args.optee_repo_dir, s)
        assert os.path.isdir(
            s_dir), "invalid --optee_repo_dir : can not find dirs: {}".format(
                s_dir)

    lib_path = os.path.join(args.tinynn_lib_install_dir, "lib/libTinyNN.a")
    assert os.path.isfile(lib_path), "can not find tinynn lib at: {}".format(
        lib_path)

    # clean may old files
    old_builds = [
        "./out-br/host/aarch64-buildroot-linux-gnu/sysroot/usr/bin/optee_example_megcc_inference",
        "out-br/build/optee_examples_ext-1.0",
        "./out-br/target/usr/bin/optee_example_megcc_inference",
        "./linux/arch/arm64/boot/Image",
        "./out/bin/Image",
        "./out-br/images/rootfs.cpio.gz",
        "./out/bin/rootfs.cpio.gz",
    ]
    for o in old_builds:
        f_p = os.path.join(args.optee_repo_dir, o)
        if os.path.isdir(f_p):
            logging.debug("find old build: {}".format(f_p))
            subprocess.check_call("rm -r {}".format(f_p), shell=True)
        if os.path.isfile(f_p):
            logging.debug("find old build: {}".format(f_p))
            subprocess.check_call("rm {}".format(f_p), shell=True)

    # clean old example
    example_dir = os.path.join(args.optee_repo_dir, "optee_examples")
    example_git_dir = os.path.join(example_dir, ".git")
    assert os.path.isdir(
        example_git_dir), "invalid optee repo, can not find: {}".format(
            example_git_dir)
    logging.debug("force clean old example_dir")
    subprocess.check_call(
        "cd {} && git clean -xdf && git reset --hard".format(example_dir),
        shell=True)

    # prepare new test example
    subprocess.check_call("cp -rf tee {}/megcc_inference".format(example_dir),
                          shell=True)
    subprocess.check_call(
        "cp -rf {} {}/megcc_inference/ta/tinynn_sdk_install".format(
            args.tinynn_lib_install_dir, example_dir),
        shell=True,
    )

    # patch optee repo build
    build_dir = os.path.join(args.optee_repo_dir, "build")
    build_git_dir = os.path.join(build_dir, ".git")
    assert os.path.isdir(
        build_git_dir), "invalid optee repo, can not find: {}".format(
            build_git_dir)
    logging.debug("force clean old build_dir")
    subprocess.check_call(
        "cd {} && git clean -xdf && git reset --hard".format(build_dir),
        shell=True)
    # patch details:
    # terminal_start.patch: will show CA and TA log, but save tty log to a raw_log, just use debug see TA log
    # terminal_start_only_show_ca_log.patch: only show CA side log, use to CI check
    patch = "terminal_start_only_show_ca_log.patch"
    subprocess.check_call(
        "cp tee/{} {} && cd {} && git apply {}".format(patch, build_dir,
                                                       build_dir, patch),
        shell=True,
    )

    # build optee repo
    logging.debug("build optee repo")
    subprocess.check_call(
        "cd {} && make -f qemu_v8.mk all -j$(nproc)".format(build_dir),
        shell=True)

    # run example
    max_time = 200
    cmd = "cd {} && rm -rf raw_log.log && make -f qemu_v8.mk run-only | tee raw_log.log".format(
        build_dir)
    logging.debug("run megcc_inference example by cmd: {} timeout: {}".format(
        cmd, max_time))
    try:
        subprocess.check_call(cmd, shell=True, timeout=max_time)
        raw_log = subprocess.check_output(
            "cd {} && cat raw_log.log".format(build_dir),
            shell=True).decode("utf-8")
        ca_key_logs = [
            "invoke init model cmd,seconds", "run megcc tee example success"
        ]
        for log in ca_key_logs:
            logging.debug("check log: {}".format(log))
            assert raw_log.find(log) > 0
    except Exception as exp:
        logging.error(
            "run failed, if you want check TA log, please change patch to terminal_start.patch"
        )
        raise exp
    finally:
        os.system("pkill -9 qemu-system-aar")


if __name__ == "__main__":
    # run at repo root dir
    os.chdir(str(Path(__file__).resolve().parent.parent))
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)
    logging.debug("run at: {}".format(os.getcwd()))

    main()
