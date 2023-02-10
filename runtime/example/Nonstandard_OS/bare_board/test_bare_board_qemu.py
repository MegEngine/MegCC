#!/usr/bin/env python3

# file runtime/example/Nonstandard_OS/bare_board/test_bare_board_qemu.py
# This file is part of MegCC, a deep learning compiler developed by Megvii.
# copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

import argparse
import logging
import os
import subprocess
from pathlib import Path


def main():
    test_aarchs = ["aarch32", "aarch64"]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--tinynn_lib_install_dir",
        help="tinynn static lib build with runtime and input install dir",
        required=True,
    )
    parser.add_argument(
        "--test_arch",
        help="test arch, now support: {}".format(test_aarchs),
        required=True,
    )
    args = parser.parse_args()

    assert os.path.isdir(
        args.tinynn_lib_install_dir
    ), "invalid --tinynn_lib_install_dir(can not find or not dir): {}".format(
        args.tinynn_lib_install_dir)
    lib_path = os.path.join(args.tinynn_lib_install_dir, "lib/libTinyNN.a")
    assert os.path.isfile(lib_path), "can not find tinynn lib at: {}".format(
        lib_path)
    assert (args.test_arch in test_aarchs
            ), "invalid --test_arch, only support: {}".format(test_aarchs)

    gnu_toolchain_link = "https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads"
    env_help_msg = {
        "aarch32": {
            "arm-none-eabi-gcc": gnu_toolchain_link,
            "qemu-system-arm": "sudo apt install qemu-system-arm",
        },
        "aarch64": {
            "aarch64-none-elf-gcc": gnu_toolchain_link,
            "qemu-system-aarch64": "sudo apt install qemu-system-arm",
        },
    }

    for key in env_help_msg[args.test_arch]:
        try:
            subprocess.check_call("which {}".format(key), shell=True)
            logging.debug("check key success!")
        except Exception as exp:
            logging.error(
                "can not find: {} in PATH, please install from: {}".format(
                    (key), env_help_msg[args.test_arch][key]))
            raise exp

    common_build_flags = " -O3 -specs=nosys.specs bare_board_qemu_arm_example.c {}_qemu/startup.s -L{}/lib -lTinyNN -I{}/include -Wl,-T,{}_qemu/link.ld -lm".format(
        args.test_arch,
        args.tinynn_lib_install_dir,
        args.tinynn_lib_install_dir,
        args.test_arch,
    )
    build_cmd = {
        "aarch32":
        "arm-none-eabi-gcc {} -o qemu_{}.bin".format(common_build_flags,
                                                     args.test_arch),
        "aarch64":
        "aarch64-none-elf-gcc {} -o qemu_{}.bin".format(
            common_build_flags, args.test_arch),
    }
    cmd = build_cmd[args.test_arch]
    logging.debug("build cmd: {}".format(cmd))
    subprocess.check_call(cmd, shell=True)

    # FIXME: QEMU return !0, check the finally result with string now
    qemu_cmd = {
        "aarch32":
        "rm -rf raw_log.log && qemu-system-arm -M vexpress-a9 -m 1024M -nographic -kernel qemu_aarch32.bin -semihosting | tee raw_log.log || true",
        "aarch64":
        "rm -rf raw_log.log && qemu-system-aarch64 -machine virt,virtualization=on -m 1024M -cpu cortex-a53 -nographic -s -kernel qemu_aarch64.bin -semihosting | tee raw_log.log || true",
    }
    max_time = 100
    logging.debug('run qemu with cmd: "{}" with timeout: {}s'.format(
        qemu_cmd[args.test_arch], max_time))
    try:
        subprocess.check_call(qemu_cmd[args.test_arch],
                              shell=True,
                              timeout=max_time)
        raw_log = subprocess.check_output("cat raw_log.log",
                                          shell=True).decode("utf-8")
        assert raw_log.find("Bye world!") > 0
        logging.debug("check key log success")
    except Exception as exp:
        logging.error("run failed")
        raise exp
    finally:
        kill_args = {
            "aarch32": "qemu-system-arm",
            "aarch64": "qemu-system-aar"
        }
        os.system("pkill -9 {}".format(kill_args[args.test_arch]))
        os.system("reset")


if __name__ == "__main__":
    # run at repo root dir
    os.chdir(str(Path(__file__).resolve().parent))
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)
    logging.debug("run at: {}".format(os.getcwd()))

    main()
