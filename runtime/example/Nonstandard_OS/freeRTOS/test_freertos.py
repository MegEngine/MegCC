#!/usr/bin/env python3

# file runtime/example/Nonstandard_OS/freeRTOS/test_freertos.py
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
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--tinynn_lib_install_dir",
        help="tinynn static lib build with runtime and input install dir",
        required=True,
    )
    parser.add_argument(
        "--free_rtos_repo_dir",
        help="freeRTOS repo dir, please download from https://github.com/FreeRTOS/FreeRTOS",
        required=True,
    )
    args = parser.parse_args()

    assert os.path.isdir(
        args.tinynn_lib_install_dir
    ), "invalid --tinynn_lib_install_dir(can not find or not dir): {}".format(
        args.tinynn_lib_install_dir
    )
    assert os.path.isdir(
        args.free_rtos_repo_dir
    ), "invalid --free_rtos_repo_dir(can not find or not dir): {}".format(
        args.free_rtos_repo_dir
    )
    free_rtos_repo_sub = [
        ".git",
        "FreeRTOS/Demo/CORTEX_MPS2_QEMU_IAR_GCC/build/gcc",
        "FreeRTOS/Demo/Common/Minimal",
    ]
    for s in free_rtos_repo_sub:
        s_dir = os.path.join(args.free_rtos_repo_dir, s)
        assert os.path.isdir(
            s_dir
        ), "invalid --free_rtos_repo_dir : can not find dirs: {}".format(s_dir)

    lib_path = os.path.join(args.tinynn_lib_install_dir, "lib/libTinyNN.a")
    assert os.path.isfile(lib_path), "can not find tinynn lib at: {}".format(lib_path)

    # clean old build and reset freeRTOS repo
    logging.debug("clean old build and reset freeRTOS repo")
    subprocess.check_call(
        "cd {} && git clean -xdf && git reset --hard".format(args.free_rtos_repo_dir),
        shell=True,
    )

    # prepare new test example
    minimal_dir = os.path.join(args.free_rtos_repo_dir, "FreeRTOS/Demo/Common/Minimal")
    subprocess.check_call(
        "cp -rf freeRTOS/tinynn_freeRTOS_example.c {}/".format(minimal_dir), shell=True
    )
    subprocess.check_call(
        "cp -rf {} {}/tinynn_sdk".format(args.tinynn_lib_install_dir, minimal_dir),
        shell=True,
    )

    # patch freeRTOS repo
    logging.debug("patch freeRTOS repo")
    subprocess.check_call(
        "cp freeRTOS/tinynn_example_test_at_minimal_rtos.patch {} && cd {} && git apply tinynn_example_test_at_minimal_rtos.patch".format(
            args.free_rtos_repo_dir, args.free_rtos_repo_dir
        ),
        shell=True,
    )

    # build freeRTOS repo
    logging.debug("build freeRTOS repo")
    iar_dir = os.path.join(
        args.free_rtos_repo_dir, "FreeRTOS/Demo/CORTEX_MPS2_QEMU_IAR_GCC/build/gcc"
    )
    subprocess.check_call("cd {} && make -j$(nproc)".format(iar_dir), shell=True)
    # freeRTOS can not shutdown by semihosting and we do not want to change
    # default error handler, so we start a thread, which sleep max_time
    # then kill qemu-system-arm, let test thread have a chance to exit
    max_time = 15

    def sleep_and_kill_thread():
        logging.debug("into sleep_and_kill_thread function")
        cnt = 0
        while True:
            logging.debug("sleep 1s cnt: {} vs max_time: {}".format(cnt, max_time))
            cnt = cnt + 1
            time.sleep(1)
            if cnt == max_time:
                os.system("pkill -9 qemu-system-arm && reset")
                break

    kill_t = threading.Thread(target=sleep_and_kill_thread)
    kill_t.start()
    # run example
    cmd = "cd {} && rm -rf raw_log.log && qemu-system-arm -semihosting -machine mps2-an385 -cpu cortex-m3 -m 16M -kernel output/RTOSDemo.out -monitor none -nographic -serial stdio | tee raw_log.log || true".format(
        iar_dir
    )
    logging.debug(
        "run tinynn_inference example by cmd: {} timeout: {}".format(cmd, max_time)
    )
    try:
        subprocess.check_call(cmd, shell=True, timeout=max_time + 5)
        raw_log = subprocess.check_output(
            "cd {} && cat raw_log.log".format(iar_dir), shell=True
        ).decode("utf-8")
        raw_log = "head" + raw_log
        key_logs = ["start tinynn inference!!!", "run tinynn FreeRTOS example success"]
        for log in key_logs:
            assert raw_log.find(log) > 0
            logging.debug("check log: {} success".format(log))
    except Exception as exp:
        logging.error("run failed")
        raise exp
    finally:
        os.system("pkill -9 qemu-system-arm && reset")
        kill_t.join()


if __name__ == "__main__":
    # run at repo root dir
    os.chdir(str(Path(__file__).resolve().parent.parent))
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.debug("run at: {}".format(os.getcwd()))

    main()
