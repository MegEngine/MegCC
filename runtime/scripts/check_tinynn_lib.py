#!/usr/bin/env python3
# file runtime/scripts/check_tinynn_lib.py
# This file is part of MegCC, a deep learning compiler developed by Megvii.
# copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

import argparse
import logging
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tinynn_lib", help="tinynn lib, check it can deploy at NONE STANDARD OS or not", required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.tinynn_lib), "can not find tinynn lib at: {}".format(
        args.tinynn_lib
    )
    # libc_apis_without_syscall means libc api do not depends on syscall, bare board toolchains
    # always imp this function, so we mark this kind of apis to white list
    libc_apis_without_syscall = [
        "memset",
        "snprintf",
        "strcpy",
        "memcpy",
        "strlen",
        "strcmp",
        "sprintf",
        "memmove",
        "qsort",
    ]
    # libm_apis_without_syscall means libm api do not depends syscall, some toolchains(tee), may
    # not imp it now, link newlib libm.a now, TODO: imp self math function
    libm_apis_without_syscall = ["exp", "floor", "fmax", "expf"]
    cmd = "nm {}".format(args.tinynn_lib)
    nm_raw_log = subprocess.check_output(cmd, shell=True).decode("utf-8").split("\n")
    find_symbols = []
    may_undefined_symbols = []
    undefined_symbols = []
    log_trait = {
        " T ": find_symbols,
        " t ": find_symbols,
        " D ": find_symbols,
        " d ": find_symbols,
        " B ": find_symbols,
        " b ": find_symbols,
        " C ": find_symbols,
        " U ": may_undefined_symbols,
    }
    for log in nm_raw_log:
        # logging.debug(log)
        for key in log_trait:
            if log.find(key) > 0:
                l = log[log.find(key) + 3 :]
                # logging.debug("put {} to {}".format(l, key))
                log_trait[key].append(l)

    for may_undefined_symbol in set(may_undefined_symbols):
        if may_undefined_symbol in libc_apis_without_syscall:
            logging.debug(
                "undefined symbol: {} in libc white list".format(may_undefined_symbol)
            )
        elif may_undefined_symbol in libm_apis_without_syscall:
            logging.debug(
                "undefined symbol: {} in libm white list".format(may_undefined_symbol)
            )
        elif may_undefined_symbol not in find_symbols:
            undefined_symbols.append(may_undefined_symbol)

    assert len(undefined_symbols) == 0, "find undefined_symbols({}): {}".format(
        len(set(undefined_symbols)), set(undefined_symbols)
    )
    assert len(find_symbols) > 0, "can not find any symbols, may invalid static lib"
    logging.debug("check {} file success!!".format(args.tinynn_lib))


if __name__ == "__main__":
    # run at repo root dir
    os.chdir(str(Path(__file__).resolve().parent.parent))
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    main()
