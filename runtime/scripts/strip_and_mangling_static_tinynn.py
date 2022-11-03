#!/usr/bin/env python3
# file runtime/scripts/strip_and_mangling_static_tinynn.py
# This file is part of MegCC, a deep learning compiler developed by Megvii.
# copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

import argparse
import logging
import os
import random
import string
import subprocess
from pathlib import Path


def create_mangling_string(len):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(len))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--tinynn_lib",
        help="tinynn lib for stip debug section and mangling symbols",
        required=True,
    )
    args = parser.parse_args()
    args.tinynn_lib = os.path.abspath(args.tinynn_lib)

    assert os.path.isfile(args.tinynn_lib), "can not find tinynn lib at: {}".format(
        args.tinynn_lib
    )
    cmd = "file {}".format(args.tinynn_lib)
    file_log = subprocess.check_output(cmd, shell=True).decode("utf-8")
    assert file_log.find("current ar archive") > 0, "%s should be a static lib".format(
        args.tinynn_lib
    )

    # please keep same with runtime/version.ld
    api_symbols = [
        "LITE_",
        "default_config",
        "default_network_io",
        "register_tinynn_cb",
    ]
    key_words = [
        "mgb",
        "mbg",
        "mge",
        "meg",
        "huawei",
        "emui",
        "hicloud",
        "hisilicon",
        "hisi",
        "kirin",
        "balong",
        "harmony",
        "senstime",
    ]
    # load from mem sym
    load_from_mem_sym = ["input_bin", "input_bin_len"]
    # internal sym
    may_internal_sym = ["$", "."]
    # handle sym type
    handle_sym_type = ["T", "t"]

    # handle static symbols
    cmd = "llvm-nm {}".format(args.tinynn_lib)
    nm_raw_log = subprocess.check_output(cmd, shell=True).decode("utf-8").split("\n")
    unique_symbols = []
    alread_push_symbols = []
    mangling_map = {}
    for log in nm_raw_log:
        # logging.debug(log)
        if log.find(" ") > 0:
            s = log.split()
            # log like '0000000000000000 n $d.1'
            if len(s) == 3 and s[2] not in alread_push_symbols:
                alread_push_symbols.append(s[2])
                unique_symbols.append(s)
                # logging.debug("push {}".format(s))
    for log in unique_symbols:
        symbol = log[2]
        symbol_type = log[1]
        need_skip = False
        if symbol[0] in may_internal_sym:
            logging.debug("skip handle internal sym: {}".format(symbol))
            need_skip = True

        if symbol_type not in handle_sym_type:
            logging.debug(
                "skip handle symbol:{} of type: {}".format(symbol, symbol_type)
            )
            need_skip = True

        for api in api_symbols + load_from_mem_sym:
            if symbol.find(api) == 0:
                logging.debug("skip handle api or model mem sym: {}".format(symbol))
                need_skip = True
                break

        # create finally mangling map
        if not need_skip:
            while True:
                mangling_s = create_mangling_string(16)
                valid_mangling = True
                if mangling_s in mangling_map:
                    valid_mangling = False
                for unique_symbol in unique_symbols:
                    if mangling_s == unique_symbol[2]:
                        valid_mangling = False
                        break
                for key_word in key_words:
                    if mangling_s.find(key_word) >= 0:
                        valid_mangling = False
                        break
                if valid_mangling:
                    mangling_map[mangling_s] = symbol
                    logging.debug(
                        "create mangling map for: {} -- {}".format(symbol, mangling_s)
                    )
                    break
    # strip debug section
    cmd = "llvm-objcopy {} --strip-debug".format(args.tinynn_lib)
    subprocess.check_call(cmd, shell=True)
    # call objcopy
    for mangling_s in mangling_map:
        symbol = mangling_map[mangling_s]
        cmd = "llvm-objcopy {} --redefine-sym={}={}".format(
            args.tinynn_lib, symbol, mangling_s
        )
        logging.debug("call cmd: {}".format(cmd))
        subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    main()
