#!/usr/bin/env python3
# -*-coding=utf-8-*-

import argparse
import struct
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the correctness of header of decrypted hako model.")
    parser.add_argument(
        "--header-file",
        type=str,
        required=True,
        help="the header file",
    )
    parser.add_argument(
        "--ans",
        type=str,
        required=True,
        help="the correct answer file",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="the model file",
    )
    args = parser.parse_args()

    ans_path = args.ans
    ans = []
    with open(ans_path) as ans_file:
        ans.append(ans_file.readline().strip('\n'))
        ans.append(ans_file.readline().strip('\n'))
    len0 = len(ans[0])
    len1 = len(ans[1])

    model_path = args.model_file
    real_model_len = os.stat(model_path).st_size

    header_path = args.header_file
    data = bytearray()
    with open(header_path, "rb") as header_file:
        data.extend(header_file.read())
        header = struct.unpack("B{}sB{}s".format(len0, len1), data[0:len0+len1+2])
        model_len = struct.unpack("i", data[len0+len1+2:])[0]
        assert header[0] == len0
        assert header[1].decode("ascii") == ans[0]
        assert header[2] == len1
        assert header[3].decode("ascii") == ans[1]
        assert model_len == real_model_len