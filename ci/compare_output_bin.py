# coding: utf-8
import os
from argparse import ArgumentParser

import numpy as np


def compare_file(file_path_0, file_path_1, eps):
    print("compare ", file_path_0, file_path_1)
    with open(file_path_0, "rb") as f:
        d0 = np.frombuffer(f.read(), dtype=np.float32)
    with open(file_path_1, "rb") as f:
        d1 = np.frombuffer(f.read(), dtype=np.float32)
    assert d0.size == d1.size, "{} == {}".format(d0.size, d1.size)
    diff = np.abs(d0 - d1) / np.maximum(1.0, np.minimum(
        np.abs(d0), np.abs(d1)))
    max_idx = np.argmax(diff.flatten())
    print(d0.shape)
    print(
        "max diff ",
        np.max(diff.flatten()),
        "abs_diff",
        np.max(np.abs(d0 - d1).flatten()),
        d0[max_idx:max_idx + 10],
        " vs ",
        d1[max_idx:max_idx + 10],
        " at ",
        max_idx,
    )
    assert np.all(diff < eps), "failed {} != {}, max ".format(
        d0, d1, np.max(diff.flatten()))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="compare tinynn and model binary outputs")
    parser.add_argument("file_or_dir", nargs=2, type=str)
    parser.add_argument("--eps", type=float, required=False, default=1e-6)
    args = parser.parse_args()
    eps = args.eps
    compare_file_cnt = 0
    if os.path.isdir(args.file_or_dir[0]) and os.path.isdir(
            args.file_or_dir[1]):
        file_names = os.listdir(args.file_or_dir[0])
        for file_name in file_names:
            file_path_0 = os.path.join(args.file_or_dir[0], file_name)
            file_path_1 = os.path.join(args.file_or_dir[1], file_name)
            assert os.path.exists(file_path_0), "can not find {}".format(
                file_path_0)
            assert os.path.exists(file_path_1), "can not find {}".format(
                file_path_1)
            compare_file(file_path_0, file_path_1, eps)
            compare_file_cnt += 1
    else:
        assert os.path.isfile(args.file_or_dir[0]), "can not find {}".format(
            args.file_or_dir[0])
        assert os.path.isfile(args.file_or_dir[1]), "can not find {}".format(
            args.file_or_dir[1])
        compare_file(args.file_or_dir[0], args.file_or_dir[1], eps)
        compare_file_cnt += 1
    if compare_file_cnt > 0:
        print("compare pass!!")
    else:
        print("no file compared!!")
        exit(-1)
