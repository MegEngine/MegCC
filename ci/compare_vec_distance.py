# coding: utf-8
import os
from argparse import ArgumentParser

import numpy as np


def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def water(m_seed_in, str, n):
    str = str.tobytes()
    str = np.frombuffer(str, "uint8").copy()
    m_seed = np.array([m_seed_in]).astype("uint32")

    def get_seed():
        m_seed[0] ^= m_seed[0] << 13
        m_seed[0] ^= m_seed[0] >> 17
        m_seed[0] ^= m_seed[0] << 5
        return m_seed[0]

    s = np.zeros(0x100).astype("uint8")
    m_seed[0] ^= n
    for i in range(0xFF):
        s[i] = i
    j = 0
    for i in range(0xFF):
        j = (j + s[i] + (get_seed() >> ((i & 3) << 3))) & 0xFF
        s[i], s[j] = s[j], s[i]

    i = 0
    j = 0
    k = 0
    nr_drop = (get_seed() & 4095) + 800
    while k < nr_drop:
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        k += 1

    for k in range(n):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        str[k] ^= s[(np.uint16(s[i]) + np.uint16(s[j])) & 0xFF]
    return np.frombuffer(str, "float32")


def piecewise_normalize(score, left, right, threshold, new_left, new_right,
                        new_threshold):
    if not (score >= left and score <= right):
        return new_left

    ret = 0
    if score < threshold:
        ret = (score - left) / (threshold - left) * (new_threshold -
                                                     new_left) + new_left
        if ret < new_left:
            ret = new_left
        if ret >= new_threshold:
            ret = new_threshold - 1
    else:
        ret = (score - threshold) / (right - threshold) * (
            new_right - new_threshold) + new_threshold
        if ret <= new_threshold:
            ret = new_threshold + 1
        if ret > new_right:
            ret = new_right
    return ret


def get_spy_similar(v1, v2):
    num = float(np.sum((v1 - v2) * (v1 - v2)))
    alpha = 2.297190
    beta = -2.659770
    threshold = 55.3
    score_full = 100.0 / (1.0 + np.exp(alpha * num + beta))
    final_score = piecewise_normalize(score_full, 0.0, 100.0, threshold, 0,
                                      100, 80)
    print("spy ", num, score_full, score_full, final_score)
    return score_full


def compare_file(file_path_0, file_path_1, score_limit):
    print("compare ", os.path.abspath(file_path_0),
          os.path.abspath(file_path_1))
    with open(file_path_0, "rb") as f:
        d0 = np.frombuffer(f.read(), dtype=np.float32)
    with open(file_path_1, "rb") as f:
        d1 = np.frombuffer(f.read(), dtype=np.float32)
    assert d0.size == d1.size, "{} == {}".format(d0.size, d1.size)
    score = get_spy_similar(d0, d1)
    assert score >= score_limit


if __name__ == "__main__":
    parser = ArgumentParser(
        description="compare tinynn and model binary outputs")
    parser.add_argument("file_or_dir", nargs=2, type=str)
    parser.add_argument("--score", type=float, required=False, default=50)
    args = parser.parse_args()
    score = args.score
    compare_file_cnt = 0
    if os.path.isdir(args.file_or_dir[0]) and os.path.isdir(
            args.file_or_dir[1]):
        file_names = os.listdir(args.file_or_dir[0])
        nr_files = len(file_names)
        for file_name in file_names:
            file_path_0 = os.path.join(args.file_or_dir[0], file_name)
            file_path_1 = os.path.join(args.file_or_dir[1], file_name)
            assert os.path.exists(file_path_0), "can not find {}".format(
                file_path_0)
            assert os.path.exists(file_path_1), "can not find {}".format(
                file_path_1)
            compare_file(file_path_0, file_path_1, score)
            compare_file_cnt += 1
        for i in range(nr_files):
            for j in range(i + 1, nr_files):
                file_path_0 = os.path.join(args.file_or_dir[0], file_names[i])
                file_path_1 = os.path.join(args.file_or_dir[0], file_names[j])
                compare_file(file_path_0, file_path_1, score)
                compare_file_cnt += 1
        for i in range(nr_files):
            for j in range(i + 1, nr_files):
                file_path_0 = os.path.join(args.file_or_dir[1], file_names[i])
                file_path_1 = os.path.join(args.file_or_dir[1], file_names[j])
                compare_file(file_path_0, file_path_1, score)
                compare_file_cnt += 1

    else:
        assert os.path.isfile(args.file_or_dir[0]), "can not find {}".format(
            args.file_or_dir[0])
        assert os.path.isfile(args.file_or_dir[1]), "can not find {}".format(
            args.file_or_dir[1])
        compare_file(args.file_or_dir[0], args.file_or_dir[1], score)
        compare_file_cnt += 1
    if compare_file_cnt > 0:
        print("compare pass!!")
    else:
        print("no file compared!!")
        exit(-1)
