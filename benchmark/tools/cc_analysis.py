#! /usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(passed_args=None):
    parser = argparse.ArgumentParser(
        description="analyze profile result",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data")
    parser.add_argument("--output", "-o", default=".", type=str)
    args = parser.parse_args(passed_args)
    if not os.path.exists(args.output) or os.path.isfile(args.output):
        os.makedirs(args.output)
    files0 = set()
    if os.path.isdir(args.data):
        for i in os.listdir(args.data):
            files0.add(str(Path(args.data) / i))
    else:
        files0.add(args.data)
    data_map = {}
    data_info = []
    model_set = set()
    for i in files0:
        path = i.split("/")
        file_name = path[len(path) - 1].split(".")
        info = file_name[0].split("-")
        if info[0] == "megcc" and info[3] == "0":
            text_file = open(i, "r")
            data = text_file.read()
            text_file.close()
            pattern = re.compile(r"\s\w+\s[\r\n]+use\s\d*\.\d+")
            results = pattern.findall(data)
            analyze_data = []
            op_totoal_nums = len(results)
            op_per_test = int(op_totoal_nums / 60)
            iter_num = 0
            total = 0.0
            for i in results:
                kernel_name_pattern = re.compile(r"\s\w+\s")
                kernel_time_pattern = re.compile(r"\d*\.\d+")
                kernel_name = kernel_name_pattern.search(i).group()
                kernel_time = float(kernel_time_pattern.search(i).group())
                if iter_num < op_per_test:
                    total = total + kernel_time
                    analyze_data.append([kernel_name, kernel_time])
                else:
                    total = total + kernel_time
                    analyze_data[iter_num % op_per_test][1] += kernel_time

                iter_num = iter_num + 1
            diff_kernel_data = {}
            for i in analyze_data:
                if not i[0] in diff_kernel_data:
                    diff_kernel_data[i[0]] = [i[1], i[1] / total]
                else:
                    diff_kernel_data[i[0]][0] += i[1]
                    diff_kernel_data[i[0]][1] += i[1] / total
            kernel_name = []
            kernel_rate = []
            for k, v in sorted(
                diff_kernel_data.items(), key=lambda item: item[1][1], reverse=True
            ):
                kernel_name.append(k)
                kernel_rate.append(v[1] * 100)

            barWidth = 0.5
            topK = 10
            kernel_name = kernel_name[0:topK]
            kernel_rate = kernel_rate[0:topK]
            br1 = np.arange(len(kernel_name))
            plt.figure(figsize=(25, 6))
            plt.title("{}-{}-{}".format(info[1], info[5], info[2]), fontsize=30)
            plt.pie(kernel_rate, labels=kernel_name, autopct="%0.1f%%")
            plt.savefig(
                "{}/{}-{}-{}-profile-top{}.png".format(
                    args.output, info[1], info[5], info[2], topK
                )
            )


if __name__ == "__main__":
    main()
