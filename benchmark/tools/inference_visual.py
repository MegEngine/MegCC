#! /usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(passed_args=None):
    parser = argparse.ArgumentParser(
        description="visualize inference result",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data")
    parser.add_argument("--no_figure", "-f", action="store_true")
    parser.add_argument("--output", "-o", default=".", type=str)
    args = parser.parse_args(passed_args)
    files0 = set()
    if not os.path.exists(args.output) or os.path.isfile(args.output):
        os.makedirs(args.output)
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

        if "{}-{}".format(info[1], info[5]) not in data_map:
            data_map["{}-{}".format(info[1], info[5])] = {}
        if info[0] not in data_map["{}-{}".format(info[1], info[5])]:
            data_map["{}-{}".format(info[1], info[5])][info[0]] = []
        if info[3] == "3":
            text_file = open(i, "r")
            data = text_file.read()
            text_file.close()
            pattern = re.compile(r"\d*\.\d+")
            result = float(pattern.search(data).group())
            #  for excel
            data_info.append([info[1], info[5], info[0], info[2], result])
            data_map["{}-{}".format(info[1], info[5])][info[0]].append(
                [info[2], result])
            model_set.add(info[2])

    model_list = []
    for model in model_set:
        model_list.append(model)
    model_list = sorted(model_list)

    for k, v in data_map.items():
        for k0, v0 in v.items():
            v1 = sorted(v0, key=lambda item: item[0])
            v1_val = []
            for i in v1:
                v1_val.append(i[1])
            data_map[k][k0] = v1_val
    for i in data_info:
        print(i[0], i[1], i[2], i[3], i[4])
    if not args.no_figure:
        print(model_list)
        print(data_map)
        # generate figure
        barWidth = 0.5
        br1 = np.arange(len(model_list))
        br2 = [x + barWidth for x in br1]
        for k, v in data_map.items():
            plt.figure(figsize=(10, 6))
            plt.title(k)
            # Make the plot
            plt.bar(br1,
                    v["megcc"],
                    width=barWidth,
                    edgecolor="grey",
                    label="megcc")

            # Adding Xticks
            plt.xlabel("model", fontweight="bold", fontsize=15)
            plt.ylabel("inference(ms)", fontweight="bold", fontsize=15)
            plt.xticks([r + barWidth for r in range(len(model_list))],
                       model_list)
            plt.grid(axis="y")
            for a, b in zip(br1, v["megcc"]):
                plt.text(a, b + 0.05, "%.2f" % b, ha="center", va="bottom")

            plt.legend()
            plt.savefig("{}/{}.png".format(args.output, k))


if __name__ == "__main__":
    main()
