import json
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
import re

import numpy as np


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def gen_input_name(shape, dtype):
    return "_".join([str(i) for i in shape]) + "_" + dtype


def gen_input(path, shape, dtype):
    if dtype == "ui8":
        res = np.random.randint(0, 255, size=shape, dtype="uint8")
    else:
        assert dtype == "f32", "" + dtype + " not support "
        res = np.random.randint(0, 255, size=shape).astype("float32") / 256
    res.tofile(path)
    np.save(path + ".npy", res, False)


def parse_model_info(model_infos, input_dir):
    input_list = []
    model_data_info = []
    model_data_info_local = []
    model_shape_info = []
    for model_info in model_infos[0]:
        model_input_info = []
        model_input_info_local = []
        model_input_shape_info = []
        for shape, dtype, input_name in model_info:
            input_file_name = gen_input_name(shape, dtype)
            input_path = os.path.join(input_dir, input_file_name)
            input_list.append({
                "input_file_name": input_file_name,
                "input_path": input_path,
                "shape": shape,
                "dtype": dtype,
                "input_name": input_name,
            })
            model_input_info.append(input_name + "=" + input_file_name)
            model_input_info_local.append(input_name + "=" + input_path)
            model_input_shape_info.append(input_name + "=(" +
                                          ",".join([str(i)
                                                    for i in shape]) + ")")
        model_data_info.append(";".join(model_input_info))
        model_shape_info.append(";".join(model_input_shape_info))
        model_data_info_local.append(";".join(model_input_info_local))
    model_data_info = ":".join(model_data_info)
    model_shape_info = ":".join(model_shape_info)
    model_data_info_local = ":".join(model_data_info_local)

    output_name_2_dtype = {}
    for output_info in model_infos[1]:
        for _, dtype, name in output_info:
            output_name_2_dtype[name] = dtype
    return input_list, model_data_info, model_shape_info, model_data_info_local, output_name_2_dtype


def local_call(cmd):
    return subprocess.check_call(cmd)


def target_call(host, cmd):
    return subprocess.check_call(["ssh", host, "-t"] + cmd)


def copy2target(host, workdir, origin_path):
    return subprocess.check_call(
        ["rsync", "-aP", "-zz", origin_path, host + ":" + workdir])


def copyftarget(host, workdir, origin_path):
    return subprocess.check_call(
        ["rsync", "-aP", "-zz", host + ":" + workdir, origin_path])


def compare_file(file_path_0, file_path_1, eps, out_dtype):
    print("compare ", file_path_0, file_path_1)
    with open(file_path_0, "rb") as f:
        d0 = np.frombuffer(f.read(), dtype=out_dtype)
    with open(file_path_1, "rb") as f:
        d1 = np.frombuffer(f.read(), dtype=out_dtype)
    assert d0.size == d1.size, "{} == {}".format(d0.size, d1.size)
    diff = (np.maximum(d0, d1) - np.minimum(d0, d1)) / np.maximum(1.0, np.minimum(
        np.abs(d0), np.abs(d1)))
    abs_diff = np.maximum(d0, d1) - np.minimum(d0, d1)
    max_idx = np.argmax(diff.flatten())
    print(d0.shape)
    print(
        "max diff ",
        np.max(diff.flatten()),
        "\nabs_diff",
        np.max(abs_diff.flatten()[max_idx]),
        "\n",
        d0[max_idx:max_idx + 10],
        " vs ",
        d1[max_idx:max_idx + 10],
        " at ",
        max_idx,
    )
    #! FIXME: the max abs diff should be zero when dtype is int?
    if out_dtype == np.int8 or out_dtype == np.uint8 or out_dtype == np.int32:
        assert np.all(abs_diff <= 1), "failed {} != {}, max : {}".format(
            d0, d1, np.max(abs_diff.flatten()))
        if np.max(abs_diff.flatten()) == 1:
            print("\033[31mWarning\033[0m: the max abs diff is 1")
    else:
        assert np.all(diff < eps), "failed {} != {}, max: {} ".format(
            d0, d1, np.max(diff.flatten()))


def compare_file_or_dir(path_0, path_1, eps, output_name_2_dtype):
    compare_file_cnt = 0
    dtype_str_2_np = {"f32": np.float32, "ui8": np.uint8,
                      "f16": np.float16, "si8": np.int8, "si32": np.int32}
    if os.path.isdir(path_0) and os.path.isdir(path_1):
        file_names = os.listdir(path_0)
        for file_name in file_names:
            output_name = None
            for i in output_name_2_dtype.keys():
                if re.search("^"+i+"_[0-9]+$", file_name):
                    output_name = i
                    break
            assert output_name
            file_path_0 = os.path.join(path_0, file_name)
            file_path_1 = os.path.join(path_1, file_name)
            assert os.path.exists(file_path_0), "can not find {}".format(
                file_path_0)
            assert os.path.exists(file_path_1), "can not find {}".format(
                file_path_1)
            compare_file(file_path_0, file_path_1, eps,
                         dtype_str_2_np[output_name_2_dtype[output_name]])
            compare_file_cnt += 1
    else:
        assert os.path.isfile(path_0), "can not find {}".format(path_0)
        assert os.path.isfile(path_1), "can not find {}".format(path_1)
        output_name = None
        for i in output_name_2_dtype.keys():
            if re.search(i+"_[0-9]+$", file_name):
                output_name = i
                break
        assert output_name
        compare_file(path_0, path_1, eps,
                     dtype_str_2_np[output_name_2_dtype[output_name]])
        compare_file_cnt += 1
    if compare_file_cnt > 0:
        print("compare pass!!")
        return True
    else:
        print("no file compared!!")
        return False


def parse_dump_dir(user_mdl_str):
    model_name_2_mdl = {}
    for model_case in user_mdl_str.split(","):
        if len(model_case) > 0:
            model_name, model_path = model_case.split(":")
            model_name_2_mdl[model_name] = model_path
    return model_name_2_mdl


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files and os.path.isfile(os.path.join(root, name)):
            return os.path.abspath(os.path.join(root, name))


def parse_env(dump_dir, bin_dir=None):
    script_abs = os.path.abspath(sys.argv[0])
    dev_flag = os.path.exists(
        os.path.join(os.path.dirname(script_abs), "release_megcc.sh"))
    print("dev flag", dev_flag)
    env = {}
    if dev_flag:
        # dev mode
        project_path = os.path.dirname(os.path.dirname(script_abs))
        env["build_script"] = os.path.join(project_path, "runtime", "scripts",
                                           "runtime_build.py")
        env["mgb_runner_path"] = find("mgb-runner", os.path.curdir)
        env["mgb_to_tinynn_path"] = find("mgb-to-tinynn", os.path.curdir)
        env["model_dir"] = dump_dir
        env["model_info_dir"] = dump_dir
        env["kern_dir"] = dump_dir
        env["build_dir"] = os.path.join(dump_dir, "build")
    else:
        # release mode
        env["build_script"] = os.path.join(dump_dir, "runtime/script",
                                           "runtime_build.py")
        bin_dir = os.path.abspath(bin_dir) if bin_dir else os.path.dirname(
            dump_dir)
        env["mgb_runner_path"] = find("mgb-runner", bin_dir)
        env["mgb_to_tinynn_path"] = find("mgb-to-tinynn", bin_dir)
        env["model_dir"] = os.path.join(dump_dir, "model")
        env["kern_dir"] = os.path.join(dump_dir, "kern")
        env["build_dir"] = os.path.join(dump_dir, "build")
        env["model_info_dir"] = os.path.join(dump_dir, "model_info")
    return env


def auto_check(model_name_2_all, eps, target_arch, target_host, env, mdl_str):
    build_script = env["build_script"]
    mgb_runner_path = env["mgb_runner_path"]
    mgb_to_tinynn_path = env["mgb_to_tinynn_path"]
    model_dir = env["model_dir"]
    kern_dir = env["kern_dir"]
    build_dir = env["build_dir"]
    model_info_dir = env["model_info_dir"]
    work_dir = os.path.join(os.path.curdir, "megcc_check_workdir")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    tinynn_test_lite_path = os.path.join(build_dir, "tinynn_test_lite")
    if not os.path.exists(build_dir):
        print("build not exit!!")
        os.mkdir(build_dir)
    subprocess.run([
        build_script,
        "--cross_build",
        "--remove_old_build",
        "--kernel_dir",
        kern_dir,
        "--cross_build_target_arch",
        target_arch,
        "--specify_build_dir",
        build_dir,
    ])

    for model_file in os.listdir(model_dir):
        if model_file.endswith(".tiny"):
            model_name = model_file[:-5]
            if model_name_2_all and model_name not in model_name_2_all.keys():
                continue
            model_path = os.path.join(model_dir, model_file)
            model_info_path = os.path.join(model_info_dir, model_file + ".txt")
            assert os.path.exists(model_info_path)
            model_infos = read_json(model_info_path)
            input_dir = os.path.join(work_dir, model_name + "_input")
            os.mkdir(input_dir)
            local_run_dir = os.path.join(work_dir, model_name + "_rundir")
            os.mkdir(local_run_dir)
            (
                input_list,
                model_data_info,
                model_shape_info,
                model_data_info_local,
                output_name_2_dtype
            ) = parse_model_info(model_infos, input_dir)
            for input_info in input_list:
                if not os.path.exists(input_info["input_path"]):
                    gen_input(
                        input_info["input_path"],
                        input_info["shape"],
                        input_info["dtype"],
                    )
            target_run_dir = model_name + "_rundir"

            target_call(target_host, ["rm", "-fr", target_run_dir])
            target_call(target_host,
                        ["mkdir", "-p", target_run_dir + "/tiny_out"])
            for input_info in input_list:
                copy2target(target_host, target_run_dir,
                            input_info["input_path"])

            copy2target(target_host, target_run_dir, tinynn_test_lite_path)
            copy2target(target_host, target_run_dir, model_path)
            target_shape_info = model_shape_info.replace("(",
                                                         "").replace(")", "")

            target_call(
                target_host,
                [
                    "cd",
                    target_run_dir,
                    "&&",
                    "./tinynn_test_lite",
                    "-m {} ".format(model_file),
                    "-o tiny_out ",
                    "-l 0 ",
                    "-d '{}' ".format(model_data_info),
                    "-s {} ".format(target_shape_info),
                ],
            )
            copyftarget(target_host, target_run_dir + "/tiny_out",
                        local_run_dir)

            if model_name in model_name_2_all:
                mdl_model_path = model_name_2_all[model_name]
                mgb_out_dir = os.path.join(local_run_dir, "mgb_out")
                os.mkdir(mgb_out_dir)
                if mdl_model_path.endswith(".emod"):
                    emod_path = mdl_model_path
                    mdl_model_path = "./decryption/" + os.path.basename(mdl_model_path) + ".mge"
                    local_call([mgb_to_tinynn_path, emod_path, "--decrypt"])
                local_call([
                    mgb_runner_path,
                    mdl_model_path,
                    mgb_out_dir,
                    "--input-shapes={}".format(model_shape_info),
                    "--input-data={}".format(model_data_info_local),
                ])
                try:
                    compare_file_or_dir(local_run_dir + "/tiny_out",
                                        mgb_out_dir, eps, output_name_2_dtype)
                except Exception as err:
                    print("{} result check failed for: {}".format(model_name, err))
                    exit(-1)
            else:
                print(
                    'can not find {} mdl path in mdl args "{}", skip correctness check'
                    .format(model_name, mdl_str))
            print("model {} done".format(model_name))


def main():
    parser = ArgumentParser(
        description=
        "auto benchmark and check correctness with mgb using random input")
    parser.add_argument(
        "--arch",
        type=str,
        required=False,
        default="aarch64",
        choices=[
            'x86_64', 'i386', 'aarch64', 'armv7-a', 'cortex-m', 'armv7-a-qemu',
            'rv64gcv0p7', 'rv64norvv'
        ],
    )
    parser.add_argument("--eps", type=float, required=False, default=3e-4)
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="target host str, `xiaomi9@phone`for termux",
    )
    parser.add_argument("dump_dir",
                        type=str,
                        help="dump dir contain model and kernels")
    parser.add_argument(
        "--mdl",
        type=str,
        required=False,
        default="",
        help=
        "input mdl path to do correctness check json str $model_name:$mdl_path,$model_name2:$mdl_path2",
    )
    parser.add_argument(
        "--bin_dir",
        type=str,
        required=False,
        default=None,
        help="bin dir contains mgb-runner",
    )
    args = parser.parse_args()
    eps = args.eps
    dump_dir = os.path.abspath(args.dump_dir)
    model_name_2_mdl = parse_dump_dir(args.mdl)
    env = parse_env(dump_dir, args.bin_dir)
    print(env)
    auto_check(
        model_name_2_mdl,
        eps=eps,
        target_arch=args.arch,
        target_host=args.target,
        env=env,
        mdl_str=args.mdl,
    )


if __name__ == "__main__":
    main()
