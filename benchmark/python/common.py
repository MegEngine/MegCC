#! /usr/bin/env python3
import logging
import os
import subprocess
from pathlib import Path

megcc_path = Path(os.path.split(
    os.path.realpath(__file__))[0]).parent.parent.absolute()
kernel_build_dirs = {}


def prepare_megcc():
    # build prepare
    MEGCC_MGB_TO_TINYNN_PATH = os.environ.get("MEGCC_MGB_TO_TINYNN_PATH")
    assert (
        MEGCC_MGB_TO_TINYNN_PATH != None
    ), "MEGCC_MGB_TO_TINYNN_PATH is not valid, please export MEGCC_MGB_TO_TINYNN_PATH to your path of mgb_to_tinynn"


def build_megcc_lib(arch_desc="x86",
                    model_config_json="",
                    kernel_build_dir=""):
    MEGCC_MGB_TO_TINYNN_PATH = os.environ.get("MEGCC_MGB_TO_TINYNN_PATH")
    # build prepare
    change_dir = ""
    if model_config_json == "":
        arch_ = arch_desc
        if arch_desc == "arm64" or arch_desc == "armv7":
            arch_ = "arm"
        model_config_json = "{}/benchmark/model/model_{}.json".format(
            megcc_path, arch_)
        change_dir = "cd {}/benchmark/model".format(megcc_path)
    else:
        change_dir = "cd {}".format(
            Path(model_config_json).parent().absolute())
    if kernel_build_dir == "":
        # WARNING: the dir path should be the same with path set in model_config_json file
        kernel_build_dir = "{}/benchmark/model/benchmark_kernel_{}".format(
            megcc_path, arch_desc)
    if not os.path.exists(kernel_build_dir) or os.path.isfile(
            kernel_build_dir):
        os.makedirs(kernel_build_dir)
    # set runtime build options
    if arch_desc == "x86":
        arch = "--baremetal"
        runtime_flag = ""
    elif arch_desc == "arm64":
        arch = "--arm64"
        runtime_flag = "--cross_build --cross_build_target_arch aarch64 --cross_build_target_os ANDROID"
    elif arch_desc == "armv7":
        arch = "--armv7"
        runtime_flag = "--cross_build --cross_build_target_arch armv7-a --cross_build_target_os ANDROID "
    elif arch_desc == "riscv":
        arch = "--baremetal"
        runtime_flag = "--cross_build --cross_build_target_arch rv64gcv0p7 --cross_build_target_os LINUX"

    # convert model
    if len(change_dir) != 0:
        cmd = "{} && {}/mgb-to-tinynn -json={} {} --dump {}".format(
            change_dir,
            MEGCC_MGB_TO_TINYNN_PATH,
            model_config_json,
            arch,
            kernel_build_dir,
        )
    else:
        cmd = "{} && {}/mgb-to-tinynn -json={} {} --dump {}".format(
            change_dir,
            MEGCC_MGB_TO_TINYNN_PATH,
            model_config_json,
            arch,
            kernel_build_dir,
        )
    subprocess.check_call(cmd, shell=True)
    # build runtime
    cmd = "python3 {}/runtime/scripts/runtime_build.py --build_with_profile --kernel_dir {}/ --remove_old_build {}".format(
        megcc_path, kernel_build_dir, runtime_flag)
    subprocess.check_call(cmd, shell=True)


def build_model_and_megcc_lib(models, models_dir, arch_str):
    # dump all models from onnx to megengine
    models.make(models_dir)
    # prepare megcc compiler
    prepare_megcc()
    # build megcc model lib
    for arch_desc in arch_str:
        kernel_build_dirs[
            arch_desc] = "{}/benchmark/model/benchmark_kernel_{}".format(
                megcc_path, arch_desc)
        build_megcc_lib(
            arch_desc,
            model_config_json="",
            kernel_build_dir=kernel_build_dirs[arch_desc],
        )


def build_benchmarker(x86_target="fallback", arch_str=None, benchmarkers=None):
    for arch_desc in arch_str:
        benchmark_build_dir = "{}/benchmark/build/{}".format(
            megcc_path, arch_desc)
        subprocess.check_call("rm -rf {}".format(benchmark_build_dir),
                              shell=True)
        if arch_desc == "x86":
            build_option = "-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/install -DRUNTIME_KERNEL_DIR={}".format(
                kernel_build_dirs[arch_desc])
        else:
            if arch_desc == "arm64":
                TOOLCHAIN_OPTION = '-DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake"  -DANDROID_NDK="$NDK_ROOT" -DANDROID_ABI=arm64-v8a  -DANDROID_NATIVE_API_LEVEL=21'
            elif arch_desc == "armv7":
                TOOLCHAIN_OPTION = '-DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake"  -DANDROID_NDK="$NDK_ROOT" -DANDROID_ABI=armeabi-v7a  -DANDROID_NATIVE_API_LEVEL=21'
            elif arch_desc == "riscv":
                TOOLCHAIN_OPTION = '-DCMAKE_TOOLCHAIN_FILE="{}/runtime/toolchains/riscv64-linux-gnu.toolchain.cmake"'.format(
                    megcc_path)
            build_option = "{} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/install -DRUNTIME_KERNEL_DIR={}".format(
                TOOLCHAIN_OPTION, kernel_build_dirs[arch_desc])
        benchmarkers[arch_desc].build(build_options=build_option)
