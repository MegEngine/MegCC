from builtins import print
import numpy as np
import argparse
import os
import struct
import textwrap
from pathlib import Path

def load_tensor_binary(fobj):
    """
    Load a tensor dumped by the :class:`BinaryOprIODump` plugin; the actual
    tensor value dump is implemented by ``mgb::debug::dump_tensor``.

    :param fobj: file object, or a string that contains the file name.
    :return: tuple ``(tensor_value, tensor_name)``.
    """
    if isinstance(fobj, str):
        with open(fobj, "rb") as fin:
            return load_tensor_binary(fin)

    DTYPE_LIST = {
        0: np.float32,
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        # 5: _mgb.intb1,
        # 6: _mgb.intb2,
        # 7: _mgb.intb4,
        8: None,
        9: np.float16,
        # quantized dtype start from 100000
        # see MEGDNN_PARAMETERIZED_DTYPE_ENUM_BASE in
        # dnn/include/megdnn/dtype.h
        100000: np.uint8,
        100001: np.int32,
        100002: np.int8,
    }

    header_fmt = struct.Struct("III")
    name_len, dtype, max_ndim = header_fmt.unpack(fobj.read(header_fmt.size))
    assert (
        DTYPE_LIST[dtype] is not None
    ), "Cannot load this tensor: dtype Byte is unsupported."

    shape = list(struct.unpack("I" * max_ndim, fobj.read(max_ndim * 4)))
    while shape[-1] == 0:
        shape.pop(-1)
    name = fobj.read(name_len).decode("ascii")
    return np.fromfile(fobj, dtype=DTYPE_LIST[dtype]).reshape(shape), name

def find_file_name(dir_path, varid):
    temp_str = "var={id:" + varid + ","
    if os.path.isdir(dir_path):
        file_names = os.listdir(dir_path)
        for file_name in file_names:
            tensor, tensor_name = load_tensor_binary(dir_path + file_name)
            if tensor_name.startswith(temp_str):
                return file_name


dir_path = "/home/liujunjie/gitlab/megcc/compiler/build_host/bin_dump/"
megcc_dir_path = "/home/liujunjie/gitlab/megcc/compiler/build_host/dump/"

varid='19615'
megcc_tensor_name = '176__tensor:228_kernel_typecvt_qsi8qsi8_1_256_16_16'
dtype=np.int8

varid='19623'
megcc_tensor_name = '178__tensor:230_kernel_conv2d_1x1_NCHW_DENSE_p0x0_s1x1_d1x1_qsi8qsi8qsi32qsi8_bias_1_256_16_16'
dtype=np.int8

file_name = find_file_name(dir_path, varid)
tensor, tensor_name = load_tensor_binary(dir_path+file_name)
np.save('./mgb_npy/'+varid+'.npy', tensor)
file_b_path = megcc_dir_path + megcc_tensor_name
with open(file_b_path, 'rb') as f:
    tensor_b = np.frombuffer(f.read(), dtype=dtype)
    tensor_b.reshape(tensor.shape)

d0 = tensor.flatten().astype("float32")
d1 = tensor_b.flatten().astype("float32")
assert d0.shape == d1.shape
diff = np.abs(d0 - d1) / np.maximum(1.0, np.minimum(np.abs(d0), np.abs(d1)))
abs_diff = np.max(np.abs(d0 - d1))

abs_sum = np.sum(np.abs(d0 - d1))
max_idx = np.argmax(np.abs(d0 - d1).flatten())
print(
    "max diff ",
    np.max(diff.flatten()),
    ",abs:",
    abs_diff,
    abs_sum,
    np.sum((d0 - d1).flatten()),
    ":",
    d0[max_idx : max_idx + 10],
    " vs ",
    d1[max_idx : max_idx + 10],
    " at ",
    max_idx,
    "shape ",
    d0.shape,
)
print("avg ", np.average(d0), np.average(d1))
print("ref head ", d0[:10], " compare to ", d1[:10])
