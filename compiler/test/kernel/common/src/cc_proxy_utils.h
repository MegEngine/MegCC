#pragma once

#include <data_struct.h>

#include "compiler/KernelGen/KernelGen.h"
#include "megbrain/common.h"
#include "megcc_test_config.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"
#include "test/kernel/common/cc_proxy.h"
#include "test/kernel/common/target_module.h"

namespace megcc {
namespace test {
namespace {

using KernType = KernelGen::KernelPack::KernType;

Layout dnnlayout_2_cclayout(const megdnn::TensorLayout& layout) {
    Layout res;
    mgb_assert(layout.ndim <= MAX_DIM);
    res.nr_dim = layout.ndim;
    for (size_t i = 0; i < layout.ndim; ++i) {
        res.dims[i] = layout.shape[i];
        res.stride[i] = layout.stride[i];
    }
    //! FIXME: detect layout
    res.format = TinyNNFormat::TinyNN_NCHW;
    return res;
}

uint32_t bit_cast(float scale) {
    uint32_t* bit_scale = (uint32_t*)&scale;
    return *bit_scale;
}

std::string dnndtype_2_str(megdnn::DType dtype, float scale) {
    switch (dtype.enumv()) {
        case megdnn::DTypeEnum::Float32:
            return "f32";
        case megdnn::DTypeEnum::Int8:
            return "si8";
        case megdnn::DTypeEnum::Int32:
            return "i32";
        case megdnn::DTypeEnum::Int16:
            return "i16";
        case megdnn::DTypeEnum::Uint8:
            return "ui8";
        case megdnn::DTypeEnum::QuantizedS8: {
            std::stringstream ss;
            ss << "qsi8<" << bit_cast(scale) << ":" << scale << ">";
            return ss.str();
        }
        case megdnn::DTypeEnum::QuantizedS32: {
            std::stringstream ss;
            ss << "qsi32<" << bit_cast(scale) << ":" << scale << ">";
            return ss.str();
        }
#if !MEGDNN_DISABLE_FLOAT16
        case megdnn::DTypeEnum::Float16:
            return "f16";
#endif
        default:
            mgb_assert(0, "no support dtype %s", dtype.name());
            break;
    }
    return "invalid";
}
megcc::CCOperand dnntensor_2_ccoperand(
        const megdnn::TensorND& tensor, bool is_dynamic) {
    megcc::CCOperand res;
    size_t unknow_shape = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < tensor.layout.ndim; ++i) {
        if (is_dynamic) {
            res.shape.push_back(unknow_shape);
        } else {
            res.shape.push_back(tensor.layout[i]);
        }
    }
    auto dtype_enum = tensor.layout.dtype.enumv();
    if (dtype_enum == megdnn::DTypeEnum::QuantizedS8) {
        res.scale = tensor.layout.dtype.param<megdnn::dtype::QuantizedS8>().scale;
    } else if (dtype_enum == megdnn::DTypeEnum::QuantizedS32) {
        res.scale = tensor.layout.dtype.param<megdnn::dtype::QuantizedS32>().scale;
    }
    res.dtype = dnndtype_2_str(tensor.layout.dtype, res.scale);
    return res;
}
void fill_operands(
        std::unordered_map<std::string, CCAttr>& map, const TensorNDArray& tensors,
        OutputScope output_idx, bool is_dynamic) {
    int tensor_size = (int)tensors.size();
    map["nr_operands"] = CCAttr(tensor_size);
    int input_cnt = 0;
    for (int i = 0; i < tensor_size; ++i) {
        if (i >= output_idx.start && i <= output_idx.end) {
            continue;
        }
        map["operand:" + std::to_string(input_cnt++)] =
                CCAttr(dnntensor_2_ccoperand(tensors[i], is_dynamic));
    }
    for (int i = output_idx.start; i <= output_idx.end; ++i) {
        map["operand:" + std::to_string(input_cnt++)] =
                CCAttr(dnntensor_2_ccoperand(tensors[i], is_dynamic));
    }
}

void gen_depend_kernels(
        KernelGen::Arch arch, const std::vector<KernelGen::KernelObj>& depends) {
    TargetModule& g_module = TargetModule::get_global_target_module();
    auto internal_kernels =
            KernelGen::KernelPack::GetKernel(KernType::InternelKernel, arch);
    for (auto& depend : depends) {
        if (!g_module.exist_internal_function(depend.kernel_symbol)) {
            if (depend.kernel_bin.size() > 0) {
                g_module.add_binary(depend.kernel_symbol, depend.kernel_bin);
            }
            g_module.add_internal_func(depend.kernel_symbol, depend.kernel_body);
            if (depend.kernel_dep.size() > 0) {
                gen_depend_kernels(arch, depend.kernel_dep);
            }
        }
    }
}

inline void add_test_mode_to_attr(std::unordered_map<std::string, CCAttr>& map) {
    map["unitest_mode"] = CCAttr(true);
}
}  // namespace
}  // namespace test
}  // namespace megcc