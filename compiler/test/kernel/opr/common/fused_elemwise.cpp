/**
 * \file
 * compiler/test/kernel/opr/common/fused_elementwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "fused_elemwise.h"
#include <string>
#include "compiler/Common/TContext.h"
#include "megdnn/handle.h"
#include "test/kernel/common/src/cc_proxy_utils.h"
#include "test/kernel/opr/common/elemwise.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = ElemwiseForward::Param::Mode;

namespace {
std::vector<std::string> split_string(const std::string& s, const char delim) {
    std::istringstream iss(s);
    std::string item;
    std::vector<std::string> result;
    while (std::getline(iss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

Mode string_to_mode(std::string mode_str) {
    if ("RELU" == mode_str) {
        return Mode::RELU;
    } else if ("EXP" == mode_str) {
        return Mode::EXP;
    }  else if ("ABS" == mode_str) {
        return Mode::ABS;
    } else if ("NEGATE" == mode_str) {
        return Mode::NEGATE;
    } else if ("H_SWISH" == mode_str) {
        return Mode::H_SWISH;
    } else if ("ADD" == mode_str) {
        return Mode::ADD;
    } else if ("SUB" == mode_str) {
        return Mode::SUB;
    } else if ("MUL" == mode_str) {
        return Mode::MUL;
    } else if ("MAX" == mode_str) {
        return Mode::MAX;
    } else if ("MIN" == mode_str) {
        return Mode::MIN;
    } else if ("SIGMOID" == mode_str) {
        return Mode::SIGMOID;
    } else if ("TRUE_DIV" == mode_str) {
        return Mode::TRUE_DIV;
    } else if ("FUSE_ADD_RELU" == mode_str) {
        return Mode::FUSE_ADD_RELU;
    } else if ("FUSE_ADD_SIGMOID" == mode_str) {
        return Mode::FUSE_ADD_SIGMOID;
    } else if ("FUSE_MUL_ADD3" == mode_str) {
        return Mode::FUSE_MUL_ADD3;
    } else if ("FUSE_MUL_ADD4" == mode_str) {
        return Mode::FUSE_MUL_ADD4;
    }
    printf("mode is not support %s\n", mode_str.c_str());
}

std::shared_ptr<TensorNDArray> fused_elemwise_compute_dnn_truth(
        std::shared_ptr<TensorNDArray> inputs, std::vector<std::string> modes,
        megdnn::Handle* dnn_handle) {
    Checker<ElemwiseForward> checker;
    auto opr = dnn_handle->template create_operator<ElemwiseForward>();
    megdnn::test::DnnOprProxy<ElemwiseForward> dnn_proxy;
    TensorNDArray mid_tensor;
    std::vector<std::shared_ptr<TensorNDArray>> storage_hold;
    for (auto& mode_str : modes) {
        auto strs = split_string(mode_str, ',');
        size_t nr_str = strs.size();
        TensorNDArray in_tensor;
        TensorLayoutArray in_layouts;
        for (size_t i = 0; i < nr_str - 2; i++) {
            if (strs[i][0] == 'I') {
                auto id = strs[i][1] - '0';
                in_tensor.push_back((*inputs)[id]);
            } else if (strs[i][0] == 'O') {
                auto id = strs[i][1] - '0';
                in_tensor.push_back(mid_tensor[id]);
            }
            in_layouts.push_back(in_tensor.back().layout);
        }
        opr->param().mode = string_to_mode(strs[nr_str - 2]);
        auto all_layouts = in_layouts;
        all_layouts.push_back({});
        //! deduce the output layout
        dnn_proxy.deduce_layout(opr.get(), all_layouts);
        auto out_layout = all_layouts.back();
        //! allocate output memory
        auto output_storage =
                megdnn::test::dnn_alloc_tensors(dnn_handle, {out_layout}, 0);
        storage_hold.push_back(output_storage);
        in_tensor.push_back((*output_storage)[0]);

        //! exec operator
        dnn_proxy.exec(opr.get(), in_tensor);
        mid_tensor.push_back(in_tensor.back());
    }
    //! the last output tensor is the output
    return storage_hold.back();
}

std::shared_ptr<TensorNDArray> fused_elemwise_compute_cc(
        std::shared_ptr<TensorNDArray> inputs, TensorLayout out_layout,
        std::vector<std::string> modes, megcc::KernelGen::Arch arch,
        megdnn::Handle* dnn_handle, const std::string& symbol) {
    CCOprProxy<ElemwiseForward> cc_proxy;
    std::unordered_map<std::string, megcc::CCAttr> proxy_attr;
    int id = 0;
    std::string key = "modes:";
    for (auto& mode_str : modes) {
        proxy_attr[key + std::to_string(id)] = mode_str;
        id++;
    }
    proxy_attr[key + "size"] = id;

    auto output_storage =
            megdnn::test::dnn_alloc_tensors(dnn_handle, {out_layout}, 0);
    auto in_tensors = *inputs;
    in_tensors.push_back((*output_storage)[0]);
    fused_elemwise_exec(in_tensors, arch, proxy_attr, symbol);
    return output_storage;
}
}  // namespace

namespace megcc {
namespace test {

void check_fuse_elemwise(TensorShapeArray shapes,
                         std::vector<std::string> modes,
                         megcc::KernelGen::Arch arch, const std::string& symbol,
                         float epsilon) {
    Checker<ElemwiseForward> checker;
    auto dnn_handle = checker.get_dnn_handle();
    auto inputs_layout = checker.make_layouts(shapes, {}, {});
    auto inputs = megdnn::test::dnn_alloc_tensors(dnn_handle, inputs_layout, 0);
    UniformIntRNG rng(-10, 10);
    std::unordered_map<size_t, RNG*> rng_map;
    for (size_t i = 0; i < shapes.size(); i++) {
        rng_map[i] = &rng;
    }
    checker.init_tensor(*inputs, {});
    auto outputs_truth =
            fused_elemwise_compute_dnn_truth(inputs, modes, dnn_handle);
    auto output_layout = (*outputs_truth)[0].layout;
    auto outputs = fused_elemwise_compute_cc(inputs, output_layout, modes, arch,
                                             dnn_handle, symbol);
#if !MEGCC_TEST_GEN
    checker.check_tensors(*outputs_truth, *outputs, epsilon, epsilon, epsilon);
#endif
}

}  // namespace test
}  // namespace megcc

// vim: syntax=cpp.doxygen
