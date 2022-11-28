/**
 * \file
 * compiler/test/kernel/common/src/cc_proxy.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "test/kernel/common/cc_proxy.h"
#include <data_struct.h>
#include <init.h>
#include <mutex>
#include <regex>
#include <string>
#include "megbrain/common.h"
#include "megcc_test_config.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"
#include "test/kernel/common/cv_opr.h"
#include "test/kernel/common/target_module.h"
#include "tinycv_c.h"

#include <malloc.h>
#include <sstream>
#include "compiler/Common/MemoryStatus.h"
#include "compiler/Common/TContext.h"
#include "compiler/KernelGen/KernelGen.h"
#include "test/kernel/common/src/cc_fill_attr.h"
#include "test/kernel/common/src/cc_proxy_utils.h"
#include "test/kernel/common/timer.h"

#if MEGCC_TEST_GEN
#include "compiler/KernelGen/JitExe.h"
#endif

using namespace megcc;
using namespace test;

namespace {
DType dnndtype_2_ccdtype(megdnn::DType dtype) {
    DType res;
    switch (dtype.enumv()) {
        case megdnn::DTypeEnum::Float32:
            res.type_enum = TinyNNDType::TinyNN_FLOAT;
            break;
        case megdnn::DTypeEnum::QuantizedS8:
            res.type_enum = TinyNNDType::TinyNN_QINT8;
            res.param.scale = dtype.param<megdnn::dtype::QuantizedS8>().scale;
            break;
        case megdnn::DTypeEnum::Int32:
            res.type_enum = TinyNNDType::TinyNN_INT;
            break;
        case megdnn::DTypeEnum::Int16:
            res.type_enum = TinyNNDType::TinyNN_INT16;
            break;
        case megdnn::DTypeEnum::Uint8:
            res.type_enum = TinyNNDType::TinyNN_UINT8;
            break;
        case megdnn::DTypeEnum::Int8:
            res.type_enum = TinyNNDType::TinyNN_INT8;
            break;
        case megdnn::DTypeEnum::QuantizedS32:
            res.type_enum = TinyNNDType::TinyNN_QINT32;
            res.param.scale = dtype.param<megdnn::dtype::QuantizedS32>().scale;
            break;
        default:
            mgb_assert(0, "no support dtype %s", dtype.name());
            break;
    }
    return res;
}

std::string input_shape_to_string(TContext* ctx) {
    int nr_input = ctx->getAttrInt("nr_operands") - 1;
    std::string result;
    for (int i = 0; i < nr_input; i++) {
        auto operand = ctx->getAttrOprand("operand:" + std::to_string(i));
        size_t nr_dim = operand.shape.size();
        if (nr_dim < 1) {
            continue;
        }
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < operand.shape.size(); ++i) {
            ss << operand.shape[i] << ", ";
        }
        ss << "]";
        result += ss.str();
    }
    return result;
}

Tensor dnntensor_2_cctensor(const megdnn::TensorND& tensor, const char* name) {
    Tensor res;
    res.ptr = tensor.raw_ptr();
    res.dtype = dnndtype_2_ccdtype(tensor.layout.dtype);
    res.layout = dnnlayout_2_cclayout(tensor.layout);
    res.name = const_cast<char*>(name);
    return res;
}
std::string to_string(Layout layout) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < layout.nr_dim; ++i) {
        ss << layout.dims[i] << "(" << layout.stride[i] << ")"
           << ", ";
    }
    ss << "]";
    return ss.str();
}
struct FreePtr {
    void operator()(void* ptr) { free(ptr); }
};

using KernType = KernelGen::KernelPack::KernType;

static inline bool check_layout_equal(const Layout& l1, const Layout& l2) {
    bool ok_dim = l1.nr_dim == l2.nr_dim && l1.format == l2.format;
    if (!ok_dim) {
        return false;
    }
    for (int i = 0; i < l1.nr_dim; ++i) {
        bool ok_shape =
                l1.dims[i] == l2.dims[i] && l1.stride[i] == l2.stride[i];
        if (!ok_shape) {
            return false;
        }
    }
    return true;
}
PerformanceResult proxy_kernel(TensorNDArray tensor_array, StdKernelCall func,
                               StdKernelInitCall init_func,
                               StdKernelWorkspaceCall workspace_func,
                               StdKernelDeduceCall deduce_func,
                               const BenchmarkOption& benchmark_option,
                               const size_t workspace_bytes,
                               OutputScope output_idx) {
    constexpr size_t mem_align_bytes = 64;
    megdnn::SmallVector<Tensor> cc_tensor_in;
    megdnn::SmallVector<std::string> cc_tensor_in_name;
    megdnn::SmallVector<Tensor> cc_tensor_out;
    for (int i = output_idx.start; i <= output_idx.end; ++i) {
        cc_tensor_out.push_back(
                dnntensor_2_cctensor(tensor_array.at(i), "cc_output"));
    }
    size_t input_cnt = 0;
    for (size_t i = 0; i < tensor_array.size(); ++i) {
        if ((int)i >= output_idx.start && (int)i <= output_idx.end) {
            continue;
        }
        cc_tensor_in_name.push_back(mgb::ssprintf("cc_input:%zu", input_cnt++));
        cc_tensor_in.push_back(dnntensor_2_cctensor(
                tensor_array[i], cc_tensor_in_name.back().c_str()));
    }
    auto make_pointer_array = [](auto&& data_vec) {
        std::vector<decltype(data_vec.data())> ret;
        ret.reserve(data_vec.size());
        for (auto&& i : data_vec) {
            ret.push_back(&i);
        }
        return ret;
    };
    //! get workspace size should be ahead of init function
    auto input_ptr_array = make_pointer_array(cc_tensor_in);
    auto output_ptr_array = make_pointer_array(cc_tensor_out);

    size_t workspace_size = 0;
    workspace_func(input_ptr_array.data(), input_ptr_array.size(), 1,
                   &workspace_size);
    mgb_assert(
            workspace_bytes == workspace_size,
            "two method workspace must equal, jit get %zu, runtime get %zu\n",
            workspace_bytes, workspace_size);
    std::vector<std::vector<char>> preprocessed_weight_storage;
    if (init_func) {
        int nr_weight_after_process = 0;

        auto input_ptr_array = make_pointer_array(cc_tensor_in);
        init_func(input_ptr_array.data(), input_ptr_array.size(), NULL,
                  &nr_weight_after_process, nullptr);

        if (nr_weight_after_process) {
            std::vector<Tensor> new_cc_weight_vec(nr_weight_after_process);
            preprocessed_weight_storage.resize(nr_weight_after_process);
            init_func(input_ptr_array.data(), input_ptr_array.size(),
                      new_cc_weight_vec.data(), NULL, nullptr);
            for (int i = 0; i < nr_weight_after_process; ++i) {
                size_t size_in_bytes =
                        tensor_length_in_byte(&new_cc_weight_vec[i]);
                preprocessed_weight_storage[i].resize(size_in_bytes);
                new_cc_weight_vec[i].ptr =
                        preprocessed_weight_storage[i].data();
            }
            init_func(input_ptr_array.data(), input_ptr_array.size(),
                      new_cc_weight_vec.data(), &nr_weight_after_process,
                      nullptr);
            for (size_t i = 0; i < input_ptr_array.size(); i++) {
                auto&& old_weight = cc_tensor_in[i];
                for (auto&& new_weight : new_cc_weight_vec) {
                    if (old_weight.name == new_weight.name) {
                        old_weight = new_weight;
                        break;
                    }
                }
            }
        }
    }

    std::unique_ptr<void, FreePtr> workspace_mem_ptr(
            memalign(mem_align_bytes, workspace_size));
    mgb_assert(workspace_mem_ptr.get());
    Workspace workspace{workspace_mem_ptr.get(), workspace_size, 0};
    Workspace* workspace_ptr = &workspace;
    RuntimeOpt* runtime_opt = nullptr;
    if (deduce_func) {
        //! test deduce func
        megdnn::SmallVector<Tensor> cc_tensor_out_shape;
        for (int i = output_idx.start; i <= output_idx.end; ++i) {
            auto tensor = dnntensor_2_cctensor(tensor_array.at(i), "cc_output");
            memset(&(tensor.layout), 0, sizeof(tensor.layout));
            cc_tensor_out_shape.push_back(tensor);
        }
        auto output_ptr_shape = make_pointer_array(cc_tensor_out_shape);
        deduce_func(input_ptr_array.data(), input_ptr_array.size(),
                    output_ptr_shape.data(), output_ptr_shape.size());
        //! check layout with dnn
        for (size_t i = 0; i < cc_tensor_out_shape.size(); ++i) {
            mgb_assert(check_layout_equal(output_ptr_shape[i]->layout,
                                          output_ptr_array[i]->layout),
                       "deduce layout must equal with dnn at output %zu :: %s "
                       "!= %s",
                       i, to_string(output_ptr_shape[i]->layout).c_str(),
                       to_string(output_ptr_array[i]->layout).c_str());
        }
    }

    func(input_ptr_array.data(), input_ptr_array.size(),
         output_ptr_array.data(), output_ptr_array.size(), workspace_ptr,
         runtime_opt);
    if (benchmark_option.valid_megcc_performance) {
        if (benchmark_option.warmup_iter > 0) {
            for (int i = 0; i < benchmark_option.warmup_iter; ++i) {
                func(input_ptr_array.data(), input_ptr_array.size(),
                     output_ptr_array.data(), output_ptr_array.size(),
                     workspace_ptr, runtime_opt);
            }
        }
        mgb_assert(benchmark_option.test_iter > 0);
        megcc::test::Timer timer;
        timer.start();
        for (int i = 0; i < benchmark_option.test_iter; ++i) {
            func(input_ptr_array.data(), input_ptr_array.size(),
                 output_ptr_array.data(), output_ptr_array.size(),
                 workspace_ptr, runtime_opt);
        }

        timer.stop();
        PerformanceResult res;
        res.valid = true;
        res.kernel_time_ms =
                timer.get_time_in_us() / 1e3 / benchmark_option.test_iter;
        return res;
    }
    return {};
}

void gen_kernel(KernelGenRet& kernels, TContext* ctx, KernelGen::Arch arch,
                const std::string& kernel_symbol, bool gen_deduce_func) {
    int usable_kern_cnt = 0;
    for (auto kernel : kernels.first) {
        if (kernel->IsAvailable(ctx)) {
            usable_kern_cnt++;
            TargetModule& g_module = TargetModule::get_global_target_module();
            auto kern_sym = kernel->GetKernelSymbol(ctx);
            auto if_match =
                    std::regex_match(kern_sym, std::regex(kernel_symbol));
            if (!if_match)
                continue;
            if (!g_module.exist(kern_sym)) {
                std::string deduce_sym = "";
                std::string deduce_body = "";
                if (gen_deduce_func) {
                    mgb_assert(kernels.second, "must have deduce func");
                    deduce_sym = kernels.second->GetDeduceSymbol(ctx);
                    deduce_body = kernels.second->GetDeduceBody(ctx);
                }
                g_module.add(
                        kern_sym, kernel->GetKernelBody(ctx),
                        kernel->GetInitSymbol(ctx), kernel->GetInitBody(ctx),
                        kernel->GetWorkspaceSymbol(ctx),
                        kernel->GetWorkspaceBody(ctx), deduce_sym, deduce_body);
                auto depends = kernel->GetDependInternalSymbol(ctx);
                gen_depend_kernels(arch, depends);
            }
            size_t workspace_bytes = 0;
#if MEGCC_TEST_GEN
            workspace_bytes =
                    megcc::KernelGen::JitExec::jit_exec_and_get_workspace(
                            kernel, ctx);
#endif
            auto workspace_size_symbol = kernel->GetWorkspaceSymbol(ctx) +
                                         input_shape_to_string(ctx);
            g_module.add_workspace_size(workspace_size_symbol, workspace_bytes);
            return;
        }
    }
    mgb_assert(0, "gen kernel failed, available %d", usable_kern_cnt);
}

PerformanceResult call_kernel(
        std::pair<std::vector<const KernelGen::KernelFunc*>,
                  const KernelGen::DeduceFunc*>& kernels,
        TContext* ctx, const TensorNDArray& tensors,
        const BenchmarkOption& benchmark_option,
        const std::string& kernel_symbol, const OutputScope output_idx,
        bool dynamic_shape) {
    std::string kern_name;
    std::string kern_init_name;
    std::string kern_deduce_name;
    std::string kern_workspace_name;
    for (auto kernel : kernels.first) {
        if (kernel->IsAvailable(ctx)) {
            auto kern_sym = kernel->GetKernelSymbol(ctx);
            auto if_match =
                    std::regex_match(kern_sym, std::regex(kernel_symbol));
            if (!if_match)
                continue;
            kern_name = kern_sym;
            kern_init_name = kernel->GetInitSymbol(ctx);
            kern_workspace_name = kernel->GetWorkspaceSymbol(ctx);
            if (dynamic_shape) {
                mgb_assert(kernels.second, "deduce func can not be null");
                kern_deduce_name = kernels.second->GetDeduceSymbol(ctx);
            }
            break;
        }
    }
    mgb_assert(kern_name.size() > 0 && kern_init_name.size() > 0,
               "gen kernel name failed");
    TargetModule& g_module = TargetModule::get_global_target_module();
    auto func = g_module.get_kernel(kern_name);
    mgb_assert(func, "can not get kernel[%s] from target module",
               kern_name.c_str());
    auto init_func = g_module.get_kernel_init(kern_init_name);
    mgb_assert(init_func, "can not get init kernel[%s] from target module",
               kern_init_name.c_str());
    auto workspace_func = g_module.get_kernel_workspace(kern_workspace_name);
    mgb_assert(workspace_func,
               "can not get workspace kernel[%s] from target module",
               kern_workspace_name.c_str());
    auto workspace_size_symbol =
            kern_workspace_name + input_shape_to_string(ctx);
    size_t workspace_bytes =
            g_module.get_kernel_workspace_size(workspace_size_symbol);
    StdKernelDeduceCall deduce_func = nullptr;
    if (dynamic_shape) {
        deduce_func = g_module.get_kernel_deduce(kern_deduce_name);
        mgb_assert(deduce_func,
                   "can not get deduce func[%s] from target module",
                   kern_deduce_name.c_str());
    }
    return proxy_kernel(tensors, func, init_func, workspace_func, deduce_func,
                        benchmark_option, workspace_bytes, output_idx);
}

}  // namespace

void megcc::test::fused_elemwise_exec(
        const TensorNDArray& tensors, KernelGen::Arch arch,
        std::unordered_map<std::string, CCAttr>& proxy_attr,
        const std::string& symbol) {
    OutputScope output_idx{-1, -1};
    output_idx.normalize((int)tensors.size());
    fill_operands(proxy_attr, tensors, output_idx, false);
    megcc::CodeGenContext ctx(proxy_attr);
    auto kernels = megcc::KernelGen::KernelPack::GetKernel(
            KernType::FusedElemwiseKernel, arch);
#if MEGCC_TEST_GEN
    gen_kernel(kernels, &ctx, arch, symbol, false);
#else
    //! call kernel
    call_kernel(kernels, &ctx, tensors, {}, symbol, output_idx, false);
#endif
}

namespace megcc {
namespace test {

//! [start, end]
template <typename Opr>
OutputScope CCOprProxy<Opr>::get_output_idx(Opr*) {
    return {-1, -1};
}
template <>
OutputScope CCOprProxy<megdnn::Argsort>::get_output_idx(megdnn::Argsort*) {
    return {-2, -1};
}
template <>
OutputScope CCOprProxy<megdnn::TopK>::get_output_idx(megdnn::TopK* opr) {
    if (opr->param().mode == megdnn::TopK::Param::Mode::KTH_ONLY)
        return {-1, -1};
    else
        return {-2, -1};
}
template <>
OutputScope CCOprProxy<megdnn::IndexingMultiAxisVec>::get_output_idx(
        megdnn::IndexingMultiAxisVec*) {
    return {1, 1};
}

template <typename Opr>
PerformanceResult CCOprProxy<Opr>::exec(
        Opr* opr, const TensorNDArray& tensors, KernelGen::Arch arch,
        const BenchmarkOption& benchmark_option,
        const std::string& kernel_symbol,
        const std::unordered_map<std::string, CCAttr>& proxy_attr,
        bool gen_dynamic) {
    std::unordered_map<std::string, CCAttr> attr_map;
    auto kernels = opr_fill_attr<Opr>(attr_map, opr, tensors, arch, proxy_attr);
    auto output_idx = get_output_idx(opr);
    output_idx.normalize((int)tensors.size());
    fill_operands(attr_map, tensors, output_idx, gen_dynamic);
    CodeGenContext ctx(attr_map);
    MGB_MARK_USED_VAR(call_kernel);
    MGB_MARK_USED_VAR(gen_kernel);
#if MEGCC_TEST_GEN
    gen_kernel(kernels, &ctx, arch, kernel_symbol, gen_dynamic);
    return {};
#else
    //! call kernel
    return call_kernel(kernels, &ctx, tensors, benchmark_option, kernel_symbol,
                       output_idx, gen_dynamic);
#endif
}

#undef FILL_MAP
#undef FILL_MAP_EX

#define DEF_CCOPRPROXY(_OPR_CLS)                                               \
    template PerformanceResult CCOprProxy<_OPR_CLS>::exec(                     \
            _OPR_CLS* opr, const TensorNDArray& tensors, KernelGen::Arch arch, \
            const BenchmarkOption& benchmark_option,                           \
            const std::string& kernel_symbol,                                  \
            const std::unordered_map<std::string, CCAttr>& proxy_attr,         \
            bool gen_dynamic)

DEF_CCOPRPROXY(megdnn::ElemwiseForward);
DEF_CCOPRPROXY(megdnn::ElemwiseMultiType);
DEF_CCOPRPROXY(megdnn::ConvolutionForward);
DEF_CCOPRPROXY(megdnn::ConvBiasForward);
DEF_CCOPRPROXY(megdnn::ConvolutionBackwardData);
DEF_CCOPRPROXY(megdnn::PoolingForward);
DEF_CCOPRPROXY(megdnn::MatrixMulForward);
DEF_CCOPRPROXY(megdnn::MatrixInverse);
DEF_CCOPRPROXY(megdnn::IndexingMultiAxisVec);
DEF_CCOPRPROXY(megdnn::IndexingOneHot);
DEF_CCOPRPROXY(megdnn::ReduceForward);
DEF_CCOPRPROXY(megdnn::WarpAffineForward);
DEF_CCOPRPROXY(megdnn::WarpPerspectiveForward);
DEF_CCOPRPROXY(megdnn::BatchedMatrixMulForward);
DEF_CCOPRPROXY(megdnn::TypeCvtForward);
DEF_CCOPRPROXY(megdnn::TopK);
DEF_CCOPRPROXY(megdnn::RelayoutForward);
DEF_CCOPRPROXY(megdnn::PowC);
DEF_CCOPRPROXY(megdnn::ResizeForward);
DEF_CCOPRPROXY(megdnn::Argsort);
DEF_CCOPRPROXY(megdnn::ConcatForward);
DEF_CCOPRPROXY(megdnn::ArgmaxForward);

#undef DEF_CCOPRPROXY

}  // namespace test
}  // namespace megcc

// vim: syntax=cpp.doxygen
