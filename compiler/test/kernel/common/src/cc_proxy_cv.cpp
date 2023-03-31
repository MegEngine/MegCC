/**
 * \file
 * compiler/test/kernel/common/src/cc_proxy_cv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <regex>

#include "test/kernel/common/cc_proxy.h"
#include "test/kernel/common/cv_opr.h"
#include "test/kernel/common/src/cc_fill_attr.h"
#include "test/kernel/common/src/cc_proxy_utils.h"
#include "test/kernel/common/target_module.h"
#include "test/kernel/common/timer.h"
#include "tinycv_c.h"

#include "compiler/Common/MemoryStatus.h"
#include "compiler/Common/TContext.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace test {
template <typename Opr>
OutputScope CCOprProxy<Opr>::get_output_idx(Opr*) {
    return {-1, -1};
}
namespace {
template <typename Opr>
class RunCvHelper {
public:
    RunCvHelper(Opr* opr);
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr);
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };
};

template <>
class RunCvHelper<megdnn::CVtranspose> {
public:
    typedef void (*CVtransposeFunc)(const TinyMat* src, const TinyMat* dst);
    RunCvHelper(megdnn::CVtranspose*){};
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        CVtransposeFunc func = (CVtransposeFunc)func_ptr;
        func(&mat_array[0], &mat_array[1]);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        TensorNDArray res = tensor_array;
        std::swap(res[0].layout[1], res[0].layout[2]);
        return res;
    };
};

template <>
class RunCvHelper<megdnn::CVflip> {
public:
    typedef void (*CVflipFunc)(
            const TinyMat* src, const TinyMat* dst, bool vertical, bool horizontal);
    RunCvHelper(megdnn::CVflip* opr) { m_param = opr->param(); };
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        CVflipFunc func = (CVflipFunc)func_ptr;
        func(&mat_array[0], &mat_array[1], m_param.vertical, m_param.horizontal);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
    megdnn::CVflip::Param m_param;
};

template <>
class RunCvHelper<megdnn::CVResize> {
public:
    typedef void (*ResizeFunc)(const TinyMat* src, const TinyMat* dst);
    RunCvHelper(megdnn::CVResize* opr){};
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        ResizeFunc func = (ResizeFunc)func_ptr;
        func(&mat_array[0], &mat_array[1]);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
};

template <>
class RunCvHelper<megdnn::CVRotate> {
public:
    typedef void (*CVrotateFunc)(
            const TinyMat* src, const TinyMat* dst, bool clockwise);
    RunCvHelper(megdnn::CVRotate* opr) { m_param = opr->param(); };
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        CVrotateFunc func = (CVrotateFunc)func_ptr;
        func(&mat_array[0], &mat_array[1], m_param.clockwise);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
    megdnn::CVRotate::Param m_param;
};

template <>
class RunCvHelper<megdnn::CVRoicopy> {
public:
    typedef void (*CVroicopyFunc)(
            const TinyMat* src, const TinyMat* dst, size_t row_from, size_t row_to,
            size_t col_from, size_t col_to);
    RunCvHelper(megdnn::CVRoicopy* opr) { m_param = opr->param(); };
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        CVroicopyFunc func = (CVroicopyFunc)func_ptr;
        func(&mat_array[0], &mat_array[1], m_param.row_from, m_param.row_to,
             m_param.col_from, m_param.col_to);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
    megdnn::CVRoicopy::Param m_param;
};

template <>
class RunCvHelper<megdnn::CVCvtColor> {
public:
    typedef void (*CVCvtColorFunc)(const TinyMat* src, const TinyMat* dst);
    RunCvHelper(megdnn::CVCvtColor* opr) { m_param = opr->param(); };
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        CVCvtColorFunc func = (CVCvtColorFunc)func_ptr;
        func(&mat_array[0], &mat_array[1]);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
    megdnn::CVCvtColor::Param m_param;
};

template <>
class RunCvHelper<megdnn::CVWarpAffine> {
public:
    typedef void (*CVWarpAffineFunc)(
            const TinyMat* src, const TinyMat* dst, const double* trans);
    typedef void (*CVWarpAffineFuncWithConst)(
            const TinyMat* src, const TinyMat* dst, const double* trans,
            uint8_t const_val);
    RunCvHelper(megdnn::CVWarpAffine* opr) { m_param = opr->param(); };
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        double mat[6];
        float* ori_mat = (float*)(mat_array[1].data);
        for (int i = 0; i < 6; ++i) {
            mat[i] = ori_mat[i];
        }
        if (m_param.border_mode ==
            megdnn::CVWarpAffine::Param::BorderMode::BORDER_CONSTANT) {
            CVWarpAffineFuncWithConst func = (CVWarpAffineFuncWithConst)func_ptr;
            uint8_t const_val = m_param.border_val;
            func(&mat_array[0], &mat_array[2], mat, const_val);
        } else {
            CVWarpAffineFunc func = (CVWarpAffineFunc)func_ptr;
            func(&mat_array[0], &mat_array[2], mat);
        }
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
    megdnn::CVWarpAffine::Param m_param;
};

template <>
class RunCvHelper<megdnn::CVGaussianBlur> {
public:
    typedef void (*CVGaussianBlurFunc)(
            const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
            double sigma1, double sigma2);
    RunCvHelper(megdnn::CVGaussianBlur* opr) { m_param = opr->param(); };
    void run_cv_kernel(megdnn::SmallVector<TinyMat>& mat_array, void* func_ptr) {
        CVGaussianBlurFunc func = (CVGaussianBlurFunc)func_ptr;
        func(&mat_array[0], &mat_array[1], m_param.kernel_height, m_param.kernel_width,
             m_param.sigma_x, m_param.sigma_y);
    };
    TensorNDArray on_tensor_before(TensorNDArray& tensor_array) {
        return tensor_array;
    };

private:
    megdnn::CVGaussianBlur::Param m_param;
};

static inline TinyMat tensor2TinyMat(const megdnn::TensorND& tensor) {
    auto layout = tensor.layout;
    mgb_assert(
            layout.ndim == 3 || ((layout.ndim == 4) && (layout[0] == 1)),
            "failed %s, %d %d", layout.to_string().c_str(), (layout.ndim == 4),
            (layout[0] == 1));
    int idx_offset = layout.ndim - 3;
    TinyMat res;
    res.rows = layout[idx_offset + 0];
    res.cols = layout[idx_offset + 1];
    res.channels = layout[idx_offset + 2];
    res.data = tensor.raw_ptr();
    return res;
}
template <typename Opr>
PerformanceResult proxy_cv_kernel(
        TensorNDArray tensor_array, RunCvHelper<Opr>& runner, void* func_ptr,
        const BenchmarkOption& benchmark_option, OutputScope output_idx) {
    megdnn::SmallVector<TinyMat> mat_vec;
    auto temp_tensor = runner.on_tensor_before(tensor_array);
    for (auto& tensor : temp_tensor) {
        mat_vec.push_back(tensor2TinyMat(tensor));
    }
    runner.run_cv_kernel(mat_vec, func_ptr);
    if (benchmark_option.valid_megcc_performance) {
        if (benchmark_option.warmup_iter > 0) {
            for (int i = 0; i < benchmark_option.warmup_iter; ++i) {
                runner.run_cv_kernel(mat_vec, func_ptr);
            }
        }
        mgb_assert(benchmark_option.test_iter > 0);
        megcc::test::Timer timer;
        timer.start();
        for (int i = 0; i < benchmark_option.test_iter; ++i) {
            runner.run_cv_kernel(mat_vec, func_ptr);
        }

        timer.stop();
        PerformanceResult res;
        res.valid = true;
        res.kernel_time_ms = timer.get_time_in_us() / 1e3 / benchmark_option.test_iter;
        return res;
    }
    return {};
}

template <typename Opr>
PerformanceResult call_kernel_cv(
        std::vector<const KernelGen::KernelFunc*>& kernels, TContext* ctx,
        const TensorNDArray& tensors, const BenchmarkOption& benchmark_option,
        const std::string& kernel_symbol, RunCvHelper<Opr>& runner,
        const OutputScope output_idx) {
    std::string kern_name;
    for (auto kernel : kernels) {
        if (kernel->IsCVAvailable(ctx)) {
            auto kern_sym = kernel->GetCVKernelSymbol(ctx);
            auto if_match = std::regex_match(kern_sym, std::regex(kernel_symbol));
            if (!if_match)
                continue;
            kern_name = kern_sym;
            break;
        }
    }
    mgb_assert(kern_name.size() > 0, "gen kernel name failed");
    TargetModule& g_module = TargetModule::get_global_target_module();
    void* func = g_module.get_cv_kernel(kern_name);
    return proxy_cv_kernel(tensors, runner, func, benchmark_option, output_idx);
}

void gen_kernel_cv(
        std::vector<const KernelGen::KernelFunc*>& kernels, TContext* ctx,
        KernelGen::Arch arch, const std::string& kernel_symbol) {
    int usable_kern_cnt = 0;
    for (auto kernel : kernels) {
        if (kernel->IsCVAvailable(ctx)) {
            usable_kern_cnt++;
            TargetModule& g_module = TargetModule::get_global_target_module();
            auto kern_sym = kernel->GetCVKernelSymbol(ctx);
            auto if_match = std::regex_match(kern_sym, std::regex(kernel_symbol));
            if (!if_match)
                continue;
            if (!g_module.exist(kern_sym)) {
                g_module.add_cv(
                        kern_sym, kernel->GetCVKernelSignature(ctx),
                        kernel->GetCVKernelBody(ctx));
                auto depends = kernel->GetDependInternalSymbol(ctx);
                gen_depend_kernels(arch, depends);
            }
            return;
        }
    }
    mgb_assert(0, "gen cv kernel failed, available %d", usable_kern_cnt);
}

}  // namespace

//! CV
#if MEGCC_TEST_GEN
#define DEFAULT_RUN_INST_CV(_Opr)                                \
    {                                                            \
        gen_kernel_cv(kernels.first, &ctx, arch, kernel_symbol); \
        return {};                                               \
    }
#else
#define DEFAULT_RUN_INST_CV(_Opr)                                                      \
    {                                                                                  \
        RunCvHelper<_Opr> runner(opr);                                                 \
        return call_kernel_cv<_Opr>(                                                   \
                kernels.first, &ctx, tensors, benchmark_option, kernel_symbol, runner, \
                output_idx);                                                           \
    }
#endif
#define DEF_CCOPRPROXY_CV(_Opr)                                                        \
    template <>                                                                        \
    PerformanceResult CCOprProxy<_Opr>::exec(                                          \
            _Opr* opr, const TensorNDArray& tensors, KernelGen::Arch arch,             \
            const BenchmarkOption& benchmark_option, const std::string& kernel_symbol, \
            const std::unordered_map<std::string, CCAttr>& proxy_attr,                 \
            bool gen_dynamic) {                                                        \
        std::unordered_map<std::string, CCAttr> attr_map;                              \
        auto kernels = opr_fill_attr<_Opr>(attr_map, opr, tensors, arch, proxy_attr);  \
        auto output_idx = get_output_idx(opr);                                         \
        output_idx.normalize((int)tensors.size());                                     \
        fill_operands(attr_map, tensors, output_idx, false);                           \
        add_test_mode_to_attr(attr_map);                                               \
        CodeGenContext ctx(attr_map);                                                  \
        DEFAULT_RUN_INST_CV(_Opr);                                                     \
    }

DEF_CCOPRPROXY_CV(megdnn::CVtranspose);

DEF_CCOPRPROXY_CV(megdnn::CVflip);
DEF_CCOPRPROXY_CV(megdnn::CVResize);
DEF_CCOPRPROXY_CV(megdnn::CVRotate);
DEF_CCOPRPROXY_CV(megdnn::CVRoicopy);
DEF_CCOPRPROXY_CV(megdnn::CVCvtColor);
DEF_CCOPRPROXY_CV(megdnn::CVWarpAffine);
DEF_CCOPRPROXY_CV(megdnn::CVGaussianBlur);

#undef DEF_CCOPRPROXY_CV
}  // namespace test
}  // namespace megcc