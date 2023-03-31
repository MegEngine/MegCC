/**
 * \file
 * compiler/test/kernel/common/src/workload_proxy.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "test/kernel/common/workload_proxy.h"
#include <cmath>
#include <cstddef>

using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;
static inline size_t get_conv_compute_workload(
        megdnn::TensorLayout src_layout, megdnn::TensorLayout filter_layout,
        megdnn::TensorLayout dst_layout, megdnn::ConvBias::Param::Format format,
        megdnn::ConvBias::Param::Sparse sparse) {
    using Param = megdnn::ConvBiasForward::Param;
    float fh = 0;
    float fw = 0;
    float icpg = 0;
    if (format == Param::Format::NCHW) {
        if (sparse == Param::Sparse::DENSE) {
            icpg = filter_layout[1];
            fh = filter_layout[2];
            fw = filter_layout[2];
        } else {
            mgb_assert(sparse == Param::Sparse::GROUP);
            icpg = filter_layout[2];
            fh = filter_layout[3];
            fw = filter_layout[4];
        }
    } else if (format == Param::Format::NCHW88) {
        if (sparse == Param::Sparse::DENSE) {
            if (src_layout.ndim == 5) {
                icpg = filter_layout[1] * 8;
                fh = filter_layout[2];
                fw = filter_layout[3];
            } else {
                mgb_assert(src_layout.ndim == 4);
                icpg = filter_layout[3];
                fh = filter_layout[1];
                fw = filter_layout[2];
            }
        } else {
            mgb_assert(sparse == Param::Sparse::GROUP);
            icpg = filter_layout[2];
            fh = filter_layout[3];
            fw = filter_layout[4];
        }
    } else {
        mgb_assert(
                format == Param::Format::NCHW44 || format == Param::Format::NCHW44_DOT);
        if (sparse == Param::Sparse::DENSE) {
            if (src_layout.ndim == 5) {
                icpg = filter_layout[1] * 4;
                fh = filter_layout[2];
                fw = filter_layout[3];
            } else {
                mgb_assert(src_layout.ndim == 4);
                icpg = filter_layout[3];
                fh = filter_layout[1];
                fw = filter_layout[2];
            }
        } else {
            mgb_assert(sparse == Param::Sparse::GROUP);
            icpg = filter_layout[2];
            fh = filter_layout[3];
            fw = filter_layout[4];
        }
    }

    return dst_layout.total_nr_elems() * icpg * fh * fw * 2;
}
namespace megcc {
namespace test {
template <>
size_t WorkloadOprProxy<megdnn::ConvBiasForward>::get_compute_workload(
        megdnn::ConvBiasForward* opr, const TensorNDArray& tensors) {
    auto param = opr->param();
    auto src_layout = tensors[0].layout;
    auto filter_layout = tensors[1].layout;
    auto dst_layout = tensors[4].layout;
    return get_conv_compute_workload(
            src_layout, filter_layout, dst_layout, param.format, param.sparse);
}
template <>
size_t WorkloadOprProxy<megdnn::ConvolutionForward>::get_compute_workload(
        megdnn::ConvolutionForward* opr, const TensorNDArray& tensors) {
    auto param = opr->param();
    auto src_layout = tensors[0].layout;
    auto filter_layout = tensors[1].layout;
    auto dst_layout = tensors[2].layout;
    return get_conv_compute_workload(
            src_layout, filter_layout, dst_layout, param.format, param.sparse);
}
template <>
size_t WorkloadOprProxy<megdnn::PoolingForward>::get_compute_workload(
        megdnn::PoolingForward* opr, const TensorNDArray& tensors) {
    auto param = opr->param();
    auto dst_layout = tensors[1].layout;
    float computation = dst_layout.total_nr_elems() * param.window_h * param.window_w;
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::MatrixMulForward>::get_compute_workload(
        megdnn::MatrixMulForward* opr, const TensorNDArray& tensors) {
    using Param = megdnn::MatrixMulForward::Param;
    auto param = opr->param();
    auto transposeA = param.transposeA;
    auto A_layout = tensors[0].layout;
    auto C_layout = tensors[2].layout;
    size_t m = A_layout[0], n = C_layout[1], k = A_layout[1];
    if (param.format == Param::Format::MK4) {
        m = A_layout[0] * 4;
        k = A_layout[1] * 4;
    }
    if (transposeA == true) {
        m = A_layout[1];
        k = A_layout[0];
    }
    return m * k * n * 2;
}

template <>
size_t WorkloadOprProxy<megdnn::TypeCvtForward>::get_compute_workload(
        megdnn::TypeCvtForward* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[1].layout;
    float computation = dst_layout.total_nr_elems();
    return computation;
}
template <>
size_t WorkloadOprProxy<megdnn::RelayoutForward>::get_compute_workload(
        megdnn::RelayoutForward* opr, const TensorNDArray& tensors) {
    auto src_layout = tensors[0].layout;
    float computation = src_layout.total_nr_elems();
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::Resize>::get_compute_workload(
        megdnn::Resize* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[1].layout;
    float computation = dst_layout.total_nr_elems() * 12;
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::CvtColor>::get_compute_workload(
        megdnn::CvtColor* opr, const TensorNDArray& tensors) {
    auto src_layout = tensors[0].layout;
    float computation = src_layout.total_nr_elems();
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::WarpAffine>::get_compute_workload(
        megdnn::WarpAffine* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[2].layout;
    float computation = dst_layout.total_nr_elems() * 12;
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::Flip>::get_compute_workload(
        megdnn::Flip* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[1].layout;
    float computation = dst_layout.total_nr_elems();
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::GaussianBlur>::get_compute_workload(
        megdnn::GaussianBlur* opr, const TensorNDArray& tensors) {
    auto param = opr->param();
    auto dst_layout = tensors[1].layout;
    float computation =
            dst_layout.total_nr_elems() * param.kernel_height * param.kernel_width;
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::Rotate>::get_compute_workload(
        megdnn::Rotate* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[1].layout;
    float computation = dst_layout.total_nr_elems();
    return computation;
}
template <>
size_t WorkloadOprProxy<megdnn::ElemwiseForward>::get_compute_workload(
        megdnn::ElemwiseForward* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[tensors.size() - 1].layout;
    float computation = dst_layout.total_nr_elems();
    return computation;
}

template <>
size_t WorkloadOprProxy<megdnn::ReduceForward>::get_compute_workload(
        megdnn::ReduceForward* opr, const TensorNDArray& tensors) {
    auto dst_layout = tensors[tensors.size() - 1].layout;
    auto src_layout = tensors[0].layout;
    float computation = dst_layout.total_nr_elems() + src_layout.total_nr_elems();
    return computation;
}
template <>
size_t WorkloadOprProxy<megdnn::ConvolutionBackwardData>::get_compute_workload(
        megdnn::ConvolutionBackwardData* opr, const TensorNDArray& tensors) {
    auto param = opr->param();
    auto src_layout = tensors[0].layout;
    auto filter_layout = tensors[1].layout;
    auto dst_layout = tensors[2].layout;
    return get_conv_compute_workload(
            src_layout, filter_layout, dst_layout, param.format, param.sparse);
}
template <>
size_t WorkloadOprProxy<megdnn::TopK>::get_compute_workload(
        megdnn::TopK* opr, const TensorNDArray& tensors) {
    auto src_layout = tensors[0].layout;
    auto dst_layout = tensors[1].layout;
    float computation = src_layout[0] * src_layout[1] * std::log2f(dst_layout[1]);
    return computation;
}
}  // namespace test
}  // namespace megcc
