/**
 * \file
 * compiler/test/kernel/common/src/cc_fill_attr.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "test/kernel/common/src/cc_fill_attr.h"
#include "megbrain/common.h"
#include "megbrain/reflection.h"
#include "megcc_test_config.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"
#include "test/kernel/common/cv_opr.h"

namespace {

void get_kernel_size(uint32_t& kh, uint32_t& kw, megdnn::TensorND weight,
                     ConvParam::Sparse sparse, ConvParam::Format format) {
    if (format == ConvParam::Format::NCHW) {
        if (sparse == ConvParam::Sparse::DENSE) {
            kh = weight.layout[2];
            kw = weight.layout[3];
        } else {
            mgb_assert(sparse == ConvParam::Sparse::GROUP);
            kh = weight.layout[3];
            kw = weight.layout[4];
        }
    } else if (format == ConvParam::Format::NCHW44 ||
               format == ConvParam::Format::NCHW44_DOT) {
        if (sparse == ConvParam::Sparse::DENSE) {
            //! dense layout is oc/4, ic/4, fh, fw, 4, 4
            if (weight.layout.ndim == 6) {
                kh = weight.layout[2];
                kw = weight.layout[3];
            } else {
                //! hybrid first layout oc/4, fh, fw, ic, 4
                kh = weight.layout[1];
                kw = weight.layout[2];
            }
        }
        if (sparse == ConvParam::Sparse::GROUP) {
            //! channel wise weight layout is g/4, 1, 1, fh, fw, 4
            if (weight.layout.ndim == 6) {
                kh = weight.layout[3];
                kw = weight.layout[4];
            } else {
                //! group conv with weight layout is g, ocpg/4, icpg/4, fh, fw,
                //! 4, 4
                kh = weight.layout[3];
                kw = weight.layout[4];
            }
        }
    } else {
        mgb_assert(0, "get_kernel_size not support format %d", (int)format);
    }
}

#define DEFINE_DNNPARAM2STR(cls)                             \
    std::string dnnparam_2_str(cls value) {                  \
        return mgb::reflection::nameOfEnumValue<cls>(value); \
    }

DEFINE_DNNPARAM2STR(ConvParam::Format)
DEFINE_DNNPARAM2STR(ConvParam::Sparse)
DEFINE_DNNPARAM2STR(ConvParam::Mode)
DEFINE_DNNPARAM2STR(ConvBiasParam::NonlineMode)
DEFINE_DNNPARAM2STR(megdnn::ElemwiseForward::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::ElemwiseMultiType::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::PoolingForward::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::MatrixMulForward::Param::Format)
DEFINE_DNNPARAM2STR(megdnn::MatrixMulForward::Param::ComputeMode)
DEFINE_DNNPARAM2STR(megdnn::Reduce::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::Reduce::Param::DataType)
DEFINE_DNNPARAM2STR(megdnn::WarpPerspectiveForward::Param::BorderMode)
DEFINE_DNNPARAM2STR(megdnn::WarpPerspectiveForward::Param::InterpolationMode)
DEFINE_DNNPARAM2STR(megdnn::CvtColor::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::Argsort::Param::Order)
DEFINE_DNNPARAM2STR(megdnn::TopK::Param::Mode)
#undef DEFINE_DNNPARAM2STR
}  // namespace

namespace megcc {
namespace test {
#define FILL_MAP(_map_name, _parm_name, _attr_name) \
    _map_name[#_attr_name] = CCAttr(_parm_name._attr_name)
#define FILL_MAP_EX(_map_name, _parm_name, _attr_name, _helper_fun) \
    _map_name[#_attr_name] = CCAttr(_helper_fun(_parm_name._attr_name))
using KernType = KernelGen::KernelPack::KernType;
template <>
KernelGenRet opr_fill_attr<megdnn::ElemwiseForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ElemwiseForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    attr_map["mode"] = CCAttr(dnnparam_2_str(param.mode));
    return KernelGen::KernelPack::GetKernel(KernType::ElemwiseKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::Argsort>(
        std::unordered_map<std::string, CCAttr>& attr_map, megdnn::Argsort* opr,
        const TensorNDArray& tensors, KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    attr_map["order"] = CCAttr(dnnparam_2_str(param.order));
    return KernelGen::KernelPack::GetKernel(KernType::ArgSortKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::TopK>(
        std::unordered_map<std::string, CCAttr>& attr_map, megdnn::TopK* opr,
        const TensorNDArray& tensors, KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    attr_map["mode"] = CCAttr(dnnparam_2_str(param.mode));
    attr_map["k"] = proxy_attr.at("k");
    return KernelGen::KernelPack::GetKernel(KernType::TopK, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ElemwiseMultiType>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ElemwiseMultiType* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    attr_map["mode"] = CCAttr(dnnparam_2_str(param.mode));
    return KernelGen::KernelPack::GetKernel(KernType::ElemwiseMultiKernel,
                                            arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::RelayoutForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::RelayoutForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    return KernelGen::KernelPack::GetKernel(KernType::RelayoutKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::IndexingMultiAxisVec>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::IndexingMultiAxisVec* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto axis = proxy_attr.at("axis").AsOperand().shape;
    for (size_t i = 0; i < axis.size(); ++i) {
        attr_map["axis:" + std::to_string(i)] = CCAttr((int)axis[i]);
    }
    return KernelGen::KernelPack::GetKernel(KernType::IndexingMultiAxisKernel,
                                            arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::IndexingOneHot>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::IndexingOneHot* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch, const std::unordered_map<std::string, CCAttr>&) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, axis);
    return KernelGen::KernelPack::GetKernel(KernType::IndexingOneHotKernel,
                                            arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::MatrixInverse>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::MatrixInverse* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    return KernelGen::KernelPack::GetKernel(KernType::MatrixInvKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVtranspose>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::CVtranspose* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    return KernelGen::KernelPack::GetKernel(KernType::CVTransposeKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVflip>(
        std::unordered_map<std::string, CCAttr>& attr_map, megdnn::CVflip* opr,
        const TensorNDArray& tensors, KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    return KernelGen::KernelPack::GetKernel(KernType::FlipKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVCvtColor>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::CVCvtColor* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::CvtColorKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVWarpAffine>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::CVWarpAffine* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, border_val);
    FILL_MAP_EX(attr_map, param, border_mode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);

    return KernelGen::KernelPack::GetKernel(KernType::WarpAffineKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ConvolutionForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ConvolutionForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    uint32_t kh = 0, kw = 0;
    get_kernel_size(kh, kw, tensors[1], param.sparse, param.format);
    attr_map["kernel_h"] = CCAttr(kh);
    attr_map["kernel_w"] = CCAttr(kw);
    FILL_MAP(attr_map, param, stride_h);
    FILL_MAP(attr_map, param, stride_w);
    FILL_MAP(attr_map, param, pad_h);
    FILL_MAP(attr_map, param, pad_w);
    FILL_MAP(attr_map, param, dilate_h);
    FILL_MAP(attr_map, param, dilate_w);
    FILL_MAP_EX(attr_map, param, sparse, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ConvKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ConvBiasForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ConvBiasForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    uint32_t kh = 0, kw = 0;
    get_kernel_size(kh, kw, tensors[1], param.sparse, param.format);
    attr_map["kernel_h"] = CCAttr(kh);
    attr_map["kernel_w"] = CCAttr(kw);
    FILL_MAP(attr_map, param, stride_h);
    FILL_MAP(attr_map, param, stride_w);
    FILL_MAP(attr_map, param, pad_h);
    FILL_MAP(attr_map, param, pad_w);
    FILL_MAP(attr_map, param, dilate_h);
    FILL_MAP(attr_map, param, dilate_w);
    FILL_MAP_EX(attr_map, param, sparse, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, nonlineMode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ConvKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::PoolingForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::PoolingForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, stride_h);
    FILL_MAP(attr_map, param, stride_w);
    FILL_MAP(attr_map, param, pad_h);
    FILL_MAP(attr_map, param, pad_w);
    FILL_MAP(attr_map, param, window_h);
    FILL_MAP(attr_map, param, window_w);

    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::PoolingKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ConcatForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ConcatForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, axis);
    return KernelGen::KernelPack::GetKernel(KernType::ConcatKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::MatrixMulForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::MatrixMulForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, transposeA);
    FILL_MAP(attr_map, param, transposeB);

    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, compute_mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::MatrixMulKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::BatchedMatrixMulForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::BatchedMatrixMulForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, transposeA);
    FILL_MAP(attr_map, param, transposeB);

    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, compute_mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::BatchMatmulKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::WarpPerspectiveForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::WarpPerspectiveForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, border_val);
    FILL_MAP_EX(attr_map, param, bmode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);

    return KernelGen::KernelPack::GetKernel(KernType::WarpPerspectiveKernel,
                                            arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::WarpAffineForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::WarpAffineForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, border_val);
    FILL_MAP_EX(attr_map, param, border_mode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);

    return KernelGen::KernelPack::GetKernel(KernType::WarpAffineKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ReduceForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ReduceForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, axis);

    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, data_type, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ReduceKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVResize>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::CVResize* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();

    FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ResizeKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ResizeForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ResizeForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();

    FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ResizeKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVRotate>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::CVRotate* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();

    FILL_MAP(attr_map, param, clockwise);
    return KernelGen::KernelPack::GetKernel(KernType::RotateKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::CVRoicopy>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::CVRoicopy* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, row_from);
    FILL_MAP(attr_map, param, row_to);
    FILL_MAP(attr_map, param, col_from);
    FILL_MAP(attr_map, param, col_to);
    return KernelGen::KernelPack::GetKernel(KernType::RoiCopyKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::PowC>(
        std::unordered_map<std::string, CCAttr>& attr_map, megdnn::PowC* opr,
        const TensorNDArray& tensors, KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, exp);
    return KernelGen::KernelPack::GetKernel(KernType::PowCKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::TypeCvtForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::TypeCvtForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    return KernelGen::KernelPack::GetKernel(KernType::TypeCvtKernel, arch);
}

template <>
KernelGenRet opr_fill_attr<megdnn::ArgmaxForward>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ArgmaxForward* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    FILL_MAP(attr_map, param, axis);
    return KernelGen::KernelPack::GetKernel(KernType::ArgmaxKernel, arch);
}
template <>
KernelGenRet opr_fill_attr<megdnn::ConvolutionBackwardData>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ConvolutionBackwardData* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    uint32_t kh = 0, kw = 0;
    get_kernel_size(kh, kw, tensors[0], param.sparse, param.format);
    attr_map["kernel_h"] = CCAttr(kh);
    attr_map["kernel_w"] = CCAttr(kw);
    FILL_MAP(attr_map, param, stride_h);
    FILL_MAP(attr_map, param, stride_w);
    FILL_MAP(attr_map, param, pad_h);
    FILL_MAP(attr_map, param, pad_w);
    FILL_MAP(attr_map, param, dilate_h);
    FILL_MAP(attr_map, param, dilate_w);
    FILL_MAP_EX(attr_map, param, sparse, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ConvBackDataKernel, arch);
}

}  // namespace test
}  // namespace megcc