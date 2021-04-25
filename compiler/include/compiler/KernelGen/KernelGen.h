/**
 * \file compiler/include/compiler/KernelGen.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include <sstream>
#include <string>
#include <vector>
#include "compiler/Common/TContext.h"

namespace megcc {
namespace KernelGen {
namespace {
//! used for ARM64V7 in dump stage
const std::string ARM64V7_COMMON_POSTFIX = "#x#";
const std::string ARM64V7_ARM64_POSTFIX = ARM64V7_COMMON_POSTFIX + "_arm64";
const std::string ARM64V7_ARMV7_POSTFIX = ARM64V7_COMMON_POSTFIX + "_armv7";
}  // namespace

//! Flag the platform
enum Arch {
    BAREMETAL = 0,
    ARM64 = 1,
    ARMV7 = 2,
    //! special arch for both dump arm64 and armv7 kern, only used in dump stage
    ARM64V7 = 3,
    //! Auto arch code is generate by mlir or tvm generator
    AUTO_BAREMETAL = 4,
    AUTO_ARM64 = 5
};

//! Flag the priority for Kernel selection
enum KernelPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
};

std::string GenCommonInclude();
std::string GenCommonCall();
std::string GenCommonRet();
std::string GenCommonInitCall();
std::string GenCommonWorkspaceCall();
std::string GenCommonDeduceCall();

struct KernelObj {
    KernelObj() = default;

    KernelObj(const std::string& kernel_symbol_in,
              const std::string& kernel_body_in,
              const std::string& guard_begin_in,
              const std::string& guard_end_in)
            : kernel_symbol(kernel_symbol_in),
              kernel_body(kernel_body_in),
              guard_begin(guard_begin_in),
              guard_end(guard_end_in){};

    KernelObj(const std::string& kernel_symbol_in,
              const std::string& kernel_body_in,
              const std::string& guard_begin_in,
              const std::string& guard_end_in,
              const std::vector<KernelObj>& kernel_dep_in)
            : kernel_symbol(kernel_symbol_in),
              kernel_body(kernel_body_in),
              guard_begin(guard_begin_in),
              guard_end(guard_end_in),
              kernel_dep(kernel_dep_in){};
    std::string kernel_symbol;
    std::string kernel_body;
    std::string guard_begin;
    std::string guard_end;
    std::vector<KernelObj> kernel_dep;
    std::vector<uint8_t> kernel_bin;
};

struct KernelFunc {
    virtual ~KernelFunc(){};
    virtual bool IsAvailable(TContext* context) const = 0;
    virtual KernelPriority GetPriority() const {
        return KernelPriority::NORMAL;
    }
    //! kernel gen
    virtual std::string GetKernelSymbol(TContext* context) const = 0;
    virtual std::string GetKernelSignature(TContext* context) const {
        return GetKernelSymbol(context) + GenCommonCall();
    };
    virtual std::string GetKernelBody(TContext* context) const = 0;
    //! cv gen
    virtual bool IsCVAvailable(TContext* context) const { return false; };
    virtual std::string GetCVKernelSymbol(TContext* context) const {
        return "";
    };
    virtual std::string GetCVKernelSignature(TContext* context) const {
        return "";
    };
    virtual std::string GetCVKernelBody(TContext* context) const { return ""; };

    //! init gen
    virtual std::string GetInitSymbol(TContext* context) const {
        return GetKernelSymbol(context) + "_init";
    };
    virtual std::string GetInitSignature(TContext* context) const {
        return GetInitSymbol(context) + GenCommonInitCall();
    };
    virtual std::string GetInitBody(TContext* context) const {
        std::stringstream ss;
        ss << GenCommonRet() << " " << GetInitSignature(context) << R"({
                if (nr_out_weight){
                    *nr_out_weight = 0;
                }
               return TinyNN_SUCCESS;
            })";
        return ss.str();
    };

    //! workspace gen
    virtual std::string GetWorkspaceSymbol(TContext* context) const {
        return GetKernelSymbol(context) + "_workspace";
    };
    virtual std::string GetWorkspaceSignature(TContext* context) const {
        return GetWorkspaceSymbol(context) + GenCommonWorkspaceCall();
    };
    virtual std::string GetWorkspaceBody(TContext* context) const {
        std::stringstream ss;
        ss << GenCommonRet() << " " << GetWorkspaceSignature(context) << R"({
               return TinyNN_SUCCESS;
            })";
        return ss.str();
    };
    //! if get workspace need Jit execute, it should not depend on extern
    //! function
    virtual std::string GetWorkspaceBodyAndJitExec(TContext* context) const {
        return GetWorkspaceBody(context);
    };
    //! All body will be warp by guard begin, guard end
    virtual std::string GetBodyGuardBegin(TContext* context) const {
        return "";
    }

    virtual std::string GetBodyGuardEnd(TContext* context) const { return ""; }

    //! The internal kernel used by the kernel function
    virtual std::vector<KernelObj> GetDependInternalSymbol(TContext*) const {
        return {};
    }
};
//! this func used to get output shape from input shape. No alloc and deduce
//! dtype here
struct DeduceFunc {
    virtual ~DeduceFunc(){};
    virtual std::string GetDeduceSymbol(TContext* context) const = 0;
    virtual std::string GetDeduceBody(TContext* context) const = 0;
    virtual std::string GetDeduceSig(TContext* context) const {
        return GetDeduceSymbol(context) + GenCommonDeduceCall();
    }
};

//! The internal function kernel used by other KernelFuncs
struct InternalKernelFunc : public KernelFunc {
    bool IsAvailable(TContext*) const override { return true; }
    std::string GetInitSymbol(TContext*) const override { return ""; };
    std::string GetInitSignature(TContext*) const override { return ""; };
    std::string GetInitBody(TContext*) const override { return ""; };
    //! workspace gen
    virtual std::string GetWorkspaceSymbol(TContext*) const override {
        return "";
    };
    virtual std::string GetWorkspaceSignature(TContext*) const override {
        return "";
    };
    virtual std::string GetWorkspaceBody(TContext*) const override {
        return "";
    };
};

struct KernelPack {
    enum class KernType {
        Unknow = 0,
        ConvKernel,
        ElemwiseKernel,
        ElemwiseMultiKernel,
        PoolingKernel,
        MatrixMulKernel,
        MatrixInvKernel,
        RelayoutKernel,
        ReduceKernel,
        IndexingMultiAxisKernel,
        IndexingOneHotKernel,
        WarpPerspectiveKernel,
        WarpAffineKernel,
        TypeCvtKernel,
        TopK,
        BatchMatmulKernel,
        PowCKernel,
        CVTransposeKernel,
        FlipKernel,
        ResizeKernel,
        RotateKernel,
        RoiCopyKernel,
        CvtColorKernel,
        ArgSortKernel,
        ArgmaxKernel,
        ConcatKernel,
        InternelKernel,
        ConvBackDataKernel,
    };
    static std::pair<std::vector<const KernelFunc*>, const DeduceFunc*>
    GetKernel(KernelPack::KernType kernel_type, Arch arch);
};

}  // namespace KernelGen
}  // namespace megcc
// vim: syntax=cpp.doxygen
