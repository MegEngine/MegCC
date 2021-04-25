/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/InternalKernel/InternalKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "Arm/ArmCommon/InternalKernel.h"
#include "Arm/Armv7/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace Armv7 {

class MatmulM4N12Kernel : public Armv7MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;

    std::string GetPackBWorkspaceBody(TContext*) const override;
};

class MatmulM4N12MK4Kernel : public Armv7MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;

    std::string GetPackBWorkspaceBody(TContext*) const override;
};

class MatmulM4N8MK4Kernel : public Armv7InternalKernelFunc {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelSignature(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
