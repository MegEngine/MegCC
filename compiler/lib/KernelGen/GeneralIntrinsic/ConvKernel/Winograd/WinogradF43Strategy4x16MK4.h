/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Winograd/WinogradF43Strategy4x16MK4.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "WinogradCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class WinogradF43Strategy4x16MK4 : public WinogradStrategyBase {
public:
    uint32_t GetKernelSize() override { return 3; }
    uint32_t GetOutputBlockSize() override { return 4; }
    uint32_t GetTileSize() override { return 68; };
    std::string DependMatmulSymbol() override;
    std::string WeightTrans(const std::vector<std::string>& strs) override;
    std::string InputFeatureTrans(
            const std::vector<std::string>& strs) override;
    std::string BatchedMatMul(const std::vector<std::string>& strs) override;
    std::string OutputFeatureTrans(const std::vector<std::string>& strs,
                                   TContext*) override;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
