#pragma once
#include <string>
#include "../WinogradCommon.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class WinogradF23Strategy8x8MK8 : public WinogradStrategyBase {
public:
    uint32_t GetKernelSize() override { return 3; }
    uint32_t GetOutputBlockSize() override { return 2; }
    std::string DependMatmulSymbol() override;
    std::string WeightTrans(const std::vector<std::string>& strs) override;
    std::string InputFeatureTrans(const std::vector<std::string>& strs) override;
    std::string OutputFeatureTrans(
            const std::vector<std::string>& strs, TContext*) override;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
