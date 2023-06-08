#pragma once
#include <string>
#include "Arm/ArmCommon/ConvKernel/Fp32/Winograd/WinogradCommon.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class WinogradFrameNchw44Int8 {
    uint32_t m_tile_per_loop = 24;

public:
    //! gen init code
    std::string GenInitCode(TContext*, WinogradStrategyBase*);

    //! gen body code without signature
    std::string GenKernelBodyCode(TContext*, WinogradStrategyBase*);

    //! gen get workspace code
    std::string GenGetWorkSpaceCode(TContext*, WinogradStrategyBase*);
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
