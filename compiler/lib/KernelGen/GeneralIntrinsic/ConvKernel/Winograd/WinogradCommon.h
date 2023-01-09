/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Winograd/WinogradCommon.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class WinogradStrategyBase {
public:
    virtual uint32_t GetKernelSize() = 0;
    virtual uint32_t GetOutputBlockSize() = 0;

    //! transform the weight to winograd space, input strings are:
    //! 0: inptr, the start pointer of the convolution weight
    //! 1: outptr, the transformed weight store memory pointer
    //! 2: OC, the output channel multiply pack_c_size
    //! 3: IC, the input channel multiply pack_c_size
    virtual std::string WeightTrans(const std::vector<std::string>& strs) = 0;

    //! transform the input feature map to the winograd, with input strings:
    //! 0: inptr, input pointer offset to group
    //! 1: transform_input_ptr, the transformed storage
    //! 2: IH
    //! 3: IW
    //! 4: IC
    //! 5: PH
    //! 6: PW, the param of the transform
    //! 7: tile_id, the start tile id
    //! 8: nr_tiles_in_loop, the number of tile in the loop
    virtual std::string InputFeatureTrans(
            const std::vector<std::string>& strs) = 0;

    //! the batched matmul conduct on the transformed input and weight, with
    //! input strings:
    //! 0:A_ptr, 1:LDA, 2:B_ptr, 3:LDB, 4:C_ptr, 5:LDC, 6:OC,
    //! 7:IC, 8:nr_tiles_in_loop
    virtual std::string BatchedMatMul(const std::vector<std::string>& strs) = 0;

    //! output transform and data post process, input strings:
    //! 0: transform_output_ptr, to be transform memory
    //! 1: outptr, the dst memory
    //! 2: bias_ptr, bias ptr, is no bias is nullptr
    //! 3: OH,
    //! 4: OW,
    //! 5: OC,
    //! 6: tile_id, the start tile id
    //! 7: nr_tiles_in_loop, the number of tile in the loop
    virtual std::string OutputFeatureTrans(const std::vector<std::string>& strs,
                                           TContext*) = 0;

    virtual std::string DependMatmulSymbol() = 0;
};

class WinogradFrameNchw44 {
    uint32_t m_tile_per_loop = 32;

public:
    //! gen init code
    std::string GenInitCode(TContext*, WinogradStrategyBase*);

    //! gen body code without signature
    std::string GenKernelBodyCode(TContext*, WinogradStrategyBase*);

    //! gen get workspace code
    std::string GenGetWorkSpaceCode(TContext*, WinogradStrategyBase*);
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
