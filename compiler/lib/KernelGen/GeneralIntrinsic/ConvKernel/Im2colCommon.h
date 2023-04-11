#pragma once
#include <string>
#include "Common/ConvKernel.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {
class Im2colStrategyBase {
public:
    virtual ~Im2colStrategyBase() = default;
    //! padding the src into im2col needed:
    //! 0: inptr, the start pointer of the convolution src
    //! 1: outptr, the transformed src store memory pointer
    //! 2: icpg, the output channel per group
    //! 3: ih, the input height
    //! 4: iw, the input width
    //! 5: pad_h, the input height pad size
    //! 6: pad_h, the input width pad size
    std::string PaddingSrc(TContext* ctx);

    //! transform the input feature map to the Im2col needed:
    //! 0: inptr, input pointer
    //! 1: transform_input_ptr, the transformed storage
    //! 2: OW
    //! 3: icpg
    //! 4: IH
    //! 5: IW
    //! 6: FH
    //! 7: FW
    //! 8: ohw_idx, the number of out block index
    //! 9: real_block_ohw out block size
    std::string Im2col(TContext* ctx);

    std::shared_ptr<TContext> cvt2matmul_ctx(TContext* ctx);

    // matmul call
    virtual KernelGen::InternalKernelFunc* GetInnerCtxMatmul(TContext* ctx) = 0;

    virtual std::string GetInnerCtxMatmulSym(TContext* ctx) = 0;

    virtual std::string PackBSym(TContext* ctx) { return "pack_B"; };

    virtual std::string PackASym(TContext* ctx) { return "pack_A"; };

    virtual std::string GetPackAWorkspaceSym(TContext* ctx) {
        return "pack_A_workspace";
    }

    virtual std::string GetPackBWorkspaceSym(TContext* ctx) {
        return "pack_B_workspace";
    };

    virtual std::string GetPackASignature(TContext* ctx);

    virtual std::string GetPackAWorkspaceSignature(TContext* ctx);

    virtual std::string GetPackBSignature(TContext* ctx);

    virtual std::string GetPackBWorkspaceSignature(TContext* ctx);

    virtual std::string GetPackBWorkspaceBody(TContext* ctx);
};

class Im2colFrameNchwxx {
public:
    //! gen init code
    std::string GenInitCode(TContext*, Im2colStrategyBase*);

    //! gen body code without signature
    std::string GenKernelBodyCode(TContext*, Im2colStrategyBase*);

    //! gen get workspace code
    std::string GenGetWorkSpaceCode(TContext*, Im2colStrategyBase*);
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
