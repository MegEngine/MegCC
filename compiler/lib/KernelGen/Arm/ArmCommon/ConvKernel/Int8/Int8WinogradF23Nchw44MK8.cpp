#include <memory>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/ArmCommon/ConvKernel.h"
#include "Arm/ArmCommon/InternalMatMul/InternalMatMul.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool WinogradFloatF23Nchw44MK8Int8::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == 3 && ctx->getAttrUInt("kernel_w") == 3 &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_h") == 1 && ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1;

    bool param_mode_ok = ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";

    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU";

    bool type_ok = is_qint8_conv_dtype(ctx);

    //! because of MK8 matmul
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ((ctx->getAttrStr("sparse") == "DENSE" &&
                       ctx->getAttrOprand("operand:1").shape.size() == 6 &&
                       ctx->getAttrOprand("operand:1").shape[0] % 2 == 0 &&
                       ctx->getAttrOprand("operand:1").shape[1] % 2 == 0) ||
                      (ctx->getAttrStr("sparse") == "GROUP" &&
                       ctx->getAttrOprand("operand:1").shape.size() == 7 &&
                       ctx->getAttrOprand("operand:1").shape[1] % 2 == 0 &&
                       ctx->getAttrOprand("operand:1").shape[2] % 2 == 0));

    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}

std::string WinogradFloatF23Nchw44MK8Int8::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include<arm_neon.h>\n";
    writer << "#include<math.h>\n";
    writer << "#include \"unroll_macro.h\"\n";
    writer << "\n\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx) << "{\n";
    writer << m_framework.GenInitCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFloatF23Nchw44MK8Int8::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream writer;
    writer << GenCommonRet() << " " << GetWorkspaceSignature(ctx) << "{\n";
    writer << m_framework.GenGetWorkSpaceCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFloatF23Nchw44MK8Int8::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include\"marm_neon.h\"\n";
    writer << "#include \"unroll_macro.h\"\n";
    writer << "\n\n";
    writer << "extern " << Int16M8N8K8MatMulKernel().GetKernelSignature(ctx) << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    writer << m_framework.GenKernelBodyCode(ctx, &m_winograd_strategy);
    writer << "return TinyNN_SUCCESS;\n}";
    return writer.str();
}

std::vector<KernelObj> WinogradFloatF23Nchw44MK8Int8::GetDependInternalSymbol(
        TContext*) const {
    auto matmul = Int16M8N8K8MatMulKernel();
    return {
            {matmul.GetKernelSymbol(nullptr), matmul.GetKernelBody(nullptr),
             matmul.GetBodyGuardBegin(nullptr), matmul.GetBodyGuardEnd(nullptr),
             matmul.GetDependInternalSymbol(nullptr)}};
}

std::string WinogradFloatF23Nchw44MK8Int8::GetKernelSymbol(TContext* context) const {
    auto symbol = ArmCommonConvImpl::GetKernelSymbol(context);
    return symbol + "_winograd_f23_int8_nchw44_mk8";
}
// vim: syntax=cpp.doxygen
