#include "WebAssembly/InternalKernel/InternalKernel.h"
#include "Utils/Utils.h"
using namespace megcc;
using namespace KernelGen;
using namespace WebAssembly;
const std::string WebAssemblyMatmulInternal::m_packa_workspace_call =
        "(int y0, int ymax, int k0, int kmax)";
const std::string WebAssemblyMatmulInternal::m_packb_workspace_call =
        "(int x0, int xmax, int k0, int kmax)";
const std::string WebAssemblyMatmulInternal::m_workspace_call =
        "(int y0, int ymax, int x0, int xmax, int k0, int kmax)";

std::string WebAssemblyMatmulInternal::GenNakedKernelCall(TContext* ctx) {
    auto dtype = ctx ? ctx->getAttrStr("dtype") : "f32";
    if (Utils::is_float_dtype(dtype)) {
        return R"((const float* pack_a, const float* pack_b, float* C,
            size_t LDC, size_t M, size_t N, size_t K, const float* bias_ptr))";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}

std::string WebAssemblyMatmulInternal::GenKernelCall(TContext* ctx) {
    auto dtype = ctx ? ctx->getAttrStr("dtype") : "f32";
    if (Utils::is_float_dtype(dtype)) {
        return R"((const float* A, size_t LDA, const float* B, size_t LDB, float* C,
            size_t LDC, size_t M, size_t N, size_t K, const float* bias_ptr, void* workspace))";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}

std::string WebAssemblyMatmulInternal::GenPackACall(TContext* ctx) {
    auto dtype = ctx ? ctx->getAttrStr("dtype") : "f32";
    if (Utils::is_float_dtype(dtype)) {
        return "(float* outptr, const float* inptr, int ldin, int y0, int "
               "ymax, int k0, int kmax)";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}

std::string WebAssemblyMatmulInternal::GenPackBCall(TContext* ctx) {
    auto dtype = ctx ? ctx->getAttrStr("dtype") : "f32";
    if (Utils::is_float_dtype(dtype)) {
        return "(float* outptr, const float* inptr, int ldin, int x0, int "
               "xmax, int k0, int kmax)";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}
