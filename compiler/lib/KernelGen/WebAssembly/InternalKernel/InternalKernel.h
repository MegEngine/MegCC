#pragma once
#include <string>
#include "WebAssembly/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace WebAssembly { 

class WebAssemblyMatmulInternal : public InternalKernelFunc {
public:
    static const std::string m_packa_workspace_call;
    static const std::string m_packb_workspace_call;
    static const std::string m_workspace_call;
    virtual std::string GetKernelSignature(TContext* ctx) const override {
        return "void " + GetKernelSymbol(ctx) + GenKernelCall(ctx);
    }
    virtual std::string GetPackASymbol(TContext* ctx) const {
        bool trans_a = ctx->getAttrBool("transposeA");
        std::string suffix = trans_a ? "t" : "n";
        return GetKernelSymbol(ctx) + "_packa_" + suffix;
    }
    virtual std::string GetPackBSymbol(TContext* ctx) const {
        bool trans_b = ctx->getAttrBool("transposeB");
        std::string suffix = trans_b ? "t" : "n";
        return GetKernelSymbol(ctx) + "_packb_" + suffix;
    }
    virtual std::string GetNakedKernelSymbol(TContext* ctx) const {
        return GetKernelSymbol(ctx) + "_naked";
    }
    virtual std::string GetPackAWorkspaceSymbol(TContext* ctx) const {
        return GetKernelSymbol(ctx) + "_workspace_a";
    }
    virtual std::string GetPackBWorkspaceSymbol(TContext* ctx) const {
        return GetKernelSymbol(ctx) + "_workspace_b";
    }

    virtual std::string GetPackAWorkspaceBody(TContext*) const { return ""; }
    virtual std::string GetPackBWorkspaceBody(TContext*) const { return ""; }

    virtual std::string GetPackASignature(TContext* ctx) const {
        return "void " + GetPackASymbol(ctx) + GenPackACall(ctx);
    }
    virtual std::string GetPackBSignature(TContext* ctx) const {
        return "void " + GetPackBSymbol(ctx) + GenPackBCall(ctx);
    }
    virtual std::string GetNakedKernelSignature(TContext* ctx) const {
        return "void " + GetNakedKernelSymbol(ctx) + GenNakedKernelCall(ctx);
    }
    virtual std::string GetPackAWorkspaceSignature(TContext* ctx) const {
        return "size_t " + GetPackAWorkspaceSymbol(ctx) + m_packa_workspace_call;
    }
    virtual std::string GetPackBWorkspaceSignature(TContext* ctx) const {
        return "size_t " + GetPackBWorkspaceSymbol(ctx) + m_packb_workspace_call;
    }
    virtual bool need_post_process(TContext*) const { return false; }
    static std::string GenKernelCall(TContext* ctx);
    static std::string GenNakedKernelCall(TContext* ctx);
    static std::string GenPackACall(TContext* ctx);
    static std::string GenPackBCall(TContext* ctx);
};


class MatmulM4N12Kernel : public WebAssemblyMatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;
 
    std::string GetPackAWorkspaceBody(TContext*) const override;

    std::string GetPackBWorkspaceBody(TContext*) const override;
};


}  // namespace WebAssembly
}  // namespace KernelGen
}  // namespace megcc