#include "./KernelTemplate.h"
#include "KernelRegister.h"

#include "compiler/Common/Logger.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/KernelGen/JitExe.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;

namespace mlir {
namespace Kernel {
class RawCodeKernelTemplate::KernelJITImpl final : public RawCodeKernelJIT {
public:
    KernelJITImpl(Registry* reg) : registry(reg) {}

private:
    size_t getWorkspace(Operation* op, TContext* ctx) const override {
        if (!op)
            return 0;

        auto kernelDef = dyn_cast<RawCodeKernelDef>(op);
        if (!kernelDef)
            return 0;

        auto&& kernelFuncMap = registry->symbol2kernelFunc;
        auto&& iter = kernelFuncMap->find(kernelDef.sym_name().str());
        if (iter == kernelFuncMap->end())
            return 0;

        auto* kernelFunc = iter->second;
        if (!kernelFunc)
            return 0;
        return megcc::KernelGen::JitExec::jit_exec_and_get_workspace(kernelFunc, ctx);
    }

    Registry* registry;
};

std::unique_ptr<RawCodeKernelJIT> RawCodeKernelJIT::make(
        KernelTemplateRegistry* registry) {
    return std::make_unique<RawCodeKernelTemplate::KernelJITImpl>(registry);
}

std::vector<RawCodeKernelTemplate*> KernelTemplateRegistry::getCandidates(
        Operation* op) {
    std::vector<RawCodeKernelTemplate*> candidates;
    for (auto&& i : kernelTemplates) {
        if (i->match(op)) {
            candidates.push_back(i.get());
        }
    }
    return candidates;
}

Operation* RawCodeKernelTemplate::instantiate(
        OpBuilder& builder, TContext* context, bool is_internal_func) {
    if (kernelFunc->IsAvailable(context)) {
        instantiateInternalKernel(builder, context);
        auto symbolName = kernelFunc->GetKernelSymbol(context);
        Operation* kernel = owner->getKernelDef(symbolName);
        if (!kernel) {
            auto signature = kernelFunc->GetKernelSignature(context);
            auto body = kernelFunc->GetKernelBody(context);
            auto initSymbolName = kernelFunc->GetInitSymbol(context);
            auto initSignature = kernelFunc->GetInitSignature(context);
            auto initBody = kernelFunc->GetInitBody(context);
            auto guard_begin = kernelFunc->GetBodyGuardBegin(context);
            auto guard_end = kernelFunc->GetBodyGuardEnd(context);
            std::string deduce_symbol = "";
            std::string deduce_sig = "";
            std::string deduce_body = "";
            if (deduceFunc) {
                deduce_symbol = deduceFunc->GetDeduceSymbol(context);
                deduce_sig = deduceFunc->GetDeduceSig(context);
                deduce_body = deduceFunc->GetDeduceBody(context);
            }
            kernel = builder.create<RawCodeKernelDef>(
                    builder.getUnknownLoc(), symbolName, signature, body,
                    initSymbolName, initSignature, initBody, guard_begin, guard_end,
                    deduce_symbol, deduce_sig, deduce_body, is_internal_func);
            owner->symbol2kernelDef->operator[](symbolName) = kernel;
            owner->symbol2kernelFunc->operator[](symbolName) = kernelFunc;
            if (!is_internal_func) {
                LOG_DEBUG << "Instantiates Kernel with symbol: "
                          << dyn_cast<RawCodeKernelDef>(kernel).sym_name().str()
                          << "\n";
            } else {
                LOG_DEBUG << "Instantiates Internal Kernel with symbol: "
                          << dyn_cast<RawCodeKernelDef>(kernel).sym_name().str()
                          << "\n";
            }
        }
        return kernel;
    }
    return nullptr;
}

void RawCodeKernelTemplate::instantiateInternalKernel(
        OpBuilder& builder, megcc::TContext* context) {
    auto depends = kernelFunc->GetDependInternalSymbol(context);
    std::function<void(
            const std::vector<megcc::KernelGen::KernelObj>&,
            std::map<std::string, Operation*>&, OpBuilder&)>
            init_dep_kern;

    init_dep_kern = [&init_dep_kern](
                            const std::vector<megcc::KernelGen::KernelObj>& depends_in,
                            std::map<std::string, Operation*>& symbol2kernelDef,
                            OpBuilder& builder) {
        for (auto& kernel_obj : depends_in) {
            auto symbolName = kernel_obj.kernel_symbol;
            if (symbol2kernelDef.count(symbolName) <= 0) {
                auto kernel = builder.create<RawCodeKernelDef>(
                        builder.getUnknownLoc(), symbolName, "", kernel_obj.kernel_body,
                        "", "", "", kernel_obj.guard_begin, kernel_obj.guard_end, "",
                        "", "", true);
                symbol2kernelDef[symbolName] = kernel;
                if (kernel_obj.kernel_dep.size() > 0) {
                    init_dep_kern(kernel_obj.kernel_dep, symbol2kernelDef, builder);
                }
            }
        }
    };
    init_dep_kern(depends, *owner->symbol2kernelDef, builder);
}

void addBuiltinTemplates(
        KernelTemplateRegistry& registry, KernelGen::Arch target_arch) {
    addBuiltinTemplatesByOperator(registry, target_arch);
    if (target_arch != megcc::KernelGen::BAREMETAL) {
        addBuiltinTemplatesByOperator(registry, megcc::KernelGen::BAREMETAL);
    }
}

}  // namespace Kernel
}  // namespace mlir

// vim: syntax=cpp.doxygen
