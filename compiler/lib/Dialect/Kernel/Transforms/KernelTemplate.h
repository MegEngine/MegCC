/**
 * \file compiler/lib/Dialect/Kernel/Transforms/KernelTemplate.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "compiler/Common/TContext.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/KernelGen/KernelGen.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace Kernel {
class RawCodeKernelTemplate {
public:
    using KernelFn = megcc::KernelGen::KernelFunc;
    using DeduceFn = megcc::KernelGen::DeduceFunc;
    using MatchFn = std::function<bool(Operation*)>;
    using PrepareFn = std::function<void(Operation*, megcc::TContext*)>;

    class KernelJITImpl;

    struct Registry {
        Registry() {
            symbol2kernelDef = std::make_shared<std::map<std::string, Operation*>>();
            symbol2kernelFunc =
                    std::make_shared<std::map<std::string, const KernelFn*>>();
            symbol2deduceFunc =
                    std::make_shared<std::map<std::string, const DeduceFn*>>();
            symbol2InternalKernel =
                    std::make_shared<std::map<std::string, std::string>>();
        }
        Registry(
                std::shared_ptr<std::map<std::string, Operation*>> symbol2kernelDefIn,
                std::shared_ptr<std::map<std::string, const KernelFn*>>
                        symbol2kernelFuncIn,
                std::shared_ptr<std::map<std::string, std::string>>
                        symbol2InternalKernelIn) {
            symbol2kernelDef = symbol2kernelDefIn;
            symbol2kernelFunc = symbol2kernelFuncIn;
            symbol2InternalKernel = symbol2InternalKernelIn;
        }
        template <typename OpTy>
        Registry& create(
                const KernelFn* kernelFn, const DeduceFn* deduceFn,
                PrepareFn prepareFn = {}) {
            return create(
                    kernelFn, deduceFn,
                    [&](Operation* op) { return op && llvm::isa<OpTy>(op); },
                    prepareFn);
        }

        Registry& create(
                const KernelFn* kernelFn, const DeduceFn* deduceFn, MatchFn matchFn,
                PrepareFn prepareFn = {}) {
            kernelTemplates.emplace_back(new RawCodeKernelTemplate(
                    this, kernelFn, matchFn, prepareFn, deduceFn));
            return *this;
        }

        std::vector<RawCodeKernelTemplate*> getCandidates(Operation*);

        Operation* getKernelDef(std::string symbol) {
            auto&& iter = symbol2kernelDef->find(symbol);
            if (iter != symbol2kernelDef->end()) {
                return iter->second;
            }
            return nullptr;
        }
        std::shared_ptr<std::map<std::string, Operation*>> symbol2kernelDef;
        std::shared_ptr<std::map<std::string, const KernelFn*>> symbol2kernelFunc;
        std::shared_ptr<std::map<std::string, const DeduceFn*>> symbol2deduceFunc;
        std::shared_ptr<std::map<std::string, std::string>> symbol2InternalKernel;

    private:
        std::vector<std::unique_ptr<RawCodeKernelTemplate>> kernelTemplates;
        friend class RawCodeKernelTemplate;
    };

private:
    RawCodeKernelTemplate(
            Registry* registry, const KernelFn* kernelFn, MatchFn matchFn,
            PrepareFn prepareFn, const DeduceFn* deduceFn)
            : owner(registry),
              kernelFunc(kernelFn),
              matchFunc(matchFn),
              prepareFunc(prepareFn),
              deduceFunc(deduceFn) {}

    Registry* owner;
    const KernelFn* kernelFunc;
    MatchFn matchFunc;
    PrepareFn prepareFunc;
    const DeduceFn* deduceFunc;

    // return null for instantiate failure
    void instantiateInternalKernel(OpBuilder& builder, megcc::TContext* context);

public:
    // return null for instantiate failure
    Operation* instantiate(
            OpBuilder& builder, megcc::TContext* context,
            bool is_internal_func = false);

    // perform early type checking on given operation
    bool match(Operation* op) { return matchFunc(op); }

    // user hook for adding extra template arguments into TContext
    void prepare(Operation* op, megcc::TContext* ctx) {
        if (prepareFunc) {
            prepareFunc(op, ctx);
        }
    }
};

using KernelTemplateRegistry = RawCodeKernelTemplate::Registry;

class RawCodeKernelJIT {
public:
    virtual ~RawCodeKernelJIT() = default;
    virtual size_t getWorkspace(Operation* op, megcc::TContext* ctx) const = 0;
    static std::unique_ptr<RawCodeKernelJIT> make(KernelTemplateRegistry* registry);

protected:
    RawCodeKernelJIT() = default;
};

void addBuiltinTemplates(
        KernelTemplateRegistry& registry, megcc::KernelGen::Arch target_arch);

}  // namespace Kernel
}  // namespace mlir

// vim: syntax=cpp.doxygen
