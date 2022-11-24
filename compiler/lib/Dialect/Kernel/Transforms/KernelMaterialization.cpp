/**
 * \file compiler/lib/Dialect/Kernel/Transforms/KernelMaterialization.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>
#include <unordered_map>
#include "./KernelTemplate.h"
#include "compiler/Common/Logger.h"
#include "compiler/Common/MlirUtils.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace megcc;

llvm::cl::opt<megcc::KernelGen::Arch> target_arch(
        llvm::cl::desc("The target architecture to execute the model"),
        llvm::cl::values(clEnumValN(megcc::KernelGen::BAREMETAL, "baremetal",
                                    "compiler for device baremetal."),
                         clEnumValN(megcc::KernelGen::ARM64, "arm64",
                                    "compiler for device arm64."),
                         clEnumValN(megcc::KernelGen::ARMV7, "armv7",
                                    "compiler for device armv7."),
                         clEnumValN(megcc::KernelGen::ARM64V7, "arm64v7",
                                    "compiler for device arm64v7.")),
        llvm::cl::init(megcc::KernelGen::BAREMETAL));

namespace mlir {
namespace {
#define GEN_PASS_CLASSES
#include "compiler/Dialect/Kernel/Transforms/Passes.h.inc"

LogicalResult fetchInputsAndOutputs(Operation* op, SmallVector<Value>& inputs,
                                    SmallVector<Value>& outputs) {
    auto memoryEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffect) {
        return failure();
    }
    for (auto&& operand : op->getOperands()) {
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        memoryEffect.getEffectsOnValue(operand, effects);
        if (llvm::any_of(
                    effects, [](const MemoryEffects::EffectInstance& instance) {
                        return isa<MemoryEffects::Read>(instance.getEffect());
                    })) {
            inputs.push_back(operand);
        }
        if (llvm::any_of(
                    effects, [](const MemoryEffects::EffectInstance& instance) {
                        return isa<MemoryEffects::Write>(instance.getEffect());
                    })) {
            outputs.push_back(operand);
        }
    }
    return success();
}

std::unordered_map<std::string, CCAttr> convertAttrToKernelAttr(
        mlir::DictionaryAttr dict_attrs) {
    std::unordered_map<std::string, CCAttr> attr_map;
    for (const auto& attr : dict_attrs) {
        auto attribute = attr.getValue();
        if (auto value = attribute.dyn_cast_or_null<BoolAttr>()) {
            attr_map[attr.getName().str()] = value.getValue();
        } else if (attribute.dyn_cast_or_null<IntegerAttr>()) {
            auto value = attribute.dyn_cast<IntegerAttr>().getValue();
            auto type = attribute.dyn_cast<IntegerAttr>().getType();
            if (type.isSignedInteger() && type.getIntOrFloatBitWidth() <= 32) {
                attr_map[attr.getName().str()] =
                        static_cast<int32_t>(value.getSExtValue());
            } else if ((type.isSignlessInteger() || type.isUnsignedInteger()) &&
                       type.getIntOrFloatBitWidth() <= 32) {
                attr_map[attr.getName().str()] =
                        static_cast<uint32_t>(value.getZExtValue());
            } else if (type.isSignlessIntOrIndex() ||
                       type.isUnsignedInteger()) {
                attr_map[attr.getName().str()] =
                        static_cast<uint64_t>(value.getZExtValue());
            } else {
                attr_map[attr.getName().str()] =
                        static_cast<int64_t>(value.getSExtValue());
            }
        } else if (attribute.dyn_cast_or_null<StringAttr>()) {
            auto value = attribute.dyn_cast<StringAttr>();
            attr_map[attr.getName().str()] = value.getValue().str();
        } else if (attribute.dyn_cast_or_null<FloatAttr>()) {
            auto value = attribute.dyn_cast<FloatAttr>();
            auto type = attribute.dyn_cast<FloatAttr>().getType();
            if (type.isF32()) {
                attr_map[attr.getName().str()] =
                        static_cast<float>(value.getValueAsDouble());
            } else {
                attr_map[attr.getName().str()] = value.getValueAsDouble();
            }
        } else if (attribute.dyn_cast_or_null<ArrayAttr>()) {
            auto attr_list = attribute.dyn_cast<ArrayAttr>();
            int cnt = 0;
            for (auto attr_iter : attr_list) {
                if (attr_iter.dyn_cast_or_null<IntegerAttr>()) {
                    auto value = attr_iter.dyn_cast<IntegerAttr>().getValue();
                    auto type = attr_iter.dyn_cast<IntegerAttr>().getType();
                    CC_ASSERT(type.getIntOrFloatBitWidth() <= 32)
                            << "is sign " << type.isSignedInteger()
                            << ", bitwise " << type.getIntOrFloatBitWidth()
                            << "\n";
                    if (type.isSignedInteger()) {
                        attr_map[attr.getName().str() + ":" +
                                 std::to_string(cnt++)] =
                                (int32_t)(value.getSExtValue());
                    } else {
                        attr_map[attr.getName().str() + ":" +
                                 std::to_string(cnt++)] =
                                (int32_t)(value.getZExtValue());
                    }
                } else if (attr_iter.dyn_cast<StringAttr>()) {
                    auto value =
                            attr_iter.dyn_cast<StringAttr>().getValue().str();
                    attr_map[attr.getName().str() + ":" +
                             std::to_string(cnt++)] = value;
                }
            }
            attr_map[attr.getName().str() + ":size"] = cnt;
        }
    }
    return attr_map;
}

std::unordered_map<std::string, CCAttr> getKernelAttr(Operation* op) {
    auto attrs = convertAttrToKernelAttr(op->getAttrDictionary());
    auto nr_operands = op->getNumOperands();
    //! nr_operands attribute is of type uint32_t
    attrs["nr_operands"] = static_cast<uint32_t>(nr_operands);
    //! add "operand:0" to dtype string attribute
    for (uint32_t i = 0; i < nr_operands; i++) {
        if (auto shapedType = op->getOperands()[i]
                                      .getType()
                                      .dyn_cast_or_null<ShapedType>()) {
            CCOperand cc_operand;
            llvm::raw_string_ostream raw_os(cc_operand.dtype);
            auto dtype = shapedType.getElementType();
            dtype.print(raw_os);
            cc_operand.shape = {shapedType.getShape().begin(),
                                shapedType.getShape().end()};
            if (dtype.isa<IntegerType>() &&
                dtype.dyn_cast<IntegerType>().isQuant()) {
                cc_operand.scale = dtype.dyn_cast<IntegerType>().getScale();
            }
            attrs[llvm::formatv("operand:{0}", i)] = cc_operand;
        }
    }
    return attrs;
}

class KernelMaterializationPass final
        : public KernelMaterializationPassBase<KernelMaterializationPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        auto op = getOperation();
        populateKernelMaterializationPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
            signalPassFailure();
        op->walk([this](Operation* op) {
            if (op->hasTrait<OpTrait::Kernel::AbstractKernelTrait>()) {
                llvm::errs() << "abstract kernel " << *op
                             << " hasn't been materialized\n";
                signalPassFailure();
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
    }
};

class DupFuncPattern final : public OpRewritePattern<FuncOp> {
public:
    DupFuncPattern(MLIRContext* ctx) : OpRewritePattern(ctx) {}
    LogicalResult matchAndRewrite(FuncOp op,
                                  PatternRewriter& rewriter) const override {
        auto op_name = op.getName();
        auto is_modi = op_name.contains(KernelGen::ARM64V7_COMMON_POSTFIX);
        if (llvm::isa<FuncOp>(op) && !is_modi) {
            auto clone_op = op.clone();
            clone_op.setName(op.getName().str() +
                             KernelGen::ARM64V7_ARMV7_POSTFIX);
            op.setName(op.getName().str() + KernelGen::ARM64V7_ARM64_POSTFIX);
            rewriter.insert(clone_op);
            return success();
        }
        return failure();
    }
};

class KernelMaterialization final
        : public OpTraitRewritePattern<OpTrait::Kernel::AbstractKernelTrait> {
    using KTRegistry = Kernel::KernelTemplateRegistry;
    using KernelJIT = Kernel::RawCodeKernelJIT;

public:
    KernelMaterialization(MLIRContext* ctx, std::unique_ptr<KTRegistry> reg,
                          std::string prefix = "")
            : OpTraitRewritePattern(ctx),
              registry(std::move(reg)),
              kernelJIT(KernelJIT::make(registry.get())),
              m_prefix(prefix) {}

    LogicalResult matchAndRewrite(Operation* op,
                                  PatternRewriter& rewriter) const override {
        SmallVector<Value> inputs;
        SmallVector<Value> outputs;
        if (failed(fetchInputsAndOutputs(op, inputs, outputs))) {
            return failure();
        }
        Operation* parent = SymbolTable::getNearestSymbolTable(op);
        if (!parent || !llvm::isa<ModuleOp>(parent)) {
            return failure();
        }
        auto func_name = mlir::dyn_cast<FuncOp>(op->getParentOp()).getName();
        if (m_prefix.size() > 0 && !func_name.endswith(m_prefix)) {
            return failure();
        }
        //! TODO: cache the kernel_attr for the same op
        auto kernel_attr = getKernelAttr(op);
        for (auto* kernelTemplate : registry->getCandidates(op)) {
            Operation* kernelDef = nullptr;
            int64_t workspaceInBytes = 0;
            {
                OpBuilder::InsertionGuard _(rewriter);
                rewriter.setInsertionPointToStart(
                        &parent->getRegion(0).front());
                CodeGenContext cgctx{kernel_attr};
                kernelTemplate->prepare(op, &cgctx);
                kernelDef = kernelTemplate->instantiate(rewriter, &cgctx);
                workspaceInBytes = kernelJIT->getWorkspace(kernelDef, &cgctx);
            }
            if (kernelDef) {
                bool is_dynamic = false;
                for (auto&& value : outputs) {
                    if (llvm::dyn_cast<Kernel::DynamicAlloc>(
                                value.getDefiningOp())) {
                        is_dynamic = true;
                        break;
                    }
                }
                Value workspace;
                if (is_dynamic) {
                    //! FIXME: add workspace body for dynamic opr, now default
                    //! zero
                    auto deduce_func_name =
                            kernelDef->getAttrOfType<StringAttr>(
                                    "deduce_sym_name");
                    CC_ASSERT(deduce_func_name.getValue().size() > 0)
                            << op->getName().getStringRef().str()
                            << "need layout deduce func\n";

                } else if (workspaceInBytes) {
                    // alloc workspace for kernel execution, corresponding
                    // deallocation op would be inserted automatically in
                    // another pass(createBufferDeallocationPass) later
                    workspace = rewriter.create<memref::AllocOp>(
                            op->getLoc(),
                            MemRefType::get({workspaceInBytes},
                                            rewriter.getIntegerType(8)));
                }
                rewriter.replaceOpWithNewOp<Kernel::KernelCall>(
                        op, op->getResultTypes(),
                        llvm::dyn_cast<Kernel::RawCodeKernelDef>(kernelDef)
                                .sym_name(),
                        inputs, outputs, workspace, op->getAttrDictionary(),
                        is_dynamic);
                return success();
            }
        }
        return failure();
    }

private:
    std::unique_ptr<KTRegistry> registry;
    std::unique_ptr<KernelJIT> kernelJIT;
    std::string m_prefix;
};

}  // namespace

void populateKernelMaterializationPatterns(RewritePatternSet& patterns) {
    if (target_arch == megcc::KernelGen::ARM64V7) {
        auto a64_registry = std::make_unique<Kernel::KernelTemplateRegistry>();
        Kernel::addBuiltinTemplates(*a64_registry, megcc::KernelGen::ARM64);
        //! a32_registry and a64_registry shared the same map to avoid
        //! generating redundant armcommon kernel
        auto a32_registry = std::make_unique<Kernel::KernelTemplateRegistry>(
                a64_registry->symbol2kernelDef, a64_registry->symbol2kernelFunc,
                a64_registry->symbol2InternalKernel);
        Kernel::addBuiltinTemplates(*a32_registry, megcc::KernelGen::ARMV7);
        //! copy function, origin function rename to
        //! old_name+KernelGen::ARM64V7_ARM64_POSTFIX, new function rename to
        //! old_name+KernelGen::ARM64V7_ARMV7_POSTFIX
        patterns.add(std::make_unique<DupFuncPattern>(patterns.getContext()));
        //! materializ function with specific kernel for different function
        //! postfix
        patterns.add(std::make_unique<KernelMaterialization>(
                patterns.getContext(), std::move(a32_registry),
                KernelGen::ARM64V7_ARMV7_POSTFIX));
        patterns.add(std::make_unique<KernelMaterialization>(
                patterns.getContext(), std::move(a64_registry),
                KernelGen::ARM64V7_ARM64_POSTFIX));
    } else {
        auto registry = std::make_unique<Kernel::KernelTemplateRegistry>();
        Kernel::addBuiltinTemplates(*registry, target_arch);
        patterns.add(std::make_unique<KernelMaterialization>(
                patterns.getContext(), std::move(registry)));
    }
}

std::unique_ptr<OperationPass<ModuleOp>> createKernelMaterializationPass() {
    return std::make_unique<KernelMaterializationPass>();
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
