/**
 * \file compiler/lib/Target/TinyNN/exporter.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "compiler/Common/Logger.h"
#include "compiler/Common/MlirUtils.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/KernelGen/KernelGen.h"
#include "compiler/Target/TinyNN/export.h"
#include "schema/model_generated.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include <unordered_map>

extern llvm::cl::opt<megcc::KernelGen::Arch> target_arch;

using namespace flatbuffers;

namespace mlir {
namespace {
class Exporter {
public:
    Exporter(ModuleOp top_module) : m_root(top_module) {}

    void save_model(std::string model_path, KernelExporter& kernel_exporter,
                    const bool save_model) {
        symbol2weight_id.clear();

        std::vector<Offset<MegCC::Weight>> weights;
        std::vector<std::string> model_meta_info;
        std::vector<Offset<MegCC::DeviceModel>> device_models;
        for (auto&& _ : *m_root.getBody()) {
            llvm::TypeSwitch<Operation*>(&_)
                    .Case([&](Kernel::WeightStorage op) {
                        weights.push_back(
                                attr_to_weight(op.value(), op.sym_name(), op.user_count()));
                        symbol2weight_id[op.sym_name().str()] =
                                weights.size() - 1;
                    })
                    .Case([&](Kernel::RawCodeKernelDef op) {
                        if (!op.internal_call()) {
                            kernel_exporter.addKernel(
                                    op.sym_name(), op.signature(), op.body(),
                                    op.guard_begin(), op.guard_end());
                            LOG_DEBUG << "Gen Kernel name: "
                                      << op.sym_name().str() << " id: "
                                      << kernel_exporter.get_kernel_id(
                                                 op.sym_name().str())
                                      << "\n";
                            kernel_exporter.addInitFunc(
                                    op.init_sym_name(), op.init_signature(),
                                    op.init_body(), op.sym_name(),
                                    op.guard_begin(), op.guard_end());
                            LOG_DEBUG << "Gen Init Kernel name: "
                                      << op.init_sym_name().str() << " id: "
                                      << kernel_exporter.get_init_id(
                                                 op.sym_name().str())
                                      << "\n";
                            if (op.deduce_sym_name().size() > 0) {
                                kernel_exporter.addDeduceShapeKernel(
                                        op.deduce_sym_name(), op.deduce_sig(),
                                        op.deduce_body(), op.sym_name());
                                LOG_DEBUG << "Gen Deduce Kernel name: "
                                          << op.init_sym_name().str() << " id: "
                                          << kernel_exporter.get_init_id(
                                                     op.sym_name().str())
                                          << "\n";
                            }
                        } else {
                            kernel_exporter.addInternalKernel(
                                    op.sym_name(), op.signature(), op.body(),
                                    op.guard_begin(), op.guard_end());
                            LOG_DEBUG << "Gen Internal Kernel name: "
                                      << op.sym_name().str();
                        }
                    })
                    .Case([&](FuncOp op) {
                        device_models.push_back(export_single_func(
                                op, kernel_exporter, model_meta_info));
                    })
                    .Default([&](Operation* op) {
                        llvm::errs() << "Unknown operation : " << *op << "\n";
                        abort();
                    });
        }

        auto weights_ = m_fbs_builder.CreateVector(weights);
        auto device_models_ = m_fbs_builder.CreateVector(device_models);

        MegCC::ModelBuilder model(m_fbs_builder);
        model.add_device_models(device_models_);
        model.add_weight_pool(weights_);
        m_fbs_builder.Finish(model.Finish());

        LOG_DEBUG << "TinyNN model information"
                  << "\n\tweights number: " << weights.size()
                  << "\n\tmodel path: " << model_path << "\n";

        std::error_code EC;
        llvm::raw_fd_stream model_file(model_path, EC);
        LOG_DEBUG << "open tiny model_file " << EC.message()
                  << ", size = " << m_fbs_builder.GetSize() << "\n";
        model_file.write(static_cast<char*>(static_cast<void*>(
                                 m_fbs_builder.GetBufferPointer())),
                         m_fbs_builder.GetSize());
        llvm::raw_fd_stream model_file_meta(model_path + ".txt", EC);
        model_file_meta << "[";
        for (size_t i = 0; i < model_meta_info.size(); ++i) {
            model_file_meta << model_meta_info[i];
            if (i != model_meta_info.size() - 1) {
                model_file_meta << ",";
            }
        }
        model_file_meta << "]";

        if (save_model) {
            writeModelToCFile(model_path + ".c", m_fbs_builder);
        }
    }
    std::string log_model_info(Value value, std::string input_name) {
        auto memref = value.getType().dyn_cast<MemRefType>();
        auto dtype = memref.getElementType();
        auto shape = memref.getShape();
        std::string ret;
        llvm::raw_string_ostream ss(ret);
        ss << "[";
        size_t cnt = 0;
        for (auto x : shape) {
            ss << x;
            ++cnt;
            if (cnt != shape.size()) {
                ss << ",";
            }
        }
        ss << "],\"";
        dtype.print(ss);
        ss << "\"";
        ss << ",\"" << input_name << "\"";
        return ret;
    }
    Offset<MegCC::DeviceModel> export_single_func(
            mlir::FuncOp func, KernelExporter& kernel_exporter,
            std::vector<std::string>& model_meta_info) {
        std::unordered_map<void*, std::pair<MegCC::TensorType, int32_t>>
                value2typed_tensor;
        std::unordered_map<void*, std::string> value2name;
        std::vector<Offset<MegCC::Tensor>> tensors;
        std::vector<uint8_t> instructions_type;
        std::vector<Offset<void>> instructions;
        std::vector<int32_t> inputs;
        std::vector<int32_t> outputs;

        uint64_t tensor_memory = 0;

        auto createTensor = [&](Value value) {
            if (auto memref = value.getType().dyn_cast<MemRefType>()) {
                auto&& iter = value2name.find(value.getAsOpaquePointer());
                if (iter == value2name.end()) {
                    tensors.push_back(memref_to_tensor(memref));
                } else {
                    tensors.push_back(memref_to_tensor(memref, iter->second));
                }
            } else {
                CC_ABORT << "invalid type of mem plan\n";
            }
            value2typed_tensor.emplace(value.getAsOpaquePointer(),
                                       std::make_pair(MegCC::TensorType_TENSOR,
                                                      tensors.size() - 1));
            return tensors.size() - 1;
        };
        std::vector<std::string> model_info_vec;
        for (size_t i = 0; i < func.getNumArguments(); ++i) {
            auto value = func.getArgument(i);
            auto name =
                    func.getArgAttrOfType<StringAttr>(i, "mgb.func_arg_name")
                            .getValue()
                            .str();
            if (name == "kGlobalBuffer") {
                // global buffer used for static memory planning
                auto memref = value.getType().dyn_cast<MemRefType>();
                CC_ASSERT(memref.getElementTypeBitWidth() == 8);
                tensor_memory = memref.getNumElements();
            } else {
                if (!value2name.emplace(value.getAsOpaquePointer(), name)
                             .second) {
                    CC_ABORT << "duplicate input buffer\n";
                }
                LOG_DEBUG << "Get TinyNN model input name: " << name << "\n";
                model_info_vec.push_back(log_model_info(value, name));
                inputs.push_back(createTensor(value));
            }
        }
        {
            //! hack: ignore armv7 in arm64v7 mode
            if (!func.getName().endswith(
                        megcc::KernelGen::ARM64V7_ARMV7_POSTFIX)) {
                std::stringstream ss;
                ss << "[";
                for (size_t i = 0; i < model_info_vec.size(); ++i) {
                    ss << "[" << model_info_vec[i] << "]";
                    if (i != model_info_vec.size() - 1) {
                        ss << ",";
                    }
                }
                ss << "]";
                model_meta_info.push_back(ss.str());
            }
        }

        auto&& block = func.getBody().front();
        for (auto&& iter = block.rbegin(); iter != block.rend(); ++iter) {
            if (auto ret = llvm::dyn_cast<ReturnOp>(*iter)) {
                CC_ASSERT(ret->getNumOperands() == func.getNumResults());
                for (size_t i = 0; i < func.getNumResults(); ++i) {
                    auto value = ret->getOperand(i);
                    auto name = func.getResultAttrOfType<StringAttr>(
                                            i, "mgb.func_result_name")
                                        .getValue()
                                        .str();
                    if (!value2name.emplace(value.getAsOpaquePointer(), name)
                                 .second) {
                        CC_ABORT << "duplicate output buffer\n";
                    }
                    LOG_DEBUG << "Get TinyNN model output name: " << name
                              << "\n";
                }
                break;
            }
        }

        for (auto&& _ : block) {
            llvm::TypeSwitch<Operation*>(&_)
                    .Case([&](memref::AllocOp op) {
                        size_t allocated = createTensor(op->getResult(0));
                        instructions_type.push_back(
                                MegCC::Instruction_DevMemAlloc);
                        instructions.push_back(MegCC::CreateDevMemAlloc(
                                                       m_fbs_builder, allocated)
                                                       .Union());
                        LOG_DEBUG << "Add DevMemAlloc instruction.\n";
                    })
                    .Case([&](memref::DeallocOp op) {
                        auto&& tensor = value2typed_tensor.at(
                                op->getOperand(0).getAsOpaquePointer());
                        if (tensor.first != MegCC::TensorType_TENSOR) {
                            CC_ABORT << "apply Free on non-Tensor\n";
                        }
                        int32_t to_free = tensor.second;
                        instructions_type.push_back(
                                MegCC::Instruction_DevMemFree);
                        instructions.push_back(
                                MegCC::CreateDevMemFree(m_fbs_builder, to_free)
                                        .Union());
                        LOG_DEBUG << "Add DevMemFree instruction.\n";
                    })
                    .Case([&](Kernel::GetWeight op) {
                        value2typed_tensor.emplace(
                                op->getResult(0).getAsOpaquePointer(),
                                std::make_pair(
                                        MegCC::TensorType_WEIGHT,
                                        symbol2weight_id[op.name().str()]));
                    })
                    .Case([&](Kernel::ExternOpr op) {
                        kernel_exporter.addInst("EXTERN_OPR");

                        std::vector<int32_t> input_tensors, output_tensors;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                        }

                        for (auto&& i : op.results()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            output_tensors.push_back(tensor.second);
                        }

                        std::string name(op.name().data(), op.name().size());
                        std::string data(op.data().data(), op.data().size());
                        uint32_t data_len = data.size();

                        LOG_DEBUG << "Add ExternOpr instruction.\n";
                        instructions_type.push_back(
                                MegCC::Instruction_ExternOpr);
                        instructions.push_back(
                                MegCC::CreateExternOpr(
                                        m_fbs_builder,
                                        m_fbs_builder.CreateVector(
                                                input_tensors),
                                        m_fbs_builder.CreateString(name),
                                        m_fbs_builder.CreateString(data),
                                        data_len,
                                        m_fbs_builder.CreateVector(
                                                output_tensors))
                                        .Union());
                    })
                    .Case([&](Kernel::MemPlan op) {
                        createTensor(op->getResult(0));
                    })
                    .Case([&](Kernel::DynamicAlloc op) {
                        createTensor(op->getResult(0));
                    })
                    .Case([&](Kernel::KernelCall op) {
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        std::vector<int32_t> output_tensors;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        if (input_tensors.size() != input_types.size()) {
                            CC_ABORT << "operator with different size of "
                                        "tensor  and tensor type\n";
                        }
                        for (auto&& i : op.results()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            if (tensor.first != MegCC::TensorType_TENSOR) {
                                CC_ABORT << "operator output must be "
                                            "Tensor\n";
                            }
                            output_tensors.push_back(tensor.second);
                        }
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto output_tensors_ =
                                m_fbs_builder.CreateVector(output_tensors);
                        auto workspace_ = value_to_workspace(op.workspace(),
                                                             op.callee().str());
                        auto type_ =
                                m_fbs_builder.CreateString(op.callee().str());
                        MegCC::OprBuilder opr_builder(m_fbs_builder);
                        opr_builder.add_inputs(input_tensors_);
                        opr_builder.add_input_types(input_types_);
                        opr_builder.add_outputs(output_tensors_);
                        opr_builder.add_kernel_id(kernel_exporter.get_kernel_id(
                                op.callee().str()));
                        opr_builder.add_init_id(
                                kernel_exporter.get_init_id(op.callee().str()));
                        if (op.dynamic_shape()) {
                            opr_builder.add_deduce_id(
                                    kernel_exporter.get_deduce_id(
                                            op.callee().str()));
                        }
                        opr_builder.add_workspace(workspace_);
                        opr_builder.add_type(type_);

                        LOG_DEBUG << "Add Opr to Call Kernel: "
                                  << op.callee().str() << " inputs id is "
                                  << input_tensors << " inputs type is "
                                  << input_types << " output id is "
                                  << output_tensors << "\n";
                        instructions_type.push_back(MegCC::Instruction_Opr);
                        instructions.push_back(opr_builder.Finish().Union());
                    })
                    .Case([&](Kernel::Reshape op) {
                        kernel_exporter.addInst("MEMFORWARD");
                        auto typed_tensor = value2typed_tensor.at(
                                op->getOperand(0).getAsOpaquePointer());
                        if (typed_tensor.first != MegCC::TensorType_TENSOR) {
                            CC_ABORT << "reshape instruction cannot be "
                                        "applied on weight\n";
                        }
                        LOG_DEBUG << "Add MemForward instruction.\n";
                        instructions_type.push_back(
                                MegCC::Instruction_MemForward);
                        instructions.push_back(
                                MegCC::CreateMemForward(
                                        m_fbs_builder, typed_tensor.second,
                                        createTensor(op), 0,
                                        MegCC::MemForwardType_RESHAPE)
                                        .Union());
                    })
                    .Case([&](Kernel::Subtensor op) {
                        kernel_exporter.addInst("MEMFORWARD");
                        auto typed_tensor = value2typed_tensor.at(
                                op->getOperand(0).getAsOpaquePointer());
                        if (typed_tensor.first != MegCC::TensorType_TENSOR) {
                            CC_ABORT << "Subtensor instruction cannot be "
                                        "applied on weight\n";
                        }
                        auto input_memref =
                                op.input().getType().dyn_cast<MemRefType>();
                        auto memref = op.getType().dyn_cast<MemRefType>();
                        int64_t offset_in, offset_out;
                        llvm::SmallVector<int64_t> stride_in, stride_out;
                        if (failed(getStridesAndOffset(input_memref, stride_in,
                                                       offset_in))) {
                            CC_ABORT << "get offset failed\n";
                        }
                        if (failed(getStridesAndOffset(memref, stride_out,
                                                       offset_out))) {
                            CC_ABORT << "get offset failed\n";
                        }
                        LOG_DEBUG << "Add MemForward instruction.\n";
                        instructions_type.push_back(
                                MegCC::Instruction_MemForward);
                        instructions.push_back(
                                MegCC::CreateMemForward(
                                        m_fbs_builder, typed_tensor.second,
                                        createTensor(op),
                                        offset_out - offset_in,
                                        MegCC::MemForwardType_SUBTENSOR)
                                        .Union());
                    })
                    .Case([&](Kernel::SubtensorIns op) {
                        kernel_exporter.addInst("SUBTENSOR");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        int32_t output_tensor;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto&& tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        if (tensor.first != MegCC::TensorType_TENSOR) {
                            CC_ABORT << "operator output must be "
                                        "Tensor\n";
                        }
                        output_tensor = tensor.second;
                        auto descs = llvm::to_vector<4>(
                                op.descs().getAsRange<ArrayAttr>());
                        auto flags= llvm::to_vector<4>(
                                op.flags().getAsRange<ArrayAttr>());
                        std::vector<Offset<MegCC::IndexDesc>> descs_;
                        std::vector<Offset<MegCC::IndexDesc>> flags_;
                        for (size_t idx = 0; idx < descs.size(); idx++) {
                            descs_.push_back(indexdesc_to_fbs(descs[idx]));
                            flags_.push_back(indexdesc_to_fbs(flags[idx]));
                        }

                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto descs_fbs = m_fbs_builder.CreateVector(descs_);
                        auto flags_fbs = m_fbs_builder.CreateVector(flags_);

                        MegCC::SubTensorBuilder subtensor_builder(m_fbs_builder);
                        subtensor_builder.add_inputs(input_tensors_);
                        subtensor_builder.add_input_types(input_types_);
                        subtensor_builder.add_output(output_tensor);
                        subtensor_builder.add_descs(descs_fbs);
                        subtensor_builder.add_flags(flags_fbs);

                        LOG_DEBUG << "Add subtensor instruction.\n";
                        instructions_type.push_back(MegCC::Instruction_SubTensor);
                        instructions.push_back(subtensor_builder.Finish().Union());
                    })
                    .Case([&](Kernel::SetSubtensorIns op) {
                        kernel_exporter.addInst("SETSUBTENSOR");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        int32_t output_tensor;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto&& tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        if (tensor.first != MegCC::TensorType_TENSOR) {
                            CC_ABORT << "operator output must be "
                                        "Tensor\n";
                        }
                        output_tensor = tensor.second;
                        auto descs = llvm::to_vector<4>(
                                op.descs().getAsRange<ArrayAttr>());
                        auto flags = llvm::to_vector<4>(
                                op.flags().getAsRange<ArrayAttr>());
                        std::vector<Offset<MegCC::IndexDesc>> descs_;
                        std::vector<Offset<MegCC::IndexDesc>> flags_;
                        for (size_t idx = 0; idx < descs.size(); idx++) {
                            descs_.push_back(indexdesc_to_fbs(descs[idx]));
                            flags_.push_back(indexdesc_to_fbs(flags[idx]));
                        }

                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto descs_fbs = m_fbs_builder.CreateVector(descs_);
                        auto flags_fbs = m_fbs_builder.CreateVector(flags_);

                        MegCC::SetSubTensorBuilder subtensor_builder(
                                m_fbs_builder);
                        subtensor_builder.add_inputs(input_tensors_);
                        subtensor_builder.add_input_types(input_types_);
                        subtensor_builder.add_output(output_tensor);
                        subtensor_builder.add_descs(descs_fbs);
                        subtensor_builder.add_flags(flags_fbs);

                        LOG_DEBUG << "Add set_subtensor instruction : \n";
                        instructions_type.push_back(
                                MegCC::Instruction_SetSubTensor);
                        instructions.push_back(
                                subtensor_builder.Finish().Union());
                    })
                    .Case([&](Kernel::GetVarShapeIns op) {
                        auto typed_tensor = value2typed_tensor.at(
                                op->getOperand(0).getAsOpaquePointer());
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        instructions_type.push_back(MegCC::Instruction_ShapeOf);
                        instructions.push_back(
                                MegCC::CreateShapeOf(
                                        m_fbs_builder, typed_tensor.second,
                                        typed_tensor.first, out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::ConcatIns op) {
                        kernel_exporter.addInst("CONCAT");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        LOG_DEBUG << "Add Concat instruction.\n";
                        instructions_type.push_back(MegCC::Instruction_Concat);
                        instructions.push_back(
                                MegCC::CreateConcat(m_fbs_builder, op.axis(),
                                                    input_tensors_,
                                                    input_types_,
                                                    out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::BroadcastIns op) {
                        kernel_exporter.addInst("BROADCAST");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        LOG_DEBUG << "Add Broadcast instruction.\n";
                        instructions_type.push_back(MegCC::Instruction_BroadCast);
                        instructions.push_back(
                                MegCC::CreateBroadCast(
                                        m_fbs_builder, input_tensors_,
                                        input_types_, out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::ReshapeIns op) {
                        kernel_exporter.addInst("RESHAPE");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        LOG_DEBUG << "Add Reshape instruction with nr_input "
                                  << input_tensors.size() << "\n";
                        instructions_type.push_back(MegCC::Instruction_Reshape);
                        instructions.push_back(
                                MegCC::CreateReshape(
                                        m_fbs_builder, input_tensors_,
                                        input_types_, out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::IndexingMultiAxisVecIns op) {
                        kernel_exporter.addInst("INDEXING_MULTI_AXIS");
                        std::vector<int> axis_vec;
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        for (auto axis : op.axis()) {
                            CC_ASSERT(axis.getType().isInteger(32));
                            auto value =
                                    axis.dyn_cast<IntegerAttr>().getValue();
                            if (axis.getType().isSignedInteger()) {
                                axis_vec.push_back(value.getSExtValue());
                            } else {
                                axis_vec.push_back(value.getZExtValue());
                            }
                        }
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto input_axis_ = m_fbs_builder.CreateVector(axis_vec);
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        LOG_DEBUG << "Add Indexing instruction with nr_input "
                                  << input_tensors.size() << "\n";
                        instructions_type.push_back(
                                MegCC::Instruction_IndexingMultiAxis);
                        instructions.push_back(
                                MegCC::CreateIndexingMultiAxis(
                                        m_fbs_builder, input_axis_,
                                        input_tensors_, input_types_,
                                        out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::DimshuffleIns op) {
                        kernel_exporter.addInst("DIMSHUFFLE");
                        auto typed_tensor = value2typed_tensor.at(
                                op->getOperand(0).getAsOpaquePointer());
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        auto pattern = op.pattern();
                        auto member = llvm::to_vector<4>(
                                pattern.getAsRange<IntegerAttr>());
                        std::vector<int32_t> pattern_v;
                        for (size_t idx = 0; idx < member.size(); idx++) {
                            pattern_v.push_back(member[idx].getInt());
                        }
                        auto pattern_ = m_fbs_builder.CreateVector(pattern_v);
                        LOG_DEBUG << "Add Dimshuffle instruction.\n";
                        instructions_type.push_back(
                                MegCC::Instruction_Dimshuffle);
                        instructions.push_back(
                                MegCC::CreateDimshuffle(m_fbs_builder, pattern_,
                                                        typed_tensor.second,
                                                        typed_tensor.first,
                                                        out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::ArithmeticIns op) {
                        kernel_exporter.addInst("ARITHMETIC");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        LOG_DEBUG << "Add Arithmetic instruction.\n";
                        instructions_type.push_back(
                                MegCC::Instruction_Arithmetic);
                        instructions.push_back(
                                MegCC::CreateArithmetic(
                                        m_fbs_builder,
                                        convert_arithmetic_mode(op.mode()),
                                        input_tensors_, input_types_,
                                        out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::TypeCvtIns op) {
                        kernel_exporter.addInst("TYPECVT");
                        auto typed_tensor = value2typed_tensor.at(
                                op->getOperand(0).getAsOpaquePointer());
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());
                        auto idtype = op.i_dtype().str();
                        auto odtype = op.o_dtype().str();
                        LOG_DEBUG << "Add TypeCvt instruction.\n";
                        instructions_type.push_back(MegCC::Instruction_TypeCvt);
                        instructions.push_back(
                                MegCC::CreateTypeCvt(
                                        m_fbs_builder,
                                        m_fbs_builder.CreateString(idtype),
                                        m_fbs_builder.CreateString(odtype),
                                        typed_tensor.second, typed_tensor.first,
                                        out_tensor.second)
                                        .Union());
                    })
                    .Case([&](Kernel::WarpPerspectiveIns op) {
                        kernel_exporter.addInst("WARP_PERSPCETIVE");
                        std::vector<int32_t> input_tensors;
                        std::vector<int8_t> input_types;
                        for (auto&& i : op.operands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            input_tensors.push_back(tensor.second);
                            input_types.push_back(tensor.first);
                        }
                        auto input_tensors_ =
                                m_fbs_builder.CreateVector(input_tensors);
                        auto input_types_ =
                                m_fbs_builder.CreateVector(input_types);
                        auto&& out_tensor = value2typed_tensor.at(
                                op.result().getAsOpaquePointer());

                        auto mat_id= op.mat_idx();
                        auto member = llvm::to_vector<4>(
                                mat_id.getAsRange<IntegerAttr>());
                        std::vector<int32_t> mat_id_v;
                        for (size_t idx = 0; idx < member.size(); idx++) {
                            mat_id_v.push_back(member[idx].getInt());
                        }
                        auto mat_id_ = m_fbs_builder.CreateVector(mat_id_v);

                        LOG_DEBUG << "Add WarpPerspective instruction.\n";
                        instructions_type.push_back(
                                MegCC::Instruction_WarpPerspective);
                        instructions.push_back(
                                MegCC::CreateWarpPerspective(
                                        m_fbs_builder,
                                        convert_bordermodemode_mode(op.bmode()),
                                        convert_interpolation_mode(op.imode()),
                                        convert_format(op.format()),
                                        op.border_val().convertToFloat(),
                                        mat_id_, input_tensors_, input_types_,
                                        out_tensor.second)
                                        .Union());
                    })
                    .Case([&](ReturnOp op) {
                        for (auto&& i : op.getOperands()) {
                            auto&& tensor = value2typed_tensor.at(
                                    i.getAsOpaquePointer());
                            if (tensor.first != MegCC::TensorType_TENSOR) {
                                CC_ABORT << "network output must be "
                                            "Tensor\n";
                            }
                            outputs.push_back(tensor.second);
                        }
                    })
                    .Default([&](Operation* op) {
                        llvm::errs() << "Unknown operation : " << *op << "\n";
                        abort();
                    });
        }
        auto tensors_ = m_fbs_builder.CreateVector(tensors);
        auto instructions_type_ = m_fbs_builder.CreateVector(instructions_type);
        auto instructions_ = m_fbs_builder.CreateVector(instructions);
        auto inputs_ = m_fbs_builder.CreateVector(inputs);
        auto outputs_ = m_fbs_builder.CreateVector(outputs);

        MegCC::DeviceModelBuilder device_model(m_fbs_builder);
        device_model.add_tensor_pool(tensors_);
        device_model.add_instructions_type(instructions_type_);
        device_model.add_instructions(instructions_);
        device_model.add_inputs(inputs_);
        device_model.add_outputs(outputs_);
        device_model.add_tensor_memory(tensor_memory);
        auto func_name = func.getName();
        device_model.add_device(get_tiny_device(func_name));

        LOG_DEBUG << "TinyNN device model information"
                  << "\n\tinstructions number : " << instructions.size()
                  << "\n\ttensors number: " << tensors.size()
                  << "\n\tinputs number: " << inputs.size()
                  << "\n\toutputs number: " << outputs.size()
                  << "\n\tdevice: " << get_tiny_device(func_name)
                  << "\n\ttotal memory for model : " << tensor_memory << "\n";

        return device_model.Finish();
    }

private:
    static void writeModelToCFile(std::string save_path,
                                  const FlatBufferBuilder& model) {
        std::error_code EC;
        llvm::raw_fd_stream os(save_path, EC);

        size_t size = model.GetSize();
        uint8_t* file_vec = static_cast<uint8_t*>(model.GetBufferPointer());
        std::stringstream ss;
        ss << "unsigned char model_tiny[] = {";
        for (size_t i = 0; i < size; i++) {
            char buf[16];
            sprintf(buf, "0x%02X, ", file_vec[i]);
            ss << std::string(buf);
            if (i % 16 == 0) {
                ss << "\n";
            }
        }
        ss << "};\n";
        ss << "unsigned int model_tiny_len = sizeof(model_tiny);\n";
        os << ss.str();
    }
    // [low, high) in byte
    std::pair<uint64_t, uint64_t> span_in_byte(mlir::MemRefType memref) {
        if (!memref.getNumElements()) {
            CC_ABORT << "empty tensor!!\n";
        }

        int64_t offset_in_bytes;
        llvm::SmallVector<int64_t> stride;
        if (failed(getStridesAndOffset(memref, stride, offset_in_bytes))) {
            CC_ABORT << "only support strided memref type\n";
        }

        if (stride.size() != memref.getShape().size()) {
            CC_ABORT << "mismatch of ndim of strides\n";
        }

        int64_t low = 0, high = 0;
        for (size_t i = 0; i < memref.getShape().size(); ++i) {
            if (stride[i] < 0) {
                low += stride[i] * (memref.getDimSize(i) - 1);
            } else {
                high += stride[i] * (memref.getDimSize(i) - 1);
            }
        }

        ++high;

        low *= memref.getElementTypeBitWidth() >> 3;
        high *= memref.getElementTypeBitWidth() >> 3;

        low += offset_in_bytes;
        high += offset_in_bytes;

        if (!(0 <= low && low < high)) {
            CC_ABORT << "invalid memory assignment\n";
        }
        return {low, high};
    }

    Offset<MegCC::DType> type_to_dtype(mlir::Type type) {
        if (type.isF32()) {
            return MegCC::CreateDType(m_fbs_builder, MegCC::DTypeEnum_Float32);
        } else if (auto inttype = type.dyn_cast_or_null<IntegerType>()) {
            if (inttype.isQuant()) {
                float scale = inttype.getScale();
                auto param = MegCC::CreateDTypeParam(m_fbs_builder, scale);
                if (inttype.isInteger(32)) {
                    return MegCC::CreateDType(m_fbs_builder,
                                              MegCC::DTypeEnum_QInt32, param);
                } else if (inttype.isInteger(8) && inttype.isUnsigned()) {
                    return MegCC::CreateDType(m_fbs_builder,
                                              MegCC::DTypeEnum_QUint8, param);
                } else if (inttype.isInteger(8)) {
                    return MegCC::CreateDType(m_fbs_builder,
                                              MegCC::DTypeEnum_QInt8, param);
                } else {
                    CC_ABORT << "unsupported dtype qint" << inttype.getWidth()
                             << "\n";
                }
            } else {
                if (inttype.isInteger(32)) {
                    return MegCC::CreateDType(m_fbs_builder,
                                              MegCC::DTypeEnum_Int32);
                } else if (inttype.isInteger(8) && inttype.isUnsigned()) {
                    return MegCC::CreateDType(m_fbs_builder,
                                              MegCC::DTypeEnum_Uint8);
                } else if (inttype.isInteger(8)) {
                    return MegCC::CreateDType(m_fbs_builder,
                                              MegCC::DTypeEnum_Int8);
                } else {
                    CC_ABORT << "unsupported dtype int" << inttype.getWidth()
                             << "\n";
                }
            }
        } else {
            std::string dtype_str;
            llvm::raw_string_ostream raw_os(dtype_str);
            type.print(raw_os);
            CC_ABORT << "unsupported dtype " << dtype_str << "\n";
        }
        return {};
    }

    Offset<Vector<int32_t>> shaped_type_to_vec(mlir::ShapedType type) {
        return m_fbs_builder.CreateVectorScalarCast<int32_t>(
                type.getShape().data(), type.getShape().size());
    }

    bool check_layout(llvm::ArrayRef<int64_t> stride,
                      llvm::ArrayRef<int64_t> shape) {
        if (stride.size() != shape.size())
            return false;
        return true;
    }

    Offset<MegCC::Workspace> value_to_workspace(mlir::Value workspace,
                                                std::string reason) {
        if (!workspace) {
            return MegCC::CreateWorkspace(m_fbs_builder, 0, 0);
        }

        auto memref = workspace.getType().dyn_cast<MemRefType>();
        if (!memref.getElementType().isInteger(8)) {
            CC_ABORT << "invalid workspace data type, it should be int8";
        }

        int64_t offset;
        llvm::SmallVector<int64_t> stride;
        if (failed(getStridesAndOffset(memref, stride, offset))) {
            CC_ABORT << "only support strided memref type\n";
        }

        if (stride.size() != 1 || stride[0] != 1) {
            CC_ABORT << "invalid workspace layout\n";
        }

        size_t size = memref.getNumElements();

        LOG_DEBUG << "create workspace from " << reason << " with size=" << size
                  << ", offset=" << offset << "\n";

        return MegCC::CreateWorkspace(m_fbs_builder, size, offset);
    }

    Offset<MegCC::Tensor> memref_to_tensor(mlir::MemRefType memref,
                                           std::string name = "") {
        int64_t offset;
        llvm::SmallVector<int64_t> stride;
        if (failed(getStridesAndOffset(memref, stride, offset))) {
            CC_ABORT << "only support strided memref type\n";
        }

        if (!check_layout(stride, memref.getShape())) {
            CC_ABORT << "invalid layout\n";
        }

        Offset<MegCC::Layout> layout = MegCC::CreateLayout(
                m_fbs_builder,
                // dims
                shaped_type_to_vec(memref),
                // stride
                m_fbs_builder.CreateVectorScalarCast<int32_t>(stride.data(),
                                                              stride.size()),
                // format
                MegCC::Format_NCHW);

        auto shape = memref.getShape();
        bool is_dynamic_shape = false;
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] < 0) {
                is_dynamic_shape = true;
            }
        }

        return MegCC::CreateTensor(m_fbs_builder,
                                   // dtype
                                   type_to_dtype(memref.getElementType()),
                                   // layout
                                   layout,
                                   // offset
                                   offset,
                                   // dynamic
                                   is_dynamic_shape,
                                   // usecount
                                   0,
                                   // name
                                   m_fbs_builder.CreateString(name));
    }

    Offset<MegCC::IndexDesc> indexdesc_to_fbs(ArrayAttr desc){
        CC_ASSERT(desc.size()==5);
        auto member = llvm::to_vector<5>(desc.getAsRange<IntegerAttr>());
        return MegCC::CreateIndexDesc(m_fbs_builder, member[0].getInt(),
                                      member[1].getInt(), member[2].getInt(),
                                      member[3].getInt(), member[4].getInt());
    }

    Offset<MegCC::Weight> attr_to_weight(Attribute attr, StringRef name, int32_t user_count) {
        auto dense = attr.cast<DenseElementsAttr>();
        auto size_in_bits = dense.getType().getSizeInBits();
        if (size_in_bits & 0x7) {
            CC_ABORT << "unsupport data type\n";
        }

        auto size_in_bytes = size_in_bits >> 3;
        std::vector<int8_t> data(size_in_bytes);
        if (dense.getType().getElementType().isF32()) {
            float* ptr = reinterpret_cast<float*>(data.data());
            for (auto&& i : dense.getValues<float>()) {
                *ptr = i;
                ++ptr;
            }
        } else if (dense.getType().getElementType().isInteger(32)) {
            int* ptr = reinterpret_cast<int*>(data.data());
            for (auto&& i : dense.getValues<int>()) {
                *ptr = i;
                ++ptr;
            }
        } else if (dense.getType().getElementType().isInteger(8) &&
                   dense.getType().getElementType().isSignedInteger()) {
            int8_t* ptr = reinterpret_cast<int8_t*>(data.data());
            for (auto&& i : dense.getValues<int8_t>()) {
                *ptr = i;
                ++ptr;
            }
        } else if (dense.getType().getElementType().isInteger(8) &&
                   dense.getType().getElementType().isUnsignedInteger()) {
            uint8_t* ptr = reinterpret_cast<uint8_t*>(data.data());
            for (auto&& i : dense.getValues<uint8_t>()) {
                *ptr = i;
                ++ptr;
            }
        } else {
            std::string type_string;
            llvm::raw_string_ostream raw_os(type_string);
            dense.getType().getElementType().print(raw_os);
            CC_ABORT << "unsupport data type: " << type_string << "\n";
        }

        auto shape = dense.getType();
        llvm::SmallVector<int64_t> stride(shape.getShape().size());
        uint64_t curr = 1;
        if (shape.getShape().size() > 0) {
            for (int i = shape.getShape().size() - 1; i >= 0; i--) {
                stride[i] = curr;
                curr *= shape.getShape()[i];
            }
        }

        Offset<MegCC::Layout> layout = MegCC::CreateLayout(
                m_fbs_builder,
                // dims
                shaped_type_to_vec(dense.getType()),
                // stride
                m_fbs_builder.CreateVectorScalarCast<int32_t>(stride.data(),
                                                              stride.size()),
                // format
                MegCC::Format_NCHW);
        return MegCC::CreateWeight(
                m_fbs_builder,
                // dtype
                type_to_dtype(dense.getType().getElementType()),
                // layout
                layout,
                // use_count
                user_count,
                // data
                m_fbs_builder.CreateVector(data),
                // name
                m_fbs_builder.CreateString(name.str()));
    }

    MegCC::ArithMode convert_arithmetic_mode(llvm::StringRef strref) {
        auto str = strref.str();
        if (str == "ROUND")
            return MegCC::ArithMode_ROUND;
        else if (str == "NEGATE")
            return MegCC::ArithMode_NEGATE;
        else if (str == "SUB")
            return MegCC::ArithMode_SUB;
        else if (str == "ADD")
            return MegCC::ArithMode_ADD;
        else if (str == "MUL")
            return MegCC::ArithMode_MUL;
        else if (str == "MAX")
            return MegCC::ArithMode_Max;
        else if (str == "MIN")
            return MegCC::ArithMode_Min;
        else if (str == "LT")
            return MegCC::ArithMode_LT;
        else if (str == "LEQ")
            return MegCC::ArithMode_LEQ;
        else if (str == "FLOOR_DIV")
            return MegCC::ArithMode_FLOORDIV;
        else if (str == "TRUE_DIV")
            return MegCC::ArithMode_TRUE_DIV;
        else {
            CC_ABORT << "Not supported arithmetic mode: " << str << "\n";
        }
    }

    MegCC::InterpolationMode convert_interpolation_mode(
            llvm::StringRef strref) {
        auto str = strref.str();
        if (str == "LINEAR")
            return MegCC::InterpolationMode_LINEAR;
        else {
            CC_ABORT << "Not supported InterpolationMode mode: " << str << "\n";
        }
    }

    MegCC::BorderModeMode convert_bordermodemode_mode(llvm::StringRef strref) {
        auto str = strref.str();
        if (str == "REPLICATE")
            return MegCC::BorderModeMode_REPLICATE;
        else {
            CC_ABORT << "Not supported BorderModeMode mode: " << str << "\n";
        }
    }

    MegCC::Format convert_format(llvm::StringRef strref) {
        auto str = strref.str();
        if (str == "NCHW")
            return MegCC::Format_NCHW;
        else {
            CC_ABORT << "Not supported format mode: " << str << "\n";
        }
    }

    MegCC::Device get_tiny_device(const llvm::StringRef& func_name) {
        switch (target_arch) {
            case megcc::KernelGen::BAREMETAL:
                return MegCC::Device_BARE_METAL;
            case megcc::KernelGen::ARM64:
                return MegCC::Device_ARM64;
            case megcc::KernelGen::ARMV7:
                return MegCC::Device_ARM32;
            case megcc::KernelGen::ARM64V7: {
                //! magic str
                if (func_name.endswith(
                            megcc::KernelGen::ARM64V7_ARM64_POSTFIX)) {
                    return MegCC::Device_ARM64;
                } else {
                    CC_ASSERT(func_name.endswith(
                            megcc::KernelGen::ARM64V7_ARMV7_POSTFIX));
                    return MegCC::Device_ARM32;
                }
            }
            default:
                CC_ABORT << "not supported device.\n";
        }
        return MegCC::Device_BARE_METAL;
    }

    ModuleOp m_root;
    FlatBufferBuilder m_fbs_builder;
    std::unordered_map<std::string, int32_t> symbol2weight_id;
};
}  // namespace

void export_tinynn_model(ModuleOp top_module, std::string save_path,
                         const bool save_model_as_symbol,
                         KernelExporter& kernel_exporter) {
    LOG_DEBUG << "\n\t\t\t Begin Export TinyNN \t\t\t\n";
    Exporter exporter(top_module);
    exporter.save_model(save_path, kernel_exporter, save_model_as_symbol);
    LOG_DEBUG << "\t\t\t End Export TinyNN \t\t\t\n\n";
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
