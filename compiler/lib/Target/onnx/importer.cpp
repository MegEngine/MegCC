#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <onnx/onnx_pb.h>
#include "compiler/Common/Logger.h"
#include "compiler/Common/MemoryStatus.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Target/onnx/helper.h"
#include "compiler/Target/onnx/import.h"
#include "onnx/common/common.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/shape_inference/implementation.h"

namespace mlir {
namespace ONNX {
class ONNXImporter {
public:
    ONNXImporter(mlir::ModuleOp mod)
            : m_module(mod), m_context(m_module->getContext()), m_builder(m_context) {
        m_context->loadDialect<mlir::MGB::MGBDialect>();
        m_context->loadDialect<mlir::StandardOpsDialect>();
    }
    ~ONNXImporter() {}

    mlir::LogicalResult import_onnx(std::string model_path) {
        OpBuilder::InsertionGuard _(m_builder);

        ONNX_NAMESPACE::ModelProto model;
        ONNX_NAMESPACE::LoadProtoFromPath<ONNX_NAMESPACE::ModelProto>(
                model_path, model);
        // comment: the shape of conv pads must be [x1_begin, x2_begin...x1_end, x2_end]
        // or it can't be inferred by ONNX_NAMESPACE::shape_inference::InferShapes
        ONNX_NAMESPACE::shape_inference::InferShapes(model);

        std::unique_ptr<ONNX_NAMESPACE::Graph> g(
                ONNX_NAMESPACE::ImportModelProto(model));

        m_builder.setInsertionPointToEnd(m_module.getBody());
        auto func = m_builder.create<mlir::FuncOp>(
                m_builder.getUnknownLoc(), g->name(),
                get_func_type(g->inputs(), g->outputs()));
        mlir::Block* entryBlock = func.addEntryBlock();
        m_builder.setInsertionPointToStart(entryBlock);

        // deal with inputs
        for (int i = 0; i < g->inputs().size(); ++i) {
            std::string name = g->inputs()[i]->uniqueName();
            m_value2value[g->inputs()[i]] = entryBlock->getArgument(i);
            func.setArgAttr(i, "mgb.func_arg_name", m_builder.getStringAttr(name));
        }

        // build a map for initializer to get its value* by name
        std::unordered_map<std::string, ONNX_NAMESPACE::Value*> init_map;
        for (auto node : g->nodes()) {
            auto inputs = node->inputs();
            auto outputs = node->outputs();
            for (ONNX_NAMESPACE::Value* input : inputs)
                init_map.emplace(input->uniqueName(), input);
            for (ONNX_NAMESPACE::Value* output : outputs)
                init_map.emplace(output->uniqueName(), output);
        }

        std::unordered_map<ONNX_NAMESPACE::Value*, ONNX_NAMESPACE::Tensor> tensor_map;
        // save initializers as ParamStorage and load by ParamProvider
        int size = g->initializers().size();
        for (int i = 0; i < size; ++i) {
            std::string initializer_name = g->initializer_names()[i];
            ONNX_NAMESPACE::Tensor initializer = g->initializers()[i];
            ONNX_NAMESPACE::Value* init_value = init_map.at(initializer_name);
            tensor_map.emplace(init_value, initializer);
            auto storage = create_param_storage(initializer, init_value);
            mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
                    m_builder.getUnknownLoc(), storage);
            m_value2value[init_value] = value;
        }

        // elemwiseMap maps elemwise opr in onnx to elemwise mode in mgb
        std::unordered_map<ONNX_NAMESPACE::BuiltinSymbol, megdnn::param::Elemwise::Mode>
                elemwiseMap;
        elemwiseMap.emplace(
                ONNX_NAMESPACE::BuiltinSymbol::kAdd,
                megdnn::param::Elemwise::Mode::ADD);
        elemwiseMap.emplace(
                ONNX_NAMESPACE::BuiltinSymbol::kSigmoid,
                megdnn::param::Elemwise::Mode::SIGMOID);
        elemwiseMap.emplace(
                ONNX_NAMESPACE::BuiltinSymbol::kMul,
                megdnn::param::Elemwise::Mode::MUL);

        // deal with oprs
        for (auto node : g->nodes()) {
            LOG_DEBUG << node->kind().toString() << "\n";
            if (!strcmp(node->kind().toString(), "Add")) {
                ONNX_NAMESPACE::Value* output = node->output();
                mlir::Value value = m_builder.create<mlir::MGB::Elemwise>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        onnxValueToMLIRValue(node->inputs()),
                        elemwiseMap[ONNX_NAMESPACE::BuiltinSymbol::kAdd]);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Mul")) {
                ONNX_NAMESPACE::Value* output = node->output();
                mlir::Value value = m_builder.create<mlir::MGB::Elemwise>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        onnxValueToMLIRValue(node->inputs()),
                        elemwiseMap[ONNX_NAMESPACE::BuiltinSymbol::kMul]);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Sigmoid")) {
                ONNX_NAMESPACE::Value* output = node->output();
                mlir::Value value = m_builder.create<mlir::MGB::Elemwise>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        onnxValueToMLIRValue(node->inputs()),
                        elemwiseMap[ONNX_NAMESPACE::BuiltinSymbol::kSigmoid]);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Conv")) {
                ONNX_NAMESPACE::Value* output = node->output();
                CC_ASSERT(node->hasAttribute(
                        ONNX_NAMESPACE::BuiltinSymbol::kkernel_shape));
                auto kernel_shape =
                        node->is(ONNX_NAMESPACE::BuiltinSymbol::kkernel_shape);
                CC_ASSERT(
                        node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kdilations));
                auto dilations = node->is(ONNX_NAMESPACE::BuiltinSymbol::kdilations);
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kpads));
                auto pads = node->is(ONNX_NAMESPACE::BuiltinSymbol::kpads);
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kstrides));
                auto strides = node->is(ONNX_NAMESPACE::BuiltinSymbol::kstrides);
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kgroup));
                auto group = node->i(ONNX_NAMESPACE::BuiltinSymbol::kgroup);
                megdnn::param::ConvBias::Sparse sparse =
                        megdnn::param::ConvBias::Sparse::DENSE;
                if (group > 1)
                    sparse = megdnn::param::ConvBias::Sparse::GROUP;
                megdnn::param::ConvBias::Mode mode =
                        megdnn::param::ConvBias::Mode::CROSS_CORRELATION;

                mlir::Value value = m_builder.create<mlir::MGB::ConvBias>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        onnxValueToMLIRValue(node->inputs()),
                        megdnn::param::ConvBias::NonlineMode::IDENTITY, mode, sparse,
                        megdnn::param::ConvBias::Format::NCHW, pads[0], pads[2],
                        strides[0], strides[1], dilations[0], dilations[1],
                        megdnn::param::ConvBias::ComputeMode::DEFAULT);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Clip")) {
                ONNX_NAMESPACE::Value* output = node->output();
                std::vector<mlir::Value> min_node_input;
                min_node_input.push_back(m_value2value.at(node->input(0)));
                min_node_input.push_back(m_value2value.at(node->input(1)));
                mlir::Value min_output_value = m_builder.create<mlir::MGB::Elemwise>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        min_node_input, megdnn::param::Elemwise::Mode::MAX);
                std::vector<mlir::Value> max_node_input;
                max_node_input.push_back(min_output_value);
                max_node_input.push_back(m_value2value.at(node->input(2)));
                mlir::Value max_output_value = m_builder.create<mlir::MGB::Elemwise>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        max_node_input, megdnn::param::Elemwise::Mode::MIN);
                m_value2value.emplace(output, max_output_value);
            } else if (!strcmp(node->kind().toString(), "Identity")) {
                ONNX_NAMESPACE::Value* output = node->output();
                m_value2value.emplace(output, m_value2value.at(node->input(0)));
            } else if (!strcmp(node->kind().toString(), "Flatten")) {
                ONNX_NAMESPACE::Value* output = node->output();
                mlir::Value value = m_builder.create<mlir::MGB::Reshape>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        onnxValueToMLIRValue(node->inputs()));
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Reshape")) {
                ONNX_NAMESPACE::Value* output = node->output();
                mlir::Value value = m_builder.create<mlir::MGB::Reshape>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input(0)));
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "GlobalAveragePool")) {
                ONNX_NAMESPACE::Value* output = node->output();
                auto sizes = node->input()->sizes();
                uint32_t window_w = sizes[sizes.size() - 2].dim;
                uint32_t window_h = sizes[sizes.size() - 1].dim;
                mlir::Value value = m_builder.create<mlir::MGB::Pooling>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input()),
                        megdnn::param::PoolingV0::Mode::AVERAGE, 0, 0, 1, 1, window_w,
                        window_h, megdnn::param::Convolution::Format::NCHW);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "MaxPool")) {
                ONNX_NAMESPACE::Value* output = node->output();
                CC_ASSERT(node->hasAttribute(
                        ONNX_NAMESPACE::BuiltinSymbol::kkernel_shape));
                auto kernel_shapes =
                        node->is(ONNX_NAMESPACE::BuiltinSymbol::kkernel_shape);
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kpads));
                auto pads = node->is(ONNX_NAMESPACE::BuiltinSymbol::kpads);
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kstrides));
                auto strides = node->is(ONNX_NAMESPACE::BuiltinSymbol::kstrides);
                mlir::Value value = m_builder.create<mlir::MGB::Pooling>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input()),
                        megdnn::param::PoolingV0::Mode::MAX, pads[0], pads[1],
                        strides[0], strides[1], kernel_shapes[0], kernel_shapes[1],
                        megdnn::param::Convolution::Format::NCHW);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Gemm")) {
                ONNX_NAMESPACE::Value* output = node->output();
                bool transA = false;
                bool transB = false;
                if (node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::ktransA))
                    transA = true;
                if (node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::ktransB))
                    transB = true;
                mlir::Value value = m_builder.create<mlir::MGB::MatrixMul>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input(0)),
                        m_value2value.at(node->input(1)), transA, transB,
                        megdnn::param::MatrixMulV1::ComputeMode::DEFAULT,
                        megdnn::param::MatrixMul::Format::DEFAULT);
                if (node->inputs().size() == 3) {
                    std::vector<mlir::Value> addValue;
                    addValue.emplace_back(value);
                    addValue.emplace_back(m_value2value.at(node->input(2)));
                    value = m_builder.create<mlir::MGB::Elemwise>(
                            m_builder.getUnknownLoc(),
                            valueToShapedType(m_context, output), addValue,
                            elemwiseMap[ONNX_NAMESPACE::BuiltinSymbol::kAdd]);
                }
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Relu")) {
                ONNX_NAMESPACE::Value* output = node->output();
                mlir::Value value = m_builder.create<mlir::MGB::Elemwise>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input(0)),
                        megdnn::param::Elemwise::Mode::RELU);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Concat")) {
                ONNX_NAMESPACE::Value* output = node->output();
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kaxis));
                auto axis = node->i(ONNX_NAMESPACE::BuiltinSymbol::kaxis);
                mlir::Value value = m_builder.create<mlir::MGB::Concat>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        onnxValueToMLIRValue(node->inputs()), axis,
                        mgb::CompNode::default_cpu());
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Slice")) {
                ONNX_NAMESPACE::Value* output = node->output();
                auto desc_flag = get_subtensor_desc_and_flag(node, tensor_map);
                auto desc = std::get<0>(desc_flag);
                auto flag = std::get<1>(desc_flag);
                // only give the first input, otherwise this op would be dealt as
                // dynamic shape during MGBToKernel conversion
                mlir::Value value = m_builder.create<mlir::MGB::Subtensor>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input(0)), desc, flag);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Transpose")) {
                ONNX_NAMESPACE::Value* output = node->output();
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kperm));
                auto perm = node->is(ONNX_NAMESPACE::BuiltinSymbol::kperm);
                std::vector<int32_t> perm_32(perm.size());
                std::transform(perm.begin(), perm.end(), perm_32.begin(), [](int n) {
                    return static_cast<int32_t>(n);
                });
                mlir::Value value = m_builder.create<mlir::MGB::Dimshuffle>(
                        m_builder.getUnknownLoc(), valueToShapedType(m_context, output),
                        m_value2value.at(node->input()), perm_32);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "ReduceMean")) {
                ONNX_NAMESPACE::Value* output = node->output();
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kaxes));
                auto axes = node->is(ONNX_NAMESPACE::BuiltinSymbol::kaxes);
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kkeepdims));
                auto keepdims = node->i(ONNX_NAMESPACE::BuiltinSymbol::kkeepdims);
                ::megdnn::param::Reduce::Mode mode =
                        ::megdnn::param::Reduce::Mode::MEAN;
                ::megdnn::param::Reduce::DataType data_type =
                        ::megdnn::param::Reduce::DataType::DEFAULT;

                // ReduceMean is simulated by Reduce and AxisAddRemove
                mlir::Value reduce_input = m_value2value.at(node->input());
                std::vector<ONNX_NAMESPACE::Dimension> previous_sizes =
                        node->input()->sizes();
                mlir::Value final_output;

                for (auto it = axes.rbegin(); it != axes.rend(); it++) {
                    size_t axis = *it;
                    // form the shape of reduce's output
                    std::vector<ONNX_NAMESPACE::Dimension> reduce_output_sizes(
                            previous_sizes.begin(), previous_sizes.begin() + axis);
                    reduce_output_sizes.emplace_back(ONNX_NAMESPACE::Dimension(1));
                    reduce_output_sizes.insert(
                            reduce_output_sizes.end(),
                            previous_sizes.begin() + axis + 1, previous_sizes.end());

                    ONNX_NAMESPACE::Value* reduce_output_resized =
                            output->setSizes(reduce_output_sizes);
                    LOG_DEBUG << "reduce_output_size: " << reduce_output_sizes.size()
                              << "\n";
                    mlir::Value reduce_output = m_builder.create<mlir::MGB::Reduce>(
                            m_builder.getUnknownLoc(),
                            valueToShapedType(m_context, reduce_output_resized),
                            reduce_input, mode, axis, data_type);
                    reduce_input = reduce_output;
                    final_output = reduce_output;
                    previous_sizes = reduce_output_sizes;

                    if (keepdims == 0) {
                        std::vector<ONNX_NAMESPACE::Dimension> reshape_output_sizes(
                                previous_sizes.begin(), previous_sizes.begin() + axis);
                        reshape_output_sizes.insert(
                                reshape_output_sizes.end(),
                                previous_sizes.begin() + axis + 1,
                                previous_sizes.end());
                        ONNX_NAMESPACE::Value* reshape_output_resized =
                                output->setSizes(reshape_output_sizes);
                        LOG_DEBUG << "reshape_output_size: "
                                  << reshape_output_sizes.size() << "\n";
                        mlir::Value axisremove_output =
                                m_builder.create<mlir::MGB::Reshape>(
                                        m_builder.getUnknownLoc(),
                                        valueToShapedType(
                                                m_context, reshape_output_resized),
                                        reduce_output);
                        reduce_input = axisremove_output;
                        final_output = axisremove_output;
                        previous_sizes = reshape_output_sizes;
                    }
                }
                m_value2value.emplace(output, final_output);
            } else if (!strcmp(node->kind().toString(), "Constant")) {
                ONNX_NAMESPACE::Value* output = node->output();
                CC_ASSERT(node->hasAttribute(ONNX_NAMESPACE::BuiltinSymbol::kvalue));
                ONNX_NAMESPACE::Tensor value_tensor =
                        node->t(ONNX_NAMESPACE::BuiltinSymbol::kvalue);
                if (value_tensor.sizes().size() == 0) {
                    std::vector<ONNX_NAMESPACE::Dimension> constant_sizes;
                    constant_sizes.emplace_back(ONNX_NAMESPACE::Dimension(1));
                    output->setSizes(constant_sizes);
                }
                auto storage = create_param_storage(value_tensor, output);
                mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
                        m_builder.getUnknownLoc(), storage);
                m_value2value.emplace(output, value);
            } else if (!strcmp(node->kind().toString(), "Undefined")) {
                continue;
            } else {
                CC_ABORT << "unsupported onnx operator type " << node->kind().toString()
                         << "\n";
            }
        }

        std::vector<mlir::Value> results;
        // deal with outputs
        for (size_t i = 0; i < g->outputs().size(); ++i) {
            std::string name = g->outputs()[i]->uniqueName();
            func.setResultAttr(
                    i, "mgb.func_result_name", m_builder.getStringAttr(name));
            results.push_back(m_value2value.at(g->outputs()[i]));
        }
        m_builder.create<ReturnOp>(m_builder.getUnknownLoc(), results);
        m_value2value.clear();
        return mlir::verify(m_module);
    }

private:
    std::tuple<
            std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>>,
            std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>>>
    get_subtensor_desc_and_flag(
            ONNX_NAMESPACE::Node* node,
            std::unordered_map<ONNX_NAMESPACE::Value*, ONNX_NAMESPACE::Tensor>&
                    tensor_map) {
        ONNX_NAMESPACE::Value* output = node->output();
        auto starts_tensor = tensor_map.at(node->input(1));
        size_t nr_elems = 1;
        for (int size : starts_tensor.sizes()) {
            nr_elems *= size;
        }
        std::vector<int64_t> starts;
        starts.resize(nr_elems);
        memcpy(starts.data(), starts_tensor.data<int64_t>(),
               nr_elems * sizeof(int64_t));

        auto ends_tensor = tensor_map.at(node->input(2));
        nr_elems = 1;
        for (int size : ends_tensor.sizes()) {
            nr_elems *= size;
        }
        std::vector<int64_t> ends;
        ends.resize(nr_elems);
        memcpy(ends.data(), ends_tensor.data<int64_t>(), nr_elems * sizeof(int64_t));
        std::vector<int64_t> axes;
        axes.reserve(output->sizes().size());
        for (int64_t i = 0; i < output->sizes().size(); i++) {
            axes.emplace_back(i);
        }
        std::vector<int64_t> steps;
        steps.reserve(starts.size());
        for (int i = 0; i < starts.size(); i++) {
            steps.emplace_back(1);
        }
        if (node->inputs().size() == 4) {
            ONNX_NAMESPACE::Tensor axes_tensor = tensor_map.at(node->input(3));
            nr_elems = 1;
            for (int size : axes_tensor.sizes()) {
                nr_elems *= size;
            }
            axes.resize(nr_elems);
            memcpy(axes.data(), axes_tensor.data<int64_t>(),
                   nr_elems * sizeof(int64_t));
        }
        if (node->inputs().size() == 5) {
            ONNX_NAMESPACE::Tensor steps_tensor = tensor_map.at(node->input(4));
            nr_elems = 1;
            for (int size : steps_tensor.sizes()) {
                nr_elems *= size;
            }
            steps.resize(nr_elems);
            memcpy(steps.data(), steps_tensor.data<int64_t>(),
                   nr_elems * sizeof(int64_t));
        }
        LOG_DEBUG << starts.size() << " " << ends.size() << " " << axes.size() << " "
                  << steps.size() << "\n";

        CC_ASSERT(starts.size() == ends.size());
        CC_ASSERT(starts.size() == axes.size());
        CC_ASSERT(starts.size() == steps.size());

        std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>> desc;
        std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>> flag;

        for (int i = 0; i < axes.size(); i++) {
            desc.push_back({axes[i], starts[i], ends[i], steps[i], -1});
            flag.push_back({0, 0, 0, 0, -1});
        }

        LOG_DEBUG << "desc size: "
                  << " " << std::get<0>(desc[0]) << " " << std::get<1>(desc[0]) << " "
                  << std::get<2>(desc[0]) << " " << std::get<3>(desc[0]) << " "
                  << std::get<4>(desc[0]) << " "
                  << "\n";

        LOG_DEBUG << "flag size: " << std::get<0>(flag[0]) << " "
                  << std::get<1>(flag[0]) << " " << std::get<2>(flag[0]) << " "
                  << std::get<3>(flag[0]) << " " << std::get<4>(flag[0]) << " "
                  << "\n";

        return {desc, flag};
    }

    // get FunctionType from inputs and outputs
    mlir::FunctionType get_func_type(
            ONNX_NAMESPACE::ArrayRef<ONNX_NAMESPACE::Value*> inputs,
            ONNX_NAMESPACE::ArrayRef<ONNX_NAMESPACE::Value*> outputs) {
        auto arg_types = llvm::to_vector<1>(llvm::map_range(
                inputs, [this](ONNX_NAMESPACE::Value* value) -> mlir::Type {
                    return valueToShapedType(m_context, value);
                }));
        auto result_types = llvm::to_vector<1>(llvm::map_range(
                outputs, [this](ONNX_NAMESPACE::Value* value) -> mlir::Type {
                    return valueToShapedType(m_context, value);
                }));
        return mlir::FunctionType::get(m_context, arg_types, result_types);
    }

    // from onnx Tensor to attr
    mlir::DenseElementsAttr tensor_to_attr(
            const ONNX_NAMESPACE::Tensor& tensor, ONNX_NAMESPACE::Value* init_value) {
#define FOR_EACH_TYPE_CTYPE(DTYPE_ENUM, CTYPE)                                       \
    if (tensor.elem_type() == DTYPE_ENUM) {                                          \
        LOG_DEBUG << "type: " << tensor.elem_type() << "\n";                         \
        size_t nr_elems = 1;                                                         \
        for (int size : tensor.sizes())                                              \
            nr_elems *= size;                                                        \
        std::vector<CTYPE> data(nr_elems);                                           \
        memcpy(data.data(), tensor.data<CTYPE>(), nr_elems * sizeof(CTYPE));         \
        return mlir::DenseElementsAttr::get(                                         \
                valueToShapedType(m_context, init_value), llvm::makeArrayRef(data)); \
    }

        FOR_EACH_TYPE_CTYPE(
                ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, float)
        FOR_EACH_TYPE_CTYPE(
                ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, int)
        FOR_EACH_TYPE_CTYPE(
                ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE,
                double)
        // MegEngine doesn't support int64, convert it to int32
        if (tensor.elem_type() ==
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64) {
            LOG_DEBUG << "type: " << tensor.elem_type() << "\n";
            size_t nr_elems = 1;
            for (int size : tensor.sizes())
                nr_elems *= size;
            std::vector<int64_t> data(nr_elems);
            memcpy(data.data(), tensor.data<int64_t>(), nr_elems * sizeof(int64_t));
            std::vector<int32_t> data_32;
            std::transform(
                    data.begin(), data.end(), std::back_inserter(data_32),
                    [](int64_t num) { return static_cast<int32_t>(num); });
            init_value->setElemType(
                    ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
            return mlir::DenseElementsAttr::get(
                    valueToShapedType(m_context, init_value),
                    llvm::makeArrayRef(data_32));
        }

        CC_ABORT << "unsupported data type " << tensor.elem_type() << '\n';
        return {};
    }

    MGB::ParamStorage& create_param_storage(
            const ONNX_NAMESPACE::Tensor& initializer,
            ONNX_NAMESPACE::Value* init_value) {
        std::string name = init_value->uniqueName();
        bool equal_flag = true;
        auto tensor_attr = tensor_to_attr(initializer, init_value);
        std::string id_str;
        llvm::raw_string_ostream id_stream(id_str);
        id_stream << tensor_attr;
        if (m_param_storage.find(id_str) == m_param_storage.end()) {
            equal_flag = false;
        } else {
            auto param = m_param_storage[id_str];
            int32_t user_count = param.user_count() + 1;
            auto user_count_attr = mlir::IntegerAttr::get(
                    param.user_countAttr().getType(), user_count);
            auto value = param.value();
            if (value != tensor_attr)
                equal_flag = false;
            else
                param.user_countAttr(user_count_attr);
        }
        // std::cout << equal_flag << std::endl;
        if (!equal_flag) {
            OpBuilder::InsertionGuard _(m_builder);
            m_builder.setInsertionPointToStart(m_module.getBody());
            auto storage = m_builder.create<MGB::ParamStorage>(
                    m_builder.getUnknownLoc(), name, tensor_attr,
                    valueToShapedType(m_context, init_value), 1);
            storage.setPrivate();
            m_param_storage.emplace(id_str, storage);
        }
        return m_param_storage[id_str];
    }

    std::vector<mlir::Value> onnxValueToMLIRValue(
            std::vector<ONNX_NAMESPACE::Value*> values) {
        std::vector<mlir::Value> inputs;
        inputs.reserve(values.size());
        for (auto v : values) {
            // find value by outputs from previous nodes
            inputs.push_back(m_value2value.at(v));
        }
        return inputs;
    }

    mlir::ModuleOp m_module;
    mlir::MLIRContext* m_context;
    mlir::OpBuilder m_builder;
    std::unordered_map<ONNX_NAMESPACE::Value*, mlir::Value> m_value2value;
    std::unordered_map<std::string, MGB::ParamStorage> m_param_storage;
};

mlir::LogicalResult removeUnusedParam(mlir::ModuleOp module) {
    mlir::PassManager pm(module->getContext());
    pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    return pm.run(module);
}

mlir::LogicalResult import_onnx(mlir::ModuleOp module, std::string model_path) {
    LOG_DEBUG << "\n\t\t\t Begin Import ONNX \t\t\t\n";
    LOG_DEBUG << "load model from " << model_path << "\n";
    ONNXImporter imp(module);
    auto result = imp.import_onnx(model_path);
    LOG_DEBUG << "\t\t\t End Import onnx \t\t\t\n\n";
    if (mlir::failed(result))
        return result;
    return removeUnusedParam(module);
}

}  // namespace ONNX
}  // namespace mlir