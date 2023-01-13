/**
 * \file compiler/lib/Target/MGB/importer.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <memory>
#include <set>

#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/MemoryStatus.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Target/Hako/hako_parse.h"
#include "compiler/Target/MGB/dummy_loader.h"
#include "compiler/Target/MGB/helper.h"
#include "compiler/Target/MGB/import.h"

#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/serialization/extern_c_opr_io.h"
#include "megbrain/serialization/serializer.h"

llvm::cl::opt<int> hako_version(
        "hako", llvm::cl::desc("specific version used for encrypt"),
        llvm::cl::init(2));
llvm::cl::opt<std::string> ExternOprOutputShape(
        "extern-opr-output-shapes", llvm::cl::Optional,
        llvm::cl::desc("specific extern opr output shapes"),
        llvm::cl::value_desc(
                "loader_name_1=output_shape_1;output_shape_2;...:"
                "loader_name_2=output_shape_1;output_shape_2;... "
                "If only one loader, \"loader_name=\" can be omitted."
                "e.g., "
                "\"loader_1=(1,3,5,5);(1,1);(3,3):loader_2=(2,2);(1,1,3,3)\""));
llvm::cl::opt<std::string> ExternOprOutputDType(
        "extern-opr-output-dtypes", llvm::cl::Optional,
        llvm::cl::desc("specific extern opr output dtypes"),
        llvm::cl::value_desc(
                "Similar to --extern-opr-output-shapes but without "
                "\"loader_name\"."
                "The available values are float32, int32, uint8, float16, "
                "int16. e.g., \"float32;int32;uint8:float16;int16\". Default "
                "value is float32."));
llvm::cl::opt<std::string> ExternOprLoaderPathWithInterface(
        "loader-path-with-interface", llvm::cl::Optional,
        llvm::cl::desc("specific extern opr loader path with interface. If "
                       "\"interface\" "
                       "is not provided, using \"mgb_c_opr_init\" default."),
        llvm::cl::value_desc("loader_path:interface"));
llvm::cl::opt<std::string> ExternOprLoaderEnv(
        "set-extern-opr-env", llvm::cl::Optional,
        llvm::cl::desc("set ENV for all extern opr loader, must surrounded by "
                       "\" if set multiple ENV."),
        llvm::cl::value_desc("\"ENV_1=VALUE_1;ENV_2=VALUE_2...\""));

using namespace mgb;
using namespace llvm;

namespace mgb {
namespace cg {
namespace {
class ComputeDepOprIter {
public:
    using VarNodeMap =
            std::unordered_map<VarNode*, std::vector<OperatorNodeBase*>>;
    using Callback = thin_function<void(OperatorNodeBase*, VarNodeMap&)>;

    ComputeDepOprIter(Callback cb, VarNodeMap& varnode_to_used_opr)
            : m_cb{std::move(cb)}, m_varnode_to_used_opr(varnode_to_used_opr) {}

    void iter(SymbolVar var) { return iter(var.node()->owner_opr()); }

    void iter(OperatorNodeBase* opr) {
        if (m_visited.count(opr))
            return;
        m_visited.insert(opr);
        for (auto& inp : opr->input()) {
            m_varnode_to_used_opr[inp].push_back(opr);
        }
        auto&& static_infer_mgr = opr->owner_graph()->static_infer_manager();
        for (auto&& i : opr->node_prop().dep_map()) {
            if (OperatorNodeProp::is_device_value_dep(i.second)) {
                iter(i.first);
            } else if (i.second & OperatorNodeProp::DepType::SHAPE) {
                if (!static_infer_mgr.infer_shape_fallible(i.first)) {
                    iter(i.first);
                }
            } else if (i.second & OperatorNodeProp::DepType::HOST_VALUE) {
                iter(i.first);
            }
        }
        m_cb(opr, m_varnode_to_used_opr);
    }

private:
    ThinHashSet<OperatorNodeBase*> m_visited;
    std::unordered_map<VarNode*, std::vector<OperatorNodeBase*>>
            m_varnode_to_used_opr;
    Callback m_cb;
};

}  // namespace
}  // namespace cg
}  // namespace mgb

namespace mlir {
namespace MGB {
namespace {
template <typename T>
void sort_inputs_with_index(std::vector<T>& inputs, std::vector<int>& axis) {
    std::vector<std::pair<int, T>> temp_vec;
    CC_ASSERT(inputs.size() == axis.size() + 1);
    for (size_t i = 0; i < axis.size(); ++i) {
        temp_vec.emplace_back(axis[i], inputs[i + 1]);
    }
    std::sort(
            temp_vec.begin(), temp_vec.end(),
            [](const std::pair<int, T>& left, const std::pair<int, T>& right) {
                return left.first < right.first;
            });
    for (size_t i = 0; i < axis.size(); ++i) {
        axis[i] = temp_vec[i].first;
        inputs[i + 1] = temp_vec[i].second;
    }
}

std::vector<int> reduceAxisFromShape(TensorShape src_shape,
                                     TensorShape dst_shape) {
    std::vector<int> res;
    CC_ASSERT(src_shape.ndim == dst_shape.ndim)
            << src_shape.to_string() << ", " << dst_shape.to_string();
    for (size_t i = 0; i < dst_shape.ndim; ++i) {
        if (dst_shape[i] == 1 && src_shape[i] > 1) {
            res.push_back(i);
        }
    }
    return res;
}

std::vector<uint8_t> read_file(std::string path) {
    std::vector<uint8_t> res;
    FILE* fin = fopen(path.c_str(), "rb");
    CC_ASSERT(fin) << "can not open " << path << "\n";
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    res.resize(size);
    fseek(fin, 0, SEEK_SET);
    auto nr = fread(res.data(), 1, size, fin);
    CC_ASSERT(nr == size);
    fclose(fin);
    return res;
}

inline std::vector<std::string> split(std::string str,
                                      const std::string& delimiter) {
    std::vector<std::string> res;
    size_t pos = 0;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        res.emplace_back(std::move(str.substr(0, pos)));
        str.erase(0, pos + delimiter.size());
    }
    res.emplace_back(std::move(str));
    return res;
}

inline void parse_extern_loader_info() {
    auto& name2outputinfo = loaderInfo.m_name_2_outputinfo;

    std::string extern_opr_output_shapes = ExternOprOutputShape;
    if (extern_opr_output_shapes.size()) {
        auto&& output_shapes_loaders = split(extern_opr_output_shapes, ":");
        size_t nr_loader = output_shapes_loaders.size();

        std::string extern_opr_output_dtypes = ExternOprOutputDType;
        bool specify_dtype = (extern_opr_output_dtypes.size() != 0);
        auto&& output_dtypes_loaders = split(extern_opr_output_dtypes, ":");
        if (specify_dtype)
            CC_ASSERT(nr_loader == output_dtypes_loaders.size());

        auto skip_whitespace = [](const std::string& str) {
            int left = 0, right = str.size() - 1;
            while (str[left] == ' ' || str[left] == '\t')
                ++left;
            while (str[right] == ' ' || str[right] == '\t')
                --right;
            return str.substr(left, right - left + 1);
        };

        auto parse_output_info = [=, &name2outputinfo](
                                         const std::string& output_shapes_str,
                                         const std::string& output_dtypes_str,
                                         const std::string& loader_name) {
            auto&& output_shapes = split(output_shapes_str, ";");

            std::vector<uint32_t> uint_output_dtypes(output_shapes.size(), 0);
            if (specify_dtype) {
                auto&& output_dtypes = split(output_dtypes_str, ";");
                CC_ASSERT((output_shapes.size() == output_dtypes.size()))
                        << "Number of extern opr output shapes("
                        << output_shapes.size()
                        << ") should equal to "
                           "number "
                           "of extern opr output dtypes("
                        << output_dtypes.size() << ").\n";
                std::unordered_map<std::string, uint32_t> dtype_str2uint{
                        {"float32", 0},
                        {"int32", 1},
                        {"uint8", 2},
                        {"float16", 3},
                        {"int16", 4}};
                for (size_t i = 0; i < output_dtypes.size(); ++i) {
                    auto&& tmp_str = skip_whitespace(output_dtypes[i]);
                    if (dtype_str2uint.find(tmp_str) != dtype_str2uint.end())
                        uint_output_dtypes[i] = dtype_str2uint.at(tmp_str);
                    else
                        CC_ASSERT(0)
                                << tmp_str
                                << " is invalid extern opr output dtype! Dtype "
                                   "should be float32, int32, uint8, float16 "
                                   "or "
                                   "int16.\n";
                }
            }

            std::vector<std::vector<uint32_t>> uint_output_shapes(
                    output_shapes.size());
            for (size_t i = 0; i < output_shapes.size(); ++i) {
                auto&& tmp_str = skip_whitespace(output_shapes[i]);
                CC_ASSERT((tmp_str[0] == '(' &&
                           tmp_str[tmp_str.size() - 1] == ')'))
                        << "The output shape needs to be surrounded by "
                           "parentheses.\n";
                tmp_str = tmp_str.substr(1, tmp_str.size() - 2);
                auto&& tmp_shape = split(tmp_str, ",");
                CC_ASSERT((tmp_shape.size() <= MGB_TENSOR_MAX_NDIM))
                        << "Maximum dimension of single output shape of extern "
                           "opr "
                           "is "
                        << MGB_TENSOR_MAX_NDIM << ".\n";
                uint_output_shapes[i].resize(tmp_shape.size());
                std::transform(tmp_shape.begin(), tmp_shape.end(),
                               uint_output_shapes[i].begin(),
                               [](const std::string& s) {
                                   return static_cast<uint32_t>(std::stoul(s));
                               });
            }

            name2outputinfo[loader_name] =
                    std::make_pair(std::move(uint_output_shapes),
                                   std::move(uint_output_dtypes));
        };

        if (nr_loader == 1) {
            auto&& name_and_shapes = split(output_shapes_loaders[0], "=");
            bool specify_name = (name_and_shapes.size() == 2);
            std::string&& loader_name =
                    (specify_name ? skip_whitespace(name_and_shapes[0]) : "_");
            const std::string& shapes =
                    specify_name ? name_and_shapes[1] : name_and_shapes[0];
            if (specify_dtype) {
                parse_output_info(shapes, output_dtypes_loaders[0],
                                  loader_name);
            } else {
                parse_output_info(shapes, "", loader_name);
            }
        } else if (nr_loader > 1) {
            for (size_t i = 0; i < nr_loader; ++i) {
                auto&& name_and_shapes = split(output_shapes_loaders[i], "=");
                CC_ASSERT((name_and_shapes.size() == 2))
                        << "When there are more than one loader, loader name "
                           "must be specified.\n";
                std::string&& loader_name = skip_whitespace(name_and_shapes[0]);
                const std::string& shapes = name_and_shapes[1];
                if (specify_dtype) {
                    parse_output_info(shapes, output_dtypes_loaders[i],
                                      loader_name);
                } else {
                    parse_output_info(shapes, "", loader_name);
                }
            }
        }
    }

    // parse ENV
    std::string env = ExternOprLoaderEnv;
    if (env.size()) {
        auto&& env_values = split(env, ";");
        for (auto&& env_value : env_values) {
            auto&& env_value_vec = split(env_value, "=");
            CC_ASSERT((env_value_vec.size() == 2))
                    << "Wrong format. Set ENV using \"ENV=VALUE\"";
            loaderInfo.m_envs[env_value_vec[0]] = env_value_vec[1];
        }
    }

    // parse loader path and interface
    std::string loaderPathWithInterface = ExternOprLoaderPathWithInterface;
    if (loaderPathWithInterface.size()) {
        auto&& loaderPath_interface = split(loaderPathWithInterface, ":");
        CC_ASSERT((loaderPath_interface.size() <= 2))
                << "Wrong format. Specify loader path and interface using "
                   "loader_path[:interface]";
        loaderInfo.m_loader_path_with_interface.first = loaderPath_interface[0];
        if (loaderPath_interface.size() == 1 || loaderPath_interface[1] == "") {
            loaderInfo.m_loader_path_with_interface.second = "mgb_c_opr_init";
        } else {
            loaderInfo.m_loader_path_with_interface.second =
                    loaderPath_interface[1];
        }
    }
}

class Importer {
    using LoadResult = serialization::GraphLoader::LoadResult;
    using Options = MGBImporterOptions;

public:
    Importer(mlir::ModuleOp mod)
            : m_module(mod),
              m_context(m_module->getContext()),
              m_builder(m_context) {
        m_context->loadDialect<mlir::MGB::MGBDialect>();
        m_context->loadDialect<mlir::StandardOpsDialect>();
    }

    mlir::LogicalResult import_mgb(std::string model_path, Options options,
                                   int hako_ver = 0) {
        std::vector<uint8_t> mdl_model_buffer;
        std::unique_ptr<serialization::InputFile> inp_file;
        hako_ver = hako_ver == 0 ? hako_version.getValue() : hako_ver;
        if (model_path.substr(model_path.size() - 5, model_path.size()) ==
            ".emod") {
            auto model_buffer = read_file(model_path);
            mdl_model_buffer = megcc::parse_hako(model_buffer, hako_ver);
            std::shared_ptr<void> ptr;
            ptr.reset((char*)malloc(mdl_model_buffer.size()));
            memcpy(ptr.get(), mdl_model_buffer.data(), mdl_model_buffer.size());
            inp_file = serialization::InputFile::make_mem_proxy(
                    ptr, mdl_model_buffer.size());
        } else {
            inp_file = serialization::InputFile::make_fs(model_path.c_str());
        }
        auto format = serialization::GraphLoader::identify_graph_dump_format(
                *inp_file);
        CC_ASSERT(format.valid()) << "invalid model: unknown model format.\n";
        m_loader = serialization::GraphLoader::make(std::move(inp_file),
                                                    format.val());

        parse_extern_loader_info();
        dummy_mgb_c_opr_init(mgb_get_extern_c_opr_api_versioned);

        LOG_DEBUG << "Process mgb graph\n";
        process_graph(options);
        return mlir::verify(m_module);
    }

private:
    mlir::ShapedType var_to_shaped_type(VarNode* var) {
        // dynamic allocation
        return tensorShapeToShapedType(m_context, var->shape(), var->dtype());
    }

    mlir::ShapedType var_to_shaped_type_with_shape(VarNode* var,
                                                   const TensorShape& shape) {
        // dynamic allocation
        return tensorShapeToShapedType(m_context, shape, var->dtype());
    }

    template <typename T>
    T get_const_value_from_var(VarNode* var, T def_val = 0) {
        if (!var) {
            return def_val;
        }
        if (auto immutabletensor_opr =
                    var->owner_opr()->try_cast_final<opr::ImmutableTensor>()) {
            auto host_tensor = immutabletensor_opr->host_value();
            return *host_tensor.ptr<T>();
        } else {
            CC_ABORT << "the opr of var :" << var->name()
                     << " is not ImmutableTensor Opr\n";
        }
    }

    template <typename T>
    std::vector<T> get_const_array_from_var(VarNode* var) {
        CC_ASSERT(var);
        std::vector<T> ret;
        if (auto immutabletensor_opr =
                    var->owner_opr()->try_cast_final<opr::ImmutableTensor>()) {
            auto host_tensor = immutabletensor_opr->host_value();
            auto layout = host_tensor.layout();
            size_t nr_elems = layout.total_nr_elems();
            ret.resize(nr_elems);
            memcpy(ret.data(), host_tensor.raw_ptr(),
                   layout.dtype.size(nr_elems));
            return ret;
        }
        return ret;
    }

    //! return the subtensor desc and corresponding flag
    std::tuple<std::vector<
                       std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>>,
               std::vector<
                       std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>>>
    get_subtensor_desc_and_flag(const opr::indexing::IndexDesc& index_desc,
                                VarNodeArray& inputs_array) {
        std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>>
                desc;
        std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>>
                flag;
        auto is_const_var = [](VarNode* var) {
            if (var->owner_opr()->try_cast_final<opr::ImmutableTensor>()) {
                return true;
            }
            return false;
        };
        //! process one node, if node exist and node is const value: flag = 0,
        //! value is the const value, if node is exist and node is dynamic :
        //! flag = 1, value is the index of the input array, else the node is
        //! not exit, the flag = -1, the value is the default value
        //!
        //! flag corresponding of subtensor param begin, end, step, index
        //! if -1, mean it is not exist
        //! if 0, mean the corresponding value is static value
        //! if 1, mean the corresponding value is dynamic, and value store the
        //! index of input
        auto process_node = [&](VarNode* var, int32_t& value, int32_t& flag,
                                int32_t default_value) {
            if (var) {
                if (!is_const_var(var)) {
                    inputs_array.push_back(var);
                    value = inputs_array.size() - 1;
                    flag = 1;
                } else {
                    value = get_const_value_from_var<int32_t>(var);
                    flag = 0;
                }
            } else {
                value = default_value;
                flag = -1;
            }
        };

        for (auto& item : index_desc) {
            int32_t axis = item.axis.get_raw();
            int32_t begin = 0, end = -1, step = 1, index = -1;
            int32_t begin_flag = -1, end_flag = -1, step_flag = -1,
                    index_flag = -1;

            process_node(item.begin.node(), begin, begin_flag, 0);
            process_node(item.end.node(), end, end_flag, -1);
            process_node(item.step.node(), step, step_flag, 1);
            process_node(item.idx.node(), index, index_flag, -1);

            desc.push_back({axis, begin, end, step, index});
            flag.push_back({0, begin_flag, end_flag, step_flag, index_flag});
        }
        return {desc, flag};
    }

    mlir::DenseElementsAttr dev_tensor_to_attr(const DeviceTensorND& dv) {
        CC_ASSERT(dv.comp_node() == CompNode::default_cpu());
        CC_ASSERT(dv.layout().is_contiguous());
#define FOR_EACH_DTYPE_CTYPE(DTYPE_ENUM, CTYPE)                             \
    if (dv.dtype().enumv() == DTYPE_ENUM) {                                 \
        size_t nr_elems = dv.layout().total_nr_elems();                     \
        std::vector<CTYPE> data(nr_elems);                                  \
        memcpy(data.data(), dv.raw_ptr(), dv.layout().span().dist_byte());  \
        return DenseElementsAttr::get(                                      \
                tensorShapeToShapedType(m_context, dv.shape(), dv.dtype()), \
                llvm::makeArrayRef(data));                                  \
    }

        FOR_EACH_DTYPE_CTYPE(::megdnn::DTypeEnum::Float32, float)
        FOR_EACH_DTYPE_CTYPE(::megdnn::DTypeEnum::Int32, int)
        FOR_EACH_DTYPE_CTYPE(::megdnn::DTypeEnum::Uint8, uint8_t)
        FOR_EACH_DTYPE_CTYPE(::megdnn::DTypeEnum::QuantizedS8, int8_t)
        FOR_EACH_DTYPE_CTYPE(::megdnn::DTypeEnum::QuantizedS32, int)

        CC_ABORT << "unsupported data type " << dv.dtype().name() << '\n';
        return {};
    }
    auto is_dynamic_value(mlir::Value value) {
        auto shape_type = value.getType().cast<ShapedType>();
        return shape_type.getNumDynamicDims() > 0;
    }

    auto var_array_to_value_array(const cg::VarNodeArray& vars,
                                  bool no_dynamic = false) {
        std::vector<mlir::Value> inputs;
        inputs.reserve(vars.size());
        int cnt = 0;
        for (auto&& i : vars) {
            if (no_dynamic) {
                auto tensor = m_var2value.at(i);
                CC_ASSERT(!is_dynamic_value(tensor))
                        << "with input " << cnt << "\n";
            }
            inputs.push_back(m_var2value.at(i));
            cnt++;
        }
        return inputs;
    }

    MGB::ParamStorage& create_param_storage(VarNode* var,
                                            const DeviceTensorND& tensor,
                                            size_t idx) {
        std::string name = var->cname();
        bool equal_flag = true;
        auto tensor_attr = dev_tensor_to_attr(tensor);
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
            if (value != tensor_attr) {
                equal_flag = false;
            } else {
                param.user_countAttr(user_count_attr);
            }
        }

        if (!equal_flag) {
            if (idx != 0) {
                name = std::to_string(idx) + "_" + name;
            }
            OpBuilder::InsertionGuard _(m_builder);
            m_builder.setInsertionPointToStart(m_module.getBody());
            auto storage = m_builder.create<MGB::ParamStorage>(
                    m_builder.getUnknownLoc(), name, tensor_attr,
                    var_to_shaped_type(var), 1);
            storage.setPrivate();
            m_param_storage.emplace(id_str, storage);
        }
        return m_param_storage[id_str];
    }

    void on_opr(
            cg::OperatorNodeBase* opr,
            std::unordered_map<VarNode*, std::vector<cg::OperatorNodeBase*>>&
                    input2opr,
            size_t idx) {
        LOG_DEBUG << "Import Operator type: " << opr->dyn_typeinfo()->name
                  << ", with name: " << opr->name() << ", output size "
                  << opr->output().size() << ", input size "
                  << opr->input().size() << "\n";
        if (opr->same_type<opr::Host2DeviceCopy>()) {
            auto&& iter = m_var2value.find(opr->output(0));
            CC_ASSERT(iter != m_var2value.end());
        } else if (auto sd = opr->try_cast_final<opr::SharedDeviceTensor>()) {
            DeviceTensorND dv(CompNode::default_cpu());
            dv.copy_from(sd->get_dev_tensor()).sync();
            auto storage = create_param_storage(opr->output(0), dv, idx);
            mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
                    m_builder.getUnknownLoc(), storage);
            m_var2value.emplace(opr->output(0), value);
        } else if (auto imm = opr->try_cast_final<opr::ImmutableTensor>()) {
            auto storage = create_param_storage(opr->output(0),
                                                imm->host_value(), idx);
            mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
                    m_builder.getUnknownLoc(), storage);
            m_var2value.emplace(opr->output(0), value);
        } else if (auto mdt = opr->try_cast_final<
                              opr::MultipleDeviceTensorHolder>()) {
            for (size_t i = 0; i < opr->output().size(); ++i) {
                DeviceTensorND dv(CompNode::default_cpu());
                dv.copy_from(*mdt->values()[i]).sync();
                auto storage = create_param_storage(opr->output(i), dv, idx);
                mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
                        m_builder.getUnknownLoc(), storage);
                m_var2value.emplace(opr->output(i), value);
            }
        } else if (auto elem = opr->try_cast_final<opr::Elemwise>()) {
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::Elemwise>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(opr->input()), elem->param().mode);
            m_var2value.emplace(out, value);
        } else if (auto elem_multi =
                           opr->try_cast_final<opr::ElemwiseMultiType>()) {
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::ElemwiseMultiType>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(opr->input()),
                    elem_multi->param().mode);
            m_var2value.emplace(out, value);
        } else if (auto shuffle = opr->try_cast_final<opr::Dimshuffle>()) {
            auto&& p = shuffle->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::Dimshuffle>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)),
                    std::vector<int32_t>{p.pattern, p.pattern + p.pattern_len});
            m_var2value.emplace(out, value);
        } else if (auto mat_inv = opr->try_cast_final<opr::MatrixInverse>()) {
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::MatrixInverse>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)));
            m_var2value.emplace(out, value);
        } else if (auto conv = opr->try_cast_final<opr::Convolution>()) {
            auto&& p = conv->param();
            auto&& out = opr->output(0);
            CC_ASSERT(!is_dynamic_value(m_var2value.at(opr->input(0))) &&
                      !is_dynamic_value(m_var2value.at(opr->input(1))));
            mlir::Value value = m_builder.create<mlir::MGB::Convolution>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)),
                    m_var2value.at(opr->input(1)), p.mode, p.pad_h, p.pad_w,
                    p.stride_h, p.stride_w, p.dilate_h, p.dilate_w, p.sparse,
                    p.format, p.compute_mode);
            m_var2value.emplace(out, value);
        } else if (auto deconv = opr->try_cast_final<
                                 opr::ConvolutionBackwardData>()) {
            auto&& p = deconv->param();
            auto&& out = opr->output(0);
            CC_ASSERT(!is_dynamic_value(m_var2value.at(opr->input(0))) &&
                      !is_dynamic_value(m_var2value.at(opr->input(1))));
            mlir::Value value =
                    m_builder.create<mlir::MGB::ConvolutionBackwardData>(
                            m_builder.getUnknownLoc(), var_to_shaped_type(out),
                            m_var2value.at(opr->input(0)),
                            m_var2value.at(opr->input(1)), p.mode, p.pad_h,
                            p.pad_w, p.stride_h, p.stride_w, p.dilate_h,
                            p.dilate_w, p.sparse, p.format, p.compute_mode);
            m_var2value.emplace(out, value);
        } else if (auto conv = opr->try_cast_final<opr::ConvBiasForward>()) {
            auto&& p = conv->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::ConvBias>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(opr->input(), true), p.nonlineMode,
                    p.mode, p.sparse, p.format, p.pad_h, p.pad_w, p.stride_h,
                    p.stride_w, p.dilate_h, p.dilate_w, p.compute_mode);
            m_var2value.emplace(out, value);
        } else if (auto resize_opr =
                           opr->try_cast_final<opr::ResizeForward>()) {
            auto&& p = resize_opr->param();
            auto&& out = opr->output(0);

            mlir::Value value = m_builder.create<mlir::MGB::Resize>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)), p.imode, p.format);
            m_var2value.emplace(out, value);
        } else if (opr->try_cast_final<opr::Reshape>()) {
            //! reshape and AxisAddRemove opr are all treated as reshape opr
            auto&& out = opr->output(0);
            auto inputs = opr->input();
            if (out->shape().ndim != 0) {
                inputs = {inputs[0]};
            }
            mlir::Value value = m_builder.create<mlir::MGB::Reshape>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(inputs));
            m_var2value.emplace(out, value);
        } else if (opr->try_cast_final<opr::AxisAddRemove>()) {
            auto&& out = opr->output(0);
            auto inputs = opr->input();
            mlir::Value value = m_builder.create<mlir::MGB::Reshape>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    // FIXME: handle symbolic shape, currently we only pass the
                    // input tensor and the statically-known output shape to
                    // reshape op
                    var_array_to_value_array(inputs));
            m_var2value.emplace(out, value);
        } else if (auto powc = opr->try_cast_final<opr::PowC>()) {
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::PowC>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)), powc->param().exp);
            m_var2value.emplace(out, value);
        } else if (auto pooling = opr->try_cast_final<opr::Pooling>()) {
            auto&& p = pooling->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::Pooling>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)), p.mode, p.pad_h, p.pad_w,
                    p.stride_h, p.stride_w, p.window_h, p.window_w, p.format);
            m_var2value.emplace(out, value);
        } else if (auto adapt_pooling =
                           opr->try_cast_final<opr::AdaptivePooling>()) {
            auto&& p = adapt_pooling->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::AdaptivePooling>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)), p.mode, p.format);
            m_var2value.emplace(out, value);
        } else if (auto matmul = opr->try_cast_final<opr::MatrixMul>()) {
            auto&& p = matmul->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::MatrixMul>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)),
                    m_var2value.at(opr->input(1)), p.transposeA, p.transposeB,
                    p.compute_mode, p.format);
            m_var2value.emplace(out, value);
        } else if (auto batched_matmul =
                           opr->try_cast_final<opr::BatchedMatrixMul>()) {
            auto&& p = batched_matmul->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::BatchedMatrixMul>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)),
                    m_var2value.at(opr->input(1)), p.transposeA, p.transposeB,
                    p.compute_mode, p.format);
            m_var2value.emplace(out, value);
        } else if (auto reduce = opr->try_cast_final<opr::Reduce>()) {
            auto&& p = reduce->param();
            auto&& out = opr->output(0);
            auto inputs = opr->input();
            int axis = p.axis;
            if (inputs.size() > 1) {
                CC_ASSERT(inputs.size() == 2 &&
                          inputs[1]
                                  ->owner_opr()
                                  ->try_cast_final<opr::ImmutableTensor>());
                auto imma = inputs[1]
                                    ->owner_opr()
                                    ->try_cast_final<opr::ImmutableTensor>();
                CC_ASSERT(imma->host_value().dtype().enumv() ==
                                  DTypeEnum::Int32 &&
                          imma->host_value().layout().total_nr_elems() == 1);
                auto shape_value = imma->host_value();
                CC_ASSERT(inputs[0]->shape().ndim != 0);
                auto out_shape = out->shape();
                auto input_shape = inputs[0]->shape();
                //! out shape == [1], and input shape nr_dim > 1, we need reduce
                //! all, add reshape to impl
                if (out_shape.ndim == 1 && input_shape.ndim != 1) {
                    TensorShape inner_shape = input_shape;
                    auto nr_elem = input_shape.total_nr_elems();
                    inner_shape.ndim = 1;
                    inner_shape[0] = nr_elem;
                    axis = 0;

                    auto inner_shape_type = tensorShapeToShapedType(
                            m_context, inner_shape, out->dtype());

                    mlir::Value inner_value =
                            m_builder.create<mlir::MGB::Reshape>(
                                    m_builder.getUnknownLoc(), inner_shape_type,
                                    var_array_to_value_array({inputs[0]}));

                    std::vector<mlir::Value> inner_inputs;
                    inner_inputs.push_back(inner_value);
                    mlir::Value value = m_builder.create<mlir::MGB::Reduce>(
                            m_builder.getUnknownLoc(), var_to_shaped_type(out),
                            inner_inputs, p.mode, axis, p.data_type);
                    m_var2value.emplace(out, value);
                    return;
                } else {
                    auto axis_vec = reduceAxisFromShape(input_shape, out_shape);
                    CC_ASSERT(axis_vec.size() == 1) << "only support one axis";
                    axis = axis_vec[0];
                    inputs = {inputs[0]};
                }
            }
            mlir::Value value = m_builder.create<mlir::MGB::Reduce>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(inputs), p.mode, axis,
                    p.data_type);
            m_var2value.emplace(out, value);
        } else if (auto subtensor = opr->try_cast_final<opr::Subtensor>()) {
            auto index_desc = subtensor->index_desc();
            auto&& out = subtensor->output(0);
            VarNodeArray inputs_array;
            inputs_array.push_back(opr->input(0));
            auto desc_flag = get_subtensor_desc_and_flag(
                    subtensor->index_desc(), inputs_array);
            auto desc = std::get<0>(desc_flag);
            auto flag = std::get<1>(desc_flag);
            mlir::Value value = m_builder.create<mlir::MGB::Subtensor>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(inputs_array), desc, flag);
            m_var2value.emplace(out, value);
        } else if (auto set_subtensor =
                           opr->try_cast_final<opr::SetSubtensor>()) {
            auto&& out = set_subtensor->output(0);
            VarNodeArray inputs_array;
            inputs_array.push_back(opr->input(0));
            inputs_array.push_back(opr->input(1));
            auto desc_flag = get_subtensor_desc_and_flag(
                    set_subtensor->index_desc(), inputs_array);
            auto desc = std::get<0>(desc_flag);
            auto flag = std::get<1>(desc_flag);
            mlir::Value value = m_builder.create<mlir::MGB::SetSubtensor>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(inputs_array), desc, flag);
            m_var2value.emplace(out, value);
        } else if (auto concat = opr->try_cast_final<opr::Concat>()) {
            auto&& p = concat->param();
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::Concat>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(opr->input()), p.axis,
                    CompNode::default_cpu());
            m_var2value.emplace(out, value);
        } else if (auto warpperspective =
                           opr->try_cast_final<opr::WarpPerspectiveForward>()) {
            auto&& p = warpperspective->param();
            auto&& out = opr->output(0);
            //! if mat_id size=0 means no mat_id tensor
            std::vector<int32_t> mat_id;
            if (opr->input().size() >= 3) {
                mat_id = get_const_array_from_var<int32_t>(opr->input(2));
            }
            VarNodeArray inputs_array;
            if (mat_id.size() < 1) {
                inputs_array = opr->input();
            } else {
                inputs_array.push_back(opr->input(0));
                inputs_array.push_back(opr->input(1));
            }
            mlir::Value value = m_builder.create<mlir::MGB::WarpPerspective>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(inputs_array), p.imode, p.bmode,
                    p.format, p.border_val, mat_id);
            m_var2value.emplace(out, value);
        } else if (auto typecvt = opr->try_cast_final<opr::TypeCvt>()) {
            auto&& out_dtype = dtype_to_type(m_context, typecvt->param());
            auto&& in_type = dtype_to_type(m_context, opr->input(0)->dtype());
            auto&& out = opr->output(0);
            //! TODO: no handle quantized type
            mlir::Value value = m_builder.create<mlir::MGB::TypeCvt>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)), in_type, out_dtype);
            m_var2value.emplace(out, value);
        } else if (auto getVarShape = opr->try_cast_final<opr::GetVarShape>()) {
            auto&& out = opr->output(0);
            auto axis = getVarShape->param().axis;
            mlir::Value value = m_builder.create<mlir::MGB::GetVarShape>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)), axis);
            m_var2value.emplace(out, value);
        } else if (auto broascast = opr->try_cast_final<opr::Broadcast>()) {
            auto&& out = opr->output(0);
            mlir::Value value = m_builder.create<mlir::MGB::Broadcast>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    var_array_to_value_array(opr->input()));
            m_var2value.emplace(out, value);
        } else if (auto idx_multi =
                           opr->try_cast_final<opr::IndexingMultiAxisVec>()) {
            auto&& out = opr->output(0);
            int slice_node_cnt = 0;
            std::vector<int> axiss;
            for (auto idx_desc : idx_multi->index_desc()) {
                axiss.push_back(idx_desc.axis.get_raw());
                if (idx_desc.begin.node()) {
                    slice_node_cnt++;
                }
                if (idx_desc.end.node()) {
                    slice_node_cnt++;
                }
                if (idx_desc.step.node()) {
                    slice_node_cnt++;
                }
            }
            if (slice_node_cnt == 0) {
                auto mlir_inputs = var_array_to_value_array(opr->input());
                sort_inputs_with_index(mlir_inputs, axiss);
                mlir::Value value =
                        m_builder.create<mlir::MGB::IndexingMultiAxisVec>(
                                m_builder.getUnknownLoc(),
                                var_to_shaped_type(out), mlir_inputs, axiss);
                m_var2value.emplace(out, value);
            } else {
                CC_ASSERT(slice_node_cnt == opr->input().size() - 1)
                        << "subtensor case \n";
                VarNodeArray inputs_array;
                inputs_array.push_back(opr->input(0));
                auto desc_flag = get_subtensor_desc_and_flag(
                        idx_multi->index_desc(), inputs_array);
                auto desc = std::get<0>(desc_flag);
                auto flag = std::get<1>(desc_flag);
                mlir::Value value = m_builder.create<mlir::MGB::Subtensor>(
                        m_builder.getUnknownLoc(), var_to_shaped_type(out),
                        var_array_to_value_array(inputs_array), desc, flag);
                m_var2value.emplace(out, value);
            }
        } else if (auto arg_sort = opr->try_cast_final<opr::Argsort>()) {
            auto&& out_val = opr->output(0);
            auto&& out_index = opr->output(1);
            auto param = arg_sort->param();

            auto values = m_builder.create<mlir::MGB::Argsort>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out_val),
                    var_to_shaped_type(out_index),
                    m_var2value.at(opr->input(0)), param.order);
            m_var2value.emplace(out_val, values.getResult(0));
            m_var2value.emplace(out_index, values.getResult(1));
        } else if (auto arg_max = opr->try_cast_final<opr::Argmax>()) {
            auto&& out_val = opr->output(0);
            auto param = arg_max->param();
            mlir::Value value = m_builder.create<mlir::MGB::Argmax>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out_val),
                    m_var2value.at(opr->input(0)), param.axis);
            m_var2value.emplace(out_val, value);
        } else if (auto arg_topk = opr->try_cast_final<opr::TopK>()) {
            CC_ASSERT(arg_topk->output().size() >= 2)
                    << "only support size >= 2 , but "
                    << arg_topk->output().size() << "\n";
            auto k_opr = arg_topk->input(1)
                                 ->owner_opr()
                                 ->try_cast_final<opr::ImmutableTensor>();
            CC_ASSERT(k_opr)
                    << "k_opr must be ImmutableTensor, but "
                    << arg_topk->input(1)->owner_opr()->dyn_typeinfo()->name
                    << "\n";
            auto k_tensor = k_opr->host_value();
            CC_ASSERT(k_tensor.shape().total_nr_elems() == 1);
            int k = k_tensor.ptr<int>()[0];
            auto&& out_val = opr->output(0);
            auto&& out_index = opr->output(1);
            auto param = arg_topk->param();
            auto values = m_builder.create<mlir::MGB::TopK>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out_val),
                    var_to_shaped_type(out_index),
                    m_var2value.at(opr->input(0)), param.mode, k);
            m_var2value.emplace(out_val, values.getResult(0));
            m_var2value.emplace(out_index, values.getResult(1));
        } else if (auto idx_one = opr->try_cast_final<opr::IndexingOneHot>()) {
            auto&& out = opr->output(0);
            auto param = idx_one->param();
            mlir::Value value = m_builder.create<mlir::MGB::IndexingOneHot>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    m_var2value.at(opr->input(0)),
                    m_var2value.at(opr->input(1)), param.axis);
            m_var2value.emplace(out, value);
        } else if (auto warp_affine = opr->try_cast_final<opr::WarpAffine>()) {
            auto&& p = warp_affine->param();
            auto&& out = opr->output(0);
            if (input2opr[out].size() == 1 &&
                input2opr[out][0]->try_cast_final<opr::Reshape>()) {
                //! FIXME: force warp shape from next reshape..., for no method
                //! to static infer warpaffine, and we do not want to add
                //! dynamic warpaffine

                mlir::Value value = m_builder.create<mlir::MGB::WarpAffine>(
                        m_builder.getUnknownLoc(),
                        var_to_shaped_type_with_shape(
                                out, input2opr[out][0]->output()[0]->shape()),
                        m_var2value.at(opr->input(0)),
                        m_var2value.at(opr->input(1)), p.imode, p.border_mode,
                        p.border_val, p.format);
                m_var2value.emplace(out, value);
            } else {
                mlir::Value value = m_builder.create<mlir::MGB::WarpAffine>(
                        m_builder.getUnknownLoc(), var_to_shaped_type(out),
                        m_var2value.at(opr->input(0)),
                        m_var2value.at(opr->input(1)), p.imode, p.border_mode,
                        p.border_val, p.format);
                m_var2value.emplace(out, value);
            }
        } else if (auto softmax = opr->try_cast_final<opr::Softmax>()) {
            auto&& p = softmax->param();
            auto&& out = opr->output(0);
            auto reduce_shape = out->shape();
            auto&& inp = opr->input(0);
            int32_t axis = p.axis < 0 ? p.axis + inp->shape().ndim : p.axis;
            CC_ASSERT(axis >= 0 &&
                      axis < static_cast<int32_t>(inp->shape().ndim))
                    << "Softmax axis param out of input dim\n";
            reduce_shape[axis] = 1;
            auto reduce_shape_type = tensorShapeToShapedType(
                    m_context, reduce_shape, inp->dtype());
            mlir::Value reduce_out = m_builder.create<mlir::MGB::Reduce>(
                    m_builder.getUnknownLoc(), reduce_shape_type,
                    m_var2value.at(opr->input(0)), opr::Reduce::Mode::MAX,
                    axis);
            std::vector<mlir::Value> sub_inps{m_var2value.at(inp), reduce_out};
            mlir::Value elemwise_sub = m_builder.create<mlir::MGB::Elemwise>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    sub_inps, opr::Elemwise::Mode::SUB);
            mlir::Value elemwise_exp = m_builder.create<mlir::MGB::Elemwise>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    elemwise_sub, opr::Elemwise::Mode::EXP);
            mlir::Value reduce_sum = m_builder.create<mlir::MGB::Reduce>(
                    m_builder.getUnknownLoc(), reduce_shape_type, elemwise_exp,
                    opr::Reduce::Mode::SUM, axis);
            mlir::Value out_value = m_builder.create<mlir::MGB::Elemwise>(
                    m_builder.getUnknownLoc(), var_to_shaped_type(out),
                    std::vector<Value>{elemwise_exp, reduce_sum},
                    opr::Elemwise::Mode::TRUE_DIV);
            m_var2value.emplace(out, out_value);
        } else if (auto extern_opr =
                           opr->try_cast_final<opr::ExternCOprRunner>()) {
            auto user_datas = MGBOprLoaderImpl::get_user_datas();
            void* extra_data = MGBOprLoaderImpl::get_extra_data();

            void* _data = nullptr;
            if (user_datas.find(opr->name()) != user_datas.end()) {
                _data = user_datas[opr->name()];
            }
            CC_ASSERT(_data) << "No data related to " << opr->name() << ".\n";
            std::string data(
                    reinterpret_cast<const char*>(static_cast<char*>(_data) +
                                                  sizeof(size_t)),
                    *(size_t*)(_data));
            uint32_t data_len = static_cast<uint32_t>(data.size());
            if (extra_data)
                data += std::string(reinterpret_cast<const char*>(
                                            static_cast<char*>(extra_data) +
                                            sizeof(size_t)),
                                    *(size_t*)(extra_data));
            free(_data);

            std::vector<mlir::Type> v_resultTypes(opr->output().size());
            for (int i = 0; i < opr->output().size(); ++i) {
                v_resultTypes[i] = var_to_shaped_type(opr->output(i));
            }

            uint32_t nr_input = static_cast<uint32_t>(opr->input().size());
            uint32_t nr_output = static_cast<uint32_t>(opr->output().size());

            auto values = m_builder.create<mlir::MGB::ExternOpr>(
                    m_builder.getUnknownLoc(), v_resultTypes,
                    var_array_to_value_array(opr->input()), opr->name(), data,
                    data_len, nr_input, nr_output);
            for (int i = 0; i < opr->output().size(); ++i) {
                m_var2value.emplace(opr->output(i), values.getResult(i));
            }
        } else {
            CC_ABORT << "unsupported mgb operator type "
                     << opr->dyn_typeinfo()->name << "\n";
        }
    }

    void replace_h2d_in_new_shape(
            const LoadResult::TensorMap& tensor_map,
            const std::map<std::string, megdnn::TensorShape>& input_shapes) {
        std::set<std::string> used_input_shape;
        if (!input_shapes.empty()) {
            for (auto&& i : tensor_map) {
                auto&& iter = input_shapes.find(i.first);
                if (iter != input_shapes.end()) {
                    LOG_DEBUG << "replace mgb H2D tensor of name " << i.first
                              << " to shape :" << iter->second.to_string()
                              << "\n";
                    i.second->resize(iter->second);
                    used_input_shape.insert(i.first);
                }
            }
        }
        for (auto& x : input_shapes) {
            CC_ASSERT(used_input_shape.find(x.first) != used_input_shape.end())
                    << "not used shape name " << x.first << "\n";
        }
    }

    SymbolVarArray disable_h2d_mem_fwd(const SymbolVarArray& dest_vars) {
        ThinHashMap<SymbolVar, SymbolVar> varmap;
        cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
            if (auto h2d = opr->try_cast_final<opr::Host2DeviceCopy>()) {
                if (h2d->output(0)->contain_flag(
                            cg::VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
                    // which means forwarding preallocated memory into output
                    // var
                    auto param = h2d->param();
                    param.allow_cpu_mem_fwd = false;
                    varmap[h2d->output(0)] = opr::Host2DeviceCopy::make(
                            *h2d->owner_graph(), h2d->host_data(), param,
                            h2d->config());
                    LOG_DEBUG << "disable H2D memory forward of operator: "
                              << h2d->name() << "\n";
                }
            }
        });
        for (auto&& i : dest_vars)
            dep.add(i);
        if (!varmap.empty()) {
            return cg::replace_vars(dest_vars, varmap);
        }
        return dest_vars;
    }

    SymbolVarArray append_decouple_weight_pack(
            const SymbolVarArray& dest_vars) {
        ThinHashMap<SymbolVar, SymbolVar> varmap;
        cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
            if (auto mdt = opr->try_cast_final<
                           opr::MultipleDeviceTensorHolder>()) {
                std::vector<SymbolVar> sdt;
                for (auto val : mdt->mutable_values()) {
                    sdt.push_back(opr::SharedDeviceTensor::make_const(
                            *mdt->owner_graph(), val));
                }
                for (size_t i = 0; i < mdt->output().size(); ++i) {
                    varmap[mdt->output()[i]] = sdt[i];
                }
            }
        });
        for (auto&& i : dest_vars)
            dep.add(i);
        if (!varmap.empty()) {
            return cg::replace_vars(dest_vars, varmap);
        }
        return dest_vars;
    }

    SymbolVarArray append_nhwc2nchw_to_h2d(
            const SymbolVarArray& dest_vars,
            serialization::GraphLoader::LoadResult::TensorMap& tensor_map) {
        ThinHashMap<SymbolVar, SymbolVar> varmap;
        cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
            if (auto h2d = opr->try_cast_final<opr::Host2DeviceCopy>()) {
                if (h2d->output(0)->shape().ndim != 4) {
                    return;
                }
                auto param = h2d->param();
                auto host_data = h2d->host_data();
                auto old_shape = host_data->shape();
                auto new_shape = TensorShape({old_shape[0], old_shape[2],
                                              old_shape[3], old_shape[1]});
                std::shared_ptr<HostTensorND> host_data_new =
                        std::make_shared<HostTensorND>(host_data->comp_node(),
                                                       new_shape,
                                                       host_data->dtype());
                std::string old_name = h2d->name();
                for (auto kv : tensor_map) {
                    if (kv.second == host_data) {
                        old_name = kv.first;
                    }
                }
                tensor_map[old_name] = host_data_new;
                auto h2d_opr = opr::Host2DeviceCopy::make(*h2d->owner_graph(),
                                                          host_data_new, param,
                                                          h2d->config());
                varmap[h2d->output(0)] =
                        opr::Dimshuffle::make(h2d_opr, {0, 3, 1, 2});
                LOG_DEBUG << "add nhwc -> nchw to h2d: " << h2d->name()
                          << ", shape " << h2d->output(0)->shape().to_string()
                          << ", dtype " << h2d->output(0)->dtype().name()
                          << ", out " << h2d_opr.node()->name()
                          << "::" << h2d_opr.shape().to_string()
                          << "::" << host_data->layout().to_string() << ":-_-"
                          << new_shape.to_string() << "\n";
            }
        });
        for (auto&& i : dest_vars)
            dep.add(i);
        if (!varmap.empty()) {
            return cg::replace_vars(dest_vars, varmap);
        }
        return dest_vars;
    }

    SymbolVarArray append_reshape_to_h2d(
            const SymbolVarArray& dest_vars,
            std::vector<TensorShape>& input_tensorshape) {
        ThinHashMap<SymbolVar, SymbolVar> varmap;
        cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
            if (auto h2d = opr->try_cast_final<opr::Host2DeviceCopy>()) {
                auto param = h2d->param();
                auto h2d_opr = opr::Host2DeviceCopy::make(*h2d->owner_graph(),
                                                          h2d->host_data(),
                                                          param, h2d->config());

                varmap[h2d->output(0)] =
                        opr::Reshape::make(h2d_opr, h2d->output(0)->shape());
                auto shape = h2d->output(0)->shape();
                input_tensorshape.push_back(shape);
                LOG_DEBUG << "add reshape to h2d: " << h2d->name() << ", shape "
                          << h2d->output(0)->shape().to_string() << ", dtype "
                          << h2d->output(0)->dtype().name()
                          << ", new name: " << h2d_opr.node()->name() << "\n";
            }
        });
        for (auto&& i : dest_vars)
            dep.add(i);
        if (!varmap.empty()) {
            return cg::replace_vars(dest_vars, varmap);
        }
        return dest_vars;
    }

    void assume_input_n_c(std::vector<TensorShape>& input_tensorshape,
                          int& hint_n, int& hint_c) {
        for (auto& shape : input_tensorshape) {
            if (shape.ndim == 4) {
                hint_n = shape[0];
                if (shape[1] > 4) {
                    hint_c = shape[3];
                } else {
                    hint_c = shape[1];
                }
            }
        }
    }

    //! FIXME: warpaffine may be dynamic, batch may not be one
    SymbolVarArray append_reshape_to_warpaffine(
            const SymbolVarArray& dest_vars,
            std::vector<TensorShape>& input_tensorshape) {
        ThinHashMap<SymbolVar, SymbolVar> varmap;
        cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
            if (auto wa = opr->try_cast_final<opr::WarpAffine>()) {
                if (auto imma =
                            wa->input()[2]
                                    ->owner_opr()
                                    ->try_cast_final<opr::ImmutableTensor>()) {
                    auto param = wa->param();
                    auto wa_opr = opr::WarpAffineForward::make(
                            wa->input()[0], wa->input()[1], wa->input()[2],
                            param, wa->config());
                    auto host_val = imma->host_value();
                    CC_ASSERT(host_val.layout().total_nr_elems() == 2);
                    auto hw_shape_ptr = host_val.ptr<int32_t>();
                    int hint_n, hint_c;
                    assume_input_n_c(input_tensorshape, hint_n, hint_c);
                    TensorShape dst_shape;
                    if (wa->param().format ==
                        opr::WarpAffine::Param::Format::NHWC) {
                        dst_shape = {(size_t)hint_n, (size_t)hw_shape_ptr[0],
                                     (size_t)hw_shape_ptr[1], (size_t)hint_c};
                    } else {
                        CC_ASSERT(wa->param().format ==
                                  opr::WarpAffine::Param::Format::NCHW);
                        dst_shape = {(size_t)hint_n, (size_t)hint_c,
                                     (size_t)hw_shape_ptr[0],
                                     (size_t)hw_shape_ptr[1]};
                    }

                    LOG_WARN << "force warpaffine shape to " << hint_n
                             << ", x, x, " << hint_c << ", this maybe a bug\n";
                    varmap[wa->output(0)] =
                            opr::Reshape::make(wa_opr, dst_shape);
                    LOG_DEBUG << "add reshape to h2d: " << wa->name()
                              << ", shape "
                              << wa->output(0)->shape().to_string()
                              << ", dtype " << wa->output(0)->dtype().name()
                              << "\n";
                }
            }
        });
        for (auto&& i : dest_vars)
            dep.add(i);
        if (!varmap.empty()) {
            return cg::replace_vars(dest_vars, varmap);
        }
        return dest_vars;
    }

    SymbolVarArray fetch_inputs(const VarNodeArray& dest_vars) {
        SymbolVarArray inputs;
        cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
            if (auto h2d = opr->try_cast_final<opr::Host2DeviceCopy>()) {
                inputs.push_back(h2d->output(0));
            }
        });
        for (auto&& i : dest_vars)
            dep.add(i);
        return inputs;
    }

    FunctionType get_func_type(SymbolVarArray inputs, SymbolVarArray outputs) {
        auto arg_types = llvm::to_vector<1>(
                llvm::map_range(inputs, [this](SymbolVar var) -> mlir::Type {
                    return var_to_shaped_type(var.node());
                }));
        auto result_types = llvm::to_vector<1>(
                llvm::map_range(outputs, [this](SymbolVar var) -> mlir::Type {
                    return var_to_shaped_type(var.node());
                }));
        return FunctionType::get(m_context, arg_types, result_types);
    }

    void process_graph(Options options) {
        OpBuilder::InsertionGuard _(m_builder);
        bool use_default_input_map = options.input_map_vec.empty();
        size_t nr_input =
                use_default_input_map ? 1 : options.input_map_vec.size();

        for (size_t idx = 0; idx < nr_input; ++idx) {
            auto graph = m_loader->load(m_load_config, true);
            if (use_default_input_map) {
                std::map<std::string, megdnn::TensorShape> default_input_map;
                replace_h2d_in_new_shape(graph.tensor_map, default_input_map);
            } else {
                replace_h2d_in_new_shape(graph.tensor_map,
                                         options.input_map_vec[idx]);
            }
            SymbolVarArray output_vars = graph.output_var_list;
            std::vector<TensorShape> input_tensorshape;
            output_vars = append_decouple_weight_pack(output_vars);
            output_vars = disable_h2d_mem_fwd(output_vars);
            output_vars = append_reshape_to_h2d(output_vars, input_tensorshape);
            output_vars = append_reshape_to_warpaffine(output_vars,
                                                       input_tensorshape);
            if (options.add_nhwc2nchw_to_input) {
                output_vars =
                        append_nhwc2nchw_to_h2d(output_vars, graph.tensor_map);
            }

            auto cg = output_vars[0].node()->owner_graph();
            cg->options().graph_opt_level = options.graph_opt_level;
            gopt::OptimizeForInferenceOptions opt_for_inference;
            if (options.enable_fuse_conv_bias_nonlinearity) {
                opt_for_inference.enable_fuse_conv_bias_nonlinearity();
                LOG_INFO << "enable_fuse_conv_bias_nonlinearity\n";
            }
            if (options.enable_fuse_conv_bias_nonlinearity_z) {
                opt_for_inference.enable_fuse_conv_bias_with_z();
                LOG_INFO << "enable_fuse_conv_bias_with_z\n";
            }
            if (options.enable_nchw44) {
                opt_for_inference.enable_nchw44();
                LOG_INFO << "enable nchw44\n";
            }
            if (options.enable_nchw44_dot) {
                opt_for_inference.enable_nchw44_dot();
                LOG_INFO << "enable nchw44_dot\n";
            }
            if (options.optimize_for_inference) {
                LOG_DEBUG << "optimize mgb model for inference\n";
                output_vars = mgb::gopt::optimize_for_inference(
                        output_vars, opt_for_inference);
            }
            cg::ComputingGraph::OutputSpec output_spec;
            for (auto&& i : output_vars) {
                output_spec.push_back({i, {}});
            }
            auto seq = cg->compile(output_spec);
            auto inputs = fetch_inputs(seq->get_output_vars());
            auto outputs = cg::to_symbol_var_array(seq->get_output_vars());
            m_builder.setInsertionPointToEnd(m_module.getBody());
            std::string func_name = options.module_name;
            if (!use_default_input_map) {
                func_name += "_" + std::to_string(idx);
            }
            auto func = m_builder.create<FuncOp>(
                    m_builder.getUnknownLoc(), func_name,
                    get_func_type(inputs, outputs));
            Block* entryBlock = func.addEntryBlock();
            m_builder.setInsertionPointToStart(entryBlock);
            CC_ASSERT(entryBlock->getNumArguments() == inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto&& h2d = inputs[i]
                                     .node()
                                     ->owner_opr()
                                     ->cast_final_safe<opr::Host2DeviceCopy>();
                m_var2value[h2d.output(0)] = entryBlock->getArgument(i);
                std::string name = h2d.output(0)->cname();

                for (auto&& i : graph.tensor_map) {
                    if (i.second == h2d.host_data()) {
                        name = i.first;
                        break;
                    }
                }
                func.setArgAttr(i, "mgb.func_arg_name",
                                m_builder.getStringAttr(name));
            }
            std::unordered_map<VarNode*, std::vector<cg::OperatorNodeBase*>>
                    varnode_to_used_opr;
            seq->iter_opr_seq([&](cg::OperatorNodeBase* op) {
                for (auto& inp : op->input()) {
                    varnode_to_used_opr[inp].push_back(op);
                }
                return true;
            });
            cg::ComputeDepOprIter dep(
                    std::bind(&Importer::on_opr, this, std::placeholders::_1,
                              std::placeholders::_2, idx),
                    varnode_to_used_opr);
            seq->iter_opr_seq([&](cg::OperatorNodeBase* op) {
                dep.iter(op);
                return true;
            });
            std::vector<Value> results;
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto&& orig_node = graph.output_var_list[i].node();
                std::string name = orig_node->cname();
                for (auto&& i : graph.output_var_map) {
                    if (i.second.node() == orig_node) {
                        name = i.first;
                        break;
                    }
                }
                func.setResultAttr(i, "mgb.func_result_name",
                                   m_builder.getStringAttr(name));
                results.push_back(m_var2value.at(outputs[i].node()));
            }
            m_builder.create<ReturnOp>(m_builder.getUnknownLoc(), results);
            m_var2value.clear();
        }
    }
    mgb::serialization::GraphLoadConfig m_load_config;
    std::unique_ptr<mgb::serialization::GraphLoader> m_loader;
    mlir::ModuleOp m_module;
    mlir::MLIRContext* m_context;
    mlir::OpBuilder m_builder;
    std::unordered_map<cg::VarNode*, mlir::Value> m_var2value;
    std::unordered_map<std::string, MGB::ParamStorage> m_param_storage;
};

mlir::LogicalResult removeUnusedParam(mlir::ModuleOp module) {
    mlir::PassManager pm(module->getContext());
    pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    return pm.run(module);
}
}  // namespace
mlir::LogicalResult import_mgb(mlir::ModuleOp module, std::string model_path,
                               MGBImporterOptions options, int hako_ver) {
    LOG_DEBUG << "\n\t\t\t Begin Import MBG \t\t\t\n";
    LOG_DEBUG << "load model from " << model_path
              << " with Options:\n\tuse_static_memory_plan="
              << options.use_static_memory_plan
              << "\n\toptimize_for_inference=" << options.optimize_for_inference
              << "\n\tuse_naive_memory_plan=" << options.use_naive_memory_plan
              << "\n\tgraph_opt_level="
              << static_cast<int>(options.graph_opt_level) << "\n";
    Importer imp(module);
    auto result = imp.import_mgb(model_path, options, hako_ver);
    LOG_DEBUG << "\t\t\t End Import MBG \t\t\t\n\n";
    if (failed(result))
        return result;
    return removeUnusedParam(module);
}

mlir::LogicalResult parseInputShapes(std::string s,
                                     mlir::MGB::MGBImporterOptions& options) {
    if (s.empty())
        return mlir::success();
    llvm::SmallVector<llvm::StringRef> shapes_group;
    llvm::SplitString(s, shapes_group, ":");
    for (auto&& shape_str : shapes_group) {
        llvm::SmallVector<llvm::StringRef> shapes;
        llvm::SplitString(shape_str, shapes, ";");
        std::map<std::string, megdnn::TensorShape> input_map;
        for (auto&& shape : shapes) {
            llvm::StringRef nameString, shapeString;
            std::tie(nameString, shapeString) = shape.split("=");
            if (!shapeString.startswith("(") || !shapeString.endswith(")")) {
                llvm::errs() << "invalid shape expression " << shape
                             << " the format of shape expression should be "
                                "tensor_name=(dim0, dim1, dim2)";
                return mlir::failure();
            }
            llvm::SmallVector<llvm::StringRef> dims;
            llvm::SplitString(shapeString.drop_front(1).drop_back(1), dims,
                              ", ");
            if (dims.empty() || dims.size() > megdnn::TensorShape::MAX_NDIM) {
                llvm::errs()
                        << "invalid dnn tensor shape " << shapeString << "\n";
                return mlir::failure();
            }
            megdnn::TensorShape tshape;
            tshape.ndim = dims.size();
            for (size_t i = 0; i < tshape.ndim; ++i) {
                if (dims[i].getAsInteger(0, tshape[i])) {
                    llvm::errs() << "invalid dimension " << dims[i] << "\n";
                    return mlir::failure();
                }
            }
            if (!input_map.emplace(nameString.str(), tshape).second) {
                llvm::errs() << "duplicated tensor name " << nameString << "\n";
                return mlir::failure();
            }
        }
        if (input_map.size() > 0) {
            llvm::outs() << "Model input tensor shape:\n";
            for (auto& it : input_map) {
                llvm::outs() << "   input name= " << it.first << ".   ";
                llvm::outs() << "shape= [";
                for (size_t dim = 0; dim < it.second.ndim; dim++) {
                    if (dim != 0)
                        llvm::outs() << ",";
                    llvm::outs() << it.second.shape[dim];
                }
                llvm::outs() << "]\n";
            }
        }
        options.input_map_vec.push_back(input_map);
    }
    return mlir::success();
}

}  // namespace MGB
}  // namespace mlir

// vim: syntax=cpp.doxygen
