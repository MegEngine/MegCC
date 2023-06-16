#pragma once

#include <map>
#include <vector>

#include "megdnn/basic_types.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace MGB {

struct MGBImporterOptions {
    bool use_static_memory_plan = true;
    bool optimize_for_inference = true;
    bool use_naive_memory_plan = false;
    bool enable_nchw44 = false;
    bool enable_nchw44_dot = false;
    bool enable_fuse_conv_bias_nonlinearity = true;
    bool enable_fuse_conv_bias_nonlinearity_z = false;
    bool add_nhwc2nchw_to_input = false;
    bool enable_ioc16 = false;
    bool enable_nchw88 = false;
    int8_t graph_opt_level = 2;
    std::string module_name;
    std::string extern_opr_output_shape;
    std::string extern_opr_output_dtype;
    std::string extern_opr_loader_env;
    std::string extern_opr_loader_path_with_interface;
    std::vector<std::map<std::string, megdnn::TensorShape>> input_map_vec;
};

mlir::LogicalResult import_mgb(
        mlir::ModuleOp module, std::string model_path, MGBImporterOptions options,
        std::vector<uint8_t>* const head);

mlir::LogicalResult parseInputShapes(
        std::string s, mlir::MGB::MGBImporterOptions& options);

}  // namespace MGB
}  // namespace mlir

// vim: syntax=cpp.doxygen
