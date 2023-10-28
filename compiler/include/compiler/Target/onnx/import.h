#pragma once

#include <map>
#include <vector>

#include "megdnn/basic_types.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace ONNX {

struct ONNXImporterOptions {
    std::string module_name;
    std::string model_path;
    std::string input_shape_str;
};

mlir::LogicalResult import_onnx(mlir::ModuleOp module, std::string model_path);

}  // namespace ONNX
}  // namespace mlir