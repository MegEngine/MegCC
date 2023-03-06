/**
 * \file compiler/tools/mgb-runner/main.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <map>
#include "compiler/Common/Version.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/serialization/serializer.h"

using namespace llvm;
using namespace mgb;

cl::opt<std::string> InputFile(
        cl::Positional, cl::Required, cl::desc("<input mgb model>"));
cl::opt<std::string> OutputDir(cl::Positional, cl::Required, cl::desc("<output dir>"));
cl::opt<bool> verboseOpt("verbose", cl::Optional, cl::desc("enable verbose output"));
cl::opt<bool> EnableNchw44(
        "enable_nchw44", cl::Optional, cl::desc("enable nchw44 trans"));
cl::opt<bool> BinDumpOpt("bin_dump", cl::Optional, cl::desc("enable bin_dump output"));
cl::opt<bool> Add_nhwc2nchw_to_input(
        "add_nhwc2nchw_to_input", cl::desc("add nhwc2nchw dimshuffle to input"));

cl::opt<std::string> InputShapes(
        "input-shapes", cl::Required, cl::desc("modify input shapes"),
        cl::value_desc("name0=(xx0,yy0);name1=(xx1,yy1,zz1)"));

cl::opt<std::string> InputData(
        "input-data", cl::Required, cl::desc("feed input data with raw binary"),
        cl::value_desc("img=filename0;lmk=filename1"));

SymbolVarArray append_nhwc2nchw_to_h2d(const SymbolVarArray& dest_vars) {
    ThinHashMap<SymbolVar, SymbolVar> varmap;
    cg::DepOprIter dep([&](cg::OperatorNodeBase* opr) {
        if (auto h2d = opr->try_cast_final<opr::Host2DeviceCopy>()) {
            if (h2d->output(0)->shape().ndim != 4) {
                return;
            }

            auto param = h2d->param();
            auto host_data = h2d->host_data();
            auto old_shape = host_data->shape();
            auto new_shape = TensorShape(
                    {old_shape[0], old_shape[2], old_shape[3], old_shape[1]});
            std::shared_ptr<HostTensorND> host_data_new =
                    std::make_shared<HostTensorND>(
                            host_data->comp_node(), new_shape, host_data->dtype());
            auto h2d_opr = opr::Host2DeviceCopy::make(
                    *h2d->owner_graph(), host_data_new, param, h2d->config());
            varmap[h2d->output(0)] = opr::Dimshuffle::make(h2d_opr, {0, 3, 1, 2});
        }
    });
    for (auto&& i : dest_vars)
        dep.add(i);
    if (!varmap.empty()) {
        return cg::replace_vars(dest_vars, varmap);
    }
    return dest_vars;
}

bool parseInputShapes(
        std::string s,
        std::vector<std::map<std::string, megdnn::TensorShape>>& map_vec) {
    if (s.empty())
        return false;
    llvm::SmallVector<llvm::StringRef> shapes_group;
    llvm::SplitString(s, shapes_group, ":");
    for (auto&& shape_str : shapes_group) {
        llvm::SmallVector<llvm::StringRef> shapes;
        llvm::SplitString(shape_str, shapes, ";");
        std::map<std::string, megdnn::TensorShape> map;
        for (auto&& shape : shapes) {
            llvm::StringRef nameString, shapeString;
            std::tie(nameString, shapeString) = shape.split("=");
            if (!shapeString.startswith("(") || !shapeString.endswith(")")) {
                llvm::errs() << "invalid shape expression " << shape
                             << " the format of shape expression should be "
                                "tensor_name=(dim0, dim1, dim2)";
                return false;
            }
            llvm::SmallVector<llvm::StringRef> dims;
            llvm::SplitString(shapeString.drop_front(1).drop_back(1), dims, ", ");
            if (dims.empty() || dims.size() > megdnn::TensorShape::MAX_NDIM) {
                llvm::errs() << "invalid dnn tensor shape " << shapeString << "\n";
                return false;
            }
            megdnn::TensorShape tshape;
            tshape.ndim = dims.size();
            for (size_t i = 0; i < tshape.ndim; ++i) {
                if (dims[i].getAsInteger(0, tshape[i])) {
                    llvm::errs() << "invalid dimension " << dims[i] << "\n";
                    return false;
                }
            }
            if (!map.emplace(nameString.str(), tshape).second) {
                llvm::errs() << "duplicated tensor name " << nameString << "\n";
                return false;
            }
        }
        if (map.size() > 0) {
            llvm::outs() << "Model input tensor shape:\n";
            for (auto& it : map) {
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
        map_vec.push_back(map);
    }
    return true;
}

bool parseInputData(
        std::string s, std::vector<std::map<std::string, std::string>>& map_vec) {
    if (s.empty())
        return false;
    llvm::SmallVector<llvm::StringRef> data_group;
    llvm::SplitString(s, data_group, ":");
    for (auto&& data_str : data_group) {
        llvm::SmallVector<llvm::StringRef> datas;
        llvm::SplitString(data_str, datas, ";");
        std::map<std::string, std::string> map;
        for (auto&& data : datas) {
            llvm::StringRef nameString, fileString;
            std::tie(nameString, fileString) = data.split("=");
            map[nameString.str()] = fileString.str();
        }
        if (map.size() > 0) {
            llvm::outs() << "Model input data:\n";
            for (auto& it : map) {
                llvm::outs() << "   input name= " << it.first << ".   ";
                llvm::outs() << "file= " << it.second << "\n";
            }
        }
        map_vec.push_back(map);
    }
    return true;
}

void run(const serialization::GraphLoader::LoadResult& graph, bool verbose) {
    auto output_vars = graph.output_var_list;
    output_vars = mgb::gopt::optimize_for_inference(output_vars);
    size_t nr_output = output_vars.size();
    cg::ComputingGraph::OutputSpec output_spec;
    HostTensorND output[nr_output];
    std::vector<std::map<std::string, megdnn::TensorShape>> input_map_vec;
    if (!parseInputShapes(InputShapes.getValue(), input_map_vec)) {
        mgb_assert(0, "parseInputShapes error %s\n", InputShapes.getValue().c_str());
        return;
    }
    std::vector<std::map<std::string, std::string>> data_map_vec;
    if (!parseInputData(InputData.getValue(), data_map_vec)) {
        mgb_assert(0, "parseInputData error\n");
        return;
    }
    auto cg = graph.output_var_list[0].node()->owner_graph();
    gopt::OptimizeForInferenceOptions opt_for_inference;
    opt_for_inference.enable_fuse_conv_bias_nonlinearity();
    if (EnableNchw44) {
        opt_for_inference.enable_nchw44();
    }
    if (Add_nhwc2nchw_to_input) {
        output_vars = append_nhwc2nchw_to_h2d(output_vars);
    }
    output_vars = mgb::gopt::optimize_for_inference(output_vars, opt_for_inference);
    for (int i = 0; i < output_vars.size(); ++i) {
        auto&& opr = output_vars[i];
        output_spec.push_back({opr, [&output, i](const DeviceTensorND& res) {
                                   output[i].copy_from(res).sync();
                               }});
    }
    auto func = cg->compile(output_spec);
    int inst_cnt = 0;
    for (auto& data_map : data_map_vec) {
        auto& input_map = input_map_vec[inst_cnt];
        for (auto&& tensor_map : graph.tensor_map) {
            auto name = tensor_map.first;
            auto tensor = tensor_map.second;
            auto shape = input_map[name];
            auto data_path = data_map[name];
            TensorLayout layout(shape, tensor->dtype(), tensor->format());
            llvm::outs() << "set layout " << layout.to_string() << "\n";
            std::unique_ptr<serialization::InputFile> data_file =
                    serialization::InputFile::make_fs(data_path.c_str());
            data_file->rewind();
            data_file->read_into_tensor(*tensor, layout);
        }

        if (verbose) {
            TextOprIODump _(cg);
            func->execute().wait();
        } else if (BinDumpOpt) {
            BinaryOprIODump _(cg, "bin_dump/");
            func->execute().wait();
        } else {
            func->execute().wait();
        }
        for (int i = 0; i < output_vars.size(); ++i) {
            auto out_path = OutputDir + "/" + output_vars[i].node()->name() + "_" +
                            std::to_string(inst_cnt);
            std::unique_ptr<serialization::OutputFile> output_file =
                    serialization::OutputFile::make_fs(out_path.c_str());
            if (output[i].dtype() == dtype::Float32()) {
                output_file->write(
                        output[i].ptr<float>(), output[i].layout().span().high_byte);
            } else if (output[i].dtype() == dtype::Uint8()) {
                output_file->write(
                        output[i].ptr<uint8_t>(), output[i].layout().span().high_byte);
            } else if (output[i].dtype() == dtype::Int8()) {
                output_file->write(
                        output[i].ptr<int8_t>(), output[i].layout().span().high_byte);
            } else {
                mgb_assert(
                        output[i].dtype() == dtype::Int32(),
                        "invalid output type (should be float32, int8, uint8 or "
                        "int32)");
                output_file->write(
                        output[i].ptr<int>(), output[i].layout().span().high_byte);
            }
        }
        ++inst_cnt;
    }
}

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    cl::ParseCommandLineOptions(argc, argv);
    std::string model_path = InputFile.getValue();
    std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_fs(model_path.c_str());
    auto format = serialization::GraphLoader::identify_graph_dump_format(*inp_file);
    mgb_assert(format.valid(), "invalid model: unknown model format");
    auto loader = serialization::GraphLoader::make(std::move(inp_file), format.val());
    serialization::GraphLoadConfig load_config;
    auto result = loader->load(load_config, false);
    run(result, verboseOpt.getValue());
    return 0;
}

// vim: syntax=cpp.doxygen
