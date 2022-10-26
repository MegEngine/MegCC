/**
 * \file compiler/tools/tinynn-exporter/tinynn-exporter.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Conversion/MGBToKernel/MGBToKernel.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Dialect/MGB/Transforms/Passes.h"
#include "compiler/KernelGen/KernelGen.h"
#include "compiler/Target/MGB/import.h"
#include "compiler/Target/TinyNN/export.h"

using namespace llvm;

cl::opt<std::string> InputFile(cl::Positional, cl::Optional,
                               cl::desc("<input megengine cpp model>"));
cl::opt<std::string> OutputDir(
        cl::Positional, cl::Optional,
        cl::desc("<output dir for tinynn model and generated kernels>"));
cl::opt<std::string> dumpDir("dump", cl::Optional,
                             cl::desc("<override output dir in json for tinynn "
                                      "model and generated kernels>"));
cl::opt<std::string> InputShapes(
        "input-shapes", cl::Optional, cl::desc("modify input shapes"),
        cl::value_desc("name0=(xx0,yy0);name1=(xx1,yy1,zz1)"));
cl::opt<bool> Verbose(
        "verbose", cl::desc("log more detail information when compiler model"));
cl::opt<bool> EnableNchw44("enable_nchw44", cl::desc("enable nchw44 trans"));
cl::opt<bool> EnableNchw44Dot("enable_nchw44_dot",
                              cl::desc("enable nchw44-dot trans"));
cl::opt<bool> MGBFuseKernel("mgb_fuse_kernel",
                            cl::desc("fuse mgb kernel as possible"));
cl::opt<bool> SaveModel("save-model", cl::desc("save model to c"));
cl::opt<bool> Add_nhwc2nchw_to_input(
        "add_nhwc2nchw_to_input",
        cl::desc("add nhwc2nchw dimshuffle to input"));

cl::opt<std::string> JsonFile("json", cl::Optional,
                              cl::desc("config app by json"),
                              cl::value_desc("<path/to/json/file>"));

cl::opt<bool> EnableCompressWeightToFp16(
        "enable_compress_fp16",
        cl::desc("enable compress model weight from fp32 to fp16, enable this "
                 "may effect model precision."));

extern llvm::cl::opt<megcc::KernelGen::Arch> target_arch;
struct DumpJson {
    struct ModelJson {
        ModelJson() {
            str_options["model_name"] = "";
            str_options["model_path"] = "";
            str_options["input_shape_str"] = "";
            bool_options["enable_nchw44"] = false;
            bool_options["enable_nchw44_dot"] = false;
            bool_options["add_nhwc2nchw_to_input"] = false;
            bool_options["mgb_fuse_kernel"] = false;
            bool_options["enable_compress_fp16"] = false;
            int_options["hako_ver"] = 0;
        }
        static ModelJson parse(json::Object& obj) {
            ModelJson res;
            for (auto& kv : res.str_options) {
                auto key = kv.first;
                auto value = obj.getString(key);
                CC_ASSERT(value)
                        << "need models/model/" << key << " string value\n";
                res.str_options[key] = value.getValue().str();
            }
            for (auto& kv : res.bool_options) {
                auto key = kv.first;
                auto value = obj.getBoolean(key);
                if (value) {
                    res.bool_options[key] = value.getValue();
                }
            }
            for (auto& kv : res.int_options) {
                auto key = kv.first;
                auto value = obj.getInteger(key);
                if (value) {
                    res.int_options[key] = value.getValue();
                }
            }
            return res;
        }
        std::map<std::string, std::string> str_options;
        std::map<std::string, bool> bool_options;
        std::map<std::string, int> int_options;
        std::string to_string() const {
            std::stringstream ss;
            for (auto& kv : str_options) {
                ss << kv.first << ": " << kv.second << "\n";
            }
            for (auto& kv : bool_options) {
                ss << kv.first << ": " << kv.second << "\n";
            }
            for (auto& kv : int_options) {
                ss << kv.first << ": " << kv.second << "\n";
            }
            return ss.str();
        }
    };

    std::string dump_dir;
    std::vector<ModelJson> models;
    std::map<std::string, std::vector<std::string>> cv_impl;

    std::string to_string() const {
        std::stringstream ss;
        ss << "dump " << models.size() << " models to dump_dir:" << dump_dir
           << "\n";
        for (auto& model : models) {
            ss << "{\n" << model.to_string() << "}\n";
        }
        ss << "cv_impl:\n";
        for (auto& kv : cv_impl) {
            ss << kv.first << ": [";
            for (auto& dtype : kv.second) {
                ss << dtype << ", ";
            }
            ss << "]\n ";
        }
        return ss.str();
    }

    static std::shared_ptr<DumpJson> make(std::string path) {
        auto res = std::make_shared<DumpJson>();
        auto buffer = MemoryBuffer::getFile(path, true);
        CC_ASSERT(buffer) << "can not open json file " << path << "\n";
        auto json_parse = json::parse((*buffer)->getBuffer());
        CC_ASSERT(json_parse);
        auto json_obj = json_parse->getAsObject();
        auto dump_dir = json_obj->getString("dump_dir");
        CC_ASSERT(dump_dir) << "need dump_dir key\n";
        res->dump_dir = dump_dir.getValue().str();
        auto model_list = json_obj->getArray("models");
        if (model_list) {
            for (auto& model : *model_list) {
                auto model_dict = model.getAsObject();
                CC_ASSERT(model_dict) << "models/model must be dict\n";
                ModelJson model_json = ModelJson::parse(*model_dict);
                res->models.push_back(model_json);
            }
        }
        auto cv_obj = json_obj->getObject("cv");
        if (cv_obj) {
            for (auto& cv_obj : *cv_obj) {
                auto cv_name = cv_obj.getFirst().str();
                auto cv_dtypes = cv_obj.getSecond().getAsArray();
                std::vector<std::string> dtype_vec;
                if (cv_dtypes) {
                    for (auto& dtype : *cv_dtypes) {
                        dtype_vec.push_back(
                                dtype.getAsString().getValue().str());
                    }
                }
                if (dtype_vec.size() > 0) {
                    res->cv_impl[cv_name] = dtype_vec;
                }
            }
        }
        return res;
    }
};

class DumpCVHelper {
public:
    using Kerns = std::vector<const megcc::KernelGen::KernelFunc*>;
    using GenKerns = megcc::KernelGen::KernelPack::KernType;
    struct CVConfig {
        GenKerns kernel_type;
        int nr_operands;
        std::map<std::string, std::string> str_param;
        std::map<std::string, float> flt_param;
    };

    DumpCVHelper() {
        m_name2gen["transpose"] = {GenKerns::CVTransposeKernel, 2};
        m_name2gen["roicopy"] = {GenKerns::RoiCopyKernel, 2};
        m_name2gen["rotate"] = {GenKerns::RotateKernel, 2};
        m_name2gen["resize_linear"] = {
                GenKerns::ResizeKernel,
                2,
                {{"imode", "LINEAR"}, {"format", "NHWC"}}};
        m_name2gen["flip"] = {GenKerns::FlipKernel, 2};
        m_name2gen["warp_affine_replicate_linear"] = {
                GenKerns::WarpAffineKernel,
                3,
                {{"imode", "LINEAR"},
                 {"format", "NHWC"},
                 {"border_mode", "REPLICATE"}}};
        m_name2gen["warp_affine_replicate_linear"].flt_param["border_val"] =
                0.f;
        m_name2gen["warp_affine_constant_linear"] = {
                GenKerns::WarpAffineKernel,
                4,
                {{"imode", "LINEAR"},
                 {"format", "NHWC"},
                 {"border_mode", "CONSTANT"}}};
        m_name2gen["warp_affine_constant_linear"].flt_param["border_val"] = 0.f;
        m_name2gen["rgb2bgr"] = {
                GenKerns::CvtColorKernel, 2, {{"mode", "RGB2BGR"}}};
        m_name2gen["rgb2yuv"] = {
                GenKerns::CvtColorKernel, 2, {{"mode", "RGB2YUV"}}};
        m_name2gen["rgb2gray"] = {
                GenKerns::CvtColorKernel, 2, {{"mode", "RGB2GRAY"}}};
        m_name2gen["yuv2bgr_nv21"] = {
                GenKerns::CvtColorKernel, 2, {{"mode", "YUV2BGR_NV21"}}};
    }

    Kerns get_kerns(const std::string& cv_name, megcc::KernelGen::Arch arch) {
        CC_ASSERT(m_name2gen.find(cv_name) != m_name2gen.end())
                << "can not find cv " << cv_name << "\n";
        auto kernel_type = m_name2gen[cv_name].kernel_type;
        auto kernels = megcc::KernelGen::KernelPack::GetKernel(kernel_type,
                                                               target_arch)
                               .first;
        {
            auto bare_kernels =
                    megcc::KernelGen::KernelPack::GetKernel(
                            kernel_type, megcc::KernelGen::Arch::BAREMETAL)
                            .first;
            for (auto x : bare_kernels) {
                kernels.push_back(x);
            }
        }
        return kernels;
    }

    CVConfig get_kern_config(const std::string& cv_name) {
        CC_ASSERT(m_name2gen.find(cv_name) != m_name2gen.end())
                << "not support cv " << cv_name << "\n";
        return m_name2gen[cv_name];
    }

private:
    std::unordered_map<std::string, CVConfig> m_name2gen;
};

static inline std::unordered_map<std::string, megcc::CCAttr> get_attr_map(
        const DumpCVHelper::CVConfig& config, const std::string& dtype) {
    std::unordered_map<std::string, megcc::CCAttr> attr_map;
    attr_map["nr_operands"] = megcc::CCAttr(config.nr_operands);
    for (int i = 0; i < config.nr_operands; ++i) {
        megcc::CCOperand operand;
        operand.dtype = dtype;
        attr_map["operand:" + std::to_string(i)] = megcc::CCAttr(operand);
    }
    for (auto& kv : config.str_param) {
        attr_map[kv.first] = megcc::CCAttr(kv.second);
    }
    for (auto& kv : config.flt_param) {
        attr_map[kv.first] = megcc::CCAttr(kv.second);
    }
    return attr_map;
}

static void export_cv_one_dtype(mlir::KernelExporter& kernel_exporter,
                                std::string& cv_name, std::string& cv_dtype) {
    static DumpCVHelper dump_cv_helper;
    auto kernels = dump_cv_helper.get_kerns(cv_name, target_arch);
    CC_ASSERT(kernels.size() > 0) << "export " << cv_name << "failed";
    auto attr_map =
            get_attr_map(dump_cv_helper.get_kern_config(cv_name), cv_dtype);
    megcc::CodeGenContext ctx(attr_map);
    std::function<void(std::vector<megcc::KernelGen::KernelObj>&)> reg_dep =
            [&](std::vector<megcc::KernelGen::KernelObj>& deps) {
                for (auto& dep_kern : deps) {
                    kernel_exporter.addInternalKernel(dep_kern.kernel_symbol,
                                                      "", dep_kern.kernel_body,
                                                      "", "");
                    reg_dep(dep_kern.kernel_dep);
                }
            };
    for (auto kernel : kernels) {
        if (kernel->IsCVAvailable(&ctx)) {
            auto kern_sym = kernel->GetCVKernelSymbol(&ctx);
            auto sig = kernel->GetCVKernelSignature(&ctx);
            auto body = kernel->GetCVKernelBody(&ctx);
            auto deps = kernel->GetDependInternalSymbol(&ctx);
            reg_dep(deps);
            kernel_exporter.addCVKernel(kern_sym, sig, body);
            return;
        }
    }
    CC_ASSERT(0) << "no usable kernel for " << cv_name << "\n";
}

static void export_cv_opr(
        mlir::KernelExporter& kernel_exporter,
        const std::map<std::string, std::vector<std::string>>& cv_impl) {
    for (auto& kv : cv_impl) {
        auto cv_name = kv.first;
        auto dtypes = kv.second;
        for (auto& dtype : dtypes) {
            export_cv_one_dtype(kernel_exporter, cv_name, dtype);
        }
    }
}

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv);
    std::shared_ptr<DumpJson> dump_info;
    if (JsonFile.length() > 0) {
        dump_info = DumpJson::make(JsonFile.getValue());
        if (dumpDir.length() > 0) {
            dump_info->dump_dir = dumpDir.getValue();
        }
        llvm::outs() << dump_info->to_string();
    } else {
        CC_ASSERT(InputFile.length() > 0);
        CC_ASSERT(OutputDir.length() > 0);
        dump_info = std::make_shared<DumpJson>();
        dump_info->dump_dir = OutputDir.getValue();
        DumpJson::ModelJson model_json;
        model_json.str_options["model_name"] = "";
        model_json.str_options["model_path"] = InputFile.getValue();
        model_json.str_options["input_shape_str"] = InputShapes.getValue();
        model_json.bool_options["enable_nchw44"] = EnableNchw44.getValue();
        model_json.bool_options["enable_nchw44_dot"] =
                EnableNchw44Dot.getValue();
        model_json.bool_options["add_nhwc2nchw_to_input"] =
                Add_nhwc2nchw_to_input.getValue();
        model_json.bool_options["mgb_fuse_kernel"] = MGBFuseKernel.getValue();
        model_json.bool_options["enable_compress_fp16"] =
                EnableCompressWeightToFp16.getValue();
        dump_info->models.push_back(model_json);
    }
    auto dump_dir = dump_info->dump_dir;
    mlir::KernelExporter kernel_exporter;
    for (auto model : dump_info->models) {
        mlir::MLIRContext ctx;
        mlir::MGB::MGBImporterOptions options;
        options.graph_opt_level = 2;
        options.use_static_memory_plan = false;
        options.enable_nchw44 = model.bool_options.at("enable_nchw44");
        options.enable_nchw44_dot = model.bool_options.at("enable_nchw44_dot");
        options.add_nhwc2nchw_to_input =
                model.bool_options.at("add_nhwc2nchw_to_input");
        bool model_mgb_fuse_kernel = model.bool_options.at("mgb_fuse_kernel");
        if (Verbose) {
            megcc::SetLogLevel(megcc::LogLevel::DEBUG);
        }
        if (failed(parseInputShapes(model.str_options["input_shape_str"],
                                    options))) {
            return -1;
        }
        auto model_name = model.str_options.at("model_name");
        if (model_name.size() > 0) {
            options.module_name = model_name;
        } else {
            llvm::SmallVector<llvm::StringRef> dir_names;
            llvm::SplitString(InputFile, dir_names, "/");
            llvm::SmallVector<llvm::StringRef> names;
            llvm::SplitString(dir_names[dir_names.size() - 1], names, ".");
            options.module_name = names[0].str();
        }
        auto model_input = model.str_options.at("model_path");
        llvm::outs() << "Import mgb/mge model from " << model_input << "\n";
        mlir::OwningOpRef<mlir::ModuleOp> mod =
                mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
        auto status = mlir::MGB::import_mgb(mod.get(), model_input, options
        , model.int_options["hako_ver"]);
        if (mlir::failed(status)) {
            llvm::outs() << "import megengine model failed\n";
            return -1;
        }
        mlir::PassManager pm(&ctx);
        if (model_mgb_fuse_kernel) {
            pm.addNestedPass<mlir::FuncOp>(mlir::createMGBFuseKernelPass());
        }
        pm.addPass(mlir::createMGBToKernelPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::createMemoryForwardingPass());
        pm.addPass(mlir::createKernelMaterializationPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::createStaticMemoryPlanningPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::createKernelFinalCleanPass());
        //! Now all the memory is allocated in runtime, the Deallocation
        //! instruction is not used.
        // pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());
        pm.addNestedPass<mlir::FuncOp>(
                mlir::bufferization::createFinalizingBufferizePass());
        llvm::outs() << "Apply createMGBToKernelPass and "
                        "createKernelMaterializationPass to the dialect.\n";
        if (failed(pm.run(mod.get()))) {
            return -1;
        }
        llvm::outs() << "Export tinynn model and kernel to dir " << dump_dir
                     << "\n";
        mlir::export_tinynn_model(
                mod.get(), dump_dir + "/" + options.module_name + ".tiny",
                SaveModel, kernel_exporter,
                model.bool_options.at("enable_compress_fp16"));
        llvm::outs() << "Mgb/mge model convert to tinynn model "
                     << options.module_name << " done.\n";
    }
    export_cv_opr(kernel_exporter, dump_info->cv_impl);
    kernel_exporter.write(dump_dir);
    llvm::outs() << "Mgb/mge model convert to tinynn kernel done.\n";
    return 0;
}

// vim: syntax=cpp.doxygen
