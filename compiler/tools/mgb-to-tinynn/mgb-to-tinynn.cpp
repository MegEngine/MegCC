#include <fstream>
#include <string>
#include <vector>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
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
#include "compiler/Target/Hako/hako_parse.h"
#include "compiler/Target/MGB/import.h"
#include "compiler/Target/TinyNN/export.h"
using namespace llvm;

cl::opt<std::string> InputFile(
        cl::Positional, cl::Optional, cl::desc("<input megengine cpp model>"));
cl::opt<std::string> OutputDir(
        cl::Positional, cl::Optional,
        cl::desc("<output dir for tinynn model and generated kernels>"));
cl::opt<std::string> dumpDir(
        "dump", cl::Optional,
        cl::desc("<override output dir in json for tinynn "
                 "model and generated kernels>"));
cl::opt<std::string> InputShapes(
        "input-shapes", cl::Optional, cl::desc("modify input shapes"),
        cl::value_desc("name0=(xx0,yy0);name1=(xx1,yy1,zz1)"));
cl::opt<bool> Verbose(
        "verbose", cl::desc("log more detail information when compiler model"));
cl::opt<bool> EnableNchw44("enable_nchw44", cl::desc("enable nchw44 trans"));
cl::opt<bool> EnableNchw44Dot("enable_nchw44_dot", cl::desc("enable nchw44-dot trans"));
cl::opt<bool> MGBFuseKernel("mgb_fuse_kernel", cl::desc("fuse mgb kernel as possible"));
cl::opt<bool> SaveModel("save-model", cl::desc("save model to c"));
cl::opt<bool> Add_nhwc2nchw_to_input(
        "add_nhwc2nchw_to_input", cl::desc("add nhwc2nchw dimshuffle to input"));

cl::opt<std::string> JsonFile(
        "json", cl::Optional, cl::desc("config app by json"),
        cl::value_desc("<path/to/json/file>"));
cl::opt<std::string> GenJsonTemplate(
        "json-template", cl::Optional,
        cl::desc("specify directory stored the output json template"),
        cl::value_desc("path/to/out_dir"));

cl::opt<bool> EnableCompressWeightToFp16(
        "enable_compress_fp16",
        cl::desc("enable compress model weight from fp32 to fp16, enable this "
                 "may effect model precision."));
cl::opt<bool> EnableIoc16("enable_ioc16", cl::desc("enable ioc16 trans"));
cl::opt<bool> EnableNchw88("enable_nchw88", cl::desc("enable nchw88 trans"));

cl::opt<bool> Decrypt(
        "decrypt",
        cl::desc("Only try to convert the input file to the mge format model and save "
                 "it in the ./decryption/<model_name>.mge"));

extern llvm::cl::opt<megcc::KernelGen::Arch> target_arch;
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
                GenKerns::ResizeKernel, 2, {{"imode", "LINEAR"}, {"format", "NHWC"}}};
        m_name2gen["flip"] = {GenKerns::FlipKernel, 2};
        m_name2gen["warp_affine_replicate_linear"] = {
                GenKerns::WarpAffineKernel,
                3,
                {{"imode", "LINEAR"},
                 {"format", "NHWC"},
                 {"border_mode", "REPLICATE"}}};
        m_name2gen["warp_affine_replicate_linear"].flt_param["border_val"] = 0.f;
        m_name2gen["warp_affine_constant_linear"] = {
                GenKerns::WarpAffineKernel,
                4,
                {{"imode", "LINEAR"}, {"format", "NHWC"}, {"border_mode", "CONSTANT"}}};
        m_name2gen["warp_affine_constant_linear"].flt_param["border_val"] = 0.f;
        m_name2gen["rgb2bgr"] = {GenKerns::CvtColorKernel, 2, {{"mode", "RGB2BGR"}}};
        m_name2gen["rgb2yuv"] = {GenKerns::CvtColorKernel, 2, {{"mode", "RGB2YUV"}}};
        m_name2gen["rgb2gray"] = {GenKerns::CvtColorKernel, 2, {{"mode", "RGB2GRAY"}}};
        m_name2gen["yuv2bgr_nv21"] = {
                GenKerns::CvtColorKernel, 2, {{"mode", "YUV2BGR_NV21"}}};
        m_name2gen["gaussian_blur_constant"] = {
                GenKerns::CVGaussianBlur, 2, {{"border_mode", "CONSTANT"}}};
        m_name2gen["gaussian_blur_replicate"] = {
                GenKerns::CVGaussianBlur, 2, {{"border_mode", "REPLICATE"}}};
        m_name2gen["gaussian_blur_reflect"] = {
                GenKerns::CVGaussianBlur, 2, {{"border_mode", "REFLECT"}}};
        m_name2gen["gaussian_blur_reflect_101"] = {
                GenKerns::CVGaussianBlur, 2, {{"border_mode", "REFLECT_101"}}};
    }

    Kerns get_kerns(const std::string& cv_name, megcc::KernelGen::Arch arch) {
        CC_ASSERT(m_name2gen.find(cv_name) != m_name2gen.end())
                << "can not find cv " << cv_name << "\n";
        auto kernel_type = m_name2gen[cv_name].kernel_type;
        auto kernels =
                megcc::KernelGen::KernelPack::GetKernel(kernel_type, target_arch).first;
        {
            auto bare_kernels = megcc::KernelGen::KernelPack::GetKernel(
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

    std::vector<std::string> get_all_cv_name() const {
        std::vector<std::string> res;
        for (auto name2gen : m_name2gen) {
            res.push_back(name2gen.first);
        }
        return res;
    }

private:
    std::unordered_map<std::string, CVConfig> m_name2gen;
};
struct DumpJson {
    struct ModelJson {
        ModelJson() {
            str_options["model_name"] = "";
            str_options_template["model_name"] = std::make_pair(
                    "[Optional], specify the name of the tiny model to be generated.",
                    "model_nchw44");

            str_options["model_path"] = "";
            str_options_template["model_path"] = std::make_pair(
                    "[Required], specify the input model path", "path/to/model");

            str_options["input_shape_str"] = "";
            str_options_template["input_shape_str"] = std::make_pair(
                    "[Optional], modify the input shape",
                    "data=(1,1,384,288):data=(1,1,288,384)");

            str_options["extern_opr_output_shape"] = "";
            str_options_template["extern_opr_output_shape"] = std::make_pair(
                    "[Optional], specific extern opr output shapes",
                    "loader_1=(1,3,5,5);(1,1);(3,3):loader_2=(2,2);(1,1,3,3)");

            str_options["extern_opr_output_dtype"] = "";
            str_options_template["extern_opr_output_dtype"] = std::make_pair(
                    "[Optional], specific extern opr output dtypes. The available "
                    "values are float32, int32, uint8, float16 and int16. Default "
                    "value is float32.",
                    "float32;int32;uint8:float16;int16");

            str_options["extern_opr_loader_env"] = "";
            str_options_template["extern_opr_loader_env"] = std::make_pair(
                    "[Optional], specific extern opr loader path with interface. If "
                    "\"interface\" "
                    "is not provided, using \"mgb_c_opr_init\" default.",
                    "loader_path:interface");

            str_options["extern_opr_loader_path_with_interface"] = "";
            str_options_template["extern_opr_loader_path_with_interface"] =
                    std::make_pair(
                            "[Optional], set ENV for all extern opr loader",
                            "ENV_1=VALUE_1;ENV_2=VALUE_2");

            bool_options["enable_nchw44"] = false;
            bool_options_template["enable_nchw44"] = std::make_pair(
                    "[Optional], whether to enable nchw44 optimization, default false",
                    true);

            bool_options["enable_nchw44_dot"] = false;
            bool_options_template["enable_nchw44_dot"] = std::make_pair(
                    "[Optional], whether to enable nchw44 dot optimization for int8, "
                    "default false",
                    false);

            bool_options["add_nhwc2nchw_to_input"] = false;
            bool_options_template["add_nhwc2nchw_to_input"] = std::make_pair(
                    "[Optional], add nhwc2nchw dimshuffle to input", false);

            bool_options["mgb_fuse_kernel"] = false;
            bool_options_template["mgb_fuse_kernel"] =
                    std::make_pair("[Optional], fuse mgb kernel as possible", false);

            bool_options["enable_compress_fp16"] = false;
            bool_options_template["enable_compress_fp16"] = std::make_pair(
                    "[Optional], whether to enable the optimization of using float16 "
                    "storage to compress the model size",
                    false);

            bool_options["enable_nchw88"] = false;
            bool_options_template["enable_nchw88"] = std::make_pair(
                    "[Optional], whether to enable nchw88 optimization, default false",
                    false);

            bool_options["enable_ioc16"] = false;
            bool_options_template["enable_ioc16"] = std::make_pair(
                    "[Optional], whether to enable optimization using float16 "
                    "calculation, default false",
                    false);
        }
        static ModelJson parse(json::Object& obj) {
            ModelJson res;
            for (auto& kv : res.str_options) {
                auto key = kv.first;
                auto value = obj.getString(key);
                if (key == "model_path")
                    CC_ASSERT(value) << "`model_path' of every model must be specified "
                                        "in given json file\n";
                if (value)
                    res.str_options[key] = value.getValue().str();
            }
            for (auto& kv : res.bool_options) {
                auto key = kv.first;
                auto value = obj.getBoolean(key);
                if (value) {
                    res.bool_options[key] = value.getValue();
                }
            }
            return res;
        }
        std::map<std::string, std::string> str_options;
        std::map<std::string, bool> bool_options;
        std::map<std::string, std::pair<std::string, std::string>> str_options_template;
        std::map<std::string, std::pair<std::string, bool>> bool_options_template;
        std::string to_string() const {
            std::stringstream ss;
            for (auto& kv : str_options) {
                ss << kv.first << ": " << kv.second << "\n";
            }
            for (auto& kv : bool_options) {
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
        ss << "dump " << models.size() << " models to dump_dir:" << dump_dir << "\n";
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
                        dtype_vec.push_back(dtype.getAsString().getValue().str());
                    }
                }
                if (dtype_vec.size() > 0) {
                    res->cv_impl[cv_name] = dtype_vec;
                }
            }
        }
        return res;
    }

    static void gen_json_template(const std::string& out_dir) {
        llvm::json::Object res;
        //! dump dir
        res["dump_dir@"] = llvm::json::Value(
                "[Required], specify the directory where the output kernel and model "
                "are stored");
        res["dump_dir"] = llvm::json::Value("kernel_dir");

        //! model
        auto model_array = llvm::json::Array();
        auto model = llvm::json::Object();
        ModelJson model_json;
        for (auto str_option : model_json.str_options) {
            model[str_option.first + "@"] = llvm::json::Value(
                    model_json.str_options_template.at(str_option.first).first);
            model[str_option.first] = llvm::json::Value(
                    model_json.str_options_template.at(str_option.first).second);
        }
        for (auto bool_option : model_json.bool_options) {
            model[bool_option.first + "@"] = llvm::json::Value(
                    model_json.bool_options_template.at(bool_option.first).first);
            model[bool_option.first] = llvm::json::Value(
                    model_json.bool_options_template.at(bool_option.first).second);
        }
        model_array.push_back(llvm::json::Value(std::move(model)));
        res["model"] = llvm::json::Value(std::move(model_array));

        //! cv
        auto cv_oprs = llvm::json::Object();
        auto&& cv_names = DumpCVHelper().get_all_cv_name();
        for (auto&& cv_name : cv_names) {
            auto cv_opr_dtype = llvm::json::Array();
            cv_opr_dtype.push_back(llvm::json::Value("ui8"));
            //! TODO: Automatically determine if dtype float32 is supported. Hard code
            //! for now.
            if (cv_name.find("resize") != std::string::npos ||
                cv_name.find("gaussian_blur") != std::string::npos)
                cv_opr_dtype.push_back(llvm::json::Value("f32"));
            cv_oprs[cv_name] =
                    llvm::json::Value(llvm::json::Value(std::move(cv_opr_dtype)));
        }
        res["cv@"] = llvm::json::Value(
                "[Optional], specify the cv operator used in non-models (e.g. in pre "
                "and post processing)");
        res["cv"] = llvm::json::Value(std::move(cv_oprs));

        std::string JSONString;
        llvm::raw_string_ostream JSONStream(JSONString);
        JSONStream << llvm::json::Value(std::move(res));

        if (!llvm::sys::fs::exists(out_dir.c_str())) {
            llvm::sys::fs::create_directories(out_dir.c_str());
        } else {
            CC_ASSERT(llvm::sys::fs::is_directory(out_dir.c_str()))
                    << out_dir
                    << "is existed and not a directory, try remove it manually or "
                       "choice another one\n";
        }

        std::error_code EC;
        std::string out_path = out_dir + "/config.json";
        llvm::raw_fd_ostream File(out_path, EC, llvm::sys::fs::OF_Text);
        if (EC) {
            llvm::errs() << "Error opening file: " << EC.message() << "\n";
            CC_ABORT;
        }

        File << JSONString << "\n";
        File.close();
        llvm::outs() << "Json template has been saved into " << out_path << ".\n";
    }
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

static void export_cv_one_dtype(
        mlir::KernelExporter& kernel_exporter, std::string& cv_name,
        std::string& cv_dtype) {
    static DumpCVHelper dump_cv_helper;
    auto kernels = dump_cv_helper.get_kerns(cv_name, target_arch);
    CC_ASSERT(kernels.size() > 0) << "export " << cv_name << "failed";
    auto attr_map = get_attr_map(dump_cv_helper.get_kern_config(cv_name), cv_dtype);
    megcc::CodeGenContext ctx(attr_map);
    std::function<void(std::vector<megcc::KernelGen::KernelObj>&)> reg_dep =
            [&](std::vector<megcc::KernelGen::KernelObj>& deps) {
                for (auto& dep_kern : deps) {
                    kernel_exporter.addInternalKernel(
                            dep_kern.kernel_symbol, "", dep_kern.kernel_body, "", "");
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
    if (Verbose) {
        megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    }

    if (GenJsonTemplate.length() > 0) {
        DumpJson::gen_json_template(GenJsonTemplate);
        return 0;
    }

    std::shared_ptr<DumpJson> dump_info;
    if (JsonFile.length() > 0) {
        dump_info = DumpJson::make(JsonFile.getValue());
        if (dumpDir.length() > 0) {
            dump_info->dump_dir = dumpDir.getValue();
        }
        llvm::outs() << dump_info->to_string();
    } else {
        CC_ASSERT(InputFile.length() > 0);
        if (!Decrypt)
            CC_ASSERT(OutputDir.length() > 0);
        dump_info = std::make_shared<DumpJson>();
        dump_info->dump_dir = OutputDir.getValue();
        DumpJson::ModelJson model_json;
        model_json.str_options["model_name"] = "";
        model_json.str_options["model_path"] = InputFile.getValue();
        model_json.str_options["input_shape_str"] = InputShapes.getValue();
        model_json.bool_options["enable_nchw44"] = EnableNchw44.getValue();
        model_json.bool_options["enable_nchw44_dot"] = EnableNchw44Dot.getValue();
        model_json.bool_options["add_nhwc2nchw_to_input"] =
                Add_nhwc2nchw_to_input.getValue();
        model_json.bool_options["mgb_fuse_kernel"] = MGBFuseKernel.getValue();
        model_json.bool_options["enable_compress_fp16"] =
                EnableCompressWeightToFp16.getValue();
        model_json.bool_options["enable_nchw88"] = EnableNchw88.getValue();
        model_json.bool_options["enable_ioc16"] = EnableIoc16.getValue();
        dump_info->models.push_back(model_json);
    }

    if (Decrypt) {
        for (auto model : dump_info->models) {
            std::string model_path = model.str_options.at("model_path");
            size_t found = model_path.find_last_of('/');
            std::string file_name =
                    model_path.substr((found == std::string::npos) ? 0 : found + 1);
            std::ifstream fin(model_path, std::ios::in | std::ios::binary);
            std::vector<uint8_t> model_buffer(std::istreambuf_iterator<char>(fin), {});
            fin.close();

            megcc::DecryptedModel&& res = megcc::parse_model(model_buffer);
            auto& mdl_model_buffer = res.model;
            megcc::EncryptionType enc_type = res.enc_type;
            if (enc_type == megcc::EncryptionType::NONE) {
                if (JsonFile.length() > 0) {
                    llvm::outs()
                            << "Warning: " << file_name << " NO need to decryption.\n";
                } else {
                    CC_ASSERT(0) << file_name << " NO need to decryption.\n";
                }
            }
            llvm::sys::fs::create_directories("./decryption", true);
            std::string out_name = "./decryption/" + file_name + ".mge";
            std::ofstream fout(out_name, std::ios::out | std::ios::binary);
            fout.write(
                    reinterpret_cast<char*>(mdl_model_buffer.data()),
                    mdl_model_buffer.size());
            fout.close();
        }
        llvm::outs() << "Decrypted model has been saved into ./decryption\n";
    } else {
        auto dump_dir = dump_info->dump_dir;
        if (!llvm::sys::fs::exists(dump_dir.c_str())) {
            llvm::sys::fs::create_directories(dump_dir.c_str());
        } else {
            CC_ASSERT(llvm::sys::fs::is_directory(dump_dir.c_str()))
            "output: " << dump_dir
                       << "is existed and not a directory, try remove it manually or "
                          "choice another one";
        }

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
            options.enable_nchw88 = model.bool_options.at("enable_nchw88");
            options.enable_ioc16 = model.bool_options.at("enable_ioc16");
            options.extern_opr_output_shape =
                    model.str_options.at("extern_opr_output_shape");
            options.extern_opr_output_dtype =
                    model.str_options.at("extern_opr_output_dtype");
            options.extern_opr_loader_env =
                    model.str_options.at("extern_opr_loader_env");
            options.extern_opr_loader_path_with_interface =
                    model.str_options.at("extern_opr_loader_path_with_interface");
            bool model_mgb_fuse_kernel = model.bool_options.at("mgb_fuse_kernel");

            if (failed(parseInputShapes(
                        model.str_options["input_shape_str"], options))) {
                return -1;
            }
            auto model_name = model.str_options.at("model_name");
            if (model_name.size() > 0) {
                options.module_name = model_name;
            } else {
                llvm::SmallVector<llvm::StringRef> dir_names;
                llvm::SplitString(model.str_options.at("model_path"), dir_names, "/");
                llvm::SmallVector<llvm::StringRef> names;
                llvm::SplitString(dir_names[dir_names.size() - 1], names, ".");
                options.module_name = names[0].str();
            }
            auto model_input = model.str_options.at("model_path");
            llvm::outs() << "Import mgb/mge model from " << model_input << "\n";
            mlir::OwningOpRef<mlir::ModuleOp> mod =
                    mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
            std::vector<uint8_t> hako_head;
            auto status =
                    mlir::MGB::import_mgb(mod.get(), model_input, options, &hako_head);
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

            std::string out_model_path = dump_dir + "/" + options.module_name + ".tiny";
            mlir::export_tinynn_model(
                    mod.get(), out_model_path, SaveModel, kernel_exporter,
                    model.bool_options.at("enable_compress_fp16"));

            //! In some cases, users need the headers added when using hako encryption,
            //! so they are saved here.
            //! Before saving, the size of the model in the header needs to be
            //! modified to the size of the `tiny` model (the original size was the
            //! size of the `mge` model)
            if (hako_head.size() >= sizeof(int)) {
                std::ifstream fin(out_model_path, std::ios::binary);
                fin.seekg(0, fin.end);
                int size = static_cast<int>(fin.tellg());
                fin.close();

                *reinterpret_cast<int*>(
                        hako_head.data() + hako_head.size() - sizeof(int)) = size;

                std::string model_name(llvm::sys::path::filename(out_model_path).str());
                std::string head_path = dump_dir + "/" + model_name + "_head.bin";
                std::ofstream fout(head_path, std::ios::out | std::ios::binary);
                fout.write(
                        reinterpret_cast<const char*>(hako_head.data()),
                        hako_head.size());
                fout.close();
            }

            llvm::outs() << "Mgb/mge model convert to tinynn model "
                         << options.module_name << " done.\n";
        }
        export_cv_opr(kernel_exporter, dump_info->cv_impl);
        kernel_exporter.write(dump_dir);
        llvm::outs() << "Mgb/mge model convert to tinynn kernel done.\n";
    }
    return 0;
}

// vim: syntax=cpp.doxygen
