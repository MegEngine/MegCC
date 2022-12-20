/**
 * \file
 * compiler/tools/kernel_exporter/config_attr.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "config_attr.h"
#include <map>
#include "compiler/Common/TContext.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "megbrain/common.h"
#include "megbrain/reflection.h"
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/cv.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/imgproc.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "megdnn/oprs/nn_int.h"
#include "utils.h"

#define megcore_check(expr)                                            \
    do {                                                               \
        megcoreStatus_t _err = (expr);                                 \
        if (_err != megcoreSuccess) {                                  \
            fprintf(stderr, "mgb failed : line=%d %s:%d\n", (int)_err, \
                    __FILE__, __LINE__);                               \
            abort();                                                   \
        }                                                              \
    } while (0)

namespace {
#define DEFINE_DNNPARAM2STR(cls)                             \
    std::string dnnparam_2_str(cls value) {                  \
        return mgb::reflection::nameOfEnumValue<cls>(value); \
    }

DEFINE_DNNPARAM2STR(ConvParam::Format)
DEFINE_DNNPARAM2STR(ConvParam::Sparse)
DEFINE_DNNPARAM2STR(ConvParam::Mode)
DEFINE_DNNPARAM2STR(megdnn::ElemwiseForward::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::ElemwiseMultiType::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::PoolingForward::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::MatrixMulForward::Param::Format)
DEFINE_DNNPARAM2STR(megdnn::MatrixMulForward::Param::ComputeMode)
DEFINE_DNNPARAM2STR(megdnn::Reduce::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::Reduce::Param::DataType)
DEFINE_DNNPARAM2STR(megdnn::WarpPerspectiveForward::Param::BorderMode)
DEFINE_DNNPARAM2STR(megdnn::WarpPerspectiveForward::Param::InterpolationMode)
DEFINE_DNNPARAM2STR(megdnn::CvtColor::Param::Mode)
DEFINE_DNNPARAM2STR(megdnn::Argsort::Param::Order)
#undef DEFINE_DNNPARAM2STR

int get_int() {
    llvm::outs() << "please input a int number"
                 << "\n";
    std::string ret;
    std::string num;
    std::cin >> num;
    int n = num.size();
    for (int i = 0; i < n; i++) {
        if (num[i] >= '0' && num[i] <= '9') {
            ret.push_back(num[i]);
        }
    }
    llvm::outs() << "input: " << stoi(ret) << "\n";

    return stoi(ret);
}

float get_float() {
    llvm::outs() << "please input a float number"
                 << "\n";
    std::string ret;
    std::string num;
    std::cin >> num;
    int n = num.size();
    for (int i = 0; i < n; i++) {
        if ((num[i] >= '0' && num[i] <= '9') || num[i] == '.') {
            ret.push_back(num[i]);
        }
    }
    llvm::outs() << "input: " << stof(ret) << "\n";

    return stof(ret);
}

std::string support_map_to_msg(const std::map<int, std::string>& m) {
    std::string msg = "\n";
    for (const auto& i : m) {
        msg += std::to_string(i.first);
        msg += " = ";
        msg += i.second;
        msg += ",\n";
    }

    return msg;
}

std::pair<std::string, std::map<int, std::string>> support_dtype() {
    std::map<int, std::string> enum2dtype{
            {0, "f32"}, {1, "si8"},          {2, "i32"},          {3, "i16"},
            {4, "ui8"}, {5, "qsi8<scale:>"}, {6, "qsi32<scale:>"}};

    return {support_map_to_msg(enum2dtype), enum2dtype};
}

std::pair<std::string, std::map<int, std::string>> support_format() {
    std::map<int, std::string> format2enum{
            {0, "NCHW"}, {7, "NCHW44"}, {8, "NCHW44_DOT"}};

    return {support_map_to_msg(format2enum), format2enum};
}

}  // namespace

namespace megcc {
namespace exporter {
#define FILL_MAP(_map_name, _parm_name, _attr_name) \
    _map_name[#_attr_name] = CCAttr(_parm_name._attr_name)
#define FILL_MAP_EX(_map_name, _parm_name, _attr_name, _helper_fun) \
    _map_name[#_attr_name] = CCAttr(_helper_fun(_parm_name._attr_name))
using KernType = KernelGen::KernelPack::KernType;
template <typename Opr>
class ParamHelper {
public:
    using Param = typename Opr::Param;
    ParamHelper() {
        megcore_check(megcoreCreateDeviceHandle(&m_device_handle,
                                                megcorePlatformCPU));
        megcore_check(megcoreCreateComputingHandle(&m_compute_handle,
                                                   m_device_handle));
        m_dnn_handle = megdnn::Handle::make(m_compute_handle, 2);
    }

    ~ParamHelper() {
        megcore_check(megcoreDestroyComputingHandle(m_compute_handle));
        megcore_check(megcoreDestroyDeviceHandle(m_device_handle));
    }

    Param create_param() {
        auto opr = m_dnn_handle->create_operator<Opr>();
        return opr->param();
    }

protected:
    megcoreDeviceHandle_t m_device_handle;
    megcoreComputingHandle_t m_compute_handle;
    std::unique_ptr<megdnn::Handle> m_dnn_handle;
};

std::vector<megcc::CodeGenContext> config_attr(KPT k_type, std::string k_name,
                                               bool use_default_attr) {
#define DEC_DTYPE()                                              \
    auto dtypes = support_dtype();                               \
    llvm::outs() << "please config \"src type\" "                \
                 << "support one of: " << dtypes.first << "\n";  \
    auto dtype_enum = get_int();                                 \
    if (dtypes.second.find(dtype_enum) == dtypes.second.end()) { \
        llvm::outs() << "invalid input"                          \
                     << "\n";                                    \
        abort();                                                 \
    }                                                            \
    std::string dtype_input = dtypes.second[dtype_enum]

#define DEC_FORMAT()                                                \
    auto formats = support_format();                                \
    llvm::outs() << "please config \"format\" "                     \
                 << "support one of: " << formats.first << "\n";    \
    auto format_input = get_int();                                  \
    if (dtypes.second.find(format_input) == formats.second.end()) { \
        llvm::outs() << "invalid input"                             \
                     << "\n";                                       \
        abort();                                                    \
    }

    std::vector<megcc::CodeGenContext> ret;
    std::unordered_map<std::string, megcc::CCAttr> attr_map;
    if (!use_default_attr) {
        llvm::outs() << "+++++++++++++++++++++++++++++++++++++\n";
        llvm::outs() << "  please config attr for " << k_name << "\n";
        llvm::outs() << "+++++++++++++++++++++++++++++++++++++\n";
    }
    switch (k_type) {
        case KPT::TopK: {
            megcc::CCOperand cc_operand;
            attr_map["nr_operands"] = megcc::CCAttr(1);
            if (use_default_attr) {
                attr_map["k"] = megcc::CCAttr(10);
                attr_map["mode"] = megcc::CCAttr("KTH_ONLY");
                cc_operand.dtype = "f32";
            } else {
                llvm::outs() << "please config \"k\""
                             << "\n";
                auto int_input = get_int();
                attr_map["k"] = megcc::CCAttr(int_input);

                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "KTH_ONLY"},
                            {1, "VALUE_IDX_NOSORT"},
                            {2, "VALUE_IDX_SORTED"}};

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"mode\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                std::string input_str = m.second[mode_enum];
                attr_map["mode"] = megcc::CCAttr(input_str);

                DEC_DTYPE();
                cc_operand.dtype = dtype_input;
            }
            attr_map[llvm::formatv("operand:{0}", 0)] = cc_operand;
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ElemwiseKernel: {
            auto&& m_helper = ParamHelper<megdnn::ElemwiseForward>();
            auto param = m_helper.create_param();
            if (use_default_attr) {
                param.mode = megdnn::Elemwise::Mode::RELU;
                attr_map["mode"] = CCAttr(dnnparam_2_str(param.mode));
                attr_map["nr_operands"] = megcc::CCAttr(2);
                megcc::CCOperand res;
                res.dtype = "f32";
                attr_map["operand:0"] = megcc::CCAttr(res);
                attr_map["operand:1"] = megcc::CCAttr(res);
                megcc::CodeGenContext ctx(attr_map);
                ret.push_back(ctx);
            } else {
                EXPORT_ERR(
                        "ElemwiseKernel have so many case , it`s hard to user "
                        "dynamic config, not support now");
            }
        } break;
        case KPT::ElemwiseMultiKernel: {
            auto&& m_helper = ParamHelper<megdnn::ElemwiseMultiType>();
            auto param = m_helper.create_param();
            if (use_default_attr) {
                param.mode = megdnn::ElemwiseMultiType::Mode::QADD;
                attr_map["mode"] = CCAttr(dnnparam_2_str(param.mode));
                attr_map["nr_operands"] = megcc::CCAttr(3);
                megcc::CCOperand res;
                res.dtype = "qsi8";
                attr_map["operand:0"] = megcc::CCAttr(res);
                res.dtype = "qsi8";
                attr_map["operand:1"] = megcc::CCAttr(res);
                res.dtype = "qsi8";
                attr_map["operand:2"] = megcc::CCAttr(res);
                megcc::CodeGenContext ctx(attr_map);
                ret.push_back(ctx);
            } else {
                EXPORT_ERR(
                        "ElemwiseMultiType have so many case , it`s hard to "
                        "user "
                        "dynamic config, not support now");
            }
        } break;
        case KPT::PoolingKernel: {
            auto&& m_helper = ParamHelper<megdnn::PoolingForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                res.dtype = "f32";
                param.stride_h = 1;
                param.stride_w = 1;
                param.pad_h = 1;
                param.pad_w = 1;
                param.window_h = 3;
                param.window_w = 3;
                param.format = ConvParam::Format::NCHW;
                param.mode = megdnn::param::PoolingV0::Mode::MAX;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;
                llvm::outs() << "please config \"stride_h\""
                             << "\n";
                auto int_input = get_int();
                param.stride_h = int_input;
                llvm::outs() << "please config \"stride_w\""
                             << "\n";
                int_input = get_int();
                param.stride_w = int_input;
                llvm::outs() << "please config \"pad_h\""
                             << "\n";
                int_input = get_int();
                param.pad_h = int_input;
                llvm::outs() << "please config \"pad_w\""
                             << "\n";
                int_input = get_int();
                param.pad_w = int_input;
                llvm::outs() << "please config \"window_h\""
                             << "\n";
                int_input = get_int();
                param.window_h = int_input;
                llvm::outs() << "please config \"window_w\""
                             << "\n";
                int_input = get_int();
                param.window_w = int_input;

                DEC_FORMAT();
                param.format = static_cast<ConvParam::Format>(format_input);
                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "MAX"},
                            {1, "AVERAGE"},
                            {2, "AVERAGE_COUNT_EXCLUDE_PADDING"}};

                    return {support_map_to_msg(enum2mode), enum2mode};
                };

                auto m = support_mode();
                llvm::outs() << "please config \"mode\""
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.mode =
                        static_cast<megdnn::param::PoolingV0::Mode>(mode_enum);
            }
            FILL_MAP(attr_map, param, stride_h);
            FILL_MAP(attr_map, param, stride_w);
            FILL_MAP(attr_map, param, pad_h);
            FILL_MAP(attr_map, param, pad_w);
            FILL_MAP(attr_map, param, window_h);
            FILL_MAP(attr_map, param, window_w);

            FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
            attr_map["nr_operands"] = megcc::CCAttr(3);
            attr_map["operand:0"] = megcc::CCAttr(res);
            attr_map["operand:1"] = megcc::CCAttr(res);
            attr_map["operand:2"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::MatrixMulKernel:
        case KPT::BatchMatmulKernel: {
            auto&& m_helper = ParamHelper<megdnn::MatrixMulForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                res.dtype = "f32";
                param.transposeA = false;
                param.transposeB = false;
                param.format = megdnn::param::MatrixMul::Format::DEFAULT;
                param.compute_mode =
                        megdnn::param::MatrixMul::ComputeMode::DEFAULT;
                FILL_MAP(attr_map, param, transposeA);
                FILL_MAP(attr_map, param, transposeB);

                FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
                FILL_MAP_EX(attr_map, param, compute_mode, dnnparam_2_str);
                megcc::CodeGenContext ctx(attr_map);
                ret.push_back(ctx);
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"transposeA\""
                             << "\n";
                auto int_input = get_int();
                param.transposeA = int_input;
                llvm::outs() << "please config \"transposeB\""
                             << "\n";
                int_input = get_int();
                param.transposeB = int_input;
                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{{0, "DEFAULT"},
                                                         {1, "MK4"},
                                                         {2, "MK8"},
                                                         {3, "MK4_DOT"}};

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"format\""
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.format = static_cast<megdnn::param::MatrixMul::Format>(
                        mode_enum);
                param.compute_mode =
                        static_cast<megdnn::param::MatrixMul::ComputeMode>(0);
            }
            FILL_MAP(attr_map, param, transposeA);
            FILL_MAP(attr_map, param, transposeB);

            FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, compute_mode, dnnparam_2_str);
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::MatrixInvKernel:
        case KPT::RelayoutKernel: {
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ReduceKernel: {
            auto&& m_helper = ParamHelper<megdnn::ReduceForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                res.dtype = "f32";
                param.axis = 1;
                param.mode = megdnn::param::Reduce::Mode::SUM;
                param.data_type = megdnn::param::Reduce::DataType::DEFAULT;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"axis\""
                             << "\n";
                auto int_input = get_int();
                param.axis = int_input;
                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "SUM"}, {1, "SUM_SQR"}, {2, "PRODUCT"},
                            {3, "MIN"}, {4, "MAX"},     {5, "MEAN"}};

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"mode\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.mode =
                        static_cast<megdnn::param::Reduce::Mode>(mode_enum);
                param.data_type =
                        static_cast<megdnn::param::Reduce::DataType>(0);
            }

            FILL_MAP(attr_map, param, axis);
            FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, data_type, dnnparam_2_str);
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::IndexingMultiAxisKernel: {
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::IndexingOneHotKernel: {
            auto&& m_helper = ParamHelper<megdnn::IndexingOneHot>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                res.dtype = "f32";
                param.axis = 1;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"axis\""
                             << "\n";
                auto int_input = get_int();
                param.axis = int_input;
            }

            FILL_MAP(attr_map, param, axis);
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::WarpPerspectiveKernel: {
            auto&& m_helper = ParamHelper<megdnn::WarpPerspectiveForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                res.dtype = "f32";
                param.border_val = 0.1;
                param.bmode = megdnn::param::WarpPerspective::BorderMode::
                        BORDER_CONSTANT;
                param.imode =
                        megdnn::param::WarpPerspective::InterpolationMode::AREA;
                param.format = megdnn::param::WarpPerspective::Format::NCHW;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                param.border_val = 0.1;
                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "REPLICATE/BORDER_REPLICATE"},
                            {1, "REFLECT/BORDER_REFLECT"},
                            {2, "REFLECT_101/BORDER_REFLECT_101"},
                            {3, "WRAP/BORDER_WRAP"},
                            {4, "CONSTANT/BORDER_CONSTANT"},
                            {5, "TRANSPARENT/BORDER_TRANSPARENT"},
                            {6, "ISOLATED/BORDER_ISOLATED"},
                    };

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"bmode\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.bmode =
                        megdnn::param::WarpPerspective::BorderMode(mode_enum);
                auto support_imode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "NEAREST/INTER_NEAREST"},
                            {1, "LINEAR/INTER_LINEAR"},
                            {2, "AREA/INTER_AREA"},
                            {3, "CUBIC/INTER_CUBIC"},
                            {4, "LANCZOS4/INTER_LANCZOS4"},
                    };

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto im = support_imode();
                llvm::outs() << "please config \"imode\" "
                             << "support one of: " << im.first << "\n";
                auto imode_enum = get_int();
                if (im.second.find(imode_enum) == im.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.imode = megdnn::param::WarpPerspective::InterpolationMode(
                        imode_enum);
                DEC_FORMAT();
                param.format =
                        megdnn::param::WarpPerspective::Format(format_input);
            }

            FILL_MAP(attr_map, param, border_val);
            FILL_MAP_EX(attr_map, param, bmode, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::WarpAffineKernel: {
            auto&& m_helper = ParamHelper<megdnn::WarpAffineForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                res.dtype = "f32";
                param.border_val = 0.1;
                param.border_mode =
                        megdnn::param::WarpAffine::BorderMode::BORDER_CONSTANT;
                param.imode =
                        megdnn::param::WarpAffine::InterpolationMode::AREA;
                param.format = megdnn::param::WarpAffine::Format::NCHW;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                param.border_val = 0.1;
                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "REPLICATE/BORDER_REPLICATE"},
                            {1, "REFLECT/BORDER_REFLECT"},
                            {2, "REFLECT_101/BORDER_REFLECT_101"},
                            {3, "WRAP/BORDER_WRAP"},
                            {4, "CONSTANT/BORDER_CONSTANT"},
                            {5, "TRANSPARENT/BORDER_TRANSPARENT"},
                            {6, "ISOLATED/BORDER_ISOLATED"},
                    };

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"bmode\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.border_mode =
                        megdnn::param::WarpAffine::BorderMode(mode_enum);
                auto support_imode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "NEAREST/INTER_NEAREST"},
                            {1, "LINEAR/INTER_LINEAR"},
                            {2, "AREA/INTER_AREA"},
                            {3, "CUBIC/INTER_CUBIC"},
                            {4, "LANCZOS4/INTER_LANCZOS4"},
                    };

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto im = support_imode();
                llvm::outs() << "please config \"imode\" "
                             << "support one of: " << im.first << "\n";
                auto imode_enum = get_int();
                if (im.second.find(imode_enum) == im.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.imode = megdnn::param::WarpAffine::InterpolationMode(
                        imode_enum);
                DEC_FORMAT();
                param.format = megdnn::param::WarpAffine::Format(format_input);
            }

            FILL_MAP(attr_map, param, border_val);
            FILL_MAP_EX(attr_map, param, border_mode, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::TypeCvtKernel: {
            if (use_default_attr) {
                attr_map["nr_operands"] = megcc::CCAttr(2);
                megcc::CCOperand res;
                res.dtype = "ui8";
                attr_map["operand:0"] = megcc::CCAttr(res);
                res.dtype = "f32";
                attr_map["operand:1"] = megcc::CCAttr(res);
            } else {
                attr_map["nr_operands"] = megcc::CCAttr(2);
                megcc::CCOperand res;
                auto dtypes = support_dtype();
                llvm::outs() << "please config \"src type\" "
                             << "support one of: " << dtypes.first << "\n";
                auto dtype_enum = get_int();
                if (dtypes.second.find(dtype_enum) == dtypes.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                std::string str_input = dtypes.second[dtype_enum];
                res.dtype = str_input;
                attr_map["operand:0"] = megcc::CCAttr(res);
                llvm::outs() << "please config \"dst type\" "
                             << "support one of: " << dtypes.first << "\n";
                dtype_enum = get_int();
                if (dtypes.second.find(dtype_enum) == dtypes.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                str_input = dtypes.second[dtype_enum];
                res.dtype = str_input;
                attr_map["operand:1"] = megcc::CCAttr(res);
            }
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::PowCKernel: {
            auto&& m_helper = ParamHelper<megdnn::PowC>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                //! init default attr
                param.exp = 2;
                res.dtype = "f32";

            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"exp float value\" "
                             << "\n";
                float f_input = get_float();
                param.exp = f_input;
            }
            FILL_MAP(attr_map, param, exp);
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::CVTransposeKernel:
        case KPT::FlipKernel: {
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ResizeKernel: {
            auto&& m_helper = ParamHelper<megdnn::Resize>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.imode = megdnn::param::Resize::InterpolationMode::NEAREST;
                param.format = megdnn::param::Resize::Format::NCHW;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "NEAREST/INTER_NEAREST"},
                            {1, "LINEAR/INTER_LINEAR"},
                            {2, "AREA/INTER_AREA"},
                            {3, "CUBIC/INTER_CUBIC"},
                            {4, "LANCZOS4/INTER_LANCZOS4"},
                    };
                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"mode\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.imode =
                        megdnn::param::Resize::InterpolationMode(mode_enum);
                DEC_FORMAT();
                param.format = megdnn::param::Resize::Format(format_input);
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            FILL_MAP_EX(attr_map, param, imode, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::RotateKernel: {
            auto&& m_helper = ParamHelper<megdnn::Rotate>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.clockwise = true;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"clockwise\" "
                             << "0 means false, other wise means true: "
                             << "\n";
                int int_input = get_int();
                param.clockwise = int_input;
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            FILL_MAP(attr_map, param, clockwise);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::RoiCopyKernel: {
            auto&& m_helper = ParamHelper<megdnn::ROICopy>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.row_from = 1;
                param.row_to = 1;
                param.col_from = 1;
                param.col_to = 1;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"row_from\" "
                             << "\n";
                int int_input = get_int();
                param.row_from = int_input;
                llvm::outs() << "please config \"row_to\" "
                             << "\n";
                int_input = get_int();
                param.row_to = int_input;
                llvm::outs() << "please config \"col_from\" "
                             << "\n";
                int_input = get_int();
                param.col_from = int_input;
                llvm::outs() << "please config \"col_to\" "
                             << "\n";
                int_input = get_int();
                param.col_to = int_input;
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            FILL_MAP(attr_map, param, row_from);
            FILL_MAP(attr_map, param, row_to);
            FILL_MAP(attr_map, param, col_from);
            FILL_MAP(attr_map, param, col_to);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::CvtColorKernel: {
            auto&& m_helper = ParamHelper<megdnn::CvtColor>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.mode = megdnn::param::CvtColor::Mode::RGB2YUV;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "RGB2GRAY"},  {1, "RGB2YUV"},  {2, "YUV2RGB"},
                            {3, "YUV2RGB"},   {4, "RGBA2RGB"}, {5, "RGBA2BGR"},
                            {6, "RGBA2GRAY"}, {7, "RGB2BGR"},  {8, "BGR2GRAY"},
                            {9, "BGR2RGB"},
                    };

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"mode\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.mode = megdnn::param::CvtColor::Mode(mode_enum);
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ArgSortKernel: {
            auto&& m_helper = ParamHelper<megdnn::Argsort>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.order = megdnn::param::Argsort::Order::ASCENDING;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{{0, "ASCENDING"},
                                                         {1, "DESCENDING"}};

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"order\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.order = megdnn::param::Argsort::Order(mode_enum);
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            attr_map["order"] = CCAttr(dnnparam_2_str(param.order));
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ArgmaxKernel: {
            auto&& m_helper = ParamHelper<megdnn::ArgmaxForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.axis = 1;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"axis\" "
                             << "\n";
                int int_input = get_int();
                param.axis = int32_t(int_input);
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            attr_map["operand:1"] = megcc::CCAttr(res);
            FILL_MAP(attr_map, param, axis);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ConcatKernel: {
            auto&& m_helper = ParamHelper<megdnn::ConcatForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            if (use_default_attr) {
                res.dtype = "f32";
                param.axis = 1;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"axis\" "
                             << "\n";
                int int_input = get_int();
                param.axis = int32_t(int_input);
            }
            attr_map["nr_operands"] = megcc::CCAttr(1);
            attr_map["operand:0"] = megcc::CCAttr(res);
            FILL_MAP(attr_map, param, axis);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        case KPT::ConvKernel:
        case KPT::ConvBackDataKernel: {
            auto&& m_helper = ParamHelper<megdnn::ConvBiasForward>();
            auto param = m_helper.create_param();
            megcc::CCOperand res;
            uint32_t kernel_h = 0, kernel_w = 0;
            if (use_default_attr) {
                res.dtype = "f32";

                kernel_h = 3;
                kernel_w = 3;
                param.sparse = ConvParam::Sparse::DENSE;
                param.format = ConvParam::Format::NCHW;
                param.stride_h = 1;
                param.stride_w = 1;
                param.pad_h = 1;
                param.pad_w = 1;
                param.dilate_h = 1;
                param.dilate_w = 1;
                param.mode = ConvParam::Mode::CONVOLUTION;
            } else {
                DEC_DTYPE();
                res.dtype = dtype_input;

                llvm::outs() << "please config \"kernel_h\" "
                             << "\n";
                int int_input = get_int();
                kernel_h = int_input;
#define CB(name)                                        \
    llvm::outs() << "please config: " << #name << "\n"; \
    int_input = get_int();                              \
    name = int_input

                CB(kernel_w);
                CB(param.stride_h);
                CB(param.stride_w);
                CB(param.pad_h);
                CB(param.pad_w);
                CB(param.dilate_h);
                CB(param.dilate_w);
#undef CB
                auto support_mode = [&]()
                        -> std::pair<std::string, std::map<int, std::string>> {
                    std::map<int, std::string> enum2mode{
                            {0, "DENSE"},
                            {1, "GROUP"},
                    };

                    return {support_map_to_msg(enum2mode), enum2mode};
                };
                auto m = support_mode();
                llvm::outs() << "please config \"sparse\" "
                             << "support one of: " << m.first << "\n";
                auto mode_enum = get_int();
                if (m.second.find(mode_enum) == m.second.end()) {
                    llvm::outs() << "invalid input"
                                 << "\n";
                    abort();
                }
                param.sparse = ConvParam::Sparse(mode_enum);

                DEC_FORMAT();
                param.format = ConvParam::Format(format_input);

                param.mode = ConvParam::Mode(1);
            }
            attr_map["nr_operands"] = megcc::CCAttr(3);
            attr_map["operand:0"] = megcc::CCAttr(res);
            attr_map["operand:1"] = megcc::CCAttr(res);
            attr_map["operand:2"] = megcc::CCAttr(res);
            attr_map["kernel_h"] = CCAttr(kernel_h);
            attr_map["kernel_w"] = CCAttr(kernel_w);
            FILL_MAP(attr_map, param, stride_h);
            FILL_MAP(attr_map, param, stride_w);
            FILL_MAP(attr_map, param, pad_h);
            FILL_MAP(attr_map, param, pad_w);
            FILL_MAP(attr_map, param, dilate_h);
            FILL_MAP(attr_map, param, dilate_w);
            FILL_MAP_EX(attr_map, param, sparse, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
            FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
            megcc::CodeGenContext ctx(attr_map);
            ret.push_back(ctx);
        } break;
        default:
            EXPORT_ERR(ssprintf("config_attr not imp for: %s", k_name.c_str()));
            break;
    }

    return ret;
#undef DEC_DTYPE
}

}  // namespace exporter
}  // namespace megcc
