#pragma once
#include "Arm/ArmCommon/Activation.h"
#include "Utils/StringTemplate.h"

static std::string store_ocx_owx(
        std::string dst, std::string reg_name, std::string bias_scale,
        std::string dst_scale,
        const megcc::KernelGen::ArmCommon::ActivationGenIntrinsicBase& act,
        const int oc_step, const int ow_step) {
    std::stringstream ss;
    ss << "{";
    for (int i = 0; i < oc_step; ++i)
        ss << megcc::KernelGen::StringTemplate::StringTemplateArgs()
                        .add("dst", dst)
                        .add("i", i)
                        .render(R"(
        int8_t* store_ptr${i} = (int8_t*)${dst} + ${i} * dst->layout.stride[1];
    )");
    for (int i = 0; i < oc_step; ++i) {
        for (int j = 0; j < ow_step; ++j) {
            ss << act.GenIntrinsicQuantStore(
                    reg_name + "[" + std::to_string(i) + "]" + "[" + std::to_string(j) +
                            "]",
                    "store_ptr" + std::to_string(i) + " + " + std::to_string(j) +
                            " * 4",
                    bias_scale, dst_scale);
        }
    }
    ss << "\n}";
    return ss.str();
}

static std::string store_ocx_ow_remain(
        std::string dst, std::string reg_name, std::string bias_scale,
        std::string dst_scale, std::string ow_remain,
        const megcc::KernelGen::ArmCommon::ActivationGenIntrinsicBase& act,
        const int oc_step) {
    std::stringstream ss;
    ss << "{";
    for (int i = 0; i < oc_step; ++i)
        ss << megcc::KernelGen::StringTemplate::StringTemplateArgs()
                        .add("dst", dst)
                        .add("i", i)
                        .render(R"(
        int8_t* store_ptr${i} = (int8_t*)${dst} + ${i} * dst->layout.stride[1];
    )");
    for (int i = 0; i < oc_step; ++i) {
        ss << megcc::KernelGen::StringTemplate::StringTemplateArgs()
                        .add("ow_remain", ow_remain)
                        .render(R"(
        for(int j = 0; j < ${ow_remain}; ++j) {
                    )");
        ss << act.GenIntrinsicQuantStore(
                reg_name + "[" + std::to_string(i) + "]" + "[j]",
                "store_ptr" + std::to_string(i) + " + j * oc_step", bias_scale,
                dst_scale);
        ss << "\n}";
    }
    ss << "\n}";
    return ss.str();
}