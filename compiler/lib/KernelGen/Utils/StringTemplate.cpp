/**
 * \file
 * compiler/lib/KernelGen/Utils/StringTemplate.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "StringTemplate.h"
#include <regex>
#include <sstream>
using namespace megcc;
using namespace KernelGen;

namespace {

void trim(std::string& s) {
    if (s.empty()) {
        return;
    }

    s.erase(0, s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
}
std::string render_with_cbk(
        const std::string& template_str, std::regex regx,
        const StringTemplate::KvMap kv_map, const StringTemplate::FuncMap func_map) {
    std::stringstream ss;
    std::sregex_iterator iter(template_str.begin(), template_str.end(), regx), end;
    std::string last_suffix = template_str;
    std::for_each(iter, end, [&](const std::smatch& match) {
        ss << match.prefix().str();
        std::string key_str = match[1].str();
        std::regex func_reg("([\\w_]+)\\((.*)\\)");
        std::smatch base_match;
        if (std::regex_match(key_str, base_match, func_reg)) {
            //! match func type
            std::regex param_reg("([^,]+)");
            auto func_name = base_match[1].str();
            auto param_str = base_match[2].str();
            std::sregex_iterator param_iter(
                    param_str.begin(), param_str.end(), param_reg),
                    param_end;
            std::vector<std::string> param_str_vec;
            std::for_each(param_iter, param_end, [&](const std::smatch& match) {
                auto str = match[1].str();
                trim(str);
                param_str_vec.push_back(str);
            });
            if (func_map.find(func_name) != func_map.end()) {
                ss << func_map.find(func_name)->second(param_str_vec);
            } else {
                CC_ABORT << "render failed, can not render " << key_str.c_str() << "\n";
            }
        } else {
            //! match var type
            if (kv_map.find(key_str) != kv_map.end()) {
                ss << kv_map.find(key_str)->second;
            } else {
                CC_ABORT << "render failed, can not render " << key_str.c_str() << "\n";
            }
        }
        last_suffix = match.suffix();
    });
    ss << last_suffix;
    return ss.str();
}

}  // namespace

std::string StringTemplate::render(
        const std::string& template_str, const KvMap& kv_map, const FuncMap& func_map) {
    //! template args used as ${args}
    return render_with_cbk(
            template_str, std::regex(R"(\$\{([^\{\}]+)\})"), kv_map, func_map);
}

void StringTemplate::StringTemplateArgs::init_builtin() {
    //! for is builtin func impl, you can use
    //! ${_unroll(nr_iter, template_body, arg0:2, arg1:copy)} to render,
    //! nr_iter is number or template arg, template_body is template arg, arg0
    //! is template in template_body.
    //! template_body like x[${_i}] = ${arg1}(y[${_i}] + ${arg0})
    //! FIXME: unroll impl is slow, optimize it
    this->add("_unroll", [&](std::vector<std::string> args) {
        StringTemplate::StringTemplateArgs& template_args = *this;
        StringTemplate::StringTemplateArgs this_temp = template_args;
        CC_ASSERT(args.size() > 1);
        auto k_str = template_args.try_get_str(args[0]);
        if (k_str.size() == 0) {
            k_str = args[0];
        }
        int k = std::atoi(k_str.c_str());
        std::regex kv_reg("([\\w_]+):([\\w_]+)");
        for (size_t i = 2; i < args.size(); ++i) {
            std::smatch base_match;
            if (std::regex_match(args[i], base_match, kv_reg)) {
                auto key_str = base_match[1].str();
                auto value_str = base_match[2].str();
                this_temp.add(key_str.c_str(), value_str.c_str());
            } else {
                CC_ABORT << "tempstr for function only accept XXX:YYY as extra "
                            "string args";
            }
        }
        std::stringstream ss;
        for (int i = 0; i < k; ++i) {
            ss << this_temp.add("_i", i).render(this_temp.get_str(args[1]));
        }
        return ss.str();
    });
}

const std::string StringTemplate::render_init_body(
        uint32_t nr_out_weight, const std::string& fill_weight_attr,
        const std::string& fill_weight_transform, const std::string& common_def) {
    const std::string body_temp = R"({
    if (out_weights == NULL && nr_out_weight != NULL) {
        *nr_out_weight = ${nr_out_weight};
        return TinyNN_SUCCESS;
    }

    ${common_def}

    if (out_weights != NULL && nr_out_weight == NULL) {
        ${fill_weight_attr}
        return TinyNN_SUCCESS;
    }
    if (out_weights != NULL && nr_out_weight != NULL) {
        ${fill_weight_transform}
    }
    return TinyNN_SUCCESS;
}
    )";
    return StringTemplate::StringTemplateArgs()
            .add("nr_out_weight", nr_out_weight)
            .add("fill_weight_attr", fill_weight_attr)
            .add("fill_weight_transform", fill_weight_transform)
            .add("common_def", common_def)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
