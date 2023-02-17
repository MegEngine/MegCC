/**
 * \file
 * compiler/lib/KernelGen/Utils/StringTemplate.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include "compiler/Common/Logger.h"
#include "compiler/Common/TContext.h"
namespace megcc {
namespace KernelGen {

class StringTemplate {
public:
    template <typename R, typename C, typename... Args>
    static std::function<R(Args...)> object_bind(
            R (C::*func)(Args...) const, C& instance) {
        return [=](Args... args) { return (instance.*func)(args...); };
    }
    using KvMap = std::unordered_map<std::string, std::string>;
    using FuncType = std::function<std::string(std::vector<std::string>)>;
    using FuncMap = std::unordered_map<std::string, FuncType>;
    using FuncTypeArg0 = std::function<std::string()>;
    using FuncTypeArg1 = std::function<std::string(const std::string&)>;
    using FuncTypeArg2 =
            std::function<std::string(const std::string&, const std::string&)>;
    using FuncTypeArg3 = std::function<std::string(
            const std::string&, const std::string&, const std::string&)>;
    using FuncTypeArg4 = std::function<std::string(
            const std::string&, const std::string&, const std::string&,
            const std::string&)>;
    static std::string render(
            const std::string& template_str, const KvMap& kv_map,
            const FuncMap& func_map = {});
    //! init function will be called third times, first return nr_out_weight,
    //! second do fill_weight_attr code, last do fill_weight_transform code
    //! common_def code will do since second call
    static const std::string render_init_body(
            uint32_t nr_out_weight, const std::string& fill_weight_attr,
            const std::string& fill_weight_transform,
            const std::string& common_def = "");
    class StringTemplateArgs {
    public:
        StringTemplateArgs(TContext* ctx = nullptr) : m_ctx(ctx) { init_builtin(); };
        StringTemplateArgs& add_ctx_int(const std::string& key) {
            CC_ASSERT(m_ctx);
            return add(key, m_ctx->getAttrInt(key));
        }
        StringTemplateArgs& add_ctx_str(const std::string& key) {
            CC_ASSERT(m_ctx);
            return add(key, m_ctx->getAttrStr(key));
        }
        StringTemplateArgs& add(const std::string& key, uint32_t value) {
            m_kv_map[key] = std::to_string(value);
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, int value) {
            m_kv_map[key] = std::to_string(value);
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, const std::string& value) {
            m_kv_map[key] = value;
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, FuncType func) {
            m_func_map[key] = func;
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, FuncTypeArg4 func) {
            m_func_map[key] = [=](std::vector<std::string> args) -> std::string {
                CC_ASSERT(args.size() == 4);
                return func(args[0], args[1], args[2], args[3]);
            };
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, FuncTypeArg3 func) {
            m_func_map[key] = [=](std::vector<std::string> args) -> std::string {
                CC_ASSERT(args.size() == 3);
                return func(args[0], args[1], args[2]);
            };
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, FuncTypeArg2 func) {
            m_func_map[key] = [=](std::vector<std::string> args) -> std::string {
                CC_ASSERT(args.size() == 2);
                return func(args[0], args[1]);
            };
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, FuncTypeArg1 func) {
            m_func_map[key] = [=](std::vector<std::string> args) -> std::string {
                CC_ASSERT(args.size() == 1);
                return func(args[0]);
            };
            return *this;
        }
        StringTemplateArgs& add(const std::string& key, FuncTypeArg0 func) {
            m_func_map[key] = [=](std::vector<std::string> args) -> std::string {
                CC_ASSERT(args.size() == 0);
                return func();
            };
            return *this;
        }
        StringTemplateArgs& remove(const std::string& key) {
            m_kv_map.erase(key);
            return *this;
        }
        std::string render(const std::string& template_str) {
            return StringTemplate::render(template_str, m_kv_map, m_func_map);
        }
        std::string try_get_str(const std::string& key) { return m_kv_map[key]; }
        std::string get_str(const std::string& key) {
            CC_ASSERT(m_kv_map.count(key) > 0);
            return m_kv_map[key];
        }

        StringTemplateArgs(const StringTemplateArgs& obj) {
            m_kv_map = obj.m_kv_map;
            m_ctx = obj.m_ctx;
            m_func_map = obj.m_func_map;
        }

    private:
        void init_builtin();
        TContext* m_ctx{nullptr};
        KvMap m_kv_map;
        FuncMap m_func_map;
    };
};

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen