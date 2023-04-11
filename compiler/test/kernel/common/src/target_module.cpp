#include "test/kernel/common/target_module.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"
#include "megcc_test_config.h"

namespace megcc {
using namespace KernelGen;
namespace test {
namespace {
void write_map(
        const std::unordered_map<std::string, std::string>& map, const std::string& dir,
        bool is_cv = false) {
    for (auto& kv : map) {
        auto file_path = dir + "/" + kv.first + ".c";
        std::ofstream out_file(file_path);
        out_file << "//! generated file by megcc, do not modify\n";
        if (!is_cv) {
            out_file << GenCommonInclude() << "\n";
        }
        out_file << "#include <stdio.h>\n";
        out_file << "#include <math.h>\n";
        out_file << kv.second;
        out_file.close();
    }
}

void write_map(
        const std::unordered_map<std::string, std::vector<uint8_t>>& map,
        const std::string& dir) {
    for (auto& kv : map) {
        auto file_path = dir + "/" + kv.first + ".o";
        std::ofstream out_file(file_path, std::ios::binary);
        auto&& vec = kv.second;
        out_file.write((const char*)vec.data(), vec.size());
        out_file.close();
    }
}
}  // namespace

bool TargetModule::exist(const std::string& sig) const {
    return m_kern_map.count(sig) != 0;
}

void TargetModule::add(
        const std::string& sig, const std::string& body, const std::string& init_sig,
        const std::string& init_body, const std::string& workspace_sig,
        const std::string& workspace_body, const std::string& deduce_sig,
        const std::string& deduce_body) {
    if (m_kern_map.count(sig) == 0) {
        m_kern_map[sig] = body;
    }
    if (m_kern_init_map.count(init_sig) == 0) {
        m_kern_init_map[init_sig] = init_body;
    }
    if (m_kern_workspace_map.count(workspace_sig) == 0) {
        m_kern_workspace_map[workspace_sig] = workspace_body;
    }
    if (m_kern_deduce_layout_map.count(deduce_sig) == 0 && deduce_sig.size() > 0) {
        m_kern_deduce_layout_map[deduce_sig] = deduce_body;
    }
}

void TargetModule::add_workspace_size(
        const std::string& workspace_size_symbol, size_t workspace_size) {
    if (m_jit_workspace_size_map.count(workspace_size_symbol) == 0) {
        m_jit_workspace_size_map[workspace_size_symbol] = workspace_size;
    }
}

void TargetModule::add_binary(const std::string& sig, const std::vector<uint8_t>& vec) {
    if (m_kern_bin_map.find(sig) == m_kern_bin_map.end()) {
        m_kern_bin_map[sig] = vec;
    }
}

void TargetModule::add_cv(
        const std::string& sym, const std::string& sig, const std::string& body) {
    if (m_cv_kern_map.count(sym) == 0) {
        m_cv_kern_map[sym] = body;
        m_cv_sig_map[sym] = sig;
    }
}

bool TargetModule::exist_internal_function(const std::string& sig) const {
    return m_internal_kern_map.count(sig) != 0;
}

void TargetModule::add_internal_func(const std::string& sig, const std::string& body) {
    if (m_internal_kern_map.count(sig) == 0) {
        m_internal_kern_map[sig] = body;
    }
}

std::string TargetModule::get_core_module_str() const {
    std::stringstream ss;
    ss << "//! generated file by megcc, do not modify\n";
    ss << GenCommonInclude() << "\n";
    ss << "#include <stdio.h>\n";
    ss << "#include <math.h>\n";
    for (auto& kv : m_kern_map) {
        ss << kv.second << "\n";
    }
    for (auto& kv : m_kern_init_map) {
        ss << kv.second << "\n";
    }
    return ss.str();
}

std::string TargetModule::get_helper_module_str() const {
    std::stringstream ss;
    ss << "//! generated file by megcc, do not modify\n";
    ss << "#include <unordered_map>\n";
    ss << "#include <string>\n";
    ss << "#include \"tinycv_c.h\" \n";
    ss << "#include \"test/kernel/common/target_module.h\"\n";
    ss << GenCommonInclude() << "\n";
    for (auto& kv : m_kern_map) {
        ss << "extern \"C\" " << GenCommonRet() << " " << kv.first << GenCommonCall()
           << ";\n";
    }
    for (auto& kv : m_kern_init_map) {
        ss << "extern \"C\" " << GenCommonRet() << " " << kv.first
           << GenCommonInitCall() << ";\n";
    }
    for (auto& kv : m_kern_workspace_map) {
        ss << "extern \"C\" " << GenCommonRet() << " " << kv.first
           << GenCommonWorkspaceCall() << ";\n";
    }
    for (auto& kv : m_kern_deduce_layout_map) {
        ss << "extern \"C\" " << GenCommonRet() << " " << kv.first
           << GenCommonDeduceCall() << ";\n";
    }
    for (auto& kv : m_cv_sig_map) {
        ss << "extern \"C\" "
           << " void " << kv.second << ";\n";
    }
    ss << "namespace megcc{\n";
    ss << "namespace test{\n";
    ss << "std::unordered_map<std::string, StdKernelCall> "
          "g_str_2_kern{";
    for (auto& kv : m_kern_map) {
        ss << "{\"" << kv.first << "\"," << kv.first << "},";
    }
    ss << "};\n";

    ss << "std::unordered_map<std::string, StdKernelInitCall> "
          "g_str_2_kern_init{";
    for (auto& kv : m_kern_init_map) {
        ss << "{\"" << kv.first << "\"," << kv.first << "},";
    }
    ss << "};\n";

    ss << "std::unordered_map<std::string, StdKernelWorkspaceCall> "
          "g_str_2_kern_workspace{";
    for (auto& kv : m_kern_workspace_map) {
        ss << "{\"" << kv.first << "\"," << kv.first << "},";
    }
    ss << "};\n";

    ss << "std::unordered_map<std::string, StdKernelDeduceCall> "
          "g_str_2_kern_deduce{";
    for (auto& kv : m_kern_deduce_layout_map) {
        ss << "{\"" << kv.first << "\"," << kv.first << "},";
    }
    ss << "};\n";

    ss << "std::unordered_map<std::string, void*> "
          "g_str_2_cv_kern{";
    for (auto& kv : m_cv_kern_map) {
        ss << "{\"" << kv.first << "\",(void*)&" << kv.first << "},";
    }
    ss << "};\n";

    ss << "std::unordered_map<std::string, size_t> "
          "g_str_2_workspace_size{";
    for (auto& kv : m_jit_workspace_size_map) {
        ss << "{\"" << kv.first << "\", " << kv.second << "},";
    }
    ss << "};\n";

    ss << "} // namespace test\n";
    ss << "} // namespace megcc\n";
    return ss.str();
}
extern std::unordered_map<std::string, StdKernelCall> g_str_2_kern;
StdKernelCall TargetModule::get_kernel(std::string& sig) const {
#if !MEGCC_TEST_GEN
    auto res = g_str_2_kern.find(sig);
    if (res == g_str_2_kern.end()) {
        return nullptr;
    } else {
        return res->second;
    }
#else
    return nullptr;
#endif
}
extern std::unordered_map<std::string, void*> g_str_2_cv_kern;
void* TargetModule::get_cv_kernel(std::string& sig) const {
#if !MEGCC_TEST_GEN
    auto res = g_str_2_cv_kern.find(sig);
    if (res == g_str_2_cv_kern.end()) {
        return nullptr;
    } else {
        return res->second;
    }
#else
    return nullptr;
#endif
}
extern std::unordered_map<std::string, StdKernelInitCall> g_str_2_kern_init;
StdKernelInitCall TargetModule::get_kernel_init(std::string& sig) const {
#if !MEGCC_TEST_GEN
    auto res = g_str_2_kern_init.find(sig);
    if (res == g_str_2_kern_init.end()) {
        return nullptr;
    } else {
        return res->second;
    }
#else
    return nullptr;
#endif
}
extern std::unordered_map<std::string, StdKernelWorkspaceCall> g_str_2_kern_workspace;
StdKernelWorkspaceCall TargetModule::get_kernel_workspace(std::string& sig) const {
#if !MEGCC_TEST_GEN
    auto res = g_str_2_kern_workspace.find(sig);
    if (res == g_str_2_kern_workspace.end()) {
        return nullptr;
    } else {
        return res->second;
    }
#else
    return nullptr;
#endif
}

extern std::unordered_map<std::string, size_t> g_str_2_workspace_size;
size_t TargetModule::get_kernel_workspace_size(std::string& sig) const {
#if !MEGCC_TEST_GEN
    auto res = g_str_2_workspace_size.find(sig);
    if (res == g_str_2_workspace_size.end()) {
        return 0;
    } else {
        return res->second;
    }
#else
    return 0;
#endif
}

extern std::unordered_map<std::string, StdKernelDeduceCall> g_str_2_kern_deduce;
StdKernelDeduceCall TargetModule::get_kernel_deduce(std::string& sig) const {
#if !MEGCC_TEST_GEN
    auto res = g_str_2_kern_deduce.find(sig);
    if (res == g_str_2_kern_deduce.end()) {
        return nullptr;
    } else {
        return res->second;
    }
#else
    return nullptr;
#endif
}

TargetModule& TargetModule::get_global_target_module() {
    static TargetModule res;
    return res;
}

void TargetModule::write_to_dir(std::string dir) const {
    write_map(m_cv_kern_map, dir, true);
    write_map(m_kern_map, dir);
    write_map(m_kern_init_map, dir);
    write_map(m_kern_workspace_map, dir);
    write_map(m_kern_deduce_layout_map, dir);
    write_map(m_internal_kern_map, dir);
    write_map(m_kern_bin_map, dir);
    auto file_path = dir + "/target_helper.cpp";
    std::ofstream out_file(file_path);
    auto helper_body = get_helper_module_str();
    out_file << helper_body;
    out_file.close();
}

}  // namespace test
}  // namespace megcc
