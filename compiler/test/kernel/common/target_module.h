#pragma once
#include <data_struct.h>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

typedef TinyNNStatus(KernelCall)(
        Tensor** inputs, int nr_input, Tensor** outputs, int nr_output,
        const Workspace* workspace, const RuntimeOpt* opt);
typedef TinyNNStatus(KernelInitCall)(
        Tensor** inputs, int nr_input, Tensor* out_weights, int* nr_out_weight,
        const RuntimeOpt* opt);

typedef TinyNNStatus(KernelWorkspaceCall)(
        Tensor** inputs, int nr_input, int nr_thread, size_t* workspace);

typedef TinyNNStatus(KernelDeduceCall)(
        Tensor** inputs, int nr_input, Tensor** outputs, int nr_output);

using StdKernelCall = std::function<KernelCall>;
using StdKernelInitCall = std::function<KernelInitCall>;
using StdKernelWorkspaceCall = std::function<KernelWorkspaceCall>;
using StdKernelDeduceCall = std::function<KernelDeduceCall>;
namespace megcc {
namespace test {
class TargetModule {
public:
    bool exist(const std::string& sig) const;
    bool exist_internal_function(const std::string& sig) const;
    void add(
            const std::string& sig, const std::string& body,
            const std::string& init_sig, const std::string& init_body,
            const std::string& workspace_sig, const std::string& workspace_body,
            const std::string& deduce_sig, const std::string& deduce_body);

    void add_workspace_size(
            const std::string& workspace_size_symbol, size_t workspace_size);

    void add_binary(const std::string& sig, const std::vector<uint8_t>& vec);

    void add_cv(
            const std::string& sym, const std::string& sig, const std::string& body);
    void add_internal_func(const std::string& sig, const std::string& body);
    std::string get_core_module_str() const;
    std::string get_helper_module_str() const;
    void* get_cv_kernel(std::string& sig) const;
    StdKernelCall get_kernel(std::string& sig) const;
    StdKernelInitCall get_kernel_init(std::string& sig) const;
    StdKernelWorkspaceCall get_kernel_workspace(std::string& sig) const;
    StdKernelDeduceCall get_kernel_deduce(std::string& sig) const;
    void write_to_dir(std::string dir) const;
    static TargetModule& get_global_target_module();

    //! get the jit exec workspace size, for check jit exec and return of the
    //! get workspace kernel
    size_t get_kernel_workspace_size(std::string& sig) const;

private:
    std::unordered_map<std::string, std::string> m_cv_kern_map;
    std::unordered_map<std::string, std::string> m_cv_sig_map;
    std::unordered_map<std::string, std::string> m_internal_kern_map;
    std::unordered_map<std::string, std::string> m_kern_map;
    std::unordered_map<std::string, std::string> m_kern_init_map;
    std::unordered_map<std::string, std::string> m_kern_workspace_map;
    std::unordered_map<std::string, std::string> m_kern_deduce_layout_map;

    //! Jit exec get workspace function when kernel gen, check it when kernel
    //! run
    std::unordered_map<std::string, size_t> m_jit_workspace_size_map;

    std::unordered_map<std::string, std::vector<uint8_t>> m_kern_bin_map;
};

}  // namespace test
}  // namespace megcc

// vim: syntax=cpp.doxygen
