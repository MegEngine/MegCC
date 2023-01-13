/**
 * \file compiler/lib/KernelGen/Jit/JitExe.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2021 Megvii Inc. All rights reserved.
 */

#include "compiler/KernelGen/JitExe.h"
#include <fstream>
#include <mutex>
#include "JitHeader.h"
#include "LibJit.h"
#include "libtcc.h"

extern "C" {
#include "data_struct.h"
}

#undef LOG_INFO
#undef LOG_DEBUG
#undef LOG_ERROR

#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;

namespace {

static inline TinyNNDType get_dtype_enum(const std::string& dtype) {
    if (dtype == "f32") {
        return TinyNNDType::TinyNN_FLOAT;
    } else if (dtype == "i32" || dtype == "si32") {
        return TinyNNDType::TinyNN_INT;
    } else if (dtype == "i8" || dtype == "si8") {
        return TinyNNDType::TinyNN_INT8;
    } else if (dtype == "i16" || dtype == "si16") {
        return TinyNNDType::TinyNN_INT16;
    } else if (dtype == "ui8") {
        return TinyNNDType::TinyNN_UINT8;
    } else if (Utils::is_quant_dtype(dtype, 8)) {
        return TinyNNDType::TinyNN_QINT8;
    } else if (Utils::is_quant_dtype(dtype, 32)) {
        return TinyNNDType::TinyNN_QINT32;
    } else {
        CC_ASSERT(dtype == "si8" || dtype == "i8") << "not support " << dtype;
        return TinyNNDType::TinyNN_INT8;
    }
}

static void compiler_error(void*, const char* msg) {
    std::string error(msg);
    LOG_ERROR << error << "\n";
}

struct AutoFile {
    std::string file_path = "/tmp/libtcc1.a";
    AutoFile() {
        std::ofstream lib;
        lib.open(file_path, std::ofstream::binary);
        lib.write(reinterpret_cast<const char*>(libtcc1_a), libtcc1_a_len);
        lib.close();
    }
    ~AutoFile() { std::remove(file_path.c_str()); }
};

}  // namespace
using namespace megcc;
using namespace KernelGen;

size_t JitExec::jit_exec_and_get_workspace(const KernelFn* func,
                                           TContext* ctx) {
    static AutoFile lib_file;
    std::string program = get_header_define();
    program += func->GetWorkspaceBodyAndJitExec(ctx);
    TCCState* state = tcc_new();
    CC_ASSERT(state) << "Can’t create a TCC context.\n";

    tcc_set_lib_path(state, "/tmp");
    tcc_add_library_path(state, "/usr/lib64");
    tcc_add_library_path(state, "/usr/lib/x86_64-linux-gnu/");
    tcc_add_library_path(state, "/usr/lib");

    tcc_set_output_type(state, TCC_OUTPUT_MEMORY);
    tcc_set_error_func(state, nullptr, compiler_error);
    auto ret = tcc_compile_string(state, program.c_str());
    CC_ASSERT(ret == 0) << "TCC jit Compilation error !\n"
                        << "source file:\n"
                        << program << "\n";

    tcc_relocate(state, TCC_RELOCATE_AUTO);

    WorkspaceFunc workspace_func = reinterpret_cast<WorkspaceFunc>(
            tcc_get_symbol(state, func->GetWorkspaceSymbol(ctx).c_str()));

    //! construct the jit param
    int nr_input = ctx->getAttrInt("nr_operands") - 1;
    std::vector<Tensor> inputs;
    std::vector<Tensor*> inputs_ptr;
    for (int i = 0; i < nr_input; i++) {
        auto operand = ctx->getAttrOprand("operand:" + std::to_string(i));
        if (operand.shape.size() < 1) {
            continue;
        }
        Tensor tensor;
        tensor.layout.nr_dim = operand.shape.size();
        size_t stride = 1;
        for (int dim = tensor.layout.nr_dim - 1; dim >= 0; dim--) {
            tensor.layout.dims[dim] = operand.shape[dim];
            tensor.layout.stride[dim] = stride;
            stride *= operand.shape[dim];
        }
        tensor.layout.nr_dim = operand.shape.size();
        tensor.dtype.type_enum = get_dtype_enum(operand.dtype);
        inputs.push_back(tensor);
    }
    //! get the pointer (the pointer will change after vector resize)
    for (int i = 0; i < nr_input; i++) {
        inputs_ptr.push_back(&inputs[i]);
    }
    //! must init to 0
    size_t workspace = 0;
    workspace_func(inputs_ptr.data(), inputs_ptr.size(), 1, &workspace);
    tcc_delete(state);
    LOG_DEBUG << "Jit get workspace size: " << workspace
              << ", with symbol: " << func->GetWorkspaceSymbol(ctx).c_str()
              << "\n";
    return workspace;
}

// vim: syntax=cpp.doxygen
