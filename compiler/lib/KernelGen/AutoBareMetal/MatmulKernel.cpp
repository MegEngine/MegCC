/**
 * \file
 * compiler/lib/KernelGen/AutoBareMetal/MatmulKernel.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>

#include "Common/ElemwiseCommon.h"
#include "MatmulKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/CodeGen/CodeGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace AutoBareMetal;

bool MatmulKernel::IsAvailable(TContext* context) const {
    auto auto_kern = codegen::GenCode(KernelPack::KernType::MatrixMulKernel);
    bool mode_kern_ok = auto_kern->IsAvailable(context);
    return mode_kern_ok;
}

std::string MatmulKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_auto_matmul_";
    if (context->getAttrBool("transposeA")) {
        ss << "t";
    } else {
        ss << "n";
    }
    if (context->getAttrBool("transposeB")) {
        ss << "t";
    } else {
        ss << "n";
    }
    return ss.str();
}

std::string MatmulKernel::GetKernelBody(TContext* context) const {
    auto op0 = context->getAttrOprand("operand:0");
    auto src_dtype = op0.dtype;
    auto specifier = Utils::cvt_dtype_specifier(src_dtype);

    auto dst_operand = Utils::get_last_operand(context);
    int nr_dim = dst_operand.shape.size();

    std::stringstream writer;
    writer << "#include <math.h> \n";
    writer << "#include <stdbool.h> \n";
    auto auto_kern = codegen::GenCode(KernelPack::KernType::MatrixMulKernel);
    auto func_name = auto_kern->GetKernelSymbol(context);
    writer << StringTemplate::StringTemplateArgs()
                      .add("func_name", func_name)
                      .add("nr_dim", nr_dim)
                      .add("specifier", specifier)
                      .render(R"(
        typedef struct MemRef_descriptor_* MemRef_descriptor;
        typedef struct MemRef_descriptor_ {
            ${specifier}* allocated;
            ${specifier}* aligned;
            size_t offset;
            size_t sizes[${nr_dim}];
            size_t strides[${nr_dim}];
        } Memref;
        #define FUNC_NAME  _mlir_ciface_${func_name}
        extern void FUNC_NAME(Memref* a, Memref* b, Memref* c);

    )");

    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);

    std::string unary_str = R"({
                ${specifier}* input_a = (${specifier}*)inputs[0]->ptr;
                TINYNN_ASSERT(input_a);
                ${specifier}* input_b = (${specifier}*)inputs[1]->ptr;
                TINYNN_ASSERT(input_b);
                ${specifier}* output_data = (${specifier}*)outputs[0]->ptr;
                TINYNN_ASSERT(output_data);


                Memref ref_a;
                const Layout src_layout_a = inputs[0]->layout;
                ${fill_memref(ref_a, src_layout_a, input_a)}

                Memref ref_b;
                const Layout src_layout_b = inputs[1]->layout;
                ${fill_memref(ref_b, src_layout_b, input_b)}

                Memref ref_c;
                const Layout dst_layout = outputs[0]->layout;
                ${fill_memref(ref_c, dst_layout, output_data)}

                Layout out_layout = outputs[0]->layout;
                size_t nr_elem = 1;
                for (size_t i = 0; i < out_layout.nr_dim; ++i) {
                    nr_elem *= out_layout.dims[i];
                }
                memset(output_data, 0, sizeof(float) * nr_elem);

                FUNC_NAME(&ref_a, &ref_b, &ref_c);
                return TinyNN_SUCCESS;
                }

                )";
    writer << StringTemplate::StringTemplateArgs()
                      .add("specifier", specifier)
                      .add("fill_memref",
                           [](const std::string& memref_var,
                              const std::string& layout_var,
                              const std::string& ptr_var) {
                               std::string body_temp = R"(
                                  ${memref_var}.aligned = ${ptr_var};
                                  ${memref_var}.allocated = ${ptr_var};
                                  ${memref_var}.offset = 0;
                                  for(int i = 0; i < ${layout_var}.nr_dim; ++i){
                                    ${memref_var}.sizes[i] = ${layout_var}.dims[i];
                                    ${memref_var}.strides[i] = ${layout_var}.stride[i];
                                  }
                              )";
                               return StringTemplate::StringTemplateArgs()
                                       .add("memref_var", memref_var)
                                       .add("layout_var", layout_var)
                                       .add("ptr_var", ptr_var)
                                       .render(body_temp);
                           })
                      .render(unary_str);

    return writer.str();
}

std::vector<KernelObj> MatmulKernel::GetDependInternalSymbol(TContext* context) const {
    auto rst = codegen::GenCode(KernelPack::KernType::MatrixMulKernel);
    return {rst->GetKernelObj(context)};
}

// vim: syntax=cpp.doxygen
