/**
 * \file runtime/test/instruction/shape_of.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string.h>
#include <memory>
#include <vector>
#include "./common/common.h"
#include "math.h"
using namespace test;

TEST(INSTRUCTION, ShapeOf) {
    //! input shape =[10, 20, 10, 10]
    std::shared_ptr<Tensor> output;
    std::vector<int> data0(20 * 10 * 10 * 10);
    std::vector<int> data1(20);

    auto create_shape_of = [&](Tensor* input) {
        auto shapeof = std::make_shared<ShapeOf>();

        output = std::make_shared<Tensor>();
        output->is_dynamic = true;

        shapeof->input = input;
        shapeof->output = output.get();
        return shapeof;
    };
    VM* vm = create_vm();
    auto test_shape_of = [&](Tensor* input, const Tensor& expect) {
        auto shapeof= create_shape_of(input );
        Instruction inst;
        inst.tag = TinyNN_INST_SHAPEOF;
        inst.workload.shape_of= *shapeof;
        vm_instruction_call(vm, &inst);
        check_tensor(*shapeof->output, expect);
        vm->model->host_dev.free(shapeof->output->ptr);
    };

    auto test_case = [&](std::vector<uint32_t> shape) {
        auto src = create_tensor(shape, TinyNNDType::TinyNN_INT, data0.data());
        int index = 0;
        for (auto i : shape) {
            data1[index++] = i;
        }
        auto trueth = create_tensor({static_cast<uint32_t>(shape.size())},
                                    TinyNNDType::TinyNN_INT, data1.data());
        test_shape_of(src.get(), *trueth);
    };

    test_case({1, 1, 2, 3});
    test_case({10, 5, 8, 3});
    test_case({1, 3});
    test_case({2, 3, 5});
}

// vim: syntax=cpp.doxygen
