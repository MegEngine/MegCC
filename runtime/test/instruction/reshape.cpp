/**
 * \file
 * runtime/test/instruction/reshape.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string.h>
#include <memory>
#include <vector>
#include "./common/common.h"
using namespace test;

TEST(INSTRUCTION, Reshape) {
    std::vector<Instruction> insts;
    std::vector<std::shared_ptr<Tensor>> out_tensors;

    //! input shape =[3, 3]
    std::vector<float> data(3 * 3 * 3);
    for (size_t i = 0; i < 3 * 3 * 3; i++) {
        data[i] = i;
    }
    auto src_tensor = create_tensor({3, 3, 3}, TinyNN_FLOAT, data.data());

    auto create_reshape = [&](Tensor* shape_tensor, std::vector<Tensor*>& inputs) {
        auto reshape = std::make_shared<Reshape>();
        inputs.push_back(src_tensor.get());
        inputs.push_back(shape_tensor);

        auto output = std::make_shared<Tensor>();
        output->is_dynamic = true;
        out_tensors.push_back(output);
        reshape->inputs = inputs.data();
        reshape->output = out_tensors.back().get();
        return reshape;
    };
    VM* vm = create_vm();
    auto test_reshape = [&](const Tensor& expect, Tensor* shape_tensor) {
        std::vector<Tensor*> inputs;
        auto reshape = create_reshape(shape_tensor, inputs);
        Instruction inst;
        inst.tag = TinyNN_INST_RESHAPE;
        inst.workload.reshape = *reshape;
        vm_instruction_call(vm, &inst);
        check_tensor(*reshape->output, expect);
    };
    {
        std::vector<int> shape_data({1, -1});
        auto shape_tensor = create_tensor({2}, TinyNN_INT, shape_data.data());
        auto result = create_tensor({1, 27}, TinyNN_FLOAT, data.data());
        test_reshape(*result, shape_tensor.get());
    }
    {
        std::vector<int> shape_data({1, -1, 3});
        auto shape_tensor = create_tensor({3}, TinyNN_INT, shape_data.data());
        auto result = create_tensor({1, 9, 3}, TinyNN_FLOAT, data.data());
        test_reshape(*result, shape_tensor.get());
    }
    {
        std::vector<int> shape_data({3, -1, 3});
        auto shape_tensor = create_tensor({3}, TinyNN_INT, shape_data.data());
        auto result = create_tensor({3, 3, 3}, TinyNN_FLOAT, data.data());
        test_reshape(*result, shape_tensor.get());
    }
    {
        std::vector<int> shape_data({-1});
        auto shape_tensor = create_tensor({1}, TinyNN_INT, shape_data.data());
        auto result = create_tensor({27}, TinyNN_FLOAT, data.data());
        test_reshape(*result, shape_tensor.get());
    }
}

// vim: syntax=cpp.doxygen
