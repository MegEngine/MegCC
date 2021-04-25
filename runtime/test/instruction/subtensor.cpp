/**
 * \file
 * runtime/test/instruction/subtensor.cpp
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

TEST(INSTRUCTION, SubTensorTest) {
    std::vector<Instruction> insts;
    std::vector<std::shared_ptr<Tensor>> in_tensors;
    std::vector<std::shared_ptr<Tensor>> out_tensors;
    std::vector<Tensor*> inputs;

    //! input shape =[20, 20, 20]
    std::vector<float> data(20 * 20 * 20);
    for (size_t i = 0; i < 20 * 20 * 20; i++) {
        data[i] = i;
    }
    auto src_tensor = create_tensor({20, 20, 20}, TinyNN_FLOAT, data.data());

    auto create_subtensor = [&](IndexDesc* index, IndexDesc* flag,
                                std::vector<uint32_t> scaler_value = {}) {
        auto subtensor = std::make_shared<SubTensor>();
        inputs.clear();
        inputs.push_back(src_tensor.get());
        if (flag->start == 1) {
            in_tensors.push_back(
                    create_scalar_tensor(scaler_value[0], TinyNN_INT));
            inputs.push_back(in_tensors.back().get());
            scaler_value.erase(scaler_value.begin());
        }
        if (flag->end == 1) {
            in_tensors.push_back(
                    create_scalar_tensor(scaler_value[0], TinyNN_INT));
            inputs.push_back(in_tensors.back().get());
            scaler_value.erase(scaler_value.begin());
        }
        if (flag->step == 1) {
            in_tensors.push_back(
                    create_scalar_tensor(scaler_value[0], TinyNN_INT));
            inputs.push_back(in_tensors.back().get());
            scaler_value.erase(scaler_value.begin());
        }
        if (flag->index == 1) {
            in_tensors.push_back(
                    create_scalar_tensor(scaler_value[0], TinyNN_INT));
            inputs.push_back(in_tensors.back().get());
            scaler_value.erase(scaler_value.begin());
        }
        auto output = std::make_shared<Tensor>();
        output->is_dynamic = true;
        out_tensors.push_back(output);

        subtensor->nr_descs = 1;
        subtensor->descs = index;
        subtensor->flags = flag;

        subtensor->nr_input = inputs.size();
        subtensor->inputs = inputs.data();
        subtensor->output = out_tensors.back().get();
        return subtensor;
    };
    VM* vm = create_vm();
    auto test_subtensor = [&](IndexDesc* index, IndexDesc* flag,
                              const Tensor& expect,
                              std::vector<uint32_t> input_idx = {}) {
        auto subtensor = create_subtensor(index, flag, input_idx);
        Instruction inst;
        inst.tag = TinyNN_INST_SUBTENSOR;
        inst.workload.subtensor = *subtensor;
        vm_instruction_call(vm, &inst);
        check_tensor(*subtensor->output, expect);
        vm->model->host_dev.free(subtensor->output->ptr);
    };
    //! item = [1, 0, 10, 1, -1], input shape =[20, 20, 20]
    {
        IndexDesc index{1, 0, 10, 1, -1}, flag{0, 0, 0, 0, -1};
        std::vector<float> truth(20 * 10 * 20);
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 20; j++) {
                if (j >= 10)
                    continue;
                for (size_t k = 0; k < 20; k++) {
                    truth[i * 10 * 20 + j * 20 + k] = i * 20 * 20 + j * 20 + k;
                }
            }
        }
        auto result = create_tensor({20, 10, 20}, TinyNN_FLOAT, truth.data());
        test_subtensor(&index, &flag, *result);
    }

    {
        //! the start index in input is 1
        //! the step index in input is 2
        IndexDesc index{1, 1, 8, 2, -1}, flag{0, 1, 0, 1, -1};
        std::vector<uint32_t> scaler_value{4, 1};
        std::vector<float> truth(20 * 4 * 20);
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 20; j++) {
                if (j < 4 || j >= 8)
                    continue;
                for (size_t k = 0; k < 20; k++) {
                    truth[i * 4 * 20 + (j - 4) * 20 + k] =
                            i * 20 * 20 + j * 20 + k;
                }
            }
        }
        auto result = create_tensor({20, 4, 20}, TinyNN_FLOAT, truth.data());
        test_subtensor(&index, &flag, *result, scaler_value);
    }

    //! item = [1, 0, 8, 1, 2], flag =[ 0, -1, -1, -1, 1]
    {
        IndexDesc index{1, 0, 8, 1, 1}, flag{0, -1, -1, -1, 1};
        std::vector<float> truth(20 * 20);
        for (size_t i = 0; i < 20; i++) {
            for (size_t j = 0; j < 20; j++) {
                if (j != 1)
                    continue;
                for (size_t k = 0; k < 20; k++) {
                    truth[i * 20 + k] = i * 20 * 20 + j * 20 + k;
                }
            }
        }
        auto result = create_tensor({20, 20}, TinyNN_FLOAT, truth.data());
        test_subtensor(&index, &flag, *result, {1});
    }
    for (auto& tensor : in_tensors) {
        free(tensor->ptr);
    }
}

// vim: syntax=cpp.doxygen
