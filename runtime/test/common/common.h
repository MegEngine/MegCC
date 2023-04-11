#pragma once

#include <gtest/gtest.h>
#include <string>
extern "C" {
#include "data_struct.h"
#include "init.h"
#include "vm.h"
#include "vm/common.h"
#include "vm/instruction.h"
#include "vm/registry.h"
}
#include <memory>
#include <sstream>

#define ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr_avg, maxerr_avg_biased) \
    ASSERT_PRED_FORMAT5(assert_tensor_eq, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased)
namespace test {
void check_tensor(
        const Tensor& expected, const Tensor& computed, float epsilon = 1e-3,
        float max_avg_error = 1e-3, float max_avg_biased_error = 1e-3);

std::shared_ptr<Tensor> create_tensor(
        std::vector<uint32_t> shape, TinyNNDType dtype_enum, void* ptr);

VM* create_vm();

template <typename T>
std::shared_ptr<Tensor> create_scalar_tensor(T val, TinyNNDType dtype_enum) {
    auto tensor = std::make_shared<Tensor>();
    tensor->is_dynamic = true;
    DType dtype;
    dtype.type_enum = dtype_enum;
    tensor->dtype = dtype;
    Layout layout;
    layout.nr_dim = 1;
    layout.dims[0] = 1;
    tensor->layout = layout;
    tensor->ptr = malloc(sizeof(T));
    *(T*)(tensor->ptr) = val;
    return tensor;
}

class SimpleCombineModel {
public:
    SimpleCombineModel(int nr_device_model, int nr_instruction) {
        m_combine_model = (CombineModel*)malloc(sizeof(CombineModel));
        m_combine_model->nr_device_model = nr_device_model;
        m_combine_model->device_models =
                (DeviceModel**)malloc(sizeof(DeviceModel*) * nr_device_model);
        m_combine_model->weights = (Tensor*)malloc(sizeof(Tensor) * 10);
        for (int i = 0; i < nr_device_model; i++) {
            DeviceModel* dev_model = (DeviceModel*)malloc(sizeof(DeviceModel));
            m_device_models.push_back(dev_model);
            m_combine_model->device_models[i] = dev_model;
            dev_model->tensors = (Tensor*)malloc(sizeof(Tensor) * 10);

            dev_model->device.device_type = TinyNN_BARE_METAL;
            init_device(&dev_model->device);
            dev_model->opt = create_runtime_opt(&dev_model->device);
            dev_model->nr_instruction = nr_instruction;
            dev_model->instructions =
                    (Instruction*)malloc(sizeof(Instruction) * nr_instruction);
            for (int j = 0; j < nr_instruction; j++) {
                Instruction* ins = dev_model->instructions + j;
                ins->tag = TinyNN_INST_OPR;
                Opr* opr = &ins->workload.opr;
                opr->inputs = (Tensor**)malloc(sizeof(Tensor*));
                opr->nr_input = 1;
                opr->nr_output = 1;
                opr->outputs = (Tensor**)malloc(sizeof(Tensor*));
            }
        }
        m_combine_model->active_device_model_idx = 0;
        m_combine_model->host_dev.device_type = TinyNN_BARE_METAL;
        init_device(&(m_combine_model->host_dev));
    }
    ~SimpleCombineModel() {
        int nr_device_model = m_combine_model->nr_device_model;
        for (int i = 0; i < nr_device_model; i++) {
            DeviceModel* dev_model = m_combine_model->device_models[i];
            int nr_instruction = dev_model->nr_instruction;
            for (int j = 0; j < nr_instruction; j++) {
                Instruction* ins = dev_model->instructions + j;
                Opr* opr = &ins->workload.opr;
                free(opr->inputs);
                free(opr->outputs);
            }
            free(dev_model->instructions);
            free(dev_model->tensors);
            free(dev_model);
        }
        free(m_combine_model->device_models);
        free(m_combine_model->weights);
        free(m_combine_model);
    }

    CombineModel* m_combine_model;
    std::vector<DeviceModel*> m_device_models;
};
}  // namespace test

// vim: syntax=cpp.doxygen
