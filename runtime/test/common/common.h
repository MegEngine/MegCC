#pragma once

#include <gtest/gtest.h>
#include <string>
#include <vector>
extern "C" {
#include "data_struct.h"
#include "init.h"
#include "lite/io_tensor.h"
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
    SimpleCombineModel(
            int nr_device_model, int nr_instruction, int nr_input = 1,
            int nr_output = 1, const std::vector<const char*>& input_name = {"data"},
            const std::vector<const char*>& output_name = {"output"},
            const std::vector<std::vector<std::vector<size_t>>>& input_shape =
                    {{{1, 3, 224, 224}}},
            const std::vector<std::vector<std::vector<size_t>>>& output_shape = {
                    {{1, 2}}}) {
        TINYNN_ASSERT(
                nr_device_model > 0 && nr_device_model == input_shape.size() &&
                nr_device_model == output_shape.size());
        TINYNN_ASSERT(
                nr_input > 0 && nr_input == input_name.size() && nr_output > 0 &&
                nr_output == output_name.size());
        for (size_t i = 0; i < nr_device_model; ++i) {
            TINYNN_ASSERT(
                    nr_input == input_shape[i].size() &&
                    nr_output == output_shape[i].size());
        }
        m_combine_model = (CombineModel*)malloc(sizeof(CombineModel));
        m_combine_model->combo_iotensor = create_combo_io_tensor();
        m_combine_model->nr_device_model = nr_device_model;
        m_combine_model->device_models =
                (DeviceModel**)malloc(sizeof(DeviceModel*) * nr_device_model);
        m_combine_model->weights = (Tensor*)malloc(sizeof(Tensor) * 10);
        for (int i = 0; i < nr_device_model; i++) {
            DeviceModel* dev_model = (DeviceModel*)malloc(sizeof(DeviceModel));
            m_device_models.push_back(dev_model);
            m_combine_model->device_models[i] = dev_model;
            dev_model->tensors = (Tensor*)malloc(sizeof(Tensor) * 10);

            dev_model->nr_input = nr_input;
            dev_model->inputs = (Tensor**)malloc(sizeof(Tensor*) * dev_model->nr_input);
            for (size_t j = 0; j < dev_model->nr_input; ++j) {
                dev_model->inputs[j] = (Tensor*)malloc(sizeof(Tensor));
                memset(dev_model->inputs[j], 0, sizeof(Tensor));
                dev_model->inputs[j]->name = (char*)input_name[j];
                dev_model->inputs[j]->layout.nr_dim = input_shape[i][j].size();
                for (size_t k = 0; k < dev_model->inputs[j]->layout.nr_dim; ++k) {
                    dev_model->inputs[j]->layout.dims[k] = input_shape[i][j][k];
                }
            }

            dev_model->nr_output = nr_output;
            dev_model->outputs =
                    (Tensor**)malloc(sizeof(Tensor*) * dev_model->nr_output);
            for (size_t j = 0; j < dev_model->nr_output; ++j) {
                dev_model->outputs[j] = (Tensor*)malloc(sizeof(Tensor));
                memset(dev_model->outputs[j], 0, sizeof(Tensor));
                dev_model->outputs[j]->name = (char*)output_name[j];
                dev_model->outputs[j]->layout.nr_dim = output_shape[i][j].size();
                for (size_t k = 0; k < dev_model->outputs[j]->layout.nr_dim; ++k) {
                    dev_model->outputs[j]->layout.dims[k] = output_shape[i][j][k];
                }
            }

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
            for (int j = 0; j < dev_model->nr_input; ++j)
                free(dev_model->inputs[j]);
            free(dev_model->inputs);
            for (int j = 0; j < dev_model->nr_output; ++j)
                free(dev_model->outputs[j]);
            free(dev_model->outputs);
            free(dev_model->instructions);
            free(dev_model->tensors);
            free(dev_model);
        }
        free(m_combine_model->device_models);
        free(m_combine_model->weights);
        destroy_combo_io_tensor(m_combine_model->combo_iotensor);
        free(m_combine_model);
    }

    CombineModel* m_combine_model;
    std::vector<DeviceModel*> m_device_models;
};
}  // namespace test

// vim: syntax=cpp.doxygen
