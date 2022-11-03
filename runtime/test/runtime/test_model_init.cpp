/**
 * \file runtime/test/runtime/test_model_init.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string.h>
#include <memory>
#include <vector>
#include "./common/common.h"
#include "init.h"
#include "math.h"
using namespace test;

KernelFunc kernels[10];
InitFunc init_kernels[10];
WorkspaceFunc workspace_func[10];
DeduceFunc deduce_func[10];

//! not thread safe
void load_kernel_init_function() {
    return ;
}

namespace{
TinyNNStatus Init0(Tensor** inputs, int nr_input, Tensor* out_weights,
                   int* nr_out_weight, const RuntimeOpt* opt) {
    if (out_weights == NULL && nr_out_weight != NULL) {
        *nr_out_weight = 1;
        return TinyNN_SUCCESS;
    }
    Tensor* in_weights = inputs[0];
    if (out_weights != NULL && nr_out_weight == NULL) {
        out_weights->layout.nr_dim = 2;
        out_weights->layout.dims[0] = 5;
        out_weights->layout.dims[1] = 2;
        out_weights->layout.stride[0] = 1;
        out_weights->layout.stride[1] = 5;
        out_weights->dtype.type_enum = TinyNN_FLOAT;
        out_weights->name = in_weights->name;

        return TinyNN_SUCCESS;
    }
    if (out_weights != NULL && nr_out_weight != NULL) {
        float* data = (float*)out_weights->ptr;
        for (int i = 0; i < 10; i++) {
            data[i] = i;
        }
    }
    return TinyNN_SUCCESS;
}

TinyNNStatus Init1(Tensor** inputs, int nr_input, Tensor* out_weights,
                   int* nr_out_weight, const RuntimeOpt* opt) {
    if (out_weights == NULL && nr_out_weight != NULL) {
        *nr_out_weight = 1;
        return TinyNN_SUCCESS;
    }
    Tensor* in_weights = inputs[0];
    if (out_weights != NULL && nr_out_weight == NULL) {
        out_weights->layout.nr_dim = 2;
        out_weights->layout.dims[0] = 5;
        out_weights->layout.dims[1] = 8;
        out_weights->layout.stride[0] = 1;
        out_weights->layout.stride[1] = 5;
        out_weights->dtype.type_enum = TinyNN_FLOAT;
        out_weights->name = in_weights->name;

        return TinyNN_SUCCESS;
    }
    if (out_weights != NULL && nr_out_weight != NULL) {
        float* data = (float*)out_weights->ptr;
        for (int i = 0; i < 40; i++) {
            data[i] = 40 - i;
        }
    }
    return TinyNN_SUCCESS;
}
}  // namespace

TEST(RUNTIME, ShareWithOldWeights) {
    SimpleCombineModel simple_model = SimpleCombineModel(1, 2);
    std::vector<float> weight1(10, 0);
    std::vector<float> weight2(40, 0);
    auto combine_model = simple_model.m_combine_model;
    auto w0 = create_tensor({2, 5}, TinyNNDType::TinyNN_FLOAT, weight1.data());
    w0->is_shared = 1;
    w0->is_dynamic= 0;
    w0->use_count = 1;
    w0->ptr= weight1.data();
    w0->size = 10 * sizeof(float);
    w0->name = (char*)"w0";
    auto w1 = create_tensor({8, 5}, TinyNNDType::TinyNN_FLOAT, weight1.data());
    w1->is_shared = 1;
    w1->use_count = 1;
    w1->is_dynamic= 0;
    w1->ptr= weight2.data();
    w1->size = 40 * sizeof(float);
    w1->name = (char*)"w1";
    *(combine_model->weights) = *w0;
    *(combine_model->weights + 1) = *w1;

    init_kernels[0] = Init0;
    init_kernels[1] = Init1;

    auto dev_model = combine_model->device_models[0];
    auto ins0 = dev_model->instructions;
    auto ins1 = dev_model->instructions + 1;
    ins0->workload.opr.inputs[0] = combine_model->weights;
    ins1->workload.opr.inputs[0] = combine_model->weights + 1;
    ins0->workload.opr.init_func = 0;
    ins1->workload.opr.init_func = 1;

    init_model_weights(combine_model);


    ASSERT_EQ(ins0->workload.opr.inputs[0]->is_shared, 1);
    ASSERT_EQ(ins1->workload.opr.inputs[0]->is_shared, 1);

    float* data0 = (float*)ins0->workload.opr.inputs[0]->ptr;
    float* data1 = (float*)ins1->workload.opr.inputs[0]->ptr;

    //! after opr init, it will copy the processed data to the original weight
    //! memory and use it in future computation
    ASSERT_EQ(data0, weight1.data());
    ASSERT_EQ(data1, weight2.data());
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(data0[i], i);
    }
    for (int i = 0; i < 40; i++) {
        ASSERT_EQ(data1[i], 40 - i);
    }
    tinynn_free(dev_model->processed_weights);
}

TEST(RUNTIME, ShareWithOldWeightsAndCrossDeviceModel) {
    SimpleCombineModel simple_model = SimpleCombineModel(2, 2);
    std::vector<float> weight1(10, 0);
    std::vector<float> weight2(40, 0);
    auto combine_model = simple_model.m_combine_model;
    auto w0 = create_tensor({2, 5}, TinyNNDType::TinyNN_FLOAT, weight1.data());
    w0->is_shared = 1;
    w0->is_dynamic= 0;
    w0->use_count = 2;
    w0->ptr= weight1.data();
    w0->size = 10 * sizeof(float);
    w0->name = (char*)"w0";
    auto w1 = create_tensor({8, 5}, TinyNNDType::TinyNN_FLOAT, weight1.data());
    w1->is_shared = 1;
    w1->use_count = 2;
    w1->is_dynamic= 0;
    w1->ptr= weight2.data();
    w1->size = 40 * sizeof(float);
    w1->name = (char*)"w1";
    *(combine_model->weights) = *w0;
    *(combine_model->weights + 1) = *w1;

    init_kernels[0] = Init0;
    init_kernels[1] = Init1;

    auto dev_model0 = combine_model->device_models[0];
    auto dev_model1 = combine_model->device_models[1];
    auto ins0_0 = dev_model0->instructions;
    auto ins0_1 = dev_model0->instructions + 1;
    ins0_0->workload.opr.inputs[0] = combine_model->weights;
    ins0_1->workload.opr.inputs[0] = combine_model->weights + 1;
    ins0_0->workload.opr.init_func = 0;
    ins0_1->workload.opr.init_func = 1;

    auto ins1_0 = dev_model1->instructions;
    auto ins1_1 = dev_model1->instructions + 1;
    ins1_0->workload.opr.inputs[0] = combine_model->weights;
    ins1_1->workload.opr.inputs[0] = combine_model->weights + 1;
    ins1_0->workload.opr.init_func = 0;
    ins1_1->workload.opr.init_func = 1;

    init_model_weights(combine_model);


    ASSERT_EQ(ins0_0->workload.opr.inputs[0]->is_shared, 1);
    ASSERT_EQ(ins0_1->workload.opr.inputs[0]->is_shared, 1);

    ASSERT_EQ(ins1_0->workload.opr.inputs[0]->is_shared, 1);
    ASSERT_EQ(ins1_1->workload.opr.inputs[0]->is_shared, 1);

    float* data0_0 = (float*)ins0_0->workload.opr.inputs[0]->ptr;
    float* data0_1 = (float*)ins0_1->workload.opr.inputs[0]->ptr;

    float* data1_0 = (float*)ins1_0->workload.opr.inputs[0]->ptr;
    float* data1_1 = (float*)ins1_1->workload.opr.inputs[0]->ptr;

    //! after opr init, it will copy the processed data to the original weight
    //! memory and use it in future computation
    ASSERT_EQ(data0_0, weight1.data());
    ASSERT_EQ(data0_1, weight2.data());

    ASSERT_EQ(data0_0, data1_0);
    ASSERT_EQ(data0_1, data1_1);
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(data0_0[i], i);
    }
    for (int i = 0; i < 40; i++) {
        ASSERT_EQ(data0_1[i], 40 - i);
    }
    tinynn_free(dev_model0->processed_weights);
    tinynn_free(dev_model1->processed_weights);
}

// vim: syntax=cpp.doxygen
