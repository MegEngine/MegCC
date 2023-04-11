#include <string.h>
#include <memory>
#include <vector>
#include "./common/common.h"
#include "math.h"
using namespace test;

TEST(INSTRUCTION, Dimshuffle) {
    //! input shape =[10, 20, 10, 10]
    std::shared_ptr<Tensor> output;
    std::vector<int> data0(20 * 10 * 10 * 10);
    std::vector<int> data1(20 * 10 * 10 * 10);
    for (size_t i = 0; i < 20 * 10 * 10 * 10; i++) {
        data0[i] = i + 1;
        data1[i] = i + 1;
    }

    auto create_dimshuffle = [&](Tensor* input, std::vector<uint32_t> pattern) {
        auto dimshuffle = std::make_shared<Dimshuffle>();
        dimshuffle->pattern_dim = pattern.size();
        for (int i = 0; i < pattern.size(); i++) {
            dimshuffle->pattern[i] = pattern[i];
        }

        output = std::make_shared<Tensor>();
        output->is_dynamic = true;

        dimshuffle->input = input;
        dimshuffle->output = output.get();
        return dimshuffle;
    };
    VM* vm = create_vm();
    auto test_dimshuffle = [&](Tensor* input, std::vector<uint32_t> pattern,
                               const Tensor& expect) {
        auto dimshuffle = create_dimshuffle(input, pattern);
        Instruction inst;
        inst.tag = TinyNN_INST_DIMSHUFFLE;
        inst.workload.dimshuffle = *dimshuffle;
        vm_instruction_call(vm, &inst);
        check_tensor(*dimshuffle->output, expect);
        vm->model->host_dev.free(dimshuffle->output->ptr);
    };

    auto test_case = [&](std::vector<uint32_t> pattern) {
        std::vector<uint32_t> shape{10, 20, 10, 10};
        auto src = create_tensor(shape, TinyNNDType::TinyNN_INT, data0.data());
        Tensor src_copy = *src;
        Layout src_layout = src->layout, out_layout = src->layout;
        for (int i = 0; i < pattern.size(); i++) {
            src_layout.dims[i] = src->layout.dims[pattern[i]];
            out_layout.dims[i] = src->layout.dims[pattern[i]];
            src_layout.stride[i] = src->layout.stride[pattern[i]];
        }
        out_layout.stride[out_layout.nr_dim - 1] = 1;
        for (int index = out_layout.nr_dim - 2; index >= 0; index--) {
            out_layout.stride[index] =
                    out_layout.dims[index + 1] * out_layout.stride[index + 1];
        }

        auto trueth = create_tensor(shape, TinyNNDType::TinyNN_INT, data1.data());
        trueth->layout = out_layout;
        NoconIter src_iter = init_iter(src_layout);
        NoconIter dst_iter = init_iter(out_layout);
        int* dst_data = static_cast<int*>(trueth->ptr);
        int* src_data = static_cast<int*>(src->ptr);
        for (size_t i = 0; i < 10 * 20 * 10 * 10; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(src_layout, &src_iter);
            inc_iter(out_layout, &dst_iter);
        }
        test_dimshuffle(src.get(), pattern, *trueth);
    };

    test_case({0, 1, 2, 3});
    test_case({0, 3, 2, 1});
    test_case({0, 2, 1, 3});
    test_case({2, 3, 0, 1});
}

// vim: syntax=cpp.doxygen
