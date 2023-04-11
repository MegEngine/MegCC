#include <string.h>
#include <memory>
#include <vector>
#include "./common/common.h"
#include "math.h"
using namespace test;

TEST(INSTRUCTION, BroadCast) {
    std::shared_ptr<Tensor> output;
    std::vector<int> data0(5 * 340);
    std::vector<int> data1(5 * 340);
    std::vector<int> data2(5 * 340);
    for (size_t i = 0; i < 5 * 340; i++) {
        data0[i] = i;
        data1[i] = i + 1;
        data2[i] = i + 2;
    }

    auto create_broadcast = [&](std::vector<Tensor*>& inputs) {
        auto broad_cast = std::make_shared<BroadCast>();

        output = std::make_shared<Tensor>();
        output->is_dynamic = true;

        broad_cast->inputs[0] = inputs[0];
        broad_cast->inputs[1] = inputs[1];
        broad_cast->output = output.get();
        return broad_cast;
    };
    VM* vm = create_vm();
    auto test_broadcast = [&](std::vector<Tensor*> inputs, const Tensor& expect) {
        auto broadcast = create_broadcast(inputs);
        Instruction inst;
        inst.tag = TinyNN_INST_BROADCAST;
        inst.workload.broadcast = *broadcast;
        vm_instruction_call(vm, &inst);
        check_tensor(*broadcast->output, expect);
        vm->model->host_dev.free(broadcast->output->ptr);
    };

    auto test_case = [&](std::vector<uint32_t> shape0, std::vector<uint32_t> shape1) {
        auto src0 = create_tensor(shape0, TinyNNDType::TinyNN_INT, data0.data());
        int index = 0;
        for (auto i : shape1) {
            data1[index++] = i;
        }
        auto src1 = create_tensor(
                {static_cast<uint32_t>(shape1.size())}, TinyNNDType::TinyNN_INT,
                data1.data());
        auto trueth = create_tensor(shape1, TinyNNDType::TinyNN_INT, data2.data());
        Tensor src_copy = *src0;
        //! broadcast input
        for (int i = 0; i < shape1.size(); i++) {
            if (src_copy.layout.dims[i] == 1 && shape1[i] != 1) {
                src_copy.layout.dims[i] = shape1[i];
                src_copy.layout.stride[i] = 0;
            }
        }
        size_t nr_elem = 1;
        for (int j = 0; j < src_copy.layout.nr_dim; ++j) {
            nr_elem *= src_copy.layout.dims[j];
        }
        auto dst_layout = trueth->layout;
        auto src_layout = src_copy.layout;

        NoconIter src_iter = init_iter(src_copy.layout);
        NoconIter dst_iter = init_iter(dst_layout);
        int* dst_data = static_cast<int*>(trueth->ptr);
        int* src_data = static_cast<int*>(src_copy.ptr);
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(src_layout, &src_iter);
            inc_iter(dst_layout, &dst_iter);
        }
        test_broadcast({src0.get(), src1.get()}, *trueth);
    };

    test_case({1, 1}, {10, 34});
    test_case({1, 34}, {10, 34});
    test_case({10, 1}, {10, 34});
    test_case({1, 10, 1}, {5, 10, 34});
}

// vim: syntax=cpp.doxygen
