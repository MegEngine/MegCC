#include "init.h"
#include "parse.h"
#include "vm.h"
#include "vm/common.h"
#include "vm/instruction.h"
#include "vm/registry.h"
#if ENABLE_INST_MEMFORWARD
static MemForwardType get_forward_type(ns(MemForwardType_enum_t) type) {
    switch (type) {
        case ns(MemForwardType_RESHAPE):
            return TinyNN_MemForward_Reshape;
        case ns(MemForwardType_SUBTENSOR):
            return TinyNN_MemForward_Subtensor;
        default: {
            LOG_ERROR("no support forward type from fbs.\n");
        }
    }
    tinynn_trap();
}

static TinyNNStatus load(flatbuffers_generic_t fbs_inst, Instruction* inst, VM* vm) {
    MemForward* memforward = &inst->workload.mem_forward;
    ns(MemForward_table_t) fbs_reshape = (ns(MemForward_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_MEM_FORWARD;
    int32_t input_idx = ns(MemForward_input(fbs_reshape));
    DeviceModel* model = get_active_device_model(vm);
    memforward->input = model->tensors + input_idx;
    int32_t output_idx = ns(MemForward_output(fbs_reshape));
    memforward->output = model->tensors + output_idx;
    memforward->offset = ns(MemForward_offset(fbs_reshape));
    memforward->type = get_forward_type(ns(MemForward_type(fbs_reshape)));
    LOG_DEBUG(
            "parse memforward, input idx: %d, output idx: %d, type=%d, "
            "offset=%d\n",
            input_idx, output_idx, memforward->type, memforward->offset);
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute(Instruction* inst, VM* vm) {
    Tensor *input = inst->workload.mem_forward.input,
           *output = inst->workload.mem_forward.output;
    // TODO: assert this memforward operation is safe
    LOG_DEBUG(
            "Memory Forward offset is %d, offset from %p to %p\n",
            inst->workload.mem_forward.offset, input->ptr,
            input->ptr + inst->workload.mem_forward.offset);
    output->ptr = input->ptr + inst->workload.mem_forward.offset;
    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct(VM* vm, Instruction* inst) {
    return TinyNN_SUCCESS;
}

void register_memforward(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_MemForward), &load);
    vm_register_instruction_call(vm, TinyNN_INST_MEM_FORWARD, &execute);
    vm_register_instruction_destruct(vm, TinyNN_INST_MEM_FORWARD, &destruct);
}
#else
void register_memforward(VM* vm) {}
#endif
// vim: syntax=cpp.doxygen
