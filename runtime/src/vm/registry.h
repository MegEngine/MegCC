#ifndef VM_REGISTRY_H
#define VM_REGISTRY_H

#include "vm.h"

void register_all(VM* vm);

void register_op(VM* vm);

void register_memory_management(VM* vm);

void register_memforward(VM* vm);

void register_subtensor(VM* vm);

void register_setsubtensor(VM* vm);

void register_dimshuffle(VM* vm);

void register_broadcast_shape_of(VM* vm);

void register_reshape(VM* vm);

void register_extern_opr(VM* vm);

#endif  // VM_REGISTRY_H

// vim: syntax=cpp.doxygen
