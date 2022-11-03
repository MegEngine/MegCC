/**
 * \file runtime/src/vm/registry.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

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

#endif  // VM_REGISTRY_H

// vim: syntax=cpp.doxygen
