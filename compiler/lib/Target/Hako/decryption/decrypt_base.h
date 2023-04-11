#pragma once
#include "misc.h"

namespace megcc {

template <int count>
struct DecryptionRegister;

}  // namespace megcc

#define CONCAT_IMPL(a, b)  a##b
#define MACRO_CONCAT(a, b) CONCAT_IMPL(a, b)

#define REGIST_DECRYPTION_METHOD(name_, func_, key_) \
    REGIST_DECRYPTION_METHOD_WITH_NUM(__COUNTER__, name_, func_, key_)

#define REGIST_DECRYPTION_METHOD_WITH_NUM(number_, name_, func_, key_)            \
    template <>                                                                   \
    struct DecryptionRegister<number_> {                                          \
        DecryptionRegister() { register_decryption_and_key(name_, func_, key_); } \
    };                                                                            \
    namespace {                                                                   \
    DecryptionRegister<number_> MACRO_CONCAT(decryption_, number_);               \
    }

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
