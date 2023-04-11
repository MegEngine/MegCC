#pragma once

#include "rc4/rc4_cryption_base.h"

#include <vector>

namespace megcc {

class RC4 {
public:
    static std::vector<uint8_t> decrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key);

    static std::vector<uint8_t> encrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key);

    static std::vector<uint8_t> get_decrypt_key();
};

class SimpleFastRC4 {
public:
    static std::vector<uint8_t> decrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key);
    static std::vector<uint8_t> encrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key);

    static std::vector<uint8_t> get_decrypt_key();
};

class NaiveEncrypt {
public:
    static std::vector<uint8_t> decrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key);

    static std::vector<uint8_t> get_decrypt_key();
};

class FakeEncrypt {
public:
    static std::vector<uint8_t> decrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key);

    static std::vector<uint8_t> get_decrypt_key();
};

}  // namespace megcc

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
