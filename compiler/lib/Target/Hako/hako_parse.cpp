/**
 * \file compiler/lib/Target/Hako/hako_parse.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "compiler/Target/Hako/hako_parse.h"
#include <string.h>
#include <fstream>
#include <memory>
#include "compiler/Common/Logger.h"
#include "rc4/rc4_cryption_base.h"
#include "rc4_cryption.h"
using namespace megcc;

//! used to check hash
uint64_t get_model_hash(
        uint64_t expected_hash, std::ifstream& model_stream, size_t size) {
    constexpr size_t CHUNK_SIZE = 1024 * 16;
    std::unique_ptr<uint8_t[]> chunk{new uint8_t[CHUNK_SIZE]};
    rc4::FastHash64 hasher(rc4::key_gen_hash_key());

    for (size_t start = 0, end; start < size; start = end) {
        end = std::min(start + CHUNK_SIZE, size);
        size_t len = end - start;
        model_stream.read((char*)chunk.get(), len);
        auto p64 = reinterpret_cast<uint64_t*>(chunk.get());
        size_t len64 = len / sizeof(uint64_t), len8 = len % sizeof(uint64_t);
        for (size_t i = 0; i < len64; ++i) {
            hasher.feed(p64[i]);
        }
        if (len8) {
            auto p8 = chunk.get() + len64 * sizeof(uint64_t);
            uint64_t last = 0;
            memcpy(&last, p8, len8);
            hasher.feed(last);
        }
    }
    return hasher.get();
}

std::vector<uint8_t> find_prefix(std::vector<uint8_t> src, std::string prefix) {
    CC_ASSERT(src.size() > prefix.size());
    CC_ASSERT(prefix.size() > 0);
    std::vector<uint8_t> res;
    for (size_t i = 0; i < src.size() - prefix.size(); ++i) {
        if (src[i] == prefix[0]) {
            bool match = true;
            for (size_t j = 0; j < prefix.size(); j++) {
                if (src[i + j] != prefix[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                res.resize(src.size() - i);
                memcpy(res.data(), ((uint8_t*)src.data()) + i, src.size() - i);
                return res;
            }
        }
    }
    return res;
}

std::vector<uint8_t> megcc::parse_hako(
        const std::vector<uint8_t>& model_buffer, int version) {
    std::vector<uint8_t> result;
    if (version == 2) {
        //! Naive is used for version2, version 1 use sfrc4
        result = NaiveEncrypt::decrypt_model(
                model_buffer.data(), model_buffer.size(),
                NaiveEncrypt::get_decrypt_key());
    } else {
        CC_ASSERT(version == 1);
        result = SimpleFastRC4::decrypt_model(
                model_buffer.data(), model_buffer.size(),
                SimpleFastRC4::get_decrypt_key());
    }
    std::vector<std::string> valid_magic{"MGBC", "MGBS"};
    std::vector<uint8_t> mdl_result;
    for (auto& prefix : valid_magic) {
        mdl_result = find_prefix(result, prefix);
        if (mdl_result.size() > 0) {
            break;
        }
    }
    CC_ASSERT(mdl_result.size() > 0)
            << "can not parse hako model as version " << version
            << ", you can change version by setting `-hako x`\n";
    return mdl_result;
}