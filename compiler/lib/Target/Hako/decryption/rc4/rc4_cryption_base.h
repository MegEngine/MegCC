/**
 * \file compiler/lib/Target/Hako/decryption/rc4/rc4_cryption_base.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace megcc {
namespace rc4 {

#define m256(x) static_cast<uint8_t>(x)

/*! \brief Pseudo-random byte stream for RC4.
 */
class RC4RandStream {
public:
    RC4RandStream() = default;

    RC4RandStream(uint64_t key) { reset(key); }

    void reset(uint64_t init_key) {
        i_ = j_ = 0;
        for (int i = 0; i < 256; i++)
            s_[i] = i;
        uint8_t j = 0;
        for (int i = 0; i < 256; i++) {
            j = j + s_[i] + m256(init_key >> ((i % 8) * 8));
            std::swap(s_[i], s_[j]);
        }
        // drop
        for (int i = 0; i < 768; i++) {
            next8();
        }
        for (int i = 0, t = next8(); i < t; i++) {
            next8();
        }
    }

    uint8_t next8() {
        i_++;
        uint8_t a = s_[i_];
        j_ += a;
        uint8_t b = s_[j_];
        s_[i_] = b;
        s_[j_] = a;
        uint8_t c = s_[m256((i_ << 5) ^ (j_ >> 3))] +
                    s_[m256((j_ << 5) ^ (i_ >> 3))];
        return (s_[m256(a + b)] + s_[c ^ 0xAA]) ^ s_[m256(j_ + b)];
    }

    uint64_t next64() {
        uint64_t rst;
        uint8_t* buf = reinterpret_cast<uint8_t*>(&rst);
        for (int i = 0; i < 8; i++) {
            buf[i] = next8();
        }
        return rst;
    }

private:
    uint8_t s_[256], i_ = 0, j_ = 0;
};
#undef m256

/*!
 * \brief fast and secure 64-bit hash
 * see https://code.google.com/p/fast-hash/
 */
class FastHash64 {
public:
    FastHash64(uint64_t seed)
            : hash_{seed},
              mul0_{key_gen_hash_mul0()},
              mul1_{key_gen_hash_mul1()} {}

    void feed(uint64_t val) {
        val ^= val >> 23;
        val *= mul0_;
        val ^= val >> 47;
        hash_ ^= val;
        hash_ *= mul1_;
    }

    uint64_t get() { return hash_; }

private:
    uint64_t hash_;
    const uint64_t mul0_, mul1_;

    static uint64_t key_gen_hash_mul0() {
        uint64_t rst;
        uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);
        buf[2] = 2;
        buf[3] = 3;
        buf[6] = 6;
        buf[1] = 1;
        buf[5] = 5;
        buf[4] = 4;
        buf[0] = 0;
        buf[7] = 7;
        return rst;
    }

    static uint64_t key_gen_hash_mul1() {
        uint64_t rst;
        uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);
        buf[6] = 6;
        buf[2] = 2;
        buf[7] = 7;
        buf[1] = 1;
        buf[5] = 5;
        buf[0] = 0;
        buf[4] = 4;
        buf[3] = 3;
        return rst;
    }
};

// The encryption keys are always inlined.
static inline uint64_t key_gen_enc_key() {
    uint64_t rst;
    uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);

    buf[4] = 4;
    buf[3] = 3;
    buf[7] = 7;
    buf[6] = 6;
    buf[0] = 0;
    buf[5] = 5;
    buf[2] = 2;
    buf[1] = 1;

    return rst;
}

static inline uint64_t key_gen_hash_key() {
    uint64_t rst;
    uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);

    buf[2] = 2;
    buf[5] = 5;
    buf[4] = 4;
    buf[7] = 7;
    buf[1] = 1;
    buf[3] = 3;
    buf[6] = 6;
    buf[0] = 0;

    return rst;
}
}  // namespace rc4
}  // namespace megcc

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
